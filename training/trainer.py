"""Ultralytics training wrapper: experiment lifecycle, resume, AMP, early
stopping, TensorBoard/W&B, best-model selection. One call trains one
detector end to end. See DESIGN.md §9.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

from core.config import load_config
from core.exceptions import ConfigError
from core.logging import get_logger
from training.callbacks import configure_experiment_tracking
from training.datasets.prepare import prepare_detector_dataset
from training.experiment import Experiment, create_experiment, latest_experiment

logger = get_logger(__name__)

# Ultralytics' own bundled dataset config — auto-downloads the full, correctly
# split, already-YOLO-formatted SKU-110K on first use. See
# datasets/manifest.yaml's note under `huggingface:` and DESIGN.md §7, §9.
SKU110K_BUILTIN_YAML = "SKU-110K.yaml"


class TrainingDefaults(BaseModel):
    # extra="forbid": a profile setting an unrecognized key (e.g. a typo, or
    # a field nobody added a schema slot for) must fail loudly at config-load
    # time, not vanish silently — see the ADR-style note in
    # configs/profiles/local_8gb.yaml on how `cache` learned this the hard way.
    model_config = ConfigDict(extra="forbid")

    epochs: int = 100
    patience: int = 20
    image_size: int = 640
    optimizer: str = "auto"
    amp: bool = True
    seed: int = 0
    workers: int = 8
    save_period: int = -1
    batch: int = 16
    device: int | str = 0
    cache: bool | str = False


class DetectorTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_weights: str


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensorboard: bool = True
    wandb: str = "auto"
    experiment_root: str = "experiments"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: TrainingDefaults = TrainingDefaults()
    detectors: dict[str, DetectorTrainingConfig] = Field(default_factory=dict)
    tracking: TrackingConfig = TrackingConfig()


@dataclass(frozen=True)
class TrainResult:
    detector: str
    experiment: Experiment
    best_weights: Path
    metrics: dict[str, float]


def _resolve_data_yaml(detector: str) -> str:
    if detector == "product":
        return SKU110K_BUILTIN_YAML
    return str(prepare_detector_dataset(detector))


def train_detector(
    detector: str, *, resume: bool = False, profile: str | None = None
) -> TrainResult:
    """Trains one of "product", "gap", "tag" end to end: prepares data (via
    training.datasets.prepare for gap/tag; ultralytics' own SKU-110K.yaml
    auto-download for product), runs Ultralytics training into a fresh (or,
    if resume=True, the most recent) experiments/<detector>/ run, and
    promotes the resulting best.pt into checkpoints/<detector>/best.pt.
    """
    cfg = load_config("training", TrainingConfig, profile=profile)
    detector_cfg = cfg.detectors.get(detector)
    if detector_cfg is None:
        raise ConfigError(f"No training config for detector '{detector}' in configs/training.yaml")

    configure_experiment_tracking()
    data_yaml = _resolve_data_yaml(detector)

    if resume:
        experiment = latest_experiment(detector)
        if experiment is None:
            raise ConfigError(f"--resume requested but no prior experiment exists for '{detector}'")
        model = YOLO(str(experiment.last_weights_path))
        logger.info("Resuming '%s' from %s", detector, experiment.last_weights_path)
    else:
        experiment = create_experiment(detector)
        model = YOLO(detector_cfg.base_weights)
        experiment.snapshot_config(
            {
                "detector": detector,
                "base_weights": detector_cfg.base_weights,
                "data_yaml": data_yaml,
                "defaults": cfg.defaults.model_dump(),
                "profile": profile,
            }
        )

    defaults = cfg.defaults
    # `exist_ok=True` because create_experiment() already made this exact
    # directory (to snapshot the config before training starts) — without
    # it, ultralytics refuses to write into a pre-existing run dir.
    results = model.train(
        data=data_yaml,
        epochs=defaults.epochs,
        patience=defaults.patience,
        imgsz=defaults.image_size,
        batch=defaults.batch,
        device=defaults.device,
        workers=defaults.workers,
        optimizer=defaults.optimizer,
        amp=defaults.amp,
        seed=defaults.seed,
        save_period=defaults.save_period,
        cache=defaults.cache,
        project=str(experiment.root.parent),
        name=experiment.run_id,
        exist_ok=True,
        resume=resume,
        plots=True,
    )

    metrics: dict[str, float] = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }
    experiment.write_metrics(metrics)
    best_weights = experiment.promote_to_checkpoints()

    logger.info("Training complete for '%s': %s", detector, metrics)
    return TrainResult(
        detector=detector, experiment=experiment, best_weights=best_weights, metrics=metrics
    )
