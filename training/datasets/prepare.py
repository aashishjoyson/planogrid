"""Orchestrates per-detector dataset preparation: resolves each configured
source (datasets/manifest.yaml + configs/datasets.yaml), runs the matching
converter, and merges + splits the result into datasets/processed/<detector>/.
See DESIGN.md §6.5, §9.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.config import REPO_ROOT, load_config
from core.exceptions import ConfigError
from core.logging import get_logger
from core.schemas import DetectionClass
from datasets.manifest import load_manifest
from training.datasets import hitl_to_yolo, roboflow_to_yolo
from training.datasets.common import merge_and_split

logger = get_logger(__name__)

PROCESSED_ROOT = REPO_ROOT / "datasets" / "processed"
STAGING_ROOT = PROCESSED_ROOT / "_staging"

_CANONICAL_CLASS = {
    "gap": DetectionClass.EMPTY_GAP.value,
    "tag": DetectionClass.PRICE_TAG.value,
}

_KAGGLE_CONVERTERS = {
    "hitl-supermarket-shelves": hitl_to_yolo.convert,
}


class _SplitConfig(BaseModel):
    # extra="forbid": an unrecognized key in configs/datasets.yaml (typo, or
    # a field nobody added a schema slot for) must fail loudly at load time,
    # not vanish silently — see training/trainer.py's TrainingDefaults for
    # the incident that motivated this across every config schema.
    model_config = ConfigDict(extra="forbid")

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 0
    strategy: str = "stratified_by_source"


class _DetectorSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(default_factory=list)
    ultralytics_builtin: str | None = None


class DatasetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: _SplitConfig = _SplitConfig()
    processed_dir: str = "datasets/processed"
    detectors: dict[str, _DetectorSources] = Field(default_factory=dict)


def _find_manifest_entry(manifest: dict, name: str) -> tuple[str, dict]:
    for source_kind in ("kaggle", "roboflow", "huggingface", "direct"):
        for entry in manifest.get(source_kind) or []:
            if entry["name"] == name:
                return source_kind, entry
    raise ConfigError(f"'{name}' (from configs/datasets.yaml) not found in datasets/manifest.yaml")


def prepare_detector_dataset(detector: str, *, force: bool = False) -> Path:
    """Returns the path to datasets/processed/<detector>/data.yaml, building
    it from configured sources if it doesn't already exist (or if
    force=True). `detector` must be "gap" or "tag" — "product" uses
    ultralytics' built-in SKU-110K.yaml and has nothing to prepare here.
    """
    if detector not in _CANONICAL_CLASS:
        raise ValueError(
            f"prepare_detector_dataset() only handles {sorted(_CANONICAL_CLASS)}; "
            f"'product' uses ultralytics' built-in SKU-110K.yaml directly, see training/trainer.py"
        )

    output_dir = PROCESSED_ROOT / detector
    data_yaml = output_dir / "data.yaml"
    if data_yaml.exists() and not force:
        logger.info(
            "datasets/processed/%s/data.yaml already exists, skipping preparation", detector
        )
        return data_yaml

    cfg = load_config("datasets", DatasetsConfig)
    detector_cfg = cfg.detectors.get(detector)
    if detector_cfg is None or not detector_cfg.sources:
        raise ConfigError(
            f"No sources configured for detector '{detector}' in configs/datasets.yaml"
        )

    manifest = load_manifest()
    staging_dirs: list[Path] = []

    for source_name in detector_cfg.sources:
        source_kind, entry = _find_manifest_entry(manifest, source_name)
        raw_dir = REPO_ROOT / entry["dest"]
        staging_dir = STAGING_ROOT / source_name

        if source_kind == "kaggle":
            converter = _KAGGLE_CONVERTERS.get(source_name)
            if converter is None:
                raise ConfigError(f"No converter registered for kaggle source '{source_name}'")
            converter(staging_dir=staging_dir)
        elif source_kind == "roboflow":
            roboflow_to_yolo.convert(raw_dir, staging_dir)
        else:
            raise ConfigError(
                f"No converter wired up for source kind '{source_kind}' (source '{source_name}')"
            )

        staging_dirs.append(staging_dir)

    merge_and_split(
        staging_dirs,
        output_dir,
        class_name=_CANONICAL_CLASS[detector],
        train=cfg.split.train,
        val=cfg.split.val,
        test=cfg.split.test,
        seed=cfg.split.seed,
    )
    return data_yaml


if __name__ == "__main__":
    import sys

    from core.logging import configure_logging

    configure_logging()
    if len(sys.argv) < 2 or sys.argv[1] not in _CANONICAL_CLASS:
        print(f"Usage: python -m training.datasets.prepare <{'|'.join(_CANONICAL_CLASS)}>")
        raise SystemExit(1)
    result_path = prepare_detector_dataset(sys.argv[1])
    print(f"Prepared: {result_path}")
