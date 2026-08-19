"""TensorBoard (always on) and optional Weights & Biases integration.

Ultralytics has both built in — this module only decides whether to *enable*
each via Ultralytics' own global SETTINGS (persisted machine-wide, defaults
to both OFF) and, for W&B, whether to log in — based on core.config.Settings.
No custom callback functions are needed: Ultralytics' own tensorboard.py /
wb.py hooks handle the actual logging once enabled. See DESIGN.md §9.
"""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


def configure_experiment_tracking() -> bool:
    """Enables TensorBoard (always) and W&B (only if WANDB_API_KEY is set)
    via ultralytics.utils.SETTINGS, logging into W&B if enabling it. Every
    other third-party integration Ultralytics supports (ClearML, Comet, DVC,
    HUB, MLflow, Neptune, Ray Tune) is explicitly disabled — this project
    only wires up the two named in DESIGN.md §9. Returns whether W&B was
    enabled.
    """
    from ultralytics.utils import SETTINGS

    settings = get_settings()
    wandb_enabled = bool(settings.wandb_api_key)

    SETTINGS.update(
        {
            "tensorboard": True,
            "wandb": wandb_enabled,
            "clearml": False,
            "comet": False,
            "dvc": False,
            "hub": False,
            "mlflow": False,
            "neptune": False,
            "raytune": False,
        }
    )

    if wandb_enabled:
        import wandb

        wandb.login(key=settings.wandb_api_key)
        logger.info("Weights & Biases logging enabled")
    else:
        logger.info("Weights & Biases logging disabled (WANDB_API_KEY not set)")

    return wandb_enabled
