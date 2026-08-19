"""CompositeDetector — loads whichever of the three specialist detectors
have a trained checkpoint on disk right now, and fuses their per-image
results into one list[Detection]. Gracefully handles partial availability
(e.g. only the gap detector trained so far, tag/product still running) —
this is not a fallback path, it's the expected steady state while training
is in progress. See DESIGN.md §7, §7.2, ADR-001.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from core.config import REPO_ROOT, load_config
from core.logging import get_logger
from core.registry import detector_registry
from core.schemas import Detection, DetectionClass
from models.detection.base import Detector

logger = get_logger(__name__)


class _DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_name: str
    task: str
    weights: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    image_size: int = 640


class _CompositeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    containment_overlap_threshold: float = 0.9


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_detector: _DetectorConfig
    gap_detector: _DetectorConfig
    tag_detector: _DetectorConfig
    composite: _CompositeSettings = _CompositeSettings()


_TASK_TO_CLASS = {
    "product": DetectionClass.PRODUCT,
    "empty_gap": DetectionClass.EMPTY_GAP,
    "price_tag": DetectionClass.PRICE_TAG,
}


class DetectorStatus(BaseModel):
    task: str
    ready: bool
    weights_path: str


class CompositeDetector:
    def __init__(self, config: ModelConfig | None = None, profile: str | None = None) -> None:
        self.config = config or load_config("model", ModelConfig, profile=profile)
        self._detectors: list[Detector] = []
        self.statuses: list[DetectorStatus] = []

        for detector_cfg in (
            self.config.product_detector,
            self.config.gap_detector,
            self.config.tag_detector,
        ):
            weights_path = REPO_ROOT / detector_cfg.weights
            ready = weights_path.exists()
            self.statuses.append(
                DetectorStatus(task=detector_cfg.task, ready=ready, weights_path=str(weights_path))
            )

            if not ready:
                logger.info(
                    "Skipping '%s' detector — no checkpoint at %s (not trained yet)",
                    detector_cfg.task,
                    weights_path,
                )
                continue

            detector_cls = detector_registry.get(detector_cfg.registry_name)
            detector = detector_cls(
                weights=str(weights_path),
                class_name=_TASK_TO_CLASS[detector_cfg.task],
                confidence_threshold=detector_cfg.confidence_threshold,
                image_size=detector_cfg.image_size,
            )
            self._detectors.append(detector)

        if not self._detectors:
            logger.warning("CompositeDetector loaded with zero trained detectors available")

    @property
    def loaded_tasks(self) -> list[str]:
        return [s.task for s in self.statuses if s.ready]

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for detector in self._detectors:
            detections.extend(detector.detect(image))
        return detections
