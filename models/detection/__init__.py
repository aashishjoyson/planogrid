"""Detector ABC + YOLO-backed implementation, and the CompositeDetector that
fuses the three specialist detectors (product/gap/tag). See DESIGN.md §7.

Importing this package registers "yolo26" in core.registry.detector_registry
as a side effect (models/detection/yolo_detector.py's @register_detector) —
CompositeDetector relies on that having happened before it resolves a
registry_name from configs/model.yaml.
"""

from models.detection import yolo_detector  # noqa: F401  (side effect: registers "yolo26")
from models.detection.base import Detector
from models.detection.composite import CompositeDetector, ModelConfig

__all__ = ["CompositeDetector", "Detector", "ModelConfig"]
