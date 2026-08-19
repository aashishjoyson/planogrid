"""YOLO-backed Detector implementation — wraps ultralytics.YOLO. Registered
as "yolo26" (works unmodified for any ultralytics-compatible checkpoint,
e.g. yolo11, since the predict API is identical — swap registry_name in
configs/model.yaml, not this file). See DESIGN.md §7.

Each instance is dedicated to exactly one of the three specialist classes
(product/gap/tag) — that's the whole point of the three-specialist
architecture (DESIGN.md ADR-001) — so every detection this produces gets the
instance's fixed `class_name`, never read back off the model's own output.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from core.registry import register_detector
from core.schemas import BBox, Detection, DetectionClass
from models.detection.base import Detector


@register_detector("yolo26")
class YoloDetector(Detector):
    def __init__(
        self,
        weights: str,
        class_name: DetectionClass,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        detector_source: str | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.class_name = class_name
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.detector_source = detector_source or f"yolo26-{class_name.value}"

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            confidence = float(box.conf[0])
            detections.append(
                Detection(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_name=self.class_name,
                    confidence=confidence,
                    detector_source=self.detector_source,
                )
            )
        return detections
