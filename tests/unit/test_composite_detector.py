"""Unit tests for CompositeDetector's partial-availability logic and
detection merging. Partial availability (some checkpoints trained, some not)
is the expected steady state while training is in progress, not an edge
case — see DESIGN.md §7, §7.2 and the plan addendum (2026-08-04).

Uses a fake registered detector so these tests never need a real trained
YOLO checkpoint; `REPO_ROOT / <absolute tmp_path>` correctly resolves to
just the absolute path (pathlib join-with-absolute-path semantics), so no
REPO_ROOT monkeypatching is needed either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.registry import Registry
from core.schemas import BBox, Detection, DetectionClass
from models.detection.base import Detector
from models.detection.composite import CompositeDetector, ModelConfig


class _FakeDetector(Detector):
    """Returns one fixed detection per call — stands in for a real
    ultralytics.YOLO-backed detector in these tests.
    """

    def __init__(
        self,
        weights: str,
        class_name: DetectionClass,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
    ) -> None:
        self.class_name = class_name
        self.detector_source = f"fake-{class_name.value}"

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(
                bbox=BBox(x1=0, y1=0, x2=10, y2=10),
                class_name=self.class_name,
                confidence=0.9,
                detector_source=self.detector_source,
            )
        ]


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    """An isolated registry with only "fake" registered — never touches the
    real global detector_registry (which has "yolo26" registered).
    """
    registry: Registry = Registry("detector")
    registry.register("fake")(_FakeDetector)
    monkeypatch.setattr("models.detection.composite.detector_registry", registry)
    return registry


def _model_config(tmp_path: Path, ready_tasks: set[str]) -> ModelConfig:
    """A ModelConfig whose weights files exist on disk only for
    `ready_tasks` — mirrors the real "some detectors trained, some not" state.
    """

    def weights_for(task: str) -> str:
        path = tmp_path / f"{task}.pt"
        if task in ready_tasks:
            path.write_bytes(b"fake-weights")
        return str(path)

    return ModelConfig.model_validate(
        {
            "product_detector": {
                "registry_name": "fake",
                "task": "product",
                "weights": weights_for("product"),
            },
            "gap_detector": {
                "registry_name": "fake",
                "task": "empty_gap",
                "weights": weights_for("empty_gap"),
            },
            "tag_detector": {
                "registry_name": "fake",
                "task": "price_tag",
                "weights": weights_for("price_tag"),
            },
        }
    )


def test_zero_checkpoints_available(tmp_path, fake_registry):
    config = _model_config(tmp_path, ready_tasks=set())
    detector = CompositeDetector(config=config)

    assert detector.loaded_tasks == []
    assert all(not s.ready for s in detector.statuses)
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_one_checkpoint_available_only_loads_that_one(tmp_path, fake_registry):
    config = _model_config(tmp_path, ready_tasks={"empty_gap"})
    detector = CompositeDetector(config=config)

    assert detector.loaded_tasks == ["empty_gap"]
    statuses_by_task = {s.task: s.ready for s in detector.statuses}
    assert statuses_by_task == {"product": False, "empty_gap": True, "price_tag": False}

    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == DetectionClass.EMPTY_GAP


def test_all_checkpoints_available_merges_every_detector(tmp_path, fake_registry):
    config = _model_config(tmp_path, ready_tasks={"product", "empty_gap", "price_tag"})
    detector = CompositeDetector(config=config)

    assert set(detector.loaded_tasks) == {"product", "empty_gap", "price_tag"}

    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(detections) == 3
    assert {d.class_name for d in detections} == {
        DetectionClass.PRODUCT,
        DetectionClass.EMPTY_GAP,
        DetectionClass.PRICE_TAG,
    }


def test_detector_status_reports_resolved_absolute_path(tmp_path, fake_registry):
    config = _model_config(tmp_path, ready_tasks={"price_tag"})
    detector = CompositeDetector(config=config)

    tag_status = next(s for s in detector.statuses if s.task == "price_tag")
    assert tag_status.weights_path == str(tmp_path / "price_tag.pt")
