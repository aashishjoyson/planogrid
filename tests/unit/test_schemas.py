import pytest
from pydantic import ValidationError

from core.schemas import (
    AnalysisResult,
    BBox,
    ComplianceFlag,
    Detection,
    DetectionClass,
    Severity,
)


def test_bbox_derived_fields():
    box = BBox(x1=10, y1=20, x2=50, y2=80)
    assert box.width == 40
    assert box.height == 60
    assert box.area == 2400
    assert box.cx == 30
    assert box.cy == 50
    assert box.as_xyxy() == (10, 20, 50, 80)


def test_bbox_degenerate_coords_do_not_go_negative():
    box = BBox(x1=50, y1=50, x2=10, y2=10)
    assert box.width == 0
    assert box.height == 0
    assert box.area == 0


def test_bbox_to_absolute_converts_normalized_coords():
    box = BBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, is_normalized=True)
    abs_box = box.to_absolute(image_width=1000, image_height=500)
    assert (abs_box.x1, abs_box.y1, abs_box.x2, abs_box.y2) == (100, 100, 500, 400)
    assert abs_box.is_normalized is False


def test_bbox_to_absolute_is_noop_when_already_absolute():
    box = BBox(x1=1, y1=2, x2=3, y2=4)
    assert box.to_absolute(1000, 1000) is box


def test_detection_ids_are_unique_by_default():
    box = BBox(x1=0, y1=0, x2=10, y2=10)
    d1 = Detection(bbox=box, class_name=DetectionClass.PRODUCT, confidence=0.9, detector_source="x")
    d2 = Detection(bbox=box, class_name=DetectionClass.PRODUCT, confidence=0.9, detector_source="x")
    assert d1.id != d2.id


def test_detection_rejects_out_of_range_confidence():
    box = BBox(x1=0, y1=0, x2=10, y2=10)
    with pytest.raises(ValidationError):
        Detection(bbox=box, class_name=DetectionClass.PRODUCT, confidence=1.5, detector_source="x")


def test_analysis_result_summary_counts_by_class_and_flags():
    box = BBox(x1=0, y1=0, x2=10, y2=10)
    detections = [
        Detection(bbox=box, class_name=DetectionClass.PRODUCT, confidence=0.9, detector_source="x"),
        Detection(bbox=box, class_name=DetectionClass.PRODUCT, confidence=0.8, detector_source="x"),
        Detection(
            bbox=box, class_name=DetectionClass.EMPTY_GAP, confidence=0.7, detector_source="x"
        ),
    ]
    flags = [
        ComplianceFlag(
            rule_name="missing_tag", severity=Severity.CRITICAL, target_id="abc", message="no tag"
        )
    ]
    result = AnalysisResult(
        image_path="shelf.jpg",
        image_width=1000,
        image_height=800,
        detections=detections,
        flags=flags,
    )
    assert result.summary == {"product": 2, "empty_gap": 1, "flags": 1}


def test_analysis_result_json_round_trip():
    result = AnalysisResult(image_path="shelf.jpg", image_width=100, image_height=100)
    restored = AnalysisResult.model_validate_json(result.model_dump_json())
    assert restored.image_path == "shelf.jpg"
    assert restored.summary == {"flags": 0}
