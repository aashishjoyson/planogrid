"""Top-level detection-inference entrypoint: an image in, an AnalysisResult
+ an annotated visualization out.

OCR, row segmentation, and compliance stages are not wired up yet (tonight's
scope is the detection half of the pipeline only — see the plan addendum);
AnalysisResult's rows/associations/flags fields stay at their schema
defaults (empty lists) until those stages exist. Adding them later is
additive, not a restructure — analyze_image()'s signature doesn't change.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.schemas import AnalysisResult, Detection, DetectionClass
from models.detection import CompositeDetector

# BGR (OpenCV convention) — matches app/frontend's hex tokens:
# gap #3B82F6, tag #14B8A6, product #F59E0B
_CLASS_COLORS_BGR: dict[DetectionClass, tuple[int, int, int]] = {
    DetectionClass.EMPTY_GAP: (246, 130, 59),
    DetectionClass.PRICE_TAG: (166, 184, 20),
    DetectionClass.PRODUCT: (11, 158, 245),
}


def analyze_image(
    image: np.ndarray, detector: CompositeDetector, image_path: str = ""
) -> AnalysisResult:
    detections = detector.detect(image)
    height, width = image.shape[:2]
    return AnalysisResult(
        image_path=image_path,
        image_width=width,
        image_height=height,
        detections=detections,
    )


def render_annotated(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Thin, color-coded boxes only — no per-box text label. SKU-110K-style
    shelf photos are densely packed (a real shelf can have 100+ products in
    one frame), so a text label per box makes the image unreadable rather
    than informative. Per-detection confidence is already surfaced in the
    dashboard's separate confidence-breakdown list, which is the right place
    to read exact numbers; the image's job is to show *where*, at a glance.
    """
    annotated = image.copy()
    for det in detections:
        color = _CLASS_COLORS_BGR.get(det.class_name, (200, 200, 200))
        x1, y1, x2, y2 = (int(v) for v in det.bbox.as_xyxy())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
    return annotated


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from core.logging import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Run the detection pipeline on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="outputs/annotated.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = CompositeDetector()
    logger.info("Loaded detectors: %s", detector.loaded_tasks)

    result = analyze_image(image, detector, image_path=args.image)
    logger.info("Found %d detections: %s", len(result.detections), result.summary)

    annotated = render_annotated(image, result.detections)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    logger.info("Wrote annotated image to %s", out_path)
