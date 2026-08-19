"""Converts the Kaggle Humans-in-the-Loop Supermarket Shelves dataset
(Supervisely JSON annotation format) into the staged YOLO layout
merge_and_split() consumes.

Only "Price" objects are extracted — "Product" boxes are unused for V1.0
(products are SKU-110K's job via ultralytics' native downloader, not this
dataset — see DESIGN.md §2.3, ADR-001).
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import REPO_ROOT
from core.logging import get_logger
from training.datasets.common import write_staged_example, xyxy_to_yolo

logger = get_logger(__name__)

DEFAULT_SOURCE = (
    REPO_ROOT
    / "datasets"
    / "raw"
    / "hitl_supermarket_shelves"
    / "Supermarket shelves"
    / "Supermarket shelves"
)
DEFAULT_STAGING = REPO_ROOT / "datasets" / "processed" / "_staging" / "hitl-supermarket-shelves"

TARGET_CLASS_TITLE = "Price"  # -> price_tag, class_id 0


def convert(source_dir: Path = DEFAULT_SOURCE, staging_dir: Path = DEFAULT_STAGING) -> int:
    images_dir = source_dir / "images"
    annotations_dir = source_dir / "annotations"
    if not images_dir.exists():
        raise FileNotFoundError(
            f"{images_dir} not found — run "
            "`python scripts/download_datasets.py --source kaggle` first."
        )

    count = 0
    for image_path in sorted(images_dir.glob("*.jpg")):
        annotation_path = annotations_dir / f"{image_path.name}.json"
        if not annotation_path.exists():
            logger.warning("No annotation for %s, skipping", image_path.name)
            continue

        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_w = data["size"]["width"]
        image_h = data["size"]["height"]

        boxes = []
        for obj in data["objects"]:
            if obj["classTitle"] != TARGET_CLASS_TITLE:
                continue
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            boxes.append(xyxy_to_yolo(x1, y1, x2, y2, image_w, image_h, class_id=0))

        write_staged_example(staging_dir, "hitl", image_path.stem, image_path, boxes)
        count += 1

    logger.info("hitl_to_yolo: staged %d images (Price boxes only) to %s", count, staging_dir)
    return count


if __name__ == "__main__":
    from core.logging import configure_logging

    configure_logging()
    convert()
