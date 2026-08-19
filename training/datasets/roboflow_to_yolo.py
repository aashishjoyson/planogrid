"""Harmonizes an already-YOLO-format Roboflow export into a single canonical
class, staged for merge_and_split().

Roboflow's own train/valid/test split is deliberately flattened back into
one pool here — the final split is redone once across every merged source by
merge_and_split(), so no single source ends up confined to one split. See
DESIGN.md §6.5, §9.
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from training.datasets.common import remap_yolo_source_to_single_class

logger = get_logger(__name__)

# Roboflow's own YOLO export always splits into these three subdirectories.
_ROBOFLOW_SPLITS = ("train", "valid", "test")


def convert(raw_dir: Path, staging_dir: Path) -> int:
    """`raw_dir` is a manifest.yaml entry's resolved `dest` (e.g.
    datasets/raw/roboflow_fyp_497), already downloaded via
    `python scripts/download_datasets.py --source roboflow`.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found — run "
            "`python scripts/download_datasets.py --source roboflow` first."
        )

    total = 0
    for split in _ROBOFLOW_SPLITS:
        images_dir = raw_dir / split / "images"
        labels_dir = raw_dir / split / "labels"
        if not images_dir.exists():
            continue
        total += remap_yolo_source_to_single_class(
            images_dir, labels_dir, staging_dir, source_prefix=f"{raw_dir.name}_{split}"
        )

    logger.info("roboflow_to_yolo: staged %d images from %s to %s", total, raw_dir, staging_dir)
    return total
