"""Shared helpers for label converters: writing YOLO label files and merging
multiple single-class staged sources into one final, split dataset. See
DESIGN.md §6.5, §9.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def as_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int, class_id: int
) -> YoloBox:
    x1, x2 = sorted((max(0.0, min(x1, image_w)), max(0.0, min(x2, image_w))))
    y1, y2 = sorted((max(0.0, min(y1, image_h)), max(0.0, min(y2, image_h))))
    return YoloBox(
        class_id=class_id,
        cx=(x1 + x2) / 2 / image_w,
        cy=(y1 + y2) / 2 / image_h,
        w=(x2 - x1) / image_w,
        h=(y2 - y1) / image_h,
    )


def write_staged_example(
    staging_dir: Path, source_prefix: str, stem: str, image_path: Path, boxes: list[YoloBox]
) -> None:
    images_dir = staging_dir / "images"
    labels_dir = staging_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    dest_name = f"{source_prefix}__{stem}{image_path.suffix.lower()}"
    shutil.copy2(image_path, images_dir / dest_name)
    label_lines = "\n".join(box.as_line() for box in boxes)
    (labels_dir / f"{source_prefix}__{stem}.txt").write_text(
        label_lines + ("\n" if label_lines else ""), encoding="utf-8"
    )


def remap_yolo_source_to_single_class(
    source_images_dir: Path,
    source_labels_dir: Path,
    staging_dir: Path,
    source_prefix: str,
    target_class_id: int = 0,
) -> int:
    """Copy an already-YOLO-format source into `staging_dir`, forcing every
    label row's class index to `target_class_id`. Every Roboflow source used
    here is independently single-class already — this makes that explicit
    and defends against a source's own class index not being 0.
    """
    count = 0
    for image_path in sorted(source_images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        boxes: list[YoloBox] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _original_class, cx, cy, w, h = parts
                boxes.append(YoloBox(target_class_id, float(cx), float(cy), float(w), float(h)))
        write_staged_example(staging_dir, source_prefix, image_path.stem, image_path, boxes)
        count += 1
    return count


def merge_and_split(
    staging_dirs: list[Path],
    output_dir: Path,
    class_name: str,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
) -> dict[str, int]:
    """Pool every staged (images/, labels/) source, shuffle with a fixed
    seed, and split ONCE across the combined set — not per-source — so no
    single source ends up confined to one split. Writes the final
    Ultralytics-ready datasets/processed/<detector>/ layout + data.yaml.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {train}+{val}+{test}")

    examples: list[tuple[Path, Path]] = []
    for staging_dir in staging_dirs:
        images_dir = staging_dir / "images"
        labels_dir = staging_dir / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            examples.append((image_path, labels_dir / f"{image_path.stem}.txt"))

    if not examples:
        raise ValueError(
            f"No staged examples found across {staging_dirs} — run the converters first."
        )

    rng = random.Random(seed)
    rng.shuffle(examples)

    n = len(examples)
    n_train = round(n * train)
    n_val = round(n * val)
    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }

    counts: dict[str, int] = {}
    for split_name, split_examples in splits.items():
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_examples:
            shutil.copy2(image_path, images_out / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, labels_out / label_path.name)
            else:
                (labels_out / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
        counts[split_name] = len(split_examples)

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: class_name},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    logger.info("Merged %d examples into %s: %s", n, output_dir, counts)
    return counts
