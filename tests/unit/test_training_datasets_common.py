from pathlib import Path

import pytest

from training.datasets.common import (
    merge_and_split,
    remap_yolo_source_to_single_class,
    write_staged_example,
    xyxy_to_yolo,
)


def test_xyxy_to_yolo_normalizes_correctly():
    box = xyxy_to_yolo(x1=100, y1=100, x2=300, y2=200, image_w=1000, image_h=500, class_id=0)
    assert box.class_id == 0
    assert box.cx == pytest.approx(0.2)
    assert box.cy == pytest.approx(0.3)
    assert box.w == pytest.approx(0.2)
    assert box.h == pytest.approx(0.2)


def test_xyxy_to_yolo_clamps_out_of_bounds_coords():
    box = xyxy_to_yolo(x1=-50, y1=-50, x2=1200, y2=600, image_w=1000, image_h=500, class_id=0)
    assert 0.0 <= box.cx <= 1.0
    assert 0.0 <= box.cy <= 1.0
    assert box.w <= 1.0
    assert box.h <= 1.0


def test_xyxy_to_yolo_handles_swapped_corners():
    # exterior points aren't always top-left/bottom-right in source data
    box = xyxy_to_yolo(x1=300, y1=200, x2=100, y2=100, image_w=1000, image_h=500, class_id=0)
    assert box.cx == pytest.approx(0.2)
    assert box.cy == pytest.approx(0.3)


def _make_fake_image(path: Path) -> None:
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")


def test_write_staged_example_creates_prefixed_files(tmp_path):
    staging_dir = tmp_path / "staging"
    src_image = tmp_path / "source" / "001.jpg"
    src_image.parent.mkdir(parents=True)
    _make_fake_image(src_image)

    box = xyxy_to_yolo(0, 0, 10, 10, 100, 100, class_id=0)
    write_staged_example(staging_dir, "mysrc", "001", src_image, [box])

    assert (staging_dir / "images" / "mysrc__001.jpg").exists()
    label_content = (staging_dir / "labels" / "mysrc__001.txt").read_text()
    assert label_content.startswith("0 ")


def test_write_staged_example_empty_boxes_writes_empty_label(tmp_path):
    staging_dir = tmp_path / "staging"
    src_image = tmp_path / "source" / "002.jpg"
    src_image.parent.mkdir(parents=True)
    _make_fake_image(src_image)

    write_staged_example(staging_dir, "mysrc", "002", src_image, [])

    assert (staging_dir / "labels" / "mysrc__002.txt").read_text() == ""


def test_remap_yolo_source_forces_target_class_id(tmp_path):
    images_dir = tmp_path / "src" / "images"
    labels_dir = tmp_path / "src" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    _make_fake_image(images_dir / "a.jpg")
    (labels_dir / "a.txt").write_text("5 0.5 0.5 0.2 0.2\n7 0.1 0.1 0.05 0.05\n")

    staging_dir = tmp_path / "staged"
    count = remap_yolo_source_to_single_class(
        images_dir, labels_dir, staging_dir, source_prefix="src", target_class_id=0
    )

    assert count == 1
    label_lines = (staging_dir / "labels" / "src__a.txt").read_text().splitlines()
    assert all(line.startswith("0 ") for line in label_lines)
    assert len(label_lines) == 2


def test_merge_and_split_pools_across_sources_and_produces_data_yaml(tmp_path):
    staging_a = tmp_path / "stage_a"
    staging_b = tmp_path / "stage_b"
    for staging_dir, count in ((staging_a, 6), (staging_b, 4)):
        for i in range(count):
            src = tmp_path / f"raw_{staging_dir.name}_{i}.jpg"
            _make_fake_image(src)
            box = xyxy_to_yolo(0, 0, 10, 10, 100, 100, class_id=0)
            write_staged_example(staging_dir, staging_dir.name, str(i), src, [box])

    output_dir = tmp_path / "merged"
    counts = merge_and_split(
        [staging_a, staging_b],
        output_dir,
        class_name="empty_gap",
        train=0.8,
        val=0.1,
        test=0.1,
        seed=0,
    )

    assert sum(counts.values()) == 10
    assert (output_dir / "data.yaml").exists()
    data_yaml_text = (output_dir / "data.yaml").read_text()
    assert "empty_gap" in data_yaml_text
    for split in ("train", "val", "test"):
        assert len(list((output_dir / "images" / split).iterdir())) == counts[split]


def test_merge_and_split_rejects_fractions_not_summing_to_one(tmp_path):
    with pytest.raises(ValueError, match="sum to 1.0"):
        merge_and_split([], tmp_path / "out", class_name="x", train=0.5, val=0.3, test=0.3)


def test_merge_and_split_raises_when_no_examples_found(tmp_path):
    empty_staging = tmp_path / "empty_stage"
    empty_staging.mkdir()
    with pytest.raises(ValueError, match="No staged examples"):
        merge_and_split([empty_staging], tmp_path / "out", class_name="x")


def test_merge_and_split_is_deterministic_with_fixed_seed(tmp_path):
    staging = tmp_path / "stage"
    for i in range(20):
        src = tmp_path / f"raw_{i}.jpg"
        _make_fake_image(src)
        write_staged_example(staging, "s", str(i), src, [])

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    counts1 = merge_and_split([staging], out1, class_name="x", seed=42)
    counts2 = merge_and_split([staging], out2, class_name="x", seed=42)

    names1 = sorted(p.name for p in (out1 / "images" / "train").iterdir())
    names2 = sorted(p.name for p in (out2 / "images" / "train").iterdir())
    assert counts1 == counts2
    assert names1 == names2
