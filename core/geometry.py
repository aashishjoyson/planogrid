"""Pure geometry helpers: IoU, row clustering, and the row-aware Hungarian
assignment used to match price tags to products. See DESIGN.md §2.4, ADR-003.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.schemas import BBox


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes in the same coordinate space."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def containment_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner`'s area overlapping `outer` — asymmetric, unlike IoU.
    Used to discard a spuriously detected product box sitting inside a gap box.
    """
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if inner.area <= 0:
        return 0.0
    return intersection / inner.area


def vertical_gap(a: BBox, b: BBox) -> float:
    """Signed vertical gap between two boxes' y-bands; 0 if they overlap vertically."""
    if a.y2 < b.y1:
        return b.y1 - a.y2
    if b.y2 < a.y1:
        return a.y1 - b.y2
    return 0.0


def cluster_into_rows(boxes: list[BBox], tolerance: float = 0.6) -> list[list[int]]:
    """Group box indices into shelf rows by 1-D y-centroid proximity.

    Sorts by centroid y, then starts a new row whenever the gap to the next
    centroid exceeds `tolerance` times the running row's median box height —
    scale-relative so it works whether shelves are cropped tight or wide.
    Returns rows top-to-bottom, each a list of indices into `boxes` left-to-right.
    """
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: boxes[i].cy)
    rows: list[list[int]] = [[order[0]]]
    row_heights: list[list[float]] = [[boxes[order[0]].height]]

    for idx in order[1:]:
        current_row = rows[-1]
        median_height = float(np.median(row_heights[-1])) or 1.0
        prev_cy = boxes[current_row[-1]].cy
        gap = boxes[idx].cy - prev_cy
        if gap <= tolerance * median_height:
            current_row.append(idx)
            row_heights[-1].append(boxes[idx].height)
        else:
            rows.append([idx])
            row_heights.append([boxes[idx].height])

    return [sorted(row, key=lambda i: boxes[i].cx) for row in rows]


def solve_assignment(
    cost_matrix: np.ndarray, max_cost: float | None = None
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment via the Hungarian algorithm.

    Returns (row_index, col_index) pairs into `cost_matrix`. If `max_cost` is
    set, pairs costing more are dropped rather than forced — callers use this
    to mean "no plausible match" instead of pairing a tag to a product a shelf
    away just because the solver must produce a full assignment otherwise.
    """
    if cost_matrix.size == 0:
        return []

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    pairs = list(zip(row_idx.tolist(), col_idx.tolist(), strict=True))

    if max_cost is None:
        return pairs
    return [(r, c) for r, c in pairs if cost_matrix[r, c] <= max_cost]
