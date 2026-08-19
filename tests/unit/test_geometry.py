import numpy as np
import pytest

from core.geometry import cluster_into_rows, containment_ratio, iou, solve_assignment, vertical_gap
from core.schemas import BBox


def test_iou_identical_boxes_is_one():
    box = BBox(x1=0, y1=0, x2=10, y2=10)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=20, y1=20, x2=30, y2=30)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = BBox(x1=0, y1=0, x2=10, y2=10)  # area 100
    b = BBox(x1=5, y1=0, x2=15, y2=10)  # area 100, intersection 5x10=50
    assert iou(a, b) == pytest.approx(50 / 150)


def test_containment_ratio_fully_inside():
    inner = BBox(x1=2, y1=2, x2=8, y2=8)
    outer = BBox(x1=0, y1=0, x2=10, y2=10)
    assert containment_ratio(inner, outer) == pytest.approx(1.0)


def test_containment_ratio_no_overlap_is_zero():
    inner = BBox(x1=100, y1=100, x2=110, y2=110)
    outer = BBox(x1=0, y1=0, x2=10, y2=10)
    assert containment_ratio(inner, outer) == 0.0


def test_vertical_gap_overlapping_boxes_is_zero():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=0, y1=5, x2=10, y2=15)
    assert vertical_gap(a, b) == 0.0


def test_vertical_gap_separated_boxes():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=0, y1=20, x2=10, y2=30)
    assert vertical_gap(a, b) == 10.0


def test_cluster_into_rows_separates_two_shelves():
    row_top = [BBox(x1=x, y1=0, x2=x + 15, y2=20) for x in (0, 20, 40)]
    row_bottom = [BBox(x1=x, y1=120, x2=x + 15, y2=140) for x in (0, 20, 40)]
    boxes = row_bottom + row_top  # deliberately out of visual order

    rows = cluster_into_rows(boxes)

    assert len(rows) == 2
    assert all(boxes[i].cy < 60 for i in rows[0])  # top row returned first
    assert all(boxes[i].cy > 60 for i in rows[1])


def test_cluster_into_rows_orders_left_to_right_within_a_row():
    boxes = [
        BBox(x1=40, y1=0, x2=55, y2=20),
        BBox(x1=0, y1=0, x2=15, y2=20),
        BBox(x1=20, y1=0, x2=35, y2=20),
    ]
    rows = cluster_into_rows(boxes)
    assert len(rows) == 1
    assert rows[0] == [1, 2, 0]


def test_cluster_into_rows_empty_input():
    assert cluster_into_rows([]) == []


def test_solve_assignment_finds_optimal_pairing():
    cost = np.array([[1.0, 10.0], [10.0, 1.0]])
    pairs = solve_assignment(cost)
    assert set(pairs) == {(0, 0), (1, 1)}


def test_solve_assignment_drops_pairs_over_max_cost():
    # A 2x2 matrix forces a full assignment either way; pick costs so the
    # globally optimal pairing (0,0)+(1,1) [5+300=305] genuinely beats the
    # alternative (0,1)+(1,0) [300+10=310] — otherwise the solver would
    # legitimately choose the other pairing and the "drop" assertion below
    # would be testing the wrong thing.
    cost = np.array([[5.0, 300.0], [10.0, 300.0]])
    pairs = solve_assignment(cost, max_cost=50.0)
    assert (0, 0) in pairs
    assert (1, 1) not in pairs


def test_solve_assignment_empty_matrix_returns_empty():
    assert solve_assignment(np.empty((0, 0))) == []
