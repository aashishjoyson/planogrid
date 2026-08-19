"""Standalone evaluation + training report.

Ultralytics' own `model.train(..., plots=True)` already writes loss curves,
confusion matrix, and PR/F1/P/R curves directly into the experiment
directory during training-time validation (on the val split). This module
adds the one thing that doesn't happen automatically: a held-out *test*-
split evaluation (the split training itself never touches), plus a single
human-readable report tying the numbers to those auto-generated plots. See
DESIGN.md §9.
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from core.logging import get_logger
from training.experiment import Experiment

logger = get_logger(__name__)


def evaluate_on_test_split(experiment: Experiment, data_yaml: str) -> dict[str, float]:
    """Runs model.val() on the held-out test split and returns
    precision/recall/mAP50/mAP50-95/F1.
    """
    model = YOLO(str(experiment.best_weights_path))
    results = model.val(
        data=data_yaml,
        split="test",
        plots=True,
        project=str(experiment.root),
        name="test_eval",
        exist_ok=True,
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    metrics = {
        "precision": precision,
        "recall": recall,
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0,
    }

    (experiment.root / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info("Test-split evaluation for %s: %s", experiment.run_id, metrics)
    return metrics


def write_training_report(
    experiment: Experiment,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> Path:
    """Writes experiments/<detector>/<run_id>/report.md — one human-readable
    summary combining metrics with pointers to the plots Ultralytics already
    generated alongside this file.
    """
    lines = [
        f"# Training report — {experiment.detector} ({experiment.run_id})",
        "",
        "## Validation-split metrics (during training)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {key} | {value:.4f} |" for key, value in train_metrics.items()]

    if test_metrics:
        lines += ["", "## Held-out test-split metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {key} | {value:.4f} |" for key, value in test_metrics.items()]

    lines += [
        "",
        "## Plots",
        "",
        "Generated automatically by Ultralytics during training/validation, "
        "alongside this report:",
        "- `results.png` — loss and metric curves across all epochs",
        "- `confusion_matrix.png` / `confusion_matrix_normalized.png`",
        "- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`",
        "- `test_eval/` — plots from the held-out test-split run above",
        "- `weights/best.pt`, `weights/last.pt`",
    ]

    report_path = experiment.root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote training report to %s", report_path)
    return report_path
