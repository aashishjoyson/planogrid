"""Generates presentation-ready training-curve charts (loss + P/R/mAP) from
a run's results.csv — a saved, shareable artifact rather than something that
only exists in a live TensorBoard session. See plan addendum (2026-08-04).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

_BG = "#0B0E14"
_CARD = "#151920"
_TEXT = "#F2F1ED"
_MUTED = "#9C9FA8"
_GRID = "#2A2F3A"
_AMBER = "#F59E0B"
_BLUE = "#3B82F6"
_TEAL = "#14B8A6"
_RED = "#EF4444"


def _style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(_CARD)
    ax.set_title(title, color=_TEXT, fontsize=12, fontweight="bold", loc="left")
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.6)
    ax.set_xlabel("epoch", color=_MUTED, fontsize=9)
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)


def generate_training_charts(detector: str, results_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=_BG)

    ax1.plot(df["epoch"], df["train/box_loss"], color=_AMBER, linewidth=1.8, label="train box_loss")
    ax1.plot(df["epoch"], df["train/cls_loss"], color=_BLUE, linewidth=1.8, label="train cls_loss")
    if "val/box_loss" in df.columns:
        ax1.plot(
            df["epoch"],
            df["val/box_loss"],
            color=_AMBER,
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label="val box_loss",
        )
    if "val/cls_loss" in df.columns:
        ax1.plot(
            df["epoch"],
            df["val/cls_loss"],
            color=_BLUE,
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label="val cls_loss",
        )
    _style_axis(ax1, f"{detector} — training loss")

    metric_cols = {
        "metrics/precision(B)": ("precision", _TEAL),
        "metrics/recall(B)": ("recall", _BLUE),
        "metrics/mAP50(B)": ("mAP50", _AMBER),
        "metrics/mAP50-95(B)": ("mAP50-95", _RED),
    }
    for col, (label, color) in metric_cols.items():
        if col in df.columns:
            ax2.plot(df["epoch"], df[col], color=color, linewidth=1.8, label=label)
    ax2.set_ylim(0, 1)
    _style_axis(ax2, f"{detector} — validation metrics")

    fig.suptitle(
        f"Planogrid — {detector} detector training ({len(df)} epochs)",
        color=_TEXT,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{detector}_training_curves.png"
    fig.savefig(out_path, facecolor=_BG, dpi=150)
    plt.close(fig)

    logger.info("Saved training charts for '%s' to %s", detector, out_path)
    return out_path


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

    from core.config import REPO_ROOT
    from core.logging import configure_logging
    from training.experiment import latest_experiment

    configure_logging()
    if len(sys.argv) < 2:
        print("Usage: python -m training.plots <detector>")
        raise SystemExit(1)

    detector_name = sys.argv[1]
    experiment = latest_experiment(detector_name)
    if experiment is None:
        raise SystemExit(f"No experiment found for '{detector_name}'")

    generate_training_charts(
        detector_name,
        experiment.root / "results.csv",
        REPO_ROOT / "reports" / "training",
    )
