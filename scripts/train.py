#!/usr/bin/env python
"""Train one of Planogrid's three detectors end to end.

    python scripts/train.py --detector gap
    python scripts/train.py --detector tag --resume
    python scripts/train.py --detector product --profile cloud_t4

Data preparation (for gap/tag, via training.datasets.prepare) and the
Ultralytics training loop both run as part of this one command — no separate
prep step required. See DESIGN.md §9.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train.py` to work without an editable install — see
# the same note in scripts/download_datasets.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logging import configure_logging, get_logger  # noqa: E402
from training.evaluate import evaluate_on_test_split, write_training_report  # noqa: E402
from training.trainer import train_detector  # noqa: E402

configure_logging()
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detector", required=True, choices=["product", "gap", "tag"])
    parser.add_argument(
        "--resume", action="store_true", help="Resume the most recent experiment for this detector."
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Compute profile override (defaults to $PLANOGRID_PROFILE or cloud_t4).",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip the held-out test-split evaluation + training report.",
    )
    args = parser.parse_args()

    result = train_detector(args.detector, resume=args.resume, profile=args.profile)

    test_metrics = None
    if not args.skip_test_eval and args.detector != "product":
        # "product" trains against ultralytics' bundled SKU-110K.yaml, which
        # doesn't expose the same processed-dataset path this project's own
        # gap/tag data.yaml does — its held-out check is left to
        # benchmarks/ (Phase 7) instead of a second .val() pass here.
        from training.datasets.prepare import PROCESSED_ROOT

        data_yaml = str(PROCESSED_ROOT / args.detector / "data.yaml")
        test_metrics = evaluate_on_test_split(result.experiment, data_yaml)

    write_training_report(result.experiment, result.metrics, test_metrics)

    logger.info(
        "Done. Best weights: %s | mAP50=%.4f mAP50-95=%.4f",
        result.best_weights,
        result.metrics["map50"],
        result.metrics["map50_95"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
