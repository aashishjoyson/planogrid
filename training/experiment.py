"""Experiment directory lifecycle: each training run gets its own
timestamped directory under experiments/<detector>/, with the resolved
config snapshotted alongside the weights/metrics/plots it produces. See
DESIGN.md §9.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from core.config import REPO_ROOT
from core.logging import get_logger

logger = get_logger(__name__)

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


@dataclass(frozen=True)
class Experiment:
    """One training run's on-disk home: experiments/<detector>/<run_id>/."""

    detector: str
    run_id: str
    root: Path

    @property
    def weights_dir(self) -> Path:
        return self.root / "weights"

    @property
    def config_snapshot_path(self) -> Path:
        return self.root / "config_snapshot.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def best_weights_path(self) -> Path:
        return self.weights_dir / "best.pt"

    @property
    def last_weights_path(self) -> Path:
        return self.weights_dir / "last.pt"

    def snapshot_config(self, resolved_config: dict[str, Any]) -> None:
        self.config_snapshot_path.write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
        )

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    def promote_to_checkpoints(self, checkpoints_dir: Path | None = None) -> Path:
        """Copy this run's best.pt into checkpoints/<detector>/best.pt — the
        stable path configs/model.yaml points at, decoupled from any
        specific run's timestamp. See DESIGN.md §7.
        """
        dest_dir = checkpoints_dir or (REPO_ROOT / "checkpoints" / self.detector)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "best.pt"
        shutil.copy2(self.best_weights_path, dest)
        logger.info("Promoted %s -> %s", self.best_weights_path, dest)
        return dest


def create_experiment(detector: str) -> Experiment:
    """Creates experiments/<detector>/<timestamp>_<git_sha>/weights/ and
    returns a handle to it. Never overwrites a prior run — each call gets a
    fresh, uniquely-named directory.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_git_short_sha()}"
    root = EXPERIMENTS_ROOT / detector / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "weights").mkdir()

    logger.info("Created experiment: %s", root)
    return Experiment(detector=detector, run_id=run_id, root=root)


def latest_experiment(detector: str) -> Experiment | None:
    """Returns the most recently created experiment for `detector`, or None
    if there isn't one yet — used to resume training without needing to know
    the exact run_id.
    """
    detector_root = EXPERIMENTS_ROOT / detector
    if not detector_root.exists():
        return None
    run_dirs = sorted((p for p in detector_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not run_dirs:
        return None
    latest = run_dirs[-1]
    return Experiment(detector=detector, run_id=latest.name, root=latest)
