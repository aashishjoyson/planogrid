"""DatasetAdapter ABC — one fetch() and one check_credentials() method per
source. Each concrete adapter owns its own fetch mechanics entirely; nothing
outside datasets/adapters/ knows how a Kaggle download differs from a
Roboflow one. See DESIGN.md §6.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    """One manifest.yaml entry. `extra` carries whatever fields are specific
    to the source (workspace/project/version for Roboflow, dataset_ref for
    Kaggle, repo_id/repo_type for HuggingFace, url for direct).
    """

    name: str
    dest: Path
    license: str
    role: str
    description: str
    extra: dict[str, Any]


@dataclass(frozen=True)
class FetchResult:
    spec_name: str
    dest: Path
    resolved_version: str | None
    already_present: bool


class DatasetAdapter(ABC):
    source_name: str

    @abstractmethod
    def check_credentials(self) -> None:
        """Raise DatasetCredentialError naming the exact env var and where to
        get it if something required is missing. Called for every spec during
        `--dry-run`, so credential problems surface without touching the
        network.
        """

    @abstractmethod
    def fetch(self, spec: DatasetSpec, *, force: bool = False) -> FetchResult:
        """Download (+ extract) `spec` into `spec.dest`. Idempotent: if dest
        already looks populated and force=False, return without a network call.
        """

    def _already_fetched(self, dest: Path) -> bool:
        # Ignore dotfiles (e.g. RoboflowAdapter's .manifest_resolved.json
        # marker) — a marker left behind by a failed/partial fetch must not
        # be mistaken for real dataset content on the next run.
        if not dest.exists():
            return False
        return any(not item.name.startswith(".") for item in dest.iterdir())
