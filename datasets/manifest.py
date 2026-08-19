"""Loads and parses datasets/manifest.yaml into DatasetSpec objects.

Kept separate from scripts/download_datasets.py so this logic is importable
and unit-testable like everything else in datasets/ — the script itself
stays a thin CLI wrapper. See DESIGN.md §6.2, §4.1.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config import REPO_ROOT
from core.exceptions import ConfigError
from datasets.adapters import DatasetSpec

MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.yaml"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_specs(manifest: dict, source: str) -> list[DatasetSpec]:
    entries = manifest.get(source) or []
    specs = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        try:
            name = entry.pop("name")
            dest = REPO_ROOT / entry.pop("dest")
            license_ = entry.pop("license")
            role = entry.pop("role")
            description = entry.pop("description")
        except KeyError as exc:
            raise ConfigError(
                f"manifest.yaml entry under '{source}' is missing required field {exc}: {entry}"
            ) from exc
        specs.append(
            DatasetSpec(
                name=name,
                dest=dest,
                license=license_,
                role=role,
                description=description,
                extra=entry,
            )
        )
    return specs
