#!/usr/bin/env python
"""Fetch Planogrid's training datasets from datasets/manifest.yaml.

    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --source kaggle
    python scripts/download_datasets.py --source kaggle --force
    python scripts/download_datasets.py --datasets sample
    python scripts/download_datasets.py --dry-run

Credentials come from .env (see .env.example) — never hardcoded, never
prompted for interactively. A missing credential fails loudly, naming the
exact env var and where to get it, rather than silently skipping. See
DESIGN.md §6.2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/download_datasets.py` to work without an editable
# install — Python only puts scripts/'s own directory on sys.path, not the
# repo root, when a script is run directly rather than via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import REPO_ROOT  # noqa: E402
from core.exceptions import DatasetCredentialError, DatasetDownloadError  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402
from datasets.adapters import ADAPTERS  # noqa: E402
from datasets.manifest import build_specs, load_manifest  # noqa: E402

configure_logging()
logger = get_logger(__name__)


def cmd_dry_run(manifest: dict) -> int:
    logger.info("Dry run — validating manifest and credentials, no network calls.\n")
    all_ok = True
    for source in sorted(ADAPTERS):
        specs = build_specs(manifest, source)
        if not specs:
            logger.info("[%s] no entries in manifest", source)
            continue

        adapter = ADAPTERS[source]()
        logger.info("[%s] %d dataset(s)", source, len(specs))
        try:
            adapter.check_credentials()
            logger.info("  credentials: OK")
        except DatasetCredentialError as exc:
            all_ok = False
            logger.warning("  credentials: MISSING — %s", exc)

        for spec in specs:
            logger.info(
                "  - %-28s -> %s  [%s]", spec.name, spec.dest.relative_to(REPO_ROOT), spec.license
            )

    sample_dest = REPO_ROOT / manifest["sample"]["dest"]
    has_samples = sample_dest.exists() and any(sample_dest.glob("*.jpg"))
    logger.info(
        "\n[sample] zero-credential path: %s", "present" if has_samples else "NOT YET POPULATED"
    )

    return 0 if all_ok else 1


def cmd_sample(manifest: dict) -> int:
    """The zero-credential path is a presence check, not a fetch — the sample
    images ship committed in git. See DESIGN.md §6.4, ADR-007: real CC0
    photos, never synthetic.
    """
    sample_dest = REPO_ROOT / manifest["sample"]["dest"]
    images = (
        sorted(sample_dest.glob("*.jpg"))
        + sorted(sample_dest.glob("*.jpeg"))
        + sorted(sample_dest.glob("*.png"))
    )
    if not images:
        logger.error(
            "No sample images found at %s. These ship committed in git — if "
            "you're seeing this, the clone or checkout may be incomplete.",
            sample_dest,
        )
        return 1
    logger.info(
        "Sample path OK — %d real (CC0-licensed) image(s) at %s, zero credentials required.",
        len(images),
        sample_dest,
    )
    return 0


def cmd_fetch(manifest: dict, sources: list[str], *, force: bool) -> int:
    failures: list[str] = []
    for source in sources:
        specs = build_specs(manifest, source)
        if not specs:
            logger.info("[%s] no entries in manifest, skipping", source)
            continue

        adapter = ADAPTERS[source]()
        try:
            adapter.check_credentials()
        except DatasetCredentialError as exc:
            logger.error("[%s] skipping all %d dataset(s): %s", source, len(specs), exc)
            failures.append(source)
            continue

        for spec in specs:
            try:
                result = adapter.fetch(spec, force=force)
                status = "already present" if result.already_present else "fetched"
                logger.info("[%s] %-28s %s -> %s", source, spec.name, status, result.dest)
            except DatasetDownloadError as exc:
                logger.error("[%s] %s: FAILED — %s", source, spec.name, exc)
                failures.append(f"{source}/{spec.name}")

    if failures:
        logger.error("\n%d item(s) failed: %s", len(failures), ", ".join(failures))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Fetch every source in the manifest.")
    mode.add_argument("--source", choices=sorted(ADAPTERS), help="Fetch only this source.")
    mode.add_argument(
        "--datasets",
        choices=["sample"],
        help="Zero-credential path — verify the bundled sample images.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and every source's credentials; no network calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination already looks populated.",
    )
    args = parser.parse_args()

    manifest = load_manifest()

    if args.datasets == "sample":
        return cmd_sample(manifest)
    if args.dry_run:
        return cmd_dry_run(manifest)

    sources = sorted(ADAPTERS) if args.all else [args.source]
    return cmd_fetch(manifest, sources, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
