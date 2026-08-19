"""Direct-HTTP adapter — plain GET plus optional zip extraction, for sources
not gated by Kaggle/Roboflow/HF.

manifest.yaml's `direct:` list is empty for the V1.0 baseline (reserved for
V2 extension datasets like Grozi-120/RPC/Freiburg — see DESIGN.md §19), but
this adapter is fully implemented so adding one later is a manifest entry,
not new code.
"""

from __future__ import annotations

import zipfile

import requests

from core.exceptions import DatasetDownloadError
from core.logging import get_logger
from datasets.adapters.base import DatasetAdapter, DatasetSpec, FetchResult

logger = get_logger(__name__)


class DirectAdapter(DatasetAdapter):
    source_name = "direct"

    def check_credentials(self) -> None:
        return  # no credentials needed by definition

    def fetch(self, spec: DatasetSpec, *, force: bool = False) -> FetchResult:
        dest = spec.dest
        if not force and self._already_fetched(dest):
            logger.info("Direct dataset '%s' already present at %s, skipping.", spec.name, dest)
            return FetchResult(spec.name, dest, resolved_version=None, already_present=True)

        url = spec.extra["url"]
        dest.mkdir(parents=True, exist_ok=True)
        archive_path = dest / "download.zip"

        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with archive_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(dest)
                archive_path.unlink()
        except Exception as exc:
            raise DatasetDownloadError(
                f"Failed to download '{spec.name}' from {url}: {exc}\nManual fallback: {url}"
            ) from exc

        logger.info("Direct dataset '%s' fetched to %s", spec.name, dest)
        return FetchResult(spec.name, dest, resolved_version=None, already_present=False)
