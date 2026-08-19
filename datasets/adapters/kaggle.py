"""Kaggle adapter — wraps kaggle.api.kaggle_api_extended.KaggleApi.

Supports both the current KAGGLE_API_TOKEN and the classic
KAGGLE_USERNAME/KAGGLE_KEY pair (core.config.Settings exposes both; whichever
is set wins, token first).
"""

from __future__ import annotations

import os

from core.config import get_settings
from core.exceptions import DatasetCredentialError, DatasetDownloadError
from core.logging import get_logger
from datasets.adapters.base import DatasetAdapter, DatasetSpec, FetchResult

logger = get_logger(__name__)


class KaggleAdapter(DatasetAdapter):
    source_name = "kaggle"

    def check_credentials(self) -> None:
        settings = get_settings()
        has_token = bool(settings.kaggle_api_token)
        has_legacy = bool(settings.kaggle_username and settings.kaggle_key)
        if not (has_token or has_legacy):
            raise DatasetCredentialError(
                "Kaggle credentials missing. Set KAGGLE_API_TOKEN in .env "
                "(kaggle.com -> Settings -> API -> 'Create New Token'), or the "
                "legacy KAGGLE_USERNAME + KAGGLE_KEY pair. See .env.example."
            )

    def _authenticated_api(self):  # type: ignore[no-untyped-def]
        # The kaggle package reads credentials from os.environ at
        # authenticate()-time; pydantic-settings loading .env doesn't itself
        # populate os.environ, so we bridge the two explicitly here.
        settings = get_settings()
        if settings.kaggle_api_token:
            os.environ.setdefault("KAGGLE_API_TOKEN", settings.kaggle_api_token)
        if settings.kaggle_username and settings.kaggle_key:
            os.environ.setdefault("KAGGLE_USERNAME", settings.kaggle_username)
            os.environ.setdefault("KAGGLE_KEY", settings.kaggle_key)

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api

    def fetch(self, spec: DatasetSpec, *, force: bool = False) -> FetchResult:
        dest = spec.dest
        if not force and self._already_fetched(dest):
            logger.info("Kaggle dataset '%s' already present at %s, skipping.", spec.name, dest)
            return FetchResult(spec.name, dest, resolved_version=None, already_present=True)

        self.check_credentials()
        api = self._authenticated_api()
        dataset_ref = spec.extra["dataset_ref"]

        dest.mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_download_files(dataset_ref, path=str(dest), unzip=True, quiet=False)
        except Exception as exc:
            raise DatasetDownloadError(
                f"Failed to download Kaggle dataset '{dataset_ref}' for '{spec.name}': {exc}\n"
                f"Manual fallback: https://www.kaggle.com/datasets/{dataset_ref}"
            ) from exc

        logger.info("Kaggle dataset '%s' fetched to %s", spec.name, dest)
        return FetchResult(spec.name, dest, resolved_version=None, already_present=False)
