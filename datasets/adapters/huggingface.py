"""HuggingFace adapter — wraps huggingface_hub.snapshot_download for dataset
or model repos (manifest.yaml's `repo_type` field picks which).
"""

from __future__ import annotations

from core.config import get_settings
from core.exceptions import DatasetCredentialError, DatasetDownloadError
from core.logging import get_logger
from datasets.adapters.base import DatasetAdapter, DatasetSpec, FetchResult

logger = get_logger(__name__)


class HuggingFaceAdapter(DatasetAdapter):
    source_name = "huggingface"

    def check_credentials(self) -> None:
        # Most HF dataset repos are public and downloadable anonymously; a
        # token is still required here for consistency with the other
        # adapters and to avoid anonymous rate limits — see DESIGN.md §6.2.
        settings = get_settings()
        if not settings.hf_token:
            raise DatasetCredentialError(
                "HuggingFace credentials missing. Set HF_TOKEN in .env "
                "(huggingface.co -> Settings -> Access Tokens -> New token, "
                "read scope). See .env.example."
            )

    def fetch(self, spec: DatasetSpec, *, force: bool = False) -> FetchResult:
        dest = spec.dest
        if not force and self._already_fetched(dest):
            logger.info("HuggingFace repo '%s' already present at %s, skipping.", spec.name, dest)
            return FetchResult(spec.name, dest, resolved_version=None, already_present=True)

        self.check_credentials()
        settings = get_settings()

        from huggingface_hub import snapshot_download

        repo_id = spec.extra["repo_id"]
        repo_type = spec.extra.get("repo_type", "dataset")

        dest.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                token=settings.hf_token,
                local_dir=str(dest),
            )
        except Exception as exc:
            kind_path = "datasets/" if repo_type == "dataset" else ""
            raise DatasetDownloadError(
                f"Failed to download HuggingFace {repo_type} '{repo_id}' for '{spec.name}': {exc}\n"
                f"Manual fallback: https://huggingface.co/{kind_path}{repo_id}"
            ) from exc

        logger.info("HuggingFace repo '%s' fetched to %s", spec.name, dest)
        return FetchResult(spec.name, dest, resolved_version=None, already_present=False)
