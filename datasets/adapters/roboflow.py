"""Roboflow adapter — wraps the roboflow package.

Always authenticates with the PRIVATE api key (ROBOFLOW_API_KEY). The
publishable/inferencejs key is deliberately never read here — it cannot
download datasets, only run inference on already-deployed models. See
DESIGN.md §2 and ADR-007's sibling note in .env.example.
"""

from __future__ import annotations

import json

from core.config import get_settings
from core.exceptions import DatasetCredentialError, DatasetDownloadError
from core.logging import get_logger
from datasets.adapters.base import DatasetAdapter, DatasetSpec, FetchResult

logger = get_logger(__name__)


class RoboflowAdapter(DatasetAdapter):
    source_name = "roboflow"

    def check_credentials(self) -> None:
        settings = get_settings()
        if not settings.roboflow_api_key:
            raise DatasetCredentialError(
                "Roboflow credentials missing. Set ROBOFLOW_API_KEY to your "
                "PRIVATE API key (app.roboflow.com -> Settings -> API Keys) — "
                "not the 'publishable' key, which cannot download datasets. "
                "See .env.example."
            )

    def fetch(self, spec: DatasetSpec, *, force: bool = False) -> FetchResult:
        dest = spec.dest
        if not force and self._already_fetched(dest):
            logger.info("Roboflow dataset '%s' already present at %s, skipping.", spec.name, dest)
            return FetchResult(spec.name, dest, resolved_version=None, already_present=True)

        self.check_credentials()
        settings = get_settings()

        from roboflow import Roboflow

        workspace = spec.extra["workspace"]
        project_slug = spec.extra["project"]
        fmt = spec.extra.get("format", "yolov8")
        requested_version = spec.extra.get("version", "latest")

        try:
            rf = Roboflow(api_key=settings.roboflow_api_key)
            project = rf.workspace(workspace).project(project_slug)
            versions = project.versions()
            if not versions:
                raise DatasetDownloadError(
                    f"Roboflow project '{workspace}/{project_slug}' has no published versions."
                )
            if requested_version == "latest":
                version_obj = max(versions, key=lambda v: int(v.version))
            else:
                match = next(
                    (v for v in versions if int(v.version) == int(requested_version)), None
                )
                if match is None:
                    raise DatasetDownloadError(
                        f"Roboflow project '{workspace}/{project_slug}' has no version "
                        f"{requested_version}. Available: {sorted(int(v.version) for v in versions)}"
                    )
                version_obj = match

            # roboflow's own download() treats an already-existing destination
            # directory as "already downloaded" and silently no-ops — it must
            # create the leaf directory itself. Only the parent may exist.
            dest.parent.mkdir(parents=True, exist_ok=True)
            version_obj.download(fmt, location=str(dest))
        except DatasetDownloadError:
            raise
        except Exception as exc:
            raise DatasetDownloadError(
                f"Failed to download Roboflow dataset '{workspace}/{project_slug}' "
                f"for '{spec.name}': {exc}\n"
                f"Manual fallback: https://universe.roboflow.com/{workspace}/{project_slug}"
            ) from exc

        resolved = str(version_obj.version)
        (dest / ".manifest_resolved.json").write_text(
            json.dumps({"name": spec.name, "resolved_version": resolved}, indent=2),
            encoding="utf-8",
        )
        logger.info("Roboflow dataset '%s' fetched to %s (version %s)", spec.name, dest, resolved)
        return FetchResult(spec.name, dest, resolved_version=resolved, already_present=False)
