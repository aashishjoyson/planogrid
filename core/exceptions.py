"""Typed exception hierarchy. Callers catch the specific subclass they can
handle, or `PlanogridError` as a catch-all — never a bare Exception.
"""

from __future__ import annotations


class PlanogridError(Exception):
    """Base class for every exception this project raises intentionally."""


class ConfigError(PlanogridError):
    """A config file is missing, malformed, or fails schema validation."""


class RegistryError(PlanogridError):
    """Base class for detector/OCR-engine registry lookup failures."""


class ComponentAlreadyRegisteredError(RegistryError):
    """Two implementations tried to register under the same name."""


class ComponentNotRegisteredError(RegistryError):
    """Config asked for a detector/OCR engine name that was never registered."""


class DatasetError(PlanogridError):
    """Base class for dataset acquisition failures."""


class DatasetCredentialError(DatasetError):
    """A required API credential is missing or invalid."""


class DatasetDownloadError(DatasetError):
    """A dataset source was reachable but the download/export failed."""


class ModelError(PlanogridError):
    """Base class for model-loading and inference failures."""


class WeightsNotFoundError(ModelError):
    """A checkpoint path was configured but the file doesn't exist."""


class PipelineError(PlanogridError):
    """A pipeline stage failed in a way callers should be able to catch specifically."""
