"""Configuration loading, in two deliberately separate mechanisms:

- `Settings` (pydantic-settings) — secrets and machine-specific values, sourced
  from .env / real environment variables. Never versioned, never referenced
  from a YAML file.
- `load_config()` — pipeline/model *behavior* config, sourced from
  configs/<name>.yaml with configs/profiles/<profile>.yaml deep-merged on top.
  Versioned, no secrets. Profile defaults to $PLANOGRID_PROFILE or "cloud_t4".

Nothing in this project reads os.environ directly outside this module — see
DESIGN.md §14.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

T = TypeVar("T", bound=BaseModel)


class Settings(BaseSettings):
    """Environment-backed secrets and machine-specific values.

    Field names map to env vars case-insensitively (kaggle_api_token <-> the
    KAGGLE_API_TOKEN this repo's .env uses). All optional at this layer —
    a missing credential is only an error at the point a dataset adapter
    actually needs it, so `--dry-run` and unrelated commands work without any
    keys configured at all.
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Kaggle — either the newer KAGGLE_API_TOKEN, or the classic username+key pair.
    kaggle_api_token: str | None = None
    kaggle_username: str | None = None
    kaggle_key: str | None = None

    # Roboflow — private key required for downloads; publishable key is
    # inference-only and deliberately never read by any adapter.
    roboflow_api_key: str | None = None
    roboflow_workspace: str | None = None

    # HuggingFace
    hf_token: str | None = None

    # Optional experiment tracking — training/callbacks.py enables W&B only if set.
    wandb_api_key: str | None = None

    planogrid_profile: str = "cloud_t4"
    planogrid_log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data


def load_config(name: str, schema: type[T], *, profile: str | None = None) -> T:
    """Load configs/<name>.yaml, deep-merge configs/profiles/<profile>.yaml's
    `name` section on top, and validate the result against `schema`.

    Raises ConfigError with the offending file path on any validation failure
    — never a bare pydantic.ValidationError with no context about which layer
    (base vs. profile) introduced the bad value.
    """
    resolved_profile = profile or get_settings().planogrid_profile

    base_path = CONFIG_DIR / f"{name}.yaml"
    profile_path = CONFIG_DIR / "profiles" / f"{resolved_profile}.yaml"

    base_data = _load_yaml(base_path)
    profile_data = _load_yaml(profile_path).get(name, {})
    merged = _deep_merge(base_data, profile_data)

    try:
        return schema.model_validate(merged)
    except Exception as exc:
        raise ConfigError(
            f"Invalid configuration for '{name}' (base={base_path}, "
            f"profile={resolved_profile}): {exc}"
        ) from exc
