import pytest
from pydantic import BaseModel

from core.config import Settings, _deep_merge, load_config
from core.exceptions import ConfigError


def test_deep_merge_overrides_nested_keys_without_dropping_siblings():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 99, "z": 3}}
    assert _deep_merge(base, override) == {"a": 1, "nested": {"x": 1, "y": 99, "z": 3}}


class _DummySchema(BaseModel):
    value: int
    label: str = "default"


def test_load_config_merges_profile_over_base(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    (configs_dir / "profiles").mkdir(parents=True)
    (configs_dir / "dummy.yaml").write_text("value: 1\nlabel: base\n")
    (configs_dir / "profiles" / "test_profile.yaml").write_text("dummy:\n  value: 2\n")
    monkeypatch.setattr("core.config.CONFIG_DIR", configs_dir)

    result = load_config("dummy", _DummySchema, profile="test_profile")

    assert result.value == 2  # overridden by the profile
    assert result.label == "base"  # untouched, kept from base


def test_load_config_falls_back_to_schema_defaults_when_files_absent(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    monkeypatch.setattr("core.config.CONFIG_DIR", configs_dir)

    class _WithDefaults(BaseModel):
        value: int = 42

    assert load_config("nonexistent", _WithDefaults, profile="whatever").value == 42


def test_load_config_raises_config_error_on_schema_violation(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "dummy.yaml").write_text("label: only-label-no-required-value\n")
    monkeypatch.setattr("core.config.CONFIG_DIR", configs_dir)

    with pytest.raises(ConfigError):
        load_config("dummy", _DummySchema, profile="none")


def test_load_config_rejects_non_mapping_yaml(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "dummy.yaml").write_text("- just\n- a\n- list\n")
    monkeypatch.setattr("core.config.CONFIG_DIR", configs_dir)

    with pytest.raises(ConfigError):
        load_config("dummy", _DummySchema)


def test_settings_reads_from_environment_not_the_repos_real_env_file(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token-value")
    monkeypatch.setenv("PLANOGRID_PROFILE", "cpu")

    # _env_file=None isolates this test from the repo's real .env — otherwise
    # it would silently assert against whatever real secrets happen to be
    # configured on the machine running the test.
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.kaggle_api_token == "test-token-value"
    assert settings.planogrid_profile == "cpu"


def test_settings_credential_fields_default_to_none(monkeypatch):
    for var in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "ROBOFLOW_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.kaggle_api_token is None
    assert settings.roboflow_api_key is None
    assert settings.hf_token is None
