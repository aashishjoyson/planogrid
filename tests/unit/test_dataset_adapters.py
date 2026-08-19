"""Unit tests for datasets/adapters/*. Every SDK call is mocked — these tests
never touch the real network or require real credentials, matching CI (see
.github/workflows/ci.yml).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import Settings
from core.exceptions import ConfigError, DatasetCredentialError, DatasetDownloadError
from datasets.adapters import DatasetSpec
from datasets.adapters.direct import DirectAdapter
from datasets.adapters.huggingface import HuggingFaceAdapter
from datasets.adapters.kaggle import KaggleAdapter
from datasets.adapters.roboflow import RoboflowAdapter
from datasets.manifest import build_specs


def _settings(**overrides: str) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


def _clear_real_credential_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Settings(_env_file=None)` skips *this repo's* .env, but pydantic-
    settings still reads real os.environ — and importing `roboflow` (see the
    fetch tests below, which patch roboflow.Roboflow and so must import the
    real package) calls its own internal load_dotenv(), leaking this repo's
    actual .env values into the process for every test that runs afterward.
    "Raises when missing" tests need the real env cleared, not just the file
    skipped, or they become order-dependent on which test ran first.
    """
    for var in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "ROBOFLOW_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


# --- Kaggle ---


def test_kaggle_check_credentials_raises_when_nothing_set(monkeypatch):
    _clear_real_credential_env_vars(monkeypatch)
    monkeypatch.setattr("datasets.adapters.kaggle.get_settings", lambda: _settings())
    with pytest.raises(DatasetCredentialError):
        KaggleAdapter().check_credentials()


def test_kaggle_check_credentials_passes_with_token(monkeypatch):
    monkeypatch.setattr(
        "datasets.adapters.kaggle.get_settings", lambda: _settings(kaggle_api_token="abc")
    )
    KaggleAdapter().check_credentials()  # must not raise


def test_kaggle_check_credentials_passes_with_legacy_pair(monkeypatch):
    monkeypatch.setattr(
        "datasets.adapters.kaggle.get_settings",
        lambda: _settings(kaggle_username="me", kaggle_key="k"),
    )
    KaggleAdapter().check_credentials()


def test_kaggle_fetch_skips_network_when_dest_already_populated(tmp_path):
    dest = tmp_path / "already_here"
    dest.mkdir()
    (dest / "existing.jpg").write_bytes(b"x")

    spec = DatasetSpec(
        name="x", dest=dest, license="CC0", role="r", description="d", extra={"dataset_ref": "a/b"}
    )
    result = KaggleAdapter().fetch(spec)

    assert result.already_present is True


# --- Roboflow ---


def test_roboflow_check_credentials_raises_when_missing(monkeypatch):
    _clear_real_credential_env_vars(monkeypatch)
    monkeypatch.setattr("datasets.adapters.roboflow.get_settings", lambda: _settings())
    with pytest.raises(DatasetCredentialError, match="PRIVATE"):
        RoboflowAdapter().check_credentials()


def test_roboflow_fetch_resolves_latest_version(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "datasets.adapters.roboflow.get_settings", lambda: _settings(roboflow_api_key="key")
    )

    version_1 = MagicMock(version="1")
    version_2 = MagicMock(version="2")
    # The real roboflow SDK creates `location` itself during download() (and
    # in fact no-ops if it already exists — see the adapter's comment on
    # dest.parent.mkdir). Simulate that side effect so this mock behaves like
    # the real call the adapter's subsequent .manifest_resolved.json write
    # depends on.
    version_2.download.side_effect = lambda *_a, location, **_kw: Path(location).mkdir(
        parents=True, exist_ok=True
    )
    project = MagicMock()
    project.versions.return_value = [version_1, version_2]
    workspace = MagicMock()
    workspace.project.return_value = project
    rf_instance = MagicMock()
    rf_instance.workspace.return_value = workspace

    spec = DatasetSpec(
        name="x",
        dest=tmp_path / "rf_dest",
        license="CC BY 4.0",
        role="r",
        description="d",
        extra={"workspace": "ws", "project": "proj", "version": "latest", "format": "yolov8"},
    )

    assert not spec.dest.exists()  # adapter must not pre-create the leaf itself

    with patch("roboflow.Roboflow", return_value=rf_instance):
        result = RoboflowAdapter().fetch(spec)

    version_2.download.assert_called_once_with("yolov8", location=str(spec.dest))
    assert result.resolved_version == "2"


def test_roboflow_fetch_raises_on_unknown_pinned_version(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "datasets.adapters.roboflow.get_settings", lambda: _settings(roboflow_api_key="key")
    )
    project = MagicMock()
    project.versions.return_value = [MagicMock(version="1")]
    workspace = MagicMock()
    workspace.project.return_value = project
    rf_instance = MagicMock()
    rf_instance.workspace.return_value = workspace

    spec = DatasetSpec(
        name="x",
        dest=tmp_path / "rf_dest2",
        license="CC BY 4.0",
        role="r",
        description="d",
        extra={"workspace": "ws", "project": "proj", "version": 99, "format": "yolov8"},
    )

    with (
        patch("roboflow.Roboflow", return_value=rf_instance),
        pytest.raises(DatasetDownloadError, match="no version 99"),
    ):
        RoboflowAdapter().fetch(spec)


# --- HuggingFace ---


def test_huggingface_check_credentials_raises_when_missing(monkeypatch):
    _clear_real_credential_env_vars(monkeypatch)
    monkeypatch.setattr("datasets.adapters.huggingface.get_settings", lambda: _settings())
    with pytest.raises(DatasetCredentialError):
        HuggingFaceAdapter().check_credentials()


def test_huggingface_fetch_calls_snapshot_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "datasets.adapters.huggingface.get_settings", lambda: _settings(hf_token="tok")
    )
    spec = DatasetSpec(
        name="x",
        dest=tmp_path / "hf_dest",
        license="research",
        role="r",
        description="d",
        extra={"repo_id": "org/repo", "repo_type": "dataset"},
    )
    with patch("huggingface_hub.snapshot_download") as mock_download:
        result = HuggingFaceAdapter().fetch(spec)

    mock_download.assert_called_once_with(
        repo_id="org/repo", repo_type="dataset", token="tok", local_dir=str(spec.dest)
    )
    assert result.already_present is False


# --- Direct ---


def test_direct_adapter_needs_no_credentials():
    DirectAdapter().check_credentials()  # must not raise


def test_already_fetched_ignores_marker_files(tmp_path):
    # Regression test: a prior real bug had RoboflowAdapter's own
    # .manifest_resolved.json marker (written after a successful fetch)
    # register as "data already present" on the *next* run even when the
    # actual dataset content never downloaded — see datasets/adapters/base.py.
    dest = tmp_path / "dest_with_only_marker"
    dest.mkdir()
    (dest / ".manifest_resolved.json").write_text("{}")

    assert DirectAdapter()._already_fetched(dest) is False

    (dest / "data.yaml").write_text("real content")
    assert DirectAdapter()._already_fetched(dest) is True


def test_direct_adapter_fetch_downloads_and_extracts_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.iter_content.return_value = [buf.getvalue()]
    mock_response.raise_for_status.return_value = None

    spec = DatasetSpec(
        name="x",
        dest=tmp_path / "direct_dest",
        license="CC0",
        role="r",
        description="d",
        extra={"url": "https://example.com/data.zip"},
    )
    with patch("requests.get", return_value=mock_response):
        result = DirectAdapter().fetch(spec)

    assert (spec.dest / "hello.txt").exists()
    assert result.already_present is False


# --- datasets.manifest.build_specs ---


def test_build_specs_parses_manifest_entries():
    manifest = {
        "kaggle": [
            {
                "name": "ds1",
                "dest": "datasets/raw/ds1",
                "license": "CC0",
                "role": "tag_detector",
                "description": "desc",
                "dataset_ref": "org/ds1",
            }
        ]
    }
    specs = build_specs(manifest, "kaggle")
    assert len(specs) == 1
    assert specs[0].name == "ds1"
    assert specs[0].extra == {"dataset_ref": "org/ds1"}


def test_build_specs_raises_config_error_on_missing_field():
    manifest = {"kaggle": [{"name": "ds1", "dest": "x"}]}
    with pytest.raises(ConfigError):
        build_specs(manifest, "kaggle")


def test_build_specs_empty_source_returns_empty_list():
    assert build_specs({}, "kaggle") == []
