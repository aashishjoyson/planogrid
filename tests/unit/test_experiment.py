import datetime as datetime_module

import pytest

from training.experiment import Experiment, create_experiment, latest_experiment


def test_create_experiment_makes_unique_timestamped_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("training.experiment.EXPERIMENTS_ROOT", tmp_path)

    experiment = create_experiment("gap")

    assert experiment.detector == "gap"
    assert experiment.root.exists()
    assert experiment.weights_dir.exists()
    assert experiment.root.parent == tmp_path / "gap"


def test_create_experiment_collision_raises(tmp_path, monkeypatch):
    fixed_now = datetime_module.datetime(2026, 1, 1, tzinfo=datetime_module.UTC)

    class _FrozenDatetime(datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("training.experiment.EXPERIMENTS_ROOT", tmp_path)
    monkeypatch.setattr("training.experiment._git_short_sha", lambda: "abc123")
    monkeypatch.setattr("training.experiment.datetime", _FrozenDatetime)

    create_experiment("gap")
    with pytest.raises(FileExistsError):
        create_experiment("gap")


def test_snapshot_config_and_write_metrics_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("training.experiment.EXPERIMENTS_ROOT", tmp_path)
    experiment = create_experiment("tag")

    experiment.snapshot_config({"epochs": 100, "base_weights": "yolo26s.pt"})
    assert experiment.config_snapshot_path.exists()
    assert "yolo26s.pt" in experiment.config_snapshot_path.read_text()

    experiment.write_metrics({"map50": 0.9})
    assert '"map50": 0.9' in experiment.metrics_path.read_text()


def test_promote_to_checkpoints_copies_best_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("training.experiment.EXPERIMENTS_ROOT", tmp_path)
    experiment = create_experiment("gap")
    experiment.best_weights_path.write_bytes(b"fake-weights")

    # An explicit checkpoints_dir is used as-is (no implicit /<detector>
    # join) — that join only happens for the default REPO_ROOT/checkpoints path.
    checkpoints_dir = tmp_path / "checkpoints_out"
    dest = experiment.promote_to_checkpoints(checkpoints_dir)

    assert dest == checkpoints_dir / "best.pt"
    assert dest.read_bytes() == b"fake-weights"


def test_latest_experiment_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("training.experiment.EXPERIMENTS_ROOT", tmp_path)
    assert latest_experiment("gap") is None


def test_latest_experiment_returns_most_recent_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr("training.experiment.EXPERIMENTS_ROOT", tmp_path)
    detector_root = tmp_path / "gap"
    detector_root.mkdir()
    (detector_root / "20260101_000000_aaa").mkdir()
    (detector_root / "20260228_000000_bbb").mkdir()
    (detector_root / "20260115_000000_ccc").mkdir()

    latest = latest_experiment("gap")

    assert isinstance(latest, Experiment)
    assert latest.run_id == "20260228_000000_bbb"
