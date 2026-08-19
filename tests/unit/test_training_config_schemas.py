"""Regression coverage for the config-schema strictness fix: a profile or
base config setting a field no schema declares must fail loudly at load
time, not vanish silently. This is exactly how `cache: disk` in
configs/profiles/local_8gb.yaml went unnoticed until a real training run
exhausted system RAM — see training/trainer.py's TrainingDefaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import load_config
from training.datasets.prepare import DatasetsConfig
from training.trainer import TrainingConfig


def test_training_config_rejects_unrecognized_field():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrainingConfig.model_validate({"defaults": {"epochs": 10, "totally_unknown_field": 1}})


def test_datasets_config_rejects_unrecognized_field():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DatasetsConfig.model_validate({"split": {"train": 0.8, "unknown_key": True}})


def test_training_config_cache_field_is_present_and_overridable():
    # The actual bug: `cache` had no schema slot at all and profile overrides
    # for it were silently dropped. This just proves it round-trips.
    cfg = TrainingConfig.model_validate({"defaults": {"cache": "disk"}})
    assert cfg.defaults.cache == "disk"


def test_local_8gb_profile_loads_and_overrides_workers_and_cache():
    cfg = load_config("training", TrainingConfig, profile="local_8gb")
    assert cfg.defaults.workers == 2
    assert cfg.defaults.cache == "disk"
    assert cfg.defaults.batch == 8


def test_cloud_t4_profile_keeps_higher_worker_count():
    cfg = load_config("training", TrainingConfig, profile="cloud_t4")
    assert cfg.defaults.workers == 8  # base default, not overridden for this profile
    assert cfg.defaults.batch == 32
