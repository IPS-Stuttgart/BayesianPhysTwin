import importlib.util
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.dlolab_active_wrapping_source import protocol

SPEC = importlib.util.spec_from_file_location(
    "active_wrapping_runner",
    Path(__file__).resolve().parents[1]
    / "scripts/remote/run_dlolab_active_probe_wrapping_source.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_registered_output_is_source_only_and_write_once():
    assert (
        str(runner.OUTPUT)
        == "/home/fpfaff/source-only/dlolab-active-probe-wrapping-source-v1"
    )
    assert protocol()["retry_authorized"] is False
    assert protocol()["protected_data_read"] is False


def test_validate_lock_rejects_alternate_root_before_read(tmp_path):
    with pytest.raises(ValueError, match="registered"):
        runner.validate_lock(tmp_path)


def test_worker_rejects_active_probe_not_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_lock", lambda output: {"artifact_id": "lock"})
    monkeypatch.setattr(runner, "_selected_probe", lambda output: 2)
    with pytest.raises(ValueError, match="selected"):
        runner.prerequisites(tmp_path, {"artifact_id": "lock"}, "active", 0, 1)


def test_baseline_rejects_nonnull_probe(tmp_path):
    with pytest.raises(ValueError, match="null"):
        runner.prerequisites(tmp_path, {"artifact_id": "lock"}, "baseline", 0, 1)
