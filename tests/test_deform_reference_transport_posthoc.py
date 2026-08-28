"""Synthetic regression tests for the explicitly post-result audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_reference_transport import (
    config_for_source,
    score_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def auditor():
    return _module(ROOT / "scripts/audit_deform_reference_transport_posthoc.py")


def _arrays():
    names = json.loads(
        (ROOT / "configs/sota/deform_reference_transport_source_v1.json").read_text()
    )["names"]
    native = np.zeros((14, 170, 12, 3), dtype=np.float32)
    native[:, :, (0, 1, 10, 11)] = 2**-4
    base = native.astype(float)
    base[:, :, 2:10] = 2**-8
    dx = np.zeros((14, 12, 3), dtype=np.float64)
    dx[:, 2:10] = 2**-9
    arrays = {
        "names": np.array(names),
        "nominal": native,
        "nominal_velocity": np.zeros_like(native),
        "incumbent": base[:, 50:].copy(),
        "offsets": base[:, 49:] - native[:, 49:].astype(float),
        "offset_velocities": np.zeros((14, 121, 12, 3)),
        "pose_increment": dx,
        "velocity_increment": np.zeros_like(dx),
        "future_actions": native[:, 50:, (0, 1, 10, 11)].copy() + np.float32(2**-5),
    }
    for arm in ("reference_initialized", "reference_centered", "zero_reference"):
        offset = np.zeros((14, 120, 12, 3), dtype=np.float32)
        if arm != "zero_reference":
            offset[:] = arrays["offsets"][:, 1:].astype(np.float32)
        center = native[:, 50:] + offset
        updated = center + dx[:, None].astype(np.float32)
        zero = np.zeros_like(center)
        traces = {
            "center_before": center,
            "center_after": center,
            "updated_before": updated,
            "updated_after": updated,
            "center_velocity_before": zero,
            "center_velocity_after": zero,
            "updated_velocity_before": zero,
            "updated_velocity_after": zero,
            "centering_dx": zero,
            "centering_dv": zero,
        }
        arrays.update({arm + "__" + k: v.copy() for k, v in traces.items()})
        arrays[arm] = arrays["incumbent"] + (
            updated.astype(float) - center.astype(float)
        )
    arrays["paired"] = arrays["zero_reference"].copy()
    return arrays, native, base


def test_native_preservation_does_not_claim_exact_commands(auditor):
    arrays, _, _ = _arrays()
    report = auditor.audit_traces(arrays)
    for arm in report.values():
        assert arm["native_clamp_bytes_preserved"]
        assert not arm["command_clamp_bytes_exact"]
        assert arm["maximum_command_difference_m"] == 2**-5


def test_changed_native_clamp_is_not_excused_by_command_mismatch(auditor):
    arrays, _, _ = _arrays()
    arrays["reference_centered__center_after"][0, -1, 0, 0] += 0.001
    with pytest.raises(ValueError):
        auditor.audit_traces(arrays)


def test_changed_centering_or_readout_is_rejected(auditor):
    arrays, _, _ = _arrays()
    arrays["reference_centered__centering_dx"][0, 4, 5, 1] += 0.001
    with pytest.raises(ValueError):
        auditor.audit_traces(arrays)
    arrays, _, _ = _arrays()
    arrays["reference_centered"][0, 4, 5, 1] += 0.001
    with pytest.raises(ValueError):
        auditor.audit_traces(arrays)


def test_metric_reproduction_does_not_rescue_failed_source_gate(auditor):
    arrays, _, _ = _arrays()
    truth = np.zeros((14, 120, 12, 3))
    result = score_predictions(
        arrays["names"].tolist(),
        {k: arrays[k] for k in auditor.ARMS},
        truth,
        config_for_source(),
    )
    report = auditor.audit_metrics(arrays, truth, result)
    assert report["metrics_verified"] == 624
    assert not report["source_value_gate_passed"]
    result["decision"]["passed"] = True
    with pytest.raises(ValueError, match="decision"):
        auditor.audit_metrics(arrays, truth, result)


def test_complete_posthoc_audit_reproduces_original_failure_without_edits(
    tmp_path, auditor
):
    runner = _module(ROOT / "scripts/remote/run_deform_reference_transport.py")
    arrays, native, base = _arrays()
    root = tmp_path / "run"
    root.mkdir()
    truth_path, parent_path = tmp_path / "truth.npz", tmp_path / "parent.npz"
    truth = np.zeros_like(base)
    np.savez_compressed(
        truth_path,
        targets=truth,
        baseline_predictions=native,
        candidate_predictions=base,
    )
    np.savez_compressed(
        parent_path,
        incumbent=arrays["incumbent"],
        incumbent_propagated_pose_velocity=arrays["paired"],
    )
    verifier_name = "scripts/verify_deform_reference_transport.py"
    lock = runner._write(
        root / "lock.json",
        {
            "plan": {
                "names": arrays["names"].tolist(),
                "archive": {"sha256": auditor.digest(truth_path)},
                "paired_archive": {"sha256": auditor.digest(parent_path)},
            },
            "source_files": {verifier_name: auditor.digest(ROOT / verifier_name)},
        },
    )
    controls = runner._write(
        root / "controls.json",
        {
            "checks": {"synthetic_control": True},
            "archived_gpu_replay_max_error_m": 0.0,
            "archived_gpu_replay_coordinate_rmse_m": 0.0,
        },
    )
    binding = runner._save_arrays(root / "predictions.npz", arrays)
    seal = runner._write(
        root / "prediction_seal.json",
        {
            "lock_id": lock["artifact_id"],
            "controls_id": controls["artifact_id"],
            "controls_sha256": auditor.digest(root / "controls.json"),
            "predictions": binding,
            "complete": True,
            "ordinary_successes": 14,
            "retained_technical_failures": 0,
            "unsealable": 0,
            "names": arrays["names"].tolist(),
        },
    )
    scores = score_predictions(
        arrays["names"].tolist(),
        {k: arrays[k] for k in auditor.ARMS},
        truth[:, 50:],
        config_for_source(),
    )
    runner._write(
        root / "result.json",
        {
            **scores,
            "prediction_seal_id": seal["artifact_id"],
            "prediction_seal_sha256": auditor.digest(root / "prediction_seal.json"),
        },
    )
    original = {p.name: auditor.digest(p) for p in root.iterdir()}
    report = auditor.audit(root, ROOT, truth_path, parent_path)
    assert report["native_preservation_and_metric_audit_passed"]
    assert report["original_failed_decision_unchanged"]
    assert not report["original_second_arithmetic_passed"]
    assert not report["literal_command_exact_check_passed"]
    assert not report["promotion_authorized"]
    assert not report["new_native_execution"]
    assert not report["independent_human_review"]
    assert report["original_failure_reproduced"]["line"] == 189
    assert {p.name: auditor.digest(p) for p in root.iterdir()} == original


def test_cli_refuses_receipt_inside_frozen_run_before_reading(
    tmp_path, monkeypatch, auditor
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--run-root",
            str(tmp_path),
            "--source-root",
            str(ROOT),
            "--truth-archive",
            "absent",
            "--parent-paired-archive",
            "absent",
            "--output",
            str(tmp_path / "posthoc.json"),
        ],
    )
    with pytest.raises(ValueError, match="must not modify"):
        auditor.main()
