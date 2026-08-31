"""Contracts for the frozen DLO4/DLO5 transfer experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "scripts" / "remote"
if str(REMOTE) not in sys.path:
    sys.path.insert(0, str(REMOTE))

from experiments.deform_dlo45_frozen_v1 import run  # noqa: E402

PROTOCOL = ROOT / "experiments" / "deform_dlo45_frozen_v1" / "protocol.json"
WORKFLOW = ROOT / ".github" / "workflows" / "deform-dlo45-frozen-transfer.yml"


def test_protocol_is_exact_and_target_jointly_sealed() -> None:
    protocol = run._load_protocol(PROTOCOL)
    assert tuple(protocol["data"]["dlos"]) == run.DLOS
    assert protocol["target_evaluation"]["joint_prediction_seal_before_scoring"]
    assert protocol["custody"]["source_gate_for_each_dlo_before_any_eval_access"]
    assert not protocol["custody"]["retry_authorized"]


def test_pre_target_recovery_is_explicit_and_progress_is_observable() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert ".github/requests/deform-dlo45-evaluate-v2.json" in workflow
    assert ".github/requests/deform-dlo45-evaluate-v3.json" in workflow

    assert '"prior_failed_run_id": 33329341107' in workflow
    assert '"retry_class": "pre-target-infrastructure-correction"' in workflow
    assert '"prior_failed_run_id": 33335970581' in workflow
    assert '"prior_failure_stage": "source-qualification-job-timeout"' in workflow
    assert '"prior_target_access": False' in workflow
    assert (
        '"retry_class": "pre-target-timeout-and-observability-correction"' in workflow
    )

    assert workflow.count("timeout-minutes: 7200") == 3
    assert "\n  source:\n" in workflow
    assert "\n  target:\n" in workflow
    assert "needs.source.result == 'success'" in workflow
    assert "DLO4-source" in workflow
    assert "DLO5-source" in workflow
    assert "DLO4-target" in workflow
    assert "DLO5-target" in workflow
    assert "source-heartbeat.jsonl" in workflow
    assert "target-heartbeat.jsonl" in workflow
    assert 'print("[progress] " + json.dumps(payload' in workflow
    assert "dlo4-source/physical/progress.json" in workflow
    assert "dlo5-source/physical/progress.json" in workflow
    assert "dlo4-target/alltrain/progress.json" in workflow
    assert "dlo5-target/alltrain/progress.json" in workflow
    assert "deform-dlo45-source-${{ github.run_id }}" in workflow

    assert 'git config --file "$safe_git_config" --add safe.directory' in workflow
    assert 'echo "GIT_CONFIG_GLOBAL=$safe_git_config" >> "$GITHUB_ENV"' in workflow
    assert 'rm -f "$DLO45_GIT_CONFIG"' in workflow


def test_protocol_rejects_post_selection_change(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["local_residual"]["shrinkage"] = 0.5
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol differs"):
        run._load_protocol(changed)


def test_partition_is_deterministic_exhaustive_and_dlo_specific() -> None:
    protocol = run._load_protocol(PROTOCOL)
    names = [f"{index}.pkl" for index in range(56)]
    first = run._partition_names(names, dlo="DLO4", protocol=protocol)
    second = run._partition_names(list(reversed(names)), dlo="DLO4", protocol=protocol)
    other = run._partition_names(names, dlo="DLO5", protocol=protocol)
    assert first == second
    assert tuple(map(len, first.values())) == (39, 9, 8)
    assert set().union(*map(set, first.values())) == set(names)
    assert all(
        set(left).isdisjoint(right)
        for index, left in enumerate(first.values())
        for right in list(first.values())[index + 1 :]
    )
    assert first != other


def test_source_gate_uses_paired_trajectory_units() -> None:
    protocol = run._load_protocol(PROTOCOL)
    target = np.zeros((8, 6, 12, 3), dtype=np.float64)
    baseline = np.ones_like(target)
    candidate = np.full_like(target, 0.8)
    result = run._source_gate(
        candidate,
        baseline,
        target,
        [f"{index}.pkl" for index in range(8)],
        protocol,
    )
    assert result["passed"]
    assert result["wins"] == 8
    assert result["relative_improvement"] == pytest.approx(0.2)


def test_target_gate_fails_one_large_regression() -> None:
    protocol = run._load_protocol(PROTOCOL)
    target = np.zeros((14, 9, 12, 3), dtype=np.float64)
    baseline = np.ones_like(target)
    candidate = np.full_like(target, 0.8)
    candidate[0] = 2.0
    summary = run._point_summary(
        candidate,
        baseline,
        target,
        [f"{index}.pkl" for index in range(14)],
    )
    result = run._target_gate(summary, protocol)
    assert result["wins"] == 13
    assert result["worst_candidate_to_baseline_ratio"] == pytest.approx(2.0)
    assert not result["passed"]


def test_point_summary_matches_official_metric_and_reports_free_nodes() -> None:
    target = np.zeros((2, 6, 12, 3), dtype=np.float64)
    baseline = np.ones_like(target)
    candidate = np.full_like(target, 0.5)
    candidate[:, :, :2] = 1000.0
    candidate[:, :, -2:] = 1000.0
    summary = run._point_summary(
        candidate,
        baseline,
        target,
        ["a.pkl", "b.pkl"],
    )
    expected = (4.0 * 1000.0 + 8.0 * 0.5) / 12.0
    assert summary["candidate_mean_l1_m"] == pytest.approx(expected)
    assert summary["baseline_mean_l1_m"] == pytest.approx(1.0)
    free = summary["free_node_diagnostic"]
    assert free["candidate_mean_l1_m"] == pytest.approx(0.5)
    assert free["baseline_mean_l1_m"] == pytest.approx(1.0)
    assert free["relative_improvement"] == pytest.approx(0.5)
