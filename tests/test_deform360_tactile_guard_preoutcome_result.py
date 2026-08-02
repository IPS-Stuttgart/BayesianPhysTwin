from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_deform360_tactile_guard_outcome_sealed import (
    REPO,
    _stage_backbone,
    _write_measurement,
    _write_protocol_bundle,
    _write_tactile_features,
)

from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (
    build_guarded_prediction,
    build_prediction_barrier,
    build_technical_fallback,
)
from bayesian_phystwin.deform360_tactile_guard_preoutcome_result import (
    PREOUTCOME_RESULT_ARTIFACT_KIND,
    build_preoutcome_impossibility_result,
)


def _sealed_predictions(tmp_path: Path, *, nontrivial_count: int) -> tuple[Path, Path, Path]:
    bundle = _write_protocol_bundle(tmp_path)
    protocol = json.loads(bundle["protocol"].read_text())
    predictions = tmp_path / "predictions"
    for rank, record in enumerate(protocol["cohort"]["cases"], start=1):
        case_root = tmp_path / f"case-{rank}"
        backbone = _stage_backbone(case_root, bundle, rank=rank)
        output = predictions / str(record["case"])
        if rank <= nontrivial_count:
            measurement = _write_measurement(case_root, backbone)
            tactile = case_root / "tactile.json"
            _write_tactile_features(tactile, case=str(record["case"]))
            build_guarded_prediction(
                output,
                repository_root=REPO,
                protocol_path=bundle["protocol"],
                backbone_dir=backbone,
                measurement_dir=measurement,
                tactile_feature_path=tactile,
                source_result_path=bundle["source_result"],
            )
        else:
            build_technical_fallback(
                output,
                protocol_path=bundle["protocol"],
                backbone_dir=backbone,
                failure_stage="synthetic",
                failure_type="RuntimeError",
                failure_message="registered fallback",
            )
    barrier_path = tmp_path / "barrier.json"
    build_prediction_barrier(
        barrier_path,
        protocol_path=bundle["protocol"],
        prediction_root=predictions,
    )
    return bundle["protocol"], predictions, barrier_path


def test_one_nontrivial_case_closes_two_win_gate_without_outcomes(
    tmp_path: Path,
) -> None:
    protocol, predictions, barrier = _sealed_predictions(
        tmp_path, nontrivial_count=1
    )
    result = build_preoutcome_impossibility_result(
        tmp_path / "result.json",
        protocol_path=protocol,
        barrier_path=barrier,
        prediction_root=predictions,
        runtime_revision="a" * 40,
    )

    assert result["artifact_kind"] == PREOUTCOME_RESULT_ARTIFACT_KIND
    assert result["guard_audit"]["accepted_update_count"] == 3
    assert result["guard_audit"]["admitted_case_count"] == 1
    assert result["guard_audit"]["nontrivial_prediction_case_count"] == 1
    assert result["gate_impossibility_proof"]["required_joint_case_wins"] == 2
    assert result["gate_impossibility_proof"][
        "maximum_possible_joint_case_wins"
    ] == 1
    assert result["advancement_decision"]["future_outcomes_opened"] is False
    assert result["information_boundary"]["outcome_manifest_read"] is False


def test_finalizer_refuses_when_joint_win_gate_remains_reachable(
    tmp_path: Path,
) -> None:
    protocol, predictions, barrier = _sealed_predictions(
        tmp_path, nontrivial_count=2
    )
    with pytest.raises(ValueError, match="joint-win gate remains reachable"):
        build_preoutcome_impossibility_result(
            tmp_path / "result.json",
            protocol_path=protocol,
            barrier_path=barrier,
            prediction_root=predictions,
            runtime_revision="b" * 40,
        )


def test_mutated_prediction_after_barrier_is_rejected(tmp_path: Path) -> None:
    protocol, predictions, barrier = _sealed_predictions(
        tmp_path, nontrivial_count=1
    )
    case = sorted(predictions.iterdir())[0]
    archive = case / "guarded_prediction.npz"
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays["guarded_prediction_m"][0, 0, 0] += 1.0
    np.savez_compressed(archive, **arrays)

    with pytest.raises(ValueError, match="guarded prediction files changed"):
        build_preoutcome_impossibility_result(
            tmp_path / "result.json",
            protocol_path=protocol,
            barrier_path=barrier,
            prediction_root=predictions,
            runtime_revision="c" * 40,
        )
