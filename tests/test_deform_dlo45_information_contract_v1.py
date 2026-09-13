from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deform_dlo45_decision_identifiability_v1._common import (
    Model,
    Protocol,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import (
    decide,
    fit_model,
)
from experiments.deform_dlo45_information_contract_v1.export import (
    METHODS,
    _case_payload,
    _deterministic_npz_bytes,
    _validate_request,
    reconstruct_decision_evidence,
)


def _protocol() -> Protocol:
    return Protocol(
        prefix_frames=5,
        horizon_frames=25,
        stride_frames=25,
        action_scales=(0.0, 0.5, 1.0),
        neighbor_grid=(8,),
        cluster_grid=(4,),
        temperature_grid=(1.0,),
        regret_tolerance_grid=(0.0, 0.05),
        kmeans_iterations=20,
        source_fit_count=39,
        source_calibration_count=9,
        source_test_count=8,
        partition_domain="adapter-test",
        source_gate_mean_ratio=1.2,
        source_gate_worst_trajectory_ratio=1.5,
        source_gate_minimum_nonfallback_fraction=0.0,
        bootstrap_replicates=100,
        bootstrap_seed=42,
    )


def _model() -> tuple[Model, np.ndarray, Protocol]:
    frozen = _protocol()
    rng = np.random.default_rng(20260902)
    features = rng.normal(size=(48, 81))
    residuals = rng.normal(scale=0.02, size=(48, 25 * 8 * 3))
    model = fit_model(
        features,
        residuals,
        cluster_count=4,
        neighbors=8,
        temperature_scale=1.0,
        regret_tolerance=0.05,
        protocol=frozen,
    )
    return model, features[0], frozen


def test_reconstruction_matches_the_frozen_policy() -> None:
    model, feature, frozen = _model()
    evidence = reconstruct_decision_evidence(feature, model, frozen)
    reference = decide(feature, model, frozen)

    assert evidence.selected_actions == {
        "fallback": 0,
        "certificate": reference.certificate_action,
        "jeffrey_point": reference.jeffrey_action,
        "kernel_point": reference.kernel_action,
        "map_point": reference.map_action,
    }
    np.testing.assert_allclose(evidence.correction, reference.correction)
    np.testing.assert_allclose(
        evidence.worst_case_regret,
        reference.worst_case_regret,
    )
    assert evidence.loss_by_hypothesis.shape == (8, 3)
    assert evidence.prior.shape == (8,)
    assert evidence.quotient_mass.sum() == pytest.approx(1.0)
    assert set(evidence.classes.tolist()) == set(range(len(evidence.quotient_mass)))


def test_payload_is_deterministic_and_pickle_free() -> None:
    arrays = {
        "truth_xyz_m": np.arange(9, dtype=np.float64).reshape(3, 3),
        "selected_action": np.asarray(1, dtype=np.int64),
        "decision_admitted": np.asarray(True),
    }
    first = _deterministic_npz_bytes(arrays)
    second = _deterministic_npz_bytes(arrays)
    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as archive:
        assert sorted(archive.files) == sorted(arrays)
        np.testing.assert_array_equal(archive["truth_xyz_m"], arrays["truth_xyz_m"])


def test_case_payload_applies_the_selected_normalized_action() -> None:
    model, feature, frozen = _model()
    evidence = reconstruct_decision_evidence(feature, model, frozen)
    truth = np.zeros((25, 8, 3), dtype=np.float64)
    baseline = np.ones_like(truth)
    actions = model.action_scales[:, None] * evidence.correction[None, :]
    realized = np.array([0.5, 0.2, 0.1])
    arrays = _case_payload(
        truth=truth,
        baseline=baseline,
        normalized_actions=actions,
        selected_action=2,
        length_scale=0.5,
        evidence=evidence,
        realized_action_loss=realized,
        certificate=True,
        regret_tolerance=model.regret_tolerance,
    )

    expected = baseline.reshape(-1, 3) + actions[2].reshape(-1, 3) * 0.5
    np.testing.assert_allclose(arrays["prediction_mean_xyz_m"], expected)
    np.testing.assert_array_equal(arrays["realized_action_loss"], realized)
    assert bool(arrays["decision_admitted"]) is evidence.certificate_admitted


def test_request_is_exact_and_fail_closed(tmp_path: Path) -> None:
    from experiments.deform_dlo45_information_contract_v1 import export

    request = {
        "contract": export.REQUEST_CONTRACT,
        "schema_version": 1,
        "status": "authorized-retrospective-replay",
        "source_run_id": export.SOURCE_RUN_ID,
        "source_artifact_id": export.SOURCE_ARTIFACT_ID,
        "source_head_sha": export.SOURCE_HEAD_SHA,
        "source_artifact_digest": export.SOURCE_ARTIFACT_DIGEST,
        "source_model_sha256": export.SOURCE_MODEL_SHA256,
        "source_result_sha256": export.SOURCE_RESULT_SHA256,
        "source_seal_sha256": export.SOURCE_SEAL_SHA256,
        "reference_result_sha256": export.REFERENCE_RESULT_SHA256,
        "reference_result_git_blob": export.REFERENCE_RESULT_GIT_BLOB,
        "deform_repository": export.DEFORM_REPOSITORY,
        "deform_commit": export.DEFORM_COMMIT,
        "prob4d_repository": export.PROB4D_REPOSITORY,
        "prob4d_commit": export.PROB4D_COMMIT,
        "target_tuning": False,
        "target_retries": False,
        "new_target_outcomes_opened": False,
        "payload_redistribution": False,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert _validate_request(path) == request

    request["target_retries"] = True
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="target_retries"):
        _validate_request(path)


def test_method_roster_is_frozen() -> None:
    assert METHODS == (
        "fallback",
        "certificate",
        "jeffrey_point",
        "kernel_point",
        "map_point",
        "oracle",
    )
