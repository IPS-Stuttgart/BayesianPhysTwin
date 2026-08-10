from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.deform360_joint_sparse_prediction_artifacts_v5 import (
    PREDICTION_ARCHIVE_FILENAME,
    PREDICTION_CHECKSUMS_FILENAME,
    PREDICTION_SEAL_FILENAME,
    load_deform360_joint_sparse_prediction_v5,
    publish_deform360_joint_sparse_prediction_v5,
)
from bayesian_phystwin.deform360_joint_sparse_prediction_v5 import (
    B0_PHYSICAL_FALLBACK,
    T1_CONTACT_ONLY,
    V1_VISUAL_GUARDED,
    VT2_VISUOTACTILE_UNGUARDED,
    VT3_VISUOTACTILE_ANCHOR_BIAS,
    Deform360JointSparsePredictionInputV5,
    run_deform360_joint_sparse_prediction_v5,
)


def _problem(*, admitted: bool = True, contact: bool = True):
    count = 4
    propagation = np.zeros((3, 2, 3, 1), dtype=np.float64)
    propagation[1:, :, 0, 0] = 1.0
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = 1.0
    kwargs: dict[str, object] = {}
    if contact:
        anchor_state = np.zeros((2, 3, 1), dtype=np.float64)
        anchor_state[:, 0, 0] = 1.0
        kwargs = {
            "anchor_innovation_m": np.asarray([[0.01, 0.0, 0.0]] * 2),
            "anchor_covariance_m2": np.asarray([np.eye(3) * 4e-6] * 2),
            "anchor_state_jacobian": anchor_state,
            "anchor_correlation_group_ids": ("contact-a", "contact-b"),
            "anchor_prior_reliability": np.ones(2),
            "anchor_prior_nominal_probability": np.full(2, 0.99),
            "anchor_composite_weight": np.ones(2),
        }
    batch = GaugeAwareObservationBatch(
        innovation_m=np.asarray([[0.012, 0.0, 0.0]] * count),
        observation_covariance_m2=np.asarray([np.eye(3) * 1e-6] * count),
        state_jacobian=state,
        gauge_jacobian=np.zeros((count, 3, 1)),
        shared_bias_jacobian=np.zeros((count, 3, 0)),
        view_bias_jacobian=np.zeros((count, 3, 0)),
        query_state_jacobian=propagation[1:3].reshape(-1, 3, 1),
        gauge_prior_covariance=np.eye(1) * 0.005**2,
        correlation_group_ids=("v-a", "v-b", "v-c", "v-d"),
        prior_reliability=np.ones(count),
        association_probability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.99),
        composite_weight=np.ones(count),
        state_prior_covariance_m2=np.eye(1) * 0.020**2,
        physical_response_scale_m=0.020,
        metadata={"causal_frame_stop": 1},
        **kwargs,
    )
    physical = np.zeros((3, 2, 3), dtype=np.float32)
    return Deform360JointSparsePredictionInputV5(
        object_id="001-test",
        episode_id=0,
        stratum="sheet",
        physical_prediction_m=physical,
        persistence_m=physical,
        last_causal_residual_m=np.full((2, 3), 0.002),
        future_state_jacobian_m=propagation,
        observation_batch=batch,
        causal_frame_stop=1,
        evaluation_frame_range_half_open=(1, 3),
        factor_admitted=admitted,
        physical_mode="warp_twin",
        source_artifact_ids={"fixture/input.json": "a" * 64},
    )


def _result(problem: Deform360JointSparsePredictionInputV5):
    return run_deform360_joint_sparse_prediction_v5(
        problem,
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=1.0,
            effective_samples_per_anchor_correlation_group=1.0,
            minimum_conditional_information_fraction=0.0,
            minimum_identifiable_fraction=1e-12,
            minimum_query_sensitivity_fraction=0.0,
        ),
    )


def _publish(path: Path, *, admitted: bool = True, contact: bool = True):
    problem = _problem(admitted=admitted, contact=contact)
    result = _result(problem)
    seal = publish_deform360_joint_sparse_prediction_v5(
        problem,
        result,
        path,
        execution_lock_id="1" * 64,
        implementation_revision="2" * 40,
        prediction_fit_artifact_id="3" * 64,
        prediction_fit_object_ids=("002-b", "003-c"),
    )
    return problem, result, seal


def test_prediction_artifact_round_trip_and_deterministic_archive(tmp_path: Path) -> None:
    _, result, first_seal = _publish(tmp_path / "first")
    _, _, second_seal = _publish(tmp_path / "second")
    loaded_seal, loaded = load_deform360_joint_sparse_prediction_v5(
        tmp_path / "first"
    )

    assert loaded.result_id == result.result_id
    assert loaded_seal["prediction_seal_id"] == first_seal["prediction_seal_id"]
    assert first_seal["archive"]["file_sha256"] == second_seal["archive"][
        "file_sha256"
    ]
    assert first_seal["prediction_fit_object_ids"] == ["002-b", "003-c"]
    assert loaded.trajectories_m[B0_PHYSICAL_FALLBACK].dtype == np.float32


def test_prediction_publication_refuses_overwrite(tmp_path: Path) -> None:
    _publish(tmp_path / "prediction")
    try:
        _publish(tmp_path / "prediction")
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("prediction artifact was silently overwritten")


def test_prediction_archive_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "prediction"
    _publish(root)
    with (root / PREDICTION_ARCHIVE_FILENAME).open("ab") as stream:
        stream.write(b"changed")

    try:
        load_deform360_joint_sparse_prediction_v5(root)
    except ValueError as error:
        assert "archive bytes changed" in str(error)
    else:
        raise AssertionError("changed prediction archive was accepted")


def test_prediction_seal_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "prediction"
    _publish(root)
    seal_path = root / PREDICTION_SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["risk_score"] += 1.0
    seal_path.write_text(json.dumps(seal), encoding="utf-8")

    try:
        load_deform360_joint_sparse_prediction_v5(root)
    except ValueError as error:
        assert "seal ID changed" in str(error)
    else:
        raise AssertionError("changed prediction seal was accepted")


def test_prediction_checksum_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "prediction"
    _publish(root)
    checksums = root / PREDICTION_CHECKSUMS_FILENAME
    checksums.write_text("0" * 64 + "  prediction-arrays.npz\n", encoding="ascii")

    try:
        load_deform360_joint_sparse_prediction_v5(root)
    except ValueError as error:
        assert "SHA256SUMS changed" in str(error)
    else:
        raise AssertionError("changed checksum manifest was accepted")


def test_prediction_loader_rejects_coerced_boolean(tmp_path: Path) -> None:
    root = tmp_path / "prediction"
    _publish(root)
    seal_path = root / PREDICTION_SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["factor_admitted"] = 1
    body = {key: value for key, value in seal.items() if key != "prediction_seal_id"}
    from bayesian_phystwin._portable_contracts import content_id

    seal["prediction_seal_id"] = content_id(body)
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    import hashlib

    for name in sorted((PREDICTION_ARCHIVE_FILENAME, PREDICTION_SEAL_FILENAME)):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {name}\n")
    (root / PREDICTION_CHECKSUMS_FILENAME).write_text(
        "".join(checksum_lines), encoding="ascii"
    )

    try:
        load_deform360_joint_sparse_prediction_v5(root)
    except ValueError as error:
        assert "factor_admitted must be Boolean" in str(error)
    else:
        raise AssertionError("coerced Boolean was accepted")


def test_rejected_methods_share_exact_b0_artifact_identity(tmp_path: Path) -> None:
    _, _, seal = _publish(
        tmp_path / "prediction",
        admitted=False,
        contact=False,
    )
    baseline = seal["method_artifact_ids"][B0_PHYSICAL_FALLBACK]

    for method_id in (
        V1_VISUAL_GUARDED,
        T1_CONTACT_ONLY,
        VT2_VISUOTACTILE_UNGUARDED,
        VT3_VISUOTACTILE_ANCHOR_BIAS,
    ):
        assert seal["method_artifact_ids"][method_id] == baseline
