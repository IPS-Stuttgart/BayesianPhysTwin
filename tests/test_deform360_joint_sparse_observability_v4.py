from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.deform360_joint_sparse_observability_v4 import (
    DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
    DEFORM360_JOINT_SPARSE_POLICY_SCHEMA,
    Deform360JointSparseDevelopmentReportV4,
    Deform360JointSparseFactorBatchV4,
    Deform360JointSparseObservabilityPolicyV4,
    build_deform360_joint_sparse_factor_batch_from_tree_sparse_v4,
    default_deform360_joint_sparse_information_boundary_v4,
    evaluate_deform360_joint_sparse_observability_v4,
    technical_failure_deform360_joint_sparse_result_v4,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def policy(**changes: object) -> Deform360JointSparseObservabilityPolicyV4:
    values: dict[str, object] = {
        "minimum_distinct_cameras": 2,
        "minimum_distinct_windows": 2,
        "minimum_distinct_spatial_clusters": 4,
        "minimum_supported_objects": 1,
        "minimum_supported_objects_per_stratum": 1,
        "require_full_query_rank": True,
        "minimum_query_precision_eigenvalue": 1e-9,
        "maximum_query_condition_number": 1e8,
        "maximum_single_camera_information_fraction": 0.8,
        "minimum_leave_one_camera_rank_fraction": 1 / 3,
        "minimum_leave_one_window_rank_fraction": 1 / 3,
        "effective_samples_per_correlation_group": 64.0,
        "shared_bias_prior_std_m": 0.02,
        "view_bias_prior_std_m": 0.01,
        "relative_rank_tolerance": 1e-9,
        "absolute_rank_tolerance": 1e-12,
    }
    values.update(changes)
    return Deform360JointSparseObservabilityPolicyV4(**values)


def batch(
    state: np.ndarray | None = None,
    *,
    cameras: tuple[str, ...] | None = None,
    windows: tuple[str, ...] | None = None,
    clusters: tuple[str, ...] | None = None,
    groups: tuple[str, ...] | None = None,
    query: np.ndarray | None = None,
    shared: np.ndarray | None = None,
    view: np.ndarray | None = None,
    object_id: str = "object-a",
    stratum: str = "sheet",
    excluded_factor_count: int = 0,
) -> Deform360JointSparseFactorBatchV4:
    if state is None:
        state = np.zeros((6, 3, 3))
        state[0, 0, 0] = 1
        state[1, 1, 0] = 1
        state[2, 2, 1] = 1
        state[3, 0, 1] = 1
        state[4, 1, 2] = 1
        state[5, 2, 2] = 1
    count = len(state)
    cameras = cameras or tuple(
        "camera-a" if index < count // 2 else "camera-b"
        for index in range(count)
    )
    windows = windows or tuple(
        "window-0" if index % 2 == 0 else "window-1"
        for index in range(count)
    )
    clusters = clusters or tuple(f"cluster-{index}" for index in range(count))
    groups = groups or tuple(f"group-{index}" for index in range(count))
    return Deform360JointSparseFactorBatchV4(
        selection_artifact_sha256="1" * 64,
        visual_provider_lock_id="2" * 64,
        observation_artifact_id="3" * 64,
        linearization_artifact_id="4" * 64,
        implementation_revision="5" * 40,
        object_id=object_id,
        episode_id=0,
        stratum=stratum,
        factor_ids=tuple(digest(f"{object_id}-{index}") for index in range(count)),
        camera_ids=cameras,
        window_ids=windows,
        spatial_cluster_ids=clusters,
        correlation_group_ids=groups,
        gauge_ids=("gauge-root",),
        gauge_prior_id="6" * 64,
        observation_covariance_m2=np.repeat(np.eye(3)[None], count, axis=0),
        state_jacobian=state,
        local_gauge_jacobian=np.zeros((count, 3, 1)),
        gauge_indices=np.zeros(count, dtype=np.int64),
        parent_indices=np.array([-1], dtype=np.int64),
        transition_matrices=np.zeros((1, 1, 1)),
        innovation_scale_tril=np.ones((1, 1, 1)),
        query_jacobian=np.eye(3) if query is None else query,
        prior_reliability=np.ones(count),
        association_probability=np.ones(count),
        composite_weight=np.ones(count),
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        excluded_factor_count=excluded_factor_count,
        source_artifacts={"source.json": "7" * 64},
        metadata={"cohort": "development"},
    )


def test_policy_roundtrip_and_default_boundary() -> None:
    value = policy()
    record = value.to_record()
    assert record["schema"] == DEFORM360_JOINT_SPARSE_POLICY_SCHEMA
    assert record["claim_boundary"] == DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY
    assert Deform360JointSparseObservabilityPolicyV4.from_record(record) == value
    assert default_deform360_joint_sparse_information_boundary_v4()[
        "confirmation_payloads_opened"
    ] is False
    with pytest.raises(ValueError, match="policy_id"):
        replace(value, minimum_distinct_cameras=3)
    bad = dict(record)
    bad["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        Deform360JointSparseObservabilityPolicyV4.from_record(bad)
    for key, expected in (
        ("schema", "schema"),
        ("schema_version", "version"),
        ("semantics", "semantics"),
        ("claim_boundary", "claim boundary"),
    ):
        bad = dict(record)
        bad[key] = "changed" if key != "schema_version" else 9
        with pytest.raises(ValueError, match=expected):
            Deform360JointSparseObservabilityPolicyV4.from_record(bad)


def test_complementary_factors_pass_jointly_and_are_content_addressed() -> None:
    item = batch(excluded_factor_count=2)
    first = evaluate_deform360_joint_sparse_observability_v4(
        item, policy(), implementation_revision="8" * 40
    )
    second = evaluate_deform360_joint_sparse_observability_v4(
        item, policy(), implementation_revision="8" * 40
    )
    assert first == second
    assert item.input_id == content_id_for_test(item.identity_record())
    assert first.query_rank == 3
    assert first.query_precision_eigenvalues == pytest.approx((2, 2, 2))
    assert first.single_camera_information_fraction == pytest.approx(
        {"camera-a": 0.5, "camera-b": 0.5}
    )
    assert first.leave_one_camera_rank_fraction == pytest.approx(
        {"camera-a": 2 / 3, "camera-b": 2 / 3}
    )
    assert first.leave_one_window_rank_fraction == pytest.approx(
        {"window-0": 1.0, "window-1": 1.0}
    )
    assert first.excluded_factor_count == 2
    assert first.gate_passed is True
    assert first.to_record()["result_id"] == first.result_id


def content_id_for_test(value: object) -> str:
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_rank_deficiency_camera_dominance_and_policy_checks_fail_closed() -> None:
    state = np.zeros((4, 3, 3))
    state[0, 0, 0] = 1
    state[1, 1, 0] = 1
    state[2, 2, 1] = 1
    state[3, 0, 1] = 1
    deficient = evaluate_deform360_joint_sparse_observability_v4(
        batch(
            state,
            cameras=("a", "a", "b", "b"),
            windows=("w0", "w1", "w0", "w1"),
            clusters=("c0", "c1", "c2", "c3"),
        ),
        policy(maximum_single_camera_information_fraction=1.0),
        implementation_revision="8" * 40,
    )
    assert deficient.query_rank == 2
    assert deficient.query_condition_number is None
    assert deficient.gate_checks["query_rank"] is False
    assert deficient.gate_checks["maximum_query_condition_number"] is False

    dominant_state = np.repeat(np.eye(3)[None], 6, axis=0)
    dominant_state[5] *= 1e-6
    dominant = evaluate_deform360_joint_sparse_observability_v4(
        batch(dominant_state, cameras=("dominant",) * 5 + ("weak",)),
        policy(maximum_single_camera_information_fraction=0.7),
        implementation_revision="8" * 40,
    )
    assert dominant.gate_checks["maximum_single_camera_information_fraction"] is False
    assert dominant.gate_passed is False

    strict = evaluate_deform360_joint_sparse_observability_v4(
        batch(),
        policy(
            minimum_distinct_cameras=3,
            minimum_distinct_windows=3,
            minimum_distinct_spatial_clusters=7,
            minimum_query_precision_eigenvalue=3,
            maximum_query_condition_number=0.5,
            minimum_leave_one_camera_rank_fraction=1,
            minimum_leave_one_window_rank_fraction=1,
        ),
        implementation_revision="8" * 40,
    )
    assert set(strict.gate_checks.values()) == {False, True}
    assert strict.gate_passed is False


def test_group_cap_bias_nuisance_and_zero_weight_paths() -> None:
    item = batch(
        groups=("same",) * 6,
        shared=np.tile(np.eye(3)[None], (6, 1, 1)),
        view=np.zeros((6, 3, 6)),
    )
    reliability = np.ones(6)
    reliability[0] = 0
    associated = np.ones(6)
    associated[1] = 0
    item = replace(
        item,
        input_id=None,
        prior_reliability=reliability,
        association_probability=associated,
        composite_weight=np.full(6, 0.5),
    )
    result = evaluate_deform360_joint_sparse_observability_v4(
        item,
        policy(
            effective_samples_per_correlation_group=2,
            maximum_single_camera_information_fraction=1,
            minimum_leave_one_camera_rank_fraction=0,
            minimum_leave_one_window_rank_fraction=0,
            require_full_query_rank=False,
        ),
        implementation_revision="8" * 40,
    )
    assert result.status == "evaluated"
    assert result.trace_query_precision >= 0
    assert result.gate_checks["query_rank"] is True


def test_tree_prior_with_child_and_singular_query_cases() -> None:
    item = batch()
    local = np.zeros((6, 3, 1))
    local[:, 0, 0] = 1
    item = replace(
        item,
        input_id=None,
        gauge_ids=("root", "child"),
        gauge_prior_id="9" * 64,
        local_gauge_jacobian=local,
        gauge_indices=np.array([0, 0, 0, 1, 1, 1]),
        parent_indices=np.array([-1, 0]),
        transition_matrices=np.array([[[0.0]], [[1.0]]]),
        innovation_scale_tril=np.array([[[1.0]], [[0.5]]]),
    )
    result = evaluate_deform360_joint_sparse_observability_v4(
        item,
        policy(maximum_single_camera_information_fraction=1),
        implementation_revision="8" * 40,
    )
    assert result.nuisance_dimension == 2

    zero_state = np.zeros((4, 3, 3))
    zero = evaluate_deform360_joint_sparse_observability_v4(
        batch(
            zero_state,
            cameras=("a", "a", "b", "b"),
            windows=("w0", "w1", "w0", "w1"),
            clusters=("c0", "c1", "c2", "c3"),
        ),
        policy(maximum_single_camera_information_fraction=1),
        implementation_revision="8" * 40,
    )
    assert zero.state_rank == 0
    assert zero.query_rank == 0
    assert zero.trace_query_precision == 0


def test_technical_failure_and_reports_preserve_information_boundary() -> None:
    p = policy(minimum_supported_objects=2)
    sheet_batch = batch(object_id="a", stratum="sheet")
    volume_batch = batch(object_id="b", stratum="volumetric")
    sheet = evaluate_deform360_joint_sparse_observability_v4(
        sheet_batch, p, implementation_revision="8" * 40
    )
    failed = technical_failure_deform360_joint_sparse_result_v4(
        volume_batch,
        p,
        implementation_revision="8" * 40,
        reason="synthetic-technical-failure",
        detail="boom",
    )
    assert failed.query_rank is None
    assert failed.failure_detail_sha256 == hashlib.sha256(b"boom").hexdigest()
    report = Deform360JointSparseDevelopmentReportV4(
        selection_artifact_sha256="1" * 64,
        visual_provider_lock_id="2" * 64,
        policy_id=p.policy_id,
        implementation_revision="8" * 40,
        results=(sheet, failed),
        source_artifacts={"manifest.json": "a" * 64},
        metadata={"cohort": "development"},
    )
    assert report.summary()["technical_failure_object_count"] == 1
    assert report.support_gate(p)["passed"] is False
    record = report.to_record(p)
    assert record["status"] == "development-technical-failures-retained"
    assert record["confirmation_access_authorized"] is False

    supported_volume = evaluate_deform360_joint_sparse_observability_v4(
        volume_batch, p, implementation_revision="8" * 40
    )
    supported = replace(
        report, report_id=None, results=(sheet, supported_volume)
    )
    assert supported.to_record(p)["status"] == "development-design-supported"
    unsupported_policy = policy(minimum_supported_objects=3)
    unsupported = Deform360JointSparseDevelopmentReportV4(
        selection_artifact_sha256="1" * 64,
        visual_provider_lock_id="2" * 64,
        policy_id=unsupported_policy.policy_id,
        implementation_revision="8" * 40,
        results=(
            replace(
                sheet, policy_id=unsupported_policy.policy_id, result_id=None
            ),
            replace(
                supported_volume,
                policy_id=unsupported_policy.policy_id,
                result_id=None,
            ),
        ),
    )
    assert unsupported.to_record(unsupported_policy)["status"] == "development-design-not-supported"
    with pytest.raises(ValueError, match="identity"):
        report.support_gate(policy())


def test_adapter_reuses_tree_sparse_contract_without_residuals() -> None:
    original = batch()
    adapted = SimpleNamespace(
        batch=SimpleNamespace(
            observation_covariance_m2=original.observation_covariance_m2,
            state_jacobian=original.state_jacobian,
            prior_reliability=original.prior_reliability,
            association_probability=original.association_probability,
            composite_weight=original.composite_weight,
            correlation_group_ids=original.correlation_group_ids,
            shared_bias_jacobian=original.shared_bias_jacobian,
            view_bias_jacobian=original.view_bias_jacobian,
        ),
        tree_gauge_design=SimpleNamespace(
            gauge_ids=original.gauge_ids,
            prior_id=original.gauge_prior_id,
            local_gauge_jacobian=original.local_gauge_jacobian,
            gauge_indices=original.gauge_indices,
            parent_indices=original.parent_indices,
            transition_matrices=original.transition_matrices,
            innovation_scale_tril=original.innovation_scale_tril,
        ),
        view_ids=original.camera_ids,
        provider_manifest_id="a" * 64,
        calibration_artifact_ids={"covariance": "b" * 64},
        runtime_revision_source="git-revision",
        observation_artifact_id=original.observation_artifact_id,
        linearization_artifact_id=original.linearization_artifact_id,
    )
    built = build_deform360_joint_sparse_factor_batch_from_tree_sparse_v4(
        adapted,
        selection_artifact_sha256=original.selection_artifact_sha256,
        visual_provider_lock_id=original.visual_provider_lock_id,
        implementation_revision=original.implementation_revision,
        object_id=original.object_id,
        episode_id=original.episode_id,
        stratum=original.stratum,
        factor_ids=original.factor_ids,
        spatial_cluster_ids=original.spatial_cluster_ids,
        query_jacobian=original.query_jacobian,
        source_artifacts=original.source_artifacts,
        metadata={"extra": True},
    )
    assert built.camera_ids == original.camera_ids
    assert built.window_ids == ("gauge-root",) * 6
    assert built.metadata["source_adapter"] == "ClaimBearingTreeSparseProb4DAdapterResult"
    assert built.information_boundary["prediction_residuals_used"] is False
    with pytest.raises(ValueError, match="view IDs"):
        build_deform360_joint_sparse_factor_batch_from_tree_sparse_v4(
            SimpleNamespace(**{**adapted.__dict__, "view_ids": ("one",)}),
            selection_artifact_sha256=original.selection_artifact_sha256,
            visual_provider_lock_id=original.visual_provider_lock_id,
            implementation_revision=original.implementation_revision,
            object_id=original.object_id,
            episode_id=0,
            stratum="sheet",
            factor_ids=original.factor_ids,
            spatial_cluster_ids=original.spatial_cluster_ids,
            query_jacobian=original.query_jacobian,
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"protocol_id": "wrong"}, "protocol"),
        ({"stratum": "wrong"}, "stratum"),
        ({"factor_ids": (digest("x"),)}, "length"),
        ({"gauge_ids": ()}, "gauge_ids"),
        ({"gauge_indices": np.full(6, 9)}, "gauge indices"),
        ({"parent_indices": np.array([0])}, "tree root"),
        ({"innovation_scale_tril": np.zeros((1, 1, 1))}, "diagonal"),
        ({"query_jacobian": np.ones((2, 3))}, "dependent"),
        ({"prior_reliability": np.full(6, 2.0)}, "probability"),
        ({"composite_weight": np.zeros(6)}, "probability"),
        ({"excluded_factor_count": -1}, "excluded_factor_count"),
    ],
)
def test_batch_contract_rejects_invalid_inputs(change: dict[str, object], match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        replace(batch(), input_id=None, **change)


def test_policy_and_result_contract_rejections() -> None:
    with pytest.raises(ValueError):
        policy(protocol_id="wrong")
    with pytest.raises(ValueError):
        policy(maximum_single_camera_information_fraction=2)
    with pytest.raises(ValueError):
        policy(minimum_distinct_cameras=0)
    with pytest.raises(ValueError):
        policy(information_boundary={})

    result = evaluate_deform360_joint_sparse_observability_v4(
        batch(), policy(), implementation_revision="8" * 40
    )
    with pytest.raises(ValueError, match="result_id"):
        replace(result, factor_count=result.factor_count + 1)
    with pytest.raises(ValueError, match="decision"):
        replace(result, gate_passed=not result.gate_passed, result_id=None)
    failure = technical_failure_deform360_joint_sparse_result_v4(
        batch(),
        policy(),
        implementation_revision="8" * 40,
        reason="failure",
        detail="detail",
    )
    with pytest.raises(ValueError, match="diagnostics"):
        replace(failure, state_dimension=3, result_id=None)
    with pytest.raises(ValueError, match="passed"):
        replace(failure, gate_passed=True, result_id=None)
