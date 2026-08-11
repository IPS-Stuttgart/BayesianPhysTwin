from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin import (
    _deform360_covariance_residual_history_decision_v1 as decision_module,
)
from bayesian_phystwin._deform360_covariance_residual_history_adapter_v1 import (
    ResidualHistoryAdapterV1,
    build_residual_history_adapter,
)
from bayesian_phystwin._deform360_covariance_residual_history_common_v1 import (
    CAMERA_PARTITION_NAMESPACE,
    DisjointCameraPartitionV1,
    ResidualHistoryDryRunPolicyV1,
    _boolean_array,
    _canonical_string,
    _finite_real,
    _integer_vector,
    _readonly_float_array,
    _required_sha256,
    _validate_covariance,
    camera_hardware_family,
    deterministic_disjoint_camera_partition,
)
from bayesian_phystwin._deform360_covariance_residual_history_decision_v1 import (
    ResidualHistoryDryRunDecisionV1,
    _horizon_bins,
    _physical_future_mean,
    run_source_only_residual_history_dry_run,
)


def _policy(**overrides: Any) -> ResidualHistoryDryRunPolicyV1:
    values: dict[str, Any] = {
        "minimum_prefix_frames": 2,
        "minimum_final_observed_count": 2,
        "minimum_final_observed_fraction": 0.5,
        "minimum_cameras_per_role": 1,
        "minimum_camera_families_per_role": 1,
        "covariance_scales": (8.0, 16.0, 16.0),
    }
    values.update(overrides)
    return ResidualHistoryDryRunPolicyV1(**values)


def _camera_ids() -> tuple[str, ...]:
    return (
        "alpha-recorder_cam0",
        "alpha-recorder_cam1",
        "beta-recorder_cam0",
        "gamma-recorder_cam0",
        "delta-recorder_cam0",
    )


def _partition(
    policy: ResidualHistoryDryRunPolicyV1 | None = None,
) -> DisjointCameraPartitionV1:
    selected = _policy() if policy is None else policy
    return deterministic_disjoint_camera_partition(_camera_ids(), policy=selected)


def _arrays(
    *,
    final_observed_count: int = 3,
) -> dict[str, np.ndarray]:
    prefix = np.zeros((2, 4, 3), dtype=np.float64)
    observation = np.full_like(prefix, np.nan)
    validity = np.zeros((2, 4), dtype=bool)
    validity[0] = True
    validity[-1, :final_observed_count] = True
    observation[validity] = prefix[validity] + np.array([0.1, -0.2, 0.3])
    future = np.zeros((3, 4, 3), dtype=np.float64)
    fallback_covariance = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (3, 4, 3, 3),
    ).copy()
    donor_covariance = np.broadcast_to(
        0.01 * np.eye(3, dtype=np.float64),
        (3, 4, 3, 3),
    ).copy()
    return {
        "physical_prefix_m": prefix,
        "provider_observation_prefix_m": observation,
        "observed_validity": validity,
        "physical_future_m": future,
        "physical_fallback_covariance_m2": fallback_covariance,
        "donor_covariance_m2": donor_covariance,
        "frame_indices": np.array([0, 1], dtype=np.int64),
        "material_ids": np.arange(4, dtype=np.int64),
        "future_horizon_bins": np.array([0, 1, 2], dtype=np.int64),
    }


def _adapter_kwargs() -> dict[str, Any]:
    arrays = _arrays()
    partition = _partition()
    residual = np.zeros_like(arrays["physical_prefix_m"])
    residual[arrays["observed_validity"]] = (
        arrays["provider_observation_prefix_m"][arrays["observed_validity"]]
        - arrays["physical_prefix_m"][arrays["observed_validity"]]
    )
    return {
        "source_unit_id": "source-unit",
        "frame_indices": arrays["frame_indices"],
        "material_ids": arrays["material_ids"],
        "residual_history_m": residual,
        "observed_validity": arrays["observed_validity"],
        "partition": partition,
        "provider_reconstruction_artifact_id": "a" * 64,
        "scoring_reconstruction_artifact_id": "b" * 64,
        "baseline_prefix_sha256": "c" * 64,
        "observation_prefix_sha256": "d" * 64,
        "policy_id": "e" * 64,
        "metadata": {"opened_source_only": True},
    }


def _build_adapter(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    policy: ResidualHistoryDryRunPolicyV1 | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResidualHistoryAdapterV1:
    selected_arrays = _arrays() if arrays is None else arrays
    selected_policy = _policy() if policy is None else policy
    partition = _partition(selected_policy)
    return build_residual_history_adapter(
        selected_arrays["physical_prefix_m"],
        selected_arrays["provider_observation_prefix_m"],
        selected_arrays["observed_validity"],
        frame_indices=selected_arrays["frame_indices"],
        material_ids=selected_arrays["material_ids"],
        camera_ids=_camera_ids(),
        provider_camera_ids=partition.provider_camera_ids,
        scoring_camera_ids=partition.scoring_camera_ids,
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
        source_unit_id="source-unit",
        policy=selected_policy,
        metadata=metadata,
    )


def _run(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    policy: ResidualHistoryDryRunPolicyV1 | None = None,
    metadata: dict[str, Any] | None = None,
    reference_predictor_id: str = "last-residual",
    covariance_donor_id: str = "independent-donor",
):
    selected_arrays = _arrays() if arrays is None else arrays
    selected_policy = _policy() if policy is None else policy
    partition = _partition(selected_policy)
    return run_source_only_residual_history_dry_run(
        selected_arrays["physical_prefix_m"],
        selected_arrays["provider_observation_prefix_m"],
        selected_arrays["observed_validity"],
        selected_arrays["physical_future_m"],
        selected_arrays["physical_fallback_covariance_m2"],
        selected_arrays["donor_covariance_m2"],
        frame_indices=selected_arrays["frame_indices"],
        material_ids=selected_arrays["material_ids"],
        future_horizon_bins=selected_arrays["future_horizon_bins"],
        camera_ids=_camera_ids(),
        provider_camera_ids=partition.provider_camera_ids,
        scoring_camera_ids=partition.scoring_camera_ids,
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
        source_unit_id="source-unit",
        reference_predictor_id=reference_predictor_id,
        covariance_donor_id=covariance_donor_id,
        policy=selected_policy,
        metadata=metadata,
    )


def _decision_kwargs(*, accepted: bool) -> dict[str, Any]:
    return {
        "source_unit_id": "source-unit",
        "adapter_id": "a" * 64,
        "policy_id": "b" * 64,
        "partition_id": "c" * 64,
        "accepted": accepted,
        "fallback_reasons": () if accepted else ("support",),
        "final_observed_count": 3,
        "final_observed_fraction": 0.75,
        "future_horizon_bins_sha256": "d" * 64,
        "physical_future_mean_sha256": "e" * 64,
        "physical_fallback_covariance_sha256": "f" * 64,
        "deployed_mean_sha256": "1" * 64,
        "deployed_covariance_sha256": "2" * 64,
        "hybrid_artifact_id": "3" * 64 if accepted else None,
        "hybrid_reference_mean_identity_preserved": accepted,
        "exact_physical_fallback_mean_identity_preserved": not accepted,
        "exact_physical_fallback_covariance_identity_preserved": not accepted,
        "metadata": {"source": "opened"},
    }


@pytest.mark.parametrize("value", ["", " padded", 1])
def test_canonical_string_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError, match="canonical string"):
        _canonical_string(value, name="value")


@pytest.mark.parametrize(
    ("value", "minimum", "message"),
    [
        (True, None, "finite real"),
        ("1", None, "finite real"),
        (np.inf, None, "finite"),
        (-0.1, 0.0, "at least"),
    ],
)
def test_finite_real_rejects_invalid_values(
    value: object,
    minimum: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _finite_real(value, name="value", minimum=minimum)
    assert _finite_real(2, name="value", minimum=1.0) == 2.0


@pytest.mark.parametrize("value", [None, "A" * 64, "a" * 63])
def test_required_digest_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _required_sha256(value, name="digest")
    assert _required_sha256("a" * 64, name="digest") == "a" * 64


def test_array_helpers_reject_shape_dtype_and_finiteness_errors() -> None:
    with pytest.raises(ValueError, match="2 dimensions"):
        _readonly_float_array([1.0], name="array", ndim=2)
    with pytest.raises(ValueError, match="real numeric"):
        _readonly_float_array([["x"]], name="array", ndim=2)
    with pytest.raises(ValueError, match="must be finite"):
        _readonly_float_array([[np.nan]], name="array", ndim=2)
    allowed_nan = _readonly_float_array(
        [[np.nan]],
        name="array",
        ndim=2,
        finite=False,
    )
    assert np.isnan(allowed_nan[0, 0])
    assert not allowed_nan.flags.writeable

    with pytest.raises(ValueError, match="one-dimensional"):
        _integer_vector([[1]], name="indices")
    with pytest.raises(ValueError, match="nonempty"):
        _integer_vector(np.array([], dtype=np.int64), name="indices")
    np.testing.assert_array_equal(_integer_vector([1, 2], name="indices"), [1, 2])

    with pytest.raises(ValueError, match="Boolean array"):
        _boolean_array(np.ones((2, 2), dtype=np.int64), name="mask", shape=(2, 2))
    with pytest.raises(ValueError, match="Boolean array"):
        _boolean_array(np.ones((1, 2), dtype=bool), name="mask", shape=(2, 2))
    mask = _boolean_array(np.ones((2, 2), dtype=bool), name="mask", shape=(2, 2))
    assert not mask.flags.writeable


def test_covariance_validation_rejects_every_structural_failure() -> None:
    shape = (2, 3, 3)
    with pytest.raises(TypeError, match="NumPy array"):
        _validate_covariance(
            [[[1.0] * 3] * 3] * 2,
            name="covariance",
            expected_shape=shape,
            preserve_identity=True,
        )
    with pytest.raises(ValueError, match="dtype float64"):
        _validate_covariance(
            np.zeros(shape, dtype=np.float32),
            name="covariance",
            expected_shape=shape,
            preserve_identity=True,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _validate_covariance(
            np.zeros(shape, dtype=np.float64)[:, :, ::-1],
            name="covariance",
            expected_shape=shape,
            preserve_identity=True,
        )
    with pytest.raises(ValueError, match="must have shape"):
        _validate_covariance(
            np.zeros((1, 3, 3), dtype=np.float64),
            name="covariance",
            expected_shape=shape,
        )
    nonfinite = np.broadcast_to(np.eye(3), shape).copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _validate_covariance(nonfinite, name="covariance", expected_shape=shape)
    asymmetric = np.broadcast_to(np.eye(3), shape).copy()
    asymmetric[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        _validate_covariance(asymmetric, name="covariance", expected_shape=shape)
    indefinite = np.broadcast_to(np.eye(3), shape).copy()
    indefinite[..., 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _validate_covariance(indefinite, name="covariance", expected_shape=shape)

    valid = np.broadcast_to(np.eye(3), shape).copy()
    assert (
        _validate_covariance(
            valid,
            name="covariance",
            expected_shape=shape,
            preserve_identity=True,
        )
        is valid
    )
    copied = _validate_covariance(
        valid.tolist(),
        name="covariance",
        expected_shape=shape,
    )
    assert not copied.flags.writeable


@pytest.mark.parametrize(
    ("camera_id", "expected"),
    [
        ("rig-cam0", "rig"),
        ("rig-camera-3-left", "rig"),
        ("rig-view-right", "rig"),
        ("rig-stream", "rig"),
        ("rig-sensor9-bottom", "rig"),
    ],
)
def test_camera_family_normalizes_supported_suffixes(
    camera_id: str,
    expected: str,
) -> None:
    assert camera_hardware_family(camera_id) == expected


@pytest.mark.parametrize("camera_id", ["---", "cam0", "camera-3-left"])
def test_camera_family_rejects_missing_stable_prefix(camera_id: str) -> None:
    with pytest.raises(ValueError, match="family|prefix"):
        camera_hardware_family(camera_id)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"minimum_prefix_frames": 0}, "minimum_prefix_frames"),
        ({"minimum_final_observed_fraction": 1.1}, "must not exceed one"),
        ({"covariance_scales": (8.0, 16.0)}, "early, middle, and late"),
        ({"covariance_scales": (0.5, 16.0, 16.0)}, "at least"),
        ({"policy_id": "f" * 64}, "does not match"),
    ],
)
def test_policy_rejects_invalid_or_mismatched_contracts(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _policy(**overrides)


def test_policy_compatibility_aliases_and_explicit_id() -> None:
    policy = _policy()
    assert policy.minimum_camera_count_per_role == policy.minimum_cameras_per_role
    assert (
        policy.minimum_camera_family_count_per_role
        == policy.minimum_camera_families_per_role
    )
    rebound = _policy(policy_id=policy.policy_id)
    assert rebound.policy_id == policy.policy_id


def _partition_kwargs() -> dict[str, Any]:
    return {
        "provider_camera_ids": ("alpha_cam0",),
        "scoring_camera_ids": ("beta_cam0",),
        "provider_family_ids": ("alpha",),
        "scoring_family_ids": ("beta",),
        "namespace": CAMERA_PARTITION_NAMESPACE,
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"provider_camera_ids": ("alpha_cam0", "alpha_cam0")}, "sorted and unique"),
        (
            {
                "scoring_camera_ids": ("alpha_cam0",),
                "scoring_family_ids": ("alpha",),
            },
            "cameras must be disjoint",
        ),
        (
            {
                "provider_family_ids": ("shared",),
                "scoring_family_ids": ("shared",),
            },
            "families must be disjoint",
        ),
        (
            {
                "provider_camera_ids": (),
                "provider_family_ids": (),
            },
            "requires both roles",
        ),
        ({"provider_family_ids": ("wrong",)}, "provider families"),
        ({"scoring_family_ids": ("wrong",)}, "scoring families"),
        ({"namespace": ""}, "canonical string"),
        ({"partition_id": "f" * 64}, "does not match"),
    ],
)
def test_partition_rejects_inconsistent_fields(
    changes: dict[str, Any],
    message: str,
) -> None:
    kwargs = _partition_kwargs()
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        DisjointCameraPartitionV1(**kwargs)


def test_partition_accepts_explicit_content_id_and_aliases() -> None:
    partition = DisjointCameraPartitionV1(**_partition_kwargs())
    rebound = DisjointCameraPartitionV1(
        **_partition_kwargs(),
        partition_id=partition.partition_id,
    )
    assert rebound.provider_camera_families == ("alpha",)
    assert rebound.scoring_camera_families == ("beta",)


def test_partition_builder_rejects_wrong_policy_and_duplicate_cameras() -> None:
    with pytest.raises(TypeError, match="policy"):
        deterministic_disjoint_camera_partition(
            _camera_ids(),
            policy=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unique"):
        deterministic_disjoint_camera_partition(
            (*_camera_ids(), _camera_ids()[0]),
            policy=_policy(),
        )


def test_adapter_dataclass_rejects_structural_and_identity_mismatches() -> None:
    kwargs = _adapter_kwargs()

    residual = np.asarray(kwargs["residual_history_m"]).copy()
    kwargs_bad_shape = dict(kwargs, residual_history_m=residual[:, :-1])
    with pytest.raises(ValueError, match="must have shape"):
        ResidualHistoryAdapterV1(**kwargs_bad_shape)

    validity = np.asarray(kwargs["observed_validity"])
    residual[~validity] = 1.0
    with pytest.raises(ValueError, match="zero storage"):
        ResidualHistoryAdapterV1(**dict(kwargs, residual_history_m=residual))

    with pytest.raises(TypeError, match="partition"):
        ResidualHistoryAdapterV1(**dict(kwargs, partition=object()))

    adapter = ResidualHistoryAdapterV1(**kwargs)
    with pytest.raises(ValueError, match="adapter_id does not match"):
        ResidualHistoryAdapterV1(**dict(kwargs, adapter_id="f" * 64))
    rebound = ResidualHistoryAdapterV1(**dict(kwargs, adapter_id=adapter.adapter_id))
    assert rebound.adapter_id == adapter.adapter_id
    assert rebound.descriptor()["metadata"] == {"opened_source_only": True}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("short-prefix", "supported shape"),
        ("wrong-last-axis", "supported shape"),
        ("observation-shape", "observation shape"),
        ("frame-length", "frame_indices length"),
        ("material-length", "material_ids length"),
        ("scoring-roster", "declared scoring cameras"),
    ],
)
def test_adapter_builder_rejects_invalid_shapes_and_rosters(
    mutation: str,
    message: str,
) -> None:
    arrays = _arrays()
    policy = _policy()
    partition = _partition(policy)
    physical_prefix = arrays["physical_prefix_m"]
    observation = arrays["provider_observation_prefix_m"]
    frame_indices = arrays["frame_indices"]
    material_ids = arrays["material_ids"]
    scoring_ids = partition.scoring_camera_ids

    if mutation == "short-prefix":
        physical_prefix = physical_prefix[:1]
        observation = observation[:1]
    elif mutation == "wrong-last-axis":
        physical_prefix = np.zeros((2, 4, 2), dtype=np.float64)
        observation = np.zeros_like(physical_prefix)
    elif mutation == "observation-shape":
        observation = observation[:, :-1]
    elif mutation == "frame-length":
        frame_indices = frame_indices[:1]
    elif mutation == "material-length":
        material_ids = material_ids[:3]
    elif mutation == "scoring-roster":
        scoring_ids = partition.provider_camera_ids

    with pytest.raises(ValueError, match=message):
        build_residual_history_adapter(
            physical_prefix,
            observation,
            arrays["observed_validity"],
            frame_indices=frame_indices,
            material_ids=material_ids,
            camera_ids=_camera_ids(),
            provider_camera_ids=partition.provider_camera_ids,
            scoring_camera_ids=scoring_ids,
            provider_reconstruction_artifact_id="a" * 64,
            scoring_reconstruction_artifact_id="b" * 64,
            source_unit_id="source-unit",
            policy=policy,
        )


def test_adapter_builder_rejects_wrong_policy_and_accepts_invalid_row_nan() -> None:
    arrays = _arrays()
    partition = _partition()
    with pytest.raises(TypeError, match="policy"):
        build_residual_history_adapter(
            arrays["physical_prefix_m"],
            arrays["provider_observation_prefix_m"],
            arrays["observed_validity"],
            frame_indices=arrays["frame_indices"],
            material_ids=arrays["material_ids"],
            camera_ids=_camera_ids(),
            provider_camera_ids=partition.provider_camera_ids,
            scoring_camera_ids=partition.scoring_camera_ids,
            provider_reconstruction_artifact_id="a" * 64,
            scoring_reconstruction_artifact_id="b" * 64,
            source_unit_id="source-unit",
            policy=object(),  # type: ignore[arg-type]
        )
    adapter = _build_adapter(arrays, metadata={"provenance": "opened-source"})
    assert adapter.metadata["provenance"] == "opened-source"
    assert np.all(adapter.residual_history_m[~adapter.observed_validity] == 0.0)


def test_decision_accepts_valid_admission_and_fallback_forms() -> None:
    accepted = ResidualHistoryDryRunDecisionV1(**_decision_kwargs(accepted=True))
    fallback = ResidualHistoryDryRunDecisionV1(**_decision_kwargs(accepted=False))
    assert accepted.descriptor()["accepted"] is True
    assert fallback.descriptor()["fallback_reasons"] == ["support"]
    assert accepted.decision_id is not None
    rebound = ResidualHistoryDryRunDecisionV1(
        **dict(_decision_kwargs(accepted=True), decision_id=accepted.decision_id)
    )
    assert rebound.decision_id == accepted.decision_id


@pytest.mark.parametrize(
    ("accepted", "changes", "message"),
    [
        (True, {"accepted": 1}, "accepted must be a Boolean"),
        (False, {"fallback_reasons": ["support"]}, "canonical tuple"),
        (
            False,
            {"fallback_reasons": ("support", "support")},
            "sorted and unique",
        ),
        (False, {"final_observed_fraction": 1.1}, "must not exceed one"),
        (False, {"future_horizon_bins_sha256": "bad"}, "lowercase"),
        (
            False,
            {"hybrid_reference_mean_identity_preserved": 1},
            "must be Booleans",
        ),
        (True, {"fallback_reasons": ("support",)}, "no fallback"),
        (True, {"hybrid_artifact_id": None}, "require one hybrid"),
        (
            True,
            {"hybrid_reference_mean_identity_preserved": False},
            "preserve its reference mean",
        ),
        (
            True,
            {"exact_physical_fallback_mean_identity_preserved": True},
            "must not claim fallback identity",
        ),
        (False, {"fallback_reasons": ()}, "require reasons"),
        (False, {"hybrid_artifact_id": "3" * 64}, "no hybrid"),
        (
            False,
            {"hybrid_reference_mean_identity_preserved": True},
            "has no hybrid reference",
        ),
        (
            False,
            {"exact_physical_fallback_mean_identity_preserved": False},
            "preserve both physical objects",
        ),
        (True, {"decision_id": "4" * 64}, "does not match"),
    ],
)
def test_decision_rejects_inconsistent_contracts(
    accepted: bool,
    changes: dict[str, Any],
    message: str,
) -> None:
    kwargs = _decision_kwargs(accepted=accepted)
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        ResidualHistoryDryRunDecisionV1(**kwargs)


def test_physical_future_contract_rejects_type_dtype_shape_layout_and_nan() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        _physical_future_mean([[[0.0, 0.0, 0.0]]], material_count=1)
    with pytest.raises(ValueError, match="dtype float64"):
        _physical_future_mean(
            np.zeros((1, 1, 3), dtype=np.float32),
            material_count=1,
        )
    with pytest.raises(ValueError, match="shape"):
        _physical_future_mean(np.zeros((0, 1, 3)), material_count=1)
    with pytest.raises(ValueError, match="C-contiguous"):
        _physical_future_mean(
            np.zeros((2, 1, 3), dtype=np.float64)[::-1],
            material_count=1,
        )
    nonfinite = np.zeros((1, 1, 3), dtype=np.float64)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _physical_future_mean(nonfinite, material_count=1)
    valid = np.zeros((1, 1, 3), dtype=np.float64)
    assert _physical_future_mean(valid, material_count=1) is valid


def test_horizon_bins_reject_wrong_length_and_accept_all_labels() -> None:
    with pytest.raises(ValueError, match="one entry"):
        _horizon_bins(np.array([0, 1]), future_count=3)
    with pytest.raises(ValueError, match="early/middle/late"):
        _horizon_bins(np.array([-1, 1, 2]), future_count=3)
    np.testing.assert_array_equal(
        _horizon_bins(np.array([0, 1, 2]), future_count=3),
        [0, 1, 2],
    )


def test_covariance_rejection_preserves_metadata_and_invalid_ids_fall_back() -> None:
    arrays = _arrays()
    arrays["donor_covariance_m2"][..., 0, 0] = -1.0
    result = _run(arrays, metadata={"trial": 7})
    assert result.decision.metadata["dry_run_metadata"] == {"trial": 7}
    assert result.decision.metadata["covariance_rejection_type"] == "ValueError"

    invalid_reference = _run(reference_predictor_id="")
    assert invalid_reference.decision.fallback_reasons == (
        "covariance-contract-rejection",
    )
    invalid_donor = _run(covariance_donor_id="")
    assert invalid_donor.decision.fallback_reasons == ("covariance-contract-rejection",)


def test_support_fallback_and_admission_preserve_supplied_metadata() -> None:
    fallback = _run(
        _arrays(final_observed_count=1),
        metadata={"case": "low-support"},
    )
    assert fallback.decision.metadata == {"case": "low-support"}
    assert fallback.mean_m is fallback.mean_m
    admitted = _run(metadata={"case": "admitted"})
    assert admitted.accepted
    assert admitted.decision.metadata == {"case": "admitted"}


def test_defensive_identity_assertions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied_hybrid(
        reference_mean: np.ndarray,
        donor_covariance_m2: object,
        **_: object,
    ) -> SimpleNamespace:
        del donor_covariance_m2
        return SimpleNamespace(
            mean_m=reference_mean.copy(),
            covariance_m2=np.zeros(reference_mean.shape + (3,), dtype=np.float64),
            record=SimpleNamespace(artifact_id="f" * 64),
        )

    monkeypatch.setattr(
        decision_module,
        "compose_covariance_only_hybrid",
        copied_hybrid,
    )
    with pytest.raises(AssertionError, match="copied the last-residual mean"):
        _run()


def test_fallback_identity_assertion_detects_internal_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied_result(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            mean_m=np.array(kwargs["mean_m"], copy=True),
            covariance_m2=np.array(kwargs["covariance_m2"], copy=True),
            adapter=kwargs["adapter"],
            decision=kwargs["decision"],
            hybrid=kwargs["hybrid"],
        )

    monkeypatch.setattr(
        decision_module,
        "ResidualHistoryDryRunResultV1",
        copied_result,
    )
    with pytest.raises(AssertionError, match="fallback object identity"):
        _run(_arrays(final_observed_count=1))
