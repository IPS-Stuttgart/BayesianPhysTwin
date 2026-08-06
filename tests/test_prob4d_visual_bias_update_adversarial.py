from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.prob4d_visual_bias_update as update_module
import test_prob4d_visual_bias_update as fixtures
from bayesian_phystwin.prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
    PROB4D_VISUAL_BIAS_REPARAMETERIZATION,
    ClaimBearingProb4DVisualBiasUpdateV2,
    Prob4DVisualBiasBindingV1,
    update_claim_bearing_prob4d_with_visual_bias_from_artifacts,
    validate_prob4d_visual_bias_nuisance,
)
from bayesian_phystwin.prospective_prob4d_update import ClaimBearingProb4DUpdateV1


def _visual_lineage(
    observation: Any,
    binding: Prob4DVisualBiasBindingV1,
    *,
    measured_projection: float = 0.0,
    **overrides: object,
) -> dict[str, object]:
    lineage = fixtures._claim_lineage(observation)
    lineage.update(
        {
            "prob4d_visual_bias_artifact_id": binding.artifact_id,
            "prob4d_visual_bias_observation_identity_sha256": (
                binding.observation_identity_sha256
            ),
            "prob4d_visual_bias_reparameterization": (
                PROB4D_VISUAL_BIAS_REPARAMETERIZATION
            ),
            "prob4d_visual_bias_prior_std_m": 0.02,
            "prob4d_visual_bias_gauge_orthogonalized": True,
            "prob4d_visual_bias_measured_gauge_projection": measured_projection,
            "prob4d_visual_bias_marginal_covariance_added": False,
        }
    )
    lineage.update(overrides)
    return lineage


def _base_update(
    observation: Any,
    binding: Prob4DVisualBiasBindingV1,
    *,
    shared_count: int | None = None,
    view_count: int = 0,
    lineage_overrides: dict[str, object] | None = None,
) -> ClaimBearingProb4DUpdateV1:
    shared = binding.latent_dimension if shared_count is None else shared_count
    adapted = fixtures._adapted_batch(
        observation,
        np.zeros((2, 3, shared), dtype=np.float64),
        np.zeros((2, 3, view_count), dtype=np.float64),
    )
    lineage = _visual_lineage(
        observation,
        binding,
        **({} if lineage_overrides is None else lineage_overrides),
    )
    batch = replace(adapted.batch, metadata=lineage)
    return ClaimBearingProb4DUpdateV1(
        result=fixtures._solver_result(batch),
        observation_artifact_id=observation.artifact_id,
        linearization_artifact_id=fixtures.LINEARIZATION_ID,
        provider_manifest_id=fixtures.PROVIDER_ID,
        calibration_artifact_ids=fixtures.CALIBRATION_IDS,
        runtime_revision_source="independent-vcs-check",
        runtime_revision_independently_verified=True,
    )


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    [
        (
            lambda: update_module._sha256(7, name="digest"),
            ValueError,
            "literal string",
        ),
        (
            lambda: update_module._sha256("bad", name="digest"),
            ValueError,
            "lowercase SHA-256",
        ),
        (
            lambda: update_module._finite_real(True, name="value"),
            ValueError,
            "finite real",
        ),
        (
            lambda: update_module._finite_real([1.0], name="value"),
            ValueError,
            "finite real",
        ),
        (
            lambda: update_module._finite_real(np.nan, name="value"),
            ValueError,
            "finite",
        ),
        (
            lambda: update_module._finite_real(-1.0, name="value", minimum=0.0),
            ValueError,
            "at least",
        ),
        (
            lambda: update_module._finite_real(
                0.0,
                name="value",
                strictly_positive=True,
            ),
            ValueError,
            "positive",
        ),
        (
            lambda: update_module._canonical_strings(["a"], name="names"),
            TypeError,
            "canonical tuple",
        ),
        (
            lambda: update_module._canonical_strings((), name="names"),
            ValueError,
            "nonempty",
        ),
        (
            lambda: update_module._canonical_strings(("a", "a"), name="names"),
            ValueError,
            "unique",
        ),
        (
            lambda: update_module._immutable_array(
                ["x"],
                dtype=np.dtype(object),
            ),
            TypeError,
            "Python objects",
        ),
        (
            lambda: update_module._validated_calibration_ids(None),
            ValueError,
            "missing",
        ),
        (
            lambda: update_module._validated_calibration_ids({}),
            ValueError,
            "missing",
        ),
        (
            lambda: update_module._validated_calibration_ids({"": "d" * 64}),
            ValueError,
            "nonempty",
        ),
        (
            lambda: update_module._validated_calibration_ids({"gauge": "bad"}),
            ValueError,
            "calibration artifact",
        ),
        (
            lambda: update_module._runtime_revision_source(7),
            ValueError,
            "nonempty literal string",
        ),
        (
            lambda: update_module._runtime_revision_source(""),
            ValueError,
            "nonempty literal string",
        ),
    ],
)
def test_private_contract_validators_fail_closed(
    operation: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        operation()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.eye(2, dtype=np.int64), "float64"),
        (np.asarray([[1.0, np.nan], [0.0, 1.0]]), "finite"),
        (np.asarray([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.asarray([[1.0, 0.0], [0.0, -1.0]]), "positive semidefinite"),
    ],
)
def test_psd_validation_rejects_invalid_covariance(
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        update_module._symmetric_psd(value, name="covariance", dimension=2)


def test_psd_validation_rejects_changed_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        update_module._symmetric_psd(
            np.eye(3, dtype=np.float64),
            name="covariance",
            dimension=2,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"row_bias_indices": np.asarray([0, 1], dtype=np.int32)},
            "int64",
        ),
        (
            {"row_bias_indices": np.zeros((1, 2), dtype=np.int64)},
            "one-dimensional",
        ),
        (
            {
                "row_bias_indices": np.zeros(0, dtype=np.int64),
                "bias_jacobian": np.zeros((0, 3, 1), dtype=np.float64),
            },
            "requires observation rows",
        ),
        (
            {"row_bias_indices": np.asarray([0, 2], dtype=np.int64)},
            "unknown bias ID",
        ),
        (
            {"bias_jacobian": np.zeros((2, 3, 1), dtype=np.float32)},
            "bias_jacobian must be float64",
        ),
        (
            {"bias_jacobian": np.zeros((2, 2, 1), dtype=np.float64)},
            "bias_jacobian must be float64",
        ),
        (
            {
                "bias_jacobian": np.asarray(
                    [
                        [[np.nan], [0.0], [0.0]],
                        [[0.0], [1.0], [0.0]],
                    ],
                    dtype=np.float64,
                )
            },
            "bias_jacobian must be finite",
        ),
        (
            {"joint_bias_covariance": np.eye(2, dtype=np.float32)},
            "joint_bias_covariance must be float64",
        ),
        (
            {
                "joint_bias_covariance": np.asarray(
                    [[1.0, np.nan], [np.nan, 1.0]],
                    dtype=np.float64,
                )
            },
            "joint_bias_covariance must be finite",
        ),
        (
            {
                "joint_bias_covariance": np.asarray(
                    [[1.0, 0.5], [0.0, 1.0]],
                    dtype=np.float64,
                )
            },
            "joint_bias_covariance must be symmetric",
        ),
        (
            {
                "joint_bias_covariance": np.asarray(
                    [[1.0, 0.0], [0.0, -1.0]],
                    dtype=np.float64,
                )
            },
            "positive semidefinite",
        ),
        ({"orthogonalization_semantics": 7}, "literal string"),
        ({"orthogonalization_semantics": "unknown"}, "unsupported"),
        ({"maximum_gauge_projection": True}, "finite real"),
        ({"gauge_projection_tolerance": 0.0}, "positive"),
        (
            {
                "maximum_gauge_projection": 0.1,
                "gauge_projection_tolerance": 0.01,
            },
            "exceeds",
        ),
    ],
)
def test_binding_rejects_malformed_content(
    changes: dict[str, object],
    message: str,
) -> None:
    observation = fixtures._observation()
    binding = fixtures._binding(observation)
    with pytest.raises((TypeError, ValueError), match=message):
        replace(binding, **changes, artifact_id=None)


def test_binding_rejects_changed_artifact_id() -> None:
    observation = fixtures._observation()
    binding = fixtures._binding(observation)
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        replace(binding, artifact_id="f" * 64)


def test_binding_summary_and_coefficient_order_are_stable() -> None:
    observation = fixtures._observation()
    binding = fixtures._binding(observation)
    assert binding.coefficient_names == (
        "camera-0:ray-depth",
        "camera-1:ray-depth",
    )
    assert binding.summary()["latent_dimension"] == 2
    assert set(binding.arrays()) == {
        "row_bias_indices",
        "bias_jacobian",
        "joint_bias_covariance",
    }


def test_validation_rejects_wrong_observation_and_sidecar_types() -> None:
    observation = fixtures._observation()
    with pytest.raises(TypeError, match="ObservationBeliefV1"):
        validate_prob4d_visual_bias_nuisance(
            object(),  # type: ignore[arg-type]
            object(),
        )
    with pytest.raises(TypeError, match="VisualBiasNuisanceV1"):
        validate_prob4d_visual_bias_nuisance(observation, object())
    with pytest.raises(ValueError, match="require_gauge_orthogonalized"):
        validate_prob4d_visual_bias_nuisance(
            observation,
            fixtures._binding(observation),
            require_gauge_orthogonalized=1,  # type: ignore[arg-type]
        )


def test_validation_rejects_changed_row_count() -> None:
    observation = fixtures._observation()
    _, _, identity_sha = fixtures.prob4d_observation_identity_summary(observation)
    sidecar = Prob4DVisualBiasBindingV1(
        observation_artifact_id=observation.artifact_id,
        observation_identity_sha256=identity_sha,
        bias_ids=("camera-0",),
        basis_names=("ray-depth",),
        row_bias_indices=np.asarray([0], dtype=np.int64),
        bias_jacobian=np.asarray(
            [[[1.0], [0.0], [0.0]]],
            dtype=np.float64,
        ),
        joint_bias_covariance=np.asarray([[1e-6]], dtype=np.float64),
        orthogonalization_semantics=PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
        maximum_gauge_projection=0.0,
        gauge_projection_tolerance=1e-8,
    )
    with pytest.raises(ValueError, match="row count"):
        validate_prob4d_visual_bias_nuisance(observation, sidecar)


def test_validation_allows_explicit_nonclaim_inspection() -> None:
    observation = fixtures._observation()
    sidecar = fixtures._binding(observation, orthogonalized=False)
    validated = validate_prob4d_visual_bias_nuisance(
        observation,
        sidecar,
        require_gauge_orthogonalized=False,
    )
    assert validated.orthogonalization_semantics == "not-orthogonalized"


@pytest.mark.parametrize(
    ("bias", "gauge", "covariance", "message"),
    [
        (
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3, 0), dtype=np.float64),
            np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            "bias_design",
        ),
        (
            np.zeros((2, 3, 1), dtype=np.float64),
            np.zeros((2, 2, 1), dtype=np.float64),
            np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            "gauge_design",
        ),
        (
            np.zeros((2, 3, 1), dtype=np.float64),
            np.zeros((2, 3, 1), dtype=np.float64),
            np.eye(3, dtype=np.float64),
            "conditional covariance",
        ),
        (
            np.full((2, 3, 1), np.nan, dtype=np.float64),
            np.zeros((2, 3, 1), dtype=np.float64),
            np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            "designs must be finite",
        ),
        (
            np.zeros((2, 3, 1), dtype=np.float64),
            np.zeros((2, 3, 1), dtype=np.float64),
            np.full((2, 3, 3), np.nan, dtype=np.float64),
            "covariance must be finite",
        ),
    ],
)
def test_gauge_projection_rejects_malformed_inputs(
    bias: np.ndarray,
    gauge: np.ndarray,
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        update_module._maximum_conditional_gauge_projection(
            bias,
            gauge,
            covariance,
        )


def test_gauge_projection_rejects_bad_covariance_blocks() -> None:
    bias = np.zeros((1, 3, 1), dtype=np.float64)
    gauge = np.zeros((1, 3, 1), dtype=np.float64)
    nonsymmetric = np.asarray(
        [
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="symmetric"):
        update_module._maximum_conditional_gauge_projection(
            bias,
            gauge,
            nonsymmetric,
        )
    with pytest.raises(ValueError, match="positive definite"):
        update_module._maximum_conditional_gauge_projection(
            bias,
            gauge,
            np.zeros((1, 3, 3), dtype=np.float64),
        )


def test_gauge_projection_handles_empty_and_numerically_empty_spans() -> None:
    covariance = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    bias = np.zeros((2, 3, 1), dtype=np.float64)
    assert (
        update_module._maximum_conditional_gauge_projection(
            bias,
            np.zeros((2, 3, 0), dtype=np.float64),
            covariance,
        )
        == 0.0
    )
    assert (
        update_module._maximum_conditional_gauge_projection(
            bias,
            np.zeros((2, 3, 1), dtype=np.float64),
            covariance,
        )
        == 0.0
    )
    gauge = np.zeros((2, 3, 1), dtype=np.float64)
    gauge[:, 0, 0] = 1.0
    assert (
        update_module._maximum_conditional_gauge_projection(
            bias,
            gauge,
            covariance,
        )
        == 0.0
    )


def test_v2_contract_rejects_wrong_types_and_dimensions() -> None:
    observation = fixtures._observation()
    binding = fixtures._binding(observation)
    valid_base = _base_update(observation, binding)
    with pytest.raises(TypeError, match="base_update"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=object(),  # type: ignore[arg-type]
            visual_bias=binding,
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=0.0,
        )
    with pytest.raises(TypeError, match="visual_bias"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=valid_base,
            visual_bias=object(),  # type: ignore[arg-type]
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=0.0,
        )
    with pytest.raises(ValueError, match="overlaps"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=valid_base,
            visual_bias=binding,
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=1e-4,
        )
    with pytest.raises(ValueError, match="coefficient dimension"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=_base_update(observation, binding, shared_count=1),
            visual_bias=binding,
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=0.0,
        )
    with pytest.raises(ValueError, match="view-bias"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=_base_update(observation, binding, view_count=1),
            visual_bias=binding,
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=0.0,
        )


def test_v2_contract_rejects_observation_and_lineage_mismatch() -> None:
    observation = fixtures._observation()
    binding = fixtures._binding(observation)
    valid_base = _base_update(observation, binding)
    mismatched = replace(
        binding,
        observation_artifact_id="1" * 64,
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="different observations"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=valid_base,
            visual_bias=mismatched,
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=0.0,
        )
    with pytest.raises(ValueError, match="does not bind"):
        ClaimBearingProb4DVisualBiasUpdateV2(
            base_update=_base_update(
                observation,
                binding,
                lineage_overrides={"prob4d_visual_bias_artifact_id": "2" * 64},
            ),
            visual_bias=binding,
            shared_bias_prior_std_m=0.02,
            measured_gauge_projection=0.0,
        )


def test_one_call_rejects_wrong_config_before_adapter() -> None:
    observation = fixtures._observation()
    with pytest.raises(TypeError, match="PriorAwareGaugeConfigV1"):
        update_claim_bearing_prob4d_with_visual_bias_from_artifacts(
            observation,
            SimpleNamespace(artifact_id=fixtures.LINEARIZATION_ID),
            visual_bias_nuisance=fixtures._binding(observation),
            physical_prediction_xyz_m=np.zeros((2, 3), dtype=np.float64),
            config=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        (
            {"prob4d_claim_bearing_provider_manifest_id": "bad"},
            "provider_manifest_id",
        ),
        (
            {"prob4d_claim_bearing_calibration_artifact_ids": {}},
            "calibration artifact IDs are missing",
        ),
        (
            {"prob4d_claim_bearing_runtime_revision_source": ""},
            "runtime_revision_source",
        ),
        (
            {
                "prob4d_claim_bearing_runtime_revision_independently_verified": (
                    False
                )
            },
            "not independently verified",
        ),
    ],
)
def test_one_call_rejects_bad_claim_bearing_metadata_before_solver(
    monkeypatch: pytest.MonkeyPatch,
    metadata_change: dict[str, object],
    message: str,
) -> None:
    observation = fixtures._observation()
    binding = fixtures._binding(observation)
    shared = binding.reparameterized_design(shared_bias_prior_std_m=0.02)
    original = fixtures._adapted_batch(
        observation,
        shared,
        np.zeros((2, 3, 0), dtype=np.float64),
    )
    metadata = dict(original.batch.metadata or {})
    metadata.update(metadata_change)
    adapted = SimpleNamespace(
        batch=replace(original.batch, metadata=metadata),
        observation_artifact_id=original.observation_artifact_id,
    )
    events: list[str] = []

    def build(*args: object, **kwargs: object) -> Any:
        events.append("build")
        return adapted

    def solve(*args: object, **kwargs: object) -> Any:
        events.append("solve")
        raise AssertionError("solver must not run")

    monkeypatch.setattr(
        update_module,
        "build_claim_bearing_gauge_aware_batch_from_artifacts",
        build,
    )
    monkeypatch.setattr(
        update_module,
        "update_prior_aware_gauge_belief",
        solve,
    )
    with pytest.raises(ValueError, match=message):
        update_claim_bearing_prob4d_with_visual_bias_from_artifacts(
            observation,
            SimpleNamespace(artifact_id=fixtures.LINEARIZATION_ID),
            visual_bias_nuisance=binding,
            physical_prediction_xyz_m=np.zeros((2, 3), dtype=np.float64),
        )
    assert events == ["build"]
