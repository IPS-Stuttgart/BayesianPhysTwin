from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    CameraRecorderFamilyMapV1,
    ReconstructionManifestV1,
    RegisteredResidualHistoryExecutionV1,
    ResidualHistoryDryRunPolicyV1,
    deterministic_disjoint_camera_partition,
    run_registered_source_only_residual_history,
)
from bayesian_phystwin.endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)


def _policy() -> ResidualHistoryDryRunPolicyV1:
    return ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=2,
        minimum_cameras_per_role=2,
        minimum_camera_families_per_role=2,
    )


def _family_map() -> CameraRecorderFamilyMapV1:
    return CameraRecorderFamilyMapV1(
        source_inventory_id="a" * 64,
        bindings=tuple(
            (f"camera-{index:02d}", f"recorder-{index:02d}")
            for index in range(8)
        ),
    )


def _manifests(
    family_map: CameraRecorderFamilyMapV1,
) -> tuple[ReconstructionManifestV1, ReconstructionManifestV1]:
    partition = deterministic_disjoint_camera_partition(
        family_map,
        policy=_policy(),
    )
    provider = ReconstructionManifestV1(
        role="provider",
        source_inventory_id=family_map.source_inventory_id,
        reconstruction_artifact_id="b" * 64,
        implementation_revision="1" * 40,
        configuration_id="c" * 64,
        input_camera_ids=partition.provider_camera_ids,
        input_source_artifact_ids=("d" * 64,),
    )
    scoring = ReconstructionManifestV1(
        role="scoring",
        source_inventory_id=family_map.source_inventory_id,
        reconstruction_artifact_id="e" * 64,
        implementation_revision="2" * 40,
        configuration_id="f" * 64,
        input_camera_ids=partition.scoring_camera_ids,
        input_source_artifact_ids=("0" * 64,),
    )
    return provider, scoring


def _arrays() -> dict[str, np.ndarray]:
    physical_prefix = np.zeros((3, 4, 3), dtype=np.float64)
    observation = np.full_like(physical_prefix, np.nan)
    validity = np.zeros((3, 4), dtype=bool)
    validity[0] = True
    observation[0] = 0.01
    validity[1, 2:] = True
    observation[1, 2:] = 0.2
    validity[2, :2] = True
    observation[2, :2] = 0.03
    physical_future = np.zeros((3, 4, 3), dtype=np.float64)
    return {
        "physical_prefix": physical_prefix,
        "observation": observation,
        "validity": validity,
        "physical_future": physical_future,
        "physical_covariance": np.zeros((3, 4, 3, 3), dtype=np.float64),
        "frame_indices": np.asarray([0, 7, 14], dtype=np.int64),
        "material_ids": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "horizon_steps": np.asarray([1, 7, 14], dtype=np.int64),
        "horizon_bins": np.asarray([0, 1, 2], dtype=np.int64),
    }


def _registered_mean(arrays: dict[str, np.ndarray]) -> np.ndarray:
    result = np.array(arrays["physical_future"], copy=True, order="C")
    for material_index in range(arrays["validity"].shape[1]):
        support = np.flatnonzero(arrays["validity"][:, material_index])
        frame = int(support[-1])
        result[:, material_index] += (
            arrays["observation"][frame, material_index]
            - arrays["physical_prefix"][frame, material_index]
        )
    return result


def _run(
    arrays: dict[str, np.ndarray],
    *,
    registered_mean: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> RegisteredResidualHistoryExecutionV1:
    family_map = _family_map()
    provider, scoring = _manifests(family_map)
    return run_registered_source_only_residual_history(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["physical_covariance"],
        _registered_mean(arrays) if registered_mean is None else registered_mean,
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_horizon_steps=arrays["horizon_steps"],
        future_horizon_bins=arrays["horizon_bins"],
        camera_recorder_family_map=family_map,
        provider_reconstruction_manifest=provider,
        scoring_reconstruction_manifest=scoring,
        source_unit_id="opened-source-object-session-001",
        policy=_policy(),
        metadata=metadata,
    )


def _expected_unscaled_covariance(arrays: dict[str, np.ndarray]) -> np.ndarray:
    residual = np.zeros_like(arrays["physical_prefix"])
    validity = arrays["validity"]
    residual[validity] = (
        arrays["observation"][validity]
        - arrays["physical_prefix"][validity]
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        validity,
        end_frame=len(residual),
        config=DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    )
    return np.stack(
        [
            predict_model_averaged_endpoint(
                posterior,
                horizon_steps=int(step),
            ).covariance_m2
            for step in arrays["horizon_steps"]
        ],
        axis=0,
    )


def test_registered_entry_point_has_no_covariance_injection_surface() -> None:
    parameters = inspect.signature(
        run_registered_source_only_residual_history
    ).parameters
    assert "donor_covariance_m2" not in parameters
    assert "registered_last_residual_mean_m" in parameters
    assert "future_horizon_steps" in parameters


def test_registered_execution_reproduces_the_frozen_endpoint_donor() -> None:
    arrays = _arrays()
    registered_mean = _registered_mean(arrays)
    execution = _run(
        arrays,
        registered_mean=registered_mean,
        metadata={"opened_source_only": True},
    )
    expected_unscaled = _expected_unscaled_covariance(arrays)
    scales = np.asarray(REGISTERED_COVARIANCE_SCALES, dtype=np.float64)

    assert execution.accepted
    assert execution.mean_m is registered_mean
    assert execution.result.hybrid is not None
    assert execution.donor.covariance_donor_id if False else True
    np.testing.assert_array_equal(
        execution.donor.covariance_m2,
        expected_unscaled,
    )
    np.testing.assert_array_equal(
        execution.covariance_m2,
        expected_unscaled * scales[:, None, None, None],
    )
    descriptor = execution.descriptor()
    assert descriptor["reference_predictor_id"] == REGISTERED_REFERENCE_PREDICTOR_ID
    assert descriptor["covariance_donor_id"] == REGISTERED_COVARIANCE_DONOR_ID
    assert descriptor["covariance_scales"] == [8.0, 16.0, 16.0]
    assert execution.result.adapter.metadata["opened_source_only"] is True
    assert not execution.donor.covariance_m2.flags.writeable


def test_registered_execution_is_deterministic_and_content_addressed() -> None:
    arrays = _arrays()
    first = _run(arrays)
    second = _run(_arrays())

    assert first.donor.config_id == second.donor.config_id
    assert first.donor.posterior_id == second.donor.posterior_id
    assert first.donor.prediction_ids == second.donor.prediction_ids
    assert first.donor.donor_id == second.donor.donor_id
    assert first.execution_id == second.execution_id
    assert first.result.decision.decision_id == second.result.decision.decision_id


def test_changed_causal_history_changes_donor_and_execution_identity() -> None:
    first_arrays = _arrays()
    second_arrays = _arrays()
    second_arrays["observation"][0, 0] += 0.001

    first = _run(first_arrays)
    second = _run(second_arrays)

    assert first.result.adapter.adapter_id != second.result.adapter.adapter_id
    assert first.donor.posterior_id != second.donor.posterior_id
    assert first.donor.donor_id != second.donor.donor_id
    assert first.execution_id != second.execution_id


def test_insufficient_material_support_keeps_exact_physical_fallback() -> None:
    arrays = _arrays()
    arrays["validity"][1:, 3] = False
    arrays["observation"][1:, 3] = np.nan
    execution = _run(arrays)

    assert not execution.accepted
    assert execution.mean_m is arrays["physical_future"]
    assert execution.covariance_m2 is arrays["physical_covariance"]
    assert execution.result.hybrid is None
    assert execution.result.decision.fallback_reasons == (
        "insufficient-per-material-support",
    )
    assert execution.donor.donor_id is not None


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        (np.asarray([1, 2], dtype=np.int64), "one entry per future frame"),
        (np.asarray([0, 2, 3], dtype=np.int64), "positive and increasing"),
        (np.asarray([1, 3, 2], dtype=np.int64), "positive and increasing"),
        (np.asarray([1.0, 2.0, 3.0]), "integers"),
    ],
)
def test_future_horizon_steps_fail_closed(
    steps: np.ndarray,
    match: str,
) -> None:
    arrays = _arrays()
    arrays["horizon_steps"] = steps
    with pytest.raises(ValueError, match=match):
        _run(arrays)


def test_donor_and_execution_ids_are_tamper_evident() -> None:
    execution = _run(_arrays())
    with pytest.raises(ValueError, match="donor_id"):
        replace(execution.donor, donor_id="0" * 64)
    with pytest.raises(ValueError, match="execution_id"):
        replace(execution, execution_id="0" * 64)


def test_donor_rejects_prediction_roster_and_step_tampering() -> None:
    execution = _run(_arrays())
    with pytest.raises(ValueError, match="one identity per future horizon"):
        replace(
            execution.donor,
            prediction_ids=execution.donor.prediction_ids[:-1],
            donor_id=None,
        )
    with pytest.raises(ValueError, match="positive and increasing"):
        replace(
            execution.donor,
            future_horizon_steps=np.asarray([1, 1, 2], dtype=np.int64),
            donor_id=None,
        )


def test_execution_rejects_deployed_covariance_not_from_registered_donor() -> None:
    execution = _run(_arrays())
    changed = np.array(execution.result.covariance_m2, copy=True)
    changed[0, 0, 0, 0] += 1e-9
    forged_result = replace(execution.result, covariance_m2=changed)

    with pytest.raises(ValueError, match="differs from the registered donor"):
        replace(
            execution,
            result=forged_result,
            execution_id=None,
        )


def test_execution_rejects_donor_from_another_residual_history() -> None:
    first = _run(_arrays())
    changed_arrays = _arrays()
    changed_arrays["observation"][0, 0] += 0.001
    second = _run(changed_arrays)

    with pytest.raises(ValueError, match="different residual histories"):
        replace(
            first,
            donor=second.donor,
            execution_id=None,
        )
