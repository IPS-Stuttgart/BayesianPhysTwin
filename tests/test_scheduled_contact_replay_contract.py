from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from bayesian_phystwin.causal4d_provider_v2 import (
    CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS,
    CAUSAL4D_PROVIDER_CAPABILITIES,
    SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION,
    ScheduledContactReplayProviderV1,
    ScheduledContactReplayRequestV1,
    ScheduledContactReplayResultV1,
    causal4d_provider_manifest,
    validate_scheduled_contact_replay_result,
)


def _request() -> ScheduledContactReplayRequestV1:
    paths = np.asarray(
        [
            [[0, 1, 1, 3], [0, 0, 2, 3]],
            [[0, 0, 1, 3], [0, 1, 1, 3]],
        ],
        dtype=np.int8,
    )
    indices = np.full((2, 2, 4, 2), -1, dtype=np.int64)
    weights = np.zeros(indices.shape, dtype=float)
    active = (paths == 1) | (paths == 2)
    for path_index, contact_index, frame_index in np.argwhere(active):
        indices[path_index, contact_index, frame_index] = (0, 1)
        weights[path_index, contact_index, frame_index] = (0.25, 0.75)
    return ScheduledContactReplayRequestV1(
        request_id="scheduled-001",
        schedule_identity="schedule-abc",
        simulator_configuration_id="sim-config-001",
        initial_state_id="twin-endpoint-001",
        contact_ids=("left", "right"),
        path_ids=("path-a", "path-b"),
        regime_paths=paths,
        prior_weights=np.asarray((0.6, 0.4)),
        retained_prior_mass=0.95,
        group_log_scales=np.asarray((0.1, -0.2)),
        controller_points_m=np.zeros((4, 2, 3)),
        position_m=np.zeros((3, 3)),
        velocity_mps=np.zeros((3, 3)),
        frame_times_s=np.asarray((0.0, 0.04, 0.08, 0.12)),
        contact_node_indices=indices,
        contact_node_weights=weights,
        normal_stiffness_npm=10.0,
        tangential_stiffness_npm=np.asarray((2.0, 3.0)),
        friction_coefficient=0.5,
    )


def _request_kwargs(
    request: ScheduledContactReplayRequestV1,
) -> dict[str, object]:
    return {field.name: getattr(request, field.name) for field in fields(request)}


def _result_kwargs(
    result: ScheduledContactReplayResultV1,
) -> dict[str, object]:
    return {field.name: getattr(result, field.name) for field in fields(result)}


def test_manifest_declares_contracts_without_claiming_backend_implementation() -> None:
    manifest = causal4d_provider_manifest(provider_revision="abc123")

    assert "scheduled_contact_replay_contracts" in CAUSAL4D_PROVIDER_CAPABILITIES
    assert "scheduled_contact_replay_contracts" in manifest["capabilities"]
    assert CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS[
        "ScheduledContactReplayRequest"
    ] == SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION
    assert manifest["artifact_schema_versions"][
        "ScheduledContactReplayResult"
    ] == SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION
    assert "scheduled_contact_replay" not in manifest["capabilities"]


def test_request_copies_freezes_and_content_addresses_inputs() -> None:
    request = _request()

    assert request.regime_paths.shape == (2, 2, 4)
    assert request.contact_node_indices.shape == (2, 2, 4, 2)
    assert request.normal_stiffness_npm.shape == (2, 2, 4)
    assert request.tangential_stiffness_npm.shape == (2, 2, 4)
    assert request.friction_coefficient.shape == (2, 2, 4)
    assert len(request.request_identity) == 64
    assert not request.regime_paths.flags.writeable
    assert not request.contact_node_weights.flags.writeable
    assert not request.position_m.flags.writeable
    with pytest.raises(ValueError):
        request.regime_paths[0, 0, 0] = 1


def test_active_contact_requires_a_normalized_finite_area_patch() -> None:
    request = _request()
    weights = np.asarray(request.contact_node_weights).copy()
    weights[0, 0, 1] = (0.2, 0.2)
    kwargs = _request_kwargs(request)
    kwargs["contact_node_weights"] = weights

    with pytest.raises(ValueError, match="normalized finite-area"):
        ScheduledContactReplayRequestV1(**kwargs)


def test_inactive_contact_cannot_apply_a_patch() -> None:
    request = _request()
    indices = np.asarray(request.contact_node_indices).copy()
    weights = np.asarray(request.contact_node_weights).copy()
    indices[0, 0, 0] = (0, 1)
    weights[0, 0, 0] = (0.5, 0.5)
    kwargs = _request_kwargs(request)
    kwargs["contact_node_indices"] = indices
    kwargs["contact_node_weights"] = weights

    with pytest.raises(ValueError, match="must not apply"):
        ScheduledContactReplayRequestV1(**kwargs)


def test_patch_indices_are_unique_and_inside_the_physical_state() -> None:
    request = _request()
    duplicate_indices = np.asarray(request.contact_node_indices).copy()
    duplicate_indices[0, 0, 1] = (0, 0)
    kwargs = _request_kwargs(request)
    kwargs["contact_node_indices"] = duplicate_indices

    with pytest.raises(ValueError, match="must not repeat"):
        ScheduledContactReplayRequestV1(**kwargs)

    outside_indices = np.asarray(request.contact_node_indices).copy()
    outside_indices[0, 0, 1] = (0, 3)
    kwargs["contact_node_indices"] = outside_indices
    with pytest.raises(ValueError, match="outside"):
        ScheduledContactReplayRequestV1(**kwargs)


def test_request_identity_changes_with_geometry_mechanics_and_timebase() -> None:
    request = _request()

    weights = np.asarray(request.contact_node_weights).copy()
    weights[0, 0, 1] = (0.5, 0.5)
    kwargs = _request_kwargs(request)
    kwargs["contact_node_weights"] = weights
    geometry_changed = ScheduledContactReplayRequestV1(**kwargs)

    kwargs = _request_kwargs(request)
    kwargs["friction_coefficient"] = 0.6
    mechanics_changed = ScheduledContactReplayRequestV1(**kwargs)

    kwargs = _request_kwargs(request)
    kwargs["frame_times_s"] = np.asarray((0.0, 0.05, 0.10, 0.15))
    timebase_changed = ScheduledContactReplayRequestV1(**kwargs)

    assert geometry_changed.request_identity != request.request_identity
    assert mechanics_changed.request_identity != request.request_identity
    assert timebase_changed.request_identity != request.request_identity


def test_result_binds_request_and_complete_continuous_trajectory_bank() -> None:
    request = _request()
    positions = np.zeros((2, 4, 3, 3))
    positions[1, :, :, 0] = 1.0
    result = ScheduledContactReplayResultV1.from_request(
        request,
        positions_m=positions,
        velocities_mps=np.zeros_like(positions),
        conditional_variance_m2=1e-6,
        provider_name="fake-phystwin",
        provider_version="0.4.0",
        provider_revision="abcdef",
    )

    assert validate_scheduled_contact_replay_result(request, result) is result
    assert result.replay_result_identity == result.result_identity
    assert len(result.result_identity) == 64
    assert not result.positions_m.flags.writeable
    assert not result.conditional_variance_m2.flags.writeable
    with pytest.raises(ValueError):
        result.positions_m[0, 0, 0, 0] = 1.0


def test_result_identity_changes_with_trajectory_variance_or_provider() -> None:
    request = _request()
    positions = np.zeros((2, 4, 3, 3))

    baseline = ScheduledContactReplayResultV1.from_request(
        request,
        positions_m=positions,
        velocities_mps=np.zeros_like(positions),
        conditional_variance_m2=1e-6,
        provider_name="fake-phystwin",
        provider_version="0.4.0",
        provider_revision="abcdef",
    )
    changed_positions = positions.copy()
    changed_positions[0, -1, 0, 0] = 1e-3
    trajectory_changed = ScheduledContactReplayResultV1.from_request(
        request,
        positions_m=changed_positions,
        velocities_mps=np.zeros_like(positions),
        conditional_variance_m2=1e-6,
        provider_name="fake-phystwin",
        provider_version="0.4.0",
        provider_revision="abcdef",
    )
    variance_changed = ScheduledContactReplayResultV1.from_request(
        request,
        positions_m=positions,
        velocities_mps=np.zeros_like(positions),
        conditional_variance_m2=2e-6,
        provider_name="fake-phystwin",
        provider_version="0.4.0",
        provider_revision="abcdef",
    )
    provider_changed = ScheduledContactReplayResultV1.from_request(
        request,
        positions_m=positions,
        velocities_mps=np.zeros_like(positions),
        conditional_variance_m2=1e-6,
        provider_name="fake-phystwin",
        provider_version="0.4.0",
        provider_revision="fedcba",
    )

    assert trajectory_changed.result_identity != baseline.result_identity
    assert variance_changed.result_identity != baseline.result_identity
    assert provider_changed.result_identity != baseline.result_identity


def test_validator_rejects_request_or_schedule_drift() -> None:
    request = _request()
    result = ScheduledContactReplayResultV1.from_request(
        request,
        positions_m=np.zeros((2, 4, 3, 3)),
        velocities_mps=np.zeros((2, 4, 3, 3)),
        conditional_variance_m2=1e-6,
        provider_name="fake-phystwin",
        provider_version="0.4.0",
        provider_revision="abcdef",
    )
    kwargs = _result_kwargs(result)
    kwargs["request_identity"] = "other-request"
    drifted = ScheduledContactReplayResultV1(**kwargs)

    with pytest.raises(ValueError, match="request identity"):
        validate_scheduled_contact_replay_result(request, drifted)


def test_runtime_provider_protocol_is_structural() -> None:
    class FakeProvider:
        simulator_configuration_id = "sim-config-001"
        provider_revision = "abcdef"

        def replay_scheduled_contacts(
            self,
            request: ScheduledContactReplayRequestV1,
        ) -> ScheduledContactReplayResultV1:
            return ScheduledContactReplayResultV1.from_request(
                request,
                positions_m=np.zeros((2, 4, 3, 3)),
                velocities_mps=np.zeros((2, 4, 3, 3)),
                conditional_variance_m2=1e-6,
                provider_name="fake-phystwin",
                provider_version="0.4.0",
                provider_revision=self.provider_revision,
            )

        def close(self) -> None:
            return None

    assert isinstance(FakeProvider(), ScheduledContactReplayProviderV1)
