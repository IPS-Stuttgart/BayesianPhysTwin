from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from bayesian_phystwin import causal4d_provider_v1 as provider


def test_provider_manifest_matches_causal4d_base_contract() -> None:
    manifest = provider.provider_manifest("0123456789abcdef")
    assert manifest.schema_version == 1
    assert set(provider.BASE_CAUSAL4D_PROVIDER_CAPABILITIES).issubset(
        manifest.capabilities
    )
    assert manifest.artifact_schema_versions == {
        "dynamic_discrepancy_correction": 1,
        "observation_belief": 1,
    }
    assert manifest.as_dict()["manifest_id"] == manifest.manifest_id


def test_manifest_content_address_changes_with_revision() -> None:
    first = provider.provider_manifest("first")
    duplicate = provider.provider_manifest("first")
    second = provider.provider_manifest("second")
    assert first.manifest_id == duplicate.manifest_id
    assert first.manifest_id != second.manifest_id


def test_replay_trajectory_is_validated_and_read_only() -> None:
    trajectory = provider.ReplayTrajectory(
        position_m=np.zeros((2, 3, 3)),
        velocity_mps=np.ones((2, 3, 3)),
        metadata={"source": "synthetic"},
    )
    assert trajectory.frame_count == 2
    assert trajectory.node_count == 3
    assert not trajectory.position_m.flags.writeable
    assert not trajectory.velocity_mps.flags.writeable
    with pytest.raises(ValueError, match="match position"):
        provider.ReplayTrajectory(
            position_m=np.zeros((2, 3, 3)),
            velocity_mps=np.zeros((1, 3, 3)),
        )


@dataclass
class _ReplayProvider:
    manifest: provider.PhysicalBeliefProviderManifest

    def replay_initial(self, *, frame_count: int) -> provider.ReplayTrajectory:
        return provider.ReplayTrajectory(
            np.zeros((frame_count, 1, 3)),
            np.zeros((frame_count, 1, 3)),
        )

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> provider.ReplayTrajectory:
        frame_count = stop_frame - start_frame
        return provider.ReplayTrajectory(
            np.repeat(np.asarray(position_m)[None], frame_count, axis=0),
            np.repeat(np.asarray(velocity_mps)[None], frame_count, axis=0),
        )


def test_replay_protocol_is_runtime_checkable() -> None:
    implementation = _ReplayProvider(provider.provider_manifest("test"))
    assert isinstance(implementation, provider.PhysTwinReplayProvider)
    replay = implementation.replay_initial(frame_count=3)
    assert replay.position_m.shape == (3, 1, 3)


def test_lazy_public_aliases_resolve_existing_primitives(tmp_path) -> None:
    assert callable(provider.target_validity)
    assert callable(provider.build_lift_map)
    assert callable(provider.initialize_simulator)
    path = tmp_path / "payload.bin"
    path.write_bytes(b"provider")
    assert provider.sha256_file(path) == (
        "5c4c1964340aca5b65393bbe9d3249cdd71be26665b3320ad694f034f2743283"
    )
