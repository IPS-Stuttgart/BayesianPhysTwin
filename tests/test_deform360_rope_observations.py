from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_rope_observations import (
    RopeSourceObservationConfig,
    rope_source_observation_artifact_sha256,
    select_contact_taxels,
    validate_source_rope_observation_artifact,
)


def test_contact_taxel_selection_uses_activity_then_strength() -> None:
    left = np.zeros((5, 16, 32), dtype=np.float32)
    right = np.zeros_like(left)
    active = np.asarray([False, True, True, True, False])
    left[1:4, 2, 3] = 0.2
    right[1:4, 4, 5] = 0.8
    left[1:3, 6, 7] = 2.0

    selected, diagnostics = select_contact_taxels(
        left,
        right,
        active,
        selected_taxel_count=2,
        minimum_selected_taxel_count=2,
        threshold=0.0,
    )

    assert selected.tolist() == [2 * (4 * 32 + 5) + 1, 2 * (2 * 32 + 3)]
    assert diagnostics["active_frame_count_by_selected_taxel"] == [3, 3]


def test_observation_config_rejects_too_many_taxels() -> None:
    with pytest.raises(ValueError, match="selected-taxel"):
        RopeSourceObservationConfig(selected_taxel_count=769)


def test_source_observation_artifact_verifies_archive(tmp_path: Path) -> None:
    archive = tmp_path / "observation.npz"
    positions = np.zeros((4, 7, 3), dtype=np.float64)
    controllers = np.zeros((4, 1, 3), dtype=np.float64)
    active = np.asarray([[False], [True], [True], [False]])
    offsets = np.zeros((1, 3), dtype=np.float64)
    np.savez_compressed(
        archive,
        frame_indices=np.asarray((0, 2, 4, 6), dtype=np.int32),
        positions_m=positions,
        controller_positions_m=controllers,
        contact_active=active,
        contact_node_indices=np.asarray((6,), dtype=np.int32),
        contact_offsets_m=offsets,
    )
    import causal4d_public.deform360_rope_observations as module

    payload = {
        "schema_version": 3,
        "artifact_kind": "Deform360SourceRopeDynamicsObservation",
        "episode_id": "001-rope/episode_0000",
        "split": "source",
        "quality": {"passed": True},
        "archive": {
            "path": str(archive),
            "sha256": module._sha256_file(archive),
            "positions_sha256": module._sha256_array(positions),
            "controller_positions_sha256": module._sha256_array(controllers),
            "contact_active_sha256": module._sha256_array(active),
            "contact_offsets_sha256": module._sha256_array(offsets),
        },
        "information_boundary": {"target_files_read": False},
    }
    payload["result_sha256"] = rope_source_observation_artifact_sha256(payload)

    assert validate_source_rope_observation_artifact(payload)["passed"]
    payload["quality"]["passed"] = False
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_rope_observation_artifact(payload, verify_archive=False)
