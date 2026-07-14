from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_replication_controls import (
    ContactTransitionEpisode,
    fit_causal_contact_transition,
)
from causal4d_public.deform360_replication_transition import (
    build_transition_episode_artifact,
    build_transition_fit_artifact,
    load_transition_episode_artifact,
    validate_transition_fit_artifact,
)


def _episode(name: str, onset: int) -> ContactTransitionEpisode:
    frames = 20
    openings = np.full((frames, 1), 0.08)
    openings[onset:] = 0.01
    controllers = np.zeros((frames, 1, 3))
    controllers[:, 0, 0] = np.linspace(0.1, 0.0, frames)
    objects = np.zeros((frames, 4, 3))
    active = np.zeros((frames, 1), dtype=bool)
    active[onset:] = True
    return ContactTransitionEpisode(
        episode_id=name,
        openings_m=openings,
        controller_positions_m=controllers,
        predicted_object_positions_m=objects,
        contact_active=active,
        dt_seconds=0.1,
    )


def _artifact(
    tmp_path: Path, name: str, split: str, onset: int
) -> dict:
    return build_transition_episode_artifact(
        _episode(name, onset),
        tmp_path / f"{name}.npz",
        object_id="fixture",
        split=split,
        pooled_fit_result_sha256="a" * 64,
        pooled_candidate_index=3,
        prefix_geometry_result_sha256="b" * 64,
        visual_contact_model={"threshold": 0.02},
    )


def test_transition_episode_archive_roundtrip_and_checksum(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "source-0", "source", 8)
    loaded = load_transition_episode_artifact(artifact)
    assert loaded.episode_id == "source-0"
    assert np.array_equal(loaded.contact_active, _episode("source-0", 8).contact_active)

    changed = copy.deepcopy(artifact)
    changed["pooled_candidate_index"] = 4
    with pytest.raises(ValueError, match="checksum"):
        load_transition_episode_artifact(changed)


def test_transition_fit_seals_source_and_calibration_boundaries(tmp_path: Path) -> None:
    source_artifacts = [
        _artifact(tmp_path, f"source-{index}", "source", 7 + index)
        for index in range(3)
    ]
    calibration_artifacts = [
        _artifact(tmp_path, "calibration-0", "calibration", 8)
    ]
    fit = fit_causal_contact_transition(
        [load_transition_episode_artifact(value) for value in source_artifacts],
        [load_transition_episode_artifact(value) for value in calibration_artifacts],
    )
    artifact = build_transition_fit_artifact(
        fit, source_artifacts, calibration_artifacts
    )
    model = validate_transition_fit_artifact(artifact)
    assert model.feature_names[0] == "gripper_openness_m"
    assert artifact["information_boundary"]["target_prefix_read"] is False
