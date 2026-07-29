from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_query import (
    CausalResponseQueryConfig,
    build_causal_response_query_schedule,
)


def _inputs() -> tuple[np.ndarray, ...]:
    node_count = 24
    camera_count = 12
    height = width = 96
    frame_zero = np.column_stack(
        (
            np.linspace(-0.18, 0.18, node_count),
            0.04 * np.sin(np.linspace(0.0, 2.0 * np.pi, node_count)),
            np.full(node_count, 2.0),
        )
    )
    graph_basis = np.zeros((node_count, 3, 16), dtype=np.float64)
    coordinate = np.linspace(-1.0, 1.0, node_count)
    for mode in range(16):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    action_support = np.linspace(0.05, 1.0, node_count)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, camera_count, endpoint=False)
    poses[:, 0, 3] = 0.08 * np.cos(angles)
    poses[:, 1, 3] = 0.08 * np.sin(angles)
    depth = np.full((camera_count, height, width), 2.0)
    masks = np.ones_like(depth, dtype=bool)
    camera_ids = np.asarray([f"camera-{index:02d}" for index in range(camera_count)])
    proposal = np.arange(0, camera_count, 2)
    validation = np.arange(1, camera_count, 2)
    return (
        frame_zero,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depth,
        masks,
        camera_ids,
        proposal,
        validation,
    )


def _build(
    inputs: tuple[np.ndarray, ...],
    *,
    config: CausalResponseQueryConfig | None = None,
):
    (
        frame_zero,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depth,
        masks,
        camera_ids,
        proposal,
        validation,
    ) = inputs
    return build_causal_response_query_schedule(
        frame_zero,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depth,
        masks,
        camera_ids=tuple(camera_ids),
        proposal_camera_indices=proposal,
        validation_camera_indices=validation,
        config=config
        or CausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def test_query_schedule_is_deterministic_and_supported_in_both_panels() -> None:
    inputs = _inputs()

    first = _build(inputs)
    second = _build(inputs)

    assert first.admitted
    assert len(first.entity_ids) == 8
    assert first.artifact_sha256 == second.artifact_sha256
    np.testing.assert_array_equal(first.entity_ids, second.entity_ids)
    assert np.all(first.selected_action_support >= 0.1)
    support = first.association_valid & (
        first.association_probability >= first.config.association_support_probability
    )
    assert np.all(np.sum(support[first.proposal_camera_indices], axis=0) >= 3)
    assert np.all(np.sum(support[first.validation_camera_indices], axis=0) >= 3)


def test_query_schedule_abstains_when_one_panel_has_no_depth_support() -> None:
    inputs = list(_inputs())
    masks = inputs[6].copy()
    validation = inputs[9]
    masks[validation] = False
    inputs[6] = masks

    schedule = _build(tuple(inputs))

    assert not schedule.admitted
    assert not len(schedule.entity_ids)
    assert schedule.eligible_entity_count == 0


def test_query_schedule_never_selects_below_action_support_threshold() -> None:
    inputs = list(_inputs())
    action_support = inputs[2].copy()
    action_support[:20] = 0.0
    inputs[2] = action_support

    schedule = _build(
        tuple(inputs),
        config=CausalResponseQueryConfig(
            query_count=4,
            graph_basis_rank=8,
        ),
    )

    assert schedule.admitted
    assert np.all(schedule.entity_ids >= 20)
    assert np.all(schedule.selected_action_support >= 0.1)


def test_query_schedule_digest_binds_frame_zero_camera_evidence() -> None:
    inputs = list(_inputs())
    first = _build(tuple(inputs))
    depth = inputs[5].copy()
    depth[0, 0, 0] += 1e-4
    inputs[5] = depth

    changed = _build(tuple(inputs))

    assert changed.artifact_sha256 != first.artifact_sha256


def test_query_schedule_rejects_overlapping_camera_panels() -> None:
    inputs = list(_inputs())
    inputs[9] = inputs[8].copy()

    with pytest.raises(ValueError, match="disjoint"):
        _build(tuple(inputs))
