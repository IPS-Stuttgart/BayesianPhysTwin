from __future__ import annotations

import numpy as np

from causal4d_public.deform360_rope_predict import (
    propagate_prefix_contact_state,
    select_visual_contact_patch,
)


def test_tactile_prefix_state_persists_until_visual_transition() -> None:
    visual = np.asarray(
        [
            [True, True],
            [True, True],
            [True, True],
            [False, True],
            [False, False],
        ]
    )

    conditioned = propagate_prefix_contact_state(visual, np.asarray([True, False]))

    assert conditioned.tolist() == [
        [True, False],
        [True, False],
        [True, False],
        [False, False],
        [False, False],
    ]


def test_visual_contact_patch_uses_nearest_taxels_and_node() -> None:
    centerline = np.column_stack((np.linspace(0.0, 0.2, 5), np.zeros(5), np.zeros(5)))
    taxels = np.asarray(
        [
            [-0.01, 0.002, 0.0],
            [-0.01, -0.002, 0.0],
            [0.18, 0.03, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )

    selected, patch, node, offset, diagnostics = select_visual_contact_patch(
        taxels, centerline, taxel_count=2
    )

    assert selected.tolist() == [0, 1]
    assert node == 0
    np.testing.assert_allclose(patch, [-0.01, 0.0, 0.0])
    np.testing.assert_allclose(offset, [0.01, 0.0, 0.0])
    assert diagnostics["patch_to_node_distance_m"] == 0.01
