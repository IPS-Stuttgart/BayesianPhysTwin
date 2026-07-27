from __future__ import annotations

import numpy as np

from bayesian_phystwin.causal4d_graph_provider_v1 import (
    controller_hand_count,
    infer_controller_groups,
)
from bayesian_phystwin.phystwin_controller_sensitivity import (
    controller_hand_count as legacy_controller_hand_count,
    infer_controller_groups as legacy_infer_controller_groups,
)


def test_controller_provider_matches_existing_semantics() -> None:
    cases = (
        "single_lift_sloth",
        "double_stretch_sloth",
        "rope_double_hand",
        "single_push_rope",
    )
    for case_name in cases:
        assert controller_hand_count(case_name) == legacy_controller_hand_count(
            case_name
        )

    point_sets = (
        np.asarray(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 0.1, 0.0],
                [1.0, -0.1, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.asarray(
            [
                [0.0, -2.0, 0.0],
                [0.0, -1.0, 0.1],
                [0.0, 1.0, -0.1],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    for points in point_sets:
        for group_count in (1, 2):
            np.testing.assert_array_equal(
                infer_controller_groups(points, group_count=group_count),
                legacy_infer_controller_groups(points, group_count=group_count),
            )
