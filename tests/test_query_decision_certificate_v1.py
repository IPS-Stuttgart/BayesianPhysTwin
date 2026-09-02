from __future__ import annotations

import numpy as np

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)


def test_query_decision_certificate_smoke() -> None:
    result = query_decision_certificate(
        prior_weights=np.array([0.25, 0.25, 0.25, 0.25]),
        quotient_weights=np.array([0.5, 0.5]),
        class_index=np.array([0, 0, 1, 1]),
        losses=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.1, 1.1, 2.1],
                [0.2, 1.2, 2.2],
                [0.3, 1.3, 2.3],
            ]
        ),
        regret_tolerance=0.05,
    )

    assert result.minimax_action_index == 0
    assert result.minimax_worst_case_regret == 0.0
    assert result.has_robustly_optimal_action
