from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.tree_block_gaussian import TreeBlockNormalSystemV1
from bayesian_phystwin.tree_separator_gaussian_parity import (
    evaluate_tree_separator_gaussian_parity,
    tree_block_normal_system_id,
)


def _system() -> TreeBlockNormalSystemV1:
    return TreeBlockNormalSystemV1(
        parent_indices=np.asarray([-1], dtype=np.int64),
        node_precision=np.asarray([[[2.0]]], dtype=np.float64),
        parent_coupling=np.zeros((1, 1, 1), dtype=np.float64),
        global_coupling=np.asarray([[[0.1]]], dtype=np.float64),
        global_precision=np.asarray([[3.0]], dtype=np.float64),
        node_right=np.asarray([[0.4]], dtype=np.float64),
        global_right=np.asarray([0.2], dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"normal_system_id": 7}, TypeError, "normal_system_id"),
        ({"normal_system_id": "not-a-digest"}, ValueError, "SHA-256"),
        ({"node_count": 0}, ValueError, "node_count"),
        ({"separator_size": -1}, ValueError, "separator_size"),
        ({"selected_node_indices": "0"}, TypeError, "sequence of integers"),
        ({"selected_node_indices": (True,)}, TypeError, "must be an integer"),
        ({"metrics": []}, TypeError, "metrics must be a mapping"),
        ({"metrics": {"unexpected": 0.0}}, ValueError, "metric fields"),
        (
            {"dense_precision_avoided_bytes": 1.0},
            TypeError,
            "must be an integer",
        ),
        (
            {"dense_precision_avoided_bytes": -1},
            ValueError,
            "must be non-negative",
        ),
        ({"passed": 1}, TypeError, "passed must be a bool"),
    ],
)
def test_report_rejects_malformed_fields(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    report = evaluate_tree_separator_gaussian_parity(
        _system(),
        maximum_condition_number=1.0e12,
    )
    with pytest.raises(error, match=message):
        replace(report, **overrides)  # type: ignore[arg-type]


def test_identity_and_evaluation_reject_wrong_types() -> None:
    with pytest.raises(TypeError, match="TreeBlockNormalSystemV1"):
        tree_block_normal_system_id(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TreeBlockNormalSystemV1"):
        evaluate_tree_separator_gaussian_parity(
            object(),  # type: ignore[arg-type]
            maximum_condition_number=1.0e12,
        )
