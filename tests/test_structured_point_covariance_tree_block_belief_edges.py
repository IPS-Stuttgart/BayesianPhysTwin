from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.tree_block_sparse_gauge_belief as belief
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
)


def _two_node_gauge() -> TreeSparseGaugeDesignV1:
    return TreeSparseGaugeDesignV1(
        local_gauge_jacobian=np.ones((1, 3, 1), dtype=np.float64),
        gauge_indices=np.asarray([0], dtype=np.int64),
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        transition_matrices=np.asarray(
            [[[0.0]], [[0.5]]],
            dtype=np.float64,
        ),
        innovation_scale_tril=np.asarray(
            [[[1.0]], [[1.0]]],
            dtype=np.float64,
        ),
        gauge_ids=("root", "unobserved-child"),
        prior_id="f" * 64,
    )


def test_build_tree_system_covers_all_optional_blocks() -> None:
    gauge = _two_node_gauge()
    layout = belief._GlobalLayout(  # noqa: SLF001
        state_count=1,
        shared_count=1,
        view_count=1,
        anchor_bias_count=1,
    )
    batch = SimpleNamespace(
        anchor_bias_prior_covariance=np.asarray([[2.0]], dtype=np.float64)
    )
    design = np.ones((1, 3, 1), dtype=np.float64)

    system = belief._build_tree_system(  # noqa: SLF001
        batch=batch,  # type: ignore[arg-type]
        gauge=gauge,
        layout=layout,
        observation_weight=np.ones(1, dtype=np.float64),
        state=design,
        local_gauge=design,
        shared=design,
        view=design,
        target=np.ones((1, 3), dtype=np.float64),
        anchor_weight=np.ones(1, dtype=np.float64),
        anchor_state=design,
        anchor_bias=design,
        anchor_target=np.ones((1, 3), dtype=np.float64),
        config=PriorAwareGaugeConfigV1(),
        state_prior_precision=np.eye(1, dtype=np.float64),
    )

    assert system.global_size == 4
    assert system.node_count == 2
    np.testing.assert_allclose(
        system.global_precision,
        system.global_precision.T,
    )
    assert np.count_nonzero(system.global_coupling[1]) == 0


def test_bias_prior_rejects_missing_anchor_covariance() -> None:
    layout = belief._GlobalLayout(  # noqa: SLF001
        state_count=1,
        shared_count=0,
        view_count=0,
        anchor_bias_count=1,
    )
    batch = SimpleNamespace(anchor_bias_prior_covariance=None)

    with pytest.raises(
        ValueError,
        match="anchor bias prior covariance is missing",
    ):
        belief._bias_prior_precision(  # noqa: SLF001
            batch,  # type: ignore[arg-type]
            layout,
            PriorAwareGaugeConfigV1(),
        )


def test_materialization_budget_accepts_exact_limit() -> None:
    belief._materialization_budget(16, required_bytes=16)  # noqa: SLF001
