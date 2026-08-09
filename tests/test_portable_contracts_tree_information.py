from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from bayesian_phystwin.tree_block_information import (
    TreeBlockInformationFactorV1,
)
from bayesian_phystwin.tree_schur_covariance import TreeSchurCovarianceV1


def _random_tree_factor_fixture() -> tuple[
    np.random.Generator,
    TreeBlockInformationFactorV1,
]:
    rng = np.random.default_rng(20260810)
    node_count = 8
    block_size = 3
    parents = np.asarray([-1, 0, 0, 1, 1, 2, 2, 5], dtype=np.int64)
    transitions = rng.normal(
        scale=0.15,
        size=(node_count, block_size, block_size),
    )
    transitions[0] = np.eye(block_size)
    scales = np.zeros_like(transitions)
    local_information = np.zeros_like(transitions)
    for index in range(node_count):
        scale = np.tril(rng.normal(scale=0.03, size=(block_size, block_size)))
        np.fill_diagonal(scale, rng.uniform(0.4, 0.9, size=block_size))
        scales[index] = scale
        design = rng.normal(size=(block_size + 1, block_size))
        local_information[index] = design.T @ design * 0.1
    factor = TreeBlockInformationFactorV1.from_transition_innovation(
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        local_information_blocks=local_information,
    )
    return rng, factor


def test_block_tree_information_matches_dense_solve_and_selected_inverse() -> None:
    rng, factor = _random_tree_factor_fixture()
    dense = factor.materialize()
    flat_right = rng.normal(size=factor.dimension)
    blocked_right = flat_right.reshape(factor.node_count, factor.block_size)
    matrix_right = rng.normal(size=(factor.dimension, 5))
    blocked_matrix_right = matrix_right.reshape(
        factor.node_count,
        factor.block_size,
        5,
    )

    np.testing.assert_allclose(
        factor.solve(flat_right),
        np.linalg.solve(dense, flat_right),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factor.solve(blocked_right).reshape(-1),
        np.linalg.solve(dense, flat_right),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factor.solve(matrix_right),
        np.linalg.solve(dense, matrix_right),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factor.solve(blocked_matrix_right).reshape(factor.dimension, 5),
        np.linalg.solve(dense, matrix_right),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factor.multiply(matrix_right),
        dense @ matrix_right,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        factor.multiply(blocked_matrix_right).reshape(factor.dimension, 5),
        dense @ matrix_right,
        atol=1e-12,
        rtol=1e-12,
    )
    assert factor.quadratic_form(flat_right) == pytest.approx(
        float(flat_right @ dense @ flat_right),
        abs=1e-11,
    )
    assert factor.quadratic_form(blocked_right) == pytest.approx(
        float(flat_right @ dense @ flat_right),
        abs=1e-11,
    )
    selected = np.asarray([0, 3, 7], dtype=np.int64)
    selected_coordinates = np.concatenate(
        [
            np.arange(
                index * factor.block_size,
                (index + 1) * factor.block_size,
            )
            for index in selected
        ]
    )
    dense_inverse = np.linalg.inv(dense)
    np.testing.assert_allclose(
        factor.marginal_covariance(selected),
        dense_inverse[np.ix_(selected_coordinates, selected_coordinates)],
        atol=1e-12,
        rtol=1e-12,
    )
    assert factor.log_determinant == pytest.approx(
        float(np.linalg.slogdet(dense)[1]),
        abs=1e-11,
    )
    assert factor.maximum_eliminated_block_condition >= 1.0
    assert factor.dense_information_materialized is False
    assert factor.stored_nbytes < factor.estimated_dense_bytes
    assert len(factor.result_id) == 64
    assert factor.descriptor()["dimension"] == factor.dimension
    with pytest.raises(MemoryError, match="exceeding"):
        factor.materialize(maximum_bytes=factor.estimated_dense_bytes - 1)
    np.testing.assert_allclose(
        factor.materialize(maximum_bytes=factor.estimated_dense_bytes),
        dense,
        atol=0.0,
        rtol=0.0,
    )


def test_block_tree_information_scales_linearly_without_dense_assembly() -> None:
    node_count = 2048
    block_size = 2
    parents = np.arange(node_count, dtype=np.int64) - 1
    transitions = np.repeat(
        (np.eye(block_size) * 0.95)[None],
        node_count,
        axis=0,
    )
    scales = np.repeat(
        (np.eye(block_size) * 0.2)[None],
        node_count,
        axis=0,
    )
    local_information = np.repeat(
        (np.eye(block_size) * 0.1)[None],
        node_count,
        axis=0,
    )
    factor = TreeBlockInformationFactorV1.from_transition_innovation(
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        local_information_blocks=local_information,
    )
    solution = factor.solve(np.ones((node_count, block_size)))

    assert solution.shape == (node_count, block_size)
    assert np.all(np.isfinite(solution))
    assert factor.estimated_dense_bytes >= 128 * 1024 * 1024
    assert factor.stored_nbytes < factor.estimated_dense_bytes // 100


@pytest.mark.parametrize(
    ("builder", "error", "match"),
    (
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1.0, 0.0]),
                diagonal_blocks=np.repeat(np.eye(2)[None], 2, axis=0),
                child_parent_blocks=np.zeros((2, 2, 2)),
            ),
            ValueError,
            "integer vector",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([], dtype=np.int64),
                diagonal_blocks=np.zeros((0, 2, 2)),
                child_parent_blocks=np.zeros((0, 2, 2)),
            ),
            ValueError,
            "at least one",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([0, 0], dtype=np.int64),
                diagonal_blocks=np.repeat(np.eye(2)[None], 2, axis=0),
                child_parent_blocks=np.zeros((2, 2, 2)),
            ),
            ValueError,
            "root",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1, 1], dtype=np.int64),
                diagonal_blocks=np.repeat(np.eye(2)[None], 2, axis=0),
                child_parent_blocks=np.zeros((2, 2, 2)),
            ),
            ValueError,
            "precede",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1], dtype=np.int64),
                diagonal_blocks=np.ones((1, 2, 3)),
                child_parent_blocks=np.zeros((1, 2, 3)),
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1], dtype=np.int64),
                diagonal_blocks=np.eye(2)[None],
                child_parent_blocks=np.zeros((1, 3, 3)),
            ),
            ValueError,
            "match",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1], dtype=np.int64),
                diagonal_blocks=np.asarray([[[1.0, np.nan], [np.nan, 1.0]]]),
                child_parent_blocks=np.zeros((1, 2, 2)),
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1], dtype=np.int64),
                diagonal_blocks=np.asarray([[[1.0, 0.2], [0.0, 1.0]]]),
                child_parent_blocks=np.zeros((1, 2, 2)),
            ),
            ValueError,
            "symmetric",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1], dtype=np.int64),
                diagonal_blocks=np.eye(2)[None],
                child_parent_blocks=np.ones((1, 2, 2)),
            ),
            ValueError,
            "root child-parent",
        ),
        (
            lambda: TreeBlockInformationFactorV1(
                parent_indices=np.asarray([-1], dtype=np.int64),
                diagonal_blocks=(-np.eye(2))[None],
                child_parent_blocks=np.zeros((1, 2, 2)),
            ),
            ValueError,
            "positive definite",
        ),
    ),
)
def test_block_tree_information_rejects_invalid_direct_contracts(
    builder: Callable[[], object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        builder()


def test_block_tree_information_rejects_invalid_builder_inputs() -> None:
    parents = np.asarray([-1, 0], dtype=np.int64)
    transitions = np.repeat(np.eye(2)[None], 2, axis=0)
    scales = np.repeat(np.eye(2)[None], 2, axis=0)
    local = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="integer vector"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents.astype(float),
            transition_matrices=transitions,
            innovation_scale_tril=scales,
        )
    with pytest.raises(ValueError, match="shape"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents,
            transition_matrices=np.ones((2, 2, 3)),
            innovation_scale_tril=np.ones((2, 2, 3)),
        )
    with pytest.raises(ValueError, match="match"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=np.ones((2, 3, 3)),
        )
    bad_scales = scales.copy()
    bad_scales[1, 0, 1] = 0.1
    with pytest.raises(ValueError, match="lower triangular"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=bad_scales,
        )
    bad_scales = scales.copy()
    bad_scales[1, 1, 1] = 0.0
    with pytest.raises(ValueError, match="positive diagonal"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=bad_scales,
        )
    with pytest.raises(ValueError, match="local_information_blocks"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=scales,
            local_information_blocks=np.zeros((1, 2, 2)),
        )
    nonsymmetric = local.copy()
    nonsymmetric[1, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_scale_tril=scales,
            local_information_blocks=nonsymmetric,
        )
    with pytest.raises(ValueError, match="precede"):
        TreeBlockInformationFactorV1.from_transition_innovation(
            parent_indices=np.asarray([-1, 1]),
            transition_matrices=transitions,
            innovation_scale_tril=scales,
        )


def _tree_schur_covariance_fixture() -> tuple[
    np.random.Generator,
    TreeSchurCovarianceV1,
    np.ndarray,
]:
    rng, factor = _random_tree_factor_fixture()
    retained = 2
    global_count = 3
    state_count = 4
    core_count = retained + global_count
    tree_core_cross = rng.normal(
        scale=0.05,
        size=(factor.dimension, core_count),
    )
    tree_core_solve = factor.solve(tree_core_cross)
    core_schur_factor = rng.normal(size=(core_count, core_count))
    core_schur = core_schur_factor.T @ core_schur_factor + np.eye(core_count)
    core_covariance = np.linalg.inv(core_schur)
    state_mapping = rng.normal(scale=0.1, size=(state_count, retained))
    residual = rng.normal(size=(state_count, state_count))
    state_prior = state_mapping @ state_mapping.T + residual @ residual.T * 0.05
    covariance = TreeSchurCovarianceV1(
        state_prior_covariance=state_prior,
        state_mapping=state_mapping,
        core_covariance=core_covariance,
        tree_factor=factor,
        tree_core_solve=tree_core_solve.reshape(
            factor.node_count,
            factor.block_size,
            core_count,
        ),
    )

    tree_information = factor.materialize()
    core_information = core_schur + tree_core_cross.T @ tree_core_solve
    reduced_information = np.block(
        [
            [core_information, tree_core_cross.T],
            [tree_core_cross, tree_information],
        ]
    )
    reduced_covariance = np.linalg.inv(reduced_information)
    reduced_core = reduced_covariance[:core_count, :core_count]
    reduced_core_tree = reduced_covariance[:core_count, core_count:]
    reduced_tree = reduced_covariance[core_count:, core_count:]
    expected = np.zeros((covariance.dimension, covariance.dimension))
    state_stop = state_count
    gauge_stop = state_stop + factor.dimension
    expected[:state_stop, :state_stop] = state_prior
    expected[:state_stop, :state_stop] += state_mapping @ (
        reduced_core[:retained, :retained] - np.eye(retained)
    ) @ state_mapping.T
    expected[:state_stop, state_stop:gauge_stop] = (
        state_mapping @ reduced_core_tree[:retained]
    )
    expected[state_stop:gauge_stop, :state_stop] = expected[
        :state_stop,
        state_stop:gauge_stop,
    ].T
    expected[:state_stop, gauge_stop:] = (
        state_mapping @ reduced_core[:retained, retained:]
    )
    expected[gauge_stop:, :state_stop] = expected[
        :state_stop,
        gauge_stop:,
    ].T
    expected[state_stop:gauge_stop, state_stop:gauge_stop] = reduced_tree
    expected[state_stop:gauge_stop, gauge_stop:] = reduced_covariance[
        core_count:,
        retained:core_count,
    ]
    expected[gauge_stop:, state_stop:gauge_stop] = expected[
        state_stop:gauge_stop,
        gauge_stop:,
    ].T
    expected[gauge_stop:, gauge_stop:] = reduced_core[
        retained:,
        retained:,
    ]
    return rng, covariance, expected


def test_tree_schur_covariance_matches_dense_apply_query_and_marginal() -> None:
    rng, covariance, expected = _tree_schur_covariance_fixture()
    vector = rng.normal(size=covariance.dimension)
    matrix = rng.normal(size=(covariance.dimension, 4))
    query = rng.normal(size=(5, covariance.dimension))
    selected = np.asarray([0, 3, 7, covariance.dimension - 1], dtype=np.int64)

    np.testing.assert_allclose(
        covariance.apply(vector),
        expected @ vector,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        covariance.apply(matrix),
        expected @ matrix,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        covariance.query_covariance(query),
        query @ expected @ query.T,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        covariance.marginal_covariance(selected),
        expected[np.ix_(selected, selected)],
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        covariance.materialize(),
        expected,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(np.asarray(covariance), expected, atol=2e-12)
    np.testing.assert_allclose(
        np.asarray(covariance, dtype=np.float32),
        expected.astype(np.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert covariance.dense_materialized is False
    assert covariance.state_count == 4
    assert covariance.retained_state_count == 2
    assert covariance.global_nuisance_count == 3
    assert covariance.gauge_parameter_count == covariance.tree_factor.dimension
    assert len(covariance.result_id) == 64
    assert covariance.descriptor()["representation"] == covariance.representation
    with pytest.raises(MemoryError, match="exceeding"):
        covariance.materialize(maximum_bytes=covariance.estimated_dense_bytes - 1)


def test_tree_schur_covariance_rejects_invalid_contracts() -> None:
    _rng, valid, _expected = _tree_schur_covariance_fixture()
    values = {
        "state_prior_covariance": valid.state_prior_covariance,
        "state_mapping": valid.state_mapping,
        "core_covariance": valid.core_covariance,
        "tree_factor": valid.tree_factor,
        "tree_core_solve": valid.tree_core_solve,
    }
    with pytest.raises(TypeError, match="tree_factor"):
        TreeSchurCovarianceV1(**{**values, "tree_factor": object()})
    with pytest.raises(ValueError, match="two dimensions"):
        TreeSchurCovarianceV1(
            **{**values, "state_prior_covariance": np.ones(3)}
        )
    with pytest.raises(ValueError, match="symmetric"):
        bad_prior = valid.state_prior_covariance.copy()
        bad_prior[0, 1] += 1.0
        TreeSchurCovarianceV1(
            **{**values, "state_prior_covariance": bad_prior}
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        TreeSchurCovarianceV1(
            **{**values, "core_covariance": -np.eye(len(valid.core_covariance))}
        )
    with pytest.raises(ValueError, match="one row"):
        TreeSchurCovarianceV1(
            **{**values, "state_mapping": np.zeros((1, 1))}
        )
    with pytest.raises(ValueError, match="smaller"):
        TreeSchurCovarianceV1(
            **{
                **values,
                "state_mapping": np.zeros((valid.state_count, 3)),
                "core_covariance": np.eye(2),
            }
        )
    with pytest.raises(ValueError, match="shape"):
        TreeSchurCovarianceV1(
            **{**values, "tree_core_solve": np.zeros((1, 1, 1))}
        )
    with pytest.raises(ValueError, match="finite"):
        bad_tree_core = valid.tree_core_solve.copy()
        bad_tree_core[0, 0, 0] = np.nan
        TreeSchurCovarianceV1(
            **{**values, "tree_core_solve": bad_tree_core}
        )
    with pytest.raises(ValueError, match="exceeds"):
        TreeSchurCovarianceV1(
            **{
                **values,
                "state_prior_covariance": np.eye(valid.state_count) * 1e-6,
                "state_mapping": np.ones_like(valid.state_mapping),
            }
        )
    with pytest.raises(ValueError, match="dimension"):
        valid.apply(np.zeros(valid.dimension - 1))
    with pytest.raises(ValueError, match="shape"):
        valid.query_covariance(np.zeros((2, valid.dimension - 1)))
    with pytest.raises(ValueError, match="integer vector"):
        valid.marginal_covariance(np.asarray([0.0]))
    with pytest.raises(ValueError, match="nonempty"):
        valid.marginal_covariance(np.asarray([], dtype=np.int64))
    with pytest.raises(ValueError, match="unique"):
        valid.marginal_covariance(np.asarray([0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="unknown"):
        valid.marginal_covariance(np.asarray([valid.dimension], dtype=np.int64))
    with pytest.raises(ValueError, match="nonnegative"):
        valid.materialize(maximum_bytes=-1)
