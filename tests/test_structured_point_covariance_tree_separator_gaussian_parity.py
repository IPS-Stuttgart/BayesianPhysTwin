from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.tree_block_gaussian import (
    TreeBlockFactorizationV1,
    TreeBlockNormalSystemV1,
)
from bayesian_phystwin.tree_separator_gaussian import (
    TreeSeparatorGaussianSolutionV1,
    solve_tree_separator_gaussian,
)
from bayesian_phystwin.tree_separator_gaussian_parity import (
    TREE_SEPARATOR_GAUSSIAN_PARITY_SCHEMA,
    TreeSeparatorGaussianParityError,
    evaluate_tree_separator_gaussian_parity,
    require_tree_separator_gaussian_parity,
    tree_block_normal_system_id,
    tree_block_normal_system_to_tree_separator,
)


def _dense_precision(system: TreeBlockNormalSystemV1) -> np.ndarray:
    dense = np.zeros((system.dimension, system.dimension), dtype=np.float64)
    separator_size = system.global_size
    dense[:separator_size, :separator_size] = system.global_precision
    block_size = system.block_size
    for index, parent_value in enumerate(system.parent_indices):
        start = separator_size + index * block_size
        node_slice = slice(start, start + block_size)
        dense[node_slice, node_slice] = system.node_precision[index]
        dense[node_slice, :separator_size] = system.global_coupling[index]
        dense[:separator_size, node_slice] = system.global_coupling[index].T
        parent = int(parent_value)
        if parent >= 0:
            parent_start = separator_size + parent * block_size
            parent_slice = slice(parent_start, parent_start + block_size)
            dense[node_slice, parent_slice] = system.parent_coupling[index]
            dense[parent_slice, node_slice] = system.parent_coupling[index].T
    return dense


def _system(
    *,
    node_count: int = 7,
    block_size: int = 2,
    separator_size: int = 3,
    seed: int = 7,
) -> TreeBlockNormalSystemV1:
    rng = np.random.default_rng(seed)
    parents = np.full(node_count, -1, dtype=np.int64)
    for index in range(1, node_count):
        parents[index] = (index - 1) // 2
    parent_coupling = np.zeros(
        (node_count, block_size, block_size),
        dtype=np.float64,
    )
    for index in range(1, node_count):
        parent_coupling[index] = rng.normal(
            scale=0.05,
            size=(block_size, block_size),
        )
    global_coupling = rng.normal(
        scale=0.06,
        size=(node_count, block_size, separator_size),
    ).astype(np.float64)
    node_precision = np.empty(
        (node_count, block_size, block_size),
        dtype=np.float64,
    )
    for index in range(node_count):
        matrix = rng.normal(size=(block_size, block_size))
        node_precision[index] = matrix.T @ matrix + np.eye(block_size) * 3.0
    separator_precision = np.eye(separator_size, dtype=np.float64) * 5.0
    if separator_size:
        matrix = rng.normal(size=(separator_size, separator_size))
        separator_precision += matrix.T @ matrix
    provisional = TreeBlockNormalSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_coupling=parent_coupling,
        global_coupling=global_coupling,
        global_precision=separator_precision,
        node_right=np.zeros((node_count, block_size), dtype=np.float64),
        global_right=np.zeros(separator_size, dtype=np.float64),
    )
    minimum_eigenvalue = float(
        np.min(np.linalg.eigvalsh(_dense_precision(provisional)))
    )
    if minimum_eigenvalue <= 0.5:
        shift = 1.5 - minimum_eigenvalue
        node_precision += np.eye(block_size)[None, :, :] * shift
        separator_precision += np.eye(separator_size) * shift
    return TreeBlockNormalSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_coupling=parent_coupling,
        global_coupling=global_coupling,
        global_precision=separator_precision,
        node_right=rng.normal(size=(node_count, block_size)),
        global_right=rng.normal(size=separator_size),
    )


def _converted_dense_in_production_order(
    system: TreeBlockNormalSystemV1,
) -> tuple[np.ndarray, np.ndarray]:
    converted = tree_block_normal_system_to_tree_separator(system)
    dense, information = converted.to_dense(
        maximum_bytes=converted.estimated_dense_precision_bytes
    )
    node_dimension = system.node_count * system.block_size
    order = np.concatenate(
        (
            np.arange(node_dimension, node_dimension + system.global_size),
            np.arange(node_dimension),
        )
    )
    return dense[np.ix_(order, order)], information[order]


def test_conversion_preserves_dense_precision_and_information() -> None:
    system = _system()
    dense, information = _converted_dense_in_production_order(system)
    np.testing.assert_array_equal(dense, _dense_precision(system))
    np.testing.assert_array_equal(
        information,
        np.concatenate((system.global_right, system.node_right.reshape(-1))),
    )


def test_shadow_parity_passes_for_branched_system() -> None:
    report = require_tree_separator_gaussian_parity(
        _system(),
        maximum_condition_number=1.0e12,
    )
    assert report.passed
    assert report.descriptor()["schema"] == TREE_SEPARATOR_GAUSSIAN_PARITY_SCHEMA
    assert report.selected_node_indices == tuple(range(7))
    assert len(report.parity_id) == 64
    assert len(report.normal_system_id) == 64
    assert all(
        report.metrics[name] <= 1.0
        for name in (
            "mean_maximum_scaled_error",
            "separator_covariance_maximum_scaled_error",
            "node_covariance_maximum_scaled_error",
            "node_separator_cross_maximum_scaled_error",
            "log_determinant_scaled_error",
            "structured_residual_scaled_error",
        )
    )
    assert report.maximum_node_condition_number >= 1.0
    assert report.separator_condition_number >= 1.0
    assert report.to_dict()["parity_id"] == report.parity_id


def test_normal_system_identity_binds_precision_and_information() -> None:
    system = _system()
    same = _system()
    assert tree_block_normal_system_id(system) == tree_block_normal_system_id(same)

    changed_right = np.array(system.global_right, copy=True)
    changed_right[0] += 1.0e-9
    changed = replace(system, global_right=changed_right)
    assert tree_block_normal_system_id(system) != tree_block_normal_system_id(changed)

    changed_precision = np.array(system.node_precision, copy=True)
    changed_precision[0, 0, 0] += 1.0e-9
    changed = replace(system, node_precision=changed_precision)
    assert tree_block_normal_system_id(system) != tree_block_normal_system_id(changed)


def test_report_rejects_a_pass_flag_inconsistent_with_metrics() -> None:
    report = evaluate_tree_separator_gaussian_parity(
        _system(),
        maximum_condition_number=1.0e12,
    )
    metrics = dict(report.metrics)
    metrics["mean_maximum_scaled_error"] = 2.0
    with pytest.raises(ValueError, match="passed disagrees"):
        replace(report, metrics=metrics)


def test_zero_separator_system_has_exact_parity() -> None:
    report = require_tree_separator_gaussian_parity(
        _system(separator_size=0),
        maximum_condition_number=1.0e12,
    )
    assert report.passed
    assert report.separator_size == 0
    assert report.metrics["separator_covariance_maximum_absolute_error"] == 0.0
    assert report.metrics["node_separator_cross_maximum_absolute_error"] == 0.0


def test_large_tree_uses_bounded_selected_covariance_checks() -> None:
    system = _system(node_count=257, seed=19)
    report = require_tree_separator_gaussian_parity(
        system,
        maximum_condition_number=1.0e12,
    )
    assert report.passed
    assert len(report.selected_node_indices) == 8
    assert report.selected_node_indices[0] == 0
    assert report.selected_node_indices[-1] == 256
    assert report.dense_precision_avoided_bytes == (
        system.estimated_dense_precision_bytes
    )


def test_parity_does_not_materialize_complete_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("complete covariance materialization was attempted")

    monkeypatch.setattr(
        TreeBlockFactorizationV1,
        "materialize_covariance",
        forbidden,
    )
    assert require_tree_separator_gaussian_parity(
        _system(),
        maximum_condition_number=1.0e12,
        node_indices=(0, 3, 6),
    ).passed


def test_parity_detects_an_independent_solver_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesian_phystwin import tree_separator_gaussian_parity as parity

    original = solve_tree_separator_gaussian

    def shifted(system: object) -> TreeSeparatorGaussianSolutionV1:
        result = original(system)  # type: ignore[arg-type]
        node_mean = np.array(result.node_mean, copy=True)
        node_mean[0, 0] += 1.0e-4
        node_mean.setflags(write=False)
        return replace(result, node_mean=node_mean)

    monkeypatch.setattr(parity, "solve_tree_separator_gaussian", shifted)
    report = evaluate_tree_separator_gaussian_parity(
        _system(),
        maximum_condition_number=1.0e12,
    )
    assert not report.passed
    with pytest.raises(TreeSeparatorGaussianParityError, match="shadow parity"):
        require_tree_separator_gaussian_parity(
            _system(),
            maximum_condition_number=1.0e12,
        )


def test_production_condition_gate_remains_authoritative() -> None:
    system = TreeBlockNormalSystemV1(
        parent_indices=np.asarray([-1], dtype=np.int64),
        node_precision=np.asarray(
            [[[1.0e-15, 0.0], [0.0, 1.0]]],
            dtype=np.float64,
        ),
        parent_coupling=np.zeros((1, 2, 2), dtype=np.float64),
        global_coupling=np.zeros((1, 2, 1), dtype=np.float64),
        global_precision=np.asarray([[1.0]], dtype=np.float64),
        node_right=np.zeros((1, 2), dtype=np.float64),
        global_right=np.zeros(1, dtype=np.float64),
    )
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        evaluate_tree_separator_gaussian_parity(
            system,
            maximum_condition_number=1.0e12,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"maximum_condition_number": True}, TypeError),
        ({"maximum_condition_number": 0.0}, ValueError),
        ({"maximum_condition_number": 1.0e12, "node_indices": ()}, ValueError),
        (
            {"maximum_condition_number": 1.0e12, "node_indices": (0, 0)},
            ValueError,
        ),
        (
            {"maximum_condition_number": 1.0e12, "node_indices": (99,)},
            IndexError,
        ),
        (
            {"maximum_condition_number": 1.0e12, "relative_tolerance": -1.0},
            ValueError,
        ),
    ],
)
def test_parity_input_validation(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        evaluate_tree_separator_gaussian_parity(
            _system(),
            **kwargs,  # type: ignore[arg-type]
        )


def test_conversion_rejects_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="TreeBlockNormalSystemV1"):
        tree_block_normal_system_to_tree_separator(object())  # type: ignore[arg-type]
