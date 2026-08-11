from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "science"
        / "run_full22_covariance_only_hybrid_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_full22_covariance_only_hybrid_v1",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def test_protocol_is_content_addressed_and_non_claim_bearing() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "full22_covariance_only_hybrid_v1.json"
    )
    protocol = MODULE.load_protocol(path)

    assert protocol["status"] == "retrospective-cross-fitted-development-only"
    assert protocol["hypothesis"]["reference_mean_candidate"] == "last_residual"
    assert protocol["hypothesis"]["point_prediction_change_allowed"] is False
    assert protocol["information_boundary"]["claim_authorized"] is False
    assert protocol["fresh_follow_up"]["deform360_v6_modified"] is False


def test_exact_reference_mean_returns_same_source_array() -> None:
    mean = np.zeros((3, 4, 3), dtype=np.float64)

    observed = MODULE._exact_reference_mean(mean, case_id="case")

    assert observed is mean


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((3, 4, 3), dtype=np.float64)[:, ::-1],
        np.full((3, 4, 3), np.nan, dtype=np.float64),
    ],
)
def test_exact_reference_mean_rejects_arrays_that_cannot_preserve_contract(
    value: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        MODULE._exact_reference_mean(value, case_id="case")


def test_scale_grid_reports_monotone_width_and_finite_scores() -> None:
    error = np.full((4, 2, 3), 0.4, dtype=np.float64)
    covariance = np.broadcast_to(
        np.eye(3, dtype=np.float64) * 0.01,
        (4, 2, 3, 3),
    ).copy()
    valid = np.ones((4, 2), dtype=bool)
    scales = (0.25, 1.0, 4.0, 16.0)

    nll, coverage, width = MODULE.score_scale_grid(
        error,
        covariance,
        valid,
        scales=scales,
        observation_std_m=0.005,
        eigenvalue_floor_m2=1e-12,
        marginal_coverage_z=1.6448536269514722,
    )

    assert np.all(np.isfinite(nll))
    assert np.all((0.0 <= coverage) & (coverage <= 1.0))
    assert np.all(np.diff(width) > 0.0)
    assert np.all(np.diff(nll) < 0.0)


def _selection_grid(case_count: int = 4) -> tuple[tuple[str, ...], np.ndarray]:
    cases = tuple(f"case-{index}" for index in range(case_count))
    grid = np.empty((case_count, 2, 3, 3), dtype=np.float64)
    # Independent prefers raw scale 1; dynamic is uniformly worse.
    grid[:, 0, :, 0] = 4.0
    grid[:, 0, :, 1] = 1.0
    grid[:, 0, :, 2] = 3.0
    grid[:, 1, :, 0] = 5.0
    grid[:, 1, :, 1] = 2.0
    grid[:, 1, :, 2] = 4.0
    return cases, grid


def test_crossfit_selection_excludes_the_held_case() -> None:
    cases, first_grid = _selection_grid()
    second_grid = first_grid.copy()
    second_grid[0, :, :, :] = np.asarray(
        [
            [[-100.0, -200.0, -300.0]] * 3,
            [[-400.0, -500.0, -600.0]] * 3,
        ]
    )

    first, _ = MODULE.crossfit_select(cases, first_grid, (0.5, 1.0, 2.0))
    second, _ = MODULE.crossfit_select(cases, second_grid, (0.5, 1.0, 2.0))

    assert first[0] == second[0]


def test_crossfit_ties_retain_raw_scale_and_independent_donor() -> None:
    cases = tuple(f"case-{index}" for index in range(4))
    grid = np.ones((4, 2, 3, 3), dtype=np.float64)

    folds, full = MODULE.crossfit_select(cases, grid, (0.5, 1.0, 2.0))

    assert all(fold.selected_donor == "independent_endpoint_v1" for fold in folds)
    assert all(fold.selected_scales == (1.0, 1.0, 1.0) for fold in folds)
    assert full["selected_donor"] == "independent_endpoint_v1"
    assert full["selected_scales"] == [1.0, 1.0, 1.0]


def test_effect_matrix_uses_each_fold_donor_and_scale() -> None:
    cases, grid = _selection_grid()
    folds, _ = MODULE.crossfit_select(cases, grid, (0.5, 1.0, 2.0))
    reference = np.full((4, 3), 3.0, dtype=np.float64)

    effects = MODULE._effect_matrices(
        reference,
        grid,
        (0.5, 1.0, 2.0),
        folds,
    )

    np.testing.assert_allclose(
        effects["crossfit_selected_scaled_covariance"],
        -2.0,
    )
    np.testing.assert_allclose(effects["independent_raw_covariance"], -2.0)
    np.testing.assert_allclose(effects["dynamic_raw_covariance"], -1.0)


def test_bootstrap_family_is_deterministic_and_detects_uniform_gain() -> None:
    matrix = -np.arange(1.0, 13.0, dtype=np.float64).reshape(4, 3)

    first = MODULE.bootstrap_family(
        {"hybrid": matrix},
        arm_order=("hybrid",),
        replicates=2000,
        seed=17,
        confidence=0.95,
    )
    second = MODULE.bootstrap_family(
        {"hybrid": matrix},
        arm_order=("hybrid",),
        replicates=2000,
        seed=17,
        confidence=0.95,
    )

    assert second == first
    overall = next(row for row in first if row["aggregation"] == "overall")
    assert overall["familywise_decision"] == "hybrid_better"
    assert overall["hybrid_better_case_count"] == 4
