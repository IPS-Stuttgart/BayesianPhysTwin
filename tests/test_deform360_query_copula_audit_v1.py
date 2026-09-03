from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/science/run_deform360_query_copula_audit_v1.py"
)


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "deform360_query_copula_audit_v1", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_controls_preserve_complete_empirical_marginals() -> None:
    module = _module()
    covariance = np.asarray(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.4, -0.1],
            [0.2, -0.1, 0.7],
        ],
        dtype=np.float64,
    )
    correlation, standard_deviations = module.covariance_to_correlation(covariance)
    full, _, _ = module.gaussian_copula_samples(
        correlation,
        standard_deviations,
        count=4096,
        seed=10,
        scores=module.normal_score_grid(4096),
    )
    independent = module.independent_copula(full, seed=11)
    scrambled = module.scrambled_copula(
        full,
        permutation=(1, 2, 0),
        signs=(1, -1, 1),
    )
    arms = {
        module.ARMS[0]: full,
        module.ARMS[1]: independent,
        module.ARMS[2]: scrambled,
    }
    assert module.marginal_parity_max_abs(full, arms) == 0.0
    for value in arms.values():
        assert np.array_equal(np.sort(value, axis=0), np.sort(full, axis=0))


def test_single_query_probabilities_match_but_joint_events_change() -> None:
    module = _module()
    correlation = np.asarray(
        [
            [1.0, 0.85, -0.4],
            [0.85, 1.0, -0.2],
            [-0.4, -0.2, 1.0],
        ],
        dtype=np.float64,
    )
    full, _, _ = module.gaussian_copula_samples(
        correlation,
        np.ones(3),
        count=8192,
        seed=20,
        scores=module.normal_score_grid(8192),
    )
    samples = {
        module.ARMS[0]: full,
        module.ARMS[1]: module.independent_copula(full, seed=21),
        module.ARMS[2]: module.scrambled_copula(
            full,
            permutation=(2, 0, 1),
            signs=(-1, 1, -1),
        ),
    }
    mean = np.asarray(
        [[-0.5, 0.0, 0.3], [0.0, 0.2, -0.2], [0.5, -0.4, 0.1]],
        dtype=np.float64,
    )
    truth = np.asarray(
        [[-0.2, 0.3, 0.4], [0.4, 0.1, -0.5], [0.8, -0.7, 0.2]],
        dtype=np.float64,
    )
    records, parity = module.evaluate_copula_arms(
        target_mean=mean,
        target_truth=truth,
        residual_samples=samples,
        thresholds=np.asarray([0.25, 0.35, 0.2]),
        query_events=("upper", "absolute", "upper"),
        query_names=("q0", "q1", "q2"),
        fallback_cost=0.1,
        probability_clip=1e-9,
    )
    assert parity == 0.0
    full_events = records[module.ARMS[0]]["events"]
    independent_events = records[module.ARMS[1]]["events"]
    assert any(full_events[name] != independent_events[name] for name in full_events)


def test_event_bank_exhausts_all_pairs_without_selection() -> None:
    module = _module()
    names = ("a", "b", "c", "d", "e")
    bank = module.composite_event_bank(names)
    assert len(bank) == 20
    assert {row["operator"] for row in bank} == {"and", "or"}
    pairs = {(row["left_query"], row["right_query"]) for row in bank}
    assert len(pairs) == 10


def test_normal_score_copula_tracks_requested_correlation() -> None:
    module = _module()
    correlation = np.asarray(
        [
            [1.0, 0.65, -0.15],
            [0.65, 1.0, 0.3],
            [-0.15, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    _, correlation_error, variance_error = module.gaussian_copula_samples(
        correlation,
        np.asarray([0.5, 1.2, 2.0]),
        count=16384,
        seed=30,
        scores=module.normal_score_grid(16384),
    )
    assert correlation_error < 0.02
    assert variance_error < 0.002
