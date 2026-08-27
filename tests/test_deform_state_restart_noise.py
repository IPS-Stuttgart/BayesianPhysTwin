"""Noise follow-up preserves correlated measurement and experimental units."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_state_restart import RestartConfig


@pytest.fixture
def noise_runner(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts/remote"))
    spec = importlib.util.spec_from_file_location(
        "noise_runner", root / "scripts/remote/run_deform_state_restart_noise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_noise_is_one_translation_per_trajectory(noise_runner):
    independent, biased = noise_runner.observation_noise(
        (14, 2, 4, 3),
        seed=260830,
        independent_std_m=0.001,
        shared_std_m=0.005,
    )
    difference = biased - independent
    np.testing.assert_allclose(
        difference, np.broadcast_to(difference[:, :1, :1], difference.shape), atol=1e-17
    )
    assert not np.array_equal(difference[0], difference[1])
    again = noise_runner.observation_noise(
        (14, 2, 4, 3), seed=260830, independent_std_m=0.001, shared_std_m=0.005
    )
    np.testing.assert_array_equal(independent, again[0])
    np.testing.assert_array_equal(biased, again[1])


@pytest.mark.parametrize(
    "shape,independent,shared",
    [((2, 4, 3), 0.001, 0.005), ((1, 2, 4, 3), -1, 0), ((1, 2, 4, 3), 0, float("nan"))],
)
def test_noise_rejects_invalid_contract(noise_runner, shape, independent, shared):
    with pytest.raises(ValueError):
        noise_runner.observation_noise(
            shape, seed=1, independent_std_m=independent, shared_std_m=shared
        )


def test_replication_does_not_create_extra_statistical_units(noise_runner):
    config = dataclasses.replace(RestartConfig(), bootstrap_replicates=100)
    truth = np.zeros((3, 120, 12, 3))
    base = np.ones_like(truth) * 0.01
    candidate = base.copy()
    candidate[1] *= 0.5
    candidate[2] *= 0.8
    names = ["103.pkl", "a", "b"]
    one = {"incumbent": base[None], "candidate": candidate[None]}
    many = {arm: np.repeat(value, 16, axis=0) for arm, value in one.items()}
    a = noise_runner.summarize_noise(one, truth, names, config)["summaries"][
        "candidate"
    ]
    b = noise_runner.summarize_noise(many, truth, names, config)["summaries"][
        "candidate"
    ]
    assert a["case_count"] == b["case_count"] == 2
    for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
        np.testing.assert_allclose(a[metric + "_delta_ci95"], b[metric + "_delta_ci95"])


def test_noise_summary_rejects_wrong_alignment(noise_runner):
    with pytest.raises(ValueError):
        noise_runner.summarize_noise(
            {"incumbent": np.zeros((3, 12, 3))},
            np.zeros((3, 120, 12, 3)),
            ["103.pkl", "a", "b"],
            RestartConfig(),
        )


def test_independent_metric_formulas_preserve_repetition_and_case_axes():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "noise_verifier", root / "scripts/verify_deform_state_restart_noise.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    truth = np.zeros((2, 4, 3, 3))
    points = np.broadcast_to([0.003, 0.004, 0.0], (5, *truth.shape)).copy()
    metrics = verifier.metric_arrays(points, truth)
    assert metrics["coordinate_l1_mm"].shape == (5, 2)
    np.testing.assert_allclose(metrics["coordinate_l1_mm"], 7 / 3)
    np.testing.assert_allclose(metrics["point_rmse_mm"], 5.0)
    np.testing.assert_allclose(metrics["fde_mm"], 5.0)
    points[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="cannot be dropped"):
        verifier.metric_arrays(points, truth)
