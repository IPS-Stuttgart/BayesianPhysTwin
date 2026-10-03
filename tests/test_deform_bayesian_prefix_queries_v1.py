"""Source-only query custody, exact conditioning, and headroom stop rules."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("scipy")

from experiments.deform_bayesian_prefix_queries_v1.core import (  # noqa: E402
    PANELS,
    Context,
    PrefixProvider,
    QueryScore,
    choose,
    context_from_gp,
    headroom_summary,
    pairs,
    prefix_times,
)
from experiments.deform_bayesian_prefix_queries_v1.gp import (  # noqa: E402
    GPConfig,
    fit_gp,
)
from experiments.deform_bayesian_prefix_queries_v1.prepare import (  # noqa: E402
    causal_replay_input,
    permitted_names,
)


@pytest.fixture
def context():
    rng = np.random.default_rng(71)
    design = rng.normal(size=(80, 8, 5))
    # Broad correlated mode provides an identifiable positive control.
    design[:, :, 0] = 1
    cov = np.repeat(np.eye(5)[None] * 1e-4, 3, axis=0)
    return Context(
        np.zeros((80, 12, 3)), design, cov, np.full(3, 1e-5), np.eye(3), 50, 25, 0
    )


def test_protocol_binds_original_data_and_stop_rule():
    protocol = json.loads(
        (
            ROOT / "experiments/deform_bayesian_prefix_queries_v1/protocol.json"
        ).read_text()
    )
    assert protocol["split_counts"] == {"fit": 39, "calibration": 9, "source_test": 8}
    assert protocol["headroom"]["minimum_relative_rmse_improvement"] == 0.1
    assert protocol["primary_budget"] == 2
    assert len(protocol["prefix_offsets"]) == 10
    assert protocol["gp"] == {
        key: value
        for key, value in GPConfig().__dict__.items()
        if key != "kernel_jitter"
    }


def test_source_test_names_never_returned():
    split = {"fit": ["a"], "calibration": ["b"], "source_test": ["closed"]}
    manifest = {
        "dlo_type": "DLO3",
        "partition": "train",
        "official_eval_read": False,
        "split": split,
    }
    protocol = {"split_counts": dict.fromkeys(split, 1)}
    assert permitted_names(manifest, protocol) == {"fit": ["a"], "calibration": ["b"]}
    manifest["split"]["calibration"] = ["a"]
    with pytest.raises(ValueError, match="overlapping"):
        permitted_names(manifest, protocol)


def test_native_replay_receives_no_future_free_coordinates():
    rng = np.random.default_rng(3)
    full = rng.normal(size=(500, 12, 3))
    changed = full.copy()
    changed[2:, 2:-2] += 1000
    assert np.array_equal(causal_replay_input(full), causal_replay_input(changed))
    assert np.array_equal(
        causal_replay_input(full)[:, (0, 1, 10, 11)], full[:, (0, 1, 10, 11)]
    )


@pytest.mark.parametrize("endpoint", [50, 150, 250])
def test_registered_times(endpoint):
    times = prefix_times(endpoint)
    assert len(times) == 10 and times[0] == endpoint - 46 and times[-1] == endpoint - 1
    assert np.all(np.diff(times) == 5)


def test_restricted_history_provider():
    values = np.arange(120).reshape(10, 4, 3)
    provider = PrefixProvider(prefix_times(50), PANELS[0], values, endpoint=50)
    assert np.array_equal(provider.reveal((2, 6)), values[:, (0, 2)])
    with pytest.raises(PermissionError):
        provider.reveal((3, 5))
    with pytest.raises(ValueError):
        provider.reveal((2, 2))
    with pytest.raises(ValueError):
        PrefixProvider(prefix_times(50) + 5, PANELS[0], values, endpoint=50)


def test_unqueried_mutations_cannot_change_forecast(context):
    values = np.ones((10, 4, 3))
    changed = values.copy()
    changed[:, (1, 3)] = 1e9
    a = PrefixProvider(prefix_times(50), PANELS[0], values, endpoint=50)
    b = PrefixProvider(prefix_times(50), PANELS[0], changed, endpoint=50)
    assert np.array_equal(
        context.forecast((2, 6), a.reveal((2, 6))).positions,
        context.forecast((2, 6), b.reveal((2, 6))).positions,
    )


def test_suffix_mutation_invariance(context):
    truth = np.zeros((80, 12, 3))
    changed = truth.copy()
    changed[50:] = 1e10
    results = []
    for full in (truth, changed):
        provider = PrefixProvider(
            prefix_times(50),
            PANELS[0],
            full[np.ix_(prefix_times(50), PANELS[0])],
            endpoint=50,
        )
        selected = choose([context.score(p) for p in pairs(0)]).nodes
        results.append(context.forecast(selected, provider.reveal(selected)).positions)
    assert np.array_equal(*results)


def test_identical_forecasts_across_policy_labels(context):
    observation = np.full((10, 2, 3), 0.01)
    predictions = [
        context.forecast((2, 4), observation).positions
        for _ in ("posterior", "bootstrap", "ridge", "random")
    ]
    assert all(np.array_equal(predictions[0], p) for p in predictions[1:])


def test_covariance_changes_acquisition_not_open_mean(context):
    cov = context.coefficient_covariance.copy()
    cov[:, 1:, 1:] *= 1e-3
    alternative = replace(context, coefficient_covariance=cov)
    assert np.array_equal(context.open_loop, alternative.open_loop)
    assert not np.isclose(
        context.score((2, 4)).expected_mse_reduction_m2,
        alternative.score((2, 4)).expected_mse_reduction_m2,
        atol=1e-12,
    )


def test_matched_distribution_parity(context):
    alternative = replace(
        context, coefficient_covariance=context.coefficient_covariance.copy()
    )
    a = [context.score(p) for p in pairs(0)]
    b = [alternative.score(p) for p in pairs(0)]
    assert a == b and choose(a) == choose(b)


def test_deterministic_ties():
    scores = [QueryScore((4, 6), 1.0, True), QueryScore((2, 8), 1.0, True)]
    assert choose(scores).nodes == (2, 8)
    assert choose(scores[::-1]).nodes == (2, 8)
    assert not choose([QueryScore((2, 4), 0, False)]).valid


@pytest.mark.parametrize("invalid", ["indefinite", "asymmetric"])
def test_bad_covariance_falls_back_bit_exact(context, invalid):
    cov = context.coefficient_covariance.copy()
    if invalid == "indefinite":
        cov[0, 0, 0] = -1
    else:
        cov[0, 0, 1] += 1
    bad = replace(context, coefficient_covariance=cov)
    forecast = bad.forecast((2, 4), np.ones((10, 2, 3)))
    assert not forecast.valid and not bad.score((2, 4)).valid
    assert np.array_equal(forecast.positions, context.open_loop[50:75])


def test_no_query_and_clamps_are_exact(context):
    context.open_loop[:] = np.random.default_rng(4).normal(size=context.open_loop.shape)
    assert np.array_equal(
        context.forecast((), np.empty((0,))).positions, context.open_loop[50:75]
    )
    prediction = context.forecast((2, 4), np.ones((10, 2, 3)))
    assert prediction.valid
    assert np.array_equal(
        prediction.positions[:, (0, 1, 10, 11)],
        context.open_loop[50:75, (0, 1, 10, 11)],
    )


def test_four_query_order_parity(context):
    rng = np.random.default_rng(19)
    obs = rng.normal(size=(10, 4, 3))
    a = context.forecast(PANELS[0], obs)
    b = context.forecast(PANELS[0][::-1], obs[:, ::-1])
    assert np.array_equal(a.positions, b.positions)


def test_analytic_reduction_matches_monte_carlo(context):
    selected, q, f, _, _ = context._operator((2, 4))
    analytic = context.score(selected).expected_mse_reduction_m2
    rng = np.random.default_rng(27)
    draws = 40000
    reductions = []
    for k in range(3):
        coefficients = rng.multivariate_normal(
            np.zeros(5), context.coefficient_covariance[k], draws
        )
        truth = coefficients @ f.T
        observation = coefficients @ q.T + rng.normal(
            0, np.sqrt(context.noise_m2[k]), (draws, len(q))
        )
        cross = f @ context.coefficient_covariance[k] @ q.T
        yy = (
            q @ context.coefficient_covariance[k] @ q.T
            + np.eye(len(q)) * context.noise_m2[k]
        )
        prediction = observation @ np.linalg.solve(yy, cross.T)
        reductions.append(float(np.mean(truth**2 - (truth - prediction) ** 2)))
    assert np.mean(reductions) == pytest.approx(analytic, rel=0.025)


def test_no_information_placebo(context):
    context.coefficient_covariance[:] = 0
    assert all(context.score(p).expected_mse_reduction_m2 == 0 for p in pairs(0))
    assert np.array_equal(
        context.forecast((2, 4), np.ones((10, 2, 3))).positions,
        context.open_loop[50:75],
    )


def test_pooled_gp_can_update_unqueried_nodes():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(60, 3))
    y = np.column_stack((x[:, 0], x[:, 0] * 2, x[:, 1])) + rng.normal(0, 0.1, (60, 3))
    gp = fit_gp(x, y, np.repeat(np.arange(3), 20), GPConfig(max_anchors=8))
    features = rng.normal(size=(80, 8, 3))
    ctx = context_from_gp(
        gp,
        features,
        np.zeros((80, 12, 3)),
        np.eye(3),
        endpoint=50,
        horizon=25,
        panel=0,
        inflation=1.0,
    )
    obs = ctx.open_loop[np.ix_(prefix_times(50), (2, 4))] + 0.01
    prediction = ctx.forecast((2, 4), obs)
    assert prediction.valid
    assert not np.array_equal(
        prediction.positions[:, PANELS[1]], ctx.open_loop[50:75][:, PANELS[1]]
    )


def test_headroom_zero_and_positive():
    names = [str(i) for i in range(9)]
    zero = np.ones((9, 3, 2, 6)) * 1e-4
    assert not headroom_summary(zero, names)["passed"]
    for i in range(9):
        zero[i, :, :, i % 6] = 1e-6
    result = headroom_summary(zero, names)
    assert result["passed"] and result["relative_rmse_headroom"] > 0.1
    assert result["statistical_unit"] == "complete_recording"


def test_incomplete_panel_cannot_pass():
    result = headroom_summary(np.ones((8, 3, 2, 6)), list(map(str, range(8))))
    assert not result["passed"]


def test_headroom_aggregates_recordings_not_coordinate_counts():
    a = np.ones((9, 3, 2, 6))
    a[0] *= 100
    result = headroom_summary(a, list(map(str, range(9))))
    assert result["best_fixed_rmse_mm"] == pytest.approx(np.sqrt(108 / 9) * 1000)


def test_experimental_runner_excludes_source_test_loading(tmp_path):
    spec = importlib.util.find_spec("experiments.deform_bayesian_prefix_queries_v1.run")
    assert spec is not None
    from experiments.deform_bayesian_prefix_queries_v1.run import load_cache

    with pytest.raises(PermissionError):
        load_cache(tmp_path, "source_test", {})
