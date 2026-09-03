"""Synthetic correctness tests only; never open a recorded-data carrier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/science"
sys.path.insert(0, str(SCRIPTS))
import deform360_joint_events_v1 as joint  # noqa: E402
import run_deform360_joint_events_v1 as runner  # noqa: E402
import verify_deform360_joint_events_v1 as verifier  # noqa: E402


@pytest.fixture
def protocol():
    value = json.loads(runner.PROTOCOL.read_text())
    value["sobol_power"] = 9
    value["integration_replicates"] = 2
    value["bootstrap_repetitions"] = 100
    return value


def residuals(seed=9):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(800, 5))
    values[:, 3] = 0.98 * values[:, 2] + 0.05 * values[:, 3]
    return values


def test_matched_marginals_and_mean(protocol):
    errors = residuals()
    samples, parity = joint.coupled_draws(
        errors, np.cov(errors, rowvar=False), protocol
    )
    assert parity["sorted_query_marginal_max_error"] == 0
    assert parity["shared_point_mean_max_error"] < 1e-14
    expected = np.sort(samples["structured_gaussian"], axis=1)
    for value in samples.values():
        np.testing.assert_array_equal(np.sort(value, axis=1), expected)


def test_true_joint_event_changes_with_dependence(protocol):
    errors = residuals()
    samples, _ = joint.coupled_draws(errors, np.cov(errors, rowvar=False), protocol)
    predictions = joint.event_predictions(np.zeros((1, 5)), np.ones(5), samples)
    # Both spatial responses exceeding a magnitude bound is not a scalar query.
    full = predictions["p_structured_gaussian"][0, 2]
    independent = predictions["p_independent"][0, 2]
    assert full > independent + 0.1
    # The one-dimensional absolute exceedance probabilities are nevertheless equal.
    for index in range(5):
        full_p = (np.abs(samples["structured_gaussian"][..., index]) > 1).mean()
        independent_p = (np.abs(samples["independent"][..., index]) > 1).mean()
        assert full_p == independent_p


@pytest.mark.parametrize(
    "event,first,second", [(0, 0, 1), (1, 0, 1), (2, 2, 3), (3, 2, 3), (4, 2, 3)]
)
def test_each_event_uses_multiple_queries(event, first, second):
    values = np.zeros((4, 5))
    values[1, first] = 2
    values[2, second] = 2
    values[3, [first, second]] = 2
    output = joint.event_values(values, np.ones(5))[:, event]
    if event in {0, 2, 4}:
        assert output.tolist() == [False, False, False, True]
    else:
        assert output.tolist() == [False, True, True, True]


def test_field_scale_cannot_fake_dependence_gain(protocol):
    errors = residuals()
    covariance = np.cov(errors, rowvar=False)
    first, _ = joint.coupled_draws(errors, covariance, protocol)
    second, _ = joint.coupled_draws(errors, 100 * covariance, protocol)
    for arm in joint.ARMS:
        np.testing.assert_array_equal(first[arm], second[arm])


def test_nonfinite_or_indefinite_covariance_rejected(protocol):
    covariance = np.eye(5)
    covariance[0, 1] = covariance[1, 0] = 3
    with pytest.raises(np.linalg.LinAlgError):
        joint.coupled_draws(residuals(), covariance, protocol)
    with pytest.raises(ValueError):
        joint.coupled_draws(residuals() * np.nan, np.eye(5), protocol)


@pytest.mark.parametrize("dimension", [96, 193, 0])
def test_invalid_sensor_dimensions_rejected(dimension):
    with pytest.raises(ValueError):
        joint.query_bank(dimension)


def test_query_bank_constant_field():
    weights = joint.query_bank(384)
    np.testing.assert_allclose(weights @ np.ones(384), [1, 0, 0, 0, 0], atol=1e-14)


def test_independent_event_implementation():
    values = np.random.default_rng(3901).normal(size=(3, 100, 5))
    threshold = np.array([0.5, 1.0, 1.2, 0.8, 1.5])
    np.testing.assert_array_equal(
        joint.event_values(values, threshold), verifier.events(values, threshold)
    )


def test_original_target_excluded_and_recording_split():
    items = [SimpleNamespace(episode_id=i) for i in [2, 0, 1]]
    train, evaluation, excluded = runner.split_descriptors(items, 3)
    assert [item.episode_id for item in train] == [0, 1]
    assert evaluation.episode_id == 2 and excluded == 3
    with pytest.raises(ValueError):
        runner.split_descriptors(items, 2)
    with pytest.raises(ValueError):
        runner.split_descriptors(items[:2], 3)


def test_write_once_receipt(tmp_path):
    path = tmp_path / "attempt.json"
    runner.write_new(path, {"launch_count": 1})
    with pytest.raises(FileExistsError):
        runner.write_new(path, {"launch_count": 2})
    assert json.loads(path.read_text())["launch_count"] == 1


def test_constant_logistic_and_fallback_literal_cost(protocol):
    features = np.arange(120, dtype=float).reshape(40, 3)
    labels = np.zeros((40, 5), dtype=bool)
    model = joint.fit_direct_logistic(features, labels, 1)
    p = joint.direct_predict(model, features)
    np.testing.assert_allclose(p, 1 / 42)
    predictions = {"p_structured_gaussian": p, "p_direct_logistic": p}
    scores = joint.score_predictions(
        predictions, np.full((40, 5), 2), np.ones(5), protocol
    )
    assert scores["metrics"]["always_fallback"]["decision_loss"] == 0.1
    assert scores["metrics"]["structured_gaussian"]["decision_loss"] == 1
    assert scores["metrics"]["matched_activity_direct"]["execute_fraction"] == 1


def synthetic_episode(identifier, seed):
    rng = np.random.default_rng(seed)
    descriptor = runner.base.EpisodeDescriptor(
        "synthetic",
        identifier,
        "lift",
        Path("robot.npy"),
        (Path("left"), Path("right")),
        (None, None),
    )
    robot = np.zeros((160, 2, 5, 3))
    robot[:, 0, 0, 2] = np.arange(160) / 160
    tactile = np.maximum(
        0, 1 + np.cumsum(rng.normal(scale=0.01, size=(160, 192)), axis=0)
    )
    return runner.base.EpisodeData(descriptor, tactile, robot, True, {})


def test_forecast_features_do_not_read_suffix_and_match_legacy(protocol):
    source = [synthetic_episode(0, 1), synthetic_episode(1, 2)]
    predecessor = runner.base.read_json(
        runner.ROOT / "protocols/deform360_action_conditioned_tactile_v2.json"
    )
    transform = runner.base.build_transform(source, predecessor, 32)
    episode = synthetic_episode(2, 3)
    starts = np.array([40])
    before = runner.causal_inputs(episode, transform, predecessor, starts)
    episode.tactile[41:] = 1e6
    after = runner.causal_inputs(episode, transform, predecessor, starts)
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)
    episode = synthetic_episode(2, 3)
    starts = runner.base.starts_for(160, 32, 4, 8)
    state, action, _ = runner.causal_inputs(episode, transform, predecessor, starts)
    legacy = runner.base.design_for_episode(
        episode, transform, predecessor, 32, "action"
    )[0]
    np.testing.assert_array_equal(np.column_stack((state, action)), legacy)


def test_complete_synthetic_fit_predict_score(protocol):
    source = [synthetic_episode(0, 10), synthetic_episode(1, 11)]
    fit = runner.fit_source(source, protocol)
    episode = synthetic_episode(2, 12)
    predictions = runner.predict_episode(episode, fit)
    values = runner.base.normalize_tactile(
        episode.tactile, fit["transform"].feature_scale, 5
    )
    truth = values[predictions["starts"] + 32] @ fit["queries"].T
    result = joint.score_predictions(predictions, truth, fit["thresholds"], protocol)
    result["object_id"] = "synthetic"
    summary = joint.aggregate([result], protocol)
    assert summary["object_count"] == 1
    assert not summary["superiority_gate"]
    assert not summary["decision_gate"]
    assert set(joint.ARMS) <= result["metrics"].keys()
    assert predictions["p_direct_logistic"].shape == truth.shape


def test_reference_copies_only_original_source_metadata():
    reference = runner.base.read_json(
        runner.ROOT / "protocols/locks/deform360_joint_events_v1_source_access.json"
    )
    assert len(reference["objects"]) == 14
    assert reference["outcome_fields_copied"] is False
    for item in reference["objects"]:
        assert "metrics" not in item and "target_fingerprint" not in item
        assert all(
            value["episode_id"] < item["excluded_original_episode_id"]
            for value in item["source_descriptors"]
        )
