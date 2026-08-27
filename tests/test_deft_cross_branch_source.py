"""Synthetic contracts; these tests never open the selected DEFT trajectory."""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin_experiments.deft_cross_branch_source as module
from bayesian_phystwin_experiments.deform_state_restart import file_digest
from bayesian_phystwin_experiments.deft_cross_branch_source import (
    ARMS,
    PRIMARY,
    _NumericUnpickler,
    branch_increments,
    load_numeric_training_source,
    pack_branched_world,
    permitted_inputs,
    predict_cross_branch,
    score_cross_branch,
)
from bayesian_phystwin_experiments.deft_native_restart import (
    PARENT_CLAMPS,
    STATE_FIELDS,
    DeftState,
)

ROOT = Path(__file__).resolve().parents[1]


def _trajectory():
    raw = np.arange(500 * 20 * 3, dtype=float).reshape(500, 20, 3) / 100000
    return pack_branched_world(raw)


def test_raw_axes_junction_duplicates_and_padding_match_upstream():
    raw = np.arange(500 * 20 * 3, dtype=float).reshape(500, 20, 3)
    result = pack_branched_world(raw)
    np.testing.assert_array_equal(result[:, 0, :, 0], -raw[:, :13, 2])
    np.testing.assert_array_equal(result[:, 0, :, 1], -raw[:, :13, 0])
    np.testing.assert_array_equal(result[:, 0, :, 2], raw[:, :13, 1])
    np.testing.assert_array_equal(result[:, 1, 0], result[:, 0, 4])
    np.testing.assert_array_equal(result[:, 2, 0], result[:, 0, 8])
    np.testing.assert_array_equal(result[:, 1, 5:], 0)
    np.testing.assert_array_equal(result[:, 2, 4:], 0)


def test_future_and_unobserved_prefix_truth_cannot_change_inputs():
    trajectory = _trajectory()
    before = permitted_inputs(trajectory)
    corrupted = trajectory.copy()
    corrupted[2:] = np.nan
    corrupted[2:172, 0][:, PARENT_CLAMPS] = trajectory[2:172, 0][:, PARENT_CLAMPS]
    for frame in (43, 51):
        corrupted[frame, 0, (2, 4, 6, 8)] = trajectory[frame, 0, (2, 4, 6, 8)]
    after = permitted_inputs(corrupted)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert after["sparse_parent_observations"].shape == (2, 4, 3)
    assert after["clamps"].shape == (170, 4, 3)


@pytest.mark.parametrize("field", ["initial", "clamp", "observation"])
def test_invalid_permitted_measurement_is_not_filled(field):
    trajectory = _trajectory()
    location = {"initial": (0, 0, 2), "clamp": (100, 0, 0), "observation": (51, 0, 8)}[
        field
    ]
    trajectory[location] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        permitted_inputs(trajectory)


def test_harmonic_extension_and_residual_velocity_reuse_deform_rule():
    reference = np.zeros((50, 3, 13, 3))
    observed = np.zeros((2, 4, 3))
    observed[0] = 0.01
    observed[1] = 0.018
    dx, dv = branch_increments(reference, observed, extend_children=True)
    np.testing.assert_allclose(dx[0, (2, 4, 6, 8)], 0.018)
    np.testing.assert_allclose(dv[0, (2, 4, 6, 8)], 0.1)
    np.testing.assert_array_equal(dx[0, PARENT_CLAMPS], 0)
    np.testing.assert_allclose(dx[1, :5], 0.018)
    np.testing.assert_allclose(dx[2, :4], 0.018)
    np.testing.assert_array_equal(dx[1, 5:], 0)
    parent_dx, parent_dv = branch_increments(reference, observed, extend_children=False)
    np.testing.assert_array_equal(parent_dx[0], dx[0])
    np.testing.assert_array_equal(parent_dx[1:, 1:], 0)
    np.testing.assert_array_equal(parent_dx[1, 0], dx[0, 4])
    np.testing.assert_array_equal(parent_dv[2, 0], dv[0, 8])


@pytest.mark.parametrize("length", [49, 51, 170])
def test_increments_reject_nonprefix_reference(length):
    with pytest.raises(ValueError):
        branch_increments(
            np.zeros((length, 3, 13, 3)), np.zeros((2, 4, 3)), extend_children=True
        )


def test_restricted_numeric_loader_matches_upstream_reshape(tmp_path, monkeypatch):
    raw = np.arange(30000, dtype=np.float64).reshape(3, 500, 20)
    path = tmp_path / "numeric.pkl"
    path.write_bytes(pickle.dumps(raw, protocol=4))
    monkeypatch.setattr(module, "SOURCE_FILE_SHA256", file_digest(path))
    result = load_numeric_training_source(path)
    np.testing.assert_array_equal(result, pack_branched_world(raw.transpose(1, 2, 0)))
    path.write_bytes(pickle.dumps(raw + 1, protocol=4))
    with pytest.raises(ValueError, match="metadata-selected"):
        load_numeric_training_source(path)


def test_unpickler_rejects_executable_globals():
    class Unsafe:
        def __reduce__(self):
            return eval, ("42",)

    with pytest.raises(ValueError, match="nonnumeric global"):
        _NumericUnpickler(io.BytesIO(pickle.dumps(Unsafe()))).load()


class _FakeNative:
    model_id = "a" * 64

    def rollout(self, initial, actions, state=None):
        start = 0 if state is None else state.prediction_index + 1
        count = len(actions) - start
        if state is None:
            position = np.zeros((3, 13, 3))
            velocity = np.zeros_like(position)
        else:
            position = state.fields["b_DLOs_vertices"]
            velocity = state.fields["b_DLOs_velocity"]
        points = (
            position[None]
            + np.arange(1, count + 1)[:, None, None, None] * 0.01 * velocity[None]
        )
        fields = {name: np.zeros((3, 13, 3)) for name in STATE_FIELDS}
        fields["b_DLOs_vertices"] = points[-1].copy()
        fields["b_DLOs_velocity"] = velocity.copy()
        final = DeftState(len(actions) - 1, self.model_id, fields)
        return points, np.broadcast_to(velocity, points.shape).copy(), final


def test_predictor_only_consumes_permitted_input_keys_and_seals_all_arms():
    inputs = {
        "initial_two": np.zeros((2, 3, 13, 3)),
        "clamps": np.zeros((170, 4, 3)),
        "sparse_parent_observations": np.zeros((2, 4, 3)),
    }
    predictions, controls = predict_cross_branch(_FakeNative(), _FakeNative(), inputs)
    assert set(predictions) == set(ARMS)
    assert controls["zero_update_byte_identical"] is True
    for array in predictions.values():
        assert array.shape == (120, 3, 13, 3)
        np.testing.assert_array_equal(array, 0)
    inputs["future_truth"] = np.ones((120, 3, 13, 3))
    with pytest.raises(ValueError, match="forbidden"):
        predict_cross_branch(_FakeNative(), _FakeNative(), inputs)


def test_metrics_ignore_duplicate_roots_and_padding_but_gate_each_child():
    truth = np.zeros((120, 3, 13, 3))
    predictions = {arm: np.ones_like(truth) * 0.02 for arm in ARMS}
    predictions[PRIMARY] *= 0.5
    before = score_cross_branch(predictions, truth)
    assert before["source_pilot_gate_passed"] is True
    for array in predictions.values():
        array[:, 1:, 0] = 1000
        array[:, 1, 5:] = 1000
        array[:, 2, 4:] = 1000
    assert score_cross_branch(predictions, truth) == before
    predictions[PRIMARY][:, 2, 1:4] = 0.025
    after = score_cross_branch(predictions, truth)
    assert after["source_pilot_gate_passed"] is False
    assert after["checks"]["child1_rmse_at_least_5pct_better_than_native_full"] is True
    assert after["checks"]["child2_rmse_at_least_5pct_better_than_native_full"] is False


def test_secondary_success_does_not_rescue_primary_failure():
    truth = np.zeros((120, 3, 13, 3))
    predictions = {arm: np.ones_like(truth) * 0.01 for arm in ARMS}
    predictions["paired_parent_only_pose_velocity"] *= 0
    result = score_cross_branch(predictions, truth)
    assert result["source_pilot_gate_passed"] is False
    assert result["independent_confirmation"] is False
    assert result["inferential_confidence_interval"] is None


def test_missing_arm_or_nonfinite_future_is_retained_failure():
    truth = np.zeros((120, 3, 13, 3))
    predictions = {arm: truth.copy() for arm in ARMS}
    with pytest.raises(ValueError):
        score_cross_branch({"native_full": truth}, truth)
    truth[10, 1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_cross_branch(predictions, truth)


def test_perfect_baseline_tie_does_not_count_as_a_five_percent_gain():
    truth = np.zeros((120, 3, 13, 3))
    result = score_cross_branch({arm: truth.copy() for arm in ARMS}, truth)
    assert result["source_pilot_gate_passed"] is False


def test_protocol_matches_fixed_implementation_and_public_training_boundary():
    protocol = json.loads(
        (ROOT / "configs/sota/deft_cross_branch_source_v1.json").read_text()
    )
    assert tuple(protocol["arms"]) == ARMS
    assert protocol["primary_arm"] == PRIMARY
    assert protocol["source"]["sha256"] == module.SOURCE_FILE_SHA256
    assert protocol["source"]["split"] == "train"
    assert protocol["source"]["recording_count"] == 1
    assert protocol["inputs"]["point_observation_budget"] == 8
    assert protocol["inputs"]["state_update_gain"] == 1.0
    assert protocol["pilot_gate"]["automatic_confirmation_authorization"] is False
