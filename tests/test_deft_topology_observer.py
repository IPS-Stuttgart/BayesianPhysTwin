"""Source-independent topology, information-boundary, and scoring contracts."""

from __future__ import annotations

import hashlib
import io
import pickle

import numpy as np
import pytest
from test_deft_cross_branch_source import _FakeNative

from bayesian_phystwin_experiments.deft_cross_branch_source import pack_branched_world
from bayesian_phystwin_experiments.deft_native_restart import PARENT_CLAMPS
from bayesian_phystwin_experiments.deft_topology_observer import (
    ARMS,
    CASE_IDS,
    COMPARATORS,
    HIDDEN,
    OBSERVED,
    PARENT_OBSERVED,
    PRIMARY,
    interpolate_topology,
    load_training_case,
    observe,
    permitted_inputs,
    predict_topology,
    score_case,
    score_study,
    synthetic_qualification,
    topology_basis,
    topology_increments,
)


def test_declared_space_observability_and_synthetic_recovery():
    report = synthetic_qualification()
    assert report["passed"]
    assert len(report["checks"]) == 6
    assert report["source_trajectory_decoded"] is False
    assert "not full dynamical observability" in report["scope"]


def test_basis_is_dirichlet_and_preserves_junction_clamp_padding_contracts():
    basis = topology_basis()
    np.testing.assert_array_equal(
        np.stack([basis[b, n] for b, n in OBSERVED]), np.eye(4)
    )
    np.testing.assert_array_equal(basis[0, PARENT_CLAMPS], 0)
    np.testing.assert_array_equal(basis[1, 0], basis[0, 4])
    np.testing.assert_array_equal(basis[2, 0], basis[0, 8])
    np.testing.assert_array_equal(basis[1, 5:], 0)
    np.testing.assert_array_equal(basis[2, 4:], 0)
    np.testing.assert_allclose(basis[1, 2], (basis[1, 0] + basis[1, 4]) / 2)


def test_parent_only_cannot_see_a_child_local_tip_displacement_in_this_basis():
    coefficients = np.zeros((4, 3))
    coefficients[2, 0] = 0.015
    field = interpolate_topology(coefficients)
    np.testing.assert_array_equal(observe(field, PARENT_OBSERVED), 0)
    assert np.linalg.norm(observe(field)) == pytest.approx(0.015)
    assert np.linalg.norm(field[1, 2]) > 0


def test_future_hidden_values_never_enter_permitted_inputs():
    points = pack_branched_world(
        np.arange(30000, dtype=float).reshape(500, 20, 3) / 100000
    )
    original = permitted_inputs(points)
    masked = np.full_like(points, np.nan)
    masked[:2] = points[:2]
    masked[2:172, 0][:, PARENT_CLAMPS] = points[2:172, 0][:, PARENT_CLAMPS]
    for frame in (43, 51):
        for branch, node in set(OBSERVED + PARENT_OBSERVED):
            masked[frame, branch, node] = points[frame, branch, node]
    copied = permitted_inputs(masked)
    for name in original:
        np.testing.assert_array_equal(copied[name], original[name])
    assert original["topology_observations"].shape == (2, 4, 3)
    assert original["parent_control_observations"].shape == (2, 4, 3)


@pytest.mark.parametrize("location", [(0, 0, 4), (51, 1, 4), (100, 0, 1)])
def test_missing_permitted_measurement_is_rejected_not_filled(location):
    trajectory = np.zeros((500, 3, 13, 3))
    trajectory[location] = np.nan
    with pytest.raises(ValueError, match="missing/nonfinite"):
        permitted_inputs(trajectory)


@pytest.mark.parametrize("frames", [49, 51, 170])
def test_update_rejects_nonprefix_reference(frames):
    with pytest.raises(ValueError, match="prefix"):
        topology_increments(np.zeros((frames, 3, 13, 3)), np.zeros((2, 4, 3)))


def test_metric_residual_slope_not_absolute_observation_velocity():
    prefix = np.zeros((50, 3, 13, 3))
    prefix[41] = 0.1
    prefix[49] = 0.2
    measured = observe(prefix[[41, 49]]) + 0.01
    dx, dv = topology_increments(prefix, measured)
    np.testing.assert_allclose(observe(dx), 0.01)
    np.testing.assert_allclose(dv, 0, atol=1e-14)


def test_source_loader_requires_both_pinned_checksums(tmp_path):
    raw = np.arange(30000, dtype=np.float64).reshape(3, 500, 20)
    payload = pickle.dumps(raw, protocol=4)
    path = tmp_path / "synthetic.pkl"
    path.write_bytes(payload)
    spec = {
        "git_blob": hashlib.sha1(
            b"blob " + str(len(payload)).encode() + b"\0" + payload
        ).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    np.testing.assert_array_equal(
        load_training_case(path, spec), pack_branched_world(raw.transpose(1, 2, 0))
    )
    for key in spec:
        with pytest.raises(ValueError, match="bytes changed"):
            load_training_case(path, {**spec, key: "0" * len(spec[key])})


def _inputs():
    return {
        "initial_two": np.zeros((2, 3, 13, 3)),
        "clamps": np.zeros((170, 4, 3)),
        "topology_observations": np.zeros((2, 4, 3)),
        "parent_control_observations": np.zeros((2, 4, 3)),
    }


def test_zero_update_preserves_all_fake_native_predictions_and_every_arm():
    arrays, controls = predict_topology(_FakeNative(), _FakeNative(), _inputs())
    assert set(arrays) == set(ARMS)
    assert controls["zero_update_byte_identical"]
    assert controls["point_observations_per_corrected_arm"] == 8
    for array in arrays.values():
        assert array.shape == (120, 3, 13, 3)
        assert array.dtype == np.float64
        np.testing.assert_array_equal(array, arrays["native_full"])


def test_linear_synthetic_response_matches_declared_gain_one_forecast():
    inputs = _inputs()
    inputs["topology_observations"][0] = 0.004
    inputs["topology_observations"][1] = 0.012
    predictions, _ = predict_topology(_FakeNative(), _FakeNative(), inputs)
    expected = (
        interpolate_topology(np.full((4, 3), 0.012))[None]
        + np.arange(1, 121)[:, None, None, None]
        * 0.01
        * interpolate_topology(np.full((4, 3), 0.1))[None]
    )
    np.testing.assert_allclose(predictions[PRIMARY], expected, atol=1e-12)
    np.testing.assert_allclose(
        predictions[PRIMARY],
        predictions["topology_readout_linear_velocity"],
        atol=1e-12,
    )


def test_predictor_rejects_hidden_truth_channel():
    inputs = _inputs()
    inputs["future_truth"] = np.zeros((120, 3, 13, 3))
    with pytest.raises(ValueError, match="forbidden"):
        predict_topology(_FakeNative(), _FakeNative(), inputs)


def test_policy_channels_cannot_leak_into_other_arm_predictions():
    original, _ = predict_topology(_FakeNative(), _FakeNative(), _inputs())
    parent_changed = _inputs()
    parent_changed["parent_control_observations"][:] = 0.03
    after, _ = predict_topology(_FakeNative(), _FakeNative(), parent_changed)
    for arm in ARMS:
        if arm != "parent_paired_pose_velocity":
            np.testing.assert_array_equal(after[arm], original[arm])
    topology_changed = _inputs()
    topology_changed["topology_observations"][:] = 0.02
    after, _ = predict_topology(_FakeNative(), _FakeNative(), topology_changed)
    np.testing.assert_array_equal(
        after["parent_paired_pose_velocity"], original["parent_paired_pose_velocity"]
    )


def test_hidden_ids_are_disjoint_from_both_observation_policies():
    all_observed = set(OBSERVED + PARENT_OBSERVED)
    hidden = {(branch, node) for branch, nodes in HIDDEN.values() for node in nodes}
    assert not hidden & all_observed
    assert len(hidden) == 10
    truth = np.zeros((120, 3, 13, 3))
    predictions = {arm: np.full_like(truth, 0.01) for arm in ARMS}
    original = score_case(predictions, truth)
    for point in all_observed | {(1, 0), (2, 0)}:
        for array in predictions.values():
            array[:, point[0], point[1]] = 1000
    assert score_case(predictions, truth) == original


def _scores(primary=0.5):
    truth = np.zeros((120, 3, 13, 3))
    predictions = {arm: np.full_like(truth, 0.02) for arm in ARMS}
    predictions[PRIMARY] *= primary
    return score_case(predictions, truth)


def test_complete_recording_gate_and_no_secondary_rescue():
    cases = {case: _scores() for case in CASE_IDS}
    result = score_study(cases)
    assert result["source_gate_passed"]
    assert all(n == 3 for n in result["recording_joint_wins"].values())
    assert len(result["checks"]) == 7 * len(COMPARATORS) + 1
    cases[CASE_IDS[0]] = _scores(2)
    assert not score_study(cases)["source_gate_passed"]
    with pytest.raises(ValueError, match="denominator"):
        score_study({CASE_IDS[0]: cases[CASE_IDS[0]]})


def test_aggregate_is_equal_recording_then_equal_child_not_pooled_points():
    cases = {case: _scores(0.5 + i * 0.1) for i, case in enumerate(CASE_IDS)}
    result = score_study(cases)
    assert result["equal_recording_mean"][PRIMARY]["equal_child_branch"][
        "point_rmse_mm"
    ] == pytest.approx(np.sqrt(3) * 20 * 0.6)


def test_perfect_baseline_tie_is_not_a_gain():
    truth = np.zeros((120, 3, 13, 3))
    row = score_case({arm: truth.copy() for arm in ARMS}, truth)
    assert not score_study({case: row for case in CASE_IDS})["source_gate_passed"]


def test_nonnumeric_pickle_is_never_executed():
    from bayesian_phystwin_experiments.deft_cross_branch_source import _NumericUnpickler

    class Unsafe:
        def __reduce__(self):
            return eval, ("42",)

    with pytest.raises(ValueError, match="nonnumeric"):
        _NumericUnpickler(io.BytesIO(pickle.dumps(Unsafe()))).load()
