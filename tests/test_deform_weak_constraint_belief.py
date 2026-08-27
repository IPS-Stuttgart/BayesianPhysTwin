from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_forecast_sensing import (
    SensingConfig,
    query_pairs,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    file_digest,
    paired_physical_readout,
)
from bayesian_phystwin_experiments.deform_weak_constraint_belief import (
    ARMS,
    NATIVE_ARMS,
    BeliefConfig,
    arm_columns,
    arm_schedule,
    calibration_scales,
    gaussian_events,
    impulse_basis,
    infer_prefix,
    load_protocol,
    marginal_covariance,
    ols_endpoint,
    physical_impulses,
    primary_decision,
    query_design,
    scaled_covariance,
    summarize_uq,
    validate_response,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _response(rod: RestartConfig | None = None) -> np.ndarray:
    rod = rod or RestartConfig()
    pose, velocity = impulse_basis(rod, BeliefConfig())
    response = np.zeros((145, rod.node_count, 3, 60))
    for step, frame in enumerate((25, 33, 41, 49)):
        for t in range(frame, 170):
            response[t - 25] += (
                pose[step] + (t - frame) * 0.01 * velocity[step]
            ).transpose(1, 2, 0)
    return response


def test_protocol_complete_opened_roster_and_fixed_controls() -> None:
    protocol, parent = load_protocol(
        ROOT / "configs/sota/deform_weak_constraint_belief_v1.json", ROOT
    )
    assert protocol["primary_arm"] == "weak_16"
    assert protocol["arms"] == list(ARMS)
    assert parent["prediction_case_count"] == 30 and parent["analysis_case_count"] == 29
    assert protocol["calibration"]["outer_rank"] == 13
    assert protocol["automatic_promotion"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("protected_data_access", True),
        ("calibration_case_count", 14),
        ("observation_schedule_adaptation", True),
        ("primary_arm", "weak_8"),
        ("calibration_object", "DLO1"),
    ],
)
def test_scope_changes_rejected(tmp_path: Path, field: str, value: object) -> None:
    record = json.loads(
        (ROOT / "configs/sota/deform_weak_constraint_belief_v1.json").read_text()
    )
    record[field] = value
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_protocol(path, ROOT)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_std_m": 0},
        {"process_velocity_std_m_s": float("nan")},
        {"anchor_frame": 33},
        {"observation_frames": (25, 33, 41, 50)},
        {"calibration_bootstrap_replicates": 0},
    ],
)
def test_bad_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BeliefConfig(**kwargs)


@pytest.mark.parametrize("nodes,clamps", [(12, (0, 1, 10, 11)), (13, (0, 1, 11, 12))])
def test_process_impulses_are_causal_and_clamp_zero(
    nodes: int, clamps: tuple[int, ...]
) -> None:
    rod = RestartConfig(node_count=nodes, clamped_nodes=clamps)
    pose, velocity = impulse_basis(rod, BeliefConfig())
    assert pose.shape == (4, 60, nodes, 3)
    np.testing.assert_array_equal(pose[:, :, clamps], 0)
    np.testing.assert_array_equal(velocity[:, :, clamps], 0)
    np.testing.assert_allclose(pose[1:], velocity[1:] * 0.04)
    response = _response(rod)
    validate_response(response, rod)
    response[0, 3, 0, 24] = 1
    with pytest.raises(ValueError, match="pre-impulse"):
        validate_response(response, rod)


@pytest.mark.parametrize("arm", NATIVE_ARMS)
def test_shared_prior_and_matched_budget(arm: str) -> None:
    rod = RestartConfig()
    design = query_design(_response(), arm, rod, BeliefConfig())
    assert design.shape == (16, 3, len(arm_columns(arm)) + 3)
    np.testing.assert_array_equal(
        design[:, :, -3:], np.broadcast_to(0.005 * np.eye(3), (16, 3, 3))
    )
    assert len(arm_schedule(arm)) == int(arm.rsplit("_", 1)[1])
    assert all(
        query_pairs(rod, SensingConfig())[i][1] not in rod.hidden_nodes
        for i in arm_schedule(arm)
    )
    if arm == "weak_8":
        assert set(arm_columns(arm)) == set(range(24)) | set(range(48, 60))


def test_unknown_arm_rejected() -> None:
    with pytest.raises(ValueError):
        arm_columns("weak_32")


@pytest.mark.parametrize("arm", NATIVE_ARMS)
def test_sequential_posterior_matches_independent_batch(arm: str) -> None:
    verifier = _load(
        "weak_verifier_batch", ROOT / "scripts/verify_deform_weak_constraint_belief.py"
    )
    rod, config = RestartConfig(), BeliefConfig()
    response = _response()
    rng = np.random.default_rng(260834)
    points = rng.normal(0, 0.008, (16, 3))
    reference = np.zeros_like(points)
    mean, cov = infer_prefix(response, reference, points, arm, rod, config)
    design = verifier.independent_design(response, arm, rod.observed_nodes)
    start = 0 if arm.endswith("16") else 8
    other_mean, other_cov = verifier.batch_posterior(design[start:], points[start:])
    np.testing.assert_allclose(mean, other_mean, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(cov, other_cov, atol=1e-10, rtol=1e-8)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_weak_constraint_positive_and_zero_placebo() -> None:
    rod, config, response = RestartConfig(), BeliefConfig(), _response()
    design = query_design(response, "weak_16", rod, config)
    latent = np.zeros(63)
    latent[24:36] = 1
    latent[36:48] = -2
    latent[48:60] = 2
    observations = np.einsum("qdk,k->qd", design, latent)
    reference = np.zeros_like(observations)
    errors = {}
    for arm in ("strong_16", "weak_16"):
        mean, _ = infer_prefix(response, reference, observations, arm, rod, config)
        fitted = np.einsum("qdk,k->qd", query_design(response, arm, rod, config), mean)
        errors[arm] = np.linalg.norm(observations - fitted)
    assert errors["weak_16"] < 0.5 * errors["strong_16"]
    mean, _ = infer_prefix(response, reference, reference, "weak_16", rod, config)
    np.testing.assert_array_equal(mean, 0)
    pose, velocity, gain = physical_impulses(mean[None], "weak_16", rod, config)
    np.testing.assert_array_equal(pose, 0)
    np.testing.assert_array_equal(velocity, 0)
    np.testing.assert_array_equal(gain, 1)
    incumbent = np.zeros((1, 120, 12, 3), dtype=np.float32)
    native = np.ones_like(incumbent)
    assert paired_physical_readout(incumbent, native, native.copy()) is incumbent


def test_prior_predictive_process_error_bank_and_guarded_recovery() -> None:
    rod, config, response = RestartConfig(), BeliefConfig(), _response()
    rng = np.random.default_rng(260834)
    latent = rng.normal(size=(256, 63))
    h = query_design(response, "weak_16", rod, config)
    observations = np.einsum("qdk,bk->bqd", h, latent)
    observations += rng.normal(0, 0.001, observations.shape)
    hidden_response = response[25:, rod.hidden_nodes]
    truth = np.einsum("tnck,bk->btnc", hidden_response, latent[:, :-3])
    errors, guarded = {}, {}
    for arm in ("strong_16", "weak_16"):
        means = np.stack(
            [
                infer_prefix(response, np.zeros((16, 3)), row, arm, rod, config)[0]
                for row in observations
            ]
        )
        gain = physical_impulses(means, arm, rod, config)[2]
        prediction = np.einsum(
            "tnck,bk->btnc", hidden_response[..., list(arm_columns(arm))], means[:, :-3]
        )
        errors[arm] = np.mean((truth - prediction) ** 2)
        guarded[arm] = np.mean((truth - prediction * gain[:, None, None, None]) ** 2)
    assert errors["weak_16"] < 0.5 * errors["strong_16"]
    assert guarded["weak_16"] < guarded["strong_16"]


@pytest.mark.parametrize("arm", NATIVE_ARMS)
def test_impulses_match_independent_interpolation_and_total_bounds(arm: str) -> None:
    verifier = _load(
        "weak_verifier_impulses",
        ROOT / "scripts/verify_deform_weak_constraint_belief.py",
    )
    rod, config = RestartConfig(), BeliefConfig()
    coefficients = np.random.default_rng(21).normal(
        0, 3, (3, len(arm_columns(arm)) + 3)
    )
    pose, velocity, gain = physical_impulses(coefficients, arm, rod, config)
    alternate = verifier.independent_impulses(
        coefficients, arm, rod.node_count, rod.clamped_nodes
    )
    for a, b in zip((pose, velocity, gain), alternate, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-13)
    assert np.linalg.norm(pose, axis=-1).sum(axis=1).max() <= 0.030000000001
    assert np.linalg.norm(velocity, axis=-1).sum(axis=1).max() <= 0.300000000001
    bias_changed = coefficients.copy()
    bias_changed[:, -3:] += 1000
    for a, b in zip(
        (pose, velocity, gain),
        physical_impulses(bias_changed, arm, rod, config),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_tangent_covariance_preserves_physical_cross_terms_and_floor() -> None:
    config = BeliefConfig()
    response = _response()[None]
    p = np.eye(63)[None]
    gains = np.ones(1)
    covariance = marginal_covariance(response, p, gains, "weak_16", config)
    assert covariance.shape == (1, 120, 12, 3, 3)
    assert np.linalg.eigvalsh(covariance).min() >= 9e-6 - 1e-12
    np.testing.assert_allclose(
        covariance[0, :, 0], np.broadcast_to(9e-6 * np.eye(3), (120, 3, 3))
    )
    more_bias = p.copy()
    more_bias[:, -3:, -3:] *= 100
    np.testing.assert_array_equal(
        covariance, marginal_covariance(response, more_bias, gains, "weak_16", config)
    )
    floor = marginal_covariance(response, p, np.zeros(1), "weak_16", config)
    np.testing.assert_allclose(floor, np.broadcast_to(9e-6 * np.eye(3), floor.shape))
    p[0, 0, 0] = -1
    with pytest.raises(ValueError, match="PSD"):
        marginal_covariance(response, p, gains, "weak_16", config)


def test_ols_uses_all_four_times_and_reproduces_known_linear_endpoint() -> None:
    rod, config = RestartConfig(), BeliefConfig()
    times = (np.asarray((25, 33, 41, 49)) - 49) * 0.01
    offsets = 0.003 + times[:, None, None] * 0.01 + np.zeros((4, 4, 3))
    incumbent = np.zeros((1, 170, 12, 3), dtype=np.float32)
    pose, velocity, gain = ols_endpoint(
        incumbent, offsets.reshape(1, 16, 3), rod, config
    )
    np.testing.assert_allclose(pose[:, rod.observed_nodes], 0.003, atol=1e-14)
    np.testing.assert_allclose(velocity[:, rod.observed_nodes], 0.01, atol=1e-14)
    np.testing.assert_array_equal(gain, 1)


def test_gaussian_metric_units_and_independent_cholesky_scores() -> None:
    verifier = _load(
        "weak_verifier_uq", ROOT / "scripts/verify_deform_weak_constraint_belief.py"
    )
    rng = np.random.default_rng(260834)
    error = rng.normal(0, 0.01, (30000, 3))
    covariance = np.broadcast_to(0.01**2 * np.eye(3), (30000, 3, 3))
    values = gaussian_events(error, covariance)
    other = verifier.independent_uq(error, covariance)
    for key in values:
        np.testing.assert_allclose(values[key], other[key], atol=1e-9, rtol=1e-10)
    assert 0.89 < values["coverage_90"].mean() < 0.91
    assert 2.95 < values["nees"].mean() < 3.05
    assert 49 < values["geometric_full_width_mm"].mean() < 51


@pytest.mark.parametrize("kind", ["negative", "nan", "asymmetric"])
def test_invalid_covariance_rejected(kind: str) -> None:
    covariance = np.eye(3)
    if kind == "negative":
        covariance[0, 0] = -1
    elif kind == "nan":
        covariance[0, 0] = np.nan
    else:
        covariance[0, 1] = 1
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        gaussian_events(np.zeros(3), covariance)


def test_calibration_exact_source_count_rank_and_horizon_alignment() -> None:
    error = np.ones((13, 120, 4, 3)) * np.arange(1, 14)[:, None, None, None] * 0.001
    covariance = np.broadcast_to(1e-6 * np.eye(3), (*error.shape, 3)).copy()
    scales = calibration_scales(error, covariance, object_name="DLO2")
    np.testing.assert_allclose(scales["moment"], np.mean(np.arange(1, 14) ** 2))
    np.testing.assert_allclose(scales["conformal"], 3 * 13**2 / 6.251388631170325)
    for name in ("DLO1", "DLO3", "DLO4"):
        with pytest.raises(ValueError):
            calibration_scales(error, covariance, object_name=name)
    with pytest.raises(ValueError):
        calibration_scales(error[:12], covariance[:12], object_name="DLO2")
    scaled = scaled_covariance(covariance, [2, 3, 4])
    np.testing.assert_allclose(scaled[:, :40], covariance[:, :40] * 2)
    np.testing.assert_allclose(scaled[:, 40:80], covariance[:, 40:80] * 3)
    np.testing.assert_allclose(scaled[:, 80:], covariance[:, 80:] * 4)
    summary = summarize_uq(error, scaled)
    assert len(summary["per_case"]["nll"]) == 13
    assert set(summary["horizons"]) == {"early", "middle", "late"}


def test_disallowed_or_repeated_queries_reuse_checked_bank() -> None:
    from bayesian_phystwin_experiments.deform_forecast_sensing import LockedQueryBank

    pairs = query_pairs(RestartConfig(), SensingConfig())
    with pytest.raises(ValueError):
        LockedQueryBank(np.zeros((16, 3)), pairs, [0, 0])
    bank = LockedQueryBank(np.zeros((16, 3)), pairs, list(range(16)))
    for frame, node in ((50, 2), (25, 3), (49, 8)):
        with pytest.raises(ValueError):
            bank.reveal(frame, node)


def _runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/remote"))
    return _load(
        "weak_test_runner", ROOT / "scripts/remote/run_deform_weak_constraint_belief.py"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ordinary_success", 29),
        ("protected_data_access", True),
        ("new_metrics_computed", True),
        ("retained_technical_failure", 1),
    ],
)
def test_barrier_rejects_invalid_denominator_before_reading_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    runner = _runner(monkeypatch)
    record = {
        "schema": "deform-weak-constraint-belief-v1-prediction-barrier",
        "source_revision": "unit",
        "source_receipt_sha256": "unit",
        "protocol_sha256": file_digest(ROOT / runner.PROTOCOL),
        "ordinary_success": 30,
        "analysis_case_count": 29,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "no_replacement": True,
        "new_metrics_computed": False,
        "protected_data_access": False,
        "objects": {name: {} for name in ("DLO1", "DLO2", "DLO3")},
    }
    record[field] = value
    (tmp_path / "prediction_barrier.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_barrier(tmp_path, {}, {}, {"revision": "unit"}, "unit")


def test_retained_failure_blocks_before_barrier_or_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(monkeypatch)
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="technical failure"):
        runner.validate_barrier(tmp_path, {}, {}, {}, "unit")


def test_transfer_score_requires_explicit_calibration_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(monkeypatch)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}")
    monkeypatch.setattr(runner, "validate_barrier", lambda *args: {})

    def no_truth(*args):
        raise AssertionError("transfer truth must remain closed")

    monkeypatch.setattr(runner, "truth_for", no_truth)
    args = Namespace(
        output=tmp_path, source_receipt=receipt_path, calibration_sha256=""
    )
    with pytest.raises(ValueError, match="explicit source-calibration digest"):
        runner.score(args, {}, {}, {})


def test_calibration_opens_only_dlo2_after_complete_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(monkeypatch)
    protocol, parent = load_protocol(ROOT / runner.PROTOCOL, ROOT)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}")
    (tmp_path / "prediction_barrier.json").write_text("{}")
    stages = []

    def barrier(*args):
        stages.append("barrier")
        return {}

    def source_truth(item, parent):
        assert stages == ["barrier"] and item["object"] == "DLO2"
        stages.append("DLO2")
        return np.zeros((14, 120, 12, 3))

    monkeypatch.setattr(runner, "validate_barrier", barrier)
    monkeypatch.setattr(runner, "truth_for", source_truth)
    monkeypatch.setattr(
        runner,
        "load_object",
        lambda *args: (
            {
                "weak_16": np.zeros((14, 120, 12, 3)),
                "previous_paired_8": np.zeros((14, 120, 12, 3)),
            },
            {"weak_16": np.broadcast_to(9e-6 * np.eye(3), (14, 120, 12, 3, 3)).copy()},
        ),
    )
    args = Namespace(output=tmp_path, source_receipt=receipt_path)
    runner.calibrate(args, protocol, parent, {"revision": "unit"})
    record = json.loads((tmp_path / "calibration.json").read_text())
    assert stages == ["barrier", "DLO2"]
    assert record["calibration_case_count"] == 13
    assert record["transfer_metrics_computed"] is False
    runner.validate_calibration(
        tmp_path / "calibration.json",
        tmp_path,
        file_digest(tmp_path / "calibration.json"),
        {"revision": "unit"},
    )


def test_primary_gate_is_joint_and_secondary_cannot_rescue_it() -> None:
    import copy

    metrics = {
        arm: {
            "coordinate_l1_mm": 8.0,
            "point_rmse_mm": 16.0,
            "late": {"point_rmse_mm": 17.0},
            "point_rmse_mm_worst_case_ratio": 0.9,
        }
        for arm in ARMS
    }
    metrics["incumbent"].update(
        coordinate_l1_mm=10.0, point_rmse_mm=20.0, late={"point_rmse_mm": 20.0}
    )
    metrics["weak_16"].update(coordinate_l1_mm=7.0, point_rmse_mm=14.0)
    rows = {
        arm: [
            {
                "coordinate_l1_mm": v["coordinate_l1_mm"],
                "point_rmse_mm": v["point_rmse_mm"],
            }
            for _ in range(8)
        ]
        for arm, v in metrics.items()
    }
    item = {
        "point": {"summaries": metrics, "per_case": rows},
        "uq": {
            "weak_16_shaped__moment": {
                "per_case": {"nll": [1.0] * 8},
                "summary": {"coverage_90": 0.9, "ellipsoid_volume_mm3": 1.0},
            },
            "weak_16_isotropic__moment": {
                "per_case": {"nll": [2.0] * 8},
                "summary": {"ellipsoid_volume_mm3": 2.0},
            },
        },
    }
    results = {name: copy.deepcopy(item) for name in ("DLO1", "DLO2", "DLO3")}
    assert primary_decision(results, BeliefConfig())[
        "development_advancement_gate_passed"
    ]
    results["DLO1"]["point"]["summaries"]["weak_16"]["point_rmse_mm"] = 16
    results["DLO1"]["point"]["summaries"]["weak_8"]["point_rmse_mm"] = 0
    decision = primary_decision(results, BeliefConfig())
    assert decision["point_gate_passed"] is False
    assert decision["uncertainty_gate_passed"] is True
    assert decision["development_advancement_gate_passed"] is False
    assert decision["automatic_target_authorization"] is False
