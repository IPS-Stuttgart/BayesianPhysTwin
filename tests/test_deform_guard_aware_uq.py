from __future__ import annotations

import copy
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_guard_aware_uq import (
    FAMILIES,
    PRIMARY,
    PROTOCOL,
    RAW_ARMS,
    VARIANTS,
    GuardUQConfig,
    build_prediction,
    calibrate_source,
    calibrated_covariance,
    fixed_mean_second_moment,
    load_protocol,
    primary_decision,
    source_full_matrices,
    validate_covariance,
)
from bayesian_phystwin_experiments.deform_state_restart import array_digest, file_digest
from bayesian_phystwin_experiments.deform_weak_constraint_belief import gaussian_events

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    rng = np.random.default_rng(260835)
    mean = rng.normal(0, 0.1, (2, 120, 4, 3)).astype(np.float32)
    response = rng.normal(0, 0.01, (*mean.shape, 24))
    factor = rng.normal(size=(2, 27, 27))
    posterior = factor @ factor.swapaxes(-1, -2) / 27
    coefficients = rng.normal(0, 0.3, (2, 27))
    return dict(
        incumbent=mean.copy(),
        deployed_mean=mean,
        response=response,
        coefficients=coefficients,
        posterior=posterior,
        gains=np.array([0.0, 0.4]),
        registered_mean_sha256=array_digest(mean),
    )


def test_lock_preserves_opened_data_scope_and_primary() -> None:
    lock, parent = load_protocol(ROOT / PROTOCOL, ROOT)
    assert lock["primary_arm"] == PRIMARY
    assert parent["prediction_case_count"] == 30
    assert parent["analysis_case_count"] == 29
    assert lock["new_native_rollouts"] is False
    assert lock["exact_posterior_claim"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("primary_arm", "shadow__moment"),
        ("protected_data_access", True),
        ("input_prediction_barrier_sha256", "0" * 64),
        ("population_coverage_guarantee", True),
        ("point_mean", "strong_8"),
        ("new_observation_queries", True),
        ("extra_unregistered_field", True),
    ],
)
def test_lock_rejects_contract_changes(tmp_path: Path, key, value) -> None:
    lock = json.loads((ROOT / PROTOCOL).read_text())
    lock[key] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        load_protocol(path, ROOT)


def test_mean_is_same_object_and_same_bytes() -> None:
    args = _inputs()
    before = args["deployed_mean"].tobytes()
    result = build_prediction(**args)
    assert result["mean"] is args["deployed_mean"]
    assert result["mean"].dtype == np.float32
    assert result["mean"].tobytes() == before
    assert array_digest(result["mean"]) == args["registered_mean_sha256"]
    for key in RAW_ARMS:
        validate_covariance(result[key])


@pytest.mark.parametrize(
    "field",
    ["deployed_mean", "registered_mean_sha256", "posterior", "response", "gains"],
)
def test_bad_or_changed_inputs_fail_closed(field: str) -> None:
    args = _inputs()
    if field == "registered_mean_sha256":
        args[field] = "0" * 64
    elif field == "posterior":
        args[field][0, 0, 0] = -100
    elif field == "gains":
        args[field][0] = -1
    else:
        args[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        build_prediction(**args)


def test_zero_guard_does_not_erase_unresolved_physical_uncertainty() -> None:
    args = _inputs()
    result = build_prediction(**args)
    np.testing.assert_array_equal(result["guard_scaled"][0], result["isotropic"][0])
    assert np.linalg.eigvalsh(result["shadow"][0] - result["isotropic"][0]).min() > 0
    assert (
        np.linalg.eigvalsh(result["fixed_mean_bridge"] - result["shadow"]).min()
        > -1e-12
    )
    assert not np.array_equal(result["fixed_mean_bridge"][0], result["isotropic"][0])


def test_joint_physical_covariance_is_used_but_bias_is_not_physical() -> None:
    args = _inputs()
    baseline = build_prediction(**args)
    args["coefficients"][:, -3:] += 100
    shifted_bias = build_prediction(**args)
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], shifted_bias[key])
    diag = np.zeros_like(args["posterior"])
    indices = np.arange(27)
    diag[:, indices, indices] = np.diagonal(args["posterior"], axis1=-2, axis2=-1)
    args["posterior"] = diag
    assert not np.allclose(build_prediction(**args)["shadow"], baseline["shadow"])


def test_fixed_mean_second_moment_and_expected_log_score_optimum() -> None:
    rng = np.random.default_rng(260835)
    mu = np.array([0.017, -0.009, 0.008])
    a = np.zeros(3)
    factor = np.array([[0.004, 0, 0], [0.002, 0.005, 0], [-0.001, 0.003, 0.002]])
    covariance = factor @ factor.T
    optimum = fixed_mean_second_moment(mu, covariance, a)
    samples = rng.multivariate_normal(mu, covariance, size=200000)
    empirical = samples.T @ samples / len(samples)
    np.testing.assert_allclose(empirical, optimum, atol=1e-6, rtol=0.025)

    def loss(sigma):
        return 0.5 * (
            np.linalg.slogdet(sigma)[1] + np.trace(np.linalg.solve(sigma, optimum))
        )

    for _ in range(30):
        perturbation = rng.normal(0, 0.01, (3, 3))
        other = perturbation @ perturbation.T + 1e-6 * np.eye(3)
        assert loss(other) >= loss(optimum) - 1e-12
    assert loss(covariance) > loss(optimum)
    assert loss(0.2**2 * covariance) > loss(covariance)
    np.testing.assert_array_equal(
        fixed_mean_second_moment(mu, covariance, mu), covariance
    )


def test_rotation_preserves_volume_and_second_moment_equivariance() -> None:
    result = build_prediction(**_inputs())
    np.testing.assert_allclose(
        np.linalg.eigvalsh(result["rotated_bridge"]),
        np.linalg.eigvalsh(result["fixed_mean_bridge"]),
        rtol=1e-11,
        atol=1e-12,
    )
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    mu = np.array([0.1, 0.2, -0.3])
    a = np.array([0.05, -0.02, 0.01])
    covariance = np.diag([0.02, 0.01, 0.005])
    s = fixed_mean_second_moment(mu, covariance, a)
    rotated = fixed_mean_second_moment(q @ mu, q @ covariance @ q.T, q @ a)
    np.testing.assert_allclose(rotated, q @ s @ q.T, atol=1e-14)


def test_source_full_is_uncentered_and_equal_case_weighted() -> None:
    error = np.broadcast_to(np.array([0.01, -0.02, 0.03]), (13, 120, 4, 3)).copy()
    matrices = source_full_matrices(error, object_name="DLO2")
    expected = np.outer(error[0, 0, 0], error[0, 0, 0]) + 1e-12 * np.eye(3)
    np.testing.assert_allclose(matrices, np.broadcast_to(expected, (3, 3, 3)))
    assert matrices[0, 0, 0] > 1e-5  # Not the centered covariance of constant errors.


@pytest.mark.parametrize(
    "object_name,count", [("DLO1", 13), ("DLO3", 13), ("DLO2", 12), ("DLO2", 14)]
)
def test_transfer_or_wrong_denominator_cannot_fit(object_name: str, count: int) -> None:
    with pytest.raises(ValueError, match="thirteen"):
        source_full_matrices(np.zeros((count, 120, 4, 3)), object_name=object_name)


def test_all_six_calibration_comparators_use_same_source_and_mean() -> None:
    rng = np.random.default_rng(7)
    error = rng.normal(size=(13, 120, 4, 3)) * [0.003, 0.012, 0.02] + [0.01, 0, 0]
    raw = {
        arm: np.broadcast_to(np.eye(3) * (i + 1) * 1e-5, (*error.shape, 3)).copy()
        for i, arm in enumerate(RAW_ARMS)
    }
    fit = calibrate_source(error, raw, object_name="DLO2")
    assert set(fit["scales"]) == set(FAMILIES)
    for family in FAMILIES:
        covariance = calibrated_covariance(raw, fit, family, "moment")
        nees = gaussian_events(error, covariance)["nees"]
        for frames in np.array_split(np.arange(120), 3):
            assert nees[:, frames].mean() == pytest.approx(3, abs=1e-11)
        conformal = calibrated_covariance(raw, fit, family, "conformal")
        validate_covariance(conformal)
    with pytest.raises(ValueError):
        calibrated_covariance(raw, fit, "unregistered", "moment")


def _decision_results():
    results = {}
    for name in ("DLO1", "DLO2", "DLO3"):
        arms = {}
        for a in FAMILIES:
            for b in VARIANTS:
                arms[a + "__" + b] = {
                    "per_case": {
                        "nll": [-9.0 if a == "fixed_mean_bridge" else -8.0] * 8
                    },
                    "summary": {
                        "coverage_90": 0.9,
                        "ellipsoid_volume_mm3": 10.0
                        if a == "fixed_mean_bridge"
                        else 12.0,
                    },
                }
        results[name] = {"uq": arms}
    return results


def test_primary_cannot_be_rescued_by_secondary_and_requires_both_objects() -> None:
    results = _decision_results()
    decision = primary_decision(results, mean_identity=True, accounted_cases=30)
    assert decision["development_advancement_gate_passed"]
    assert not decision["automatic_target_authorization"]
    assert not primary_decision(results, mean_identity=False, accounted_cases=30)[
        "development_advancement_gate_passed"
    ]
    assert not primary_decision(results, mean_identity=True, accounted_cases=29)[
        "development_advancement_gate_passed"
    ]
    results["DLO3"]["uq"][PRIMARY]["per_case"]["nll"] = [-7.0] * 8
    assert not primary_decision(results, mean_identity=True, accounted_cases=30)[
        "development_advancement_gate_passed"
    ]


@pytest.mark.parametrize("mode", ["coverage", "volume", "missing_arm", "missing_case"])
def test_gate_rejects_bad_calibration_or_incomplete_denominator(mode: str) -> None:
    results = _decision_results()
    if mode == "coverage":
        results["DLO1"]["uq"][PRIMARY]["summary"]["coverage_90"] = 0.99
    elif mode == "volume":
        results["DLO1"]["uq"][PRIMARY]["summary"]["ellipsoid_volume_mm3"] = 13
    elif mode == "missing_arm":
        del results["DLO1"]["uq"]["source_full__moment"]
    else:
        results["DLO1"]["uq"][PRIMARY]["per_case"]["nll"] = [-9] * 7
    if mode.startswith("missing"):
        with pytest.raises(ValueError):
            primary_decision(results, mean_identity=True, accounted_cases=30)
    else:
        assert not primary_decision(results, mean_identity=True, accounted_cases=30)[
            "development_advancement_gate_passed"
        ]


def _runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/remote"))
    spec = importlib.util.spec_from_file_location(
        "guard_uq_test_runner", ROOT / "scripts/remote/run_deform_guard_aware_uq.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_runner_barrier_no_truth_and_write_once(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(monkeypatch)
    protocol, parent = load_protocol(ROOT / PROTOCOL, ROOT)
    protocol = {
        **protocol,
        "run_root": str(tmp_path / "run"),
        "attempt_ledger": str(tmp_path / "attempt.json"),
    }
    args = Namespace(
        output=tmp_path / "run",
        source_receipt=tmp_path / "source.json",
        calibration_sha256="",
    )
    args.source_receipt.write_text("{}")
    receipt = {"revision": "a" * 40}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner.np, "__version__", "1.24.3")
    monkeypatch.setattr(runner, "input_barrier", lambda p: {})

    def no_truth(*args, **kwargs):
        raise AssertionError("prediction or barrier opened truth")

    monkeypatch.setattr(runner.parent_runner, "truth_for", no_truth)

    def predictions(item, protocol, parent):
        mean = np.zeros((len(item["names"]), 120, 4, 3), dtype=np.float32)
        cov = np.broadcast_to(np.eye(3) * 1e-5, (*mean.shape, 3)).copy()
        return {
            "names": np.asarray(item["names"]),
            "mean": mean,
            "shadow_mean": mean.astype(float),
            **{a: cov for a in RAW_ARMS},
        }, array_digest(mean)

    monkeypatch.setattr(runner, "expected_arrays", predictions)
    runner.predict(args, protocol, parent, receipt)
    verified = runner.validate_barrier(args, protocol, parent, receipt)
    assert set(verified) == {"DLO1", "DLO2", "DLO3"}
    with pytest.raises(FileExistsError):
        runner.predict(args, protocol, parent, receipt)
    with pytest.raises(ValueError, match="root"):
        runner.consume_attempt(protocol, tmp_path / "second-root", receipt)
    with pytest.raises(ValueError, match="digest"):
        runner.validate_calibration(args, receipt)
    barrier = json.loads((args.output / "prediction_barrier.json").read_text())
    barrier["ordinary_success"] = 29
    (args.output / "prediction_barrier.json").write_text(json.dumps(barrier))
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_barrier(args, protocol, parent, receipt)


def test_changed_parent_carrier_stops_before_array_loading(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(monkeypatch)
    path = tmp_path / "prediction_barrier.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="input prediction barrier"):
        runner.input_barrier(
            {"input_root": str(tmp_path), "input_prediction_barrier_sha256": "0" * 64}
        )
    assert file_digest(path) != "0" * 64


def test_invalid_config_and_covariance() -> None:
    with pytest.raises(ValueError):
        GuardUQConfig(floor_std_m=0)
    with pytest.raises(ValueError):
        validate_covariance(np.full((3, 3), np.nan))
    with pytest.raises(np.linalg.LinAlgError):
        validate_covariance(-np.eye(3))
    args = _inputs()
    changed = copy.deepcopy(args)
    changed["coefficients"] = np.zeros((2, 24))
    with pytest.raises(ValueError, match="dimensions"):
        build_prediction(**changed)


def test_independent_verifier_on_synthetic_matrix_bank(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import verify_deform_guard_aware_uq as verifier

    args = _inputs()
    actual = build_prediction(**args)
    expected = verifier.independent_prediction(
        args["incumbent"],
        args["deployed_mean"],
        args["response"],
        args["coefficients"],
        args["posterior"],
        args["gains"],
    )
    assert set(actual) == set(expected)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-11, atol=1e-12)
    rng = np.random.default_rng(16)
    error = rng.normal(0, 0.02, (13, 120, 4, 3))
    raw = {
        a: np.broadcast_to(
            np.diag([1e-5, 2e-5, 3e-5]) * (i + 1), (*error.shape, 3)
        ).copy()
        for i, a in enumerate(RAW_ARMS)
    }
    verifier.tree_close(
        calibrate_source(error, raw, object_name="DLO2"),
        verifier.independent_calibration(error, raw),
    )
    with pytest.raises(AssertionError):
        verifier.tree_close({"passed": False}, {"passed": True})
