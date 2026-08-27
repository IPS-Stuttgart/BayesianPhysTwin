from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_forecast_sensing import (
    LockedQueryBank,
    SensingConfig,
    bounded_increments,
    clean_arm_names,
    covariance_update,
    greedy_schedule,
    infer_coefficients,
    load_protocol,
    material_basis,
    native_arm_names,
    noise_arm_names,
    planning_matrices,
    primary_decision,
    query_noise,
    query_pairs,
    schedules_for_case,
    temporal_controls,
    validate_schedule,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    array_digest,
    file_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def _response(rod: RestartConfig | None = None) -> np.ndarray:
    rod = rod or RestartConfig()
    rng = np.random.default_rng(260831)
    response = rng.normal(0, 0.006, (145, rod.node_count, 3, 24))
    response[:, rod.clamped_nodes] = 0
    return response


def test_frozen_protocol_and_opened_denominator() -> None:
    protocol, parent = load_protocol(
        ROOT / "configs/sota/deform_forecast_aware_sensing_v1.json", ROOT
    )
    assert protocol["primary_arm"] == "forecast_8"
    assert parent["prediction_case_count"] == 30
    assert parent["analysis_case_count"] == 29
    assert [len(x["names"]) for x in parent["objects"]] == [8, 14, 8]
    assert protocol["automatic_promotion"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("protected_data_access", True),
        ("measurement_values_used_in_schedule_selection", True),
        ("primary_arm", "uniform_8"),
    ],
)
def test_protocol_refuses_scope_changes(
    tmp_path: Path, field: str, value: object
) -> None:
    protocol = json.loads(
        (ROOT / "configs/sota/deform_forecast_aware_sensing_v1.json").read_text()
    )
    protocol[field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        load_protocol(path, ROOT)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_std_m": 0},
        {"measurement_std_m": float("nan")},
        {"query_frames": (25, 33, 49, 50)},
        {"query_frames": (25, 33, 33, 49)},
        {"budgets": (4, 8)},
    ],
)
def test_invalid_sensing_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SensingConfig(**kwargs)


@pytest.mark.parametrize("nodes,clamps", [(12, (0, 1, 10, 11)), (13, (0, 1, 11, 12))])
def test_material_basis_reproduces_query_values_and_zero_clamps(
    nodes: int, clamps: tuple[int, ...]
) -> None:
    rod = RestartConfig(node_count=nodes, clamped_nodes=clamps)
    basis = material_basis(rod)
    assert basis.shape == (12, nodes, 3)
    np.testing.assert_array_equal(basis[:, clamps], 0)
    np.testing.assert_array_equal(
        basis[:, rod.observed_nodes].reshape(12, 12), np.eye(12)
    )


def test_planning_objective_is_future_not_observed_shape() -> None:
    design = np.zeros((2, 3, 2))
    design[0, 0, 0] = 1
    design[1, 0, 1] = 0.5
    future = np.diag([1, 100])
    current = np.diag([100, 1])
    assert greedy_schedule(design, future, 1, 0.1) == (1,)
    assert greedy_schedule(design, current, 1, 0.1) == (0,)


def test_schedules_use_only_model_response_and_are_causal() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    response = _response(rod)
    before = array_digest(response)
    plans, design, weight = schedules_for_case(response, rod, sensing, seed=12)
    again = schedules_for_case(response.copy(), rod, sensing, seed=12)[0]
    assert plans == again
    assert array_digest(response) == before
    assert plans["uniform_8"] == tuple(range(8, 16))
    assert plans["uniform_16"] == plans["forecast_16"] == tuple(range(16))
    pairs = query_pairs(rod, sensing)
    for arm, indices in plans.items():
        assert len(indices) == int(arm.split("_")[1])
        assert len(set(indices)) == len(indices)
        assert all(
            pairs[i][0] < 50 and pairs[i][1] not in rod.hidden_nodes for i in indices
        )
    assert design.shape == (16, 3, 27)
    assert weight.shape == (27, 27)
    np.testing.assert_array_equal(weight[-3:], 0)
    assert np.linalg.eigvalsh(weight).min() >= -1e-12


def test_future_objective_uses_hidden_model_predictions_not_query_nodes() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    response = _response()
    design, future, current = planning_matrices(response, rod, sensing)
    changed = response.copy()
    changed[25:, rod.observed_nodes] *= 100
    new_design, new_future, new_current = planning_matrices(changed, rod, sensing)
    np.testing.assert_array_equal(design, new_design)
    np.testing.assert_array_equal(future, new_future)
    np.testing.assert_array_equal(current, new_current)


def test_bias_nuisance_is_not_independent_noise() -> None:
    covariance = np.eye(2)
    same = np.zeros((3, 2))
    same[0] = (1, 1)
    first, _ = covariance_update(covariance, same, 1e-6)
    repeated, _ = covariance_update(first, same, 1e-6)
    assert repeated[0, 0] > 0.49
    contrast = same.copy()
    contrast[0, 1] = -1
    identified, _ = covariance_update(first, contrast, 1e-6)
    assert identified[0, 0] < 1e-5


@pytest.mark.parametrize("schedule", [[0, 0], [0, 16], [-1, 2], [0.0, 1]])
def test_duplicate_or_noncanonical_queries_rejected(schedule: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_schedule(schedule, 16, 2)


def test_bank_refuses_future_hidden_repeat_and_out_of_order_reads() -> None:
    pairs = query_pairs(RestartConfig(), SensingConfig())
    points = np.zeros((16, 3))
    bank = LockedQueryBank(points, pairs, [0, 15])
    for pair in ((49, 8), (25, 3), (50, 2)):
        with pytest.raises(ValueError):
            bank.reveal(*pair)
    bank.reveal(25, 2)
    with pytest.raises(ValueError):
        bank.reveal(25, 2)
    bank.reveal(49, 8)
    with pytest.raises(ValueError):
        bank.reveal(49, 8)
    assert bank.access_log == [(25, 2), (49, 8)]


def test_unqueried_values_cannot_change_inference() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    pairs = query_pairs(rod, sensing)
    plans, design, _ = schedules_for_case(_response(), rod, sensing, seed=9)
    chosen = plans["forecast_8"]
    values = np.random.default_rng(7).normal(0, 0.01, (16, 3))
    changed = values.copy()
    changed[[i for i in range(16) if i not in chosen]] += 1e6
    args = (design, np.zeros_like(values))
    first = infer_coefficients(
        *args, LockedQueryBank(values, pairs, chosen), pairs, chosen, 0.001
    )
    second = infer_coefficients(
        *args, LockedQueryBank(changed, pairs, chosen), pairs, chosen, 0.001
    )
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_posterior_matches_independent_batch_solution() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    pairs = query_pairs(rod, sensing)
    design = planning_matrices(_response(), rod, sensing)[0]
    rng = np.random.default_rng(90)
    selected = (0, 1, 5, 7, 8, 9, 14, 15)
    reference = rng.normal(size=(16, 3))
    observations = reference + rng.normal(0, 0.01, (16, 3))
    mean, covariance = infer_coefficients(
        design,
        reference,
        LockedQueryBank(observations, pairs, selected),
        pairs,
        selected,
        0.001,
    )
    matrix = design[list(selected)].reshape(-1, 27)
    residual = (observations - reference)[list(selected)].reshape(-1)
    precision = np.eye(27) + matrix.T @ matrix / 1e-6
    expected_covariance = np.linalg.solve(precision, np.eye(27))
    expected_mean = np.linalg.solve(precision, matrix.T @ residual / 1e-6)
    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-11)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_synthetic_known_correction_recovered_and_zero_residual_exact() -> None:
    pairs = ((25, 2), (25, 4), (25, 6), (25, 8))
    design = np.zeros((4, 3, 6))
    design[:2, :, :3] = 0.01 * np.eye(3)
    design[2:, :, 3:] = 0.01 * np.eye(3)
    target = np.array([0.2, -0.1, 0.3, -0.2, 0.5, 0.1])
    values = np.einsum("qik,k->qi", design, target)
    schedule = (0, 1, 2, 3)
    mean, covariance = infer_coefficients(
        design,
        np.zeros((4, 3)),
        LockedQueryBank(values, pairs, schedule),
        pairs,
        schedule,
        1e-6,
    )
    assert np.linalg.norm(mean - target) / np.linalg.norm(target) < 1e-7
    assert np.trace(covariance) < 1e-6
    zero, _ = infer_coefficients(
        design, values, LockedQueryBank(values, pairs, schedule), pairs, schedule, 0.001
    )
    np.testing.assert_array_equal(zero, 0)


def test_zero_budget_keeps_prior_and_observes_nothing() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    pairs = query_pairs(rod, sensing)
    design = planning_matrices(_response(), rod, sensing)[0]
    bank = LockedQueryBank(np.zeros((16, 3)), pairs, [])
    mean, covariance = infer_coefficients(
        design, np.zeros((16, 3)), bank, pairs, [], 0.001
    )
    np.testing.assert_array_equal(mean, 0)
    np.testing.assert_array_equal(covariance, np.eye(27))
    assert bank.access_log == []


def test_radial_bound_is_common_and_clamps_are_preserved() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    values = np.zeros((3, 27))
    values[1, 0] = 10
    values[1, 12] = 1
    values[2, -3:] = 10000
    pose, velocity, gain = bounded_increments(values, rod, sensing)
    np.testing.assert_allclose(gain, [1, 0.3, 1])
    assert np.linalg.norm(pose, axis=-1).max() <= 0.03 + 1e-12
    assert np.linalg.norm(velocity, axis=-1).max() <= 0.3 + 1e-12
    np.testing.assert_array_equal(pose[:, rod.clamped_nodes], 0)
    np.testing.assert_array_equal(velocity[:, rod.clamped_nodes], 0)
    np.testing.assert_array_equal(pose[2], 0)
    np.testing.assert_array_equal(velocity[2], 0)


def test_temporal_controls_use_exactly_two_times_and_horizon_seconds() -> None:
    rod, sensing = RestartConfig(), SensingConfig()
    incumbent = np.zeros((1, 170, 12, 3), dtype=np.float32)
    observed = np.zeros((1, 2, 4, 3))
    observed[:, 0, :, 0] = 0.008
    observed[:, 1, :, 0] = 0.016
    controls = temporal_controls(incumbent, observed, rod, sensing)
    assert len(controls) == 9
    assert np.shares_memory(controls["incumbent"], incumbent)
    np.testing.assert_allclose(controls["temporal_static"][0, :, 3, 0], 0.016)
    np.testing.assert_allclose(
        controls["temporal_linear"][0, [0, -1], 3, 0], [0.017, 0.136]
    )
    expected = np.exp(-1.2 / 0.3) * (0.016 + 1.2 * 0.1)
    np.testing.assert_allclose(controls["temporal_decay_300ms"][0, -1, 3, 0], expected)
    for value in controls.values():
        np.testing.assert_array_equal(value[:, :, rod.clamped_nodes], 0)


def test_noise_shared_across_all_query_times_not_independent_evidence() -> None:
    shape = (8, 16, 3)
    independent = query_noise(shape, seed=4, shared=False)
    combined = query_noise(shape, seed=4, shared=True)
    difference = combined - independent
    np.testing.assert_allclose(
        difference, np.broadcast_to(difference[:, :1], shape), atol=1e-17
    )
    np.testing.assert_array_equal(independent, query_noise(shape, seed=4, shared=False))


def _gate_fixture() -> dict[str, object]:
    def row(l1: float, rmse: float, late: float) -> dict[str, object]:
        return {
            "coordinate_l1_mm": l1,
            "point_rmse_mm": rmse,
            "late": {"point_rmse_mm": late},
            "joint_wins": 6,
        }

    return {
        name: {
            "clean": {
                "summaries": {
                    "incumbent": row(10, 20, 20),
                    "uniform_8": row(9, 18, 18),
                    "forecast_8": row(8, 16, 17),
                    "temporal_static": row(9, 17, 18),
                }
            }
        }
        for name in ("DLO1", "DLO2", "DLO3")
    }


def test_gate_fails_late_regression_despite_mean_improvement() -> None:
    values = _gate_fixture()
    assert primary_decision(values)["development_advancement_gate_passed"] is True
    values["DLO1"]["clean"]["summaries"]["forecast_8"]["late"]["point_rmse_mm"] = 21
    result = primary_decision(values)
    assert result["development_advancement_gate_passed"] is False
    assert result["automatic_target_authorization"] is False


def test_gate_requires_actual_scheduling_gain_and_strong_temporal_controls() -> None:
    values = _gate_fixture()
    values["DLO3"]["clean"]["summaries"]["uniform_8"]["point_rmse_mm"] = 16
    assert primary_decision(values)["development_advancement_gate_passed"] is False
    values = _gate_fixture()
    values["DLO3"]["clean"]["summaries"]["temporal_static"]["point_rmse_mm"] = 15
    assert primary_decision(values)["development_advancement_gate_passed"] is False


@pytest.fixture
def runner():
    path = ROOT / "scripts/remote/run_deform_forecast_aware_sensing.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("sensing_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(path.parent))
    sys.modules.pop(spec.name, None)


def test_npz_write_once_and_array_contracts(runner, tmp_path: Path) -> None:
    path = tmp_path / "points.npz"
    arrays = {"points": np.arange(24, dtype=np.float32).reshape(2, 4, 3)}
    record = runner.save_arrays(path, arrays)
    restored = runner.verified_arrays(path, record)
    assert array_digest(restored["points"]) == array_digest(arrays["points"])
    with pytest.raises(FileExistsError):
        runner.save_arrays(path, arrays)
    record["arrays"]["points"] = "0" * 64
    with pytest.raises(ValueError, match="array identity"):
        runner.verified_arrays(path, record)


def test_nonfinite_predictions_cannot_be_successes(runner, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        runner.save_arrays(tmp_path / "nan.npz", {"points": np.array([np.nan])})
    assert not (tmp_path / "nan.npz").exists()


def test_failure_or_incomplete_barrier_blocks_scoring_before_input_access(
    runner, tmp_path: Path
) -> None:
    protocol, parent = load_protocol(
        ROOT / "configs/sota/deform_forecast_aware_sensing_v1.json", ROOT
    )
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="technical failure"):
        runner.validate_barrier(
            tmp_path, protocol, parent, {"revision": "a" * 40}, "b" * 64
        )
    (tmp_path / "failure.json").unlink()
    (tmp_path / "prediction_barrier.json").write_text("{}")
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_barrier(
            tmp_path, protocol, parent, {"revision": "a" * 40}, "b" * 64
        )


def _synthetic_barrier(runner, root: Path):
    protocol, original_parent = load_protocol(
        ROOT / "configs/sota/deform_forecast_aware_sensing_v1.json", ROOT
    )
    parent = copy.deepcopy(original_parent)
    receipt, receipt_digest = {"revision": "a" * 40}, "b" * 64
    sensing = SensingConfig()
    completed, plan_records = {}, {}

    def put(path: Path, value: dict[str, object]) -> None:
        path.write_text(json.dumps(value, sort_keys=True))

    for item in parent["objects"]:
        directory = root / item["object"]
        directory.mkdir()
        batch, count = len(item["names"]), item["node_count"]
        names = np.asarray(item["names"])
        incumbent = np.zeros((batch, 170, count, 3), dtype=np.float32)
        source_path = directory / "source-incumbent.npz"
        source_record = runner.save_arrays(
            source_path, {"names": names, "candidate": incumbent}
        )
        item["archive"] = {
            "path": str(source_path),
            "sha256": source_record["sha256"],
            "incumbent_key": "candidate",
        }
        model = {
            "names": names,
            "incumbent": incumbent,
            "nominal_from_anchor": np.zeros((batch, 145, count, 3)),
            "response": np.zeros((batch, 145, count, 3, 24)),
            "query_design": np.zeros((batch, 16, 3, 27)),
            "future_objective": np.zeros((batch, 27, 27)),
            "current_objective": np.zeros((batch, 27, 27)),
        }
        model["query_design"][..., -3:] = 0.005 * np.eye(3)
        files = {"model.npz": runner.save_arrays(directory / "model.npz", model)}
        plans = {
            arm: [list(range(16 - int(arm.split("_")[1]), 16)) for _ in range(batch)]
            for arm in native_arm_names(sensing)
        }
        for case in range(batch):
            for budget in sensing.budgets:
                plans[f"forecast_{budget}"][case] = list(range(budget))
            plans["current_8"][case] = list(range(8))
            rng = np.random.default_rng(
                sensing.random_seed + item["noise_seed_offset"] + case
            )
            for repetition in range(sensing.random_repetitions):
                plans[f"random_8_seed{repetition}"][case] = sorted(
                    rng.choice(16, 8, replace=False).tolist()
                )
        put(
            directory / "plans.json",
            {
                "schema": "deform-forecast-sensing-query-plans-v1",
                "object": item["object"],
                "names": item["names"],
                "model_sha256": files["model.npz"]["sha256"],
                "query_pairs": [
                    [t, n]
                    for t in sensing.query_frames
                    for n in parent["observed_nodes"]
                ],
                "schedules": plans,
                "measurement_values_read": False,
                "future_truth_read": False,
            },
        )
        controls = {
            key: True
            for key in (
                "parent_incumbent_byte_identical",
                "parent_native_future_byte_identical",
                "zero_native_update_byte_identical",
                "zero_readout_returns_original_object",
                "clamp_response_exactly_zero",
                "full_budget_schedule_identical",
            )
        }
        controls["archived_gpu_max_error_m"] = 0.0
        put(directory / "controls.json", controls)
        base = incumbent[:, 50:]
        for condition in ("clean", *protocol["noise"]["conditions"]):
            arms = (
                clean_arm_names(sensing)
                if condition == "clean"
                else noise_arm_names(sensing)
            )
            points = base if condition == "clean" else np.repeat(base[None], 8, axis=0)
            predictions = {"names": names, **{arm: points for arm in arms}}
            filename = f"{condition}.npz"
            files[filename] = runner.save_arrays(directory / filename, predictions)
            fits = {}
            for arm in (
                native_arm_names(sensing)
                if condition == "clean"
                else ("uniform_8", "forecast_8")
            ):
                budget = int(arm.split("_")[1])
                values = {
                    "coefficients": np.zeros((batch, 27)),
                    "covariance": np.repeat(np.eye(27)[None], batch, axis=0),
                    "gain": np.ones(batch),
                    "observed_positions": np.zeros((batch, budget, 3)),
                }
                for key, value in values.items():
                    fits[f"{arm}__{key}"] = (
                        value
                        if condition == "clean"
                        else np.repeat(value[None], 8, axis=0)
                    )
            filename = f"fits_{condition}.npz"
            files[filename] = runner.save_arrays(directory / filename, fits)
        plan_digest = file_digest(directory / "plans.json")
        put(
            directory / "prediction_seal.json",
            {
                "schema": "deform-forecast-sensing-object-seal-v1",
                "object": item["object"],
                "names": item["names"],
                "case_count": batch,
                "files": files,
                "plans_sha256": plan_digest,
                "controls_sha256": file_digest(directory / "controls.json"),
                "clean_arms": list(clean_arm_names(sensing)),
                "previous_paired_prediction_byte_identical": True,
                "future_free_node_truth_used": False,
                "new_metrics_computed": False,
                "protected_data_access": False,
            },
        )
        completed[item["object"]] = {
            "ordinary_success": batch,
            "seal_sha256": file_digest(directory / "prediction_seal.json"),
        }
        plan_records[item["object"]] = {
            "plans_sha256": plan_digest,
            "model_sha256": files["model.npz"]["sha256"],
        }
    protocol_digest = file_digest(
        ROOT / "configs/sota/deform_forecast_aware_sensing_v1.json"
    )
    put(
        root / "preflight.json",
        {
            "schema": "deform-forecast-sensing-preflight-v1",
            "source_revision": receipt["revision"],
            "source_receipt_sha256": receipt_digest,
            "protocol_sha256": protocol_digest,
            "prediction_case_count": 30,
            "analysis_case_count": 29,
            "protected_data_access": False,
            "new_metrics_computed": False,
        },
    )
    put(
        root / "query_plan_barrier.json",
        {
            "schema": "deform-forecast-sensing-plan-barrier-v1",
            "source_receipt_sha256": receipt_digest,
            "protocol_sha256": protocol_digest,
            "objects": plan_records,
            "case_count": 30,
            "measurement_values_revealed": False,
            "new_metrics_computed": False,
            "protected_data_access": False,
        },
    )
    put(
        root / "prediction_barrier.json",
        {
            "schema": "deform-forecast-sensing-prediction-barrier-v1",
            "source_revision": receipt["revision"],
            "source_receipt_sha256": receipt_digest,
            "protocol_sha256": protocol_digest,
            "preflight_sha256": file_digest(root / "preflight.json"),
            "query_plan_barrier_sha256": file_digest(root / "query_plan_barrier.json"),
            "objects": completed,
            "ordinary_success": 30,
            "analysis_case_count": 29,
            "retained_technical_failure": 0,
            "unsealable": 0,
            "new_metrics_computed": False,
            "protected_data_access": False,
            "no_replacement": True,
        },
    )
    return protocol, parent, receipt, receipt_digest


def test_complete_synthetic_barrier_validates_and_rejects_missing_arm(
    runner, tmp_path: Path
) -> None:
    protocol, parent, receipt, digest = _synthetic_barrier(runner, tmp_path)
    assert (
        runner.validate_barrier(tmp_path, protocol, parent, receipt, digest)[
            "ordinary_success"
        ]
        == 30
    )
    seal_path = tmp_path / "DLO1/prediction_seal.json"
    seal = json.loads(seal_path.read_text())
    seal["clean_arms"].remove("forecast_8")
    seal_path.write_text(json.dumps(seal, sort_keys=True))
    barrier_path = tmp_path / "prediction_barrier.json"
    barrier = json.loads(barrier_path.read_text())
    barrier["objects"]["DLO1"]["seal_sha256"] = file_digest(seal_path)
    barrier_path.write_text(json.dumps(barrier, sort_keys=True))
    with pytest.raises(ValueError, match="required clean"):
        runner.validate_barrier(tmp_path, protocol, parent, receipt, digest)


def test_registered_incumbent_is_checked_not_only_self_consistency(
    runner, tmp_path: Path
) -> None:
    points = np.zeros((2, 170, 12, 3), dtype=np.float32)
    source_path = tmp_path / "incumbent.npz"
    record = runner.save_arrays(
        source_path, {"names": np.array(["a", "b"]), "candidate": points}
    )
    item = {
        "names": ["a", "b"],
        "archive": {
            "path": str(source_path),
            "sha256": record["sha256"],
            "incumbent_key": "candidate",
        },
    }
    runner.verify_incumbent_source(item, points)
    with pytest.raises(ValueError, match="registered incumbent"):
        runner.verify_incumbent_source(item, points.astype(np.float64))
    with pytest.raises(ValueError, match="registered incumbent"):
        runner.verify_incumbent_source(item, points + 0.01)


@pytest.mark.parametrize("seed", [4, 7, 11, 260831])
def test_independent_information_form_verifier_agrees(seed: int) -> None:
    path = ROOT / "scripts/verify_deform_forecast_aware_sensing.py"
    spec = importlib.util.spec_from_file_location("sensing_verifier_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(seed)
    design = rng.normal(0, 0.01, (16, 3, 27))
    target = rng.normal(0, 0.02, (60, 27))
    weight = target.T @ target
    for budget in (4, 8, 12, 16):
        assert module.independent_plan(design, weight, budget) == greedy_schedule(
            design, weight, budget, 0.001
        )


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_native_verifier_owns_contiguous_causal_inputs(layout: str) -> None:
    path = ROOT / "scripts/verify_deform_forecast_aware_sensing.py"
    spec = importlib.util.spec_from_file_location("sensing_verifier_layout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = np.arange(3 * 500 * 12 * 3, dtype=np.float64).reshape(3, 500, 12, 3)
    if layout == "fortran":
        raw = np.asfortranarray(raw)
    elif layout == "strided":
        raw = raw[:, :, ::-1]
    original = raw.copy()
    clamps = (0, 1, 10, 11)
    initial, actions = module.native_replay_inputs(raw, clamps)
    assert initial.flags.c_contiguous and actions.flags.c_contiguous
    assert not np.shares_memory(initial, raw)
    assert not np.shares_memory(actions, raw)
    np.testing.assert_array_equal(initial, raw[:, :2])
    np.testing.assert_array_equal(actions, raw[:, 2:172, clamps])
    np.testing.assert_array_equal(raw, original)
