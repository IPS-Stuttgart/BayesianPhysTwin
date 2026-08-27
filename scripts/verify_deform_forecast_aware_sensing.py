#!/usr/bin/env python3
"""Independent batch inference, schedule, temporal-control, and metric checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]


def batch_posterior(
    design: np.ndarray, values: np.ndarray, variance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    matrix = design.reshape(-1, design.shape[-1])
    precision = np.eye(matrix.shape[1]) + matrix.T @ matrix / variance
    covariance = np.linalg.solve(precision, np.eye(len(precision)))
    mean = np.linalg.solve(precision, matrix.T @ values.reshape(-1) / variance)
    return mean, covariance


def independent_plan(
    design: np.ndarray, weight: np.ndarray, budget: int
) -> tuple[int, ...]:
    precision = np.eye(design.shape[-1])
    remaining, chosen = list(range(len(design))), []
    for _ in range(budget):
        candidates = [precision + design[i].T @ design[i] / 1e-6 for i in remaining]
        costs = [float(np.trace(np.linalg.solve(p, weight))) for p in candidates]
        index = int(np.argmin(costs))
        precision = candidates[index]
        chosen.append(remaining.pop(index))
    return tuple(sorted(chosen))


def verify_plans(
    model: dict[str, np.ndarray],
    plan: dict[str, Any],
    item: dict[str, Any],
    protocol: dict[str, Any],
    parent: dict[str, Any],
) -> None:
    frames, nodes = protocol["sensing"]["query_frames"], parent["observed_nodes"]
    pairs = [(t, n) for t in frames for n in nodes]
    assert plan["query_pairs"] == [list(p) for p in pairs]
    response = model["response"]
    for case in range(len(item["names"])):
        design = np.stack([response[case, t - 25, n] for t, n in pairs])
        design = np.concatenate(
            (design, np.broadcast_to(0.005 * np.eye(3), (16, 3, 3))), axis=-1
        )
        np.testing.assert_array_equal(design, model["query_design"][case])
        future_rows = response[case, 25:][:, parent["hidden_nodes"]].reshape(-1, 24)
        free = [i for i in range(item["node_count"]) if i not in item["clamped_nodes"]]
        current_rows = response[case, 24, free].reshape(-1, 24)
        weights = []
        for rows, key in (
            (future_rows, "future_objective"),
            (current_rows, "current_objective"),
        ):
            weight = np.zeros((27, 27))
            weight[:24, :24] = 3 * rows.T @ rows / len(rows)
            np.testing.assert_allclose(weight, model[key][case], rtol=1e-12, atol=1e-14)
            weights.append(weight)
        for budget in protocol["sensing"]["budgets"]:
            assert plan["schedules"][f"uniform_{budget}"][case] == list(
                range(16 - budget, 16)
            )
            expected = independent_plan(design, weights[0], budget)
            assert tuple(plan["schedules"][f"forecast_{budget}"][case]) == expected
        assert tuple(plan["schedules"]["current_8"][case]) == independent_plan(
            design, weights[1], 8
        )
        rng = np.random.default_rng(
            protocol["sensing"]["random_seed"] + item["noise_seed_offset"] + case
        )
        for seed in range(protocol["sensing"]["random_repetitions"]):
            expected_random = sorted(rng.choice(16, size=8, replace=False).tolist())
            assert plan["schedules"][f"random_8_seed{seed}"][case] == expected_random


def expected_temporal(
    base: np.ndarray,
    incumbent: np.ndarray,
    measurements: np.ndarray,
    item: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, np.ndarray]:
    nodes, clamps = parent["observed_nodes"], item["clamped_nodes"]
    residual = (
        measurements[:, -8:].reshape(len(base), 2, 4, 3)
        - incumbent[:, [41, 49]][:, :, nodes]
    )
    knot_ids = sorted(nodes + clamps)
    pose, velocity = np.zeros((2, len(base), item["node_count"], 3))
    for case in range(len(base)):
        for axis in range(3):
            p_values, v_values = [], []
            for k in knot_ids:
                if k in nodes:
                    j = nodes.index(k)
                    p_values.append(residual[case, 1, j, axis])
                    v_values.append(
                        (residual[case, 1, j, axis] - residual[case, 0, j, axis]) / 0.08
                    )
                else:
                    p_values.append(0.0)
                    v_values.append(0.0)
            pose[case, :, axis] = np.interp(
                np.arange(item["node_count"]), knot_ids, p_values
            )
            velocity[case, :, axis] = np.interp(
                np.arange(item["node_count"]), knot_ids, v_values
            )
    horizon = np.arange(1, 121)[None, :, None, None] * 0.01
    result = {
        "temporal_static": base + pose[:, None],
        "temporal_linear": base + pose[:, None] + horizon * velocity[:, None],
    }
    for tau in (0.1, 0.3, 1.0):
        suffix = f"{int(round(tau * 1000))}ms"
        decay = np.exp(-horizon / tau)
        result[f"temporal_decay_{suffix}"] = base + decay * (
            pose[:, None] + horizon * velocity[:, None]
        )
        result[f"temporal_damped_velocity_{suffix}"] = (
            base + pose[:, None] + tau * (1 - decay) * velocity[:, None]
        )
    return result


def independent_basis(item: dict[str, Any], parent: dict[str, Any]) -> np.ndarray:
    nodes = parent["observed_nodes"]
    knots = sorted(nodes + item["clamped_nodes"])
    basis = np.zeros((12, item["node_count"], 3))
    for node_index, node in enumerate(nodes):
        values = [float(k == node) for k in knots]
        profile = np.interp(np.arange(item["node_count"]), knots, values)
        for axis in range(3):
            basis[3 * node_index + axis, :, axis] = profile
    return basis


def verify_fits(
    fits: dict[str, np.ndarray],
    model: dict[str, np.ndarray],
    plan: dict[str, Any],
    measurements: np.ndarray,
    item: dict[str, Any],
    parent: dict[str, Any],
    repetition: int | None,
) -> None:
    pairs = plan["query_pairs"]
    reference = np.stack([model["incumbent"][:, t, n] for t, n in pairs], axis=1)
    arms = [
        k.removesuffix("__coefficients") for k in fits if k.endswith("__coefficients")
    ]
    basis = independent_basis(item, parent)
    for arm in arms:

        def selected(key: str, arm: str = arm) -> np.ndarray:
            value = fits[f"{arm}__{key}"]
            return value if repetition is None else value[repetition]

        for case, schedule in enumerate(plan["schedules"][arm]):
            values = measurements[case, schedule]
            np.testing.assert_array_equal(selected("observed_positions")[case], values)
            design = model["query_design"][case, schedule]
            residual = values - reference[case, schedule]
            mean, covariance = batch_posterior(design, residual)
            np.testing.assert_allclose(
                mean, selected("coefficients")[case], rtol=2e-8, atol=2e-8
            )
            np.testing.assert_allclose(
                covariance, selected("covariance")[case], rtol=2e-8, atol=2e-9
            )
            assert np.linalg.eigvalsh(selected("covariance")[case]).min() > -1e-10
            pose = 0.01 * np.tensordot(mean[:12], basis, axes=1)
            velocity = 0.1 * np.tensordot(mean[12:24], basis, axes=1)
            gain = 1 / max(
                1,
                np.linalg.norm(pose, axis=-1).max() / 0.03,
                np.linalg.norm(velocity, axis=-1).max() / 0.3,
            )
            np.testing.assert_allclose(
                gain, selected("gain")[case], rtol=2e-8, atol=1e-9
            )


def native_replay_inputs(
    raw: np.ndarray, clamped_nodes: list[int] | tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    # The upstream .view consumer requires owned C-order action buffers.
    initial = raw[:, :2].copy(order="C")
    actions = raw[:, 2:172, clamped_nodes].copy(order="C")
    return initial, actions


def verify_native_primary(
    model: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    fits: dict[str, np.ndarray],
    raw: np.ndarray,
    item: dict[str, Any],
    parent: dict[str, Any],
    modules: Any,
    torch: Any,
) -> None:
    import run_deform_multiobject_state_restart as multi

    from bayesian_phystwin_experiments.deform_multiobject_restart import (
        config_for_object,
    )
    from bayesian_phystwin_experiments.deform_state_restart import RodState

    config = config_for_object(parent, item)
    initial, actions = native_replay_inputs(raw, item["clamped_nodes"])
    checkpoint = torch.load(
        item["checkpoint"]["path"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    rod = multi.MultiObjectRod(modules, torch, checkpoint, config, item["object"])
    _, anchor = rod.rollout(rod.initialize(initial), actions[:, :26])
    basis = independent_basis(item, parent)
    for arm in ("uniform_8", "forecast_8"):
        means = fits[f"{arm}__coefficients"]
        gains = fits[f"{arm}__gain"]
        pose = (
            np.einsum("bk,knc->bnc", means[:, :12], basis) * 0.01 * gains[:, None, None]
        )
        velocity = (
            np.einsum("bk,knc->bnc", means[:, 12:24], basis)
            * 0.1
            * gains[:, None, None]
        )
        state = RodState(
            anchor.positions + torch.tensor(pose, dtype=torch.float32),
            anchor.velocity + torch.tensor(velocity, dtype=torch.float32),
            anchor.previous_positions.clone(),
            anchor.material_u0.clone(),
            anchor.theta.clone(),
            anchor.prediction_index,
        )
        future, _ = rod.rollout(state, actions[:, 26:])
        expected = (
            model["incumbent"][:, 50:]
            + future[:, 24:].astype(np.float64)
            - model["nominal_from_anchor"][:, 25:]
        )
        np.testing.assert_allclose(expected, predictions[arm], atol=1e-12, rtol=1e-12)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    prediction_root = getattr(args, "prediction_source_root", None) or ROOT
    sys.path.insert(0, str(prediction_root / "scripts/remote"))
    import run_deform_dlo_source as source
    import run_deform_forecast_aware_sensing as runner
    import verify_deform_multiobject_state_restart as metric_verifier

    receipt = runner.multi.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    protocol, parent = runner.load_protocol(
        prediction_root / runner.PROTOCOL, prediction_root
    )
    barrier = runner.validate_barrier(
        args.run, protocol, parent, receipt, file_digest(args.source_receipt)
    )
    result = json.loads((args.run / "result.json").read_text())
    assert result["prediction_barrier_sha256"] == file_digest(
        args.run / "prediction_barrier.json"
    )
    assert (
        result["source_revision"] == receipt["revision"] == barrier["source_revision"]
    )
    assert result["protected_data_access"] is False
    assert result["original_results_modified"] is False
    modules: Any = None
    torch: Any = None
    if args.native_replay:
        import torch as torch_module

        torch = torch_module
        torch.set_num_threads(1)
        modules = source._load_upstream(Path(parent["upstream_root"]))
    verified = 0
    native_count = 0
    for item in parent["objects"]:
        manifest = runner.multi.verify_input_files(item)
        data = source._load_named_trajectories(
            manifest, item["names"], frame_count=500, node_count=item["node_count"]
        )
        raw = np.stack([data[n] for n in item["names"]])
        directory = args.run / item["object"]
        seal = json.loads((directory / "prediction_seal.json").read_text())
        model = runner.verified_arrays(
            directory / "model.npz", seal["files"]["model.npz"]
        )
        plan = json.loads((directory / "plans.json").read_text())
        verify_plans(model, plan, item, protocol, parent)
        clean_points = np.stack(
            [raw[:, t + 2, n] for t, n in plan["query_pairs"]], axis=1
        )
        base = model["incumbent"][:, 50:]
        keep = [
            i
            for i, name in enumerate(item["names"])
            if name != item["excluded_design_case"]
        ]
        truth = raw[:, 52:172]
        for condition_index, condition in enumerate(
            ("clean", *protocol["noise"]["conditions"])
        ):
            arrays = runner.verified_arrays(
                directory / f"{condition}.npz", seal["files"][f"{condition}.npz"]
            )
            predictions = {k: v for k, v in arrays.items() if k != "names"}
            fits = runner.verified_arrays(
                directory / f"fits_{condition}.npz",
                seal["files"][f"fits_{condition}.npz"],
            )
            repeats = 1 if condition == "clean" else protocol["noise"]["repetitions"]
            for repetition in range(repeats):
                measurements = clean_points.copy()
                if condition != "clean":
                    rng = np.random.default_rng(
                        protocol["noise"]["seed"]
                        + item["noise_seed_offset"]
                        + repetition
                    )
                    error = rng.normal(0, 0.001, clean_points.shape)
                    shared = rng.normal(0, 0.005, (len(clean_points), 1, 3))
                    measurements += error + (shared if condition_index == 2 else 0)
                verify_fits(
                    fits,
                    model,
                    plan,
                    measurements,
                    item,
                    parent,
                    None if condition == "clean" else repetition,
                )
                controls = expected_temporal(
                    base, model["incumbent"], measurements, item, parent
                )
                for arm, expected in controls.items():
                    actual = (
                        predictions[arm]
                        if condition == "clean"
                        else predictions[arm][repetition]
                    )
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            verified += metric_verifier.verify_rows(
                predictions,
                truth,
                result["objects"][item["object"]][condition],
                keep,
                parent["hidden_nodes"],
                parent,
            )
            if condition == "clean":
                assert array_digest(predictions["uniform_16"]) == array_digest(
                    predictions["forecast_16"]
                )
                parent_arrays = runner.parent_prediction(item, protocol)
                assert array_digest(predictions["previous_paired_8"]) == array_digest(
                    parent_arrays["incumbent_propagated_pose_velocity"]
                )
                if args.native_replay:
                    with torch.no_grad():
                        verify_native_primary(
                            model, predictions, fits, raw, item, parent, modules, torch
                        )
                    native_count += 2 * len(item["names"])
        print(
            json.dumps(
                {"verified_object": item["object"], "forecasts_so_far": verified}
            ),
            flush=True,
        )
    checks = {}
    for name in ("DLO1", "DLO3"):
        summaries = result["objects"][name]["clean"]["summaries"]
        base, candidate, uniform = [
            summaries[k] for k in ("incumbent", "forecast_8", "uniform_8")
        ]
        metrics = ("coordinate_l1_mm", "point_rmse_mm")
        temporal = [v for k, v in summaries.items() if k.startswith("temporal_")]
        checks[name] = {
            "both_metrics_improve_over_incumbent": all(
                candidate[k] < base[k] for k in metrics
            ),
            "both_metrics_improve_over_uniform": all(
                candidate[k] < uniform[k] for k in metrics
            ),
            "at_least_2percent_rmse_gain_over_uniform": candidate[metrics[1]]
            <= 0.98 * uniform[metrics[1]],
            "beats_all_frozen_temporal_controls_on_both": all(
                candidate[k] < t[k] for t in temporal for k in metrics
            ),
            "late_rmse_nonincreasing_over_incumbent": candidate["late"][metrics[1]]
            <= base["late"][metrics[1]],
            "at_least_5_of_8_joint_wins": candidate["joint_wins"] >= 5,
        }
    assert checks == result["decision"]["checks"]
    assert (
        all(all(v.values()) for v in checks.values())
        == result["decision"]["development_advancement_gate_passed"]
    )
    assert result["decision"]["automatic_target_authorization"] is False
    assert result["decision"]["incumbent_promoted"] is False
    for label, objects in (
        ("transfer_only", ("DLO1", "DLO3")),
        ("all_three_including_discovery", ("DLO1", "DLO2", "DLO3")),
    ):
        for condition, arms in result["aggregate"][label].items():
            for arm, summary in arms.items():
                rows = [
                    result["objects"][o][condition]["summaries"][arm] for o in objects
                ]
                for metric in ("coordinate_l1", "point_rmse"):
                    np.testing.assert_allclose(
                        summary[metric + "_mean_object_change_percent"],
                        np.mean([r[metric + "_mm_change_percent"] for r in rows]),
                        atol=1e-12,
                    )
                assert summary["joint_wins"] == sum(r["joint_wins"] for r in rows)
                assert summary["case_count"] == sum(r["case_count"] for r in rows)
    record = {
        "schema": "deform-forecast-sensing-independent-verification-v1",
        "passed": True,
        "source_revision": receipt["revision"],
        "source_receipt_sha256": file_digest(args.source_receipt),
        "verifier_file_sha256": file_digest(Path(__file__).resolve()),
        "prediction_source_root": str(prediction_root.resolve()),
        "prediction_barrier_sha256": file_digest(args.run / "prediction_barrier.json"),
        "result_sha256": file_digest(args.run / "result.json"),
        "forecasts_metric_recomputed": verified,
        "native_primary_forecasts_replayed": native_count,
        "independent_batch_posterior": True,
        "independent_information_form_planning": True,
        "independent_temporal_controls": True,
        "protected_data_access": False,
        "independent_analysis_code_not_independent_person": True,
    }
    write_json_once(args.output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-replay", action="store_true")
    parser.add_argument(
        "--prediction-source-root",
        type=Path,
        help="Verify an untouched prediction checkout with a separate analysis script.",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
