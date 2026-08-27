#!/usr/bin/env python3
"""Source-locked public-data development test of forecast-oriented sparse queries."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_multiobject_state_restart as multi

from bayesian_phystwin_experiments.deform_forecast_sensing import (
    LockedQueryBank,
    SensingConfig,
    bounded_increments,
    clean_arm_names,
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
from bayesian_phystwin_experiments.deform_multiobject_restart import (
    config_for_object,
    summarize_predictions,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    RodState,
    array_digest,
    file_digest,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_forecast_aware_sensing_v1.json"


def freeze(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("freeze requires a committed clean experiment")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "configs/sota/deform_*",
            "scripts/remote/run_deform*",
            "scripts/verify_deform*",
            "tests/test_deform*",
            "docs/deform_*",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deform-state-restart-source-receipt-v1",
            "experiment": "deform-forecast-aware-sensing-v1",
            "revision": revision,
            "git_clean": True,
            "files": {p: file_digest(ROOT / p) for p in paths},
            "new_outcomes_read": False,
            "protected_data_access": False,
        },
    )
    print(
        json.dumps({"source_revision": revision, "bound_files": len(paths)}), flush=True
    )


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    for value in arrays.values():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(
                "nonfinite arrays cannot be sealed as successful forecasts"
            )
    payload: dict[str, Any] = dict(arrays)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return {
        "sha256": file_digest(path),
        "arrays": {k: array_digest(v) for k, v in arrays.items()},
    }


def verified_arrays(path: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    if file_digest(path) != record["sha256"]:
        raise ValueError("sealed NPZ file changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(record["arrays"]):
            raise ValueError("sealed NPZ member set changed")
        values = {k: archive[k].copy() for k in archive.files}
    if any(array_digest(v) != record["arrays"][k] for k, v in values.items()):
        raise ValueError("sealed NPZ array identity changed")
    if any(v.dtype.kind in "fc" and not np.isfinite(v).all() for v in values.values()):
        raise ValueError("nonfinite sealed arrays cannot pass the prediction barrier")
    return values


def verify_incumbent_source(item: dict[str, Any], incumbent: np.ndarray) -> None:
    spec = item["archive"]
    if file_digest(Path(spec["path"])) != spec["sha256"]:
        raise ValueError("registered incumbent source archive changed")
    with np.load(spec["path"], allow_pickle=False) as archive:
        original = archive[spec["incumbent_key"]][:, :170]
        if archive["names"].tolist() != item["names"] or array_digest(
            original
        ) != array_digest(incumbent):
            raise ValueError("model carrier does not contain the registered incumbent")


def parent_prediction(
    item: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, np.ndarray]:
    root = Path(protocol["parent_prediction_root"])
    if (
        file_digest(root / "prediction_barrier.json")
        != protocol["parent_prediction_barrier_sha256"]
    ):
        raise ValueError("parent prediction barrier changed")
    barrier = json.loads((root / "prediction_barrier.json").read_text())
    directory = root / item["object"]
    if (
        file_digest(directory / "prediction_seal.json")
        != barrier["objects"][item["object"]]["seal_sha256"]
    ):
        raise ValueError("parent object seal changed")
    seal = json.loads((directory / "prediction_seal.json").read_text())
    if seal["names"] != item["names"]:
        raise ValueError("parent identity order changed")
    return verified_arrays(directory / "clean.npz", seal["files"]["clean"])


@dataclass
class PreparedObject:
    config: RestartConfig
    rod: Any
    anchor: RodState
    endpoint: RodState
    actions: np.ndarray
    incumbent: np.ndarray
    nominal: np.ndarray
    response: np.ndarray
    plans: dict[str, np.ndarray]
    design: np.ndarray
    old_predictions: dict[str, np.ndarray]
    model_record: dict[str, Any]
    controls: dict[str, Any]


def prepare_object(
    item: dict[str, Any],
    protocol: dict[str, Any],
    parent: dict[str, Any],
    manifest: dict[str, Any],
    modules: Any,
    torch: Any,
    output: Path,
) -> PreparedObject:
    import run_deform_dlo_source as source

    sensing = SensingConfig()
    config = config_for_object(parent, item)
    output.mkdir(exist_ok=False)
    raw_map = source._load_named_trajectories(
        manifest, item["names"], frame_count=500, node_count=config.node_count
    )
    initial, actions = multi.native.causal_model_inputs(
        np.stack([raw_map[n] for n in item["names"]]), config
    )
    # No free-node prefix or future measurements are retained by model preparation.
    del raw_map
    with np.load(item["archive"]["path"], allow_pickle=False) as archive:
        if archive["names"].tolist() != item["names"]:
            raise ValueError("incumbent identity order differs")
        incumbent = archive[item["archive"]["incumbent_key"]][
            :, : config.forecast_end
        ].copy()
        archived_physical = archive[item["archive"]["physical_key"]][
            :, : config.forecast_end
        ].copy()
    old = parent_prediction(item, protocol)
    checkpoint = torch.load(
        item["checkpoint"]["path"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    rod = multi.MultiObjectRod(modules, torch, checkpoint, config, item["object"])
    prefix, anchor = rod.rollout(
        rod.initialize(initial), actions[:, : sensing.anchor_frame + 1]
    )
    remainder, endpoint = rod.rollout(
        anchor.clone(), actions[:, sensing.anchor_frame + 1 : config.prefix_length]
    )
    future, _ = rod.rollout(endpoint.clone(), actions[:, config.prefix_length :])
    full = np.concatenate((prefix, remainder, future), axis=1)
    nominal = full[:, sensing.anchor_frame :]
    zero = torch.zeros_like(anchor.positions)
    unchanged = update_rod_state(
        anchor, zero, zero, gain=0, clamped_nodes=config.clamped_nodes
    )
    replay, _ = rod.rollout(unchanged, actions[:, sensing.anchor_frame + 1 :])
    zero_future = replay[:, config.prefix_length - sensing.anchor_frame - 1 :]
    base = incumbent[:, config.prefix_length :]
    controls = {
        "parent_incumbent_byte_identical": array_digest(base)
        == array_digest(old["incumbent"]),
        "parent_native_future_byte_identical": array_digest(future)
        == array_digest(old["physical_nominal"]),
        "zero_native_update_byte_identical": array_digest(future)
        == array_digest(zero_future),
        "zero_readout_returns_original_object": paired_physical_readout(
            base, future, zero_future
        )
        is base,
        "archived_gpu_max_error_m": float(
            np.max(np.abs(full.astype(np.float64) - archived_physical))
        ),
        "source_model_count": 1,
    }
    if (
        not all(
            controls[k]
            for k in (
                "parent_incumbent_byte_identical",
                "parent_native_future_byte_identical",
                "zero_native_update_byte_identical",
                "zero_readout_returns_original_object",
            )
        )
        or controls["archived_gpu_max_error_m"]
        > parent["controls"]["archived_gpu_replay_max_error_m"]
    ):
        raise ValueError(
            "native or incumbent parity failed before any measurement reveal"
        )
    basis = material_basis(config)
    rank = 2 * len(basis)
    response = np.empty((*nominal.shape, rank), dtype=np.float64)
    asymmetry = []
    for mode in range(rank):
        pose_mode = mode < len(basis)
        scale = sensing.position_std_m if pose_mode else sensing.velocity_std_m_s
        increment = np.broadcast_to(
            basis[mode % len(basis)] * scale * sensing.finite_difference_fraction,
            (len(item["names"]), config.node_count, 3),
        ).copy()
        values = []
        for sign in (-1, 1):
            delta = torch.tensor(sign * increment, dtype=torch.float32)
            perturbed = update_rod_state(
                anchor,
                delta if pose_mode else zero,
                zero if pose_mode else delta,
                gain=1,
                clamped_nodes=config.clamped_nodes,
            )
            subsequent, _ = rod.rollout(
                perturbed, actions[:, sensing.anchor_frame + 1 :]
            )
            values.append(
                np.concatenate(
                    (perturbed.positions.detach().cpu().numpy()[:, None], subsequent),
                    axis=1,
                ).astype(np.float64)
            )
        response[..., mode] = (values[1] - values[0]) / (
            2 * sensing.finite_difference_fraction
        )
        asymmetry.append(float(np.max(np.abs(0.5 * (values[1] + values[0]) - nominal))))
        if (mode + 1) % 4 == 0:
            print(
                json.dumps(
                    {
                        "stage": "native-response",
                        "object": item["object"],
                        "completed_modes": mode + 1,
                        "mode_count": rank,
                    }
                ),
                flush=True,
            )
    if np.any(response[:, :, config.clamped_nodes]):
        raise ValueError("a physical response modified a prescribed clamp")
    plans: dict[str, list[tuple[int, ...]]] = {}
    designs, future_weights, current_weights = [], [], []
    for case in range(len(item["names"])):
        schedule, design, future_weight = schedules_for_case(
            response[case],
            config,
            sensing,
            seed=sensing.random_seed + item["noise_seed_offset"] + case,
        )
        designs.append(design)
        future_weights.append(future_weight)
        current_weights.append(planning_matrices(response[case], config, sensing)[2])
        for key, indices in schedule.items():
            plans.setdefault(key, []).append(indices)
    arrays = {
        "incumbent": incumbent,
        "nominal_from_anchor": nominal,
        "response": response,
        "query_design": np.stack(designs),
        "future_objective": np.stack(future_weights),
        "current_objective": np.stack(current_weights),
        "names": np.asarray(item["names"]),
    }
    model_record = save_arrays(output / "model.npz", arrays)
    plan_arrays = {
        key: np.asarray(value, dtype=np.int64) for key, value in plans.items()
    }
    pairs = query_pairs(config, sensing)
    write_json_once(
        output / "plans.json",
        {
            "schema": "deform-forecast-sensing-query-plans-v1",
            "object": item["object"],
            "names": item["names"],
            "model_sha256": model_record["sha256"],
            "query_pairs": [list(p) for p in pairs],
            "schedules": {k: v.tolist() for k, v in plan_arrays.items()},
            "measurement_values_read": False,
            "future_truth_read": False,
            "schedules_are_preplanned_not_measurement_adaptive": True,
        },
    )
    controls["maximum_symmetric_response_asymmetry_m"] = max(asymmetry)
    controls["response_asymmetry_is_diagnostic_not_selection"] = True
    controls["clamp_response_exactly_zero"] = True
    controls["full_budget_schedule_identical"] = np.array_equal(
        plan_arrays["uniform_16"], plan_arrays["forecast_16"]
    )
    write_json_once(output / "controls.json", controls)
    return PreparedObject(
        config,
        rod,
        anchor.clone(),
        endpoint.clone(),
        actions,
        incumbent,
        nominal,
        response,
        plan_arrays,
        np.stack(designs),
        old,
        model_record,
        controls,
    )


def native_prediction(
    prepared: PreparedObject, coefficients: np.ndarray, torch: Any
) -> tuple[np.ndarray, np.ndarray]:
    sensing, rod = SensingConfig(), prepared.config
    pose, velocity, gain = bounded_increments(coefficients, rod, sensing)
    state = update_rod_state(
        prepared.anchor,
        torch.tensor(pose, dtype=torch.float32),
        torch.tensor(velocity, dtype=torch.float32),
        gain=1,
        clamped_nodes=rod.clamped_nodes,
    )
    continuation, _ = prepared.rod.rollout(
        state, prepared.actions[:, sensing.anchor_frame + 1 :]
    )
    points = continuation[:, rod.prefix_length - sensing.anchor_frame - 1 :]
    base = prepared.incumbent[:, rod.prefix_length :]
    nominal = prepared.nominal[:, rod.prefix_length - sensing.anchor_frame :]
    return paired_physical_readout(base, nominal, points), gain


def infer_arm(
    prepared: PreparedObject, points: np.ndarray, arm: str, torch: Any
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sensing = SensingConfig()
    pairs = query_pairs(prepared.config, sensing)
    reference = np.stack([prepared.incumbent[:, t, n] for t, n in pairs], axis=1)
    means, covariances, observations = [], [], []
    for case, schedule in enumerate(prepared.plans[arm]):
        bank = LockedQueryBank(points[case], pairs, schedule)
        mean, covariance = infer_coefficients(
            prepared.design[case],
            reference[case],
            bank,
            pairs,
            schedule,
            sensing.measurement_std_m,
        )
        expected = [pairs[i] for i in schedule]
        if bank.access_log != expected:
            raise ValueError("empirical query access differs from the sealed plan")
        means.append(mean)
        covariances.append(covariance)
        observations.append(points[case, schedule])
    coefficients = np.stack(means)
    prediction, gains = native_prediction(prepared, coefficients, torch)
    return prediction, {
        "coefficients": coefficients,
        "covariance": np.stack(covariances),
        "gain": gains,
        "observed_positions": np.stack(observations),
    }


def original_observations(
    prepared: PreparedObject, bank_points: np.ndarray
) -> np.ndarray:
    pairs = query_pairs(prepared.config, SensingConfig())
    indices = list(range(len(pairs) - 8, len(pairs)))
    result = []
    for case in range(len(bank_points)):
        bank = LockedQueryBank(bank_points[case], pairs, indices)
        result.append(np.stack([bank.reveal(*pairs[i]) for i in indices]))
    return np.stack(result).reshape(len(bank_points), 2, 4, 3)


def previous_prediction(
    prepared: PreparedObject, observed: np.ndarray, torch: Any
) -> np.ndarray:
    config = prepared.config
    pose, velocity = sparse_state_increments(
        prepared.incumbent[:, : config.prefix_length], observed, config
    )
    state = update_rod_state(
        prepared.endpoint,
        torch.tensor(pose, dtype=torch.float32),
        torch.tensor(velocity, dtype=torch.float32),
        gain=1,
        clamped_nodes=config.clamped_nodes,
    )
    points, _ = prepared.rod.rollout(state, prepared.actions[:, config.prefix_length :])
    nominal = prepared.nominal[:, config.prefix_length - SensingConfig().anchor_frame :]
    return paired_physical_readout(
        prepared.incumbent[:, config.prefix_length :], nominal, points
    )


def predict_object(
    prepared: PreparedObject,
    item: dict[str, Any],
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    torch: Any,
    output: Path,
) -> dict[str, Any]:
    import run_deform_dlo_source as source

    config, sensing = prepared.config, SensingConfig()
    raw = source._load_named_trajectories(
        manifest, item["names"], frame_count=500, node_count=config.node_count
    )
    pairs = query_pairs(config, sensing)
    points = np.stack(
        [np.stack([raw[name][t + 2, n] for t, n in pairs]) for name in item["names"]]
    )
    del raw
    observed = original_observations(prepared, points)
    predictions = temporal_controls(prepared.incumbent, observed, config, sensing)
    previous = previous_prediction(prepared, observed, torch)
    if array_digest(previous) != array_digest(
        prepared.old_predictions["incumbent_propagated_pose_velocity"]
    ):
        raise ValueError("the previous positive paired forecast changed")
    predictions["previous_paired_8"] = previous
    fits: dict[str, np.ndarray] = {}
    for arm in prepared.plans:
        if arm == "forecast_16":
            predictions[arm] = predictions["uniform_16"]
            for name in ("coefficients", "covariance", "gain", "observed_positions"):
                fits[f"{arm}__{name}"] = fits[f"uniform_16__{name}"]
            continue
        predictions[arm], fit = infer_arm(prepared, points, arm, torch)
        fits.update({f"{arm}__{k}": v for k, v in fit.items()})
        print(
            json.dumps(
                {"stage": "clean-prediction", "object": item["object"], "arm": arm}
            ),
            flush=True,
        )
    clean_arms = list(predictions)
    if tuple(clean_arms) != clean_arm_names(sensing):
        raise ValueError("required clean arm set changed before sealing")
    files = {
        "model.npz": prepared.model_record,
        "clean.npz": save_arrays(
            output / "clean.npz", {"names": np.asarray(item["names"]), **predictions}
        ),
        "fits_clean.npz": save_arrays(output / "fits_clean.npz", fits),
    }
    del predictions, fits
    for condition_index, condition in enumerate(protocol["noise"]["conditions"]):
        draws: dict[str, list[np.ndarray]] = {}
        fit_draws: dict[str, list[np.ndarray]] = {}
        for repetition in range(protocol["noise"]["repetitions"]):
            noisy = points + query_noise(
                points.shape,
                seed=protocol["noise"]["seed"] + item["noise_seed_offset"] + repetition,
                shared=bool(condition_index),
            )
            sparse = original_observations(prepared, noisy)
            forecasts = temporal_controls(prepared.incumbent, sparse, config, sensing)
            forecasts["previous_paired_8"] = previous_prediction(
                prepared, sparse, torch
            )
            for arm in ("uniform_8", "forecast_8"):
                forecasts[arm], fit = infer_arm(prepared, noisy, arm, torch)
                for key, value in fit.items():
                    fit_draws.setdefault(f"{arm}__{key}", []).append(value)
            for key, value in forecasts.items():
                draws.setdefault(key, []).append(value)
            if tuple(forecasts) != noise_arm_names(sensing):
                raise ValueError("required noisy arm set changed before sealing")
            print(
                json.dumps(
                    {
                        "stage": "noise-prediction",
                        "object": item["object"],
                        "condition": condition,
                        "repetition": repetition + 1,
                    }
                ),
                flush=True,
            )
        filename = f"{condition}.npz"
        files[filename] = save_arrays(
            output / filename,
            {
                "names": np.asarray(item["names"]),
                **{k: np.stack(v) for k, v in draws.items()},
            },
        )
        fitname = f"fits_{condition}.npz"
        files[fitname] = save_arrays(
            output / fitname, {k: np.stack(v) for k, v in fit_draws.items()}
        )
    seal = {
        "schema": "deform-forecast-sensing-object-seal-v1",
        "object": item["object"],
        "names": item["names"],
        "case_count": len(item["names"]),
        "files": files,
        "plans_sha256": file_digest(output / "plans.json"),
        "controls_sha256": file_digest(output / "controls.json"),
        "clean_arms": clean_arms,
        "previous_paired_prediction_byte_identical": True,
        "future_free_node_truth_used": False,
        "new_metrics_computed": False,
        "protected_data_access": False,
    }
    write_json_once(output / "prediction_seal.json", seal)
    return {
        "seal_sha256": file_digest(output / "prediction_seal.json"),
        "ordinary_success": len(item["names"]),
    }


def predict(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    import run_deform_dlo_source as source
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    stage, completed = "runtime-preflight", {}
    try:
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or str(torch.__version__) != parent["runtime"]["torch"]
        ):
            raise ValueError("the frozen CPU-only runtime is required")
        torch.set_num_threads(1)
        source._seed_everything(torch, parent["bootstrap_seed"])
        upstream = source._assert_upstream(
            Path(parent["upstream_root"]), parent["upstream_commit"]
        )
        manifests = {
            item["object"]: multi.verify_input_files(item) for item in parent["objects"]
        }
        modules = source._load_upstream(Path(parent["upstream_root"]))
        write_json_once(
            args.output / "preflight.json",
            {
                "schema": "deform-forecast-sensing-preflight-v1",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "upstream": upstream,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "threads": 1,
                },
                "prediction_case_count": parent["prediction_case_count"],
                "analysis_case_count": parent["analysis_case_count"],
                "protected_data_access": False,
                "new_metrics_computed": False,
            },
        )
        prepared: dict[str, PreparedObject] = {}
        with torch.no_grad():
            for item in parent["objects"]:
                stage = "prepare-model-and-plan-" + item["object"]
                prepared[item["object"]] = prepare_object(
                    item,
                    protocol,
                    parent,
                    manifests[item["object"]],
                    modules,
                    torch,
                    args.output / item["object"],
                )
            write_json_once(
                args.output / "query_plan_barrier.json",
                {
                    "schema": "deform-forecast-sensing-plan-barrier-v1",
                    "source_receipt_sha256": file_digest(args.source_receipt),
                    "protocol_sha256": file_digest(ROOT / PROTOCOL),
                    "objects": {
                        item["object"]: {
                            "plans_sha256": file_digest(
                                args.output / item["object"] / "plans.json"
                            ),
                            "model_sha256": prepared[item["object"]].model_record[
                                "sha256"
                            ],
                        }
                        for item in parent["objects"]
                    },
                    "case_count": parent["prediction_case_count"],
                    "measurement_values_revealed": False,
                    "new_metrics_computed": False,
                    "protected_data_access": False,
                },
            )
            for item in parent["objects"]:
                stage = "predict-" + item["object"]
                completed[item["object"]] = predict_object(
                    prepared[item["object"]],
                    item,
                    manifests[item["object"]],
                    protocol,
                    torch,
                    args.output / item["object"],
                )
        write_json_once(
            args.output / "prediction_barrier.json",
            {
                "schema": "deform-forecast-sensing-prediction-barrier-v1",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "preflight_sha256": file_digest(args.output / "preflight.json"),
                "query_plan_barrier_sha256": file_digest(
                    args.output / "query_plan_barrier.json"
                ),
                "objects": completed,
                "ordinary_success": parent["prediction_case_count"],
                "analysis_case_count": parent["analysis_case_count"],
                "retained_technical_failure": 0,
                "unsealable": 0,
                "new_metrics_computed": False,
                "protected_data_access": False,
                "no_replacement": True,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "all-predictions-sealed",
                    "case_count": parent["prediction_case_count"],
                    "new_metrics_computed": False,
                }
            ),
            flush=True,
        )
    except Exception as error:
        write_json_once(
            args.output / "failure.json",
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_objects": completed,
                "no_automatic_retry": True,
                "protected_data_access": False,
            },
        )
        raise


def validate_barrier(
    output: Path,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
    receipt_digest: str,
) -> dict[str, Any]:
    if (output / "failure.json").exists():
        raise ValueError("retained technical failure blocks scoring")
    barrier = json.loads((output / "prediction_barrier.json").read_text())
    required = {
        "schema": "deform-forecast-sensing-prediction-barrier-v1",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": receipt_digest,
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "ordinary_success": parent["prediction_case_count"],
        "analysis_case_count": parent["analysis_case_count"],
        "retained_technical_failure": 0,
        "unsealable": 0,
        "new_metrics_computed": False,
        "protected_data_access": False,
        "no_replacement": True,
    }
    if any(barrier.get(k) != v for k, v in required.items()):
        raise ValueError("incomplete denominator, source identity, or scoring boundary")
    for filename in ("preflight", "query_plan_barrier"):
        if file_digest(output / f"{filename}.json") != barrier[f"{filename}_sha256"]:
            raise ValueError("preflight or pre-measurement plan barrier changed")
    preflight = json.loads((output / "preflight.json").read_text())
    if any(
        preflight.get(k) != v
        for k, v in {
            "schema": "deform-forecast-sensing-preflight-v1",
            "source_revision": receipt["revision"],
            "source_receipt_sha256": receipt_digest,
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "prediction_case_count": parent["prediction_case_count"],
            "analysis_case_count": parent["analysis_case_count"],
            "protected_data_access": False,
            "new_metrics_computed": False,
        }.items()
    ):
        raise ValueError("preflight source or information boundary differs")
    plan_barrier = json.loads((output / "query_plan_barrier.json").read_text())
    if (
        plan_barrier["schema"] != "deform-forecast-sensing-plan-barrier-v1"
        or plan_barrier["measurement_values_revealed"] is not False
        or plan_barrier["new_metrics_computed"] is not False
        or plan_barrier["protected_data_access"] is not False
        or plan_barrier["case_count"] != parent["prediction_case_count"]
        or plan_barrier["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or plan_barrier["source_receipt_sha256"] != receipt_digest
    ):
        raise ValueError("query plan barrier did not precede measurements")
    names = {item["object"] for item in parent["objects"]}
    if set(barrier["objects"]) != names or set(plan_barrier["objects"]) != names:
        raise ValueError("all three objects are required before scoring")
    for item in parent["objects"]:
        directory = output / item["object"]
        record = barrier["objects"][item["object"]]
        if (
            record["ordinary_success"] != len(item["names"])
            or file_digest(directory / "prediction_seal.json") != record["seal_sha256"]
        ):
            raise ValueError("object prediction denominator or seal changed")
        seal = json.loads((directory / "prediction_seal.json").read_text())
        if (
            seal["schema"] != "deform-forecast-sensing-object-seal-v1"
            or seal["object"] != item["object"]
            or seal["names"] != item["names"]
            or seal["case_count"] != len(item["names"])
            or seal["new_metrics_computed"] is not False
            or seal["protected_data_access"] is not False
            or seal["future_free_node_truth_used"] is not False
            or seal["previous_paired_prediction_byte_identical"] is not True
        ):
            raise ValueError("object identity or information boundary changed")
        if (
            file_digest(directory / "controls.json") != seal["controls_sha256"]
            or file_digest(directory / "plans.json") != seal["plans_sha256"]
        ):
            raise ValueError("source controls or query plans changed")
        controls = json.loads((directory / "controls.json").read_text())
        for key in (
            "parent_incumbent_byte_identical",
            "parent_native_future_byte_identical",
            "zero_native_update_byte_identical",
            "zero_readout_returns_original_object",
            "clamp_response_exactly_zero",
            "full_budget_schedule_identical",
        ):
            if controls[key] is not True:
                raise ValueError("a required no-op/native control failed")
        if (
            not 0
            <= controls["archived_gpu_max_error_m"]
            <= parent["controls"]["archived_gpu_replay_max_error_m"]
        ):
            raise ValueError("native model parity exceeds the frozen tolerance")
        plan = json.loads((directory / "plans.json").read_text())
        if (
            plan["measurement_values_read"] is not False
            or plan["future_truth_read"] is not False
            or plan["names"] != item["names"]
            or plan["object"] != item["object"]
        ):
            raise ValueError("plans are not target-free and case-aligned")
        if plan_barrier["objects"][item["object"]] != {
            "plans_sha256": seal["plans_sha256"],
            "model_sha256": seal["files"]["model.npz"]["sha256"],
        }:
            raise ValueError("prediction plans differ from pre-measurement plans")
        expected_files = {"model.npz", "clean.npz", "fits_clean.npz"}
        expected_files.update(f"{c}.npz" for c in protocol["noise"]["conditions"])
        expected_files.update(f"fits_{c}.npz" for c in protocol["noise"]["conditions"])
        if set(seal["files"]) != expected_files:
            raise ValueError("incomplete condition or inference record set")
        arrays = {
            name: verified_arrays(directory / name, spec)
            for name, spec in seal["files"].items()
        }
        config = config_for_object(parent, item)
        sensing = SensingConfig()
        batch, node_count = len(item["names"]), config.node_count
        model = arrays["model.npz"]
        expected_model_shapes = {
            "incumbent": (batch, 170, node_count, 3),
            "nominal_from_anchor": (batch, 145, node_count, 3),
            "response": (batch, 145, node_count, 3, 24),
            "query_design": (batch, 16, 3, 27),
            "future_objective": (batch, 27, 27),
            "current_objective": (batch, 27, 27),
        }
        if (
            set(model) != {"names", *expected_model_shapes}
            or model["names"].tolist() != item["names"]
        ):
            raise ValueError("model carrier fields or identity order differ")
        if any(model[k].shape != shape for k, shape in expected_model_shapes.items()):
            raise ValueError(
                "model carrier frame, identity, or covariance dimensions differ"
            )
        verify_incumbent_source(item, model["incumbent"])
        if plan.get("model_sha256") != seal["files"]["model.npz"]["sha256"]:
            raise ValueError("plans use a different physical response model")
        if set(plan["schedules"]) != set(native_arm_names(sensing)):
            raise ValueError("required sensing policy is missing")
        if tuple(seal["clean_arms"]) != clean_arm_names(sensing):
            raise ValueError("required clean forecast arm is missing")
        if plan["query_pairs"] != [
            list(p) for p in query_pairs(config, SensingConfig())
        ]:
            raise ValueError("query pool changed")
        for arm, schedules in plan["schedules"].items():
            budget = int(arm.split("_")[1])
            if len(schedules) != len(item["names"]):
                raise ValueError("query schedule denominator changed")
            for schedule in schedules:
                validate_schedule(schedule, len(plan["query_pairs"]), budget)
        for case in range(batch):
            expected_plans, design, objective = schedules_for_case(
                model["response"][case],
                config,
                sensing,
                seed=sensing.random_seed + item["noise_seed_offset"] + case,
            )
            if any(
                list(value) != plan["schedules"][arm][case]
                for arm, value in expected_plans.items()
            ):
                raise ValueError(
                    "sealed schedule is not the registered model-only policy"
                )
            if (
                not np.array_equal(design, model["query_design"][case])
                or not np.array_equal(objective, model["future_objective"][case])
                or not np.array_equal(
                    planning_matrices(model["response"][case], config, sensing)[2],
                    model["current_objective"][case],
                )
            ):
                raise ValueError(
                    "query Jacobians or forecast objective differ from the model"
                )
        for condition in ("clean", *protocol["noise"]["conditions"]):
            predictions = arrays[f"{condition}.npz"]
            expected_arms = (
                clean_arm_names(sensing)
                if condition == "clean"
                else noise_arm_names(sensing)
            )
            if set(predictions) != {"names", *expected_arms}:
                raise ValueError("a required forecast comparison is missing")
            shape: tuple[int, ...] = (batch, 120, node_count, 3)
            if condition != "clean":
                shape = (protocol["noise"]["repetitions"], *shape)
            if any(predictions[k].shape != shape for k in expected_arms):
                raise ValueError(
                    "forecast frame, identity, or repetition denominator changed"
                )
            fits = arrays[f"fits_{condition}.npz"]
            fit_arms = (
                native_arm_names(sensing)
                if condition == "clean"
                else ("uniform_8", "forecast_8")
            )
            expected_fits = {
                f"{arm}__{key}"
                for arm in fit_arms
                for key in ("coefficients", "covariance", "gain", "observed_positions")
            }
            if set(fits) != expected_fits:
                raise ValueError("a required inference audit field is missing")
            for arm in fit_arms:
                shapes = {
                    "coefficients": (batch, 27),
                    "covariance": (batch, 27, 27),
                    "gain": (batch,),
                    "observed_positions": (batch, int(arm.split("_")[1]), 3),
                }
                for key, expected_shape in shapes.items():
                    if condition != "clean":
                        expected_shape = (
                            protocol["noise"]["repetitions"],
                            *expected_shape,
                        )
                    if fits[f"{arm}__{key}"].shape != expected_shape:
                        raise ValueError("inference audit dimensions differ")
            if predictions["names"].tolist() != item["names"]:
                raise ValueError("forecast identity order changed")
            base = arrays["model.npz"]["incumbent"][:, config.prefix_length :]
            reference = predictions["incumbent"]
            if condition == "clean":
                if list(k for k in predictions if k != "names") != seal["clean_arms"]:
                    raise ValueError("clean arm set/order changed")
                reference = reference[None]
            elif reference.shape[0] != protocol["noise"]["repetitions"]:
                raise ValueError("noise denominator changed")
            if any(array_digest(row) != array_digest(base) for row in reference):
                raise ValueError("the incumbent forecast was modified")
    return barrier


def summarize_aggregates(results: dict[str, Any]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for label, objects in (
        ("transfer_only", ("DLO1", "DLO3")),
        ("all_three_including_discovery", tuple(results)),
    ):
        aggregate[label] = {}
        for condition in results[objects[0]]:
            if condition not in (
                "clean",
                "independent_1mm",
                "independent_1mm_shared_5mm",
            ):
                continue
            aggregate[label][condition] = {}
            for arm in results[objects[0]][condition]["summaries"]:
                rows = [results[o][condition]["summaries"][arm] for o in objects]
                aggregate[label][condition][arm] = {
                    "coordinate_l1_mean_object_change_percent": float(
                        np.mean([r["coordinate_l1_mm_change_percent"] for r in rows])
                    ),
                    "point_rmse_mean_object_change_percent": float(
                        np.mean([r["point_rmse_mm_change_percent"] for r in rows])
                    ),
                    "coordinate_l1_mm": float(
                        np.mean([r["coordinate_l1_mm"] for r in rows])
                    ),
                    "point_rmse_mm": float(np.mean([r["point_rmse_mm"] for r in rows])),
                    "joint_wins": sum(r["joint_wins"] for r in rows),
                    "case_count": sum(r["case_count"] for r in rows),
                }
    return aggregate


def score(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    import run_deform_dlo_source as source

    barrier = validate_barrier(
        args.output, protocol, parent, receipt, file_digest(args.source_receipt)
    )
    results = {}
    for item in parent["objects"]:
        config = config_for_object(parent, item)
        manifest = multi.verify_input_files(item)
        raw = source._load_named_trajectories(
            manifest, item["names"], frame_count=500, node_count=config.node_count
        )
        truth = np.stack(
            [
                raw[n][config.prefix_length + 2 : config.forecast_end + 2]
                for n in item["names"]
            ]
        )
        result = {}
        for condition in ("clean", *protocol["noise"]["conditions"]):
            with np.load(
                args.output / item["object"] / f"{condition}.npz", allow_pickle=False
            ) as archive:
                values = {k: archive[k].copy() for k in archive.files if k != "names"}
            result[condition] = summarize_predictions(
                values, truth, item["names"], config
            )
        results[item["object"]] = result
    record = {
        "schema": "deform-forecast-aware-sensing-result-v1",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": file_digest(args.source_receipt),
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "prediction_barrier_sha256": file_digest(
            args.output / "prediction_barrier.json"
        ),
        "objects": results,
        "aggregate": summarize_aggregates(results),
        "decision": primary_decision(results),
        "ordinary_success": barrier["ordinary_success"],
        "analysis_case_count": parent["analysis_case_count"],
        "retained_technical_failure": 0,
        "unsealable": 0,
        "protected_data_access": False,
        "original_results_modified": False,
        "calibrated_predictive_distribution_claim": False,
        "population_confirmation_or_sota_claim": False,
    }
    write_json_once(args.output / "result.json", record)
    print(
        json.dumps(
            {
                "stage": "scored",
                "decision": record["decision"],
                "transfer_clean": record["aggregate"]["transfer_only"]["clean"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "predict", "score", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-receipt-sha256")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
        return
    if args.source_receipt is None or not args.source_receipt_sha256:
        parser.error("source receipt and its expected SHA-256 are required")
    receipt = multi.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    if receipt.get("experiment") != "deform-forecast-aware-sensing-v1":
        raise ValueError("source receipt belongs to another experiment")
    protocol, parent = load_protocol(ROOT / PROTOCOL, ROOT)
    if args.command == "predict":
        predict(args, protocol, parent, receipt)
    elif args.command == "score":
        score(args, protocol, parent, receipt)
    else:
        barrier = validate_barrier(
            args.output, protocol, parent, receipt, file_digest(args.source_receipt)
        )
        print(
            json.dumps(
                {
                    "barrier_valid": True,
                    "ordinary_success": barrier["ordinary_success"],
                    "new_metrics_computed": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
