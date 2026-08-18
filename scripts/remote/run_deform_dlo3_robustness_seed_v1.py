#!/usr/bin/env python3
"""Run one fixed-seed DLO3 source model and seal predictions before scoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    fit_deform_local_residual,
    predict_deform_local_residual,
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    augment_deform_local_residual_full_covariance,
    calibrate_deform_full_covariance,
    evaluate_deform_dlo3_source_gate,
    evaluate_deform_predictive_distribution,
    fit_deform_local_residual_variant,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_full_covariance,
    predict_deform_local_residual_variant,
    scale_deform_coordinate_covariance,
    validate_deform_dlo3_source_manifest,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

Array = np.ndarray[Any, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), default="run")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(
    path: Path,
    payload: dict[str, object],
    *,
    immutable: bool = True,
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if immutable and path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        if immutable:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _identity(path: Path, *, update: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if update is not None:
        result["update"] = update
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _save_variant(path: Path, model: Mapping[str, object]) -> None:
    np.savez_compressed(
        path,
        arm=np.asarray([str(model["arm"])]),
        coordinate_frame=np.asarray([str(model["coordinate_frame"])]),
        node_count=np.asarray([int(cast(Any, model["node_count"]))], dtype=np.int64),
        prediction_horizon=np.asarray(
            [int(cast(Any, model["prediction_horizon"]))], dtype=np.int64
        ),
        feature_indices=np.asarray(model["feature_indices"], dtype=np.int64),
        feature_location=np.asarray(model["feature_location"]),
        feature_scale=np.asarray(model["feature_scale"]),
        coefficients=np.asarray(model["coefficients"]),
        ridge=np.asarray([float(cast(Any, model["ridge"]))]),
    )


def _mean_l1(prediction: Array, target: Array) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    training = _mapping(protocol.get("physical_training"), label="physical training")
    residual = _mapping(protocol.get("local_residual"), label="local residual")
    compute = _mapping(
        protocol.get("compute_matched_control"), label="compute-matched control"
    )
    seeds = tuple(int(value) for value in cast(Any, training["audit_seeds"]))
    if args.seed not in seeds:
        raise ValueError("requested seed is outside the frozen stability audit")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"seed output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_protocol = _mapping(protocol.get("upstream"), label="upstream")
    upstream = source_runtime._assert_upstream(
        args.upstream_root, str(upstream_protocol["commit"])
    )
    manifest = _read_json(manifest_path)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO4")
    source_runtime._install_eval_read_guard(data_root / "DLO5")
    data = _mapping(protocol.get("data"), label="data")
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-seed-preflight-v1",
        "mode": args.mode,
        "seed": args.seed,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "upstream": upstream,
        "partition_counts": {name: len(values) for name, values in partitions.items()},
        "source_payload_deserialized": False,
        "source_test_opened": False,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "reserve_payload_read": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    cublas_config = str(training["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, args.seed)
    fit_names = list(partitions["fit"])
    calibration_names = list(partitions["calibration"])
    development_names = fit_names + calibration_names
    frame_count = int(cast(Any, data["frame_count"]))
    node_count = int(cast(Any, data["node_count"]))
    development = source_runtime._load_named_trajectories(
        manifest,
        development_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    fit_trajectories = {name: development[name] for name in fit_names}
    calibration_trajectories = {name: development[name] for name in calibration_names}
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    optimizer = source_runtime._official_optimizer(torch, model)
    orientations = source_runtime._precompute_material_u0(
        fit_trajectories,
        modules=modules,
        model_function=model_function,
        torch=torch,
        device=args.device,
    )
    registered_updates = int(cast(Any, training["total_updates"]))
    maximum_extra = int(cast(Any, compute["maximum_additional_updates"]))
    schedule_updates = registered_updates + maximum_extra
    batch_size = int(cast(Any, training["batch_size"]))
    horizon = int(cast(Any, training["unroll_horizon_frames"]))
    trajectory_indices, start_indices = source_runtime._make_schedule(
        fit_names=fit_names,
        updates=schedule_updates,
        batch_size=batch_size,
        frame_count=frame_count,
        horizon=horizon,
        seed=args.seed,
    )
    schedule_path = output_root / "window_schedule.npz"
    np.savez_compressed(
        schedule_path,
        fit_names=np.asarray(fit_names),
        trajectory_indices=trajectory_indices,
        start_indices=start_indices,
    )
    protocol_sha256 = sha256_file(protocol_path)
    schedule_sha256 = sha256_file(schedule_path)
    checkpoint_updates = set(
        int(value) for value in cast(Any, training["checkpoint_updates"])
    )
    checkpoints: list[dict[str, object]] = []
    training_losses: list[dict[str, object]] = []
    started = time.perf_counter()

    def save_checkpoint(update: int, *, label: str = "registered") -> Path:
        path: Path = output_root / "checkpoints" / f"{label}_update_{update:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": update,
                "seed": args.seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "protocol_sha256": protocol_sha256,
                "source_manifest_sha256": sha256_file(manifest_path),
                "schedule_sha256": schedule_sha256,
                "official_eval_read": False,
            },
            path,
        )
        identity = _identity(path, update=update)
        identity["label"] = label
        checkpoints.append(identity)
        return path

    if args.mode == "run":
        save_checkpoint(0)
    updates = 1 if args.mode == "smoke" else registered_updates
    for update_index in range(updates):
        batch = source_runtime._assemble_batch(
            fit_trajectories,
            orientations,
            fit_names,
            trajectory_indices[update_index],
            start_indices[update_index],
            horizon=horizon,
            torch=torch,
            device=args.device,
        )
        update_started = time.perf_counter()
        training_l1_m = source_runtime._train_update(
            modules=modules,
            model_function=model_function,
            model=model,
            optimizer=optimizer,
            batch=batch,
            torch=torch,
            device=args.device,
        )
        torch.cuda.synchronize(args.device)
        update = update_index + 1
        training_losses.append(
            {
                "update": update,
                "position_l1_m": training_l1_m,
                "seconds": time.perf_counter() - update_started,
            }
        )
        if args.mode == "run" and update in checkpoint_updates:
            save_checkpoint(update)
        if update == 1 or update % 10 == 0:
            progress = {
                "mode": args.mode,
                "seed": args.seed,
                "completed_updates": update,
                "requested_updates": updates,
                "latest_position_l1_m": training_l1_m,
                "elapsed_seconds": time.perf_counter() - started,
                "source_test_opened": False,
                "official_eval_read": False,
            }
            _write_json(output_root / "progress.json", progress, immutable=False)
            print(json.dumps(progress, sort_keys=True), flush=True)

    if args.mode == "smoke":
        smoke = {
            "schema_version": 1,
            "contract": "deform-dlo3-robustness-seed-smoke-v1",
            "seed": args.seed,
            "training": training_losses,
            "source_test_opened": False,
            "primary_eval_read": False,
            "target_authorized": False,
        }
        _write_json(output_root / "smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0

    final_candidates = [
        record
        for record in checkpoints
        if int(cast(Any, record["update"])) == registered_updates
        and record["label"] == "registered"
    ]
    if len(final_candidates) != 1:
        raise RuntimeError("seed run omitted its unique update-6400 checkpoint")
    final_checkpoint = final_candidates[0]
    final_checkpoint_path = Path(str(final_checkpoint["path"]))
    bundle = torch.load(final_checkpoint_path, map_location="cpu", weights_only=True)
    state = bundle["model_state_dict"]

    local_started = time.perf_counter()
    fit_rollout = posterior_runtime._evaluate_state(
        state,
        fit_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    fit_initial, fit_action = local_runtime._causal_inputs(fit_trajectories, fit_names)
    local_model = fit_deform_local_residual(
        fit_initial,
        fit_action,
        np.asarray(fit_rollout["predictions"]),
        np.asarray(fit_rollout["targets"]),
        fit_names,
        ridge=float(cast(Any, residual["ridge"])),
        variance_floor_m2=float(cast(Any, residual["coordinate_variance_floor_m2"])),
    )
    local_model_path = output_root / "local_residual_model.npz"
    np.savez_compressed(
        local_model_path, **serialize_deform_local_residual_model(local_model)
    )
    local_wall_seconds = time.perf_counter() - local_started

    full_model = augment_deform_local_residual_full_covariance(
        local_model,
        fit_initial,
        fit_action,
        np.asarray(fit_rollout["predictions"]),
        np.asarray(fit_rollout["targets"]),
        fit_names,
    )
    full_model_path = output_root / "full_covariance_model.npz"
    full_payload = serialize_deform_local_residual_model(full_model)
    full_payload["coefficient_covariance_full"] = np.asarray(
        full_model["coefficient_covariance_full"]
    )
    full_payload["residual_covariance_full"] = np.asarray(
        full_model["residual_covariance_full"]
    )
    np.savez_compressed(full_model_path, **full_payload)

    calibration_rollout = posterior_runtime._evaluate_state(
        state,
        calibration_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    calibration_initial, calibration_action = local_runtime._causal_inputs(
        calibration_trajectories, calibration_names
    )
    calibration_prediction = predict_deform_local_residual_full_covariance(
        full_model,
        calibration_initial,
        calibration_action,
        np.asarray(calibration_rollout["predictions"]),
        shrinkage=float(cast(Any, residual["shrinkage"])),
    )
    calibration = calibrate_deform_full_covariance(
        calibration_prediction["predictions"],
        np.asarray(calibration_rollout["targets"]),
        calibration_prediction["coordinate_covariance_m2"],
    )
    calibration_record = {
        **calibration,
        "trajectory_scores": [
            float(value) for value in np.asarray(calibration["trajectory_scores"])
        ],
        "source_test_opened": False,
        "official_eval_read": False,
    }
    calibration_path = output_root / "covariance_calibration.json"
    _write_json(calibration_path, calibration_record)

    mechanism_models = {
        "physical-plus-intercept-only": fit_deform_local_residual_variant(
            fit_initial,
            fit_action,
            np.asarray(fit_rollout["predictions"]),
            np.asarray(fit_rollout["targets"]),
            fit_names,
            ridge=float(cast(Any, residual["ridge"])),
            arm="intercept-only",
        ),
        "physical-plus-full-no-action": fit_deform_local_residual_variant(
            fit_initial,
            fit_action,
            np.asarray(fit_rollout["predictions"]),
            np.asarray(fit_rollout["targets"]),
            fit_names,
            ridge=float(cast(Any, residual["ridge"])),
            arm="full-no-action",
        ),
        "physical-plus-full-global-frame": fit_deform_local_residual_variant(
            fit_initial,
            fit_action,
            np.asarray(fit_rollout["predictions"]),
            np.asarray(fit_rollout["targets"]),
            fit_names,
            ridge=float(cast(Any, residual["ridge"])),
            arm="full-global",
        ),
        "persistence-plus-full-local": fit_deform_local_residual_variant(
            fit_initial,
            fit_action,
            np.asarray(fit_rollout["persistence"]),
            np.asarray(fit_rollout["targets"]),
            fit_names,
            ridge=float(cast(Any, residual["ridge"])),
            arm="full-local",
        ),
    }
    mechanism_identities: dict[str, dict[str, object]] = {}
    mechanism_root = output_root / "mechanism_models"
    mechanism_root.mkdir(parents=True, exist_ok=True)
    for label, variant in mechanism_models.items():
        path = mechanism_root / f"{label}.npz"
        _save_variant(path, variant)
        mechanism_identities[label] = _identity(path)

    recent = np.asarray(
        [float(cast(Any, record["seconds"])) for record in training_losses[-100:]],
        dtype=np.float64,
    )
    median_update_seconds = float(np.median(recent))
    if not math.isfinite(median_update_seconds) or median_update_seconds <= 0.0:
        raise RuntimeError("compute-matched update duration is invalid")
    additional_updates = int(math.ceil(local_wall_seconds / median_update_seconds))
    minimum_extra = int(cast(Any, compute["minimum_additional_updates"]))
    if not minimum_extra <= additional_updates <= maximum_extra:
        raise RuntimeError("compute-matched additional update count is outside bounds")
    compute_match = {
        "schema_version": 1,
        "contract": "deform-dlo3-compute-match-v1",
        "seed": args.seed,
        "local_residual_wall_seconds": local_wall_seconds,
        "median_update_seconds_6301_6400": median_update_seconds,
        "additional_updates": additional_updates,
        "start_update": registered_updates,
        "end_update": registered_updates + additional_updates,
        "source_test_opened": False,
        "official_eval_read": False,
    }
    compute_match_path = output_root / "compute_match.json"
    _write_json(compute_match_path, compute_match)
    for offset in range(additional_updates):
        schedule_index = registered_updates + offset
        batch = source_runtime._assemble_batch(
            fit_trajectories,
            orientations,
            fit_names,
            trajectory_indices[schedule_index],
            start_indices[schedule_index],
            horizon=horizon,
            torch=torch,
            device=args.device,
        )
        source_runtime._train_update(
            modules=modules,
            model_function=model_function,
            model=model,
            optimizer=optimizer,
            batch=batch,
            torch=torch,
            device=args.device,
        )
    torch.cuda.synchronize(args.device)
    compute_checkpoint_path = save_checkpoint(
        registered_updates + additional_updates, label="compute-matched"
    )

    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-method-seal-v1",
        "seed": args.seed,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "window_schedule": _identity(schedule_path),
        "physical_checkpoint": final_checkpoint,
        "compute_matched_checkpoint": _identity(
            compute_checkpoint_path, update=registered_updates + additional_updates
        ),
        "local_residual_model": _identity(local_model_path),
        "full_covariance_model": _identity(full_model_path),
        "covariance_calibration": _identity(calibration_path),
        "mechanism_models": mechanism_identities,
        "ridge": float(cast(Any, residual["ridge"])),
        "shrinkage": float(cast(Any, residual["shrinkage"])),
        "source_test_opened": False,
        "official_eval_read": False,
        "target_selection": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    source_test_names = list(partitions["source_test"])
    source_test_trajectories = source_runtime._load_named_trajectories(
        manifest,
        source_test_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    source_rollout = posterior_runtime._evaluate_state(
        state,
        source_test_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    compute_bundle = torch.load(
        compute_checkpoint_path, map_location="cpu", weights_only=True
    )
    compute_rollout = posterior_runtime._evaluate_state(
        compute_bundle["model_state_dict"],
        source_test_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=node_count,
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_test_trajectories, source_test_names
    )
    shrinkage = float(cast(Any, residual["shrinkage"]))
    source_prediction = predict_deform_local_residual_full_covariance(
        full_model,
        source_initial,
        source_action,
        np.asarray(source_rollout["predictions"]),
        shrinkage=shrinkage,
    )
    calibrated_covariance = scale_deform_coordinate_covariance(
        source_prediction["coordinate_covariance_m2"],
        float(cast(Any, calibration["variance_scale"])),
    )
    mechanism_predictions = {
        "physical-only": np.asarray(source_rollout["predictions"]),
        "persistence-plus-full-local": predict_deform_local_residual_variant(
            mechanism_models["persistence-plus-full-local"],
            source_initial,
            source_action,
            np.asarray(source_rollout["persistence"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-intercept-only": predict_deform_local_residual_variant(
            mechanism_models["physical-plus-intercept-only"],
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-full-no-action": predict_deform_local_residual_variant(
            mechanism_models["physical-plus-full-no-action"],
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-full-global-frame": predict_deform_local_residual_variant(
            mechanism_models["physical-plus-full-global-frame"],
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=shrinkage,
        )["predictions"],
        "physical-plus-full-local-unshrunk": predict_deform_local_residual(
            local_model,
            source_initial,
            source_action,
            np.asarray(source_rollout["predictions"]),
            shrinkage=1.0,
        )["predictions"],
        "physical-plus-full-local-fixed": source_prediction["predictions"],
    }
    predictions_path = output_root / "source_predictions.npz"
    prediction_payload: dict[str, Array] = {
        "names": np.asarray(source_test_names),
        "physical": np.asarray(source_rollout["predictions"]),
        "compute_matched_physical": np.asarray(compute_rollout["predictions"]),
        "candidate": np.asarray(source_prediction["predictions"]),
        "coordinate_covariance_m2": np.asarray(
            source_prediction["coordinate_covariance_m2"]
        ),
        "calibrated_coordinate_covariance_m2": calibrated_covariance,
    }
    prediction_payload.update(
        {
            f"mechanism_{label}": np.asarray(values)
            for label, values in mechanism_predictions.items()
        }
    )
    np.savez_compressed(predictions_path, **cast(dict[str, Any], prediction_payload))
    prediction_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-prediction-seal-v1",
        "seed": args.seed,
        "method_seal": _identity(method_seal_path),
        "predictions": _identity(predictions_path),
        "source_test_case_count": len(source_test_names),
        "source_outcomes_scored": False,
        "official_eval_read": False,
    }
    prediction_seal_path = output_root / "prediction_seal.json"
    _write_json(prediction_seal_path, prediction_seal)

    targets = np.asarray(source_rollout["targets"])
    baseline_predictions = np.asarray(source_rollout["predictions"])
    primary_gate = evaluate_deform_dlo3_source_gate(
        np.asarray(source_prediction["predictions"]),
        baseline_predictions,
        targets,
        source_test_names,
        protocol,
    )
    mechanism_results = {
        label: evaluate_deform_dlo3_source_gate(
            np.asarray(values),
            baseline_predictions,
            targets,
            source_test_names,
            protocol,
        )
        for label, values in mechanism_predictions.items()
    }
    raw_distribution = evaluate_deform_predictive_distribution(
        np.asarray(source_prediction["predictions"]),
        targets,
        np.asarray(source_prediction["coordinate_covariance_m2"]),
    )
    calibrated_distribution = evaluate_deform_predictive_distribution(
        np.asarray(source_prediction["predictions"]),
        targets,
        calibrated_covariance,
    )
    compute_matched_l1_m = _mean_l1(np.asarray(compute_rollout["predictions"]), targets)
    result = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-seed-result-v1",
        "claim_boundary": "DLO3 train source panel only; official evaluation unopened.",
        "seed": args.seed,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "method_seal": _identity(method_seal_path),
        "prediction_seal": _identity(prediction_seal_path),
        "physical_checkpoint": final_checkpoint,
        "compute_match": {
            **compute_match,
            "checkpoint": _identity(
                compute_checkpoint_path,
                update=registered_updates + additional_updates,
            ),
            "source_mean_l1_m": compute_matched_l1_m,
        },
        "primary_source_gate": primary_gate,
        "mechanism_ablation": mechanism_results,
        "bayesian_audit": {
            "calibration": calibration_record,
            "uncalibrated": raw_distribution,
            "calibrated": calibrated_distribution,
            "point_mean_unchanged": True,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "training_losses": training_losses,
        "checkpoints": checkpoints,
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    result_path = output_root / "source_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
