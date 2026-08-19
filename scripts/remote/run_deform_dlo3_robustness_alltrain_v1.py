#!/usr/bin/env python3
"""Refit the source-authorized DLO3 method on all 56 train trajectories."""

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
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    augment_deform_local_residual_full_covariance,
    load_deform_dlo_robustness_v1_protocol,
    validate_deform_bayesian_audit_v1,
    validate_deform_dlo3_alltrain_compute_match_v1,
    validate_deform_dlo3_backend_result_v1,
    validate_deform_dlo3_sensitivity_result_v1,
    validate_deform_dlo3_source_manifest,
    verify_deform_dlo3_backend_artifacts_v1,
    verify_deform_dlo3_seed_bayesian_artifacts_v1,
    verify_deform_dlo3_seed_diagnostic_artifacts_v1,
    verify_deform_dlo3_sensitivity_artifacts_v1,
    verify_deform_dlo3_stability_artifacts_v1,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--primary-seed-result", type=Path, required=True)
    parser.add_argument("--stability-gate", type=Path, required=True)
    parser.add_argument("--sensitivity-result", type=Path, required=True)
    parser.add_argument("--backend-result", type=Path, required=True)
    parser.add_argument("--custody-deviation", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), default="run")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path, *, update: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if update is not None:
        result["update"] = update
    return result


def _write_json(
    path: Path, payload: dict[str, object], *, immutable: bool = True
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and immutable:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked all-train output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _assert_authorization(
    *,
    protocol_path: Path,
    manifest_path: Path,
    primary_path: Path,
    stability_path: Path,
    sensitivity_path: Path,
    backend_path: Path,
    deviation_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    protocol_digest = sha256_file(protocol_path)
    manifest_digest = sha256_file(manifest_path)
    primary = _read_json(primary_path)
    stability = _read_json(stability_path)
    sensitivity = _read_json(sensitivity_path)
    backend = _read_json(backend_path)
    deviation = _read_json(deviation_path)
    primary_protocol = _mapping(primary.get("protocol"), label="primary protocol")
    primary_manifest = _mapping(
        primary.get("source_manifest"), label="primary manifest"
    )
    if (
        primary.get("contract") != "deform-dlo3-robustness-seed-result-v1"
        or primary.get("seed") != 42
        or primary_protocol.get("sha256") != protocol_digest
        or primary_manifest.get("sha256") != manifest_digest
        or _mapping(primary.get("primary_source_gate"), label="primary gate").get(
            "passed"
        )
        is not True
        or primary.get("primary_eval_read") is not False
        or primary.get("target_authorized") is not False
    ):
        raise ValueError("DLO3 primary source authorization differs")
    primary_bayesian_audit = validate_deform_bayesian_audit_v1(
        primary, context="source"
    )
    primary_bayesian_artifacts = verify_deform_dlo3_seed_bayesian_artifacts_v1(primary)
    primary_diagnostic_artifacts = verify_deform_dlo3_seed_diagnostic_artifacts_v1(
        primary, protocol
    )
    stability_artifacts = verify_deform_dlo3_stability_artifacts_v1(stability, protocol)
    if (
        stability.get("contract") != "deform-dlo3-training-stability-gate-v1"
        or stability.get("protocol_sha256") != protocol_digest
        or stability.get("source_manifest_sha256") != manifest_digest
        or stability.get("passed") is not True
        or stability.get("alltrain_fit_authorized") is not True
        or stability.get("target_authorized") is not False
        or stability.get("primary_eval_read") is not False
        or stability.get("bayesian_audit_complete") is not True
        or stability.get("bayesian_artifacts_verified") is not True
        or stability.get("diagnostic_artifacts_verified") is not True
        or int(cast(Any, stability.get("diagnostic_seed_count", -1))) != 3
        or int(cast(Any, stability.get("bayesian_distribution_count", -1)))
        != len(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS)
        or stability.get("bayesian_distribution_selection") != "none"
    ):
        raise ValueError("DLO3 stability authorization differs")
    for payload, contract, label in (
        (
            sensitivity,
            "deform-dlo3-physics-solver-sensitivity-result-v1",
            "sensitivity",
        ),
        (backend, "deform-dlo3-pyelastica-source-result-v1", "backend"),
    ):
        identity = _mapping(payload.get("protocol"), label=f"{label} protocol")
        if (
            payload.get("contract") != contract
            or identity.get("sha256") != protocol_digest
            or payload.get("source_test_opened") is not True
            or payload.get("primary_eval_read") is not False
            or payload.get("retry_authorized") is not False
            or payload.get("held_v8_access") is not False
        ):
            raise ValueError(f"DLO3 {label} audit differs")
    sensitivity_verification = validate_deform_dlo3_sensitivity_result_v1(
        sensitivity, protocol
    )
    sensitivity_artifacts = verify_deform_dlo3_sensitivity_artifacts_v1(
        sensitivity, protocol
    )
    primary_digest = sha256_file(primary_path)
    if (
        _mapping(
            stability_artifacts.get("seed_result_sha256_by_seed"),
            label="stability seed digests",
        ).get("42")
        != primary_digest
        or sensitivity_artifacts.get("parent_seed_result_sha256") != primary_digest
    ):
        raise ValueError("DLO3 primary diagnostic lineage differs")
    backend_verification = validate_deform_dlo3_backend_result_v1(backend, protocol)
    backend_artifacts = verify_deform_dlo3_backend_artifacts_v1(backend, protocol)
    emitted = _mapping(deviation.get("emitted_information"), label="deviation emission")
    if (
        deviation.get("contract") != "deform-dlo3-count-only-custody-deviation-v1"
        or emitted.get("eval_file_count") != 14
        or emitted.get("file_names_emitted") is not False
        or emitted.get("file_bytes_opened") is not False
        or deviation.get("official_eval_read") is not False
        or deviation.get("held_v8_access") is not False
    ):
        raise ValueError("DLO3 count-only custody deviation differs")
    return primary, {
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "primary_seed_result": _identity(primary_path),
        "stability_gate": _identity(stability_path),
        "sensitivity_result": _identity(sensitivity_path),
        "backend_result": _identity(backend_path),
        "custody_deviation": _identity(deviation_path),
        "primary_bayesian_audit": primary_bayesian_audit,
        "primary_bayesian_artifacts": primary_bayesian_artifacts,
        "primary_diagnostic_artifacts": primary_diagnostic_artifacts,
        "stability_artifacts": stability_artifacts,
        "sensitivity_verification": sensitivity_verification,
        "sensitivity_artifacts": sensitivity_artifacts,
        "backend_verification": backend_verification,
        "backend_artifacts": backend_artifacts,
    }


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    compute_contract = _mapping(
        protocol.get("compute_matched_control"), label="compute-matched control"
    )
    primary, authorization = _assert_authorization(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        primary_path=args.primary_seed_result.resolve(),
        stability_path=args.stability_gate.resolve(),
        sensitivity_path=args.sensitivity_result.resolve(),
        backend_path=args.backend_result.resolve(),
        deviation_path=args.custody_deviation.resolve(),
    )
    manifest = _read_json(manifest_path)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    all_names = sorted(name for values in partitions.values() for name in values)
    if len(all_names) != 56 or len(set(all_names)) != 56:
        raise ValueError("DLO3 all-train authorization does not cover 56 trajectories")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"all-train output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        str(_mapping(protocol["upstream"], label="upstream")["commit"]),
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO4")
    source_runtime._install_eval_read_guard(data_root / "DLO5")
    method_spec = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-alltrain-method-spec-v1",
        "authorization": authorization,
        "physical_backend": "official-DEFORM-PBD-all-56",
        "physical_checkpoint_update": 6400,
        "seed": 42,
        "ridge": 1.0,
        "shrinkage": 0.25,
        "covariance": "trajectory-clustered-full-coordinate-covariance-v1",
        "variance_scale": "reuse-seed42-source-calibration-without-refit",
        "compute_matched_control": (
            "frozen-wall-time-equivalent-DEFORM-continuation-v1"
        ),
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "source_bayesian_audit_complete": True,
        "source_diagnostics_verified": True,
        "primary_diagnostic_artifacts": authorization["primary_diagnostic_artifacts"],
        "sensitivity_artifacts": authorization["sensitivity_artifacts"],
        "backend_target_arm": authorization["backend_artifacts"],
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "official_eval_read": False,
    }
    method_spec_path = output_root / "method_spec.json"
    _write_json(method_spec_path, method_spec)
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-alltrain-preflight-v1",
        "mode": args.mode,
        "authorization": authorization,
        "method_spec": _identity(method_spec_path),
        "upstream": upstream,
        "train_trajectory_count": len(all_names),
        "primary_eval_enumerated_by_this_runner": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    trajectories = source_runtime._load_named_trajectories(
        manifest,
        all_names,
        frame_count=500,
        node_count=12,
    )
    training = _mapping(protocol.get("physical_training"), label="training")
    cublas_config = str(training["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    parent_runtime = _mapping(primary.get("runtime"), label="primary runtime")
    if torch.__version__ != parent_runtime.get(
        "torch"
    ) or torch.version.cuda != parent_runtime.get("cuda"):
        raise RuntimeError("DLO3 all-train runtime differs from primary source run")
    modules = source_runtime._load_upstream(args.upstream_root)
    seed = int(cast(Any, training["primary_seed"]))
    source_runtime._seed_everything(torch, seed)
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        args.device,
        dlo_type="DLO3",
        node_count=12,
    )
    optimizer = source_runtime._official_optimizer(torch, model)
    orientations = source_runtime._precompute_material_u0(
        trajectories,
        modules=modules,
        model_function=model_function,
        torch=torch,
        device=args.device,
    )
    registered_updates = int(cast(Any, training["total_updates"]))
    maximum_extra = int(cast(Any, compute_contract["maximum_additional_updates"]))
    schedule_updates = registered_updates + maximum_extra
    updates = 1 if args.mode == "smoke" else registered_updates
    trajectory_indices, start_indices = source_runtime._make_schedule(
        fit_names=all_names,
        updates=schedule_updates,
        batch_size=int(cast(Any, training["batch_size"])),
        frame_count=500,
        horizon=int(cast(Any, training["unroll_horizon_frames"])),
        seed=seed,
    )
    schedule_path = output_root / "window_schedule.npz"
    np.savez_compressed(
        schedule_path,
        fit_names=np.asarray(all_names),
        trajectory_indices=trajectory_indices,
        start_indices=start_indices,
    )
    protocol_sha256 = sha256_file(protocol_path)
    schedule_sha256 = sha256_file(schedule_path)
    method_spec_sha256 = sha256_file(method_spec_path)
    checkpoint_updates = set(
        int(value) for value in cast(Any, training["checkpoint_updates"])
    )
    checkpoints: list[dict[str, object]] = []
    losses: list[dict[str, object]] = []
    started = time.perf_counter()

    def save_checkpoint(update: int) -> Path:
        path: Path = output_root / "checkpoints" / f"update_{update:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": update,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "protocol_sha256": protocol_sha256,
                "schedule_sha256": schedule_sha256,
                "method_spec_sha256": method_spec_sha256,
                "official_eval_read": False,
            },
            path,
        )
        checkpoints.append(_identity(path, update=update))
        return path

    if args.mode == "run":
        save_checkpoint(0)
    final_checkpoint_path: Path | None = None
    for update_index in range(updates):
        batch = source_runtime._assemble_batch(
            trajectories,
            orientations,
            all_names,
            trajectory_indices[update_index],
            start_indices[update_index],
            horizon=int(cast(Any, training["unroll_horizon_frames"])),
            torch=torch,
            device=args.device,
        )
        update_started = time.perf_counter()
        loss = source_runtime._train_update(
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
        losses.append(
            {
                "update": update,
                "position_l1_m": loss,
                "seconds": time.perf_counter() - update_started,
            }
        )
        if args.mode == "run" and update in checkpoint_updates:
            final_checkpoint_path = save_checkpoint(update)
        if update == 1 or update % 10 == 0:
            progress: dict[str, object] = {
                "completed_updates": update,
                "requested_updates": updates,
                "elapsed_seconds": time.perf_counter() - started,
                "primary_eval_read": False,
            }
            _write_json(output_root / "progress.json", progress, immutable=False)
            print(json.dumps(progress, sort_keys=True), flush=True)
    if args.mode == "smoke":
        smoke = {
            "schema_version": 1,
            "contract": "deform-dlo3-robustness-alltrain-smoke-v1",
            "losses": losses,
            "primary_eval_read": False,
            "target_authorized": False,
        }
        _write_json(output_root / "smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0
    if final_checkpoint_path is None or checkpoints[-1].get("update") != 6400:
        raise RuntimeError(
            "DLO3 all-train run omitted its final update-6400 checkpoint"
        )

    state = torch.load(final_checkpoint_path, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    local_started = time.perf_counter()
    rollout = posterior_runtime._evaluate_state(
        state,
        trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO3",
        node_count=12,
    )
    initial, action = local_runtime._causal_inputs(trajectories, all_names)
    residual = _mapping(protocol.get("local_residual"), label="local residual")
    local_model = fit_deform_local_residual(
        initial,
        action,
        np.asarray(rollout["predictions"]),
        np.asarray(rollout["targets"]),
        all_names,
        ridge=float(cast(Any, residual["ridge"])),
        variance_floor_m2=float(cast(Any, residual["coordinate_variance_floor_m2"])),
    )
    local_model_path = output_root / "local_residual_model.npz"
    np.savez_compressed(
        local_model_path, **serialize_deform_local_residual_model(local_model)
    )
    local_wall_seconds = time.perf_counter() - local_started
    recent = np.asarray(
        [float(cast(Any, record["seconds"])) for record in losses[-100:]],
        dtype=np.float64,
    )
    median_update_seconds = float(np.median(recent))
    if not math.isfinite(median_update_seconds) or median_update_seconds <= 0.0:
        raise RuntimeError("all-train compute-matched update duration is invalid")
    additional_updates = int(math.ceil(local_wall_seconds / median_update_seconds))
    minimum_extra = int(cast(Any, compute_contract["minimum_additional_updates"]))
    if not minimum_extra <= additional_updates <= maximum_extra:
        raise RuntimeError("all-train compute-matched update count is outside bounds")
    compute_match = {
        "schema_version": 1,
        "contract": "deform-dlo3-alltrain-compute-match-v1",
        "seed": seed,
        "local_residual_wall_seconds": local_wall_seconds,
        "median_update_seconds_6301_6400": median_update_seconds,
        "additional_updates": additional_updates,
        "start_update": registered_updates,
        "end_update": registered_updates + additional_updates,
        "selection_effect": "none",
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "primary_eval_read": False,
    }
    compute_match_verification = validate_deform_dlo3_alltrain_compute_match_v1(
        compute_match, protocol
    )
    compute_match_path = output_root / "compute_match.json"
    _write_json(compute_match_path, compute_match)
    for offset in range(additional_updates):
        schedule_index = registered_updates + offset
        batch = source_runtime._assemble_batch(
            trajectories,
            orientations,
            all_names,
            trajectory_indices[schedule_index],
            start_indices[schedule_index],
            horizon=int(cast(Any, training["unroll_horizon_frames"])),
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
    compute_checkpoint_path = save_checkpoint(registered_updates + additional_updates)
    full_model = augment_deform_local_residual_full_covariance(
        local_model,
        initial,
        action,
        np.asarray(rollout["predictions"]),
        np.asarray(rollout["targets"]),
        all_names,
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
    source_calibration = _mapping(
        _mapping(primary.get("bayesian_audit"), label="primary Bayesian audit").get(
            "calibration"
        ),
        label="source calibration",
    )
    if (
        source_calibration.get("contract")
        != "deform-dlo-full-covariance-calibration-v1"
        or source_calibration.get("rank") != 9
        or float(cast(Any, source_calibration.get("variance_scale", 0.0))) < 1.0
    ):
        raise ValueError("DLO3 source covariance calibration differs")
    calibration_record = {
        **source_calibration,
        "policy": "reuse-frozen-seed42-source-calibration-without-refit",
        "alltrain_target_read": False,
    }
    calibration_path = output_root / "covariance_calibration.json"
    _write_json(calibration_path, calibration_record)
    final_method = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-alltrain-final-method-v1",
        "method_spec": _identity(method_spec_path),
        "window_schedule": _identity(schedule_path),
        "physical_checkpoint": _identity(final_checkpoint_path, update=6400),
        "compute_matched_checkpoint": _identity(
            compute_checkpoint_path,
            update=registered_updates + additional_updates,
        ),
        "compute_match": _identity(compute_match_path),
        "compute_match_verification": compute_match_verification,
        "local_residual_model": _identity(local_model_path),
        "full_covariance_model": _identity(full_model_path),
        "covariance_calibration": _identity(calibration_path),
        "ridge": float(cast(Any, residual["ridge"])),
        "shrinkage": float(cast(Any, residual["shrinkage"])),
        "variance_scale": float(cast(Any, source_calibration["variance_scale"])),
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "source_bayesian_audit_complete": True,
        "source_diagnostics_verified": True,
        "primary_diagnostic_artifacts": authorization["primary_diagnostic_artifacts"],
        "sensitivity_artifacts": authorization["sensitivity_artifacts"],
        "backend_target_arm": authorization["backend_artifacts"],
        "distribution_selection": "none",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "primary_eval_read": False,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    final_method_path = output_root / "final_method.json"
    _write_json(final_method_path, final_method)
    result = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-alltrain-result-v1",
        "claim_boundary": "All 56 DLO3 train trajectories only; official evaluation unopened.",
        "authorization": authorization,
        "final_method": _identity(final_method_path),
        "compute_match": _identity(compute_match_path),
        "compute_match_verification": compute_match_verification,
        "checkpoints": checkpoints,
        "training_losses": losses,
        "runtime": final_method["runtime"],
        "elapsed_seconds": time.perf_counter() - started,
        "bayesian_audit_complete": True,
        "bayesian_distribution_count": len(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS),
        "primary_eval_enumerated_by_this_runner": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    result_path = output_root / "alltrain_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
