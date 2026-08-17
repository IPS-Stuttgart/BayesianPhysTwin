#!/usr/bin/env python3
"""Refit the source-confirmed DLO2 local residual on all 56 train cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.experiments.deform_dlo_local_residual import (
    fit_deform_local_residual,
    load_deform_dlo2_local_residual_alltrain_v7_protocol,
    load_deform_dlo2_local_residual_v6_protocol,
    serialize_deform_local_residual_model,
    validate_deform_dlo2_local_residual_alltrain_v7_authorization,
)
from bayesian_phystwin.experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
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


def _verified_manifest(
    path: Path,
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    identity = protocol["source_manifest"]
    if not isinstance(identity, Mapping) or sha256_file(path) != identity.get("sha256"):
        raise ValueError("DLO2 all-train source manifest identity does not verify")
    manifest = _read_json(path)
    trajectories = manifest.get("trajectories")
    split = manifest.get("split")
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO2"
        or manifest.get("partition") != "train"
        or manifest.get("official_eval_read") is not False
        or not isinstance(trajectories, dict)
        or not isinstance(split, dict)
    ):
        raise ValueError("DLO2 all-train source manifest contract differs")
    names = sorted(str(name) for name in trajectories)
    split_names: list[str] = []
    for key in ("fit", "validation", "source_test"):
        values = split.get(key)
        if not isinstance(values, list) or not all(
            isinstance(name, str) for name in values
        ):
            raise ValueError("DLO2 all-train source split is invalid")
        split_names.extend(values)
    expected_count = int(protocol["data"]["trajectory_count"])
    if (
        len(names) != expected_count
        or len(split_names) != expected_count
        or len(set(split_names)) != expected_count
        or set(split_names) != set(names)
    ):
        raise ValueError("DLO2 all-train manifest does not cover all 56 train cases")
    return manifest, names


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    source_protocol_path = args.source_protocol.resolve()
    source_result_path = args.source_result.resolve()
    source_manifest_path = args.source_manifest.resolve()
    protocol = load_deform_dlo2_local_residual_alltrain_v7_protocol(protocol_path)
    source_protocol = load_deform_dlo2_local_residual_v6_protocol(source_protocol_path)
    source_result = _read_json(source_result_path)
    source_protocol_sha256 = sha256_file(source_protocol_path)
    source_result_sha256 = sha256_file(source_result_path)
    source_authorization = (
        validate_deform_dlo2_local_residual_alltrain_v7_authorization(
            protocol,
            source_protocol,
            source_result,
            source_protocol_sha256=source_protocol_sha256,
            source_result_sha256=source_result_sha256,
        )
    )

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream = source_runtime._assert_upstream(
        args.upstream_root, str(protocol["upstream"]["commit"])
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    manifest, all_names = _verified_manifest(source_manifest_path, protocol)
    frame_count = int(protocol["data"]["frame_count"])
    node_count = int(protocol["data"]["node_count"])
    trajectories = source_runtime._load_named_trajectories(
        manifest,
        all_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    fixed = protocol["local_residual"]["fixed_arm"]
    method_spec = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-method-spec-v7",
        "official_eval_read": False,
        "physical_backend": "DEFORM-official-model-refit-all-56",
        "physical_checkpoint_update": 6400,
        "local_residual_operator": protocol["local_residual"]["operator"],
        "fixed_arm": fixed,
        "query_evidence": protocol["local_residual"]["query_evidence"],
        "validation_reselection": False,
        "source_reselection": False,
        "target_reselection": False,
        "source_authorization": source_authorization,
    }
    method_spec_path = output_root / "method_spec.json"
    _write_json(method_spec_path, method_spec)
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-alltrain-preflight-v7",
        "mode": args.mode,
        "protocol": _identity(protocol_path),
        "source_protocol": _identity(source_protocol_path),
        "source_result": _identity(source_result_path),
        "source_manifest": _identity(source_manifest_path),
        "method_spec": _identity(method_spec_path),
        "source_authorization": source_authorization,
        "upstream": upstream,
        "validated_train_trajectory_count": len(trajectories),
        "frame_count": frame_count,
        "node_count": node_count,
        "official_eval_read": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    cublas_config = str(protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    source_runtime_info = source_result.get("runtime")
    if (
        not isinstance(source_runtime_info, dict)
        or torch.__version__ != source_runtime_info.get("torch")
        or torch.version.cuda != source_runtime_info.get("cuda")
    ):
        raise RuntimeError("all-train runtime differs from passed DLO2 source")
    modules = source_runtime._load_upstream(args.upstream_root)
    seed = int(protocol["training"]["random_seed"])
    source_runtime._seed_everything(torch, seed)
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        args.device,
        dlo_type="DLO2",
        node_count=node_count,
    )
    optimizer = source_runtime._official_optimizer(torch, model)
    orientations = source_runtime._precompute_material_u0(
        trajectories,
        modules=modules,
        model_function=model_function,
        torch=torch,
        device=args.device,
    )
    registered_updates = int(protocol["training"]["total_updates"])
    updates = 1 if args.mode == "smoke" else registered_updates
    batch_size = int(protocol["training"]["batch_size"])
    horizon = int(protocol["training"]["unroll_horizon_frames"])
    trajectory_indices, start_indices = source_runtime._make_schedule(
        fit_names=all_names,
        updates=registered_updates,
        batch_size=batch_size,
        frame_count=frame_count,
        horizon=horizon,
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
    checkpoint_updates = set(protocol["checkpoint_updates"])
    checkpoints: list[dict[str, object]] = []
    training_losses: list[dict[str, object]] = []
    started = time.perf_counter()

    def save_checkpoint(update: int) -> Path:
        path = output_root / "checkpoints" / f"update_{update:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": update,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "alltrain_protocol_sha256": protocol_sha256,
                "schedule_sha256": schedule_sha256,
                "method_spec_sha256": method_spec_sha256,
            },
            path,
        )
        checkpoints.append(_identity(path, update=update))
        return path

    if args.mode == "run":
        save_checkpoint(0)
    for update_index in range(updates):
        batch = source_runtime._assemble_batch(
            trajectories,
            orientations,
            all_names,
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
                "completed_updates": update,
                "requested_updates": updates,
                "latest_position_l1_m": training_l1_m,
                "elapsed_seconds": time.perf_counter() - started,
                "official_eval_read": False,
            }
            _write_json(output_root / "progress.json", progress, immutable=False)
            print(json.dumps(progress, sort_keys=True), flush=True)

    if args.mode == "smoke":
        smoke = {
            "schema_version": 1,
            "contract": "deform-dlo2-local-residual-alltrain-smoke-v7",
            "official_eval_read": False,
            "official_eval_execution_authorized": False,
            "training": training_losses,
        }
        _write_json(output_root / "smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0

    indexed = {int(record["update"]): record for record in checkpoints}
    final_update = int(protocol["training"]["physical_checkpoint_update"])
    if final_update not in indexed:
        raise RuntimeError("all-train run omitted the frozen final checkpoint")
    final_checkpoint = indexed[final_update]
    bundle = torch.load(
        Path(str(final_checkpoint["path"])), map_location="cpu", weights_only=True
    )
    physical_rollout = posterior_runtime._evaluate_state(
        bundle["model_state_dict"],
        trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
        dlo_type="DLO2",
        node_count=node_count,
    )
    initial, action = local_runtime._causal_inputs(trajectories, all_names)
    local_model = fit_deform_local_residual(
        initial,
        action,
        np.asarray(physical_rollout["predictions"]),
        np.asarray(physical_rollout["targets"]),
        all_names,
        ridge=float(fixed["ridge"]),
        variance_floor_m2=float(
            protocol["local_residual"]["coordinate_variance_floor_m2"]
        ),
    )
    local_model_path = output_root / "local_residual_model.npz"
    np.savez_compressed(
        local_model_path, **serialize_deform_local_residual_model(local_model)
    )
    final_method = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-final-method-v7",
        "official_eval_read": False,
        "physical_checkpoint": final_checkpoint,
        "local_residual_model": _identity(local_model_path),
        "fixed_arm": fixed,
        "method_spec": _identity(method_spec_path),
        "window_schedule": _identity(schedule_path),
        "validation_reselection": False,
        "source_reselection": False,
        "target_reselection": False,
        "fallback": "alltrain-physical-checkpoint-exact",
    }
    final_method_path = output_root / "final_method.json"
    _write_json(final_method_path, final_method)
    result = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-alltrain-result-v7",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "official_eval_execution_authorized": True,
        "protocol": _identity(protocol_path),
        "source_authorization": source_authorization,
        "source_manifest": _identity(source_manifest_path),
        "window_schedule": _identity(schedule_path),
        "method_spec": _identity(method_spec_path),
        "final_method": _identity(final_method_path),
        "physical_checkpoint": final_checkpoint,
        "local_residual_model": _identity(local_model_path),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "training_losses": training_losses,
        "checkpoints": checkpoints,
    }
    result_path = output_root / "alltrain_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
