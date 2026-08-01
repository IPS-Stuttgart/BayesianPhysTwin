#!/usr/bin/env python3
"""Refit one authorized DLO2 ensemble member on all training trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import run_deform_dlo2_alltrain as single_alltrain_runtime
import run_deform_dlo_checkpoint_belief as source_belief_runtime
import run_deform_dlo_deep_ensemble as ensemble_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_deep_alltrain_protocol,
    validate_deform_dlo2_deep_alltrain_authorization,
)
from bayesian_phystwin.deform_dlo_deep_ensemble import (
    load_deform_dlo2_deep_ensemble_protocol,
    validate_deform_two_seed_manifests,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ensemble-protocol", type=Path, required=True)
    parser.add_argument("--ensemble-result", type=Path, required=True)
    parser.add_argument("--seed42-source-protocol", type=Path, required=True)
    parser.add_argument("--seed42-source-result", type=Path, required=True)
    parser.add_argument("--seed43-source-protocol", type=Path, required=True)
    parser.add_argument("--seed43-source-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "run"),
        default="run",
    )
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


def _checkpoint_identity(path: Path, update: int) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "update": update,
    }


def _verified_selection_seal(
    ensemble_result: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    identity = ensemble_result.get("selection_seal")
    if not isinstance(identity, Mapping):
        raise ValueError("DLO2 ensemble result omits its selection seal")
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError("DLO2 ensemble selection seal does not verify")
    return path, _read_json(path)


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo2_deep_alltrain_protocol(protocol_path)
    ensemble_protocol_path = args.ensemble_protocol.resolve()
    load_deform_dlo2_deep_ensemble_protocol(ensemble_protocol_path)
    parents = protocol["parents"]
    if sha256_file(ensemble_protocol_path) != parents["ensemble_protocol"]["sha256"]:
        raise ValueError("deep all-train binds a different ensemble protocol")

    source_protocol_paths = {
        42: args.seed42_source_protocol.resolve(),
        43: args.seed43_source_protocol.resolve(),
    }
    source_result_paths = {
        42: args.seed42_source_result.resolve(),
        43: args.seed43_source_result.resolve(),
    }
    source_protocols = {}
    source_results = {}
    manifests = {}
    for seed in (42, 43):
        source_protocols[seed] = load_deform_dlo_source_protocol(
            source_protocol_paths[seed]
        )
        expected_identity = parents[f"seed{seed}_source_protocol"]
        if (
            sha256_file(source_protocol_paths[seed])
            != expected_identity["sha256"]
            or int(source_protocols[seed]["training"]["random_seed"]) != seed
        ):
            raise ValueError(f"deep all-train seed-{seed} protocol differs")
        source_results[seed] = _read_json(source_result_paths[seed])
        source_belief_runtime._validate_source_result(
            source_results[seed],
            source_protocol_sha256=sha256_file(source_protocol_paths[seed]),
            upstream_commit=str(source_protocols[seed]["upstream"]["commit"]),
        )
        _, manifests[seed] = single_alltrain_runtime._verified_manifest(
            source_results[seed],
            source_protocol_sha256=sha256_file(source_protocol_paths[seed]),
        )
    validate_deform_two_seed_manifests(
        manifests[42], manifests[43], dlo_type="DLO2"
    )
    if ensemble_runtime._runtime_identity(
        source_results[42]
    ) != ensemble_runtime._runtime_identity(source_results[43]):
        raise ValueError("deep all-train source runtimes differ")

    ensemble_result_path = args.ensemble_result.resolve()
    ensemble_result = _read_json(ensemble_result_path)
    selection_path, selection_seal = _verified_selection_seal(ensemble_result)
    selected = validate_deform_dlo2_deep_alltrain_authorization(
        protocol,
        source_results,
        ensemble_result,
        selection_seal,
        source_protocol_sha256s={
            seed: sha256_file(source_protocol_paths[seed]) for seed in (42, 43)
        },
        source_result_sha256s={
            seed: sha256_file(source_result_paths[seed]) for seed in (42, 43)
        },
        ensemble_protocol_sha256=sha256_file(ensemble_protocol_path),
        selection_seal_sha256=sha256_file(selection_path),
    )

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_commit = str(source_protocols[args.seed]["upstream"]["commit"])
    upstream = source_runtime._assert_upstream(args.upstream_root, upstream_commit)
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")

    manifest = manifests[args.seed]
    all_names = sorted(str(name) for name in manifest["trajectories"])
    if len(all_names) != int(protocol["data"]["trajectory_count"]):
        raise ValueError("deep all-train manifest does not contain all trajectories")
    frame_count = int(protocol["data"]["frame_count"])
    node_count = int(protocol["data"]["node_count"])
    trajectories = source_runtime._load_named_trajectories(
        manifest,
        all_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    method_spec = {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-alltrain-seed-method-v1",
        "official_eval_read": False,
        "seed": args.seed,
        "operator": selected["operator"],
        "seed_weight": selected["weights"][args.seed],
        "selected_update": selected["member_updates"][args.seed],
        "comparison_baseline_seed": selected["comparison_baseline_seed"],
        "selected_arm": selected["selected_arm"],
        "variance_calibration": {
            "scale": selected["validation_fitted_variance_scale"],
            "floor_m2": selected["variance_floor_m2"],
            "nominal_coordinate_coverage": selected[
                "nominal_coordinate_coverage"
            ],
        },
        "ensemble_result": {
            "path": str(ensemble_result_path),
            "sha256": sha256_file(ensemble_result_path),
        },
        "selection_seal": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "source_results": {
            str(seed): {
                "path": str(source_result_paths[seed]),
                "sha256": sha256_file(source_result_paths[seed]),
            }
            for seed in (42, 43)
        },
    }
    method_spec_path = output_root / "method_spec.json"
    _write_json(method_spec_path, method_spec)
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-alltrain-seed-preflight-v1",
        "mode": args.mode,
        "official_eval_read": False,
        "seed": args.seed,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "method_spec": {
            "path": str(method_spec_path),
            "sha256": sha256_file(method_spec_path),
        },
        "upstream": upstream,
        "validated_train_trajectory_count": len(trajectories),
        "frame_count": frame_count,
        "node_count": node_count,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    cublas_config = str(protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("deep all-train cuBLAS configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    source_runtime_info = source_results[args.seed].get("runtime")
    if (
        not isinstance(source_runtime_info, Mapping)
        or torch.__version__ != source_runtime_info.get("torch")
        or torch.version.cuda != source_runtime_info.get("cuda")
    ):
        raise RuntimeError("deep all-train runtime differs from fresh source")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, args.seed)
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
        seed=args.seed,
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
    checkpoints = []
    training_losses = []
    started = time.perf_counter()

    def save_checkpoint(update: int) -> None:
        path = output_root / "checkpoints" / f"update_{update:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seed": args.seed,
                "update": update,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "deep_alltrain_protocol_sha256": protocol_sha256,
                "schedule_sha256": schedule_sha256,
                "method_spec_sha256": method_spec_sha256,
            },
            path,
        )
        checkpoints.append(_checkpoint_identity(path, update))

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
                "seed": args.seed,
                "completed_updates": update,
                "requested_updates": updates,
                "latest_position_l1_m": training_l1_m,
                "elapsed_seconds": time.perf_counter() - started,
                "official_eval_read": False,
            }
            _write_json(output_root / "progress.json", progress, immutable=False)
            print(json.dumps(progress, sort_keys=True), flush=True)

    if args.mode == "smoke":
        result = {
            "schema_version": 1,
            "contract": "deform-dlo2-deep-alltrain-seed-smoke-v1",
            "seed": args.seed,
            "official_eval_read": False,
            "assembly_authorized": False,
            "training": training_losses,
        }
        _write_json(output_root / "smoke_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    indexed = {int(record["update"]): record for record in checkpoints}
    selected_update = int(selected["member_updates"][args.seed])
    if selected_update not in indexed:
        raise RuntimeError("deep all-train omitted the selected member checkpoint")
    final_member = {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-alltrain-seed-final-v1",
        "official_eval_read": False,
        "seed": args.seed,
        "operator": selected["operator"],
        "weight": selected["weights"][args.seed],
        "selected_update": selected_update,
        "selected_checkpoint": indexed[selected_update],
        "method_spec": {
            "path": str(method_spec_path),
            "sha256": method_spec_sha256,
        },
        "window_schedule": {
            "path": str(schedule_path),
            "sha256": schedule_sha256,
        },
    }
    final_member_path = output_root / "final_member.json"
    _write_json(final_member_path, final_member)
    result = {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-alltrain-seed-result-v1",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "seed": args.seed,
        "assembly_authorized": True,
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "ensemble_result": {
            "path": str(ensemble_result_path),
            "sha256": sha256_file(ensemble_result_path),
        },
        "method_spec": {
            "path": str(method_spec_path),
            "sha256": method_spec_sha256,
        },
        "window_schedule": {
            "path": str(schedule_path),
            "sha256": schedule_sha256,
        },
        "final_member": {
            "path": str(final_member_path),
            "sha256": sha256_file(final_member_path),
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
    }
    result_path = output_root / "alltrain_seed_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
