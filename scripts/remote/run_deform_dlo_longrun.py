#!/usr/bin/env python3
"""Continue the frozen DEFORM DLO1 source model under a new long-run protocol."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_longrun import load_deform_dlo_longrun_protocol
from bayesian_phystwin.deform_dlo_source import (
    choose_deform_validation_checkpoint,
    evaluate_deform_source_gate,
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--starting-checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("preflight", "run"),
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
    immutable: bool,
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and immutable:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _validate_parent(
    protocol: dict[str, object],
    source_result: dict[str, object],
    source_result_path: Path,
    source_manifest_path: Path,
    starting_checkpoint_path: Path,
) -> None:
    expected_result = protocol["source_result"]
    if sha256_file(source_result_path) != expected_result["sha256"]:
        raise ValueError("long-run parent source-result identity differs")
    source_gate = source_result.get("source_gate")
    if (
        source_result.get("contract") != "deform-dlo-source-reproduction-result-v1"
        or source_result.get("official_eval_read") is not False
        or source_result.get("advancement_authorized") is not False
        or not isinstance(source_gate, dict)
        or source_gate.get("passed") is not False
    ):
        raise ValueError("long-run parent must be the failed frozen source result")
    manifest_identity = source_result.get("source_manifest")
    if not isinstance(manifest_identity, dict) or sha256_file(
        source_manifest_path
    ) != manifest_identity.get("sha256"):
        raise ValueError("long-run source-manifest identity differs")
    selected = source_result.get("selected_checkpoint")
    expected_checkpoint = protocol["starting_checkpoint"]
    selected_checkpoint = (
        selected.get("checkpoint") if isinstance(selected, dict) else None
    )
    if (
        not isinstance(selected, dict)
        or not isinstance(selected_checkpoint, dict)
        or int(selected.get("update", -1)) != int(expected_checkpoint["global_update"])
        or selected_checkpoint.get("sha256") != expected_checkpoint["sha256"]
        or sha256_file(starting_checkpoint_path) != expected_checkpoint["sha256"]
        or starting_checkpoint_path.stat().st_size
        != int(expected_checkpoint["size_bytes"])
    ):
        raise ValueError("long-run starting-checkpoint identity differs")


def main() -> int:
    args = _parse_args()
    protocol = load_deform_dlo_longrun_protocol(args.protocol)
    source_protocol = load_deform_dlo_source_protocol(args.source_protocol)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_result_path = args.source_result.resolve()
    source_result = _read_json(source_result_path)
    source_manifest_path = args.source_manifest.resolve()
    starting_checkpoint_path = args.starting_checkpoint.resolve()
    _validate_parent(
        protocol,
        source_result,
        source_result_path,
        source_manifest_path,
        starting_checkpoint_path,
    )
    manifest = _read_json(source_manifest_path)
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO1"
        or manifest.get("official_eval_read") is not False
    ):
        raise ValueError("long-run requires the frozen DLO1 source manifest")

    upstream_commit = str(source_protocol["upstream"]["commit"])
    upstream = source_runtime._assert_upstream(args.upstream_root, upstream_commit)
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")

    training = protocol["training"]
    source_training = source_protocol["training"]
    if int(training["batch_size"]) != int(source_training["batch_size"]) or int(
        training["unroll_horizon_frames"]
    ) != int(source_training["unroll_horizon_frames"]):
        raise ValueError("long-run training geometry differs from the source run")
    cublas_config = str(training["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    parent_runtime = source_result["runtime"]
    if (
        torch.__version__ != parent_runtime["torch"]
        or torch.version.cuda != parent_runtime["cuda"]
    ):
        raise RuntimeError("long-run PyTorch/CUDA runtime differs from its parent")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(
        torch,
        int(source_protocol["training"]["random_seed"]),
    )
    model_function, model = source_runtime._build_dlo1_model(
        modules,
        torch,
        args.device,
    )
    optimizer = source_runtime._official_optimizer(torch, model)
    parent_bundle = torch.load(
        starting_checkpoint_path,
        map_location=args.device,
        weights_only=True,
    )
    if (
        int(parent_bundle.get("update", -1))
        != int(protocol["starting_checkpoint"]["global_update"])
        or parent_bundle.get("protocol_sha256") != sha256_file(args.source_protocol)
        or parent_bundle.get("schedule_sha256")
        != source_result["window_schedule"]["sha256"]
    ):
        raise ValueError("long-run parent checkpoint payload differs")
    model.load_state_dict(parent_bundle["model_state_dict"], strict=True)
    optimizer.load_state_dict(parent_bundle["optimizer_state_dict"])

    frame_count = int(source_protocol["data"]["expected_frames_per_trajectory"])
    node_count = int(source_protocol["data"]["expected_node_count"]["DLO1"])
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    development_names = fit_names + validation_names
    development = source_runtime._load_named_trajectories(
        manifest,
        development_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    fit_trajectories = {name: development[name] for name in fit_names}

    continuation_updates = int(training["continuation_updates"])
    batch_size = int(training["batch_size"])
    horizon = int(training["unroll_horizon_frames"])
    trajectory_indices, start_indices = source_runtime._make_schedule(
        fit_names=fit_names,
        updates=continuation_updates,
        batch_size=batch_size,
        frame_count=frame_count,
        horizon=horizon,
        seed=int(training["continuation_random_seed"]),
    )
    schedule_path = output_root / "continuation_schedule.npz"
    np.savez_compressed(
        schedule_path,
        parent_source_result_sha256=np.asarray(protocol["source_result"]["sha256"]),
        parent_global_update=np.asarray(
            int(protocol["starting_checkpoint"]["global_update"])
        ),
        fit_names=np.asarray(fit_names),
        trajectory_indices=trajectory_indices,
        start_indices=start_indices,
    )
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo-longrun-preflight-v2",
        "mode": args.mode,
        "official_eval_read": False,
        "source_test_opened": False,
        "parent_source_result_sha256": sha256_file(source_result_path),
        "parent_checkpoint_sha256": sha256_file(starting_checkpoint_path),
        "continuation_schedule": {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
        },
        "upstream": upstream,
        "validated_fit_trajectory_count": len(fit_names),
        "validated_validation_trajectory_count": len(validation_names),
        "continuation_updates": continuation_updates,
        "final_global_update": int(training["final_global_update"]),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
    }
    _write_json(output_root / "preflight.json", preflight, immutable=True)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    orientations = source_runtime._precompute_material_u0(
        fit_trajectories,
        modules=modules,
        model_function=model_function,
        torch=torch,
        device=args.device,
    )

    validation_records = [source_result["selected_checkpoint"]]
    checkpoint_records = []
    training_losses = []
    checkpoint_updates = set(int(value) for value in training["checkpoint_updates"])
    starting_update = int(protocol["starting_checkpoint"]["global_update"])
    started = time.perf_counter()

    def save_checkpoint(global_update: int) -> Path:
        path = output_root / "checkpoints" / f"update_{global_update:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_update": global_update,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "longrun_protocol_sha256": sha256_file(args.protocol),
                "source_protocol_sha256": sha256_file(args.source_protocol),
                "parent_checkpoint_sha256": protocol["starting_checkpoint"]["sha256"],
                "parent_source_result_sha256": protocol["source_result"]["sha256"],
                "continuation_schedule_sha256": sha256_file(schedule_path),
            },
            path,
        )
        identity = source_runtime._checkpoint_identity(path, global_update)
        checkpoint_records.append(identity)
        return path

    for continuation_index in range(continuation_updates):
        batch = source_runtime._assemble_batch(
            fit_trajectories,
            orientations,
            fit_names,
            trajectory_indices[continuation_index],
            start_indices[continuation_index],
            horizon=horizon,
            torch=torch,
            device=args.device,
        )
        update_started = time.perf_counter()
        position_l1_m = source_runtime._train_update(
            modules=modules,
            model_function=model_function,
            model=model,
            optimizer=optimizer,
            batch=batch,
            torch=torch,
            device=args.device,
        )
        torch.cuda.synchronize(args.device)
        global_update = starting_update + continuation_index + 1
        training_losses.append(
            {
                "global_update": global_update,
                "position_l1_m": position_l1_m,
                "seconds": time.perf_counter() - update_started,
            }
        )
        if global_update in checkpoint_updates:
            checkpoint_path = save_checkpoint(global_update)
            validation = source_runtime._rollout_records(
                {name: development[name] for name in validation_names},
                modules=modules,
                model_function=model_function,
                model=model,
                torch=torch,
                device=args.device,
            )
            validation_records.append(
                {
                    "update": global_update,
                    "validation_l1_m": float(
                        np.mean([record["model_l1_m"] for record in validation])
                    ),
                    "checkpoint": source_runtime._checkpoint_identity(
                        checkpoint_path,
                        global_update,
                    ),
                    "cases": validation,
                }
            )
        if global_update == starting_update + 1 or global_update % 10 == 0:
            progress = {
                "completed_continuation_updates": continuation_index + 1,
                "requested_continuation_updates": continuation_updates,
                "global_update": global_update,
                "latest_position_l1_m": position_l1_m,
                "elapsed_seconds": time.perf_counter() - started,
                "official_eval_read": False,
            }
            _write_json(output_root / "progress.json", progress, immutable=False)
            print(json.dumps(progress, sort_keys=True), flush=True)

    selected = choose_deform_validation_checkpoint(validation_records)
    selected_update = int(selected["update"])
    if selected_update == starting_update:
        selected_path = starting_checkpoint_path
    else:
        selected_path = next(
            Path(record["path"])
            for record in checkpoint_records
            if int(record["update"]) == selected_update
        )
    selected_bundle = torch.load(
        selected_path,
        map_location=args.device,
        weights_only=True,
    )
    model.load_state_dict(selected_bundle["model_state_dict"], strict=True)

    source_names = list(manifest["split"]["source_test"])
    source_trajectories = source_runtime._load_named_trajectories(
        manifest,
        source_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    source_test = source_runtime._rollout_records(
        source_trajectories,
        modules=modules,
        model_function=model_function,
        model=model,
        torch=torch,
        device=args.device,
    )
    source_gate = evaluate_deform_source_gate(
        source_test,
        published_reference_l1_m=float(
            protocol["source_gate"]["published_reference_l1_m"]
        ),
        published_error_multiplier_max=float(
            protocol["source_gate"]["published_error_multiplier_max"]
        ),
        minimum_persistence_wins=int(
            protocol["source_gate"]["minimum_persistence_wins"]
        ),
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo-longrun-result-v2",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "parent": {
            "source_result_path": str(source_result_path),
            "source_result_sha256": sha256_file(source_result_path),
            "starting_checkpoint_path": str(starting_checkpoint_path),
            "starting_checkpoint_sha256": sha256_file(starting_checkpoint_path),
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "source_protocol": {
            "path": str(args.source_protocol.resolve()),
            "sha256": sha256_file(args.source_protocol),
        },
        "continuation_schedule": {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
        },
        "upstream": upstream,
        "runtime": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "training_losses": training_losses,
        "checkpoints": checkpoint_records,
        "validation": validation_records,
        "selected_checkpoint": selected,
        "source_test": source_test,
        "source_gate": source_gate,
        "checkpoint_posterior_authorized": bool(source_gate["passed"]),
        "fresh_dlo2_reproduction_authorized": bool(source_gate["passed"]),
    }
    result_path = output_root / "longrun_result.json"
    _write_json(result_path, result, immutable=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
