#!/usr/bin/env python3
"""Refit the confirmed DLO2 posterior method on all official training data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import run_deform_dlo_checkpoint_belief as source_belief_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_alltrain_protocol,
    validate_deform_dlo2_alltrain_authorization,
)
from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    average_deform_checkpoint_states,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--posterior-result", type=Path, required=True)
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


def _verified_selection_seal(
    posterior_result: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    identity = posterior_result.get("selection_seal")
    if not isinstance(identity, dict):
        raise ValueError("DLO2 posterior omits its selection-seal identity")
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError("DLO2 posterior selection seal does not verify")
    return path, _read_json(path)


def _verified_manifest(
    source_result: dict[str, object],
    *,
    source_protocol_sha256: str,
) -> tuple[Path, dict[str, object]]:
    identity = source_result.get("source_manifest")
    if not isinstance(identity, dict):
        raise ValueError("DLO2 source result omits its manifest identity")
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError("DLO2 source manifest does not verify")
    manifest = _read_json(path)
    protocol_identity = manifest.get("protocol")
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO2"
        or manifest.get("partition") != "train"
        or manifest.get("official_eval_read") is not False
        or not isinstance(protocol_identity, dict)
        or protocol_identity.get("sha256") != source_protocol_sha256
    ):
        raise ValueError("DLO2 all-train manifest lineage differs")
    return path, manifest


def main() -> int:
    args = _parse_args()
    protocol = load_deform_dlo2_alltrain_protocol(args.protocol)
    source_protocol = load_deform_dlo_source_protocol(args.source_protocol)
    source_protocol_sha256 = sha256_file(args.source_protocol)
    if source_protocol_sha256 != protocol["parent_source_protocol"][
        "sha256"
    ] or source_protocol["dlo_types"] != ("DLO2",):
        raise ValueError("all-train refit binds a different DLO2 source protocol")

    source_result_path = args.source_result.resolve()
    posterior_result_path = args.posterior_result.resolve()
    source_result = _read_json(source_result_path)
    posterior_result = _read_json(posterior_result_path)
    source_belief_runtime._validate_source_result(
        source_result,
        source_protocol_sha256=source_protocol_sha256,
        upstream_commit=str(source_protocol["upstream"]["commit"]),
    )
    selection_path, selection_seal = _verified_selection_seal(posterior_result)
    selected_method = validate_deform_dlo2_alltrain_authorization(
        protocol,
        source_result,
        posterior_result,
        selection_seal,
        source_protocol_sha256=source_protocol_sha256,
        source_result_sha256=sha256_file(source_result_path),
    )
    manifest_path, manifest = _verified_manifest(
        source_result,
        source_protocol_sha256=source_protocol_sha256,
    )
    upstream_commit = str(source_protocol["upstream"]["commit"])
    upstream = source_runtime._assert_upstream(args.upstream_root, upstream_commit)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")

    all_names = sorted(str(name) for name in manifest["trajectories"])
    expected_count = int(protocol["data"]["trajectory_count"])
    if len(all_names) != expected_count:
        raise ValueError("DLO2 all-train manifest does not contain all trajectories")
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
        "contract": "deform-dlo2-alltrain-method-spec-v1",
        "official_eval_read": False,
        "operator": selected_method["operator"],
        "checkpoint_weights": selected_method["weights"],
        "comparison_baseline_update": selected_method[
            "comparison_baseline_update"
        ],
        "validation_fitted_variance_scale": selected_method[
            "validation_fitted_variance_scale"
        ],
        "variance_floor_m2": selected_method["variance_floor_m2"],
        "nominal_coordinate_coverage": selected_method["nominal_coordinate_coverage"],
        "source_protocol": {
            "path": str(args.source_protocol.resolve()),
            "sha256": source_protocol_sha256,
        },
        "source_result": {
            "path": str(source_result_path),
            "sha256": sha256_file(source_result_path),
        },
        "posterior_result": {
            "path": str(posterior_result_path),
            "sha256": sha256_file(posterior_result_path),
        },
        "selection_seal": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
    }
    method_spec_path = output_root / "method_spec.json"
    _write_json(method_spec_path, method_spec)
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo2-alltrain-preflight-v1",
        "mode": args.mode,
        "official_eval_read": False,
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
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
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    source_runtime_info = source_result.get("runtime")
    if (
        not isinstance(source_runtime_info, dict)
        or torch.__version__ != source_runtime_info.get("torch")
        or torch.version.cuda != source_runtime_info.get("cuda")
    ):
        raise RuntimeError("all-train runtime differs from fresh DLO2 source")
    modules = source_runtime._load_upstream(args.upstream_root)
    seed = int(protocol["training"]["random_seed"])
    source_runtime._seed_everything(torch, seed)
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        args.device,
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
    protocol_sha256 = sha256_file(args.protocol)
    schedule_sha256 = sha256_file(schedule_path)
    method_spec_sha256 = sha256_file(method_spec_path)
    checkpoint_updates = set(protocol["checkpoint_updates"])
    checkpoints = []
    training_losses = []
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
        checkpoints.append(source_runtime._checkpoint_identity(path, update))
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
        result = {
            "schema_version": 1,
            "contract": "deform-dlo2-alltrain-smoke-v1",
            "official_eval_read": False,
            "official_eval_execution_authorized": False,
            "training": training_losses,
        }
        _write_json(output_root / "smoke_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    indexed_checkpoints = {int(record["update"]): record for record in checkpoints}
    weights = selected_method["weights"]
    baseline_update = int(selected_method["comparison_baseline_update"])
    if (
        baseline_update not in indexed_checkpoints
        or not set(weights).issubset(indexed_checkpoints)
    ):
        raise RuntimeError("all-train run omitted a selected posterior checkpoint")
    selected_members = {
        str(update): indexed_checkpoints[update] for update in sorted(weights)
    }
    parameter_mean_identity = None
    if selected_method["operator"] == "parameter_mean":
        states = {}
        for update in sorted(weights):
            bundle = torch.load(
                Path(str(indexed_checkpoints[update]["path"])),
                map_location="cpu",
                weights_only=True,
            )
            states[update] = bundle["model_state_dict"]
        averaged = average_deform_checkpoint_states(states, weights)
        parameter_mean_path = output_root / "final_parameter_mean.pt"
        torch.save(
            {
                "model_state_dict": averaged,
                "alltrain_protocol_sha256": protocol_sha256,
                "schedule_sha256": schedule_sha256,
                "method_spec_sha256": method_spec_sha256,
            },
            parameter_mean_path,
        )
        parameter_mean_identity = {
            "path": str(parameter_mean_path),
            "sha256": sha256_file(parameter_mean_path),
            "size_bytes": parameter_mean_path.stat().st_size,
        }
    final_method = {
        "schema_version": 1,
        "contract": "deform-dlo2-alltrain-final-method-v1",
        "official_eval_read": False,
        "operator": selected_method["operator"],
        "checkpoint_weights": weights,
        "comparison_baseline_checkpoint": indexed_checkpoints[baseline_update],
        "member_checkpoints": selected_members,
        "parameter_mean_checkpoint": parameter_mean_identity,
        "method_spec": {
            "path": str(method_spec_path),
            "sha256": method_spec_sha256,
        },
        "variance_calibration": {
            "scale": selected_method["validation_fitted_variance_scale"],
            "floor_m2": selected_method["variance_floor_m2"],
            "nominal_coordinate_coverage": selected_method[
                "nominal_coordinate_coverage"
            ],
        },
    }
    final_method_path = output_root / "final_method.json"
    _write_json(final_method_path, final_method)
    result = {
        "schema_version": 1,
        "contract": "deform-dlo2-alltrain-result-v1",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "official_eval_execution_authorized": True,
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": protocol_sha256,
        },
        "window_schedule": {
            "path": str(schedule_path),
            "sha256": schedule_sha256,
        },
        "method_spec": {
            "path": str(method_spec_path),
            "sha256": method_spec_sha256,
        },
        "final_method": {
            "path": str(final_method_path),
            "sha256": sha256_file(final_method_path),
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
    result_path = output_root / "alltrain_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
