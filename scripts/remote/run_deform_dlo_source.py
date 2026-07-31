#!/usr/bin/env python3
"""Run the frozen source-only DEFORM DLO reproduction on an external checkout."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bayesian_phystwin.deform_dlo_source import (
    build_deform_dlo_source_manifest,
    choose_deform_validation_checkpoint,
    evaluate_deform_source_gate,
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dlo-type", default="DLO1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "run"),
        default="run",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object], *, immutable: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and immutable:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _assert_upstream(root: Path, expected_commit: str) -> dict[str, object]:
    root = root.resolve()
    actual_commit = subprocess.check_output(
        ("git", "-C", str(root), "rev-parse", "HEAD"), text=True
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"DEFORM upstream commit differs: {actual_commit} != {expected_commit}"
        )
    tracked_status = subprocess.check_output(
        ("git", "-C", str(root), "status", "--short", "--untracked-files=no"),
        text=True,
    ).strip()
    if tracked_status:
        raise RuntimeError("DEFORM upstream has tracked modifications")
    required = ("DEFORM_func.py", "DEFORM_sim.py", "train_DEFORM.py", "util.py")
    identities = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        identities[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    function_source = (root / "DEFORM_func.py").read_text(encoding="utf-8")
    if (
        "DenseLinearization" not in function_source
        or "CholeskyDenseSolver" not in function_source
    ):
        raise RuntimeError(
            "locked DEFORM checkout does not expose the dense solver path"
        )
    return {
        "root": str(root),
        "commit": actual_commit,
        "tracked_clean": True,
        "source_files": identities,
    }


def _install_dense_import_shim() -> dict[str, object]:
    sksparse = types.ModuleType("sksparse")
    cholmod = types.ModuleType("sksparse.cholmod")

    class Factor:
        pass

    def analyze_aat(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sparse CHOLMOD path invoked in dense-only DEFORM run")

    cholmod.Factor = Factor
    cholmod.analyze_AAt = analyze_aat
    sksparse.cholmod = cholmod
    sys.modules["sksparse"] = sksparse
    sys.modules["sksparse.cholmod"] = cholmod
    return {
        "contract": "sksparse-import-only-stub-v1",
        "sparse_path_invocation": "raises",
    }


def _load_upstream(root: Path) -> SimpleNamespace:
    _install_dense_import_shim()
    sys.path.insert(0, str(root.resolve()))
    from DEFORM_func import DEFORM_func
    from DEFORM_sim import DEFORM_sim
    from util import computeEdges

    return SimpleNamespace(
        DEFORM_func=DEFORM_func,
        DEFORM_sim=DEFORM_sim,
        computeEdges=computeEdges,
    )


def _install_eval_read_guard(eval_root: Path) -> None:
    forbidden = eval_root.resolve()

    def guard(event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args:
            return
        raw_path = args[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(raw_path).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return
        if path == forbidden or forbidden in path.parents:
            raise PermissionError(f"official DEFORM eval read is forbidden: {path}")

    sys.addaudithook(guard)


def _load_trajectory(path: Path, *, frame_count: int, node_count: int) -> np.ndarray:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    array = np.asarray(raw, dtype=np.float32)
    expected_shape = (frame_count, 3, node_count)
    if array.shape != expected_shape:
        raise ValueError(f"{path}: expected {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: trajectory contains non-finite values")
    nodes = np.transpose(array, (0, 2, 1)).copy()
    nodes[:, :, 2] = np.clip(nodes[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return nodes


def _load_named_trajectories(
    manifest: dict[str, object],
    names: list[str],
    *,
    frame_count: int,
    node_count: int,
) -> dict[str, np.ndarray]:
    identities = manifest["trajectories"]
    result = {}
    for name in names:
        identity = identities[name]
        path = Path(identity["path"])
        if sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"source trajectory changed after manifesting: {name}")
        result[name] = _load_trajectory(
            path,
            frame_count=frame_count,
            node_count=node_count,
        )
    return result


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_dlo_model(
    modules: SimpleNamespace,
    torch: Any,
    device: str,
    *,
    node_count: int,
) -> tuple[Any, Any]:
    if device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("registered DEFORM source run requires a CUDA device")
    if node_count < 5:
        raise ValueError("registered DEFORM source run requires at least five nodes")
    edge_count = node_count - 1
    model_function = modules.DEFORM_func(
        n_vert=node_count,
        n_edge=edge_count,
        device=device,
    )
    model = modules.DEFORM_sim(
        n_vert=node_count,
        n_edge=edge_count,
        pbd_iter=10,
        device=device,
    )
    model.DEFORM_func.bend_stiffness = torch.nn.Parameter(
        5e-5 * torch.ones((1, edge_count), device=device)
    )
    model.DEFORM_func.twist_stiffness = torch.nn.Parameter(
        2e-5 * torch.ones((1, edge_count), device=device)
    )
    return model_function, model


def _build_dlo1_model(
    modules: SimpleNamespace,
    torch: Any,
    device: str,
) -> tuple[Any, Any]:
    return _build_dlo_model(
        modules,
        torch,
        device,
        node_count=13,
    )


def _official_optimizer(torch: Any, model: Any) -> Any:
    network_lr = 1e-4
    lr_scale = 0.1
    groups = [
        {"params": model.integration_ratio, "lr": 1e-5 * lr_scale},
        {"params": model.velocity_ratio, "lr": 1e-5 * lr_scale},
        {"params": model.rest_vert, "lr": 1e-5 * lr_scale},
        {"params": model.mocap_mass, "lr": 1e-5 * lr_scale},
        {"params": model.DEFORM_func.bend_stiffness, "lr": 1e-11 * lr_scale},
        {"params": model.DEFORM_func.twist_stiffness, "lr": 1e-11 * lr_scale},
        {"params": model.vert_conv1.parameters(), "lr": network_lr * lr_scale},
        {"params": model.vert_conv2.parameters(), "lr": network_lr * lr_scale},
        {"params": model.delta_vert_conv1.parameters(), "lr": network_lr * lr_scale},
        {"params": model.delta_vert_conv2.parameters(), "lr": network_lr * lr_scale},
        {"params": model.fc.parameters(), "lr": network_lr * lr_scale},
    ]
    return torch.optim.SGD(groups)


def _initial_direction(torch: Any, device: str) -> Any:
    return torch.tensor(
        ((0.0, 0.6, 0.8), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _precompute_material_u0(
    trajectories: dict[str, np.ndarray],
    *,
    modules: SimpleNamespace,
    model_function: Any,
    torch: Any,
    device: str,
) -> dict[str, Any]:
    initial = _initial_direction(torch, device)
    result = {}
    with torch.no_grad():
        for name, array in trajectories.items():
            vertices = torch.from_numpy(array).to(device=device)
            sequence = []
            current_u0 = None
            for index in range(array.shape[0] - 2):
                if index == 0:
                    current_edges = modules.computeEdges(
                        vertices[index + 1].unsqueeze(0)
                    )
                    current_u0 = model_function.compute_u0(
                        current_edges[:, 0].float(), initial[:, 0]
                    )
                else:
                    previous_edges = modules.computeEdges(vertices[index].unsqueeze(0))
                    current_edges = modules.computeEdges(
                        vertices[index + 1].unsqueeze(0)
                    )
                    current_u0 = model_function.parallelTransportFrame(
                        previous_edges[:, 0], current_edges[:, 0], current_u0
                    )
                sequence.append(current_u0.squeeze(0).detach().cpu())
            result[name] = torch.stack(sequence)
    return result


def _make_schedule(
    *,
    fit_names: list[str],
    updates: int,
    batch_size: int,
    frame_count: int,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    trajectory_indices = rng.integers(
        0,
        len(fit_names),
        size=(updates, batch_size),
        endpoint=False,
        dtype=np.int64,
    )
    start_indices = rng.integers(
        0,
        frame_count - 1 - horizon,
        size=(updates, batch_size),
        endpoint=False,
        dtype=np.int64,
    )
    return trajectory_indices, start_indices


def _assemble_batch(
    trajectories: dict[str, np.ndarray],
    orientations: dict[str, Any],
    fit_names: list[str],
    trajectory_indices: np.ndarray,
    start_indices: np.ndarray,
    *,
    horizon: int,
    torch: Any,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    previous = []
    current = []
    target = []
    material_u0 = []
    for trajectory_index, start in zip(
        trajectory_indices.tolist(), start_indices.tolist(), strict=True
    ):
        name = fit_names[trajectory_index]
        array = trajectories[name]
        previous.append(array[start : start + horizon])
        current.append(array[start + 1 : start + 1 + horizon])
        target.append(array[start + 2 : start + 2 + horizon])
        material_u0.append(orientations[name][start : start + horizon])
    return (
        torch.from_numpy(np.stack(previous)).to(device=device),
        torch.from_numpy(np.stack(current)).to(device=device),
        torch.from_numpy(np.stack(target)).to(device=device),
        torch.stack(material_u0).to(device=device),
    )


def _train_update(
    *,
    modules: SimpleNamespace,
    model_function: Any,
    model: Any,
    optimizer: Any,
    batch: tuple[Any, Any, Any, Any],
    torch: Any,
    device: str,
) -> float:
    previous_vertices, vertices, target_vertices, material_u0 = batch
    batch_size, horizon, node_count, _ = vertices.shape
    clamped_selection = torch.tensor((0, 1, -2, -1), device=device)
    clamped_index = torch.zeros(node_count, device=device)
    clamped_index[clamped_selection] = 1.0
    initial = _initial_direction(torch, device).repeat(batch_size, 1, 1)
    inputs = target_vertices[:, :, clamped_selection]
    theta_full = torch.zeros(batch_size, node_count - 1, device=device)
    loss_function = torch.nn.L1Loss()
    loss = torch.zeros((), device=device)
    position_loss = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    propagated_u0 = None
    pred_vertices = None
    vert = None
    current_velocity = None
    for frame in range(horizon):
        if frame == 0:
            current_velocity = (vertices[:, frame] - previous_vertices[:, frame]).div(
                model.dt
            )
            pred_vertices, current_velocity, theta_full = model(
                vertices[:, frame],
                current_velocity,
                initial,
                clamped_index,
                material_u0[:, frame],
                inputs[:, frame],
                clamped_selection,
                theta_full,
            )
        else:
            if frame == 1:
                previous_vert = previous_vertices[:, frame]
                frame_u0 = material_u0[:, frame]
            else:
                previous_vert = vert
                frame_u0 = propagated_u0
            vert = pred_vertices
            previous_edges = modules.computeEdges(previous_vert)
            current_edges = modules.computeEdges(vert)
            propagated_u0 = model_function.parallelTransportFrame(
                previous_edges[:, 0], current_edges[:, 0], frame_u0
            )
            pred_vertices, current_velocity, theta_full = model(
                vert.clone(),
                current_velocity.clone(),
                initial,
                clamped_index,
                propagated_u0,
                inputs[:, frame],
                clamped_selection,
                theta_full,
            )
        target_velocity = (target_vertices[:, frame] - vertices[:, frame]).div(model.dt)
        frame_position_loss = loss_function(pred_vertices, target_vertices[:, frame])
        frame_velocity_loss = loss_function(current_velocity, target_velocity)
        loss = loss + frame_position_loss + frame_velocity_loss
        position_loss = position_loss + frame_position_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("DEFORM training loss became non-finite")
    loss.backward(retain_graph=True)
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("DEFORM training gradient became non-finite")
    optimizer.step()
    return float((position_loss / horizon).detach().cpu())


def _rollout_records(
    trajectories: dict[str, np.ndarray],
    *,
    modules: SimpleNamespace,
    model_function: Any,
    model: Any,
    torch: Any,
    device: str,
) -> list[dict[str, object]]:
    names = list(trajectories)
    arrays = np.stack([trajectories[name] for name in names])
    values = torch.from_numpy(arrays).to(device=device)
    previous = values[:, :-2]
    vertices = values[:, 1:-1]
    targets = values[:, 2:]
    batch_size, horizon, node_count, _ = targets.shape
    clamped_selection = torch.tensor((0, 1, -2, -1), device=device)
    clamped_index = torch.zeros(node_count, device=device)
    clamped_index[clamped_selection] = 1.0
    initial = _initial_direction(torch, device).repeat(batch_size, 1, 1)
    inputs = targets[:, :, clamped_selection]
    theta_full = torch.zeros(batch_size, node_count - 1, device=device)
    total_absolute = torch.zeros(batch_size, device=device)
    thirds = torch.zeros(batch_size, 3, device=device)
    third_counts = torch.zeros(3, dtype=torch.int64, device=device)
    model.eval()
    with torch.no_grad():
        rest_edges = modules.computeEdges(vertices[:, 0])
        material_u0 = model_function.compute_u0(rest_edges[:, 0].float(), initial[:, 0])
        current_velocity = (vertices[:, 0] - previous[:, 0]).div(model.dt)
        rest_lengths = model.m_restEdgeL.repeat(batch_size, 1)
        model.m_restWprev, model.m_restWnext, model.learned_pmass = model.Rod_Init(
            batch_size,
            initial,
            rest_lengths,
            clamped_index,
        )
        pred_vertices = None
        vert = None
        for frame in range(horizon):
            if frame == 0:
                pred_vertices, current_velocity, theta_full = model(
                    vertices[:, frame],
                    current_velocity,
                    initial,
                    clamped_index,
                    material_u0,
                    inputs[:, frame],
                    clamped_selection,
                    theta_full,
                    mode="evaluation",
                )
            else:
                if frame == 1:
                    previous_vert = previous[:, frame]
                else:
                    previous_vert = vert
                vert = pred_vertices
                previous_edges = modules.computeEdges(previous_vert)
                current_edges = modules.computeEdges(vert)
                material_u0 = model_function.parallelTransportFrame(
                    previous_edges[:, 0], current_edges[:, 0], material_u0
                )
                pred_vertices, current_velocity, theta_full = model(
                    vert.clone(),
                    current_velocity.clone(),
                    initial,
                    clamped_index,
                    material_u0,
                    inputs[:, frame],
                    clamped_selection,
                    theta_full,
                    mode="evaluation",
                )
            absolute = torch.abs(pred_vertices - targets[:, frame]).sum(dim=(1, 2))
            total_absolute += absolute
            third = min(2, (3 * frame) // horizon)
            thirds[:, third] += absolute
            third_counts[third] += node_count * 3
    denominator = horizon * node_count * 3
    model_errors = total_absolute / denominator
    persistence = vertices[:, 0].unsqueeze(1).repeat(1, horizon, 1, 1)
    persistence[:, :, clamped_selection] = targets[:, :, clamped_selection]
    persistence_errors = torch.mean(torch.abs(persistence - targets), dim=(1, 2, 3))
    third_errors = thirds / third_counts.unsqueeze(0)
    records = []
    for index, name in enumerate(names):
        records.append(
            {
                "name": name,
                "model_l1_m": float(model_errors[index].cpu()),
                "persistence_l1_m": float(persistence_errors[index].cpu()),
                "early_l1_m": float(third_errors[index, 0].cpu()),
                "middle_l1_m": float(third_errors[index, 1].cpu()),
                "late_l1_m": float(third_errors[index, 2].cpu()),
            }
        )
    return records


def _checkpoint_identity(path: Path, update: int) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "update": update,
    }


def main() -> int:
    args = _parse_args()
    protocol = load_deform_dlo_source_protocol(args.protocol)
    if args.dlo_type not in protocol["dlo_types"]:
        raise ValueError("requested DLO type is outside the registered protocol")
    upstream = _assert_upstream(args.upstream_root, protocol["upstream"]["commit"])
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = args.upstream_root.resolve() / "data_set"
    manifest = build_deform_dlo_source_manifest(
        args.protocol,
        data_root,
        dlo_type=args.dlo_type,
    )
    manifest_path = output_root / "source_manifest.json"
    _write_json(manifest_path, manifest, immutable=True)
    _install_eval_read_guard(data_root / args.dlo_type / "eval")

    frame_count = int(protocol["data"]["expected_frames_per_trajectory"])
    node_count = int(protocol["data"]["expected_node_count"][args.dlo_type])
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    source_test_names = list(manifest["split"]["source_test"])
    development_names = fit_names + ([] if args.mode == "smoke" else validation_names)
    development_trajectories = _load_named_trajectories(
        manifest,
        development_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo-source-preflight-v1",
        "mode": args.mode,
        "official_eval_read": False,
        "source_test_opened": False,
        "source_test_hash_bound_count": len(source_test_names),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "upstream": upstream,
        "validated_development_trajectory_count": len(development_trajectories),
        "frame_count": frame_count,
        "node_count": node_count,
    }
    _write_json(output_root / "preflight.json", preflight, immutable=True)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    cublas_config = str(protocol["training"]["cublas_workspace_config"])
    existing_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas_config not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    modules = _load_upstream(args.upstream_root)
    seed = int(protocol["training"]["random_seed"])
    _seed_everything(torch, seed)
    model_function, model = _build_dlo_model(
        modules,
        torch,
        args.device,
        node_count=node_count,
    )
    optimizer = _official_optimizer(torch, model)
    fit_trajectories = {name: development_trajectories[name] for name in fit_names}
    orientations = _precompute_material_u0(
        fit_trajectories,
        modules=modules,
        model_function=model_function,
        torch=torch,
        device=args.device,
    )
    registered_updates = int(protocol["training"]["total_updates"])
    updates = 1 if args.mode == "smoke" else registered_updates
    batch_size = int(protocol["training"]["batch_size"])
    horizon = int(protocol["training"]["unroll_horizon_frames"])
    trajectory_indices, start_indices = _make_schedule(
        fit_names=fit_names,
        updates=registered_updates,
        batch_size=batch_size,
        frame_count=frame_count,
        horizon=horizon,
        seed=seed,
    )
    schedule_path = output_root / "window_schedule.npz"
    np.savez_compressed(
        schedule_path,
        fit_names=np.asarray(fit_names),
        trajectory_indices=trajectory_indices,
        start_indices=start_indices,
    )
    training_losses = []
    checkpoint_records = []
    validation_records = []
    checkpoint_updates = set(
        int(value) for value in protocol["training"]["checkpoint_updates"]
    )
    started = time.perf_counter()

    def save_checkpoint(update: int) -> Path:
        checkpoint_path = output_root / "checkpoints" / f"update_{update:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": update,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "protocol_sha256": sha256_file(args.protocol),
                "schedule_sha256": sha256_file(schedule_path),
            },
            checkpoint_path,
        )
        checkpoint_records.append(_checkpoint_identity(checkpoint_path, update))
        return checkpoint_path

    if args.mode == "run":
        checkpoint_path = save_checkpoint(0)
        validation = _rollout_records(
            {name: development_trajectories[name] for name in validation_names},
            modules=modules,
            model_function=model_function,
            model=model,
            torch=torch,
            device=args.device,
        )
        validation_records.append(
            {
                "update": 0,
                "validation_l1_m": float(
                    np.mean([record["model_l1_m"] for record in validation])
                ),
                "checkpoint": _checkpoint_identity(checkpoint_path, 0),
                "cases": validation,
            }
        )

    for update_index in range(updates):
        batch = _assemble_batch(
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
        training_l1_m = _train_update(
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
            checkpoint_path = save_checkpoint(update)
            validation = _rollout_records(
                {name: development_trajectories[name] for name in validation_names},
                modules=modules,
                model_function=model_function,
                model=model,
                torch=torch,
                device=args.device,
            )
            validation_records.append(
                {
                    "update": update,
                    "validation_l1_m": float(
                        np.mean([record["model_l1_m"] for record in validation])
                    ),
                    "checkpoint": _checkpoint_identity(checkpoint_path, update),
                    "cases": validation,
                }
            )
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
        fit_rollout = _rollout_records(
            {fit_names[0]: fit_trajectories[fit_names[0]]},
            modules=modules,
            model_function=model_function,
            model=model,
            torch=torch,
            device=args.device,
        )
        result = {
            "schema_version": 1,
            "contract": "deform-dlo-runtime-smoke-v1",
            "official_eval_read": False,
            "registered_source_gate_evaluated": False,
            "source_test_opened": False,
            "training": training_losses,
            "fit_rollout": fit_rollout,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "smoke_result.json", result, immutable=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    selected = choose_deform_validation_checkpoint(validation_records)
    selected_checkpoint = Path(selected["checkpoint"]["path"])
    bundle = torch.load(selected_checkpoint, map_location=args.device)
    model.load_state_dict(bundle["model_state_dict"])
    source_test_trajectories = _load_named_trajectories(
        manifest,
        source_test_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    source_test_records = _rollout_records(
        source_test_trajectories,
        modules=modules,
        model_function=model_function,
        model=model,
        torch=torch,
        device=args.device,
    )
    gate_config = protocol["source_gate"]
    gate = evaluate_deform_source_gate(
        source_test_records,
        published_reference_l1_m=float(
            gate_config["published_reference_l1_m"][args.dlo_type]
        ),
        published_error_multiplier_max=float(
            gate_config["published_error_multiplier_max"]
        ),
        minimum_persistence_wins=int(gate_config["minimum_persistence_wins"]),
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo-source-reproduction-result-v1",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "window_schedule": {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
        },
        "upstream": upstream,
        "runtime": {
            "python": sys.version,
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
        "source_test": source_test_records,
        "source_gate": gate,
        "advancement_authorized": bool(gate["passed"]),
    }
    result_path = output_root / "source_result.json"
    _write_json(result_path, result, immutable=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
