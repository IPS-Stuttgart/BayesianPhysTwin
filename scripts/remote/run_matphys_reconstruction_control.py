#!/usr/bin/env python3
"""Run a clearly labeled all-frame MatPhys reconstruction control."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MATPHYS_PYDEPS = os.environ.get("MATPHYS_PYDEPS")
if _MATPHYS_PYDEPS:
    sys.path.append(_MATPHYS_PYDEPS)

from run_matphys_causal import (  # noqa: E402
    _ACCESSED_FRAME_PATHS,
    _ACCESSED_FRAMES,
    _FINITE_OPTIMIZER_STATS,
    _OBJECTIVE_END_FRAMES,
    FINITE_OPTIMIZER_CONTRACT,
    MATPHYS_REPOSITORY,
    _causal_video_loader,
    _checkpoint_finiteness_report,
    _collect_distributed_access_logs,
    _configure_matphys_imports,
    _install_finite_adamw,
    _install_source_supervised_objective_guard,
    _install_torchvision_nms_stub,
    _model_spring_y,
    _prepare_proxy,
    _rollout_model_output,
    _source_commit,
)

from bayesian_phystwin.matphys_causal_bridge import (  # noqa: E402
    sha256_file,
)
from bayesian_phystwin.matphys_reconstruction_control import (  # noqa: E402
    MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
    MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY,
    MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
    MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
    validate_matphys_reconstruction_audit,
    write_matphys_reconstruction_audit,
)
from bayesian_phystwin.phystwin_official_evaluation import (  # noqa: E402
    evaluate_official_phystwin_files,
)

GLOBAL_PARAMETER_NAMES = (
    "collide_elas",
    "collide_fric",
    "collide_object_elas",
    "collide_object_fric",
    "collision_dist",
    "dashpot_damping",
    "drag_damping",
)


def _single_case(value: str) -> str:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if len(values) != 1:
        raise ValueError("reconstruction control requires exactly one case")
    return values[0]


def _split(data_root: Path, case: str) -> dict[str, object]:
    path = data_root / case / "split.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    train_end = int(value["train"][1])
    frame_len = int(value.get("frame_len", value["test"][1]))
    if not 1 < train_end < frame_len:
        raise ValueError("reconstruction case must have a distinct future interval")
    return value


def _frame_lengths(data_root: Path, case: str) -> dict[str, int]:
    split = _split(data_root, case)
    return {case: int(split.get("frame_len", split["test"][1]))}


def _training_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "architecture": "released-SimpleVideoMaterialPhysicsModel",
        "case_count": 1,
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "learning_rate": float(args.learning_rate),
        "random_seed": int(args.random_seed),
        "fit_all_frames": True,
        "video_scope": MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
        "training_scope": MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
        "checkpoint_policy": MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
        "proxy_contract": "causal-dino-graph-voronoi-parts-v1",
        "missing_public_artifact_boundary": (
            "MatPhys does not release the final per-case train_ready.pt bundle; "
            "the registered deterministic DINO graph-part proxy is used."
        ),
        "videomae_model": str(args.videomae_model),
        "lambda_track": 1.0,
        "lambda_geo": 1.0,
        "lambda_render": 0.0,
        "lambda_acc_smooth": 0.01,
        "finite_optimizer_guard": FINITE_OPTIMIZER_CONTRACT,
    }


def _proxy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        proxy_root=args.proxy_root,
        graph_parts=True,
        compact_unused_edge_semantics=False,
        matphys_root=args.matphys_root,
        data_root=args.data_root,
        dino_model=args.dino_model,
        dino_image_size=args.dino_image_size,
        device=args.device,
        dino_keyframes=args.dino_keyframes,
        experiments_optimization_dir=args.experiments_optimization_dir,
        part_count=args.part_count,
        semantic_edge_weight=args.semantic_edge_weight,
    )


def train(args: argparse.Namespace) -> None:
    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    case = _single_case(args.case)
    frame_len_by_case = _frame_lengths(data_root, case)
    _ACCESSED_FRAMES.clear()
    _ACCESSED_FRAME_PATHS.clear()
    _OBJECTIVE_END_FRAMES.clear()
    for key in _FINITE_OPTIMIZER_STATS:
        _FINITE_OPTIMIZER_STATS[key] = 0
    if args.epochs < 1 or args.eval_every < 1 or args.epochs % args.eval_every:
        raise ValueError("epochs must be positive and divisible by eval_every")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if args.random_seed < 0:
        raise ValueError("random seed must be nonnegative")

    proxy = _prepare_proxy(_proxy_args(args), [case], frame_len_by_case)
    if proxy.get("contract") != "causal-dino-graph-voronoi-parts-v1":
        raise ValueError("reconstruction control requires the registered DINO proxy")
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    import train_model_video_material_simple as training

    training.load_video_frames = _causal_video_loader(frame_len_by_case)
    _install_source_supervised_objective_guard(training, data_root, frame_len_by_case)
    _install_finite_adamw(training)
    sys.argv = [
        "train_model_video_material_simple.py",
        "--case_name",
        case,
        "--save_dir",
        str(output_dir),
        "--base_path",
        str(data_root),
        "--experiments_dir",
        str(Path(args.experiments_dir).resolve()),
        "--experiments_optimization_dir",
        str(Path(args.experiments_optimization_dir).resolve()),
        "--case_to_material",
        str(proxy["mapping"]["path"]),
        "--results_dir",
        str(proxy["results_dir"]),
        "--sem_cache_dir",
        str(proxy["semantic_cache_dir"]),
        "--gaussian_root",
        "__disabled__",
        "--videomae_model",
        str(args.videomae_model),
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--epochs",
        str(args.epochs),
        "--eval_every",
        str(args.eval_every),
        "--device",
        str(args.device),
        "--lr",
        str(args.learning_rate),
        "--seed",
        str(args.random_seed),
        "--lambda_track",
        "1.0",
        "--lambda_geo",
        "1.0",
        "--lambda_render",
        "0.0",
        "--lambda_phys_prior",
        "0.0",
        "--lambda_acc_smooth",
        "0.01",
        "--grad_clip",
        "5.0",
        "--fit_all_frames",
        "--save_best_only",
        "--vis_every",
        "0",
    ]
    training.main()
    access_logs = _collect_distributed_access_logs(output_dir)
    if access_logs is None:
        raise RuntimeError("reconstruction control does not support nonzero DDP ranks")
    accessed_frames, accessed_paths, objective_ends, logs, optimizer_summaries = (
        access_logs
    )
    checkpoint = output_dir / "last_checkpoint.pth"
    if not checkpoint.is_file():
        raise RuntimeError("MatPhys training did not write its terminal checkpoint")
    finiteness = _checkpoint_finiteness_report(checkpoint)
    finiteness["optimizer_rank_summaries"] = optimizer_summaries
    finiteness_path = output_dir / "checkpoint_finiteness.json"
    finiteness_path.write_text(
        json.dumps(finiteness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logs.append(finiteness_path)
    if not finiteness["finite"]:
        raise RuntimeError("MatPhys terminal checkpoint is non-finite")
    if set(accessed_frames) != {case} or set(objective_ends) != {case}:
        raise RuntimeError(
            "reconstruction access log does not contain exactly one case"
        )
    audit = write_matphys_reconstruction_audit(
        checkpoint,
        output_dir / "reconstruction_training_audit.json",
        protocol_path=protocol_path,
        source_repository=MATPHYS_REPOSITORY,
        source_commit=_source_commit(matphys_root),
        data_root=data_root,
        case_name=case,
        split_path=data_root / case / "split.json",
        accessed_frame_indices=sorted(accessed_frames[case]),
        accessed_frame_paths=accessed_paths[case],
        objective_end_frame_exclusive=objective_ends[case],
        proxy_summary_path=proxy["summary_path"],
        training_configuration=_training_configuration(args),
        runtime_access_log_paths=logs,
        implementation_paths=(Path(__file__).resolve(),),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def _checkpoint_namespace(raw: dict[str, object], args: argparse.Namespace):
    values = dict(raw)
    values.update(
        {
            "base_path": str(Path(args.data_root).resolve()),
            "experiments_dir": str(Path(args.experiments_dir).resolve()),
            "experiments_optimization_dir": str(
                Path(args.experiments_optimization_dir).resolve()
            ),
            "case_to_material": str(
                Path(args.proxy_root).resolve() / "case_to_material.json"
            ),
            "results_dir": str(Path(args.proxy_root).resolve() / "results"),
            "sem_cache_dir": str(Path(args.proxy_root).resolve() / "semantic_cache"),
            "gaussian_root": "__disabled__",
            "device": str(args.device),
            "rank": 0,
            "lambda_render": 0.0,
            "fit_all_frames": True,
        }
    )
    values.setdefault("logk_residual_scale", 1.0)
    values.setdefault("logk_soft_clamp", 0.25)
    return SimpleNamespace(**values)


def _global_parameters(model_out: dict[str, object]) -> dict[str, float]:
    values = {}
    for name in GLOBAL_PARAMETER_NAMES:
        if name not in model_out:
            raise ValueError(f"MatPhys output omits {name}")
        value = np.asarray(model_out[name].detach().cpu(), dtype=np.float64).reshape(-1)
        if value.shape != (1,) or not np.isfinite(value[0]):
            raise ValueError(f"MatPhys output has invalid {name}")
        values[name] = float(value[0])
    return values


def export(args: argparse.Namespace) -> None:
    import torch

    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    case = _single_case(args.case)
    audit = validate_matphys_reconstruction_audit(args.audit, checkpoint_path)
    if audit["case"]["name"] != case:
        raise ValueError("reconstruction audit case differs from export request")
    if Path(audit["proxy"]["path"]).resolve() != (
        Path(args.proxy_root).resolve() / "proxy_summary.json"
    ):
        raise ValueError("export proxy differs from the audited training proxy")
    split = _split(data_root, case)
    frame_len = int(split.get("frame_len", split["test"][1]))
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    import warnings

    import warp._src.utils as warp_utils

    if not hasattr(warp_utils, "warn"):
        warp_utils.warn = warnings.warn
    import train_model_video_material_simple as training
    from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = _checkpoint_namespace(checkpoint["args"], args)
    device = torch.device(args.device)
    model = training.SimpleVideoMaterialPhysicsModel(
        videomae_model=model_args.videomae_model,
        d_motion=model_args.d_motion,
        d_mat=model_args.mat_codebook_dim,
        hidden_dim=model_args.hidden_dim,
        num_materials=model_args.num_materials,
        logk_base=model_args.logk_base,
        logk_min=model_args.logk_min,
        logk_max=model_args.logk_max,
        logk_residual_scale=model_args.logk_residual_scale,
        logk_soft_clamp=model_args.logk_soft_clamp,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = MaterialParamDataset(
        MaterialDatasetConfig(
            base_path=model_args.base_path,
            sem_cache_dir=model_args.sem_cache_dir,
            experiments_dir=model_args.experiments_dir,
            experiments_optimization_dir=model_args.experiments_optimization_dir,
            case_to_material_path=model_args.case_to_material,
            results_dir=model_args.results_dir,
            use_knn_topology=model_args.use_knn_topology,
            object_knn=model_args.object_knn,
            object_radius=model_args.object_radius,
            object_max_neighbours=model_args.object_max_neighbours,
            controller_radius=model_args.controller_radius,
            controller_max_neighbours=model_args.controller_max_neighbours,
        )
    )
    matches = [sample for sample in dataset.samples if sample["case_name"] == case]
    if len(matches) != 1:
        raise ValueError("audited proxy must contain exactly one requested sample")
    sample = matches[0]
    batch = {
        key: [value] if key != "case_name" else [case] for key, value in sample.items()
    }
    pixel_values = _causal_video_loader({case: frame_len})(
        case,
        model_args.base_path,
        T=model_args.num_video_frames,
        image_size=model_args.videomae_image_size,
        device=device,
    )
    train_end = int(sample["train_frame"].item())
    runtime = training._init_runtime(case, train_end, model_args)
    with torch.no_grad():
        model_out = training.forward_case(model, batch, 0, device, pixel_values)
    model_logk, spring_y = _model_spring_y(training, runtime, model_out, device)
    trajectory = _rollout_model_output(
        training,
        runtime,
        model_out,
        device,
        train_end,
        model_logk=model_logk,
    )
    case_dir = output_dir / "cases" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = case_dir / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    spring_path = case_dir / "candidate_spring_y.npy"
    np.save(spring_path, spring_y, allow_pickle=False)
    globals_path = case_dir / "candidate_global_parameters.json"
    globals_path.write_text(
        json.dumps(_global_parameters(model_out), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation = evaluate_official_phystwin_files(
        trajectory_path,
        data_root / case / "final_data.pkl",
        data_root / case / "gt_track_3d.pkl",
        data_root / case / "split.json",
    )
    evaluation["claim_boundary"] = MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY
    evaluation_path = case_dir / "official_reconstruction_metrics.json"
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "contract": "matphys-all-frame-reconstruction-export-v1",
        "claim_boundary": MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY,
        "future_observations_used": True,
        "predictive_use_authorized": False,
        "source_repository": MATPHYS_REPOSITORY,
        "source_commit": audit["source_commit"],
        "training_audit": {
            "path": audit["audit_path"],
            "sha256": audit["audit_sha256"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "case": {
            "name": case,
            "released_train_end_frame_exclusive": train_end,
            "frame_len": frame_len,
            "trajectory": {
                "path": str(trajectory_path),
                "sha256": sha256_file(trajectory_path),
            },
            "spring_field": {
                "path": str(spring_path),
                "sha256": sha256_file(spring_path),
                "count": int(len(spring_y)),
                "minimum": float(np.min(spring_y)),
                "maximum": float(np.max(spring_y)),
                "geometric_mean": float(np.exp(np.mean(np.log(spring_y)))),
            },
            "global_parameters": {
                "path": str(globals_path),
                "sha256": sha256_file(globals_path),
            },
            "official_metrics": {
                "path": str(evaluation_path),
                "sha256": sha256_file(evaluation_path),
            },
        },
    }
    manifest_path = output_dir / "reconstruction_export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--matphys-root", required=True)
    common.add_argument("--data-root", required=True)
    common.add_argument("--experiments-dir", required=True)
    common.add_argument("--experiments-optimization-dir", required=True)
    common.add_argument("--proxy-root", required=True)
    common.add_argument("--case", required=True)
    common.add_argument("--device", default="cuda:0")

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--protocol", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    train_parser.add_argument("--random-seed", type=int, default=42)
    train_parser.add_argument("--videomae-model", default="MCG-NJU/videomae-base")
    train_parser.add_argument("--dino-model", default="dinov2_vitl14_reg")
    train_parser.add_argument("--dino-image-size", type=int, default=518)
    train_parser.add_argument("--dino-keyframes", type=int, default=4)
    train_parser.add_argument("--part-count", type=int, default=5)
    train_parser.add_argument("--semantic-edge-weight", type=float, default=4.0)
    train_parser.set_defaults(handler=train)

    export_parser = subparsers.add_parser("export", parents=[common])
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--audit", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.set_defaults(handler=export)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    _install_torchvision_nms_stub()
    args.handler(args)


if __name__ == "__main__":
    main()
