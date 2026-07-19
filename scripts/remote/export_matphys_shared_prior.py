#!/usr/bin/env python3
"""Export a causal MatPhys proposal direction around released PhysTwin."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


_MATPHYS_PYDEPS = os.environ.get("MATPHYS_PYDEPS")
if _MATPHYS_PYDEPS:
    sys.path.append(_MATPHYS_PYDEPS)

from bayesian_phystwin.matphys_causal_bridge import (  # noqa: E402
    numeric_frame_paths,
    sha256_file,
)
from bayesian_phystwin.matphys_shared_prior import (  # noqa: E402
    MATPHYS_MATERIAL_NAMES,
    MATPHYS_SHARED_PRIOR_CONTRACT,
    assess_matphys_prediction_competence,
    build_matphys_spring_direction,
    material_distribution_from_weights,
    validate_material_distributions,
)
from bayesian_phystwin.matphys_teacher_residual import (  # noqa: E402
    load_matphys_teacher_bundle,
)
from run_matphys_causal import (  # noqa: E402
    MATPHYS_REPOSITORY,
    _ACCESSED_FRAME_PATHS,
    _ACCESSED_FRAMES,
    _causal_video_loader,
    _configure_matphys_imports,
    _install_torchvision_nms_stub,
    _source_commit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pinned MatPhys checkpoint on causal video and export its "
            "one-dimensional spring proposal around released PhysTwin."
        )
    )
    parser.add_argument("--matphys-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--proxy-root", required=True)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--experiments-optimization-dir", required=True)
    parser.add_argument("--material-prior", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--evidence-end-frame-exclusive", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--video-scope",
        choices=("causal-prefix", "numeric-all-frame-reconstruction-control"),
        default="causal-prefix",
    )
    return parser.parse_args()


def _material_matrix(
    path: Path,
    case_name: str,
    part_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("material-prior artifact must be an object")
    names = tuple(payload.get("material_names", MATPHYS_MATERIAL_NAMES))
    if names != MATPHYS_MATERIAL_NAMES:
        raise ValueError("material-prior class order differs from MatPhys")
    cases = payload.get("cases")
    if not isinstance(cases, Mapping) or case_name not in cases:
        raise ValueError(f"material-prior artifact omits {case_name}")
    record = cases[case_name]
    if not isinstance(record, Mapping):
        raise ValueError(f"{case_name}: material-prior record must be an object")

    if "part_distributions" in record:
        values = record["part_distributions"]
        source = "part_distributions"
    elif "part_weights" in record:
        weights = record["part_weights"]
        if not isinstance(weights, list):
            raise ValueError(f"{case_name}: part_weights must be a list")
        values = [material_distribution_from_weights(row) for row in weights]
        source = "part_weights"
    elif "shared_weights" in record:
        weights = record["shared_weights"]
        if not isinstance(weights, Mapping):
            raise ValueError(f"{case_name}: shared_weights must be an object")
        row = material_distribution_from_weights(weights)
        values = np.repeat(row[None, :], part_count, axis=0)
        source = "shared_weights"
    else:
        raise ValueError(
            f"{case_name}: expected part_distributions, part_weights, or shared_weights"
        )
    matrix = validate_material_distributions(values, expected_parts=part_count)
    provenance = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_field": source,
        "claim_boundary": str(payload.get("claim_boundary", "unspecified")),
    }
    return matrix, provenance


def _tensor_values(value) -> np.ndarray:
    return np.asarray(value.detach().cpu(), dtype=np.float64).reshape(-1)


def main() -> None:
    args = _parse_args()
    matphys_root = Path(args.matphys_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    proxy_root = Path(args.proxy_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matphys_imports(matphys_root)
    _install_torchvision_nms_stub()

    import torch

    from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset
    from train_model_video_material_simple import SimpleVideoMaterialPhysicsModel

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_args = checkpoint["args"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = MaterialParamDataset(
        MaterialDatasetConfig(
            base_path=str(data_root),
            sem_cache_dir=str(proxy_root / "semantic_cache"),
            experiments_dir=str(Path(args.experiments_dir).resolve()),
            experiments_optimization_dir=str(
                Path(args.experiments_optimization_dir).resolve()
            ),
            case_to_material_path=str(proxy_root / "case_to_material.json"),
            results_dir=str(proxy_root / "results"),
            use_knn_topology=bool(checkpoint_args.get("use_knn_topology", False)),
            object_knn=int(checkpoint_args.get("object_knn", 30)),
            object_radius=float(checkpoint_args.get("object_radius", 0.02)),
            object_max_neighbours=int(
                checkpoint_args.get("object_max_neighbours", 30)
            ),
            controller_radius=float(
                checkpoint_args.get("controller_radius", 0.04)
            ),
            controller_max_neighbours=int(
                checkpoint_args.get("controller_max_neighbours", 50)
            ),
        )
    )
    matches = [sample for sample in dataset.samples if sample["case_name"] == args.case]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one proxy sample for {args.case}")
    sample = matches[0]
    part_count = int(sample["material_dist"].shape[0])
    material_dist, material_provenance = _material_matrix(
        Path(args.material_prior).resolve(), args.case, part_count
    )

    available_frames = numeric_frame_paths(data_root / args.case / "color" / "0")
    evidence_end = int(args.evidence_end_frame_exclusive)
    if args.video_scope == "numeric-all-frame-reconstruction-control":
        evidence_end = max(available_frames) + 1
    if evidence_end < 1:
        raise ValueError("evidence endpoint must be positive")
    video_loader = _causal_video_loader({args.case: evidence_end})
    pixel_values = video_loader(
        args.case,
        str(data_root),
        T=int(checkpoint_args.get("num_video_frames", 16)),
        image_size=int(checkpoint_args.get("videomae_image_size", 224)),
        device=device,
    )

    model = SimpleVideoMaterialPhysicsModel(
        videomae_model=str(checkpoint_args["videomae_model"]),
        d_motion=int(checkpoint_args["d_motion"]),
        d_mat=int(checkpoint_args["mat_codebook_dim"]),
        hidden_dim=int(checkpoint_args["hidden_dim"]),
        num_materials=int(checkpoint_args["num_materials"]),
        logk_residual_scale=float(checkpoint_args.get("logk_residual_scale", 1.0)),
        logk_soft_clamp=float(checkpoint_args.get("logk_soft_clamp", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        model_out = model(
            pixel_values,
            sample["z_geo"].to(device),
            torch.as_tensor(material_dist, dtype=torch.float32, device=device),
            sample["edge_part_idx"].to(device),
            geo_stats=sample["geo_stats"].to(device),
            ctrl_rest_length=sample["ctrl_rest_length"].to(device),
            ctrl_part_idx=sample["ctrl_part_idx"].to(device),
        )

    teacher = load_matphys_teacher_bundle(
        args.case,
        args.experiments_dir,
        args.experiments_optimization_dir,
    )
    predicted_object_log_y = _tensor_values(model_out["log_k"])
    competence = assess_matphys_prediction_competence(
        teacher_object_log_y=teacher.spring_log_y[
            : int(sample["num_object_springs"].item())
        ],
        predicted_object_log_y=predicted_object_log_y,
        stiffness_minimum=float(checkpoint_args.get("logk_min", 1.0e3)),
        stiffness_maximum=float(checkpoint_args.get("logk_max", 1.0e5)),
    )
    direction = build_matphys_spring_direction(
        teacher_log_y=teacher.spring_log_y,
        predicted_object_log_y=predicted_object_log_y,
        predicted_controller_log_y=(
            _tensor_values(model_out["ctrl_log_k"])
            if "ctrl_log_k" in model_out
            else None
        ),
        object_spring_count=int(sample["num_object_springs"].item()),
    )

    basis_path = output_dir / "matphys_spring_direction.npz"
    np.savez_compressed(
        basis_path,
        weights=direction.weights,
        prior_coefficient=np.asarray(direction.prior_coefficient),
        raw_log_difference=direction.raw_log_difference,
        material_distributions=material_dist,
        edge_part_idx=_tensor_values(sample["edge_part_idx"]).astype(np.int64),
        object_spring_count=np.asarray(direction.object_spring_count),
    )
    frame_records = []
    for frame_id in sorted(_ACCESSED_FRAMES[args.case]):
        path = _ACCESSED_FRAME_PATHS[args.case][frame_id]
        frame_records.append(
            {"frame_id": int(frame_id), "path": str(path), "sha256": sha256_file(path)}
        )
    global_names = (
        "dashpot_damping",
        "drag_damping",
        "collide_elas",
        "collide_fric",
        "collide_object_elas",
        "collide_object_fric",
        "collision_dist",
    )
    summary = {
        "schema_version": 1,
        "contract": MATPHYS_SHARED_PRIOR_CONTRACT,
        "case_name": args.case,
        "video_scope": args.video_scope,
        "future_observations_used": args.video_scope != "causal-prefix",
        "declared_evidence_end_frame_exclusive": int(
            args.evidence_end_frame_exclusive
        ),
        "effective_evidence_end_frame_exclusive": evidence_end,
        "source_repository": MATPHYS_REPOSITORY,
        "source_commit": _source_commit(matphys_root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)),
        },
        "teacher": teacher.manifest(),
        "material_prior": material_provenance,
        "material_names": list(MATPHYS_MATERIAL_NAMES),
        "material_distributions": material_dist.tolist(),
        "video_frames": frame_records,
        "competence_gate": competence.diagnostics(),
        "direction": direction.diagnostics(),
        "predicted_global_parameters": {
            name: float(_tensor_values(model_out[name])[0]) for name in global_names
        },
        "basis": {"path": str(basis_path), "sha256": sha256_file(basis_path)},
        "claim_boundary": (
            "This is a MatPhys-informed proposal direction, not a posterior or "
            "a MatPhys reproduction. Coefficient zero is the exact released "
            "PhysTwin spring field. Predictive use requires both a competent "
            "spatial direction and a disjoint prefix gate."
        ),
    }
    summary_path = output_dir / "matphys_shared_prior_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
