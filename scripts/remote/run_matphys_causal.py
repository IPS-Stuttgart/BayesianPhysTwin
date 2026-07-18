#!/usr/bin/env python3
"""Train and export a future-blind MatPhys global-material ablation.

This wrapper leaves the pinned MatPhys checkout unchanged. It replaces only
the video loader, records every selected frame, and stubs optional Gaussian
rendering modules because the experiment sets render loss to zero.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np


_MATPHYS_PYDEPS = os.environ.get("MATPHYS_PYDEPS")
if _MATPHYS_PYDEPS:
    # Append so the active CUDA environment keeps its ABI-matched torch,
    # torchvision, transformers, and tokenizers ahead of supplemental modules.
    sys.path.append(_MATPHYS_PYDEPS)

from bayesian_phystwin.matphys_causal_bridge import (
    causal_uniform_frame_ids,
    numeric_frame_paths,
    prepare_global_material_proxy,
    sha256_file,
    validate_causal_training_audit,
    write_causal_training_audit,
)
from bayesian_phystwin.matphys_dino_features import CausalDinoNodeExtractor
from bayesian_phystwin.matphys_graph_parts import (
    GRAPH_PART_PROXY_CONTRACT,
    prepare_graph_part_proxy,
)
from bayesian_phystwin.matphys_part_model import (
    PART_AWARE_MODEL_CONTRACT,
    install_part_aware_simple_model,
    summarize_part_spring_ratios,
)
from bayesian_phystwin.matphys_teacher_residual import (
    TEACHER_PARAMETERIZATION,
    MatPhysTeacherBundle,
    apply_matphys_teacher_residual,
    load_matphys_teacher_bundle,
    load_matphys_teacher_manifest,
)
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_external_backbone import (
    EXTERNAL_COORDINATE_FRAME,
    EXTERNAL_VERTEX_CONTRACT,
)


MATPHYS_REPOSITORY = "https://github.com/Yrainy0615/MatPhys"
_ACCESSED_FRAMES: dict[str, set[int]] = {}
_ACCESSED_FRAME_PATHS: dict[str, dict[int, Path]] = {}
_OBJECTIVE_END_FRAMES: dict[str, int] = {}
_TORCHVISION_STUB_LIBRARY = None


def _unavailable(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("Gaussian rendering is disabled for this MatPhys experiment")


def _install_torchvision_nms_stub() -> None:
    """Permit transform-only torchvision imports when its optional ops are absent."""

    global _TORCHVISION_STUB_LIBRARY
    import torch

    try:
        __import__("torchvision")
        return
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise
        for name in tuple(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_STUB_LIBRARY = library


def _install_render_stubs() -> None:
    """Satisfy eager optional-render imports without a CUDA rasterizer build."""

    gaussian = types.ModuleType("gaussian_splatting")
    gaussian.__path__ = []
    scene = types.ModuleType("gaussian_splatting.scene")
    scene.__path__ = []
    gaussian_model = types.ModuleType("gaussian_splatting.scene.gaussian_model")
    cameras = types.ModuleType("gaussian_splatting.scene.cameras")
    renderer = types.ModuleType("gaussian_splatting.gaussian_renderer")
    dynamic = types.ModuleType("gaussian_splatting.dynamic_utils")
    utilities = types.ModuleType("gaussian_splatting.utils")
    utilities.__path__ = []
    graphics = types.ModuleType("gaussian_splatting.utils.graphics_utils")
    rotations = types.ModuleType("gaussian_splatting.rotation_utils")
    gs_render = types.ModuleType("gs_render")
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
    pyplot.__spec__ = importlib.machinery.ModuleSpec(
        "matplotlib.pyplot", loader=None
    )

    class GaussianModel:
        pass

    class Camera:
        pass

    gaussian_model.GaussianModel = GaussianModel
    gaussian_model.BasicPointCloud = object
    cameras.Camera = Camera
    renderer.render = _unavailable
    for name in (
        "interpolate_motions_speedup",
        "knn_weights",
        "knn_weights_sparse",
        "get_topk_indices",
        "calc_weights_vals_from_indices",
    ):
        setattr(dynamic, name, _unavailable)
    graphics.getWorld2View2 = _unavailable
    graphics.focal2fov = lambda focal, pixels: 2.0 * math.atan(
        float(pixels) / (2.0 * float(focal))
    )
    graphics.fov2focal = lambda fov, pixels: float(pixels) / (
        2.0 * math.tan(float(fov) / 2.0)
    )
    rotations.quaternion_multiply = _unavailable
    rotations.matrix_to_quaternion = _unavailable
    gs_render.remove_gaussians_with_low_opacity = _unavailable
    gs_render.remove_gaussians_with_point_mesh_distance = _unavailable
    pyplot.cm = SimpleNamespace(
        rainbow=lambda values: np.column_stack(
            (
                np.asarray(values, dtype=float),
                1.0 - np.asarray(values, dtype=float),
                np.full_like(np.asarray(values, dtype=float), 0.5),
                np.ones_like(np.asarray(values, dtype=float)),
            )
        )
    )
    matplotlib.pyplot = pyplot
    sys.modules.update(
        {
            "gaussian_splatting": gaussian,
            "gaussian_splatting.scene": scene,
            "gaussian_splatting.scene.gaussian_model": gaussian_model,
            "gaussian_splatting.scene.cameras": cameras,
            "gaussian_splatting.gaussian_renderer": renderer,
            "gaussian_splatting.dynamic_utils": dynamic,
            "gaussian_splatting.utils": utilities,
            "gaussian_splatting.utils.graphics_utils": graphics,
            "gaussian_splatting.rotation_utils": rotations,
            "gs_render": gs_render,
            "pyrender": types.ModuleType("pyrender"),
            "trimesh": types.ModuleType("trimesh"),
            "matplotlib": matplotlib,
            "matplotlib.pyplot": pyplot,
            "cma": types.ModuleType("cma"),
        }
    )


def _install_open3d_stub() -> None:
    """Provide the two deterministic KD-tree queries used by MatPhys."""

    module = types.ModuleType("open3d")

    class PointCloud:
        def __init__(self):
            self.points = np.zeros((0, 3), dtype=np.float64)

    class KDTreeFlann:
        def __init__(self, point_cloud):
            self.points = np.asarray(point_cloud.points, dtype=np.float64)

        def _ordered(self, query):
            delta = self.points - np.asarray(query, dtype=np.float64)
            distance_sq = np.einsum("ij,ij->i", delta, delta)
            indices = np.arange(len(self.points))
            order = np.lexsort((indices, distance_sq))
            return indices[order], distance_sq[order]

        def search_knn_vector_3d(self, query, count):
            indices, distance_sq = self._ordered(query)
            selected = indices[: int(count)]
            return len(selected), selected.tolist(), distance_sq[: len(selected)].tolist()

        def search_hybrid_vector_3d(self, query, radius, maximum):
            indices, distance_sq = self._ordered(query)
            keep = distance_sq <= float(radius) ** 2
            selected = indices[keep][: int(maximum)]
            selected_distance = distance_sq[keep][: len(selected)]
            return len(selected), selected.tolist(), selected_distance.tolist()

    module.geometry = SimpleNamespace(PointCloud=PointCloud, KDTreeFlann=KDTreeFlann)
    module.utility = SimpleNamespace(
        Vector3dVector=lambda value: np.asarray(value, dtype=np.float64)
    )
    sys.modules["open3d"] = module


def _install_wandb_stub() -> None:
    """Keep MatPhys's disabled tracking import free of binary dependencies."""

    module = types.ModuleType("wandb")
    module.init = lambda *args, **kwargs: None
    module.log = lambda *args, **kwargs: None
    module.finish = lambda *args, **kwargs: None
    module.Video = lambda *args, **kwargs: SimpleNamespace(
        args=args,
        kwargs=kwargs,
    )
    sys.modules["wandb"] = module


def _configure_matphys_imports(matphys_root: Path) -> None:
    semantic = matphys_root / "semantic"
    sys.path.insert(0, str(semantic))
    sys.path.insert(0, str(matphys_root))
    _install_render_stubs()
    _install_open3d_stub()
    _install_wandb_stub()


def _source_commit(matphys_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=matphys_root, text=True
    ).strip()


def _case_names(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise ValueError("at least one case is required")
    if len(result) != len(set(result)):
        raise ValueError("case names must be unique")
    return result


def _causal_video_loader(evidence_end_by_case: dict[str, int]) -> Callable:
    def load_video_frames(
        case_name: str,
        base_path: str,
        T: int = 16,
        image_size: int = 224,
        device=None,
    ):
        from PIL import Image
        import torch
        from torchvision import transforms

        color_dir = Path(base_path) / case_name / "color" / "0"
        frame_files = numeric_frame_paths(color_dir)
        if case_name not in evidence_end_by_case:
            raise RuntimeError(f"missing causal evidence boundary for {case_name}")
        evidence_end = int(evidence_end_by_case[case_name])
        indices = causal_uniform_frame_ids(frame_files, evidence_end, T)
        _ACCESSED_FRAMES.setdefault(case_name, set()).update(
            int(index) for index in indices
        )
        selected_paths = _ACCESSED_FRAME_PATHS.setdefault(case_name, {})
        selected_paths.update(
            {int(index): frame_files[int(index)] for index in indices}
        )
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        frames = [
            transform(Image.open(frame_files[int(index)]).convert("RGB"))
            for index in indices
        ]
        tensor = torch.stack(frames, dim=0).unsqueeze(0)
        return tensor.to(device) if device is not None else tensor

    return load_video_frames


def _validated_existing_proxy(
    proxy_root: Path,
    cases: list[str],
    expected_contract: str,
) -> dict[str, object] | None:
    summary_path = proxy_root / "proxy_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract") != expected_contract:
        raise ValueError("existing MatPhys proxy uses a different contract")
    records = summary.get("cases")
    if not isinstance(records, list) or {
        str(record.get("name")) for record in records
    } != set(cases):
        raise ValueError("existing MatPhys proxy cases do not match the request")
    for record in records:
        for key in ("node_sem", "train_ready"):
            identity = record.get(key)
            if not isinstance(identity, dict):
                raise ValueError(f"proxy record omits {key}")
            path = Path(identity["path"])
            if sha256_file(path) != identity["sha256"]:
                raise ValueError(f"proxy {key} bytes changed")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def _prepare_proxy(
    args,
    cases: list[str],
    evidence_end_by_case: dict[str, int],
) -> dict[str, object]:
    proxy_root = Path(args.proxy_root).resolve()
    expected_contract = (
        GRAPH_PART_PROXY_CONTRACT
        if args.graph_parts
        else "global-onehot-single-part-v1"
    )
    existing = _validated_existing_proxy(proxy_root, cases, expected_contract)
    if existing is not None:
        return existing
    mapping_path = (
        Path(args.matphys_root)
        / "semantic"
        / "case_to_material_different_types.json"
    )
    if not args.graph_parts:
        return prepare_global_material_proxy(
            args.data_root,
            cases,
            mapping_path,
            proxy_root,
        )

    extractor = CausalDinoNodeExtractor(
        model_name=args.dino_model,
        image_size=args.dino_image_size,
        device=args.device,
    )
    node_features = {}
    contributor_counts = {}
    graph_edges = {}
    provenance = {}
    data_root = Path(args.data_root).resolve()
    optimization_root = Path(args.experiments_optimization_dir).resolve()
    for case in cases:
        available = numeric_frame_paths(data_root / case / "color" / "0")
        frame_ids = causal_uniform_frame_ids(
            available,
            evidence_end_by_case[case],
            args.dino_keyframes,
        )
        features, counts, case_provenance = extractor.extract_case(
            data_root / case,
            frame_ids,
        )
        with (data_root / case / "final_data.pkl").open("rb") as handle:
            final_data = pickle.load(handle)
        optimal_path = optimization_root / case / "optimal_params.pkl"
        with optimal_path.open("rb") as handle:
            optimal = pickle.load(handle)
        structure_points = np.concatenate(
            (
                np.asarray(final_data["object_points"])[0],
                np.asarray(final_data["surface_points"]),
                np.asarray(final_data["interior_points"]),
            ),
            axis=0,
        )
        controller_points = np.asarray(final_data["controller_points"])[0]
        graph = build_phystwin_spring_graph(
            structure_points,
            controller_points,
            config=PhysTwinSpringGraphConfig(
                object_radius=float(optimal["object_radius"]),
                object_max_neighbours=int(optimal["object_max_neighbours"]),
                controller_radius=float(optimal["controller_radius"]),
                controller_max_neighbours=int(optimal["controller_max_neighbours"]),
            ),
        )
        node_features[case] = features
        contributor_counts[case] = counts
        graph_edges[case] = graph.springs[: graph.num_object_springs]
        provenance[case] = {
            **case_provenance,
            "evidence_end_frame_exclusive": evidence_end_by_case[case],
            "optimal_params": {
                "path": str(optimal_path),
                "sha256": sha256_file(optimal_path),
            },
            "object_spring_count": graph.num_object_springs,
        }
    return prepare_graph_part_proxy(
        data_root,
        cases,
        mapping_path,
        proxy_root,
        node_features_by_case=node_features,
        graph_edges_by_case=graph_edges,
        contributor_count_by_case=contributor_counts,
        provenance_by_case=provenance,
        part_count=args.part_count,
        semantic_edge_weight=args.semantic_edge_weight,
    )


def _splits(data_root: Path, cases: list[str]) -> dict[str, object]:
    return {
        case: json.loads(
            (data_root / case / "split.json").read_text(encoding="utf-8")
        )
        for case in cases
    }


def _evidence_ends(
    data_root: Path,
    cases: list[str],
    fit_fraction: float,
) -> dict[str, int]:
    if not 0.0 < fit_fraction <= 1.0:
        raise ValueError("fit fraction must lie in (0, 1]")
    result: dict[str, int] = {}
    for case, split in _splits(data_root, cases).items():
        train_start, train_end = (int(value) for value in split["train"])
        evidence_end = train_start + math.floor(
            fit_fraction * (train_end - train_start)
        )
        if not train_start + 2 <= evidence_end <= train_end:
            raise ValueError(f"{case}: fit fraction leaves too little evidence")
        result[case] = evidence_end
    return result


def _install_causal_objective_guard(
    training,
    data_root: Path,
    evidence_end_by_case: dict[str, int],
) -> None:
    original = training._resolve_train_frame

    def resolve_train_frame(model_args, case_name: str, dataset_train_frame: int):
        released_resolved = int(original(model_args, case_name, dataset_train_frame))
        split = json.loads(
            (data_root / case_name / "split.json").read_text(encoding="utf-8")
        )
        released_train_end = int(split["train"][1])
        if not 1 <= released_resolved <= released_train_end:
            raise RuntimeError(
                f"{case_name}: MatPhys objective crosses the released prefix"
            )
        resolved = int(evidence_end_by_case[case_name])
        if resolved > released_resolved:
            raise RuntimeError(f"{case_name}: evidence boundary exceeds objective data")
        previous = _OBJECTIVE_END_FRAMES.setdefault(case_name, resolved)
        if previous != resolved:
            raise RuntimeError(f"{case_name}: objective boundary changed during training")
        return resolved

    training._resolve_train_frame = resolve_train_frame


def _teacher_configuration(
    args, cases: list[str]
) -> tuple[dict[str, object] | None, dict[str, MatPhysTeacherBundle]]:
    scale = args.teacher_residual_log_scale
    teacher_root = args.teacher_experiments_dir
    if scale is None:
        if teacher_root is not None:
            raise ValueError(
                "--teacher-experiments-dir requires --teacher-residual-log-scale"
            )
        return None, {}
    if teacher_root is None:
        raise ValueError(
            "--teacher-residual-log-scale requires --teacher-experiments-dir"
        )
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("teacher residual log scale must be finite and nonnegative")
    bundles = {
        case: load_matphys_teacher_bundle(
            case,
            teacher_root,
            args.experiments_optimization_dir,
        )
        for case in cases
    }
    parameterization = {
        "name": TEACHER_PARAMETERIZATION,
        "residual_log_scale": float(scale),
        "identity_scale": 0.0,
        "global_parameters": "frozen at released PhysTwin values",
        "teacher": {
            "cases": [bundles[case].manifest() for case in cases],
        },
    }
    return parameterization, bundles


def _install_teacher_parameterization(
    training,
    bundles: dict[str, MatPhysTeacherBundle],
    residual_log_scale: float,
) -> None:
    original_forward_case = training.forward_case

    def forward_case(model, batch, idx, device, pixel_values):
        output = original_forward_case(model, batch, idx, device, pixel_values)
        case = str(batch["case_name"][idx])
        if case not in bundles:
            raise RuntimeError(f"teacher parameterization omits {case}")
        return apply_matphys_teacher_residual(
            output, bundles[case], residual_log_scale
        )

    training.forward_case = forward_case


def train(args) -> None:
    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    cases = _case_names(args.cases)
    evidence_end_by_case = _evidence_ends(data_root, cases, args.fit_fraction)
    if args.graph_parts and args.teacher_residual_log_scale is None:
        raise ValueError("graph parts require the identity-preserving teacher residual")
    proxy = _prepare_proxy(args, cases, evidence_end_by_case)
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    import train_model_video_material_simple as training

    if proxy["contract"] == GRAPH_PART_PROXY_CONTRACT:
        semantic_dimensions = {
            int(record["semantic_dimension"]) for record in proxy["cases"]
        }
        if len(semantic_dimensions) != 1:
            raise ValueError("part-aware training requires one DINO feature width")
        install_part_aware_simple_model(
            training,
            part_feature_dim=semantic_dimensions.pop(),
            part_feature_scale=args.part_feature_scale,
        )
    parameterization, teacher_bundles = _teacher_configuration(args, cases)
    if parameterization is not None and proxy["contract"] == GRAPH_PART_PROXY_CONTRACT:
        parameterization.update(
            {
                "part_model_contract": PART_AWARE_MODEL_CONTRACT,
                "part_feature_scale": float(args.part_feature_scale),
                "proxy_contract": GRAPH_PART_PROXY_CONTRACT,
                "fit_fraction": float(args.fit_fraction),
            }
        )
    if parameterization is not None:
        _install_teacher_parameterization(
            training,
            teacher_bundles,
            float(parameterization["residual_log_scale"]),
        )
    training.load_video_frames = _causal_video_loader(evidence_end_by_case)
    _install_causal_objective_guard(training, data_root, evidence_end_by_case)
    train_args = [
        "train_model_video_material_simple.py",
        "--save_dir",
        str(Path(args.output_dir).resolve()),
        "--base_path",
        str(data_root),
        "--experiments_dir",
        str(
            Path(args.teacher_experiments_dir).resolve()
            if args.teacher_experiments_dir is not None
            else data_root
        ),
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
        "--epochs",
        str(args.epochs),
        "--eval_every",
        str(args.eval_every),
        "--device",
        args.device,
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--train_ratio",
        "1.0",
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
        "--save_best_only",
        "--vis_every",
        "0",
    ]
    if len(cases) == 1:
        train_args.extend(("--case_name", cases[0]))
    sys.argv = train_args
    training.main()
    output_dir = Path(args.output_dir).resolve()
    checkpoints = [output_dir / "last_checkpoint.pth"]
    if not checkpoints:
        raise RuntimeError("MatPhys training did not write a checkpoint")
    audit = write_causal_training_audit(
        checkpoints,
        output_dir / "causal_training_audit.json",
        source_repository=MATPHYS_REPOSITORY,
        source_commit=_source_commit(matphys_root),
        data_root=data_root,
        accessed_frame_indices={
            case: sorted(indices) for case, indices in _ACCESSED_FRAMES.items()
        },
        accessed_frame_paths=_ACCESSED_FRAME_PATHS,
        objective_end_frames_exclusive=_OBJECTIVE_END_FRAMES,
        evidence_end_frames_exclusive=evidence_end_by_case,
        split_by_case=_splits(data_root, cases),
        proxy_summary_path=proxy["summary_path"],
        parameterization=parameterization,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def _namespace_from_checkpoint(raw: dict[str, object], args) -> SimpleNamespace:
    values = dict(raw)
    values.update(
        {
            "base_path": str(Path(args.data_root).resolve()),
            "experiments_dir": str(Path(args.data_root).resolve()),
            "experiments_optimization_dir": str(
                Path(args.experiments_optimization_dir).resolve()
            ),
            "case_to_material": str(
                Path(args.proxy_root).resolve() / "case_to_material.json"
            ),
            "results_dir": str(Path(args.proxy_root).resolve() / "results"),
            "sem_cache_dir": str(
                Path(args.proxy_root).resolve() / "semantic_cache"
            ),
            "gaussian_root": "__disabled__",
            "device": args.device,
            "rank": 0,
            "lambda_render": 0.0,
            "fit_all_frames": False,
        }
    )
    values.setdefault("logk_residual_scale", 1.0)
    values.setdefault("logk_soft_clamp", 0.25)
    return SimpleNamespace(**values)


def _rollout_model_output(training, runtime, model_out, device, train_end: int):
    """Apply one spring field and run one fresh official Warp trajectory."""

    import torch
    import warp as wp

    sim = runtime.sim
    model_logk = training._build_model_logk(
        model_out, runtime, sim, device
    )
    if model_logk is None:
        raise RuntimeError("MatPhys spring field does not match the runtime topology")
    training._apply_model_out_to_sim(model_out, model_logk, sim)
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    vertices = [
        wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False)
        .detach()
        .cpu()
        .numpy()
    ]
    frame_count = int(runtime.trainer.dataset.frame_len)
    for frame_index in range(1, frame_count):
        sim.set_controller_target(
            frame_index, pure_inference=frame_index >= train_end
        )
        if sim.object_collision_flag:
            sim.update_collision_graph()
        sim.step()
        vertices.append(
            wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
            .detach()
            .cpu()
            .numpy()
        )
        sim.clear_loss()
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
    return np.asarray(vertices, dtype=np.float32)


def export(args) -> None:
    import torch

    matphys_root = Path(args.matphys_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    audit = validate_causal_training_audit(args.training_audit, checkpoint_path)
    cases = _case_names(args.cases)
    evidence_end_by_case = {
        str(record["name"]): int(record["evidence_end_frame_exclusive"])
        for record in audit["cases"]
    }
    if set(evidence_end_by_case) != set(cases):
        raise ValueError("training audit cases do not match the export request")
    audit_proxy = audit.get("proxy")
    if not isinstance(audit_proxy, dict):
        raise ValueError("training audit omits its proxy")
    proxy_summary = json.loads(
        Path(audit_proxy["path"]).read_text(encoding="utf-8")
    )
    args.graph_parts = proxy_summary.get("contract") == GRAPH_PART_PROXY_CONTRACT
    if not (Path(args.proxy_root).resolve() / "proxy_summary.json").is_file():
        raise FileNotFoundError("export requires the byte-bound training proxy")
    proxy = _prepare_proxy(args, cases, evidence_end_by_case)
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    from material_param_dataset import (
        MaterialDatasetConfig,
        MaterialParamDataset,
    )
    import train_model_video_material_simple as training

    parameterization = audit.get("parameterization")
    if proxy["contract"] == GRAPH_PART_PROXY_CONTRACT:
        if not isinstance(parameterization, dict):
            raise ValueError("part-aware checkpoint omits its parameterization")
        if parameterization.get("part_model_contract") != PART_AWARE_MODEL_CONTRACT:
            raise ValueError("part-aware checkpoint uses an unknown model adapter")
        install_part_aware_simple_model(
            training,
            part_feature_dim=int(proxy["cases"][0]["semantic_dimension"]),
            part_feature_scale=float(parameterization["part_feature_scale"]),
        )
    if parameterization is not None:
        teacher_bundles = load_matphys_teacher_manifest(
            parameterization["teacher"]
        )
        _install_teacher_parameterization(
            training,
            teacher_bundles,
            float(parameterization["residual_log_scale"]),
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = _namespace_from_checkpoint(checkpoint["args"], args)
    device = torch.device(args.device)
    model = training.SimpleVideoMaterialPhysicsModel(
        videomae_model=model_args.videomae_model,
        d_motion=model_args.d_motion,
        d_mat=model_args.mat_codebook_dim,
        hidden_dim=model_args.hidden_dim,
        num_materials=model_args.num_materials,
        logk_residual_scale=model_args.logk_residual_scale,
        logk_soft_clamp=model_args.logk_soft_clamp,
        **(
            {
                "part_feature_dim": int(proxy["cases"][0]["semantic_dimension"]),
                "part_feature_scale": float(parameterization["part_feature_scale"]),
            }
            if proxy["contract"] == GRAPH_PART_PROXY_CONTRACT
            else {}
        ),
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
    sample_by_case = {sample["case_name"]: sample for sample in dataset.samples}
    load_video = _causal_video_loader(evidence_end_by_case)
    output_root = Path(args.output_dir).resolve()
    manifest_cases = []
    for case in cases:
        sample = sample_by_case[case]
        train_end = int(sample["train_frame"].item())
        pixel_values = load_video(
            case,
            model_args.base_path,
            T=model_args.num_video_frames,
            image_size=model_args.videomae_image_size,
            device=device,
        )
        batch = {
            key: [value] if key != "case_name" else [case]
            for key, value in sample.items()
        }
        runtime = training._init_runtime(case, train_end, model_args)
        with torch.no_grad():
            model_out = training.forward_case(model, batch, 0, device, pixel_values)
        spring_field_identity = None
        teacher_control_identity = None
        teacher_trajectory = None
        if parameterization is not None:
            teacher_bundle = teacher_bundles[case]
            object_log_k = (
                model_out["log_k"].detach().cpu().numpy().reshape(-1)
            )
            edge_part_index = np.asarray(
                sample["edge_part_idx"].detach().cpu().numpy(),
                dtype=np.int64,
            )
            spring_summary = summarize_part_spring_ratios(
                object_log_k,
                teacher_bundle.spring_log_y[: len(object_log_k)],
                edge_part_index,
            )
            spring_summary.update(
                {
                    "case": case,
                    "parameterization": parameterization["name"],
                    "residual_log_scale": float(
                        parameterization["residual_log_scale"]
                    ),
                }
            )
            spring_summary_path = (
                output_root / "cases" / case / "spring_field_summary.json"
            )
            spring_summary_path.parent.mkdir(parents=True, exist_ok=True)
            spring_summary_path.write_text(
                json.dumps(spring_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            spring_field_identity = {
                "path": str(spring_summary_path),
                "sha256": sha256_file(spring_summary_path),
            }
            teacher_runtime = training._init_runtime(case, train_end, model_args)
            teacher_model_out = apply_matphys_teacher_residual(
                model_out,
                teacher_bundle,
                0.0,
            )
            teacher_trajectory = _rollout_model_output(
                training,
                teacher_runtime,
                teacher_model_out,
                device,
                train_end,
            )
            teacher_path = (
                output_root / "cases" / case / "teacher_control_trajectory.pkl"
            )
            teacher_path.parent.mkdir(parents=True, exist_ok=True)
            with teacher_path.open("wb") as handle:
                pickle.dump(
                    teacher_trajectory,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            teacher_control_identity = {
                "contract": "paired-exact-teacher-same-export-v1",
                "trajectory": str(teacher_path),
                "sha256": sha256_file(teacher_path),
            }
        trajectory = _rollout_model_output(
            training,
            runtime,
            model_out,
            device,
            train_end,
        )
        causal_validation_identity = None
        if teacher_trajectory is not None:
            validation_files = {}
            for family, values in (
                ("candidate", trajectory),
                ("teacher", teacher_trajectory),
            ):
                validation_path = (
                    output_root
                    / "cases"
                    / case
                    / f"{family}_fit_validation_trajectory.pkl"
                )
                with validation_path.open("wb") as handle:
                    pickle.dump(
                        values[:train_end],
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                validation_files[family] = {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                }
            causal_validation_identity = {
                "contract": "paired-fit-validation-trajectories-v1",
                "fit_end_frame_exclusive": evidence_end_by_case[case],
                "validation_end_frame_exclusive": train_end,
                **validation_files,
            }
        trajectory_path = output_root / "cases" / case / "trajectory.pkl"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_path.open("wb") as handle:
            pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest_cases.append(
            {
                "name": case,
                "trajectory": str(trajectory_path),
                "sha256": sha256_file(trajectory_path),
                "evidence_end_frame_exclusive": evidence_end_by_case[case],
                "initial_alignment_tolerance_m": args.initial_alignment_tolerance_m,
                **(
                    {"spring_field_summary": spring_field_identity}
                    if spring_field_identity is not None
                    else {}
                ),
                **(
                    {"teacher_control": teacher_control_identity}
                    if teacher_control_identity is not None
                    else {}
                ),
                **(
                    {"causal_validation": causal_validation_identity}
                    if causal_validation_identity is not None
                    else {}
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "backbone": {
            "name": (
                "MatPhys causal graph-part released-PhysTwin residual"
                if proxy["contract"] == GRAPH_PART_PROXY_CONTRACT
                else "MatPhys causal released-PhysTwin residual ablation"
                if parameterization is not None
                else "MatPhys causal global-material ablation"
            ),
            "source_repository": MATPHYS_REPOSITORY,
            "source_commit": audit["source_commit"],
            "future_observations_used": False,
            "coordinate_frame": EXTERNAL_COORDINATE_FRAME,
            "vertex_contract": EXTERNAL_VERTEX_CONTRACT,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "causal_training_audit": {
                "path": audit["audit_path"],
                "sha256": audit["audit_sha256"],
            },
            "proxy_contract": proxy["contract"],
            "claim_boundary": proxy["claim_boundary"],
            "parameterization": parameterization,
        },
        "cases": manifest_cases,
    }
    manifest_path = output_root / "external_backbone_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--matphys-root", required=True)
    common.add_argument("--data-root", required=True)
    common.add_argument("--experiments-optimization-dir", required=True)
    common.add_argument("--proxy-root", required=True)
    common.add_argument("--cases", required=True)
    common.add_argument("--device", default="cuda:0")

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--eval-every", type=int, default=5)
    train_parser.add_argument("--teacher-experiments-dir")
    train_parser.add_argument("--teacher-residual-log-scale", type=float)
    train_parser.add_argument("--fit-fraction", type=float, default=1.0)
    train_parser.add_argument("--graph-parts", action="store_true")
    train_parser.add_argument("--dino-model", default="dinov2_vitl14_reg")
    train_parser.add_argument("--dino-image-size", type=int, default=518)
    train_parser.add_argument("--dino-keyframes", type=int, default=4)
    train_parser.add_argument("--part-count", type=int, default=5)
    train_parser.add_argument("--semantic-edge-weight", type=float, default=4.0)
    train_parser.add_argument("--part-feature-scale", type=float, default=1.0)
    train_parser.set_defaults(handler=train)

    export_parser = subparsers.add_parser("export", parents=[common])
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--training-audit", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--initial-alignment-tolerance-m", type=float, default=1e-6)
    export_parser.set_defaults(handler=export)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    _install_torchvision_nms_stub()
    args.handler(args)


if __name__ == "__main__":
    main()
