#!/usr/bin/env python3
"""Train and export a future-blind MatPhys global-material ablation.

This wrapper leaves the pinned MatPhys checkout unchanged. It replaces only
the video loader, records every selected frame, and stubs optional Gaussian
rendering modules because the experiment sets render loss to zero.
"""

from __future__ import annotations

import argparse
import copy
import importlib.machinery
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
import time
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np


_MATPHYS_PYDEPS = os.environ.get("MATPHYS_PYDEPS")
if _MATPHYS_PYDEPS:
    # Append so the active CUDA environment keeps its ABI-matched torch,
    # torchvision, transformers, and tokenizers ahead of supplemental modules.
    sys.path.append(_MATPHYS_PYDEPS)

from bayesian_phystwin.matphys_causal_bridge import (  # noqa: E402
    MATPHYS_GENERIC_MOTION_BACKBONE,
    MATPHYS_SOURCE_SUPERVISED_AUDIT_CONTRACT,
    causal_uniform_frame_ids,
    matphys_fresh_fold_initialization,
    numeric_frame_paths,
    prepare_global_material_proxy,
    sha256_file,
    validate_causal_training_audit,
    validate_source_supervised_training_audit,
    write_causal_training_audit,
    write_source_supervised_training_audit,
)
from bayesian_phystwin.matphys_dino_features import CausalDinoNodeExtractor  # noqa: E402
from bayesian_phystwin.matphys_graph_parts import (  # noqa: E402
    GRAPH_PART_COMPACT_PROXY_CONTRACT,
    GRAPH_PART_PROXY_CONTRACT,
    prepare_graph_part_proxy,
)
from bayesian_phystwin.matphys_part_model import (  # noqa: E402
    PART_AWARE_MODEL_CONTRACT,
    install_part_aware_simple_model,
    summarize_part_spring_ratios,
)
from bayesian_phystwin.matphys_teacher_residual import (  # noqa: E402
    TEACHER_PARAMETERIZATION,
    MatPhysTeacherBundle,
    apply_matphys_teacher_residual,
    load_matphys_teacher_bundle,
    load_matphys_teacher_manifest,
)
from bayesian_phystwin.phystwin_graph import (  # noqa: E402
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_external_backbone import (  # noqa: E402
    EXTERNAL_COORDINATE_FRAME,
    EXTERNAL_VERTEX_CONTRACT,
)


MATPHYS_REPOSITORY = "https://github.com/Yrainy0615/MatPhys"
UNEVEN_DDP_TRAINING_CONTRACT = "ddp-join-uneven-case-steps-v1"
FINITE_OPTIMIZER_CONTRACT = "transactional-finite-adamw-v1"
TRUNK_INITIALIZATION_CONTRACT = "public-matphys-trunk-zero-residual-head-v1"
SINGLE_BACKWARD_AUXILIARY_CONTRACT = "intermediate-hook-single-backward-aux-v1"
_RESIDUAL_HEAD_PREFIXES = (
    "spring_base.5.",
    "ctrl_spring_base.5.",
    "spring_mod.",
)
_ACCESSED_FRAMES: dict[str, set[int]] = {}
_ACCESSED_FRAME_PATHS: dict[str, dict[int, Path]] = {}
_OBJECTIVE_END_FRAMES: dict[str, int] = {}
_FINITE_OPTIMIZER_STATS = {
    "attempted_steps": 0,
    "accepted_steps": 0,
    "rejected_pre_step": 0,
    "rejected_post_step": 0,
}
_TORCHVISION_STUB_LIBRARY = None


def _unavailable(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("Gaussian rendering is disabled for this MatPhys experiment")


def _tree_is_finite(value) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all().item()
        )
    if isinstance(value, Mapping):
        return all(_tree_is_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tree_is_finite(item) for item in value)
    return True


def _clone_optimizer_state(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_state(item) for item in value)
    return copy.deepcopy(value)


def _transactional_finite_optimizer(base_optimizer, diagnostics):
    """Wrap an optimizer so a non-finite update cannot poison a checkpoint."""

    import torch

    class TransactionalFiniteOptimizer(base_optimizer):
        finite_optimizer_contract = FINITE_OPTIMIZER_CONTRACT

        @torch.no_grad()
        def step(self, closure=None):
            diagnostics["attempted_steps"] += 1
            parameters = [
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            ]
            pre_step_values = [parameter.detach().clone() for parameter in parameters]
            pre_step_state = {
                parameter: _clone_optimizer_state(self.state[parameter])
                for parameter in parameters
                if parameter in self.state
            }
            finite_inputs = all(_tree_is_finite(parameter) for parameter in parameters)
            finite_inputs = finite_inputs and all(
                parameter.grad is None or _tree_is_finite(parameter.grad)
                for parameter in parameters
            )
            finite_inputs = finite_inputs and _tree_is_finite(self.state)
            if not finite_inputs:
                diagnostics["rejected_pre_step"] += 1
                return None

            result = super().step(closure=closure)
            finite_outputs = all(_tree_is_finite(parameter) for parameter in parameters)
            finite_outputs = finite_outputs and _tree_is_finite(self.state)
            if finite_outputs:
                diagnostics["accepted_steps"] += 1
                return result

            for parameter, previous in zip(parameters, pre_step_values, strict=True):
                parameter.copy_(previous)
                if parameter in pre_step_state:
                    self.state[parameter] = pre_step_state[parameter]
                else:
                    self.state.pop(parameter, None)
            diagnostics["rejected_post_step"] += 1
            return None

    TransactionalFiniteOptimizer.__name__ = (
        f"TransactionalFinite{base_optimizer.__name__}"
    )
    return TransactionalFiniteOptimizer


def _install_finite_adamw(training) -> None:
    base_optimizer = training.torch.optim.AdamW
    if getattr(base_optimizer, "finite_optimizer_contract", None) is not None:
        raise RuntimeError("finite optimizer guard was installed more than once")
    training.torch.optim.AdamW = _transactional_finite_optimizer(
        base_optimizer,
        _FINITE_OPTIMIZER_STATS,
    )


def _checkpoint_finiteness_report(checkpoint_path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def inspect_tree(value, prefix: str) -> tuple[int, list[dict[str, object]]]:
        nonfinite_count = 0
        examples: list[dict[str, object]] = []
        stack = [(prefix, value)]
        while stack:
            name, item = stack.pop()
            if isinstance(item, torch.Tensor):
                if not (item.is_floating_point() or item.is_complex()):
                    continue
                count = int((~torch.isfinite(item)).sum().item())
                nonfinite_count += count
                if count and len(examples) < 20:
                    examples.append(
                        {
                            "name": name,
                            "nonfinite_count": count,
                            "element_count": int(item.numel()),
                        }
                    )
            elif isinstance(item, Mapping):
                stack.extend((f"{name}.{key}", child) for key, child in item.items())
            elif isinstance(item, (tuple, list)):
                stack.extend(
                    (f"{name}[{index}]", child) for index, child in enumerate(item)
                )
        return nonfinite_count, examples

    model_count, model_examples = inspect_tree(
        checkpoint.get("model_state_dict", {}), "model_state_dict"
    )
    optimizer_count, optimizer_examples = inspect_tree(
        checkpoint.get("optimizer_state_dict", {}), "optimizer_state_dict"
    )
    return {
        "contract": "finite-model-and-optimizer-checkpoint-v1",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "finite": model_count == 0 and optimizer_count == 0,
        "model_nonfinite_count": model_count,
        "optimizer_nonfinite_count": optimizer_count,
        "examples": model_examples + optimizer_examples,
    }


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
        library.define(
            "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
        )
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
    pyplot.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)

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
    graphics.focal2fov = lambda focal, pixels: (
        2.0 * math.atan(float(pixels) / (2.0 * float(focal)))
    )
    graphics.fov2focal = lambda fov, pixels: (
        float(pixels) / (2.0 * math.tan(float(fov) / 2.0))
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
    from scipy.spatial import cKDTree

    class PointCloud:
        def __init__(self):
            self.points = np.zeros((0, 3), dtype=np.float64)

    class KDTreeFlann:
        def __init__(self, point_cloud):
            self.points = np.asarray(point_cloud.points, dtype=np.float64)
            self.tree = cKDTree(self.points)

        def _ordered(self, query, indices):
            indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            if not len(indices):
                return indices, np.empty(0, dtype=np.float64)
            delta = self.points[indices] - np.asarray(query, dtype=np.float64)
            distance_sq = np.einsum("ij,ij->i", delta, delta)
            order = np.lexsort((indices, distance_sq))
            return indices[order], distance_sq[order]

        def search_knn_vector_3d(self, query, count):
            count = min(max(int(count), 0), len(self.points))
            if count == 0:
                return 0, [], []
            _, raw_indices = self.tree.query(query, k=count)
            selected, distance_sq = self._ordered(query, raw_indices)
            return len(selected), selected.tolist(), distance_sq.tolist()

        def search_hybrid_vector_3d(self, query, radius, maximum):
            raw_indices = self.tree.query_ball_point(query, r=float(radius))
            indices, distance_sq = self._ordered(query, raw_indices)
            selected = indices[: int(maximum)]
            selected_distance = distance_sq[: len(selected)]
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
        GRAPH_PART_COMPACT_PROXY_CONTRACT
        if args.graph_parts and getattr(args, "compact_unused_edge_semantics", False)
        else GRAPH_PART_PROXY_CONTRACT
        if args.graph_parts
        else "global-onehot-single-part-v1"
    )
    existing = _validated_existing_proxy(proxy_root, cases, expected_contract)
    if existing is not None:
        return existing
    if args.graph_parts and getattr(args, "compact_unused_edge_semantics", False):
        raise FileNotFoundError(
            "compact graph proxy must be created from a byte-bound full proxy first"
        )
    mapping_path = (
        Path(args.matphys_root) / "semantic" / "case_to_material_different_types.json"
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
        case: json.loads((data_root / case / "split.json").read_text(encoding="utf-8"))
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
            raise RuntimeError(
                f"{case_name}: objective boundary changed during training"
            )
        return resolved

    training._resolve_train_frame = resolve_train_frame


def _install_source_supervised_objective_guard(
    training,
    data_root: Path,
    evidence_end_by_case: dict[str, int],
) -> None:
    """Allow complete source outcomes while keeping source video causal."""

    original = training._resolve_train_frame

    def resolve_train_frame(model_args, case_name: str, dataset_train_frame: int):
        split = json.loads(
            (data_root / case_name / "split.json").read_text(encoding="utf-8")
        )
        released_train_end = int(split["train"][1])
        frame_len = int(split.get("frame_len", split["test"][1]))
        evidence_end = int(evidence_end_by_case[case_name])
        if not 1 <= evidence_end <= released_train_end <= frame_len:
            raise RuntimeError(f"{case_name}: invalid source-supervised boundaries")
        requested = int(original(model_args, case_name, dataset_train_frame))
        if requested != frame_len:
            raise RuntimeError(
                f"{case_name}: source-supervised mode requires --fit_all_frames"
            )
        previous = _OBJECTIVE_END_FRAMES.setdefault(case_name, frame_len)
        if previous != frame_len:
            raise RuntimeError(
                f"{case_name}: objective boundary changed during training"
            )
        return frame_len

    training._resolve_train_frame = resolve_train_frame


def _collect_distributed_access_logs(
    output_dir: Path,
) -> (
    tuple[
        dict[str, set[int]],
        dict[str, dict[int, Path]],
        dict[str, int],
        list[Path],
        list[dict[str, int]],
    ]
    | None
):
    """Merge runtime frame/objective logs after MatPhys destroys its DDP group."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    log_dir = output_dir / "runtime_access_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rank_path = log_dir / f"rank_{rank:03d}.json"
    rank_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "accessed_frame_indices": {
                    case: sorted(indices) for case, indices in _ACCESSED_FRAMES.items()
                },
                "accessed_frame_paths": {
                    case: {str(frame_id): str(path) for frame_id, path in paths.items()}
                    for case, paths in _ACCESSED_FRAME_PATHS.items()
                },
                "objective_end_frames_exclusive": _OBJECTIVE_END_FRAMES,
                "finite_optimizer": dict(_FINITE_OPTIMIZER_STATS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if rank != 0:
        return None

    expected = [log_dir / f"rank_{index:03d}.json" for index in range(world_size)]
    deadline = time.monotonic() + 120.0
    while not all(path.is_file() for path in expected):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in expected if not path.is_file()]
            raise RuntimeError(f"timed out waiting for DDP access logs: {missing}")
        time.sleep(0.25)

    merged_frames: dict[str, set[int]] = {}
    merged_paths: dict[str, dict[int, Path]] = {}
    merged_objectives: dict[str, int] = {}
    optimizer_summaries: list[dict[str, int]] = []
    for path in expected:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["world_size"]) != world_size:
            raise RuntimeError("DDP access log world size changed")
        for case, indices in payload["accessed_frame_indices"].items():
            merged_frames.setdefault(case, set()).update(
                int(value) for value in indices
            )
        for case, raw_paths in payload["accessed_frame_paths"].items():
            case_paths = merged_paths.setdefault(case, {})
            for raw_id, raw_path in raw_paths.items():
                frame_id = int(raw_id)
                candidate = Path(raw_path).resolve()
                previous = case_paths.setdefault(frame_id, candidate)
                if previous != candidate:
                    raise RuntimeError(f"{case}: DDP ranks disagree on frame source")
        for case, raw_end in payload["objective_end_frames_exclusive"].items():
            objective_end = int(raw_end)
            previous = merged_objectives.setdefault(case, objective_end)
            if previous != objective_end:
                raise RuntimeError(f"{case}: DDP ranks disagree on objective boundary")
        optimizer_summaries.append(
            {
                key: int(value)
                for key, value in payload.get("finite_optimizer", {}).items()
            }
        )
    return (
        merged_frames,
        merged_paths,
        merged_objectives,
        expected,
        optimizer_summaries,
    )


def _authoritative_uneven_ddp_rank(forward_counts: list[int]) -> int:
    if not forward_counts or any(count < 0 for count in forward_counts):
        raise ValueError("DDP forward counts must be nonnegative")
    return max(
        range(len(forward_counts)), key=lambda rank: (forward_counts[rank], rank)
    )


def _broadcast_optimizer_state(optimizer, source_rank: int, device, dist) -> None:
    """Synchronize Adam state tensor-wise after an uneven DDP epoch.

    ``broadcast_object_list`` serializes CUDA optimizer tensors as one opaque
    payload. That path corrupted Adam moments on the production NCCL runtime.
    Here only small shape/type metadata uses object transport; every optimizer
    tensor follows a native checked collective.
    """

    import torch

    rank = dist.get_rank()
    source_state = optimizer.state_dict() if rank == source_rank else None
    if source_state is not None:
        state_metadata = {}
        for parameter_id, values in source_state["state"].items():
            state_metadata[parameter_id] = {}
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    state_metadata[parameter_id][name] = {
                        "kind": "tensor",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype).removeprefix("torch."),
                    }
                else:
                    state_metadata[parameter_id][name] = {
                        "kind": "value",
                        "value": copy.deepcopy(value),
                    }
        metadata = {
            "state": state_metadata,
            "param_groups": copy.deepcopy(source_state["param_groups"]),
        }
    else:
        metadata = None
    payload = [metadata]
    dist.broadcast_object_list(payload, src=source_rank, device=device)
    metadata = payload[0]
    if not isinstance(metadata, Mapping):
        raise RuntimeError("optimizer synchronization received invalid metadata")

    received_state = {}
    for parameter_id, values in metadata["state"].items():
        received_state[parameter_id] = {}
        for name, specification in values.items():
            if specification["kind"] == "value":
                received_state[parameter_id][name] = copy.deepcopy(
                    specification["value"]
                )
                continue
            dtype_name = str(specification["dtype"])
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise RuntimeError(f"unsupported optimizer dtype: {dtype_name}")
            if source_state is not None:
                transfer = (
                    source_state["state"][parameter_id][name]
                    .detach()
                    .to(device=device, dtype=dtype)
                    .contiguous()
                )
            else:
                transfer = torch.empty(
                    tuple(int(value) for value in specification["shape"]),
                    dtype=dtype,
                    device=device,
                )
            dist.broadcast(transfer, src=source_rank)
            received_state[parameter_id][name] = transfer

    if rank != source_rank:
        optimizer.load_state_dict(
            {
                "state": received_state,
                "param_groups": metadata["param_groups"],
            }
        )
    local_finite = torch.tensor(
        [int(_tree_is_finite(optimizer.state))],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
    if int(local_finite.item()) != 1:
        raise RuntimeError("optimizer synchronization produced non-finite state")


def _rank_local_training_device(requested_device: str) -> str:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return requested_device
    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank is None or int(raw_rank) < 0:
        raise RuntimeError("distributed MatPhys training requires LOCAL_RANK")
    return f"cuda:{int(raw_rank)}"


def _enable_unused_parameter_ddp(training) -> None:
    base_ddp = training.DDP

    def ddp_with_unused_detection(*args, **kwargs):
        kwargs["find_unused_parameters"] = True
        return base_ddp(*args, **kwargs)

    training.DDP = ddp_with_unused_detection


def _install_single_backward_auxiliary_loss(training) -> None:
    """Fold an auxiliary output loss into the simulator's one DDP backward."""

    import torch

    original = training._rollout_aux_loss
    applied_output_keys = (
        "log_k",
        "ctrl_log_k",
        "collide_elas",
        "collide_fric",
        "collide_object_elas",
        "collide_object_fric",
        "collision_dist",
        "dashpot_damping",
        "drag_damping",
    )

    def rollout_aux_loss(model_out, batch, idx, device, args, epoch):
        auxiliary, stats = original(model_out, batch, idx, device, args, epoch)
        if not auxiliary.requires_grad:
            return auxiliary, stats
        targets = []
        for key in applied_output_keys:
            value = model_out.get(key)
            if isinstance(value, torch.Tensor) and value.requires_grad:
                targets.append(value)
        gradients = torch.autograd.grad(
            auxiliary,
            targets,
            retain_graph=True,
            allow_unused=True,
        )
        for target, gradient in zip(targets, gradients, strict=True):
            if gradient is None:
                continue
            frozen_gradient = gradient.detach()

            def add_auxiliary_gradient(incoming, *, addition=frozen_gradient):
                return incoming + addition

            target.register_hook(add_auxiliary_gradient)
        return auxiliary.detach(), stats

    training._rollout_aux_loss = rollout_aux_loss
    training._single_backward_auxiliary_contract = SINGLE_BACKWARD_AUXILIARY_CONTRACT


def _install_uneven_ddp_training(training) -> None:
    """Make variable-length interaction shards safe for MatPhys DDP.

    MatPhys performs one model forward and optimizer update per simulation frame,
    while its distributed sampler shards whole interactions. The resulting ranks
    execute different numbers of forwards. DDP join shadows the missing
    collectives; the last active rank then supplies both model and optimizer
    state before the next epoch.
    """

    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    original_train_epoch = training.train_epoch
    original_is_distributed = training._is_distributed

    def train_epoch(model, train_loader, optimizer, device, runtimes, args, epoch):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "uneven DDP training requires an initialized process group"
            )
        if not hasattr(model, "join"):
            raise TypeError("uneven DDP training requires DistributedDataParallel")
        original_forward_case = training.forward_case
        local_forward_count = 0

        def counted_forward_case(*forward_args, **forward_kwargs):
            nonlocal local_forward_count
            local_forward_count += 1
            return original_forward_case(*forward_args, **forward_kwargs)

        training.forward_case = counted_forward_case
        # The upstream epoch reduces statistics before a short rank can enter
        # DDP's join context. Defer those reductions until every rank has joined.
        training._is_distributed = lambda: False
        try:
            with model.join(divide_by_initial_world_size=True, enable=True):
                stats = original_train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    runtimes,
                    args,
                    epoch,
                )
        finally:
            training.forward_case = original_forward_case
            training._is_distributed = original_is_distributed

        count = torch.tensor([local_forward_count], dtype=torch.int64, device=device)
        gathered = [torch.zeros_like(count) for _ in range(world_size)]
        dist.all_gather(gathered, count)
        forward_counts = [int(value.item()) for value in gathered]
        source_rank = _authoritative_uneven_ddp_rank(forward_counts)

        _broadcast_optimizer_state(optimizer, source_rank, device, dist)

        metric_keys = (
            "loss",
            "track",
            "geo",
            "render",
            "acc",
            "phys_part",
            "phys_global",
            "teacher_log_k",
            "teacher_global",
        )
        local_graphs = float(stats["num_graphs"])
        packed = torch.tensor(
            [float(stats[key]) * local_graphs for key in metric_keys] + [local_graphs],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        graph_count = float(packed[-1].item())
        if graph_count <= 0.0:
            raise RuntimeError("uneven DDP epoch produced no valid source graphs")
        result = {
            key: float(packed[index].item() / graph_count)
            for index, key in enumerate(metric_keys)
        }
        result["num_graphs"] = int(graph_count)
        return result

    training.train_epoch = train_epoch


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


def _load_trunk_initialization(
    args,
) -> tuple[Mapping[str, object] | None, dict[str, object] | None]:
    raw_path = args.initialization_checkpoint
    expected_hash = args.initialization_sha256
    if raw_path is None:
        if expected_hash is not None:
            raise ValueError(
                "--initialization-sha256 requires --initialization-checkpoint"
            )
        return None, None
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("initialization checkpoint hash does not match")

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("initialization checkpoint contains no model state")
    if not _tree_is_finite(state):
        raise ValueError("initialization checkpoint contains non-finite tensors")
    return state, {
        "contract": TRUNK_INITIALIZATION_CONTRACT,
        "checkpoint": {"path": str(path), "sha256": actual_hash},
        "excluded_prefixes": list(_RESIDUAL_HEAD_PREFIXES),
        "reason": (
            "reuse the source-trained representation while preserving the "
            "exact zero-residual PhysTwin identity"
        ),
    }


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
        return apply_matphys_teacher_residual(output, bundles[case], residual_log_scale)

    training.forward_case = forward_case


def train(args) -> None:
    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    cases = _case_names(args.cases)
    source_supervised = args.training_contract == "source-supervised-meta"
    target_cases = _case_names(args.target_cases) if args.target_cases else []
    if source_supervised:
        if not target_cases or args.split_registration is None:
            raise ValueError(
                "source-supervised-meta requires --target-cases and "
                "--split-registration"
            )
        if set(cases) & set(target_cases):
            raise ValueError("source and target cases must be disjoint")
    elif target_cases or args.split_registration is not None:
        raise ValueError(
            "target cases and split registration require source-supervised-meta"
        )
    _ACCESSED_FRAMES.clear()
    _ACCESSED_FRAME_PATHS.clear()
    _OBJECTIVE_END_FRAMES.clear()
    for key in _FINITE_OPTIMIZER_STATS:
        _FINITE_OPTIMIZER_STATS[key] = 0
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if (
        not np.isfinite(args.teacher_proximity_weight)
        or args.teacher_proximity_weight < 0
    ):
        raise ValueError("teacher proximity weight must be finite and nonnegative")
    if not np.isfinite(args.grad_clip) or args.grad_clip <= 0.0:
        raise ValueError("gradient clip must be finite and positive")
    if args.random_seed < 0:
        raise ValueError("random seed must be nonnegative")
    evidence_end_by_case = _evidence_ends(data_root, cases, args.fit_fraction)
    target_evidence_end_by_case = (
        _evidence_ends(data_root, target_cases, args.target_fit_fraction)
        if source_supervised
        else {}
    )
    if args.graph_parts and args.teacher_residual_log_scale is None:
        raise ValueError("graph parts require the identity-preserving teacher residual")
    proxy = _prepare_proxy(args, cases, evidence_end_by_case)
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    import train_model_video_material_simple as training

    initialization_state, initialization_identity = _load_trunk_initialization(args)
    if initialization_state is not None and not args.graph_parts:
        raise ValueError("trunk initialization currently requires --graph-parts")
    if (
        initialization_state is None
        and args.videomae_model != MATPHYS_GENERIC_MOTION_BACKBONE
    ):
        raise ValueError("fresh training requires the locked generic VideoMAE backbone")
    if proxy["contract"] in (
        GRAPH_PART_PROXY_CONTRACT,
        GRAPH_PART_COMPACT_PROXY_CONTRACT,
    ):
        semantic_dimensions = {
            int(record["semantic_dimension"]) for record in proxy["cases"]
        }
        if len(semantic_dimensions) != 1:
            raise ValueError("part-aware training requires one DINO feature width")
        install_part_aware_simple_model(
            training,
            part_feature_dim=semantic_dimensions.pop(),
            part_feature_scale=args.part_feature_scale,
            initialization_state_dict=initialization_state,
            initialization_excluded_prefixes=_RESIDUAL_HEAD_PREFIXES,
        )
    parameterization, teacher_bundles = _teacher_configuration(args, cases)
    if parameterization is not None and proxy["contract"] in (
        GRAPH_PART_PROXY_CONTRACT,
        GRAPH_PART_COMPACT_PROXY_CONTRACT,
    ):
        parameterization.update(
            {
                "part_model_contract": PART_AWARE_MODEL_CONTRACT,
                "part_feature_scale": float(args.part_feature_scale),
                "proxy_contract": GRAPH_PART_PROXY_CONTRACT,
                "fit_fraction": float(args.fit_fraction),
                "optimization": {
                    "learning_rate": float(args.learning_rate),
                    "teacher_proximity_weight": float(args.teacher_proximity_weight),
                    "gradient_clip": float(args.grad_clip),
                    "finite_optimizer_guard": bool(args.finite_optimizer_guard),
                    "auxiliary_backward": (
                        SINGLE_BACKWARD_AUXILIARY_CONTRACT
                        if args.teacher_proximity_weight > 0.0
                        else "upstream"
                    ),
                },
            }
        )
        if initialization_identity is not None:
            parameterization["initialization"] = initialization_identity
        else:
            parameterization["initialization"] = matphys_fresh_fold_initialization(
                args.random_seed
            )
    if parameterization is not None:
        _install_teacher_parameterization(
            training,
            teacher_bundles,
            float(parameterization["residual_log_scale"]),
        )
    training.load_video_frames = _causal_video_loader(evidence_end_by_case)
    if args.finite_optimizer_guard:
        _install_finite_adamw(training)
    if args.teacher_proximity_weight > 0.0:
        _install_single_backward_auxiliary_loss(training)
    if source_supervised:
        _install_source_supervised_objective_guard(
            training, data_root, evidence_end_by_case
        )
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            _enable_unused_parameter_ddp(training)
            _install_uneven_ddp_training(training)
            if parameterization is not None:
                parameterization["distributed_training_contract"] = (
                    UNEVEN_DDP_TRAINING_CONTRACT
                )
    else:
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
        _rank_local_training_device(args.device),
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--train_ratio",
        "1.0",
        "--lr",
        str(args.learning_rate),
        "--seed",
        str(args.random_seed),
        "--videomae_model",
        str(args.videomae_model),
        "--grad_clip",
        str(args.grad_clip),
        "--lambda_track",
        "1.0",
        "--lambda_geo",
        "1.0",
        "--lambda_render",
        "0.0",
        "--lambda_phys_prior",
        "0.0",
        "--lambda_teacher_k",
        str(args.teacher_proximity_weight),
        "--lambda_acc_smooth",
        "0.01",
        "--save_best_only",
        "--vis_every",
        "0",
    ]
    if len(cases) == 1:
        train_args.extend(("--case_name", cases[0]))
    if source_supervised:
        train_args.append("--fit_all_frames")
    sys.argv = train_args
    training.main()
    output_dir = Path(args.output_dir).resolve()
    access_logs = _collect_distributed_access_logs(output_dir)
    if access_logs is None:
        return
    (
        accessed_frames,
        accessed_paths,
        objective_ends,
        access_log_paths,
        optimizer_summaries,
    ) = access_logs
    checkpoints = [output_dir / "last_checkpoint.pth"]
    if not checkpoints[0].is_file():
        raise RuntimeError("MatPhys training did not write a checkpoint")
    finiteness = _checkpoint_finiteness_report(checkpoints[0])
    finiteness_path = output_dir / "checkpoint_finiteness.json"
    finiteness_path.write_text(
        json.dumps(finiteness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    access_log_paths.append(finiteness_path)
    if not finiteness["finite"]:
        raise RuntimeError(
            "MatPhys terminal checkpoint failed the finite competence gate"
        )
    if parameterization is not None:
        parameterization["optimizer_guard"] = {
            "contract": (
                FINITE_OPTIMIZER_CONTRACT if args.finite_optimizer_guard else "disabled"
            ),
            "rank_summaries": optimizer_summaries,
            "total_rejected_steps": int(
                sum(
                    summary.get("rejected_pre_step", 0)
                    + summary.get("rejected_post_step", 0)
                    for summary in optimizer_summaries
                )
            ),
        }
    common_audit = {
        "source_repository": MATPHYS_REPOSITORY,
        "source_commit": _source_commit(matphys_root),
        "data_root": data_root,
        "accessed_frame_indices": {
            case: sorted(indices) for case, indices in accessed_frames.items()
        },
        "accessed_frame_paths": accessed_paths,
        "objective_end_frames_exclusive": objective_ends,
        "evidence_end_frames_exclusive": evidence_end_by_case,
        "proxy_summary_path": proxy["summary_path"],
        "parameterization": parameterization,
        "runtime_access_log_paths": access_log_paths,
    }
    if source_supervised:
        implementation_paths = [
            Path(__file__).resolve(),
            Path(inspect.getsourcefile(install_part_aware_simple_model)).resolve(),
        ]
        if args.implementation_amendment is not None:
            implementation_paths.append(Path(args.implementation_amendment).resolve())
        audit = write_source_supervised_training_audit(
            checkpoints,
            output_dir / "source_supervised_training_audit.json",
            source_cases=cases,
            target_cases=target_cases,
            target_evidence_end_frames_exclusive=target_evidence_end_by_case,
            split_by_case=_splits(data_root, cases + target_cases),
            split_registration_path=args.split_registration,
            implementation_paths=implementation_paths,
            **common_audit,
        )
    else:
        audit = write_causal_training_audit(
            checkpoints,
            output_dir / "causal_training_audit.json",
            split_by_case=_splits(data_root, cases),
            **common_audit,
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
            "sem_cache_dir": str(Path(args.proxy_root).resolve() / "semantic_cache"),
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


def _model_spring_y(training, runtime, model_out, device):
    """Materialize the complete positive spring field applied by MatPhys."""

    import torch

    model_logk = training._build_model_logk(model_out, runtime, runtime.sim, device)
    if model_logk is None:
        raise RuntimeError("MatPhys spring field does not match the runtime topology")
    spring_y = torch.exp(model_logk).detach().cpu().numpy().reshape(-1)
    expected_count = int(runtime.sim.wp_spring_Y.shape[0])
    if spring_y.shape != (expected_count,):
        raise RuntimeError("MatPhys produced an invalid complete spring field")
    if not np.isfinite(spring_y).all() or np.any(spring_y <= 0.0):
        raise RuntimeError("MatPhys produced a non-positive or non-finite spring field")
    return model_logk, spring_y.astype(np.float32, copy=False)


def _rollout_model_output(
    training,
    runtime,
    model_out,
    device,
    train_end: int,
    *,
    model_logk=None,
):
    """Apply one spring field and run one fresh official Warp trajectory."""

    import warp as wp

    sim = runtime.sim
    if model_logk is None:
        model_logk, _ = _model_spring_y(training, runtime, model_out, device)
    training._apply_model_out_to_sim(model_out, model_logk, sim)
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    vertices = [
        wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).detach().cpu().numpy()
    ]
    frame_count = int(runtime.trainer.dataset.frame_len)
    for frame_index in range(1, frame_count):
        sim.set_controller_target(frame_index, pure_inference=frame_index >= train_end)
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
    raw_audit = json.loads(Path(args.training_audit).read_text(encoding="utf-8"))
    source_supervised = (
        raw_audit.get("contract") == MATPHYS_SOURCE_SUPERVISED_AUDIT_CONTRACT
    )
    audit = (
        validate_source_supervised_training_audit(args.training_audit, checkpoint_path)
        if source_supervised
        else validate_causal_training_audit(args.training_audit, checkpoint_path)
    )
    cases = _case_names(args.cases)
    audit_case_records = audit["target_cases"] if source_supervised else audit["cases"]
    evidence_end_by_case = {
        str(record["name"]): int(record["evidence_end_frame_exclusive"])
        for record in audit_case_records
    }
    if source_supervised:
        if not set(cases).issubset(evidence_end_by_case):
            raise ValueError("export request includes an unregistered target case")
        evidence_end_by_case = {case: evidence_end_by_case[case] for case in cases}
    elif set(evidence_end_by_case) != set(cases):
        raise ValueError("training audit cases do not match the export request")
    audit_proxy = audit.get("proxy")
    if not isinstance(audit_proxy, dict):
        raise ValueError("training audit omits its proxy")
    proxy_summary = json.loads(Path(audit_proxy["path"]).read_text(encoding="utf-8"))
    args.graph_parts = proxy_summary.get("contract") in (
        GRAPH_PART_PROXY_CONTRACT,
        GRAPH_PART_COMPACT_PROXY_CONTRACT,
    )
    if source_supervised and args.graph_parts:
        source_proxy_cases = proxy_summary.get("cases", [])
        if not isinstance(source_proxy_cases, list) or not source_proxy_cases:
            raise ValueError("source graph proxy contains no cases")
        provenances = [record.get("provenance") for record in source_proxy_cases]
        if not all(isinstance(record, dict) for record in provenances):
            raise ValueError("source graph proxy omits DINO provenance")

        def one_proxy_value(field: str):
            values = {record[field] for record in provenances}
            if len(values) != 1:
                raise ValueError(f"source graph proxy disagrees on {field}")
            return values.pop()

        frame_counts = {len(record["frame_ids"]) for record in provenances}
        if len(frame_counts) != 1:
            raise ValueError("source graph proxy disagrees on keyframe count")
        args.dino_model = str(one_proxy_value("model_name"))
        args.dino_image_size = int(one_proxy_value("image_size"))
        args.dino_keyframes = int(frame_counts.pop())
        args.part_count = int(proxy_summary["part_count"])
        args.semantic_edge_weight = float(proxy_summary["semantic_edge_weight"])
    if not (Path(args.proxy_root).resolve() / "proxy_summary.json").is_file():
        if not source_supervised:
            raise FileNotFoundError("export requires the byte-bound training proxy")
    proxy = _prepare_proxy(args, cases, evidence_end_by_case)
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    # MatPhys's warning filter targets a Warp-private helper removed in newer
    # Warp builds. Supplying the standard equivalent keeps export-only replays
    # compatible without changing any simulation path.
    import warnings

    import warp._src.utils as warp_utils

    if not hasattr(warp_utils, "warn"):
        warp_utils.warn = warnings.warn
    from material_param_dataset import (
        MaterialDatasetConfig,
        MaterialParamDataset,
    )
    import train_model_video_material_simple as training

    parameterization = audit.get("parameterization")
    if proxy["contract"] in (
        GRAPH_PART_PROXY_CONTRACT,
        GRAPH_PART_COMPACT_PROXY_CONTRACT,
    ):
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
        if source_supervised:
            if args.target_teacher_experiments_dir is None:
                raise ValueError(
                    "source-supervised export requires --target-teacher-experiments-dir"
                )
            teacher_bundles = {
                case: load_matphys_teacher_bundle(
                    case,
                    args.target_teacher_experiments_dir,
                    args.experiments_optimization_dir,
                )
                for case in cases
            }
            parameterization = {
                **parameterization,
                "teacher": {
                    "cases": [teacher_bundles[case].manifest() for case in cases]
                },
                "training_teacher_scope": "registered-source-cases-only-v1",
            }
        else:
            teacher_bundles = load_matphys_teacher_manifest(parameterization["teacher"])
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
            if proxy["contract"]
            in (GRAPH_PART_PROXY_CONTRACT, GRAPH_PART_COMPACT_PROXY_CONTRACT)
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
            model_logk, candidate_spring_y = _model_spring_y(
                training, runtime, model_out, device
            )
            object_log_k = model_out["log_k"].detach().cpu().numpy().reshape(-1)
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
                    "residual_log_scale": float(parameterization["residual_log_scale"]),
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
            spring_field_path = output_root / "cases" / case / "candidate_spring_y.npy"
            np.save(spring_field_path, candidate_spring_y, allow_pickle=False)
            spring_field_identity = {
                "path": str(spring_summary_path),
                "sha256": sha256_file(spring_summary_path),
                "complete_spring_y": {
                    "path": str(spring_field_path),
                    "sha256": sha256_file(spring_field_path),
                    "count": int(len(candidate_spring_y)),
                    "minimum": float(np.min(candidate_spring_y)),
                    "maximum": float(np.max(candidate_spring_y)),
                },
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
            **({"model_logk": model_logk} if parameterization is not None else {}),
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
                "MatPhys source-supervised causal graph-part PhysTwin residual"
                if source_supervised
                and proxy["contract"]
                in (GRAPH_PART_PROXY_CONTRACT, GRAPH_PART_COMPACT_PROXY_CONTRACT)
                else "MatPhys source-supervised causal PhysTwin residual"
                if source_supervised
                else "MatPhys causal graph-part released-PhysTwin residual"
                if proxy["contract"]
                in (GRAPH_PART_PROXY_CONTRACT, GRAPH_PART_COMPACT_PROXY_CONTRACT)
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
            "training_scope": (
                "registered-source-supervised-target-disjoint-v1"
                if source_supervised
                else "independent-per-case-fixed-terminal-v1"
            ),
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
    train_parser.add_argument("--compact-unused-edge-semantics", action="store_true")
    train_parser.add_argument(
        "--training-contract",
        choices=("causal-prefix-only", "source-supervised-meta"),
        default="causal-prefix-only",
    )
    train_parser.add_argument("--target-cases")
    train_parser.add_argument("--split-registration")
    train_parser.add_argument("--implementation-amendment")
    train_parser.add_argument("--target-fit-fraction", type=float, default=0.75)
    train_parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    train_parser.add_argument("--random-seed", type=int, default=42)
    train_parser.add_argument(
        "--videomae-model", default=MATPHYS_GENERIC_MOTION_BACKBONE
    )
    train_parser.add_argument("--grad-clip", type=float, default=5.0)
    train_parser.add_argument("--teacher-proximity-weight", type=float, default=0.0)
    train_parser.add_argument("--finite-optimizer-guard", action="store_true")
    train_parser.add_argument("--initialization-checkpoint")
    train_parser.add_argument("--initialization-sha256")
    train_parser.set_defaults(handler=train)

    export_parser = subparsers.add_parser("export", parents=[common])
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--training-audit", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--target-teacher-experiments-dir")
    export_parser.add_argument(
        "--initial-alignment-tolerance-m", type=float, default=1e-6
    )
    export_parser.set_defaults(handler=export)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    _install_torchvision_nms_stub()
    args.handler(args)


if __name__ == "__main__":
    main()
    (matphys_fresh_fold_initialization,)
