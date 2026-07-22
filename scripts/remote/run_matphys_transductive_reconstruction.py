#!/usr/bin/env python3
"""Run MatPhys's per-case all-frame reconstruction recipe with hard provenance.

This is deliberately not a causal predictor. Both the optimization objective and
the video encoder may consume frames from the released test interval. The CLI
therefore requires an explicit acknowledgement and exports a separate artifact
contract that cannot be mistaken for a Bayesian-PhysTwin future rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_REMOTE_DIR = Path(__file__).resolve().parent
if str(_REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(_REMOTE_DIR))

from run_matphys_causal import (  # noqa: E402
    MATPHYS_REPOSITORY,
    _checkpoint_finiteness_report,
    _configure_matphys_imports,
    _install_torchvision_nms_stub,
    _rollout_model_output,
    _source_commit,
    sha256_file,
)

from bayesian_phystwin.phystwin_official_evaluation import (  # noqa: E402
    evaluate_official_phystwin_files,
)


TRANSDUCTIVE_CONTRACT = "matphys-offline-all-frame-reconstruction-v1"
SELECTION_CONTRACT = "upstream-lexicographic-uniform-video-v1"
PINNED_MATPHYS_COMMIT = "c16b858dfb79bf21024ead24b45a710600de7b4f"
CLAIM_BOUNDARY = (
    "offline per-case reconstruction control only; future observations and "
    "released test outcomes are used during fitting; not causal future prediction"
)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _require_acknowledgement(acknowledged: bool) -> None:
    if not acknowledged:
        raise ValueError(
            "this command uses future observations; pass "
            "--acknowledge-future-observations explicitly"
        )


def _validated_source_commit(matphys_root: Path) -> str:
    commit = _source_commit(matphys_root)
    if commit != PINNED_MATPHYS_COMMIT:
        raise ValueError(
            f"MatPhys source is {commit}, expected pinned {PINNED_MATPHYS_COMMIT}"
        )
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=matphys_root,
        text=True,
    )
    if status.strip():
        raise ValueError("MatPhys tracked source tree is dirty")
    return commit


def _file_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _validate_file_identity(identity: Mapping[str, object], label: str) -> None:
    path = Path(str(identity.get("path", ""))).resolve()
    expected = str(identity.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} changed after training: {path}")


def _case_split(data_root: Path, case_name: str) -> dict[str, object]:
    split_path = data_root / case_name / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = [int(value) for value in split["train"]]
    test = [int(value) for value in split["test"]]
    frame_len = int(split["frame_len"])
    if train[0] != 0 or train[1] != test[0] or test[1] != frame_len:
        raise ValueError(f"{case_name}: split is not contiguous through frame_len")
    if not 1 < train[1] < frame_len:
        raise ValueError(f"{case_name}: split has no usable train/test boundary")
    return {
        "frame_len": frame_len,
        "train": train,
        "test": test,
        "identity": _file_identity(split_path),
    }


def _lexicographic_frame_selection(
    data_root: Path,
    case_name: str,
    frame_count: int,
) -> list[dict[str, object]]:
    """Exactly mirror public MatPhys frame sorting and uniform subsampling."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    color_dir = data_root / case_name / "color" / "0"
    frame_paths = sorted(
        path.resolve()
        for path in color_dir.iterdir()
        if path.is_file() and path.name.endswith(".png")
    )
    if not frame_paths:
        raise FileNotFoundError(f"no PNG frames in {color_dir}")
    positions = np.linspace(0, len(frame_paths) - 1, frame_count, dtype=int)
    return [
        {
            "slot": slot,
            "lexicographic_position": int(position),
            "filename": frame_paths[int(position)].name,
            **_file_identity(frame_paths[int(position)]),
        }
        for slot, position in enumerate(positions.tolist())
    ]


class _VideoAccessLog:
    def __init__(
        self,
        expected: list[dict[str, object]] | None = None,
    ) -> None:
        self.expected = expected
        self.calls = 0
        self.selected: list[dict[str, object]] = []


def _all_frame_video_loader(
    access: _VideoAccessLog,
    *,
    expected_case: str,
    expected_data_root: Path,
):
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

        resolved_root = Path(base_path).resolve()
        if case_name != expected_case or resolved_root != expected_data_root:
            raise RuntimeError("all-frame loader received an unregistered case or root")
        selected = _lexicographic_frame_selection(resolved_root, case_name, T)
        if access.expected is not None and selected != access.expected:
            raise RuntimeError("all-frame video selection differs from training audit")
        if access.selected and selected != access.selected:
            raise RuntimeError("all-frame video selection changed during this run")
        access.selected = selected
        access.calls += 1

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
            transform(Image.open(record["path"]).convert("RGB"))
            for record in selected
        ]
        tensor = torch.stack(frames, dim=0).unsqueeze(0)
        return tensor.to(device) if device is not None else tensor

    return load_video_frames


def _common_input_identities(args, split: Mapping[str, object]) -> dict[str, object]:
    data_root = Path(args.data_root).resolve()
    case_root = data_root / args.case
    result_root = Path(args.results_dir).resolve() / args.case
    result_files = sorted(path for path in result_root.rglob("*") if path.is_file())
    if not result_files:
        raise FileNotFoundError(f"material result tree is empty: {result_root}")
    teacher_files = sorted(
        (Path(args.experiments_dir).resolve() / args.case / "train").glob(
            "best_*.pth"
        )
    )
    if not teacher_files:
        raise FileNotFoundError(f"no best teacher checkpoint for {args.case}")
    return {
        "split": split["identity"],
        "final_data": _file_identity(case_root / "final_data.pkl"),
        "gt_track_3d": _file_identity(case_root / "gt_track_3d.pkl"),
        "metadata": _file_identity(case_root / "metadata.json"),
        "calibration": _file_identity(case_root / "calibrate.pkl"),
        "optimal_params": _file_identity(
            Path(args.experiments_optimization_dir).resolve()
            / args.case
            / "optimal_params.pkl"
        ),
        "teacher_checkpoint": _file_identity(teacher_files[-1]),
        "case_to_material": _file_identity(args.case_to_material),
        "node_semantics": _file_identity(
            Path(args.sem_cache_dir).resolve() / f"{args.case}_node_sem.npz"
        ),
        "material_results": [_file_identity(path) for path in result_files],
    }


def _training_argv(args) -> list[str]:
    return [
        "train_model_video_material_simple.py",
        "--case_name",
        args.case,
        "--save_dir",
        str(Path(args.output_dir).resolve()),
        "--base_path",
        str(Path(args.data_root).resolve()),
        "--experiments_dir",
        str(Path(args.experiments_dir).resolve()),
        "--experiments_optimization_dir",
        str(Path(args.experiments_optimization_dir).resolve()),
        "--case_to_material",
        str(Path(args.case_to_material).resolve()),
        "--results_dir",
        str(Path(args.results_dir).resolve()),
        "--sem_cache_dir",
        str(Path(args.sem_cache_dir).resolve()),
        "--gaussian_root",
        "__disabled__",
        "--videomae_model",
        args.videomae_model,
        "--videomae_image_size",
        "224",
        "--num_video_frames",
        "16",
        "--d_motion",
        "128",
        "--mat_codebook_dim",
        "32",
        "--hidden_dim",
        "256",
        "--num_materials",
        "10",
        "--batch_size",
        "4",
        "--num_workers",
        "0",
        "--epochs",
        str(args.epochs),
        "--eval_every",
        str(args.eval_every),
        "--device",
        args.device,
        "--lr",
        "3e-4",
        "--seed",
        "42",
        "--lambda_track",
        "1.0",
        "--lambda_geo",
        "1.0",
        "--lambda_render",
        "0",
        "--grad_scale",
        "1000.0",
        "--grad_clip",
        "5.0",
        "--logk_residual_scale",
        "1.0",
        "--logk_soft_clamp",
        "0.25",
        "--lambda_phys_prior",
        "0.001",
        "--phys_prior_part_mode",
        "empirical_kl",
        "--phys_prior_global_mode",
        "barrier",
        "--phys_prior_start_epoch",
        "5",
        "--lambda_acc_smooth",
        "0.01",
        "--fit_all_frames",
        "--save_best_only",
        "--vis_every",
        "0",
    ]


def _validate_training_schedule(epochs: int, eval_every: int) -> None:
    if epochs < 1 or eval_every < 1:
        raise ValueError("epochs and eval_every must be positive")
    if epochs % eval_every != 0:
        raise ValueError("epochs must be divisible by eval_every to save a final checkpoint")


def _checkpoint_namespace(raw: Mapping[str, object], args) -> SimpleNamespace:
    values = dict(raw)
    values.update(
        {
            "base_path": str(Path(args.data_root).resolve()),
            "experiments_dir": str(Path(args.experiments_dir).resolve()),
            "experiments_optimization_dir": str(
                Path(args.experiments_optimization_dir).resolve()
            ),
            "case_to_material": str(Path(args.case_to_material).resolve()),
            "results_dir": str(Path(args.results_dir).resolve()),
            "sem_cache_dir": str(Path(args.sem_cache_dir).resolve()),
            "gaussian_root": "__disabled__",
            "device": args.device,
            "rank": 0,
            "lambda_render": 0.0,
            "fit_all_frames": True,
        }
    )
    values.setdefault("logk_residual_scale", 1.0)
    values.setdefault("logk_soft_clamp", 0.25)
    return SimpleNamespace(**values)


def _environment_record() -> dict[str, object]:
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }


def train(args) -> None:
    _require_acknowledgement(args.acknowledge_future_observations)
    _validate_training_schedule(args.epochs, args.eval_every)
    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    split = _case_split(data_root, args.case)
    source_commit = _validated_source_commit(matphys_root)
    inputs = _common_input_identities(args, split)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matphys_imports(matphys_root)
    _install_torchvision_nms_stub()
    import train_model_video_material_simple as training

    access = _VideoAccessLog()
    training.load_video_frames = _all_frame_video_loader(
        access,
        expected_case=args.case,
        expected_data_root=data_root,
    )
    argv = _training_argv(args)
    previous_argv = sys.argv
    started = time.time()
    try:
        with _working_directory(matphys_root):
            sys.argv = argv
            training.main()
    finally:
        sys.argv = previous_argv
    elapsed = time.time() - started
    if access.calls < 1 or not access.selected:
        raise RuntimeError("MatPhys training did not access all-frame video")

    checkpoint_records = {}
    for name in ("best_checkpoint.pth", "last_checkpoint.pth"):
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"MatPhys training did not write {name}")
        finiteness = _checkpoint_finiteness_report(path)
        if not finiteness["finite"]:
            raise RuntimeError(f"non-finite MatPhys checkpoint: {path}")
        checkpoint_records[name] = finiteness

    source_files = {
        "training_implementation": _file_identity(
            matphys_root / "semantic" / "train_model_video_material_simple.py"
        ),
        "published_recipe": _file_identity(matphys_root / "scripts" / "ours" / "train_all.sh"),
        "wrapper": _file_identity(__file__),
    }
    audit = {
        "schema_version": 1,
        "contract": TRANSDUCTIVE_CONTRACT,
        "claim_boundary": CLAIM_BOUNDARY,
        "future_observations_used": True,
        "released_test_outcomes_used_in_objective": True,
        "case_name": args.case,
        "source_repository": MATPHYS_REPOSITORY,
        "source_commit": source_commit,
        "tracked_source_clean": True,
        "paths": {
            "matphys_root": str(matphys_root),
            "data_root": str(data_root),
            "experiments_dir": str(Path(args.experiments_dir).resolve()),
            "experiments_optimization_dir": str(
                Path(args.experiments_optimization_dir).resolve()
            ),
            "results_dir": str(Path(args.results_dir).resolve()),
            "sem_cache_dir": str(Path(args.sem_cache_dir).resolve()),
            "case_to_material": str(Path(args.case_to_material).resolve()),
        },
        "split": split,
        "objective_end_frame_exclusive": int(split["frame_len"]),
        "video": {
            "contract": SELECTION_CONTRACT,
            "call_count": access.calls,
            "selected_frames": access.selected,
        },
        "inputs": inputs,
        "source_files": source_files,
        "training": {
            "argv": argv,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "elapsed_seconds": elapsed,
        },
        "environment": _environment_record(),
        "checkpoints": checkpoint_records,
        "default_export_checkpoint": "best_checkpoint.pth",
    }
    audit_path = output_dir / "transductive_training_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audit": str(audit_path), **audit}, indent=2, sort_keys=True))


def _iter_identity_records(value, prefix: str = "input"):
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield prefix, value
            return
        for key, item in value.items():
            yield from _iter_identity_records(item, f"{prefix}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _iter_identity_records(item, f"{prefix}[{index}]")


def _load_training_audit(path: Path, args) -> dict[str, object]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("contract") != TRANSDUCTIVE_CONTRACT:
        raise ValueError("training audit is not a transductive MatPhys control")
    if audit.get("future_observations_used") is not True:
        raise ValueError("training audit does not declare future observations")
    if audit.get("released_test_outcomes_used_in_objective") is not True:
        raise ValueError("training audit does not declare test-outcome fitting")
    if audit.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("training audit claim boundary changed")
    if audit.get("case_name") != args.case:
        raise ValueError("training audit case differs from export case")
    matphys_root = Path(args.matphys_root).resolve()
    if audit.get("source_commit") != _validated_source_commit(matphys_root):
        raise ValueError("MatPhys source commit differs from training")
    if audit.get("tracked_source_clean") is not True:
        raise ValueError("training audit does not establish a clean source tree")
    expected_paths = {
        "matphys_root": matphys_root,
        "data_root": Path(args.data_root).resolve(),
        "experiments_dir": Path(args.experiments_dir).resolve(),
        "experiments_optimization_dir": Path(
            args.experiments_optimization_dir
        ).resolve(),
        "results_dir": Path(args.results_dir).resolve(),
        "sem_cache_dir": Path(args.sem_cache_dir).resolve(),
        "case_to_material": Path(args.case_to_material).resolve(),
    }
    for name, expected in expected_paths.items():
        if Path(str(audit["paths"][name])).resolve() != expected:
            raise ValueError(f"{name} differs from the training audit")
    for label, identity in _iter_identity_records(audit.get("inputs", {})):
        _validate_file_identity(identity, label)
    for label, identity in _iter_identity_records(audit.get("source_files", {})):
        _validate_file_identity(identity, label)
    for label, identity in _iter_identity_records(
        audit.get("video", {}).get("selected_frames", []), "video"
    ):
        _validate_file_identity(identity, label)
    return audit


def _validate_checkpoint(audit: Mapping[str, object], checkpoint: Path) -> None:
    checkpoints = audit.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("training audit omits checkpoints")
    matching = []
    for record in checkpoints.values():
        if not isinstance(record, Mapping):
            continue
        identity = record.get("checkpoint")
        if isinstance(identity, Mapping) and Path(str(identity["path"])).resolve() == checkpoint:
            matching.append(identity)
    if len(matching) != 1:
        raise ValueError("export checkpoint is not uniquely registered in training audit")
    _validate_file_identity(matching[0], "checkpoint")
    report = _checkpoint_finiteness_report(checkpoint)
    if not report["finite"]:
        raise ValueError("export checkpoint is non-finite")


def export(args) -> None:
    _require_acknowledgement(args.acknowledge_future_observations)
    audit_path = Path(args.training_audit).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    audit = _load_training_audit(audit_path, args)
    _validate_checkpoint(audit, checkpoint_path)

    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matphys_imports(matphys_root)
    _install_torchvision_nms_stub()
    import torch
    import train_model_video_material_simple as training
    from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = _checkpoint_namespace(checkpoint["args"], args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    matches = [sample for sample in dataset.samples if sample["case_name"] == args.case]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one dataset sample for {args.case}")
    sample = matches[0]

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

    expected_video = audit["video"]["selected_frames"]
    access = _VideoAccessLog(expected=expected_video)
    load_video = _all_frame_video_loader(
        access,
        expected_case=args.case,
        expected_data_root=data_root,
    )
    pixel_values = load_video(
        args.case,
        str(data_root),
        T=model_args.num_video_frames,
        image_size=model_args.videomae_image_size,
        device=device,
    )
    batch = {key: [value] for key, value in sample.items()}
    frame_len = int(audit["split"]["frame_len"])
    with _working_directory(matphys_root):
        runtime = training._init_runtime(args.case, frame_len, model_args)
        with torch.no_grad():
            model_out = training.forward_case(model, batch, 0, device, pixel_values)
        trajectory = _rollout_model_output(
            training,
            runtime,
            model_out,
            device,
            frame_len,
        )
    if trajectory.shape[0] != frame_len:
        raise RuntimeError("exported trajectory length differs from audited frame_len")

    trajectory_path = output_dir / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    case_root = data_root / args.case
    official = evaluate_official_phystwin_files(
        trajectory_path,
        case_root / "final_data.pkl",
        case_root / "gt_track_3d.pkl",
        case_root / "split.json",
    )
    result = {
        "schema_version": 1,
        "contract": TRANSDUCTIVE_CONTRACT,
        "claim_boundary": CLAIM_BOUNDARY,
        "future_observations_used": True,
        "released_test_outcomes_used_in_objective": True,
        "case_name": args.case,
        "training_audit": _file_identity(audit_path),
        "checkpoint": _file_identity(checkpoint_path),
        "trajectory": {
            **_file_identity(trajectory_path),
            "shape": list(trajectory.shape),
        },
        "official_evaluation": official,
    }
    result_path = output_dir / "transductive_reconstruction_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": str(result_path), **result}, indent=2, sort_keys=True))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matphys-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--experiments-optimization-dir", required=True)
    parser.add_argument("--case-to-material", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--sem-cache-dir", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--acknowledge-future-observations",
        action="store_true",
        help="Required: this control fits released future observations and outcomes.",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    _add_common_arguments(train_parser)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--videomae-model", default="MCG-NJU/videomae-base")
    train_parser.set_defaults(handler=train)

    export_parser = subparsers.add_parser("export")
    _add_common_arguments(export_parser)
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--training-audit", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.set_defaults(handler=export)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    args.handler(args)


if __name__ == "__main__":
    main()
