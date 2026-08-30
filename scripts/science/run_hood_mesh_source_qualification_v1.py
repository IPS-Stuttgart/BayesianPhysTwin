#!/usr/bin/env python3
"""Run the one registered public HOOD mesh-sequence source qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin import hood_source_qualification_v1 as contracts
from bayesian_phystwin._canonical_contracts import plain_json
from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.hood_source_qualification_v1 import (
    REPLACEMENT_PLAN_SCHEMA,
    ROLLOUT_STEPS,
    HoodSourceQualificationPlanV1,
    assess_hood_source_replays_v1,
    build_hood_source_result_v1,
    consume_hood_source_attempt,
    file_sha256,
    load_hood_source_qualification_plan,
    save_hood_source_result_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--hood-root", type=Path, required=True)
    parser.add_argument("--hood-data", type=Path, required=True)
    parser.add_argument("--public-archive", type=Path, required=True)
    return parser


def _tree_digest(root: Path) -> str:
    records: list[bytes] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        records.append(f"{file_sha256(path)}  {relative}\n".encode())
    digest = hashlib.sha256()
    for record in records:
        digest.update(record)
    return digest.hexdigest()


def _git_archive_digest(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _pip_freeze_digest(python: Path) -> str:
    completed = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    canonical = "\n".join(sorted(completed.stdout.splitlines())) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_checkout(root: Path, revision: str, archive_sha256: str) -> None:
    if (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        != revision
    ):
        raise ValueError(f"source revision changed: {root}")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ):
        raise ValueError(f"source checkout must be clean: {root}")
    if _git_archive_digest(root) != archive_sha256:
        raise ValueError(f"source archive changed: {root}")


def _verify_parent_failure(
    plan: HoodSourceQualificationPlanV1,
    implementation_root: Path,
) -> None:
    if plan.value["schema"] != REPLACEMENT_PLAN_SCHEMA:
        return
    parent = plan.value["parent_failure"]
    receipt = implementation_root / parent["terminal_receipt_relative_path"]
    if file_sha256(receipt) != parent["terminal_receipt_file_sha256"]:
        raise ValueError("parent terminal receipt changed")
    for path_key, digest_key in (
        ("attempt_ledger_path", "attempt_ledger_sha256"),
        ("failure_path", "failure_sha256"),
    ):
        source = Path(parent[path_key])
        if file_sha256(source) != parent[digest_key]:
            raise ValueError(f"retained parent artifact changed: {path_key}")


def _verify_environment(
    plan: HoodSourceQualificationPlanV1,
    *,
    implementation_root: Path,
    hood_root: Path,
    hood_data: Path,
    public_archive: Path,
) -> None:
    runtime = plan.value["runtime"]
    implementation = plan.value["implementation"]
    upstream = plan.value["upstream"]
    public_source = plan.value["public_source"]
    if Path(sys.executable).resolve(strict=True) != plan.base_python_path.resolve(
        strict=True
    ):
        raise ValueError("runner is not using the registered base Python")
    if Path(contracts.__file__).resolve() != (
        implementation_root / "src/bayesian_phystwin/hood_source_qualification_v1.py"
    ):
        raise ValueError("qualification contract imported outside implementation root")
    if Path(__file__).resolve() != (
        implementation_root / "scripts/science/run_hood_mesh_source_qualification_v1.py"
    ):
        raise ValueError("runner executed outside implementation root")
    if file_sha256(plan.base_python_path) != runtime["base_python_sha256"]:
        raise ValueError("base Python executable changed")
    if _pip_freeze_digest(plan.base_python_path) != runtime["base_freeze_sha256"]:
        raise ValueError("base Python package freeze changed")
    if _tree_digest(plan.python_overlay_path) != runtime["python_overlay_tree_sha256"]:
        raise ValueError("Python overlay tree changed")
    _verify_checkout(hood_root, upstream["revision"], upstream["git_archive_sha256"])
    _verify_checkout(
        implementation_root,
        implementation["revision"],
        implementation["source_archive_sha256"],
    )
    config = hood_root / upstream["config_relative_path"]
    if file_sha256(config) != upstream["config_sha256"]:
        raise ValueError("HOOD configuration changed")
    if file_sha256(public_archive) != public_source["archive_sha256"]:
        raise ValueError("public HOOD archive changed")
    if public_archive.stat().st_size != public_source["archive_byte_count"]:
        raise ValueError("public HOOD archive size changed")
    for relative_key, digest_key in (
        ("checkpoint_relative_path", "checkpoint_sha256"),
        ("mesh_sequence_relative_path", "mesh_sequence_sha256"),
        ("garment_template_relative_path", "garment_template_sha256"),
        ("garment_obj_relative_path", "garment_obj_sha256"),
    ):
        source = hood_data.parent / public_source[relative_key]
        if file_sha256(source) != public_source[digest_key]:
            raise ValueError(f"public source changed: {relative_key}")
    for relative, digest in plan.implementation_source_files.items():
        if file_sha256(implementation_root / relative) != digest:
            raise ValueError(f"implementation source changed: {relative}")
    _verify_parent_failure(plan, implementation_root)


def _verify_imported_versions(plan: HoodSourceQualificationPlanV1) -> None:
    import pytorch3d
    import torch
    import torch_geometric

    actual = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_geometric_version": torch_geometric.__version__,
        "pytorch3d_version": pytorch3d.__version__,
    }
    for key, version in actual.items():
        if version != plan.value["runtime"][key]:
            raise ValueError(f"imported runtime changed: {key}")


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _apply_registered_dataset_correction(
    plan: HoodSourceQualificationPlanV1,
    dataset_config: Any,
) -> None:
    registered = dataset_config.smpl_model
    if registered != "smpl/SMPL_FEMALE.pkl":
        raise ValueError("HOOD example smpl_model field changed")
    if plan.value["schema"] == REPLACEMENT_PLAN_SCHEMA:
        if plan.value["execution"]["smpl_model_override"] is not None:
            raise ValueError("replacement smpl_model override changed")
        dataset_config.smpl_model = None


def _run_replay(
    *,
    plan: HoodSourceQualificationPlanV1,
    seed: int,
    hood_root: Path,
    hood_data: Path,
) -> dict[str, Any]:
    os.environ["HOOD_PROJECT"] = str(hood_root)
    os.environ["HOOD_DATA"] = str(hood_data)
    import torch
    from utils.arguments import create_modules, load_params

    _seed_everything(torch, seed)
    modules, config = load_params("aux/from_any_pose")
    dataset_config = config.dataloader.dataset.from_any_pose
    _apply_registered_dataset_correction(plan, dataset_config)
    if config.device != "cuda:0":
        raise ValueError("HOOD configuration device changed")
    if dataset_config.pose_sequence_type != "mesh":
        raise ValueError("HOOD source must use the public mesh sequence")
    if dataset_config.pose_sequence_path != "fromanypose/mesh_sequence.pkl":
        raise ValueError("HOOD mesh sequence path changed")
    if dataset_config.garment_template_path != "fromanypose/tshirt.pkl":
        raise ValueError("HOOD garment template path changed")
    dataloader_module, _, runner, _ = create_modules(
        modules,
        config,
        create_aux_modules=False,
    )
    checkpoint_path = hood_data / "trained_models/postcvpr.pth"
    state = torch.load(checkpoint_path, map_location="cpu")
    if set(state) != {"training_module"}:
        raise ValueError("HOOD checkpoint fields changed")
    runner.load_state_dict(state["training_module"])
    runner.to(config.device)
    runner.eval()
    dataloader = dataloader_module.create_dataloader(is_eval=True)
    sample = next(iter(dataloader))
    with torch.no_grad():
        output = runner.valid_rollout(
            sample,
            n_steps=ROLLOUT_STEPS,
            bare=True,
            record_time=True,
        )
    return dict(output)


def _write_failure(output_root: Path, plan_id: str, error: BaseException) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "bayesian-phystwin.hood-mesh-source-qualification-failure",
        "schema_version": 1,
        "plan_id": plan_id,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "retry_authorized": False,
        "information_boundary": {
            "fourddress_payload_read": False,
            "physical_outcomes_read": False,
            "held_v8_read": False,
            "dlo4_or_dlo5_read": False,
        },
    }
    write_atomic_json(payload, output_root / "failure.json", overwrite=False)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_hood_source_qualification_plan(args.plan)
    output_root = plan.output_root
    if output_root.exists():
        raise ValueError("registered output root already exists")
    _verify_environment(
        plan,
        implementation_root=args.implementation_root.resolve(strict=True),
        hood_root=args.hood_root.resolve(strict=True),
        hood_data=args.hood_data.resolve(strict=True),
        public_archive=args.public_archive.resolve(strict=True),
    )
    consume_hood_source_attempt(plan)
    start = time.monotonic()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(plan.cuda_visible_device)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(plan.python_overlay_path))
        sys.path.insert(1, str(args.hood_root.resolve(strict=True)))
        _verify_imported_versions(plan)
        replays = [
            _run_replay(
                plan=plan,
                seed=plan.value["execution"]["random_seed"],
                hood_root=args.hood_root.resolve(strict=True),
                hood_data=args.hood_data.resolve(strict=True),
            )
            for _ in range(plan.value["execution"]["replay_count"])
        ]
        assessment = assess_hood_source_replays_v1(
            [value["pred"] for value in replays],
            [value["obstacle"] for value in replays],
            [value["cloth_faces"] for value in replays],
            [value["obstacle_faces"] for value in replays],
        )
        output_root.mkdir(parents=True)
        archive = output_root / "hood_source_replays.npz"
        np.savez_compressed(
            archive,
            prediction_0=np.asarray(replays[0]["pred"]),
            prediction_1=np.asarray(replays[1]["pred"]),
            obstacle_0=np.asarray(replays[0]["obstacle"]),
            obstacle_1=np.asarray(replays[1]["obstacle"]),
            cloth_faces_0=np.asarray(replays[0]["cloth_faces"]),
            cloth_faces_1=np.asarray(replays[1]["cloth_faces"]),
            obstacle_faces_0=np.asarray(replays[0]["obstacle_faces"]),
            obstacle_faces_1=np.asarray(replays[1]["obstacle_faces"]),
        )
        result = build_hood_source_result_v1(
            plan=plan,
            assessment=assessment,
            replay_archive_sha256=file_sha256(archive),
            elapsed_seconds=time.monotonic() - start,
        )
        save_hood_source_result_v1(
            result,
            output_root / "source_qualification_result.json",
        )
        print(json.dumps(plain_json(result), sort_keys=True))
        return 0 if assessment.passed else 2
    except BaseException as error:
        _write_failure(output_root, plan.plan_id, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
