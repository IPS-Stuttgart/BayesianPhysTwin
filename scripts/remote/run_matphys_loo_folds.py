#!/usr/bin/env python3
"""Train and export every sealed object-disjoint MatPhys fold."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from bayesian_phystwin.matphys_causal_bridge import (
    sha256_file,
    validate_source_supervised_training_audit,
)


def _validated_identity(identity: object, label: str) -> Path:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} must be a file identity")
    path = Path(str(identity.get("path", ""))).resolve()
    expected = str(identity.get("sha256", ""))
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"{label} bytes changed")
    return path


def _run(command: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    if dry_run:
        print(rendered)
        return
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {rendered}\n")
        stream.flush()
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace_manifest")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--matphys-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--experiments-optimization-dir", required=True)
    parser.add_argument("--initialization-checkpoint", required=True)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--folds", help="Optional comma-separated fold indices.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be positive")

    manifest_path = Path(args.workspace_manifest).resolve()
    workspace = json.loads(manifest_path.read_text(encoding="utf-8"))
    if workspace.get("contract") != "matphys-object-disjoint-loo-workspace-v1":
        raise ValueError("unsupported MatPhys LOO workspace")
    if workspace.get("future_opened") is not False:
        raise ValueError("MatPhys LOO workspace has already opened future metrics")
    protocol_path = _validated_identity(workspace.get("protocol"), "protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    training = protocol["source_training"]
    graph = training["graph_parts"]
    initialization = Path(args.initialization_checkpoint).resolve()
    expected_initialization = str(training["initialization_checkpoint_sha256"])
    if sha256_file(initialization) != expected_initialization:
        raise ValueError("MatPhys initialization checkpoint bytes changed")
    runner = Path(args.runner).resolve()
    python = Path(args.python).resolve()
    if not runner.is_file() or not python.is_file():
        raise FileNotFoundError("runner or Python executable is missing")
    selected_folds = (
        {int(value) for value in args.folds.split(",") if value.strip()}
        if args.folds
        else None
    )
    env = dict(os.environ)
    repository_root = runner.parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository_root / "src"), env.get("PYTHONPATH", "")))
    )
    env["NCCL_P2P_DISABLE"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"

    completed = []
    for fold in workspace["folds"]:
        index = int(fold["fold_index"])
        if selected_folds is not None and index not in selected_folds:
            continue
        registration = _validated_identity(
            fold.get("registration"), f"fold {index} registration"
        )
        source_summary = _validated_identity(
            fold.get("source_proxy"), f"fold {index} source proxy"
        )
        target_summary = _validated_identity(
            fold.get("target_proxy"), f"fold {index} target proxy"
        )
        fold_root = Path(str(fold["root"])).resolve()
        train_root = fold_root / "training"
        export_root = fold_root / "matphys_export"
        checkpoint = train_root / "last_checkpoint.pth"
        audit = train_root / "source_supervised_training_audit.json"
        source_cases = ",".join(str(case) for case in fold["source_cases"])
        target_cases = ",".join(str(case) for case in fold["target_cases"])
        training_complete = False
        if args.resume and checkpoint.is_file() and audit.is_file():
            validate_source_supervised_training_audit(audit, checkpoint)
            training_complete = True
        train_command = [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            str(runner),
            "train",
            "--matphys-root",
            str(Path(args.matphys_root).resolve()),
            "--data-root",
            str(Path(args.data_root).resolve()),
            "--experiments-optimization-dir",
            str(Path(args.experiments_optimization_dir).resolve()),
            "--proxy-root",
            str(source_summary.parent),
            "--cases",
            source_cases,
            "--device",
            "cuda:0",
            "--output-dir",
            str(train_root),
            "--epochs",
            str(int(training["epochs"])),
            "--eval-every",
            str(int(training["epochs"])),
            "--teacher-experiments-dir",
            str(Path(args.experiments_dir).resolve()),
            "--teacher-residual-log-scale",
            str(float(training["teacher_residual_log_scale"])),
            "--fit-fraction",
            "0.75",
            "--graph-parts",
            "--part-count",
            str(int(graph["part_count"])),
            "--dino-model",
            str(graph["dino_model"]),
            "--dino-keyframes",
            str(int(graph["dino_keyframes"])),
            "--semantic-edge-weight",
            str(float(graph["semantic_edge_weight"])),
            "--part-feature-scale",
            str(float(graph["part_feature_scale"])),
            "--compact-unused-edge-semantics",
            "--training-contract",
            str(training["contract"]),
            "--target-cases",
            target_cases,
            "--split-registration",
            str(registration),
            "--implementation-amendment",
            str(protocol_path),
            "--target-fit-fraction",
            "0.75",
            "--learning-rate",
            str(float(training["learning_rate"])),
            "--grad-clip",
            str(float(training["gradient_clip"])),
            "--teacher-proximity-weight",
            str(float(training["teacher_proximity_weight"])),
            "--finite-optimizer-guard",
            "--initialization-checkpoint",
            str(initialization),
            "--initialization-sha256",
            expected_initialization,
        ]
        if not training_complete:
            _run(train_command, fold_root / "train.log", env, args.dry_run)
        if args.dry_run:
            training_complete = True
        elif not checkpoint.is_file() or not audit.is_file():
            raise RuntimeError(f"fold {index} did not produce a training audit")

        export_manifest = export_root / "external_backbone_manifest.json"
        if not (args.resume and export_manifest.is_file()):
            export_command = [
                str(python),
                str(runner),
                "export",
                "--matphys-root",
                str(Path(args.matphys_root).resolve()),
                "--data-root",
                str(Path(args.data_root).resolve()),
                "--experiments-optimization-dir",
                str(Path(args.experiments_optimization_dir).resolve()),
                "--proxy-root",
                str(target_summary.parent),
                "--cases",
                target_cases,
                "--device",
                "cuda:0",
                "--checkpoint",
                str(checkpoint),
                "--training-audit",
                str(audit),
                "--output-dir",
                str(export_root),
                "--target-teacher-experiments-dir",
                str(Path(args.experiments_dir).resolve()),
            ]
            _run(export_command, fold_root / "export.log", env, args.dry_run)
        if not args.dry_run and not export_manifest.is_file():
            raise RuntimeError(f"fold {index} did not produce an export manifest")
        completed.append(
            {
                "fold_index": index,
                "held_out_object": fold["held_out_object"],
                "checkpoint": (
                    {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
                    if checkpoint.is_file()
                    else None
                ),
                "training_audit": (
                    {"path": str(audit), "sha256": sha256_file(audit)}
                    if audit.is_file()
                    else None
                ),
                "export_manifest": (
                    {
                        "path": str(export_manifest),
                        "sha256": sha256_file(export_manifest),
                    }
                    if export_manifest.is_file()
                    else None
                ),
            }
        )

    execution = {
        "schema_version": 1,
        "contract": "matphys-object-disjoint-loo-execution-v1",
        "workspace": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "future_metrics_opened": False,
        "dry_run": args.dry_run,
        "folds": completed,
    }
    if not args.dry_run:
        destination = manifest_path.parent / "loo_execution_manifest.json"
        destination.write_text(
            json.dumps(execution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        execution["execution_manifest"] = str(destination)
    print(json.dumps(execution, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
