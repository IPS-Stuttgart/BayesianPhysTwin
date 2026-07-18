#!/usr/bin/env python3
"""Run one sequential shard of causal per-case MatPhys fits and exports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from bayesian_phystwin.matphys_causal_bridge import (
    validate_causal_training_audit,
)
from bayesian_phystwin.phystwin_external_backbone import (
    validate_external_backbone_manifest,
)


def _cases(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("cases must be a nonempty unique list")
    return result


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--matphys-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiments-optimization-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cases", type=_cases, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--teacher-experiments-dir")
    parser.add_argument("--teacher-residual-log-scale", type=float)
    parser.add_argument("--fit-fraction", type=float, default=1.0)
    parser.add_argument("--graph-parts", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_every < 1:
        parser.error("epochs and eval-every must be positive")

    wrapper = Path(args.wrapper).resolve()
    run_root = Path(args.run_root).resolve()
    data_root = Path(args.data_root).resolve()
    records: list[dict[str, object]] = []
    for index, case in enumerate(args.cases, start=1):
        started = time.monotonic()
        case_root = run_root / case
        checkpoint = case_root / "checkpoint" / "last_checkpoint.pth"
        audit = case_root / "checkpoint" / "causal_training_audit.json"
        export_root = case_root / "export_last"
        manifest = export_root / "external_backbone_manifest.json"
        print(
            f"[{index}/{len(args.cases)}] {case}: checking {args.device}",
            flush=True,
        )
        trained = False
        if checkpoint.is_file() and audit.is_file():
            validate_causal_training_audit(audit, checkpoint)
        else:
            train_command = [
                sys.executable,
                str(wrapper),
                "train",
                "--matphys-root",
                args.matphys_root,
                "--data-root",
                str(data_root),
                "--experiments-optimization-dir",
                args.experiments_optimization_dir,
                "--proxy-root",
                str(case_root / "proxy"),
                "--cases",
                case,
                "--device",
                args.device,
                "--output-dir",
                str(case_root / "checkpoint"),
                "--epochs",
                str(args.epochs),
                "--eval-every",
                str(args.eval_every),
                "--fit-fraction",
                str(args.fit_fraction),
            ]
            if args.teacher_experiments_dir is not None:
                train_command.extend(
                    ("--teacher-experiments-dir", args.teacher_experiments_dir)
                )
            if args.teacher_residual_log_scale is not None:
                train_command.extend(
                    (
                        "--teacher-residual-log-scale",
                        str(args.teacher_residual_log_scale),
                    )
                )
            if args.graph_parts:
                train_command.append("--graph-parts")
            _run(train_command, case_root / "train.log")
            validate_causal_training_audit(audit, checkpoint)
            trained = True

        exported = False
        if manifest.is_file():
            validate_external_backbone_manifest(
                data_root, manifest, require_full_cohort=False
            )
        else:
            _run(
                [
                    sys.executable,
                    str(wrapper),
                    "export",
                    "--matphys-root",
                    args.matphys_root,
                    "--data-root",
                    str(data_root),
                    "--experiments-optimization-dir",
                    args.experiments_optimization_dir,
                    "--proxy-root",
                    str(case_root / "proxy"),
                    "--cases",
                    case,
                    "--device",
                    args.device,
                    "--checkpoint",
                    str(checkpoint),
                    "--training-audit",
                    str(audit),
                    "--output-dir",
                    str(export_root),
                ],
                case_root / "export.log",
            )
            validate_external_backbone_manifest(
                data_root, manifest, require_full_cohort=False
            )
            exported = True
        record = {
            "case": case,
            "device": args.device,
            "trained": trained,
            "exported": exported,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint),
            "audit": str(audit),
            "manifest": str(manifest),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "device": args.device,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "teacher_experiments_dir": args.teacher_experiments_dir,
        "teacher_residual_log_scale": args.teacher_residual_log_scale,
        "fit_fraction": args.fit_fraction,
        "graph_parts": args.graph_parts,
        "cases": records,
    }
    destination = run_root / f"shard_{args.device.replace(':', '_')}.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"completed shard: {destination}", flush=True)


if __name__ == "__main__":
    main()
