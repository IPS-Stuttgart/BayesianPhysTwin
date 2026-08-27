#!/usr/bin/env python3
"""Freeze, stage, predict, then separately score one public DEFT training pilot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deft_cross_branch_source import (
    ARMS,
    load_numeric_training_source,
    permitted_inputs,
    physics_shadow,
    predict_cross_branch,
    score_cross_branch,
)
from bayesian_phystwin_experiments.deft_native_restart import NativeDeft

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deft_cross_branch_source_v1.json"
BOUND_PATHS = (
    "src",
    PROTOCOL,
    "configs/sota/deft_native_source_v1.json",
    "scripts/remote/run_deft_cross_branch_source.py",
    "scripts/remote/run_deft_native_source.py",
    "scripts/remote/run_deform_dlo_source.py",
    "tests/test_deft_cross_branch_source.py",
    "tests/test_deft_native_restart.py",
    "docs/deft_cross_branch_source_v1.md",
    "docs/deft_native_source_v1.md",
)


def freeze_source(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("source pilot must be committed before freeze")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-files", *BOUND_PATHS], cwd=ROOT, text=True
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deft-cross-branch-source-receipt-v1",
            "revision": revision,
            "git_clean": True,
            "source_payload_decoded": False,
            "files": {path: file_digest(ROOT / path) for path in paths},
        },
    )
    print(
        json.dumps(
            {
                "revision": revision,
                "bound_files": len(paths),
                "receipt_sha256": file_digest(output / "source_receipt.json"),
            },
            sort_keys=True,
        )
    )


def verify_source(path: Path, expected: str) -> dict[str, Any]:
    if file_digest(path) != expected:
        raise ValueError("source receipt hash differs")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema") != "deft-cross-branch-source-receipt-v1"
        or receipt.get("git_clean") is not True
        or receipt.get("source_payload_decoded") is not False
    ):
        raise ValueError("source receipt is not an outcome-blind committed lock")
    for relative, digest in receipt["files"].items():
        path_ = ROOT / relative
        if (
            not path_.resolve(strict=True).is_relative_to(ROOT.resolve())
            or file_digest(path_) != digest
        ):
            raise ValueError(f"source changed: {relative}")
    return receipt


def verify_native_qualification(path: Path, protocol: dict[str, Any]) -> None:
    if file_digest(path) != protocol["native_qualification"]["result_sha256"]:
        raise ValueError("native qualification identity differs")
    result = json.loads(path.read_text())
    if (
        result["status"] != "pass"
        or len(result["checks"]) != 6
        or not all(value is True for value in result["checks"].values())
    ):
        raise ValueError("native qualification did not pass")
    if result["boundaries"]["trajectory_dataset_decoded"] is not False:
        raise ValueError("native qualification was not source independent")


def stage(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    verify_native_qualification(args.qualification_result, protocol)
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(
        args.run_root / "stage_attempt.json",
        {
            "source_receipt_sha256": args.source_receipt_sha256,
            "source_sha256": protocol["source"]["sha256"],
        },
    )
    trajectory = load_numeric_training_source(args.training_source)
    inputs = permitted_inputs(trajectory)
    del trajectory
    path = args.run_root / "permitted_inputs.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **inputs)
    result = {
        "schema": "deft-cross-branch-staged-input-v1",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": args.source_receipt_sha256,
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "source_file_sha256": protocol["source"]["sha256"],
        "input_sha256": file_digest(path),
        "array_sha256s": {name: array_digest(value) for name, value in inputs.items()},
        "source_payload_decoded_by_stager": True,
        "future_free_node_values_published": False,
        "evaluation_or_test_split_read": False,
        "recording_count": 1,
        "point_observation_budget": 8,
    }
    write_json_once(args.run_root / "input_manifest.json", result)
    print(
        json.dumps(
            {
                "stage": "source-inputs-sealed",
                "recording_count": 1,
                "input_manifest_sha256": file_digest(
                    args.run_root / "input_manifest.json"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def read_inputs(
    root: Path, protocol_sha256: str, source_sha256: str
) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "input_manifest.json").read_text())
    if (
        manifest["schema"] != "deft-cross-branch-staged-input-v1"
        or manifest["protocol_sha256"] != protocol_sha256
        or manifest["source_receipt_sha256"] != source_sha256
        or manifest["future_free_node_values_published"] is not False
    ):
        raise ValueError("staged input identity or boundary differs")
    path = root / "permitted_inputs.npz"
    if file_digest(path) != manifest["input_sha256"]:
        raise ValueError("staged input bytes changed")
    with np.load(path, allow_pickle=False) as data:
        inputs = {name: data[name].copy() for name in data.files}
    if {name: array_digest(value) for name, value in inputs.items()} != manifest[
        "array_sha256s"
    ]:
        raise ValueError("staged input array identity differs")
    return inputs


def predict(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    verify_native_qualification(args.qualification_result, protocol)
    inputs = read_inputs(
        args.run_root, file_digest(ROOT / PROTOCOL), args.source_receipt_sha256
    )
    write_json_once(
        args.run_root / "prediction_attempt.json",
        {
            "source_receipt_sha256": args.source_receipt_sha256,
            "source_future_scoring_opened": False,
        },
    )
    started = time.monotonic()
    try:
        import torch
        from run_deform_dlo_source import _install_dense_import_shim

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU-only pilot must explicitly disable CUDA")
        _install_dense_import_shim()
        import numba
        import pytorch3d
        import theseus

        versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "theseus": theseus.__version__,
            "pytorch3d": pytorch3d.__version__,
            "numba": numba.__version__,
        }
        if any(protocol["runtime"][name] != value for name, value in versions.items()):
            raise ValueError("pilot runtime differs from lock")
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        full = NativeDeft(args.upstream, args.checkpoint)
        shadow = NativeDeft(args.upstream, args.checkpoint)
        physics_shadow(shadow)
        arrays, controls = predict_cross_branch(full, shadow, inputs)
        path = args.run_root / "predictions.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        barrier = {
            "schema": "deft-cross-branch-prediction-barrier-v1",
            "source_revision": receipt["revision"],
            "source_receipt_sha256": args.source_receipt_sha256,
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "input_manifest_sha256": file_digest(args.run_root / "input_manifest.json"),
            "prediction_file_sha256": file_digest(path),
            "array_sha256s": {arm: array_digest(arrays[arm]) for arm in ARMS},
            "controls": controls,
            "runtime": versions,
            "ordinary_successful_recordings": 1,
            "technical_failures": 0,
            "unsealable": 0,
            "complete_arm_count": len(arrays),
            "locked_recording_count": 1,
            "source_future_scoring_opened": False,
            "future_free_node_input": False,
            "protected_data_read": False,
            "wall_seconds": time.monotonic() - started,
        }
        write_json_once(args.run_root / "prediction_barrier.json", barrier)
        print(
            json.dumps(
                {
                    "stage": "all-source-predictions-sealed",
                    "recordings": 1,
                    "arms": len(arrays),
                    "barrier_sha256": file_digest(
                        args.run_root / "prediction_barrier.json"
                    ),
                    "wall_seconds": barrier["wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:
        write_json_once(
            args.run_root / "prediction_failure.json",
            {
                "traceback": traceback.format_exc(),
                "source_future_scoring_opened": False,
                "replacement_authorized": False,
            },
        )
        raise


def score(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    barrier_path = args.run_root / "prediction_barrier.json"
    if file_digest(barrier_path) != args.prediction_barrier_sha256:
        raise ValueError("prediction barrier identity differs")
    barrier = json.loads(barrier_path.read_text())
    if (
        barrier["schema"] != "deft-cross-branch-prediction-barrier-v1"
        or barrier["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or barrier["source_receipt_sha256"] != args.source_receipt_sha256
        or barrier["complete_arm_count"] != len(ARMS)
        or barrier["ordinary_successful_recordings"] != 1
        or barrier["source_future_scoring_opened"] is not False
        or barrier["controls"]["zero_update_byte_identical"] is not True
    ):
        raise ValueError(
            "complete source prediction barrier is required before scoring"
        )
    path = args.run_root / "predictions.npz"
    if file_digest(path) != barrier["prediction_file_sha256"]:
        raise ValueError("prediction bytes changed")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if {name: array_digest(value) for name, value in arrays.items()} != barrier[
        "array_sha256s"
    ] or set(arrays) != set(ARMS):
        raise ValueError("prediction array identities differ")
    write_json_once(
        args.run_root / "score_attempt.json",
        {
            "prediction_barrier_sha256": args.prediction_barrier_sha256,
            "source_receipt_sha256": args.source_receipt_sha256,
        },
    )
    truth = load_numeric_training_source(args.training_source)[52:172]
    result = {
        "schema": "deft-cross-branch-source-pilot-result-v1",
        "source_revision": receipt["revision"],
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "prediction_barrier_sha256": args.prediction_barrier_sha256,
        **score_cross_branch(arrays, truth),
        "ordinary_successful_recordings": 1,
        "technical_failures": 0,
        "unsealable": 0,
        "protected_data_read": False,
        "public_evaluation_or_test_split_read": False,
        "outcome_publication": "local-or-private-paper-evidence-only",
    }
    write_json_once(args.run_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    freeze = commands.add_parser("freeze-source")
    freeze.add_argument("--output", type=Path, required=True)
    for mode in ("stage", "predict", "score"):
        command = commands.add_parser(mode)
        command.add_argument("--source-receipt", type=Path, required=True)
        command.add_argument("--source-receipt-sha256", required=True)
        command.add_argument("--run-root", type=Path, required=True)
        if mode in ("stage", "score"):
            command.add_argument("--training-source", type=Path, required=True)
        if mode in ("stage", "predict"):
            command.add_argument("--qualification-result", type=Path, required=True)
        if mode == "predict":
            command.add_argument("--upstream", type=Path, required=True)
            command.add_argument("--checkpoint", type=Path, required=True)
        if mode == "score":
            command.add_argument("--prediction-barrier-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "freeze-source":
        freeze_source(args.output)
        return
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    if args.run_root.resolve() != Path(protocol["registered_run_root"]):
        raise ValueError("only the registered one-recording output root is allowed")
    if tuple(protocol["arms"]) != ARMS:
        raise ValueError("arm roster differs from protocol")
    receipt = verify_source(args.source_receipt, args.source_receipt_sha256)
    {"stage": stage, "predict": predict, "score": score}[args.mode](
        args, protocol, receipt
    )


if __name__ == "__main__":
    main()
