#!/usr/bin/env python3
"""Source-code/reference-geometry-only native DEFT qualification on CPU."""

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
from bayesian_phystwin_experiments.deft_native_restart import (
    PARENT_CLAMPS,
    NativeDeft,
    update_deft_state,
    verify_upstream,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deft_native_source_v1.json"
BOUND_PATHS = (
    "src",
    PROTOCOL,
    "scripts/remote/run_deft_native_source.py",
    "scripts/remote/run_deform_dlo_source.py",
    "tests/test_deft_native_restart.py",
    "docs/deft_native_source_v1.md",
)


def freeze_source(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit source-only changes before freezing qualification")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-files", *BOUND_PATHS], cwd=ROOT, text=True
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "deft-native-source-receipt-v1",
        "revision": revision,
        "git_clean": True,
        "files": {path: file_digest(ROOT / path) for path in paths},
        "dataset_read": False,
        "synthetic_reference_only": True,
    }
    write_json_once(output / "source_receipt.json", receipt)
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


def verify_source(path: Path, digest: str) -> dict[str, Any]:
    if file_digest(path) != digest:
        raise ValueError("source receipt identity differs")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema") != "deft-native-source-receipt-v1"
        or receipt.get("git_clean") is not True
        or receipt.get("dataset_read") is not False
    ):
        raise ValueError("receipt does not authorize source-only qualification")
    for relative, expected in receipt["files"].items():
        source = ROOT / relative
        if (
            not source.resolve(strict=True).is_relative_to(ROOT.resolve())
            or file_digest(source) != expected
        ):
            raise ValueError(f"frozen implementation changed: {relative}")
    return receipt


def run_qualification(args: argparse.Namespace) -> None:
    receipt = verify_source(args.source_receipt, args.source_receipt_sha256)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    if any(protocol["boundaries"].values()):
        raise ValueError(
            "qualification must not open dataset or advancement permissions"
        )
    verify_upstream(args.upstream)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("qualification must explicitly disable CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema": "deft-native-source-qualification-result-v1",
        "status": "technical_failure",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": args.source_receipt_sha256,
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "boundaries": {
            "synthetic_reference_only": True,
            "trajectory_dataset_decoded": False,
            "accuracy_outcomes_read": False,
            "public_evaluation_split_read": False,
            "protected_targets_read": False,
            "held_v8_read": False,
            "existing_deform_modules_modified": False,
            "empirical_advancement_authorized": False,
        },
    }
    write_json_once(
        args.output / "attempt.json",
        {
            "source_receipt_sha256": args.source_receipt_sha256,
            "protocol_sha256": result["protocol_sha256"],
            "synthetic_only": True,
        },
    )
    started = time.monotonic()
    try:
        import torch
        from run_deform_dlo_source import _install_dense_import_shim

        result["sparse_import"] = _install_dense_import_shim()
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
        if any(protocol["runtime"][key] != value for key, value in versions.items()):
            raise ValueError("runtime differs from the qualification lock")
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1)
        np.random.seed(1)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = NativeDeft(args.upstream, args.checkpoint)
        result["runtime"] = {
            **versions,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
            "pytorch3d_rotation_conversions_sha256": file_digest(
                Path(pytorch3d.transforms.rotation_conversions.__file__)
            ),
        }
        initial = np.repeat(model.reference_geometry[None], 2, axis=0)
        settings = protocol["synthetic_qualification"]
        count, split = settings["timesteps"], settings["pause_after_completed_steps"]
        shift = (
            np.linspace(1 / count, 1.0, count)[:, None, None]
            * np.array(settings["prescribed_clamp_translation_m"])[None, None]
        )
        actions = initial[1, 0, PARENT_CLAMPS][None] + shift
        print("native monolithic qualification", flush=True)
        native_x, native_v = model.native_rollout(initial, actions)
        full_x, full_v, full_state = model.rollout(initial, actions)
        prefix_x, prefix_v, prefix_state = model.rollout(initial, actions[:split])
        state_before = prefix_state.digests()
        suffix_x, suffix_v, suffix_state = model.rollout(initial, actions, prefix_state)
        zero_state = update_deft_state(
            prefix_state, np.zeros((3, 13, 3)), np.zeros((3, 13, 3))
        )
        zero_x, zero_v, zero_final = model.rollout(initial, actions, zero_state)
        combined_x = np.concatenate([prefix_x, suffix_x])
        combined_v = np.concatenate([prefix_v, suffix_v])
        clamp_error = float(np.max(np.abs(native_x[:, 0][:, PARENT_CLAMPS] - actions)))
        checks = {
            "finite_native_positions_and_velocities": bool(
                np.isfinite(native_x).all() and np.isfinite(native_v).all()
            ),
            "resumable_vs_native_bitwise_positions_and_velocities": bool(
                np.array_equal(native_x, full_x) and np.array_equal(native_v, full_v)
            ),
            "segmented_vs_monolithic_bitwise_positions_and_velocities": bool(
                np.array_equal(full_x, combined_x)
                and np.array_equal(full_v, combined_v)
                and full_state.digests() == suffix_state.digests()
            ),
            "zero_update_bitwise_positions_velocities_and_final_internal_state": bool(
                np.array_equal(suffix_x, zero_x)
                and np.array_equal(suffix_v, zero_v)
                and suffix_state.digests() == zero_final.digests()
            ),
            "input_state_not_mutated_by_continuation": prefix_state.digests()
            == state_before,
            "native_parent_clamp_error_at_most_1e-6_m": clamp_error <= 1e-6,
        }
        if set(checks) != set(settings["required_checks"]):
            raise ValueError("qualification checks differ from source lock")
        result.update(
            status="pass" if all(checks.values()) else "qualification_failed",
            checks=checks,
            clamp_max_error_m=clamp_error,
            final_state_sha256s=full_state.digests(),
            model_id=model.model_id,
        )
        with (args.output / "synthetic_predictions.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                native_x=native_x,
                native_v=native_v,
                resumable_x=full_x,
                resumable_v=full_v,
                segmented_x=combined_x,
                segmented_v=combined_v,
                zero_update_x=zero_x,
                zero_update_v=zero_v,
                initial=initial,
                clamps=actions,
            )
        result["prediction_file_sha256"] = file_digest(
            args.output / "synthetic_predictions.npz"
        )
        result["native_position_array_sha256"] = array_digest(native_x)
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result["wall_seconds"] = time.monotonic() - started
    write_json_once(args.output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    freeze = commands.add_parser("freeze-source")
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("qualify")
    run.add_argument("--source-receipt", type=Path, required=True)
    run.add_argument("--source-receipt-sha256", required=True)
    run.add_argument("--upstream", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze-source":
        freeze_source(args.output)
    else:
        run_qualification(args)


if __name__ == "__main__":
    main()
