#!/usr/bin/env python3
"""Commit/freeze, stage, seal all three source predictions, then separately score."""

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
from run_deft_cross_branch_source import verify_native_qualification

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deft_cross_branch_source import physics_shadow
from bayesian_phystwin_experiments.deft_native_restart import NativeDeft
from bayesian_phystwin_experiments.deft_topology_observer import (
    ARMS,
    CASE_IDS,
    COMPARATORS,
    PRIMARY,
    load_training_case,
    permitted_inputs,
    predict_topology,
    score_case,
    score_study,
    synthetic_qualification,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deft_topology_observer_source_v1.json"
SCHEMA = "deft-topology-observer-source-v1"
BOUND = (
    "src",
    PROTOCOL,
    "configs/sota/deft_native_source_v1.json",
    "scripts/remote/run_deft_topology_observer.py",
    "scripts/remote/run_deft_cross_branch_source.py",
    "scripts/remote/run_deform_dlo_source.py",
    "scripts/verify_deft_topology_observer.py",
    "scripts/verify_deft_cross_branch_source.py",
    "tests/test_deft_topology_observer.py",
    "tests/test_deft_topology_runner.py",
    "tests/test_deft_native_restart.py",
    "docs/deft_topology_observer_source_v1.md",
)


def load_protocol() -> dict[str, Any]:
    value = json.loads((ROOT / PROTOCOL).read_text())
    if (
        value["schema"] != SCHEMA
        or tuple(value["arms"]) != ARMS
        or value["primary_arm"] != PRIMARY
        or tuple(x["id"] for x in value["source_cases"]) != CASE_IDS
        or tuple(value["gate"]["comparators"]) != COMPARATORS
        or value["inputs"]["observation_raw_frames"] != [43, 51]
        or value["inputs"]["state_update_gain"] != 1.0
        or value["inputs"]["forecast_raw_frames_half_open"] != [52, 172]
        or value["boundaries"]["future_free_node_input"] is not False
    ):
        raise ValueError("frozen method, roster, or information boundary changed")
    if value["gate"] != {
        "comparators": list(COMPARATORS),
        "each_child_aggregate_rmse_gain_fraction": 0.05,
        "each_child_aggregate_l1_and_late_nonincreasing": True,
        "minimum_recording_joint_rmse_l1_wins_per_comparator": 2,
        "maximum_any_recording_rmse_ratio_to_native": 1.10,
        "all_24_forecasts_sealed_before_any_score": True,
        "all_three_ordinary_successes_required": True,
        "secondary_arm_cannot_rescue_primary": True,
        "automatic_next_evaluation_authorization": False,
    }:
        raise ValueError("frozen scientific decision rule changed")
    for spec in value["source_cases"]:
        if Path(spec["filename"]).name != spec["filename"] or not spec[
            "filename"
        ].endswith(".pkl"):
            raise ValueError("source filename must be one declared training basename")
        for field, length in (("git_blob", 40), ("sha256", 64)):
            if len(spec[field]) != length or any(
                c not in "0123456789abcdef" for c in spec[field]
            ):
                raise ValueError("source content identity is not a canonical digest")
    return value


def freeze_source(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit the complete implementation before source freeze")
    load_protocol()
    qualification = synthetic_qualification()
    if not qualification["passed"]:
        raise ValueError("synthetic topology qualification failed")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-files", *BOUND], cwd=ROOT, text=True
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / "synthetic_qualification.json", qualification)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": SCHEMA + "-receipt",
            "revision": revision,
            "git_clean": True,
            "new_source_trajectory_decoded": False,
            "synthetic_qualification_sha256": file_digest(
                output / "synthetic_qualification.json"
            ),
            "files": {path: file_digest(ROOT / path) for path in paths},
        },
    )
    print(
        json.dumps(
            {
                "revision": revision,
                "files": len(paths),
                "source_receipt_sha256": file_digest(output / "source_receipt.json"),
            }
        ),
        flush=True,
    )


def verify_source(path: Path, expected: str) -> dict[str, Any]:
    if file_digest(path) != expected:
        raise ValueError("source receipt digest changed")
    value = json.loads(path.read_text())
    if (
        value["schema"] != SCHEMA + "-receipt"
        or value["git_clean"] is not True
        or value["new_source_trajectory_decoded"] is not False
    ):
        raise ValueError("receipt is not a clean pre-decode source lock")
    for name, digest in value["files"].items():
        relative = Path(name)
        candidate = ROOT / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or candidate.is_symlink()
            or not candidate.resolve(strict=True).is_relative_to(ROOT.resolve())
            or file_digest(candidate) != digest
        ):
            raise ValueError(f"committed source changed: {name}")
    if PROTOCOL not in value["files"] or not synthetic_qualification()["passed"]:
        raise ValueError("protocol or synthetic qualification is not bound")
    return value


def _input_manifest(root: Path, source_digest: str) -> dict[str, Any]:
    record = json.loads((root / "input_manifest.json").read_text())
    if (
        record["schema"] != SCHEMA + "-input"
        or record["source_receipt_sha256"] != source_digest
        or record["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or record["future_free_node_values_published"] is not False
    ):
        raise ValueError("staged source inputs do not match the lock")
    if file_digest(root / "permitted_inputs.npz") != record["input_sha256"]:
        raise ValueError("staged input archive changed")
    return record


def stage(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    verify_native_qualification(args.qualification_result, protocol)
    args.run_root.mkdir(parents=True, exist_ok=False)
    write_json_once(
        args.run_root / "stage_attempt.json",
        {
            "source_receipt_sha256": args.source_receipt_sha256,
            "future_scoring_opened": False,
        },
    )
    outcomes = {}
    for spec in protocol["source_cases"]:
        case_root = args.run_root / spec["id"]
        case_root.mkdir()
        try:
            trajectory = load_training_case(args.training_root / spec["filename"], spec)
            inputs = permitted_inputs(trajectory)
            del trajectory
            with (case_root / "permitted_inputs.npz").open("xb") as stream:
                np.savez_compressed(stream, **inputs)
            write_json_once(
                case_root / "input_manifest.json",
                {
                    "schema": SCHEMA + "-input",
                    "case_id": spec["id"],
                    "source_revision": receipt["revision"],
                    "source_receipt_sha256": args.source_receipt_sha256,
                    "protocol_sha256": file_digest(ROOT / PROTOCOL),
                    "source_file_sha256": spec["sha256"],
                    "input_sha256": file_digest(case_root / "permitted_inputs.npz"),
                    "array_sha256s": {
                        name: array_digest(value) for name, value in inputs.items()
                    },
                    "source_container_decoded_by_stager": True,
                    "future_free_node_values_published": False,
                    "point_observation_budget_per_corrected_arm": 8,
                },
            )
            outcomes[spec["id"]] = {
                "status": "staged",
                "manifest_sha256": file_digest(case_root / "input_manifest.json"),
            }
        except Exception:
            write_json_once(
                case_root / "input_failure.json",
                {"traceback": traceback.format_exc(), "replacement_authorized": False},
            )
            outcomes[spec["id"]] = {
                "status": "unsealable",
                "failure_sha256": file_digest(case_root / "input_failure.json"),
            }
    write_json_once(
        args.run_root / "input_barrier.json",
        {
            "schema": SCHEMA + "-input-barrier",
            "source_receipt_sha256": args.source_receipt_sha256,
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "cases": outcomes,
            "source_future_scoring_opened": False,
            "protected_data_read": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "inputs-sealed",
                "staged": sum(v["status"] == "staged" for v in outcomes.values()),
                "locked_recordings": 3,
                "input_barrier_sha256": file_digest(
                    args.run_root / "input_barrier.json"
                ),
            }
        ),
        flush=True,
    )
    if any(value["status"] != "staged" for value in outcomes.values()):
        raise ValueError(
            "retained source input failure; no replacement or prediction authorization"
        )


def validate_input_barrier(
    root: Path, expected: str, source_digest: str
) -> dict[str, Any]:
    if file_digest(root / "input_barrier.json") != expected:
        raise ValueError("input barrier digest changed")
    barrier = json.loads((root / "input_barrier.json").read_text())
    if (
        barrier["schema"] != SCHEMA + "-input-barrier"
        or barrier["source_receipt_sha256"] != source_digest
        or barrier["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or set(barrier["cases"]) != set(CASE_IDS)
        or barrier["source_future_scoring_opened"] is not False
    ):
        raise ValueError("input barrier is not the complete unopened source roster")
    for case in CASE_IDS:
        record = barrier["cases"][case]
        if record["status"] != "staged" or record["manifest_sha256"] != file_digest(
            root / case / "input_manifest.json"
        ):
            raise ValueError("an input failure or changed manifest blocks the study")
        manifest = _input_manifest(root / case, source_digest)
        if manifest["case_id"] != case:
            raise ValueError("source case identity changed")
    return barrier


def predict(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    verify_native_qualification(args.qualification_result, protocol)
    validate_input_barrier(
        args.run_root, args.input_barrier_sha256, args.source_receipt_sha256
    )
    write_json_once(
        args.run_root / "prediction_attempt.json",
        {
            "source_receipt_sha256": args.source_receipt_sha256,
            "input_barrier_sha256": args.input_barrier_sha256,
            "source_future_scoring_opened": False,
        },
    )
    started = time.monotonic()
    try:
        import torch
        from run_deform_dlo_source import _install_dense_import_shim

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("the frozen CPU pilot requires CUDA_VISIBLE_DEVICES empty")
        _install_dense_import_shim()
        import numba
        import pytorch3d
        import theseus

        runtime = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "theseus": theseus.__version__,
            "numba": numba.__version__,
            "pytorch3d": pytorch3d.__version__,
        }
        if any(protocol["runtime"][key] != value for key, value in runtime.items()):
            raise ValueError("native compatibility runtime changed")
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        outcomes = {}
        for case in CASE_IDS:
            case_root = args.run_root / case
            try:
                manifest = _input_manifest(case_root, args.source_receipt_sha256)
                with np.load(
                    case_root / "permitted_inputs.npz", allow_pickle=False
                ) as archive:
                    inputs = {key: archive[key].copy() for key in archive.files}
                if {
                    name: array_digest(value) for name, value in inputs.items()
                } != manifest["array_sha256s"]:
                    raise ValueError("permitted input arrays changed")
                full = NativeDeft(args.upstream, args.checkpoint)
                shadow = NativeDeft(args.upstream, args.checkpoint)
                physics_shadow(shadow)
                arrays, controls = predict_topology(full, shadow, inputs)
                with (case_root / "predictions.npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                seal = {
                    "schema": SCHEMA + "-case-seal",
                    "case_id": case,
                    "source_receipt_sha256": args.source_receipt_sha256,
                    "protocol_sha256": file_digest(ROOT / PROTOCOL),
                    "input_manifest_sha256": file_digest(
                        case_root / "input_manifest.json"
                    ),
                    "prediction_sha256": file_digest(case_root / "predictions.npz"),
                    "array_sha256s": {arm: array_digest(arrays[arm]) for arm in ARMS},
                    "controls": controls,
                    "source_future_scoring_opened": False,
                }
                write_json_once(case_root / "prediction_seal.json", seal)
                outcomes[case] = {
                    "status": "ordinary-success",
                    "seal_sha256": file_digest(case_root / "prediction_seal.json"),
                }
                print(
                    json.dumps(
                        {
                            "stage": "prediction-sealed",
                            "completed_recordings": len(outcomes),
                            "arms": len(arrays),
                        }
                    ),
                    flush=True,
                )
            except Exception:
                write_json_once(
                    case_root / "prediction_failure.json",
                    {
                        "traceback": traceback.format_exc(),
                        "replacement_authorized": False,
                        "future_scoring_opened": False,
                    },
                )
                outcomes[case] = {
                    "status": "technical-failure",
                    "failure_sha256": file_digest(
                        case_root / "prediction_failure.json"
                    ),
                }
        write_json_once(
            args.run_root / "prediction_barrier.json",
            {
                "schema": SCHEMA + "-prediction-barrier",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": args.source_receipt_sha256,
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "input_barrier_sha256": args.input_barrier_sha256,
                "cases": outcomes,
                "ordinary_successful_recordings": sum(
                    x["status"] == "ordinary-success" for x in outcomes.values()
                ),
                "technical_failures": sum(
                    x["status"] == "technical-failure" for x in outcomes.values()
                ),
                "locked_recording_count": 3,
                "source_future_scoring_opened": False,
                "protected_data_read": False,
                "runtime": runtime,
                "wall_seconds": time.monotonic() - started,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "prediction-barrier",
                    "prediction_barrier_sha256": file_digest(
                        args.run_root / "prediction_barrier.json"
                    ),
                    "ordinary_success": sum(
                        x["status"] == "ordinary-success" for x in outcomes.values()
                    ),
                    "locked_recordings": 3,
                }
            ),
            flush=True,
        )
        if any(x["status"] != "ordinary-success" for x in outcomes.values()):
            raise ValueError(
                "technical failure retained; incomplete study cannot advance"
            )
    except Exception:
        write_json_once(
            args.run_root / "prediction_attempt_failure.json",
            {
                "traceback": traceback.format_exc(),
                "future_scoring_opened": False,
                "retry_authorized": False,
            },
        )
        raise


def read_predictions(
    root: Path, expected: str, source_digest: str
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    if file_digest(root / "prediction_barrier.json") != expected:
        raise ValueError("prediction barrier digest changed")
    barrier = json.loads((root / "prediction_barrier.json").read_text())
    if (
        barrier["schema"] != SCHEMA + "-prediction-barrier"
        or barrier["source_receipt_sha256"] != source_digest
        or barrier["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or set(barrier["cases"]) != set(CASE_IDS)
        or barrier["ordinary_successful_recordings"] != 3
        or barrier["technical_failures"] != 0
        or barrier["source_future_scoring_opened"] is not False
    ):
        raise ValueError(
            "all three ordinary-success prediction seals are required before outcomes"
        )
    validate_input_barrier(root, barrier["input_barrier_sha256"], source_digest)
    predictions = {}
    for case in CASE_IDS:
        case_root = root / case
        case_record = barrier["cases"][case]
        if (
            case_record["status"] != "ordinary-success"
            or file_digest(case_root / "prediction_seal.json")
            != case_record["seal_sha256"]
        ):
            raise ValueError("case seal changed or retained a technical failure")
        seal = json.loads((case_root / "prediction_seal.json").read_text())
        if (
            seal["case_id"] != case
            or seal["source_receipt_sha256"] != source_digest
            or seal["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
            or seal["source_future_scoring_opened"] is not False
            or seal["controls"]["zero_update_byte_identical"] is not True
            or seal["input_manifest_sha256"]
            != file_digest(case_root / "input_manifest.json")
            or seal["prediction_sha256"] != file_digest(case_root / "predictions.npz")
        ):
            raise ValueError("case prediction or exact-fallback identity changed")
        with np.load(case_root / "predictions.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if (
            set(arrays) != set(ARMS)
            or {arm: array_digest(value) for arm, value in arrays.items()}
            != seal["array_sha256s"]
            or any(
                x.shape != (120, 3, 13, 3) or not np.isfinite(x).all()
                for x in arrays.values()
            )
        ):
            raise ValueError("sealed prediction arrays changed")
        predictions[case] = arrays
    return barrier, predictions


def score(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    _, predictions = read_predictions(
        args.run_root, args.prediction_barrier_sha256, args.source_receipt_sha256
    )
    write_json_once(
        args.run_root / "score_attempt.json",
        {
            "prediction_barrier_sha256": args.prediction_barrier_sha256,
            "source_receipt_sha256": args.source_receipt_sha256,
        },
    )
    cases = {}
    for spec in protocol["source_cases"]:
        truth = load_training_case(args.training_root / spec["filename"], spec)[52:172]
        cases[spec["id"]] = score_case(predictions[spec["id"]], truth)
    result = {
        "schema": SCHEMA + "-result",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": args.source_receipt_sha256,
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "prediction_barrier_sha256": args.prediction_barrier_sha256,
        **score_study(cases),
        "ordinary_successful_recordings": 3,
        "technical_failures": 0,
        "unsealable": 0,
        "physical_object_count": 1,
        "checkpoint_training_exposure": True,
        "independent_confirmation": False,
        "confidence_interval": None,
        "protected_data_read": False,
        "public_evaluation_or_test_content_inspected": False,
        "automatic_next_experiment_authorization": False,
        "outcome_publication": "local-or-private-paper-evidence-only",
    }
    write_json_once(args.run_root / "result.json", result)
    print(
        json.dumps(
            {
                "stage": "source-scored",
                "source_gate_passed": result["source_gate_passed"],
                "equal_recording_mean": result["equal_recording_mean"],
                "recording_joint_wins": result["recording_joint_wins"],
                "result_sha256": file_digest(args.run_root / "result.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


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
            command.add_argument("--training-root", type=Path, required=True)
        if mode in ("stage", "predict"):
            command.add_argument("--qualification-result", type=Path, required=True)
        if mode == "predict":
            command.add_argument("--input-barrier-sha256", required=True)
            command.add_argument("--upstream", type=Path, required=True)
            command.add_argument("--checkpoint", type=Path, required=True)
        if mode == "score":
            command.add_argument("--prediction-barrier-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "freeze-source":
        freeze_source(args.output)
        return
    protocol = load_protocol()
    if args.run_root.resolve() != Path(protocol["registered_run_root"]):
        raise ValueError("only the registered write-once source output root is allowed")
    receipt = verify_source(args.source_receipt, args.source_receipt_sha256)
    {"stage": stage, "predict": predict, "score": score}[args.mode](
        args, protocol, receipt
    )


if __name__ == "__main__":
    main()
