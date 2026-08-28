#!/usr/bin/env python3
"""Post-result arithmetic audit; never supersedes the failed frozen verifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

CLAMPS = (0, 1, 10, 11)
ARMS = ("incumbent", "paired", "reference_initialized", "reference_centered")
HORIZONS = {"all": (0, 120), "early": (0, 40), "middle": (40, 80), "late": (80, 120)}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    identity = record.pop("artifact_id")
    if identity != canonical(record):
        raise ValueError("canonical record identity differs")
    return {**record, "artifact_id": identity}


def array_id(array: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def exact(a: np.ndarray, b: np.ndarray) -> None:
    if array_id(a) != array_id(b):
        raise ValueError("posthoc byte identity differs")


def close(a: Any, b: Any, tolerance: float = 1e-9) -> float:
    difference = float(
        np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )
    if not np.isfinite(difference) or difference > tolerance:
        raise ValueError(f"posthoc arithmetic differs: {difference}")
    return difference


def audit_traces(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Check preserved native clamps, and separately report commanded clamps."""
    native_clamps = arrays["nominal"][:, 50:170, CLAMPS]
    commands = arrays["future_actions"]
    if commands.shape != native_clamps.shape or not np.isfinite(commands).all():
        raise ValueError("command shape or finite contract differs")
    reports: dict[str, Any] = {}
    for arm in ("reference_initialized", "reference_centered", "zero_reference"):

        def trace(name: str, arm: str = arm) -> np.ndarray:
            value = arrays[arm + "__" + name]
            if value.shape != (14, 120, 12, 3) or not np.isfinite(value).all():
                raise ValueError("native trace shape or finite contract differs")
            return value

        before, after = trace("center_before"), trace("center_after")
        updated, updated_after = trace("updated_before"), trace("updated_after")
        vbefore, vafter = (
            trace("center_velocity_before"),
            trace("center_velocity_after"),
        )
        vupdated = trace("updated_velocity_before")
        vupdated_after = trace("updated_velocity_after")
        shift, vshift = trace("centering_dx"), trace("centering_dv")
        offset: np.ndarray = arrays["offsets"].astype(np.float32)
        voffset: np.ndarray = arrays["offset_velocities"].astype(np.float32)
        if arm == "zero_reference":
            offset, voffset = np.zeros_like(offset), np.zeros_like(voffset)
        exact(before[:, 0], arrays["nominal"][:, 49] + offset[:, 0])
        exact(vbefore[:, 0], arrays["nominal_velocity"][:, 49] + voffset[:, 0])
        exact(updated[:, 0], before[:, 0] + arrays["pose_increment"].astype(np.float32))
        exact(
            vupdated[:, 0],
            vbefore[:, 0] + arrays["velocity_increment"].astype(np.float32),
        )
        exact(before[:, 1:], after[:, :-1] + shift[:, :-1])
        exact(updated[:, 1:], updated_after[:, :-1] + shift[:, :-1])
        exact(vbefore[:, 1:], vafter[:, :-1] + vshift[:, :-1])
        exact(vupdated[:, 1:], vupdated_after[:, :-1] + vshift[:, :-1])
        if arm == "reference_centered":
            expected = arrays["nominal"][:, 50:169] - after[:, :-1] + offset[:, 1:-1]
            expected_v = (
                arrays["nominal_velocity"][:, 50:169]
                - vafter[:, :-1]
                + voffset[:, 1:-1]
            )
            expected_v[:, :, CLAMPS] = 0
            exact(shift[:, :-1], expected)
            exact(vshift[:, :-1], expected_v)
        elif np.count_nonzero(shift) or np.count_nonzero(vshift):
            raise ValueError("unregistered centering")
        if np.count_nonzero(shift[:, -1]) or np.count_nonzero(vshift[:, -1]):
            raise ValueError("unregistered post-horizon centering")
        readout_error = close(
            arrays[arm],
            arrays["incumbent"].astype(float)
            + updated_after.astype(float)
            - after.astype(float),
            1e-12,
        )
        exact(after[:, :, CLAMPS], native_clamps)
        exact(updated_after[:, :, CLAMPS], native_clamps)
        difference = after[:, :, CLAMPS].astype(float) - commands.astype(float)
        reports[arm] = {
            "native_clamp_bytes_preserved": True,
            "command_clamp_bytes_exact": array_id(after[:, :, CLAMPS])
            == array_id(commands),
            "maximum_command_difference_m": float(np.abs(difference).max()),
            "maximum_command_difference_by_axis_m": np.abs(difference)
            .max(axis=(0, 1, 2))
            .tolist(),
            "readout_maximum_arithmetic_difference_m": readout_error,
        }
    return reports


def audit_metrics(
    arrays: dict[str, np.ndarray], truth: np.ndarray, result: dict[str, Any]
) -> dict[str, Any]:
    names = arrays["names"].tolist()
    if len(names) != 14 or len(set(names)) != 14 or names.count("103.pkl") != 1:
        raise ValueError("frozen source denominator differs")
    case_metrics: dict[str, Any] = {}
    maximum_difference = 0.0
    for i, name in enumerate(names):
        if name == "103.pkl":
            continue
        case_metrics[name] = {}
        for arm in ARMS:
            case_metrics[name][arm] = {}
            for horizon, (start, end) in HORIZONS.items():
                error = arrays[arm][i, start:end][:, (3, 5, 7, 9)].astype(
                    float
                ) - truth[i, start:end][:, (3, 5, 7, 9)].astype(float)
                rows = error.reshape(-1, 3)
                values = {
                    "coordinate_l1_mm": float(np.abs(rows).sum()) / rows.size * 1000,
                    "point_rmse_mm": math.sqrt(float(np.square(rows).sum()) / len(rows))
                    * 1000,
                    "fde_mm": sum(
                        math.sqrt(float(np.square(v).sum())) for v in error[-1]
                    )
                    * 250,
                }
                case_metrics[name][arm][horizon] = values
                for metric, value in values.items():
                    maximum_difference = max(
                        maximum_difference,
                        close(
                            value, result["case_metrics"][name][arm][horizon][metric]
                        ),
                    )
    names = sorted(case_metrics)
    means = {
        arm: {
            h: {
                metric: sum(case_metrics[n][arm][h][metric] for n in names) / 13
                for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
            }
            for h in HORIZONS
        }
        for arm in ARMS
    }
    decision = result["decision"]
    for arm in ARMS:
        for h in HORIZONS:
            for metric, value in means[arm][h].items():
                maximum_difference = max(
                    maximum_difference, close(value, decision["means"][arm][h][metric])
                )
    differences = np.array(
        [
            case_metrics[n]["reference_centered"]["all"]["point_rmse_mm"]
            - case_metrics[n]["paired"]["all"]["point_rmse_mm"]
            for n in names
        ]
    )
    indices = np.random.default_rng(260929).integers(0, 13, (10000, 13))
    interval = np.asarray(np.percentile(differences[indices].mean(axis=1), [2.5, 97.5]))
    maximum_difference = max(
        maximum_difference, close(interval, decision["paired_rmse_difference_95pct_mm"])
    )
    wins = sum(
        all(
            case_metrics[n]["reference_centered"]["all"][m]
            < case_metrics[n]["paired"]["all"][m]
            for m in ("coordinate_l1_mm", "point_rmse_mm")
        )
        for n in names
    )
    worst = max(
        case_metrics[n]["reference_centered"]["all"]["point_rmse_mm"]
        / max(case_metrics[n]["paired"]["all"]["point_rmse_mm"], 1e-12)
        for n in names
    )
    base, candidate = means["paired"], means["reference_centered"]
    checks = {
        "two_percent_l1_gain": candidate["all"]["coordinate_l1_mm"]
        <= 0.98 * base["all"]["coordinate_l1_mm"],
        "two_percent_rmse_gain": candidate["all"]["point_rmse_mm"]
        <= 0.98 * base["all"]["point_rmse_mm"],
        "late_rmse_nonincreasing": candidate["late"]["point_rmse_mm"]
        <= base["late"]["point_rmse_mm"],
        "eight_of_thirteen_joint_wins": wins >= 8,
        "worst_rmse_ratio_at_most_1_05": worst <= 1.05,
        "rmse_bootstrap_upper_below_zero": bool(interval[1] < 0),
    }
    if checks != decision["checks"] or all(checks.values()) != decision["passed"]:
        raise ValueError("source value decision differs")
    if wins != decision["primary_joint_wins"]:
        raise ValueError("win count differs")
    close(worst, decision["primary_worst_case_rmse_ratio"])
    return {
        "metrics_verified": 13 * 4 * 4 * 3,
        "maximum_arithmetic_difference": maximum_difference,
        "checks": checks,
        "source_value_gate_passed": all(checks.values()),
    }


def audit(
    root: Path, source: Path, truth_path: Path, parent_path: Path
) -> dict[str, Any]:
    lock, seal, controls, result = (
        read(root / name)
        for name in (
            "lock.json",
            "prediction_seal.json",
            "controls.json",
            "result.json",
        )
    )
    if (
        seal["lock_id"] != lock["artifact_id"]
        or result["prediction_seal_id"] != seal["artifact_id"]
        or result["prediction_seal_sha256"] != digest(root / "prediction_seal.json")
        or seal["controls_id"] != controls["artifact_id"]
        or seal["controls_sha256"] != digest(root / "controls.json")
        or seal["complete"] is not True
        or seal["ordinary_successes"] != 14
        or seal["retained_technical_failures"] != 0
        or seal["unsealable"] != 0
        or not all(v is True for v in controls["checks"].values())
        or (root / "technical_failure.json").exists()
    ):
        raise ValueError("prediction barrier differs")
    for name, expected in lock["source_files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError("frozen source file differs")
    verifier_path = source / "scripts/verify_deform_reference_transport.py"
    if "scripts/verify_deform_reference_transport.py" not in lock["source_files"]:
        raise ValueError("original verifier is not source-bound")
    spec = importlib.util.spec_from_file_location(
        "frozen_reference_verifier", verifier_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load original verifier")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    # Reproduce the original failure, without changing its code or writing its receipt.
    try:
        verifier.verify(root, source, truth_path, parent_path)
    except ValueError as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        frame = next(
            f
            for f in frames
            if f.name == "verify"
            and Path(f.filename).resolve() == verifier_path.resolve()
        )
        if str(exc) != "state-transport arithmetic differs" or frame.line not in (
            'exact(after[:, :, (0, 1, 10, 11)], arrays["future_actions"])',
            'exact(updated_after[:, :, (0, 1, 10, 11)], arrays["future_actions"])',
        ):
            raise ValueError("original verifier failed at another contract") from exc
        original_failure = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "line": frame.lineno,
            "assertion": frame.line,
        }
    else:
        raise ValueError(
            "posthoc audit requires the retained original verifier failure"
        )
    plan = lock["plan"]
    if (
        digest(truth_path) != plan["archive"]["sha256"]
        or digest(parent_path) != plan["paired_archive"]["sha256"]
    ):
        raise ValueError("source archive differs")
    if digest(root / "predictions.npz") != seal["predictions"]["file_sha256"]:
        raise ValueError("sealed prediction file differs")
    with np.load(root / "predictions.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    if set(arrays) != set(seal["predictions"]["arrays"]):
        raise ValueError("sealed array members differ")
    for name, value in arrays.items():
        if seal["predictions"]["arrays"][name] != {
            "sha256": array_id(value),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }:
            raise ValueError("sealed array identity differs")
    if arrays["names"].tolist() != plan["names"] or seal["names"] != plan["names"]:
        raise ValueError("source roster differs")
    with np.load(parent_path, allow_pickle=False) as parent:
        exact(arrays["incumbent"], parent["incumbent"])
        exact(arrays["paired"], parent["incumbent_propagated_pose_velocity"])
    exact(arrays["zero_reference"], arrays["paired"])
    trace_report = audit_traces(arrays)
    with np.load(truth_path, allow_pickle=False) as archive:
        truth = archive["targets"][:, 50:170].copy()
        raw = archive["baseline_predictions"][:, :170].copy()
        base = archive["candidate_predictions"][:, :170].copy()
    exact(arrays["offsets"], base[:, 49:].astype(float) - raw[:, 49:].astype(float))
    exact(
        arrays["offset_velocities"],
        np.diff(base[:, 48:].astype(float) - raw[:, 48:].astype(float), axis=1) / 0.01,
    )
    native_error = arrays["nominal"].astype(float) - raw.astype(float)
    maximum_replay = float(np.abs(native_error).max())
    replay_rmse = math.sqrt(float(np.square(native_error).mean()))
    close(maximum_replay, controls["archived_gpu_replay_max_error_m"])
    close(replay_rmse, controls["archived_gpu_replay_coordinate_rmse_m"])
    if maximum_replay > 0.002 or replay_rmse > 0.0002:
        raise ValueError("registered numerical replay gate failed")
    metrics = audit_metrics(arrays, truth, result)
    if metrics["source_value_gate_passed"]:
        raise ValueError(
            "this posthoc diagnostic cannot authorize a passing source gate"
        )
    return {
        "schema": "deform-reference-transport-posthoc-arithmetic-v1",
        "posthoc": True,
        "original_second_arithmetic_passed": False,
        "original_failure_reproduced": original_failure,
        "original_verifier_sha256": digest(verifier_path),
        "literal_command_exact_check_passed": False,
        "native_preservation_and_metric_audit_passed": True,
        "original_failed_decision_unchanged": True,
        "promotion_authorized": False,
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "source_files_verified": len(lock["source_files"]),
        "arrays_verified": len(arrays),
        "traces": trace_report,
        "metrics": metrics,
        "raw_native_command_maximum_difference_m": float(
            np.abs(
                arrays["nominal"][:, 50:170, CLAMPS].astype(float)
                - arrays["future_actions"].astype(float)
            ).max()
        ),
        "independent_human_review": False,
        "new_native_execution": False,
        "new_recordings": False,
        "target_access": False,
        "held_v8_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-root",
        "source-root",
        "truth-archive",
        "parent-paired-archive",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(args.run_root.resolve()):
        raise ValueError("posthoc receipt must not modify the frozen run root")
    result = audit(
        args.run_root, args.source_root, args.truth_archive, args.parent_paired_archive
    )
    result["auditor_sha256"] = digest(Path(__file__))
    result["artifact_id"] = canonical(result)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
