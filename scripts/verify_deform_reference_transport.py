#!/usr/bin/env python3
"""Second arithmetic implementation; imports no experiment code or simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
        raise ValueError("canonical record identity changed")
    return {**record, "artifact_id": identity}


def array_id(array: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def verify(
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
        or not seal["complete"]
        or seal["ordinary_successes"] != 14
        or seal["retained_technical_failures"] != 0
        or seal["unsealable"] != 0
        or not all(v is True for v in controls["checks"].values())
        or (root / "technical_failure.json").exists()
    ):
        raise ValueError("complete qualified source barrier does not verify")
    plan = lock["plan"]
    for name, expected in lock["source_files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError("bound source file changed")
    if (
        digest(truth_path) != plan["archive"]["sha256"]
        or digest(parent_path) != plan["paired_archive"]["sha256"]
    ):
        raise ValueError("opened source archive identity differs")
    if digest(root / "predictions.npz") != seal["predictions"]["file_sha256"]:
        raise ValueError("prediction file changed")
    with np.load(root / "predictions.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    if set(arrays) != set(seal["predictions"]["arrays"]):
        raise ValueError("prediction members changed")
    for name, array in arrays.items():
        if seal["predictions"]["arrays"][name] != {
            "sha256": array_id(array),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }:
            raise ValueError("prediction array identity changed")
    if arrays["names"].tolist() != plan["names"] or seal["names"] != plan["names"]:
        raise ValueError("source roster differs")
    with np.load(parent_path, allow_pickle=False) as parent:
        for key, old in (
            ("incumbent", "incumbent"),
            ("paired", "incumbent_propagated_pose_velocity"),
        ):
            if array_id(arrays[key]) != array_id(parent[old]):
                raise ValueError("existing forecast changed")
    if array_id(arrays["zero_reference"]) != array_id(arrays["paired"]):
        raise ValueError("zero-reference control differs")

    max_continuous_difference = 0.0

    def close(a: Any, b: Any, tolerance: float = 1e-9) -> None:
        nonlocal max_continuous_difference
        delta = float(
            np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
        )
        if not np.isfinite(delta) or delta > tolerance:
            raise ValueError(f"arithmetic mismatch: {delta}")
        max_continuous_difference = max(max_continuous_difference, delta)

    def exact(a: np.ndarray, b: np.ndarray) -> None:
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or a.tobytes(order="C") != b.tobytes(order="C")
        ):
            raise ValueError("state-transport arithmetic differs")

    for arm in ("reference_initialized", "reference_centered", "zero_reference"):

        def trace(name: str, arm: str = arm) -> np.ndarray:
            value = arrays[arm + "__" + name]
            if value.shape != (14, 120, 12, 3) or not np.isfinite(value).all():
                raise ValueError("native trace shape/finite contract differs")
            return value

        before, after = trace("center_before"), trace("center_after")
        updated, updated_after = trace("updated_before"), trace("updated_after")
        vbefore, vafter = (
            trace("center_velocity_before"),
            trace("center_velocity_after"),
        )
        vupdated, vupdated_after = (
            trace("updated_velocity_before"),
            trace("updated_velocity_after"),
        )
        shift, vshift = trace("centering_dx"), trace("centering_dv")
        offset = (
            arrays["offsets"].astype(np.float32)
            if arm != "zero_reference"
            else np.zeros_like(arrays["offsets"], dtype=np.float32)
        )
        voffset = (
            arrays["offset_velocities"].astype(np.float32)
            if arm != "zero_reference"
            else np.zeros_like(arrays["offset_velocities"], dtype=np.float32)
        )
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
            expected_v[:, :, (0, 1, 10, 11)] = 0
            exact(shift[:, :-1], expected)
            exact(vshift[:, :-1], expected_v)
        elif np.count_nonzero(shift) or np.count_nonzero(vshift):
            raise ValueError("non-centered control applied a centering defect")
        if np.count_nonzero(shift[:, -1]) or np.count_nonzero(vshift[:, -1]):
            raise ValueError("unregistered post-horizon centering")
        close(
            arrays[arm],
            arrays["incumbent"].astype(float)
            + (updated_after.astype(float) - after.astype(float)),
            1e-12,
        )
        exact(after[:, :, (0, 1, 10, 11)], arrays["future_actions"])
        exact(updated_after[:, :, (0, 1, 10, 11)], arrays["future_actions"])

    # Truth access follows full custody and transport checks, never precedes them.
    with np.load(truth_path, allow_pickle=False) as archive:
        truth = archive["targets"][:, 50:170].copy()
        raw_physical = archive["baseline_predictions"][:, :170].copy()
        base = archive["candidate_predictions"][:, :170].copy()
    expected_offset = base[:, 49:].astype(float) - raw_physical[:, 49:].astype(float)
    exact(arrays["offsets"], expected_offset)
    expected_velocity = (
        np.diff(base[:, 48:].astype(float) - raw_physical[:, 48:].astype(float), axis=1)
        / 0.01
    )
    exact(arrays["offset_velocities"], expected_velocity)
    native_error = arrays["nominal"].astype(float) - raw_physical
    close(controls["archived_gpu_replay_max_error_m"], np.abs(native_error).max())
    close(
        controls["archived_gpu_replay_coordinate_rmse_m"],
        math.sqrt(float(np.square(native_error).mean())),
    )
    if (
        np.abs(native_error).max() > 0.002
        or np.sqrt(np.square(native_error).mean()) > 0.0002
    ):
        raise ValueError("native replay tolerance failed")
    arms = ("incumbent", "paired", "reference_initialized", "reference_centered")
    horizons = {
        "all": (0, 120),
        "early": (0, 40),
        "middle": (40, 80),
        "late": (80, 120),
    }
    case_metrics: dict[str, Any] = {}
    for i, name in enumerate(plan["names"]):
        if name == "103.pkl":
            continue
        case_metrics[name] = {}
        for arm in arms:
            case_metrics[name][arm] = {}
            for horizon, (start, end) in horizons.items():
                e = arrays[arm][i, start:end][:, (3, 5, 7, 9)].astype(float) - truth[
                    i, start:end
                ][:, (3, 5, 7, 9)].astype(float)
                rows = e.reshape(-1, 3)
                values = {
                    "coordinate_l1_mm": 1000 * float(np.abs(rows).sum()) / rows.size,
                    "point_rmse_mm": 1000
                    * math.sqrt(float(np.square(rows).sum()) / len(rows)),
                    "fde_mm": 1000
                    * sum(math.sqrt(float(np.square(v).sum())) for v in e[-1])
                    / 4,
                }
                case_metrics[name][arm][horizon] = values
                for metric, value in values.items():
                    close(value, result["case_metrics"][name][arm][horizon][metric])
    names = sorted(case_metrics)
    means = {
        a: {
            h: {
                m: sum(case_metrics[n][a][h][m] for n in names) / 13
                for m in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
            }
            for h in horizons
        }
        for a in arms
    }
    for a in arms:
        for h in horizons:
            for metric in means[a][h]:
                close(means[a][h][metric], result["decision"]["means"][a][h][metric])
    delta = np.array(
        [
            case_metrics[n]["reference_centered"]["all"]["point_rmse_mm"]
            - case_metrics[n]["paired"]["all"]["point_rmse_mm"]
            for n in names
        ]
    )
    rng = np.random.default_rng(260929)
    bootstrap = np.array(
        [sum(delta[rng.integers(0, 13, 13)]) / 13 for _ in range(10000)]
    )
    interval = np.asarray(np.percentile(bootstrap, [2.5, 97.5]))
    close(interval, result["decision"]["paired_rmse_difference_95pct_mm"])
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
    b, c = means["paired"], means["reference_centered"]
    checks = {
        "two_percent_l1_gain": c["all"]["coordinate_l1_mm"]
        <= 0.98 * b["all"]["coordinate_l1_mm"],
        "two_percent_rmse_gain": c["all"]["point_rmse_mm"]
        <= 0.98 * b["all"]["point_rmse_mm"],
        "late_rmse_nonincreasing": c["late"]["point_rmse_mm"]
        <= b["late"]["point_rmse_mm"],
        "eight_of_thirteen_joint_wins": wins >= 8,
        "worst_rmse_ratio_at_most_1_05": worst <= 1.05,
        "rmse_bootstrap_upper_below_zero": bool(interval[1] < 0),
    }
    if (
        checks != result["decision"]["checks"]
        or all(checks.values()) != result["decision"]["passed"]
    ):
        raise ValueError("source decision differs")
    if wins != result["decision"]["primary_joint_wins"]:
        raise ValueError("win count differs")
    close(worst, result["decision"]["primary_worst_case_rmse_ratio"])
    return {
        "schema": "deform-reference-transport-second-arithmetic-v1",
        "passed": True,
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "source_files_verified": len(lock["source_files"]),
        "arrays_verified": len(arrays),
        "trajectory_horizon_arm_metrics_verified": 13 * 4 * 4 * 3,
        "paired_native_trace_steps_verified": 14 * 120 * 3,
        "maximum_continuous_arithmetic_difference": max_continuous_difference,
        "independent_human_review": False,
        "new_native_execution": False,
        "new_recordings": False,
        "target_access": False,
        "held_v8_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in (
        "run-root",
        "source-root",
        "truth-archive",
        "parent-paired-archive",
        "output",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_root, args.source_root, args.truth_archive, args.parent_paired_archive
    )
    result["verifier_sha256"] = digest(Path(__file__))
    result["artifact_id"] = canonical(result)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
