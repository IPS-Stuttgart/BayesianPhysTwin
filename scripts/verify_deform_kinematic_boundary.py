#!/usr/bin/env python3
"""Second source arithmetic implementation; no experiment/native imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("incumbent", "paired", "hard_baseline", "hard_paired")
CLAMPS = (0, 1, 10, 11)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    identity = record.pop("artifact_id")
    if canonical(record) != identity:
        raise ValueError("canonical identity differs")
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
        raise ValueError("byte identity differs")


def close(a: Any, b: Any, tolerance: float = 1e-9) -> float:
    difference = float(
        np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max()
    )
    if not np.isfinite(difference) or difference > tolerance:
        raise ValueError(f"arithmetic differs: {difference}")
    return difference


def verify(
    root: Path, source: Path, archive_path: Path, parent_path: Path
) -> dict[str, Any]:
    lock, seal, controls, result = (
        read(root / n)
        for n in ("lock.json", "prediction_seal.json", "controls.json", "result.json")
    )
    plan = lock["plan"]
    if (
        lock["schema"] != "deform-kinematic-boundary-source-v1-lock"
        or seal["lock_id"] != lock["artifact_id"]
        or seal["controls_id"] != controls["artifact_id"]
        or seal["controls_sha256"] != digest(root / "controls.json")
        or result["prediction_seal_id"] != seal["artifact_id"]
        or result["prediction_seal_sha256"] != digest(root / "prediction_seal.json")
        or seal["complete"] is not True
        or seal["ordinary_successes"] != 14
        or any(
            seal[k] != 0
            for k in ("retained_technical_failures", "unsealable", "replacements")
        )
        or not all(v is True for v in controls["checks"].values())
        or len(controls["checks"]) != 10
        or (root / "technical_failure.json").exists()
    ):
        raise ValueError("complete source barrier required")
    for name, sha in lock["source_files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != sha:
            raise ValueError("bound source bytes differ")
    if (
        digest(archive_path) != plan["archive"]["sha256"]
        or digest(parent_path) != plan["paired_archive"]["sha256"]
    ):
        raise ValueError("registered source archive differs")
    if digest(root / "predictions.npz") != seal["predictions"]["file_sha256"]:
        raise ValueError("sealed prediction file differs")
    with np.load(root / "predictions.npz", allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if set(arrays) != set(seal["predictions"]["arrays"]):
        raise ValueError("sealed member set differs")
    for name, value in arrays.items():
        if seal["predictions"]["arrays"][name] != {
            "sha256": array_id(value),
            "shape": list(value.shape),
            "dtype": value.dtype.str,
        }:
            raise ValueError("sealed array identity differs")
        if name != "names" and not np.isfinite(value).all():
            raise ValueError("nonfinite sealed array")
    names = arrays["names"].tolist()
    if names != plan["names"] or seal["names"] != names or len(set(names)) != 14:
        raise ValueError("source roster differs")
    with np.load(parent_path, allow_pickle=False) as parent:
        exact(arrays["incumbent"], parent["incumbent"])
        exact(arrays["paired"], parent["incumbent_propagated_pose_velocity"])
    exact(arrays["incumbent"], arrays["incumbent_full"][:, 50:])
    exact(arrays["hard_zero"], arrays["hard_native"][:, 50:])
    for name in ("m_restWprev", "m_restWnext", "learned_pmass"):
        exact(arrays["native_rest__" + name], arrays["hard_rest__" + name])
    exact(
        arrays["hard_native"][:, :, CLAMPS], arrays["known_actions"].astype(np.float32)
    )
    exact(
        arrays["hard_updated"][:, :, CLAMPS],
        arrays["known_actions"][:, 50:].astype(np.float32),
    )
    offset = arrays["incumbent_full"].astype(float) - arrays["archived_native"].astype(
        float
    )
    if np.count_nonzero(offset[:, :, CLAMPS]):
        raise ValueError("readout changed prescribed boundary nodes")
    hard_full = arrays["hard_native"].astype(float) + offset
    exact(arrays["hard_baseline"], hard_full[:, 50:])
    close(
        arrays["hard_paired"],
        hard_full[:, 50:]
        + (
            arrays["hard_updated"].astype(float)
            - arrays["hard_native"][:, 50:].astype(float)
        ),
        1e-12,
    )
    close(
        arrays["paired"],
        arrays["incumbent"].astype(float)
        + (
            arrays["native_updated"].astype(float)
            - arrays["native"][:, 50:].astype(float)
        ),
        1e-12,
    )
    knots = (0, 1, 2, 4, 6, 8, 10, 11)
    for label, reference in (("old", arrays["incumbent_full"]), ("hard", hard_full)):
        residual = arrays["sparse_observations"].astype(float) - reference[:, (41, 49)][
            :, :, (2, 4, 6, 8)
        ].astype(float)
        for name, values in (
            ("dx", residual[:, 1]),
            ("dv", (residual[:, 1] - residual[:, 0]) / 0.08),
        ):
            expected = np.zeros((14, 12, 3))
            knot_values = np.zeros((14, 8, 3))
            knot_values[:, (2, 3, 4, 5)] = values
            for i in range(14):
                for axis in range(3):
                    expected[i, :, axis] = np.interp(
                        np.arange(12), knots, knot_values[i, :, axis]
                    )
            exact(arrays[label + "_" + name], expected)
    exact(
        arrays["hard_updated_endpoint_positions"],
        arrays["hard_endpoint_positions"] + arrays["hard_dx"].astype(np.float32),
    )
    exact(
        arrays["hard_updated_endpoint_velocity"],
        arrays["hard_endpoint_velocity"] + arrays["hard_dv"].astype(np.float32),
    )
    exact(arrays["hard_endpoint_positions"], arrays["hard_native"][:, 49])
    # Source truth is accessed only after custody and prediction identities verify.
    with np.load(archive_path, allow_pickle=False) as archive:
        exact(arrays["archived_native"], archive["baseline_predictions"][:, :170])
        exact(arrays["incumbent_full"], archive["candidate_predictions"][:, :170])
        truth = archive["targets"][:, 50:170].copy()
    replay = arrays["native"].astype(float) - arrays["archived_native"].astype(float)
    close(np.abs(replay).max(), controls["archived_gpu_replay_max_error_m"])
    close(
        np.sqrt(np.square(replay).mean()),
        controls["archived_gpu_replay_coordinate_rmse_m"],
    )
    if np.abs(replay).max() > 0.002 or np.sqrt(np.square(replay).mean()) > 0.0002:
        raise ValueError("original numerical replay tolerance failed")
    horizons = {
        "all": (0, 120),
        "early": (0, 40),
        "middle": (40, 80),
        "late": (80, 120),
    }
    cases: dict[str, Any] = {}
    maximum_difference = 0.0
    for i, name in enumerate(names):
        if name == "103.pkl":
            continue
        cases[name] = {}
        for arm in ARMS:
            cases[name][arm] = {}
            for horizon, (start, end) in horizons.items():
                error = arrays[arm][i, start:end][:, (3, 5, 7, 9)].astype(
                    float
                ) - truth[i, start:end][:, (3, 5, 7, 9)].astype(float)
                rows = error.reshape(-1, 3)
                metrics = {
                    "coordinate_l1_mm": float(np.abs(rows).sum()) / rows.size * 1000,
                    "point_rmse_mm": math.sqrt(float(np.square(rows).sum()) / len(rows))
                    * 1000,
                    "fde_mm": sum(
                        math.sqrt(float(np.square(v).sum())) for v in error[-1]
                    )
                    * 250,
                }
                cases[name][arm][horizon] = metrics
                for metric, value in metrics.items():
                    maximum_difference = max(
                        maximum_difference,
                        close(
                            value, result["case_metrics"][name][arm][horizon][metric]
                        ),
                    )
    if len(cases) != 13:
        raise ValueError("source analysis denominator differs")
    means = {
        a: {
            h: {
                m: sum(cases[n][a][h][m] for n in sorted(cases)) / 13
                for m in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
            }
            for h in horizons
        }
        for a in ARMS
    }
    decision = result["decision"]
    for arm in ARMS:
        for h in horizons:
            for metric, value in means[arm][h].items():
                maximum_difference = max(
                    maximum_difference, close(value, decision["means"][arm][h][metric])
                )
    delta = np.array(
        [
            cases[n]["hard_paired"]["all"]["point_rmse_mm"]
            - cases[n]["paired"]["all"]["point_rmse_mm"]
            for n in sorted(cases)
        ]
    )
    rng = np.random.default_rng(260929)
    bootstrap = np.array(
        [sum(delta[rng.integers(0, 13, 13)]) / 13 for _ in range(10000)]
    )
    interval = np.asarray(np.percentile(bootstrap, [2.5, 97.5]))
    maximum_difference = max(
        maximum_difference, close(interval, decision["paired_rmse_difference_95pct_mm"])
    )
    wins = sum(
        all(
            c["hard_paired"]["all"][m] < c["paired"]["all"][m]
            for m in ("coordinate_l1_mm", "point_rmse_mm")
        )
        for c in cases.values()
    )
    worst = max(
        c["hard_paired"]["all"]["point_rmse_mm"]
        / max(c["paired"]["all"]["point_rmse_mm"], 1e-12)
        for c in cases.values()
    )
    a, b, c = means["hard_paired"], means["paired"], means["hard_baseline"]
    checks = {
        "two_percent_l1_gain": a["all"]["coordinate_l1_mm"]
        <= 0.98 * b["all"]["coordinate_l1_mm"],
        "two_percent_rmse_gain": a["all"]["point_rmse_mm"]
        <= 0.98 * b["all"]["point_rmse_mm"],
        "late_rmse_nonincreasing": a["late"]["point_rmse_mm"]
        <= b["late"]["point_rmse_mm"],
        "eight_of_thirteen_joint_wins": wins >= 8,
        "worst_rmse_ratio_at_most_1_05": worst <= 1.05,
        "rmse_bootstrap_upper_below_zero": bool(interval[1] < 0),
        "sparse_update_improves_hard_baseline": all(
            a["all"][m] < c["all"][m] for m in ("coordinate_l1_mm", "point_rmse_mm")
        ),
    }
    if (
        checks != decision["checks"]
        or all(checks.values()) != decision["passed"]
        or wins != decision["primary_joint_wins"]
    ):
        raise ValueError("frozen source decision differs")
    close(worst, decision["primary_worst_case_rmse_ratio"])
    return {
        "schema": "deform-kinematic-boundary-second-arithmetic-v1",
        "passed": True,
        "lock_id": lock["artifact_id"],
        "result_id": result["artifact_id"],
        "source_files_verified": len(lock["source_files"]),
        "arrays_verified": len(arrays),
        "metrics_verified": 624,
        "maximum_arithmetic_difference": maximum_difference,
        "source_value_gate_passed": decision["passed"],
        "independent_human_review": False,
        "new_native_execution": False,
        "target_access": False,
        "held_v8_access": False,
        "transfer_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-root",
        "source-root",
        "source-archive",
        "parent-paired-archive",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_root, args.source_root, args.source_archive, args.parent_paired_archive
    )
    result["verifier_sha256"] = digest(Path(__file__))
    result["artifact_id"] = canonical(result)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
