#!/usr/bin/env python3
"""Second arithmetic implementation: pivoted QR, direct metrics, and gates.

This checker imports no production experiment code. It is not an independent
human review and does not generate a new empirical prediction or tune an arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import qr


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_bound(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value["artifact_id"]
    identity = {key: item for key, item in value.items() if key != "artifact_id"}
    if canonical(identity) != expected:
        raise ValueError(f"canonical digest mismatch: {path.name}")
    return value


def array_id(value: np.ndarray) -> str:
    header = {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C"}
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw + b"\0" + value.tobytes(order="C")).hexdigest()


def qr_projection(
    geometry: np.ndarray, displacement: np.ndarray, clamps: tuple[int, ...]
) -> np.ndarray:
    points = np.asarray(geometry, dtype=np.float64)
    delta = np.asarray(displacement, dtype=np.float64)
    n = len(points)
    if points.shape != (n, 3) or delta.shape != points.shape:
        raise ValueError("aligned geometry and displacement required")
    free = [i for i in range(n) if i not in clamps]
    incidence = np.eye(n)[1:] - np.eye(n)[:-1]
    edges = incidence @ points
    lengths = np.sqrt((edges * edges).sum(axis=-1))
    if not np.isfinite(lengths).all() or np.any(lengths < 1e-8):
        raise ValueError("invalid nominal geometry")
    tangents = edges / lengths[:, None]
    rows = np.einsum("ij,ik->ijk", incidence, tangents)[:, free].reshape(n - 1, -1)
    q, r, _ = qr(rows.T, mode="full", pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank = int(np.count_nonzero(diagonal > 1e-10 * diagonal.max()))
    null = q[:, rank:]
    result = np.zeros_like(delta)
    result[free] = (null @ (null.T @ delta[free].reshape(-1))).reshape(-1, 3)
    if np.max(np.abs(rows @ result[free].reshape(-1))) > 1e-9:
        raise ValueError("QR projection violates the frozen constraint tolerance")
    return result


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    squared_point_error = (error * error).sum(axis=2)
    return {
        "coordinate_l1_mm": float(1000 * np.abs(error).sum() / error.size),
        "point_rmse_mm": float(
            1000 * np.sqrt(squared_point_error.sum() / squared_point_error.size)
        ),
        "fde_mm": float(1000 * np.sqrt(squared_point_error[-1]).mean()),
    }


def verify(root: Path, repo: Path, input_root: Path | None) -> dict[str, Any]:
    lock = read_bound(root / "source-lock.json")
    seal = read_bound(root / "prediction-seal.json")
    result = read_bound(root / "result.json")
    attempt = read_bound(root / "prediction-attempt.json")
    if not (
        seal["source_lock_id"] == lock["artifact_id"] == result["source_lock_id"]
        and result["prediction_seal_id"] == seal["artifact_id"]
        and result["prediction_seal_sha256"] == digest(root / "prediction-seal.json")
        and seal["attempt_sha256"] == digest(root / "prediction-attempt.json")
        and attempt["source_lock_id"] == lock["artifact_id"]
        and attempt["attempt"] == 1
        and attempt["output_root"] == lock["output_root"]
    ):
        raise ValueError("source/prediction/result chain differs")
    for path, expected in lock["source_files"].items():
        raw = subprocess.check_output(
            ["git", "show", f"{lock['revision']}:{path}"], cwd=repo
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("committed source differs")
    protocol_path = "configs/sota/deform_constraint_split_source_v1.json"
    raw = subprocess.check_output(
        ["git", "show", f"{lock['revision']}:{protocol_path}"], cwd=repo
    )
    if hashlib.sha256(raw).hexdigest() != lock["protocol_sha256"]:
        raise ValueError("protocol differs")
    protocol = json.loads(raw)
    paths = {}
    for label, spec in protocol["inputs"].items():
        if label == "parent_protocol":
            path = repo / spec["path"]
        else:
            path = (
                input_root / Path(spec["path"]).name
                if input_root
                else Path(spec["path"])
            )
        if digest(path) != spec["sha256"]:
            raise ValueError(f"frozen input differs: {label}")
        paths[label] = path
    old_seal = json.loads(paths["prediction_seal"].read_text())
    with np.load(paths["clean"], allow_pickle=False) as archive:
        original = {key: archive[key] for key in archive.files}
    if any(
        array_id(value) != old_seal["files"]["clean"]["arrays"][key]
        for key, value in original.items()
    ):
        raise ValueError("parent arrays differ")
    if digest(root / "predictions.npz") != seal["predictions_sha256"]:
        raise ValueError("prediction bytes differ")
    with np.load(root / "predictions.npz", allow_pickle=False) as archive:
        forecasts = {key: archive[key] for key in archive.files}
    if set(forecasts) != {"names", *protocol["arms"]} or any(
        array_id(value) != seal["array_sha256s"][key]
        for key, value in forecasts.items()
    ):
        raise ValueError("sealed arm bank differs")
    names = forecasts.pop("names").tolist()
    parent = json.loads(paths["parent_protocol"].read_text())
    roster = next(
        item["names"] for item in parent["objects"] if item["object"] == "DLO2"
    )
    if names != roster or names != seal["names"] or len(names) != 14:
        raise ValueError("complete opened-source roster differs")
    mappings = {
        "incumbent": "incumbent",
        "paired": "incumbent_propagated_pose_velocity",
        "readout": "readout_sparse_pose",
    }
    for arm, key in mappings.items():
        if array_id(forecasts[arm]) != array_id(original[key]):
            raise ValueError("unchanged comparator was altered")
    maximum_prediction_error = 0.0
    clamps = (0, 1, 10, 11)
    for case in range(14):
        base = forecasts["incumbent"][case]
        dynamic = forecasts["paired"][case] - base
        offset = forecasts["readout"][case, 0] - base[0]
        tangents = []
        primary = []
        for frame in range(120):
            geometry = original["physical_nominal"][case, frame]
            projected = qr_projection(geometry, dynamic[frame], clamps)
            normal = offset - qr_projection(geometry, offset, clamps)
            tangents.append(base[frame] + projected)
            primary.append(base[frame] + projected + normal)
        replay = {
            "tangent_only": np.asarray(tangents),
            "constraint_split": np.asarray(primary),
            "half_blend": base + 0.5 * dynamic + 0.5 * offset,
        }
        for arm, value in replay.items():
            error = float(np.max(np.abs(value - forecasts[arm][case])))
            maximum_prediction_error = max(maximum_prediction_error, error)
            if error > 1e-12:
                raise ValueError("second projection arithmetic differs")
    with np.load(paths["source_truth"], allow_pickle=False) as archive:
        if archive["names"].tolist() != names:
            raise ValueError("truth identities differ")
        truth = archive["targets"][:, 50:170]
    cases = [i for i, name in enumerate(names) if name != "103.pkl"]
    horizons = {
        "all": (0, 120),
        "early": (0, 40),
        "middle": (40, 80),
        "late": (80, 120),
    }
    measured: dict[str, Any] = {}
    metric_count = 0
    for arm, forecast in forecasts.items():
        measured[arm] = {}
        for horizon, (start, end) in horizons.items():
            rows = [
                metric(
                    forecast[i, start:end][:, (3, 5, 7, 9)],
                    truth[i, start:end][:, (3, 5, 7, 9)],
                )
                for i in cases
            ]
            measured[arm][horizon] = rows
            for i, row in zip(cases, rows, strict=True):
                for key, value in row.items():
                    metric_count += 1
                    if (
                        abs(
                            value
                            - result["metrics"][arm]["cases"][names[i]][horizon][key]
                        )
                        > 1e-10
                    ):
                        raise ValueError("per-trajectory metric differs")
            for key in rows[0]:
                metric_count += 1
                if (
                    abs(
                        np.mean([row[key] for row in rows])
                        - result["metrics"][arm]["mean"][horizon][key]
                    )
                    > 1e-10
                ):
                    raise ValueError("equal-trajectory aggregation differs")
    rng = np.random.default_rng(260913)
    sampled = rng.integers(0, 13, (10000, 13))
    controls = ("incumbent", "paired", "readout", "half_blend")
    primary = measured["constraint_split"]
    differences = {}
    for arm in controls:
        delta = np.array(
            [
                a["point_rmse_mm"] - b["point_rmse_mm"]
                for a, b in zip(primary["all"], measured[arm]["all"], strict=True)
            ]
        )
        ci = np.quantile(delta[sampled].mean(axis=1), (0.025, 0.975))
        np.testing.assert_allclose(
            ci,
            result["contrasts"][arm]["trajectory_bootstrap_ci95_mm"],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            delta.mean(),
            result["contrasts"][arm]["rmse_difference_mm"],
            atol=1e-10,
            rtol=0,
        )
        differences[arm] = ci
    def mean(arm: str, horizon: str, key: str) -> float:
        return float(np.mean([row[key] for row in measured[arm][horizon]]))
    wins = sum(
        all(a[key] < b[key] for key in ("coordinate_l1_mm", "point_rmse_mm"))
        for a, b in zip(primary["all"], measured["paired"]["all"], strict=True)
    )
    checks = {
        "all_14_ordinary_predictions": seal["ordinary_success"] == 14,
        "at_least_2_percent_rmse_gain_over_every_control": all(
            mean("constraint_split", "all", "point_rmse_mm")
            <= 0.98 * mean(arm, "all", "point_rmse_mm")
            for arm in controls
        ),
        "lower_l1_than_every_control": all(
            mean("constraint_split", "all", "coordinate_l1_mm")
            < mean(arm, "all", "coordinate_l1_mm")
            for arm in controls
        ),
        "at_least_9_of_13_joint_wins_over_paired": wins >= 9,
        "late_rmse_no_worse_than_incumbent_and_paired": all(
            mean("constraint_split", "late", "point_rmse_mm")
            <= mean(arm, "late", "point_rmse_mm")
            for arm in ("incumbent", "paired")
        ),
        "every_case_rmse_at_most_1_05_times_incumbent": all(
            a["point_rmse_mm"] <= 1.05 * b["point_rmse_mm"]
            for a, b in zip(primary["all"], measured["incumbent"]["all"], strict=True)
        ),
        "rmse_difference_upper_bound_below_zero_against_every_control": all(
            ci[1] < 0 for ci in differences.values()
        ),
    }
    if (
        checks != result["gate"]["checks"]
        or all(checks.values()) != result["gate"]["passed"]
        or wins != result["joint_wins_over_paired"]
    ):
        raise ValueError("source gate differs")
    if any(
        result[key] is not False
        for key in (
            "protected_data_access",
            "transfer_objects_accessed",
            "native_replays_performed",
            "new_transfer_or_target_execution_authorized",
            "fresh_confirmation",
        )
    ):
        raise ValueError("claim boundary differs")
    report = {
        "schema": "deform-constraint-split-second-arithmetic-v1",
        "verifier_file_sha256": digest(Path(__file__)),
        "result_id": result["artifact_id"],
        "source_revision": lock["revision"],
        "method": "pivoted-QR-nullspace-versus-production-SVD",
        "projection_cases": 14,
        "projection_frames": 1680,
        "maximum_projection_prediction_difference_m": maximum_prediction_error,
        "metric_checks": metric_count,
        "bootstrap_contrasts": 4,
        "gate_checks": len(checks),
        "source_gate_passed": bool(all(checks.values())),
        "verified": True,
        "independent_human_review": False,
        "new_empirical_run": False,
        "protected_data_access": False,
    }
    return {**report, "artifact_id": canonical(report)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.evidence_root, args.repo, args.input_root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
