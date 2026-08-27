"""Independently recompute sealed state-restart metrics without replaying physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)


def verify(run: Path, archive: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    barrier = json.loads((run / "prediction_barrier.json").read_text())
    controls = json.loads((run / "controls.json").read_text())
    result = json.loads((run / "result.json").read_text())
    preflight = json.loads((run / "preflight.json").read_text())
    if file_digest(archive) != protocol["source_archive_sha256"]:
        raise ValueError("opened archive digest differs")
    if preflight["protocol_sha256"] != file_digest(protocol_path):
        raise ValueError("protocol identity differs from preflight")
    for filename, key in (
        ("predictions.npz", "predictions_sha256"),
        ("controls.json", "controls_sha256"),
        ("preflight.json", "preflight_sha256"),
    ):
        if file_digest(run / filename) != barrier[key]:
            raise ValueError(f"barrier member changed: {filename}")
    if result["prediction_barrier_sha256"] != file_digest(
        run / "prediction_barrier.json"
    ):
        raise ValueError("result barrier identity differs")
    if barrier["arms"] != protocol["arms"] or result["controls"] != controls:
        raise ValueError("arm or control custody differs")
    for key in (
        "native_adapter_max_error_m",
        "archived_gpu_replay_max_error_m",
        "archived_gpu_replay_coordinate_rmse_m",
    ):
        if not 0 <= controls[key] <= protocol["controls"][key]:
            raise ValueError("recorded native runtime control failed")
    if not (
        controls["passed"] is True
        and controls["zero_update_continuation_byte_identical"] is True
        and controls["incumbent_zero_update_returns_original_object"] is True
        and controls["synthetic_recovery_fraction"]
        >= protocol["controls"]["synthetic_minimum_recovery_fraction"]
    ):
        raise ValueError("recorded native state control failed")
    with np.load(archive, allow_pickle=False) as data:
        names = data["names"].tolist()
        start, end = protocol["prefix_length"], protocol["forecast_end"]
        original = data["candidate_predictions"][:, start:end].copy()
        target = data["targets"][:, start:end].copy()
        prefix = data["candidate_predictions"][:, :start].copy()
        observed = data["targets"][:, start - 1, protocol["observed_nodes"]].copy()
    with np.load(run / "predictions.npz", allow_pickle=False) as data:
        predictions = {name: data[name].copy() for name in data.files}
    if names != protocol["expected_names"] or barrier["case_count"] != len(names):
        raise ValueError("fixed roster differs")
    if list(predictions) != protocol["arms"] or set(result["per_case"]) != set(
        predictions
    ):
        raise ValueError("missing or extra forecast arm")
    if array_digest(predictions["incumbent"]) != array_digest(original):
        raise ValueError("incumbent bytes changed")
    # Independent interpolation calculation, without the production update helper.
    knots = sorted(protocol["observed_nodes"] + protocol["clamped_nodes"])
    weights = np.zeros((protocol["node_count"], len(protocol["observed_nodes"])))
    for node in range(protocol["node_count"]):
        if node in protocol["observed_nodes"]:
            weights[node, protocol["observed_nodes"].index(node)] = 1
        elif node not in protocol["clamped_nodes"]:
            right = int(np.searchsorted(knots, node))
            lo, hi = knots[right - 1], knots[right]
            for anchor, weight in (
                (lo, (hi - node) / (hi - lo)),
                (hi, (node - lo) / (hi - lo)),
            ):
                if anchor in protocol["observed_nodes"]:
                    weights[node, protocol["observed_nodes"].index(anchor)] = weight
    residual = observed - prefix[:, -1, protocol["observed_nodes"]]
    readout = original + np.einsum("nm,bmc->bnc", weights, residual)[:, None]
    np.testing.assert_allclose(
        predictions["readout_sparse_pose"], readout, atol=1e-15, rtol=1e-14
    )
    keep = [i for i, name in enumerate(names) if name != protocol["design_case"]]
    draws = np.random.default_rng(protocol["seed"]).integers(
        0,
        len(keep),
        size=(protocol["bootstrap_replicates"], len(keep)),
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for arm, points in predictions.items():
        if array_digest(points) != barrier["array_digests"][arm]:
            raise ValueError("prediction array identity differs")
        if points.shape != original.shape or not np.isfinite(points).all():
            raise ValueError("invalid point-array shape or values")
        rows[arm] = []
        for index in range(len(names)):
            residual = (
                points[index][:, protocol["hidden_nodes"]].astype(float)
                - target[index][:, protocol["hidden_nodes"]]
            )
            row: dict[str, Any] = {}
            for label, frames in [
                ("all", np.arange(end - start)),
                *zip(
                    ("early", "middle", "late"),
                    np.array_split(np.arange(end - start), 3),
                    strict=True,
                ),
            ]:
                error = residual[frames]
                values = {
                    "coordinate_l1_mm": float(
                        sum(np.abs(error).ravel()) / error.size * 1000
                    ),
                    "point_rmse_mm": float(
                        np.sqrt(
                            np.einsum("tnc,tnc->", error, error)
                            / np.prod(error.shape[:2])
                        )
                        * 1000
                    ),
                    "fde_mm": float(
                        np.sqrt((error[-1] ** 2).sum(axis=1)).mean() * 1000
                    ),
                }
                recorded = result["per_case"][arm][index]
                recorded = recorded if label == "all" else recorded[label]
                for metric, value in values.items():
                    np.testing.assert_allclose(
                        value, recorded[metric], rtol=1e-12, atol=1e-12
                    )
                if label == "all":
                    row.update(values)
                else:
                    row[label] = values
            rows[arm].append(row)
    for arm, case_rows in rows.items():
        summary = result["summaries"][arm]
        for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
            candidate = np.array([case_rows[i][metric] for i in keep])
            base = np.array([rows["incumbent"][i][metric] for i in keep])
            delta = candidate - base
            np.testing.assert_allclose(candidate.mean(), summary[metric], rtol=1e-12)
            interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            np.testing.assert_allclose(
                interval, summary[metric + "_delta_ci95"], rtol=1e-11, atol=1e-12
            )
        for label in ("early", "middle", "late"):
            for metric, recorded in summary[label].items():
                np.testing.assert_allclose(
                    np.mean([case_rows[i][label][metric] for i in keep]),
                    recorded,
                    rtol=1e-12,
                )
    return {
        "schema": "deform-state-restart-metric-verification-v1",
        "passed": True,
        "prediction_arrays_verified": len(predictions),
        "case_predictions_verified": len(names) * len(predictions),
        "reported_trajectory_count": len(keep),
        "barrier_sha256": file_digest(run / "prediction_barrier.json"),
        "result_sha256": file_digest(run / "result.json"),
        "independent_metric_recomputation": True,
        "independent_native_physics_reexecution": False,
        "native_controls_basis": "hash-bound runner record plus source and tensor-state tests",
        "protected_cohort_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.archive, args.protocol)
    write_json_once(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
