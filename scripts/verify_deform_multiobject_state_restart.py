#!/usr/bin/env python3
"""Independently recompute matched multi-object metrics and sealed controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]


def metrics(points: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    if points.ndim != 5 or points.shape[1:] != truth.shape:
        raise ValueError("metric arrays must align in repetition/case/time/identity")
    if not np.isfinite(points).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite values cannot be excluded")
    error = points.astype(np.float64) - truth.astype(np.float64)[None]
    distance2 = np.einsum("rbthc,rbthc->rbth", error, error)
    return {
        "coordinate_l1_mm": 1000 * np.abs(error).mean(axis=(2, 3, 4)),
        "point_rmse_mm": 1000 * np.sqrt(distance2.mean(axis=(2, 3))),
        "fde_mm": 1000 * np.sqrt(distance2[:, :, -1]).mean(axis=2),
    }


def verify_rows(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    recorded: dict[str, Any],
    keep: list[int],
    hidden: list[int],
    protocol: dict[str, Any],
) -> int:
    arrays = {k: v[None] if v.ndim == 4 else v for k, v in predictions.items()}
    expected = {
        arm: metrics(v[:, :, :, hidden], truth[:, :, hidden])
        for arm, v in arrays.items()
    }
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(
        0, len(keep), size=(protocol["bootstrap_replicates"], len(keep))
    )
    for arm, values in expected.items():
        summary = recorded["summaries"][arm]
        assert summary["case_count"] == len(keep)
        assert summary["noise_repetitions"] == len(arrays[arm])
        for key, repeated in values.items():
            rows = repeated.mean(axis=0)
            reference = expected["incumbent"][key].mean(axis=0)[keep]
            selected = rows[keep]
            np.testing.assert_allclose(
                rows,
                [r[key] for r in recorded["per_case"][arm]],
                atol=1e-11,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                selected.mean(), summary[key], atol=1e-11, rtol=1e-12
            )
            np.testing.assert_allclose(
                100 * (selected.mean() / reference.mean() - 1),
                summary[key + "_change_percent"],
                atol=1e-10,
            )
            np.testing.assert_allclose(
                np.max(selected / reference),
                summary[key + "_worst_case_ratio"],
                atol=1e-12,
            )
            interval = np.quantile(
                (selected - reference)[draws].mean(axis=1), [0.025, 0.975]
            )
            np.testing.assert_allclose(
                interval, summary[key + "_delta_ci95"], atol=1e-11, rtol=1e-11
            )
            assert int(np.sum(selected - reference < -1e-10)) == summary[key + "_wins"]
        joint = np.logical_and(
            values["coordinate_l1_mm"].mean(axis=0)[keep]
            < expected["incumbent"]["coordinate_l1_mm"].mean(axis=0)[keep],
            values["point_rmse_mm"].mean(axis=0)[keep]
            < expected["incumbent"]["point_rmse_mm"].mean(axis=0)[keep],
        )
        assert int(joint.sum()) == summary["joint_wins"]
        for label, frames in zip(
            ("early", "middle", "late"),
            np.array_split(np.arange(truth.shape[1]), 3),
            strict=True,
        ):
            partial = metrics(
                arrays[arm][:, :, frames][:, :, :, hidden],
                truth[:, frames][:, :, hidden],
            )
            for key, repeated in partial.items():
                rows = repeated.mean(axis=0)
                np.testing.assert_allclose(
                    rows,
                    [r[label][key] for r in recorded["per_case"][arm]],
                    atol=1e-11,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    rows[keep].mean(), summary[label][key], atol=1e-11, rtol=1e-12
                )
    return sum(v.shape[0] * v.shape[1] for v in arrays.values())


def verify_readout(
    predictions: dict[str, np.ndarray],
    raw: np.ndarray,
    incumbent: np.ndarray,
    item: dict[str, Any],
    protocol: dict[str, Any],
    condition: str,
) -> None:
    observed_nodes = protocol["observed_nodes"]
    frames = (
        np.asarray(protocol["observation_frames"]) + protocol["dataset_frame_offset"]
    )
    observed = raw[:, frames][:, :, observed_nodes].astype(np.float64)
    reference = incumbent[:, : protocol["prefix_length"]]
    base = incumbent[:, protocol["prefix_length"] : protocol["forecast_end"]]
    repeats = 1 if condition == "clean" else protocol["noise"]["repetitions"]
    points = (
        predictions["readout_sparse_pose"][None]
        if condition == "clean"
        else predictions["readout_sparse_pose"]
    )
    base_points = (
        predictions["incumbent"][None]
        if condition == "clean"
        else predictions["incumbent"]
    )
    knots = sorted(observed_nodes + item["clamped_nodes"])
    for repeat in range(repeats):
        assert array_digest(base_points[repeat]) == array_digest(base)
        error = np.zeros_like(observed)
        if condition != "clean":
            noise = protocol["noise"]
            rng = np.random.default_rng(
                noise["seed"] + item["noise_seed_offset"] + repeat
            )
            error = rng.normal(0, noise["independent_std_m"], observed.shape)
            shared = rng.normal(0, noise["shared_std_m"], (len(observed), 1, 1, 3))
            if condition == "independent_1mm_shared_5mm":
                error += shared
        residual = observed[:, -1] + error[:, -1] - reference[:, -1, observed_nodes]
        update = np.zeros((len(observed), item["node_count"], 3))
        for case in range(len(observed)):
            for axis in range(3):
                values = [
                    residual[case, observed_nodes.index(k), axis]
                    if k in observed_nodes
                    else 0.0
                    for k in knots
                ]
                update[case, :, axis] = np.interp(
                    np.arange(item["node_count"]), knots, values
                )
        np.testing.assert_allclose(
            points[repeat], base + update[:, None], atol=1e-14, rtol=1e-13
        )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts/remote"))
    import run_deform_dlo_source as source
    import run_deform_multiobject_state_restart as runner

    protocol = json.loads(args.protocol.read_text())
    receipt = runner.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    barrier = runner.validate_barrier(
        args.run, protocol, file_digest(args.source_receipt)
    )
    result = json.loads((args.run / "result.json").read_text())
    assert result["prediction_barrier_sha256"] == file_digest(
        args.run / "prediction_barrier.json"
    )
    assert (
        result["source_revision"] == receipt["revision"] == barrier["source_revision"]
    )
    assert result["protected_data_access"] is False
    assert result["original_results_modified"] is False
    verified = 0
    for item in protocol["objects"]:
        manifest = runner.verify_input_files(item)
        data = source._load_named_trajectories(
            manifest, item["names"], frame_count=500, node_count=item["node_count"]
        )
        raw = np.stack([data[n] for n in item["names"]])
        truth = raw[:, protocol["prefix_length"] + 2 : protocol["forecast_end"] + 2]
        keep = [
            i for i, n in enumerate(item["names"]) if n != item["excluded_design_case"]
        ]
        with np.load(item["archive"]["path"], allow_pickle=False) as archive:
            incumbent = archive[item["archive"]["incumbent_key"]][
                :, : protocol["forecast_end"]
            ].copy()
        for condition in ("clean", *protocol["noise"]["conditions"]):
            with np.load(
                args.run / item["object"] / f"{condition}.npz", allow_pickle=False
            ) as archive:
                predictions = {k: archive[k] for k in archive.files if k != "names"}
            verify_readout(predictions, raw, incumbent, item, protocol, condition)
            verified += verify_rows(
                predictions,
                truth,
                result["objects"][item["object"]][condition],
                keep,
                protocol["hidden_nodes"],
                protocol,
            )
            if item["object"] == "DLO1":
                hidden = [
                    i
                    for i in range(item["node_count"])
                    if i not in (*protocol["observed_nodes"], *item["clamped_nodes"])
                ]
                verify_rows(
                    predictions,
                    truth,
                    result["dlo1_all_hidden_secondary"][condition],
                    keep,
                    hidden,
                    protocol,
                )
    checks = {}
    for name in protocol["transfer_objects"]:
        rows = result["objects"][name]["clean"]["summaries"]
        candidate, base, readout = (
            rows[k]
            for k in (protocol["primary_arm"], "incumbent", "readout_sparse_pose")
        )
        checks[name] = {
            "coordinate_l1_improves": candidate["coordinate_l1_mm"]
            < base["coordinate_l1_mm"],
            "point_rmse_improves": candidate["point_rmse_mm"] < base["point_rmse_mm"],
            "beats_matched_readout_on_both": candidate["coordinate_l1_mm"]
            < readout["coordinate_l1_mm"]
            and candidate["point_rmse_mm"] < readout["point_rmse_mm"],
            "late_point_rmse_nonincreasing": candidate["late"]["point_rmse_mm"]
            <= base["late"]["point_rmse_mm"],
            "joint_wins": candidate["joint_wins"]
            >= protocol["transfer_gate"]["minimum_joint_wins_per_transfer_object"],
        }
    assert checks == result["assessment"]["checks"]
    assert (
        all(all(v.values()) for v in checks.values())
        == result["assessment"]["primary_transfer_gate_passed"]
    )
    for group, objects in (
        ("transfer_only", ["DLO1", "DLO3"]),
        ("all_three_including_discovery", ["DLO1", "DLO2", "DLO3"]),
    ):
        for condition in ("clean", *protocol["noise"]["conditions"]):
            arms = (
                protocol["clean_arms"]
                if condition == "clean"
                else protocol["noise_arms"]
            )
            for arm in arms:
                rows = [
                    result["objects"][o][condition]["summaries"][arm] for o in objects
                ]
                summary = result["assessment"]["object_balanced"][group][condition][arm]
                for key in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
                    np.testing.assert_allclose(
                        np.mean([r[key] for r in rows]), summary[key], atol=1e-12
                    )
                    np.testing.assert_allclose(
                        np.mean([r[key + "_change_percent"] for r in rows]),
                        summary[key + "_mean_object_change_percent"],
                        atol=1e-12,
                    )
                assert sum(r["case_count"] for r in rows) == summary["case_count"]
                assert sum(r["joint_wins"] for r in rows) == summary["joint_wins"]
    return {
        "schema": "deform-multiobject-state-restart-verification-v1",
        "passed": True,
        "case_predictions_verified": verified,
        "analyzed_trajectories": 29,
        "transfer_trajectories": 16,
        "physical_objects": 3,
        "independent_native_physics_reexecution": False,
        "independent_metric_and_bootstrap_recomputation": True,
        "registered_mean_byte_identity_verified": True,
        "matched_readout_and_noise_recomputed": True,
        "result_sha256": file_digest(args.run / "result.json"),
        "barrier_sha256": file_digest(args.run / "prediction_barrier.json"),
        "protected_data_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    write_json_once(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
