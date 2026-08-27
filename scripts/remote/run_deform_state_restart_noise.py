#!/usr/bin/env python3
"""Frozen simulated-noise follow-up to the opened DEFORM state-restart test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_sparse_state_restart as native

from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    array_digest,
    file_digest,
    paired_physical_readout,
    prediction_metrics,
    sparse_state_increments,
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_state_restart_noise_dev_v1.json"


def observation_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    independent_std_m: float,
    shared_std_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(shape) != 4 or shape[1:] != (2, 4, 3):
        raise ValueError("noise contract requires eight 3D observations per case")
    if not all(np.isfinite(x) and x >= 0 for x in (independent_std_m, shared_std_m)):
        raise ValueError("noise scales must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    independent = rng.normal(0, independent_std_m, shape)
    shared = rng.normal(0, shared_std_m, (shape[0], 1, 1, 3))
    return independent, independent + shared


def freeze(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit the robustness experiment before freezing")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            PROTOCOL,
            native.PROTOCOL,
            "scripts/remote/run_deform_sparse_state_restart.py",
            "scripts/remote/run_deform_state_restart_noise.py",
            "scripts/remote/run_deform_dlo_source.py",
            "scripts/remote/run_deform_dlo_longrun_posterior.py",
            "scripts/verify_deform_state_restart.py",
            "tests/test_deform_state_restart.py",
            "tests/test_deform_state_restart_noise.py",
            "docs/deform_state_restart_noise_dev_v1.md",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deform-state-restart-source-receipt-v1",
            "revision": revision,
            "git_clean": True,
            "files": {p: file_digest(ROOT / p) for p in paths},
            "data_read_during_freeze": False,
            "parent_results_already_opened": True,
        },
    )
    print(json.dumps({"source_revision": revision, "bound_files": len(paths)}))


def summarize_noise(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    names: list[str],
    config: RestartConfig,
) -> dict[str, Any]:
    keep = [i for i, name in enumerate(names) if name != config.design_case]
    per_case: dict[str, list[dict[str, Any]]] = {}
    shape = next(iter(predictions.values())).shape
    if (
        len(shape) != 5
        or shape[1:] != truth.shape
        or any(v.shape != shape for v in predictions.values())
    ):
        raise ValueError(
            "noise forecasts must be (repetition, case, time, identity, 3)"
        )
    for arm, values in predictions.items():
        rows = []
        for case, name in enumerate(names):
            scores = [
                prediction_metrics(
                    v[case][:, config.hidden_nodes], truth[case][:, config.hidden_nodes]
                )
                for v in values
            ]
            row: dict[str, Any] = {"case": name}
            for metric in scores[0]:
                row[metric] = float(np.mean([s[metric] for s in scores]))
            row["noise_repetition_metrics"] = scores
            rows.append(row)
        per_case[arm] = rows
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, len(keep), size=(config.bootstrap_replicates, len(keep)))
    summaries = {}
    for arm, rows in per_case.items():
        summary: dict[str, Any] = {
            "case_count": len(keep),
            "noise_repetitions": shape[0],
        }
        for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
            values = np.array([rows[i][metric] for i in keep])
            base = np.array([per_case["incumbent"][i][metric] for i in keep])
            summary[metric] = float(values.mean())
            summary[metric + "_change_percent"] = float(
                100 * (values.mean() / base.mean() - 1)
            )
            summary[metric + "_delta_ci95"] = np.quantile(
                (values - base)[indices].mean(axis=1), [0.025, 0.975]
            ).tolist()
            summary[metric + "_wins"] = int(np.sum(values < base))
        summary["joint_wins"] = sum(
            rows[i]["coordinate_l1_mm"] < per_case["incumbent"][i]["coordinate_l1_mm"]
            and rows[i]["point_rmse_mm"] < per_case["incumbent"][i]["point_rmse_mm"]
            for i in keep
        )
        summaries[arm] = summary
    return {
        "per_case": per_case,
        "summaries": summaries,
        "bootstrap_unit": "whole-trajectory-after-averaging-noise-repetitions",
    }


def run(args: argparse.Namespace) -> None:
    receipt = native.verify_source(args.source_receipt, args.source_receipt_sha256)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    base_protocol = json.loads((ROOT / native.PROTOCOL).read_text())
    config = RestartConfig()
    for name, key in (
        ("result.json", "parent_result_sha256"),
        ("prediction_barrier.json", "parent_barrier_sha256"),
        ("predictions.npz", "parent_predictions_sha256"),
    ):
        if file_digest(args.parent_run / name) != protocol[key]:
            raise ValueError(f"parent artifact changed: {name}")
    parent = json.loads((args.parent_run / "result.json").read_text())
    if (
        parent["source_revision"] != protocol["parent_source_revision"]
        or parent["controls"]["passed"] is not True
    ):
        raise ValueError("parent source/control gate differs")
    for path, expected in (
        (args.archive, base_protocol["source_archive_sha256"]),
        (args.checkpoint, base_protocol["checkpoint_sha256"]),
        (args.evaluation_manifest, args.evaluation_manifest_sha256),
    ):
        if file_digest(path) != expected:
            raise ValueError("a frozen input digest differs")
    args.output.mkdir(parents=True, exist_ok=False)
    stage = "native-control"
    started = time.perf_counter()
    try:
        import run_deform_dlo_source as source
        import torch

        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or str(torch.__version__) != "2.0.1+cu118"
        ):
            raise ValueError("frozen CPU runtime differs")
        torch.set_num_threads(1)
        source._seed_everything(torch, config.seed)
        upstream = source._assert_upstream(
            args.upstream_root, base_protocol["upstream_commit"]
        )
        modules = source._load_upstream(args.upstream_root)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        manifest = json.loads(args.evaluation_manifest.read_text())
        names = manifest["ordered_names"]
        if names != base_protocol["expected_names"] or set(
            manifest["trajectories"]
        ) != set(names):
            raise ValueError("opened roster differs")
        data = source._load_named_trajectories(
            manifest, names, frame_count=500, node_count=12
        )
        initial, actions = native.causal_model_inputs(
            np.stack([data[name] for name in names]), config
        )
        with np.load(args.archive, allow_pickle=False) as archive:
            if archive["names"].tolist() != names:
                raise ValueError("archive identity order differs")
            prefix = archive["candidate_predictions"][:, : config.prefix_length].copy()
            base = archive["candidate_predictions"][
                :, config.prefix_length : config.forecast_end
            ].copy()
            observed = archive["targets"][:, config.observation_frames][
                :, :, config.observed_nodes
            ].copy()
        with np.load(
            args.parent_run / "predictions.npz", allow_pickle=False
        ) as archive:
            parent_nominal = archive["physical_nominal"].copy()
        with torch.no_grad():
            rod = native.NativeRod(
                modules, torch, checkpoint["model_state_dict"], config
            )
            _, snapshot = rod.rollout(
                rod.initialize(initial), actions[:, : config.prefix_length]
            )
            future_actions = actions[:, config.prefix_length :]
            nominal, _ = rod.rollout(snapshot.clone(), future_actions)
            if array_digest(nominal) != array_digest(parent_nominal):
                raise ValueError("parent native continuation bytes changed")
            zeros = torch.zeros_like(snapshot.positions)
            zero_state = update_rod_state(
                snapshot, zeros, zeros, gain=0, clamped_nodes=config.clamped_nodes
            )
            zero_future, _ = rod.rollout(zero_state, future_actions)
            if (
                array_digest(zero_future) != array_digest(nominal)
                or paired_physical_readout(base, nominal, zero_future) is not base
            ):
                raise ValueError("exact no-update path changed")
            write_json_once(
                args.output / "preflight.json",
                {
                    "source_revision": receipt["revision"],
                    "source_receipt_sha256": file_digest(args.source_receipt),
                    "protocol_sha256": file_digest(ROOT / PROTOCOL),
                    "upstream": upstream,
                    "parent_barrier_sha256": protocol["parent_barrier_sha256"],
                    "native_parent_replay_byte_identical": True,
                    "zero_update_byte_identical": True,
                    "protected_cohort_read": False,
                    "original_results_modified": False,
                },
            )
            stage = "noise-predictions"
            collected: dict[str, dict[str, list[np.ndarray]]] = {
                condition: {arm: [] for arm in protocol["arms"]}
                for condition in protocol["conditions"]
            }
            noise_hashes = []
            for repeat in range(protocol["repetitions"]):
                noises = observation_noise(
                    observed.shape,
                    seed=protocol["seed"] + repeat,
                    independent_std_m=protocol["independent_noise_std_m"],
                    shared_std_m=protocol["shared_bias_std_m"],
                )
                for condition, noise in zip(
                    protocol["conditions"], noises, strict=True
                ):
                    dx, dv = sparse_state_increments(prefix, observed + noise, config)
                    group = collected[condition]
                    group["incumbent"].append(base)
                    group["readout_sparse_pose"].append(base + dx[:, None])
                    for arm, gain in zip(
                        protocol["arms"][2:], protocol["gains"], strict=True
                    ):
                        updated = update_rod_state(
                            snapshot,
                            torch.tensor(dx, dtype=torch.float32),
                            torch.tensor(dv, dtype=torch.float32),
                            gain=gain,
                            clamped_nodes=config.clamped_nodes,
                        )
                        points, _ = rod.rollout(updated, future_actions)
                        group[arm].append(
                            paired_physical_readout(base, nominal, points)
                        )
                    noise_hashes.append(
                        {
                            "condition": condition,
                            "repetition": repeat,
                            "noise_sha256": array_digest(noise),
                        }
                    )
                print(
                    json.dumps({"stage": stage, "repetitions_complete": repeat + 1}),
                    flush=True,
                )
        sealed = {}
        for condition, groups in collected.items():
            values = {arm: np.stack(points) for arm, points in groups.items()}
            hashes = native._save_predictions(args.output / f"{condition}.npz", values)
            sealed[condition] = {
                "file_sha256": file_digest(args.output / f"{condition}.npz"),
                "array_digests": hashes,
            }
        write_json_once(
            args.output / "prediction_barrier.json",
            {
                "schema": "deform-state-restart-noise-barrier-v1",
                "source_revision": receipt["revision"],
                "case_count": len(names),
                "noise_repetitions": protocol["repetitions"],
                "conditions": sealed,
                "noise_hashes": noise_hashes,
                "new_noise_metrics_computed": False,
                "protected_cohort_read": False,
                "preflight_sha256": file_digest(args.output / "preflight.json"),
            },
        )
        stage = "noise-metrics"
        with np.load(args.archive, allow_pickle=False) as archive:
            truth = archive["targets"][
                :, config.prefix_length : config.forecast_end
            ].copy()
        results = {}
        for condition in protocol["conditions"]:
            with np.load(
                args.output / f"{condition}.npz", allow_pickle=False
            ) as archive:
                values = {arm: archive[arm].copy() for arm in protocol["arms"]}
            results[condition] = summarize_noise(values, truth, names, config)
        write_json_once(
            args.output / "result.json",
            {
                "schema": "deform-state-restart-noise-result-v1",
                "conditions": results,
                "source_revision": receipt["revision"],
                "scope": protocol["scope"],
                "prediction_barrier_sha256": file_digest(
                    args.output / "prediction_barrier.json"
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "fresh_confirmation_claim": False,
            },
        )
        print(
            json.dumps(
                {"stage": "complete", "elapsed_seconds": time.perf_counter() - started}
            ),
            flush=True,
        )
    except Exception as error:
        write_json_once(
            args.output / "failure.json",
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "protected_cohort_read": False,
                "retained_failure": True,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-source-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-receipt-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--evaluation-manifest-sha256")
    args = parser.parse_args()
    if args.freeze_source_only:
        freeze(args.output)
    else:
        for field in (
            "parent_run",
            "source_receipt",
            "source_receipt_sha256",
            "archive",
            "checkpoint",
            "upstream_root",
            "evaluation_manifest",
            "evaluation_manifest_sha256",
        ):
            if getattr(args, field) is None:
                parser.error(f"--{field.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
