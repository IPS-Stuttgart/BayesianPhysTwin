#!/usr/bin/env python3
"""Freeze, predict, then score one isolated opened-DLO2 source experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_reference_transport import (
    ARMS,
    SCHEMA,
    config_for_source,
    content_id,
    learned_reference_offsets,
    score_predictions,
    transport_pair,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_reference_transport_source_v1.json"
RUNNERS = (
    "scripts/remote/run_deform_reference_transport.py",
    "scripts/remote/run_deform_sparse_state_restart.py",
    "scripts/remote/run_deform_dlo_source.py",
    "scripts/verify_deform_reference_transport.py",
)


def _write(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if "artifact_id" in value:
        raise ValueError("artifact already has an identity")
    value = {**value, "artifact_id": content_id(value)}
    write_json_once(path, value)
    return value


def _read(path: Path, digest: str | None = None) -> dict[str, Any]:
    if digest is not None and file_digest(path) != digest:
        raise ValueError("artifact file digest differs")
    value = json.loads(path.read_text())
    identity = {k: v for k, v in value.items() if k != "artifact_id"}
    if value.get("artifact_id") != content_id(identity):
        raise ValueError("artifact canonical digest differs")
    return dict(value)


def _plan() -> dict[str, Any]:
    plan = json.loads((ROOT / PROTOCOL).read_text())
    config = config_for_source()
    required = {
        "schema": SCHEMA,
        "object": "DLO2",
        "prefix_length": 50,
        "forecast_end": 170,
        "observation_frames": list(config.observation_frames),
        "observed_nodes": list(config.observed_nodes),
        "hidden_nodes": list(config.hidden_nodes),
        "clamped_nodes": list(config.clamped_nodes),
        "gain": 1.0,
        "primary_arm": "reference_centered",
        "arms": list(ARMS),
        "dataset_frame_offset": 2,
        "maximum_native_attempts": 1,
        "future_free_node_truth_in_prediction": False,
        "fresh_transfer_authorized": False,
        "target_access": False,
        "held_v8_access": False,
        "gpu_execution": False,
    }
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError("frozen plan contract changed")
    if len(set(plan["names"])) != 14 or plan["excluded_design_case"] != "103.pkl":
        raise ValueError("source roster differs")
    return dict(plan)


def freeze(output: Path) -> None:
    plan = _plan()
    if str(output.resolve()) != plan["output_root"]:
        raise ValueError("output root is not the registered one-attempt root")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("source must be committed and clean before freeze")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            *RUNNERS,
            PROTOCOL,
            "docs/deform_reference_transport_source_v1.md",
            "tests/test_deform_reference_transport.py",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for label in ("manifest", "checkpoint", "archive", "paired_archive"):
        if file_digest(Path(plan[label]["path"])) != plan[label]["sha256"]:
            raise ValueError(f"frozen {label} input changed")
    output.mkdir(parents=True, exist_ok=False)
    lock = _write(
        output / "lock.json",
        {
            "schema": SCHEMA + "-lock",
            "plan": plan,
            "revision": revision,
            "source_files": {p: file_digest(ROOT / p) for p in paths},
            "source_clean": True,
            "payload_decoded": False,
            "future_truth_scored": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "frozen",
                "lock_id": lock["artifact_id"],
                "lock_sha256": file_digest(output / "lock.json"),
            }
        ),
        flush=True,
    )


def verify_lock(output: Path, digest: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _read(output / "lock.json", digest)
    plan = _plan()
    if (
        lock.get("schema") != SCHEMA + "-lock"
        or lock.get("plan") != plan
        or lock.get("source_clean") is not True
        or str(output.resolve()) != plan["output_root"]
    ):
        raise ValueError("registered source lock or root differs")
    for path, sha in lock["source_files"].items():
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT) or file_digest(resolved) != sha:
            raise ValueError(f"frozen source changed: {path}")
    for label in ("manifest", "checkpoint", "archive", "paired_archive"):
        if file_digest(Path(plan[label]["path"])) != plan[label]["sha256"]:
            raise ValueError(f"frozen {label} input changed")
    return lock, plan


def permitted_inputs(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = config_for_source()
    if raw.shape != (14, 500, config.node_count, 3):
        raise ValueError("source trajectory shape differs")
    initial = raw[:, :2].copy()
    actions = raw[:, 2:172, config.clamped_nodes].copy()
    observations = raw[:, np.asarray(config.observation_frames) + 2][
        :, :, config.observed_nodes
    ].copy()
    if not all(np.isfinite(x).all() for x in (initial, actions, observations)):
        raise ValueError("permitted source inputs are nonfinite")
    return initial, actions, observations


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    payload: dict[str, Any] = dict(arrays)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return {
        "file_sha256": file_digest(path),
        "arrays": {
            name: {
                "sha256": array_digest(value),
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
            for name, value in arrays.items()
        },
    }


def _load_arrays(path: Path, binding: dict[str, Any]) -> dict[str, np.ndarray]:
    if file_digest(path) != binding["file_sha256"]:
        raise ValueError("sealed array archive changed")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(binding["arrays"]):
            raise ValueError("sealed array members differ")
        arrays = {k: data[k].copy() for k in data.files}
    for name, value in arrays.items():
        if binding["arrays"][name] != {
            "sha256": array_digest(value),
            "shape": list(value.shape),
            "dtype": value.dtype.str,
        }:
            raise ValueError("sealed array identity differs")
    return arrays


def predict(output: Path, digest: str) -> None:
    lock, plan = verify_lock(output, digest)
    _write(
        output / "attempt.json",
        {
            "schema": SCHEMA + "-attempt",
            "lock_id": lock["artifact_id"],
            "maximum_attempts": 1,
            "retry_authorized": False,
        },
    )
    started = time.monotonic()
    try:
        _predict(output, lock, plan, started)
    except Exception as exc:
        _write(
            output / "technical_failure.json",
            {
                "schema": SCHEMA + "-technical-failure",
                "lock_id": lock["artifact_id"],
                "error_type": type(exc).__name__,
                "message": str(exc),
                "prediction_barrier_complete": False,
                "future_scoring_authorized": False,
                "locked_case_count": 14,
                "ordinary_successes": 0,
                "retained_technical_failures": 14,
                "retry_authorized": False,
                "future_free_truth_scored": False,
                "target_access": False,
            },
        )
        raise


def _predict(
    output: Path, lock: dict[str, Any], plan: dict[str, Any], started: float
) -> None:
    import run_deform_dlo_source as source
    import run_deform_sparse_state_restart as native
    import torch

    config = config_for_source()
    runtime = plan["runtime"]
    if (
        platform.python_version_tuple()[:2] != tuple(runtime["python"].split("."))
        or torch.__version__ != runtime["torch"]
        or np.__version__ != runtime["numpy"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("exact frozen CPU runtime is required")
    torch.set_num_threads(1)
    torch.manual_seed(runtime["seed"])
    upstream = source._assert_upstream(
        Path(plan["upstream_root"]), plan["upstream_commit"]
    )
    modules = source._load_upstream(Path(plan["upstream_root"]))
    manifest = json.loads(Path(plan["manifest"]["path"]).read_text())
    if manifest["ordered_names"] != plan["names"]:
        raise ValueError("source manifest roster differs")
    data = source._load_named_trajectories(
        manifest, plan["names"], frame_count=500, node_count=12
    )
    raw = np.stack([data[n] for n in plan["names"]])
    initial, actions, observations = permitted_inputs(raw)
    del raw, data
    with np.load(plan["archive"]["path"], allow_pickle=False) as archive:
        if archive["names"].tolist() != plan["names"]:
            raise ValueError("incumbent roster differs")
        incumbent = archive["candidate_predictions"][:, :170].copy()
        archived_physical = archive["baseline_predictions"][:, :170].copy()
    with np.load(plan["paired_archive"]["path"], allow_pickle=False) as archive:
        if archive["names"].tolist() != plan["names"]:
            raise ValueError("previous paired roster differs")
        parent_base = archive["incumbent"].copy()
        parent_paired = archive["incumbent_propagated_pose_velocity"].copy()
    checkpoint = torch.load(
        plan["checkpoint"]["path"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    offsets, offset_velocities = learned_reference_offsets(
        incumbent, archived_physical, config
    )
    dx, dv = sparse_state_increments(incumbent[:, :50], observations, config)
    base = incumbent[:, 50:]
    with torch.no_grad():
        rod = native.NativeRod(modules, torch, checkpoint, config)
        state = rod.initialize(initial)
        states = []
        for frame in range(170):
            state = rod.advance(state, actions[:, frame])
            states.append(state.clone())
        nominal = np.stack([s.positions.numpy().copy() for s in states], axis=1)
        velocities = np.stack([s.velocity.numpy().copy() for s in states], axis=1)
        snapshot, future_actions = states[49], actions[:, 50:]
        tx, tv = (
            torch.tensor(dx, dtype=torch.float32),
            torch.tensor(dv, dtype=torch.float32),
        )
        updated = update_rod_state(
            snapshot, tx, tv, gain=1, clamped_nodes=config.clamped_nodes
        )
        paired_future, _ = rod.rollout(updated, future_actions)
        paired = paired_physical_readout(base, nominal[:, 50:], paired_future)
        offset_args = {
            "position_offsets": torch.tensor(offsets, dtype=torch.float32),
            "velocity_offsets": torch.tensor(offset_velocities, dtype=torch.float32),
        }
        common = dict(
            advance=rod.advance,
            nominal_states=states[49:],
            future_actions=future_actions,
            incumbent=base,
            pose_increment=tx,
            velocity_increment=tv,
            clamped_nodes=config.clamped_nodes,
        )
        zero_reference, zero_trace = transport_pair(
            **common,
            position_offsets=torch.zeros_like(offset_args["position_offsets"]),
            velocity_offsets=torch.zeros_like(offset_args["velocity_offsets"]),
            mode="reference_centered",
        )
        noop_common = {
            **common,
            "pose_increment": torch.zeros_like(tx),
            "velocity_increment": torch.zeros_like(tv),
        }
        noop, _ = transport_pair(
            **noop_common, **offset_args, mode="reference_centered"
        )
        replay = nominal.astype(np.float64) - archived_physical
        checks = {
            "incumbent_byte_identical": array_digest(base) == array_digest(parent_base),
            "paired_byte_identical": array_digest(paired)
            == array_digest(parent_paired),
            "zero_reference_byte_identical": array_digest(zero_reference)
            == array_digest(paired),
            "zero_innovation_returns_original": noop is base,
            "archived_gpu_replay_max_pass": np.max(np.abs(replay)) <= 0.002,
            "archived_gpu_replay_rmse_pass": np.sqrt(np.mean(replay**2)) <= 0.0002,
        }
        controls = _write(
            output / "controls.json",
            {
                "schema": SCHEMA + "-controls",
                "checks": {k: bool(v) for k, v in checks.items()},
                "passed": bool(all(checks.values())),
                "upstream": upstream,
                "archived_gpu_replay_max_error_m": float(np.max(np.abs(replay))),
                "archived_gpu_replay_coordinate_rmse_m": float(
                    np.sqrt(np.mean(replay**2))
                ),
            },
        )
        print(
            json.dumps(
                {
                    "stage": "controls",
                    "passed": controls["passed"],
                    "checks": controls["checks"],
                }
            ),
            flush=True,
        )
        if not controls["passed"]:
            raise ValueError("native source qualification failed; no retry")
        arrays = {
            "names": np.asarray(plan["names"]),
            "incumbent": base,
            "paired": paired,
            "nominal": nominal,
            "nominal_velocity": velocities,
            "offsets": offsets,
            "offset_velocities": offset_velocities,
            "pose_increment": dx,
            "velocity_increment": dv,
            "future_actions": future_actions,
            "zero_reference": zero_reference,
        }
        arrays.update({"zero_reference__" + k: v for k, v in zero_trace.items()})
        for mode in ARMS[2:]:
            prediction, trace = transport_pair(**common, **offset_args, mode=mode)
            arrays[mode] = prediction
            arrays.update({mode + "__" + k: v for k, v in trace.items()})
            print(
                json.dumps({"stage": "prediction", "arm": mode, "cases": 14}),
                flush=True,
            )
        binding = _save_arrays(output / "predictions.npz", arrays)
    seal = _write(
        output / "prediction_seal.json",
        {
            "schema": SCHEMA + "-prediction-seal",
            "lock_id": lock["artifact_id"],
            "names": plan["names"],
            "ordinary_successes": 14,
            "retained_technical_failures": 0,
            "unsealable": 0,
            "locked_case_count": 14,
            "replacements": 0,
            "controls_sha256": file_digest(output / "controls.json"),
            "controls_id": controls["artifact_id"],
            "predictions": binding,
            "complete": True,
            "source_future_scoring_authorized": True,
            "future_truth_scored": False,
            "future_truth_in_prediction": False,
            "fresh_transfer_authorized": False,
            "target_access": False,
            "held_v8_access": False,
            "runtime": runtime,
            "elapsed_s": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                "stage": "sealed",
                "seal_id": seal["artifact_id"],
                "seal_sha256": file_digest(output / "prediction_seal.json"),
                "ordinary_successes": 14,
                "future_truth_scored": False,
            }
        ),
        flush=True,
    )


def score(output: Path, lock_digest: str, seal_digest: str) -> None:
    lock, plan = verify_lock(output, lock_digest)
    if (output / "technical_failure.json").exists():
        raise ValueError("technical failure is terminal")
    seal = _read(output / "prediction_seal.json", seal_digest)
    controls = _read(output / "controls.json", seal["controls_sha256"])
    if (
        seal.get("schema") != SCHEMA + "-prediction-seal"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("names") != plan["names"]
        or seal.get("complete") is not True
        or seal.get("ordinary_successes") != 14
        or seal.get("retained_technical_failures") != 0
        or seal.get("unsealable") != 0
        or seal.get("replacements") != 0
        or controls.get("artifact_id") != seal.get("controls_id")
        or set(controls.get("checks", {}))
        != {
            "incumbent_byte_identical",
            "paired_byte_identical",
            "zero_reference_byte_identical",
            "zero_innovation_returns_original",
            "archived_gpu_replay_max_pass",
            "archived_gpu_replay_rmse_pass",
        }
        or not all(v is True for v in controls["checks"].values())
    ):
        raise ValueError("complete qualified source prediction barrier required")
    arrays = _load_arrays(output / "predictions.npz", seal["predictions"])
    if arrays["names"].tolist() != plan["names"]:
        raise ValueError("prediction roster differs")
    # This is the first metric-stage load of future free-node truth.
    with np.load(plan["archive"]["path"], allow_pickle=False) as archive:
        truth = archive["targets"][:, 50:170].copy()
    scores = score_predictions(
        plan["names"], {k: arrays[k] for k in ARMS}, truth, config_for_source()
    )
    result = _write(
        output / "result.json",
        {
            "schema": SCHEMA + "-result",
            "lock_id": lock["artifact_id"],
            "prediction_seal_id": seal["artifact_id"],
            "prediction_seal_sha256": seal_digest,
            "source_implementation_revision": lock["revision"],
            "source_cases_already_opened": True,
            "prediction_case_count": 14,
            "analysis_case_count": 13,
            "ordinary_successes": 14,
            "retained_technical_failures": 0,
            "unsealable": 0,
            "native_controls_passed": True,
            **scores,
            "existing_deform_unchanged": True,
            "target_access": False,
            "held_v8_access": False,
            "fresh_transfer_authorized": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "scored",
                "result_id": result["artifact_id"],
                "decision": result["decision"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "predict", "score"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lock-sha256")
    parser.add_argument("--prediction-seal-sha256")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output_root)
    elif not args.lock_sha256:
        parser.error("predict/score requires exact lock SHA-256")
    elif args.stage == "predict":
        predict(args.output_root, args.lock_sha256)
    elif not args.prediction_seal_sha256:
        parser.error("score requires exact complete prediction seal SHA-256")
    else:
        score(args.output_root, args.lock_sha256, args.prediction_seal_sha256)


if __name__ == "__main__":
    main()
