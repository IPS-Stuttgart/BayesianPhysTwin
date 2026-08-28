#!/usr/bin/env python3
"""One frozen source-only hard-position boundary experiment; CPU only."""

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
import run_deform_reference_transport as custody

from bayesian_phystwin_experiments.deform_kinematic_boundary import (
    ARMS,
    CLAMPS,
    SCHEMA,
    config_for_source,
    hard_boundary_readout,
    install_hard_position_projection,
    score_predictions,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_kinematic_boundary_source_v1.json"
CHECKS = {
    "incumbent_byte_identical",
    "paired_byte_identical",
    "native_rest_initialization_byte_identical",
    "hard_restart_byte_identical",
    "hard_clamps_exact_float32_commands",
    "hard_updated_clamps_exact_float32_commands",
    "disabled_readout_returns_original",
    "zero_innovation_returns_hard_baseline",
    "archived_gpu_replay_max_pass",
    "archived_gpu_replay_rmse_pass",
}
BOUND_FILES = (
    PROTOCOL,
    "docs/deform_kinematic_boundary_source_v1.md",
    "scripts/remote/run_deform_kinematic_boundary.py",
    "scripts/remote/run_deform_reference_transport.py",
    "scripts/remote/run_deform_sparse_state_restart.py",
    "scripts/remote/run_deform_dlo_source.py",
    "scripts/verify_deform_kinematic_boundary.py",
    "tests/test_deform_kinematic_boundary.py",
    "results/sota/deform_kinematic_boundary_source_v1/control-only-diagnostic.json",
)


def _plan() -> dict[str, Any]:
    plan = json.loads((ROOT / PROTOCOL).read_text())
    previous = custody._plan()
    shared = (
        "object",
        "names",
        "excluded_design_case",
        "runtime",
        "upstream_root",
        "upstream_commit",
        "manifest",
        "checkpoint",
        "archive",
        "paired_archive",
        "prefix_length",
        "forecast_end",
        "observation_frames",
        "observed_nodes",
        "hidden_nodes",
        "clamped_nodes",
        "dataset_frame_offset",
        "gain",
        "maximum_native_attempts",
        "future_free_node_truth_in_prediction",
        "source_cases_already_opened",
        "fresh_transfer_authorized",
        "target_access",
        "held_v8_access",
        "new_recordings",
        "gpu_execution",
        "publication",
    )
    expected = {key: previous[key] for key in shared}
    expected.update(
        schema=SCHEMA,
        primary_arm="hard_paired",
        arms=list(ARMS),
        projection_iterations=10,
        projection_change="skip_edge_only_if_both_endpoints_are_prescribed",
        prescribed_segment_inextensibility_claimed=False,
        readout="hard_native_plus_unchanged_archived_incumbent_minus_archived_native",
        sparse_innovation_reference="hard_baseline_permitted_prefix",
        output_root="/home/florianpfaff/source-only/deform-kinematic-boundary-source-v1/run-v1",
        source_gate={
            **previous["source_gate"],
            "sparse_update_must_beat_hard_baseline_l1_and_rmse": True,
        },
        controls={
            **{key: True for key in CHECKS if not key.startswith("archived_gpu")},
            "archived_gpu_replay_max_error_m": 0.002,
            "archived_gpu_replay_coordinate_rmse_m": 0.0002,
        },
    )
    if plan != expected:
        raise ValueError("the complete frozen source plan differs")
    return dict(plan)


def freeze(output: Path) -> None:
    plan = _plan()
    if str(output.resolve()) != plan["output_root"]:
        raise ValueError("only the registered one-attempt root is allowed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit clean source before freezing")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-files", "src", *BOUND_FILES], cwd=ROOT, text=True
    ).splitlines()
    if not set(BOUND_FILES).issubset(paths):
        raise ValueError("source/protocol/test/verifier binding is incomplete")
    for label in ("manifest", "checkpoint", "archive", "paired_archive"):
        if file_digest(Path(plan[label]["path"])) != plan[label]["sha256"]:
            raise ValueError("registered source input differs")
    output.mkdir(parents=True, exist_ok=False)
    lock = custody._write(
        output / "lock.json",
        {
            "schema": SCHEMA + "-lock",
            "revision": revision,
            "plan": plan,
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
    lock = custody._read(output / "lock.json", digest)
    plan = _plan()
    if (
        lock.get("schema") != SCHEMA + "-lock"
        or lock.get("source_clean") is not True
        or lock.get("plan") != plan
        or str(output.resolve()) != plan["output_root"]
        or not set(BOUND_FILES).issubset(lock.get("source_files", {}))
    ):
        raise ValueError("source lock or output root differs")
    for name, expected in lock["source_files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_digest(path) != expected:
            raise ValueError("frozen source bytes differ")
    for label in ("manifest", "checkpoint", "archive", "paired_archive"):
        if file_digest(Path(plan[label]["path"])) != plan[label]["sha256"]:
            raise ValueError("frozen source input differs")
    return lock, plan


def predict(output: Path, digest: str) -> None:
    lock, plan = verify_lock(output, digest)
    custody._write(
        output / "attempt.json",
        {
            "schema": SCHEMA + "-attempt",
            "lock_id": lock["artifact_id"],
            "maximum_attempts": 1,
            "retry_authorized": False,
        },
    )
    try:
        _predict(output, lock, plan)
    except Exception as exc:
        custody._write(
            output / "technical_failure.json",
            {
                "schema": SCHEMA + "-technical-failure",
                "lock_id": lock["artifact_id"],
                "error_type": type(exc).__name__,
                "message": str(exc),
                "qualified_prediction_barrier_complete": False,
                "ordinary_successes": 0,
                "retained_technical_failures": 14,
                "locked_case_count": 14,
                "future_scoring_authorized": False,
                "retry_authorized": False,
                "future_free_truth_scored": False,
                "target_access": False,
            },
        )
        raise


def _predict(output: Path, lock: dict[str, Any], plan: dict[str, Any]) -> None:
    import run_deform_dlo_source as source
    import run_deform_sparse_state_restart as native
    import torch

    started = time.monotonic()
    config, runtime = config_for_source(), plan["runtime"]
    if (
        platform.python_version_tuple()[:2] != tuple(runtime["python"].split("."))
        or torch.__version__ != runtime["torch"]
        or np.__version__ != runtime["numpy"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("exact frozen CPU runtime required")
    torch.set_num_threads(1)
    torch.manual_seed(runtime["seed"])
    upstream = source._assert_upstream(
        Path(plan["upstream_root"]), plan["upstream_commit"]
    )
    modules = source._load_upstream(Path(plan["upstream_root"]))
    manifest = json.loads(Path(plan["manifest"]["path"]).read_text())
    if manifest["ordered_names"] != plan["names"]:
        raise ValueError("source manifest roster differs")
    raw_by_name = source._load_named_trajectories(
        manifest, plan["names"], frame_count=500, node_count=12
    )
    initial, actions, observations = custody.permitted_inputs(
        np.stack([raw_by_name[n] for n in plan["names"]])
    )
    del raw_by_name
    with np.load(plan["archive"]["path"], allow_pickle=False) as data:
        if data["names"].tolist() != plan["names"]:
            raise ValueError("incumbent roster differs")
        incumbent = data["candidate_predictions"][:, :170].copy()
        archived = data["baseline_predictions"][:, :170].copy()
    with np.load(plan["paired_archive"]["path"], allow_pickle=False) as data:
        if data["names"].tolist() != plan["names"]:
            raise ValueError("paired reference roster differs")
        old_base = data["incumbent"].copy()
        old_paired = data["incumbent_propagated_pose_velocity"].copy()
    checkpoint = torch.load(
        plan["checkpoint"]["path"], map_location="cpu", weights_only=True
    )["model_state_dict"]

    def trajectory(rod: Any, *, hard_boundaries: bool = False) -> list[Any]:
        state = rod.initialize(initial)
        if hard_boundaries:
            install_hard_position_projection(rod.model, enabled=True)
        states = []
        for frame in range(170):
            state = rod.advance(state, actions[:, frame])
            states.append(state.clone())
        return states

    def points(states: list[Any], field: str = "positions") -> np.ndarray:
        return np.stack([getattr(s, field).numpy().copy() for s in states], axis=1)

    def inject(state: Any, dx: np.ndarray, dv: np.ndarray) -> Any:
        return update_rod_state(
            state,
            torch.tensor(dx, dtype=torch.float32),
            torch.tensor(dv, dtype=torch.float32),
            gain=1,
            clamped_nodes=CLAMPS,
        )

    with torch.no_grad():
        original = native.NativeRod(modules, torch, checkpoint, config)
        states = trajectory(original)
        nominal = points(states)
        dx, dv = sparse_state_increments(incumbent[:, :50], observations, config)
        updated, _ = original.rollout(inject(states[49], dx, dv), actions[:, 50:])
        base = incumbent[:, 50:]
        paired = paired_physical_readout(base, nominal[:, 50:], updated)
        print(
            json.dumps({"stage": "unchanged-native-complete", "cases": 14}), flush=True
        )
        hard = native.NativeRod(modules, torch, checkpoint, config)
        hard_states = trajectory(hard, hard_boundaries=True)
        hard_native = points(hard_states)
        hard_full = hard_boundary_readout(
            incumbent, archived, hard_native, enabled=True
        )
        hx, hv = sparse_state_increments(hard_full[:, :50], observations, config)
        hard_endpoint = inject(hard_states[49], hx, hv)
        hard_updated, _ = hard.rollout(hard_endpoint, actions[:, 50:])
        hard_zero, _ = hard.rollout(hard_states[49].clone(), actions[:, 50:])
        hard_base = hard_full[:, 50:]
        hard_paired = paired_physical_readout(
            hard_base, hard_native[:, 50:], hard_updated
        )
        noop = paired_physical_readout(hard_base, hard_native[:, 50:], hard_zero)
        replay = nominal.astype(float) - archived.astype(float)
        expected_actions = actions.astype(np.float32)
        checks = {
            "incumbent_byte_identical": array_digest(base) == array_digest(old_base),
            "paired_byte_identical": array_digest(paired) == array_digest(old_paired),
            "native_rest_initialization_byte_identical": all(
                array_digest(getattr(hard.model, name).detach().numpy())
                == array_digest(getattr(original.model, name).detach().numpy())
                for name in ("m_restWprev", "m_restWnext", "learned_pmass")
            ),
            "hard_restart_byte_identical": array_digest(hard_zero)
            == array_digest(hard_native[:, 50:]),
            "hard_clamps_exact_float32_commands": array_digest(
                hard_native[:, :, CLAMPS]
            )
            == array_digest(expected_actions),
            "hard_updated_clamps_exact_float32_commands": array_digest(
                hard_updated[:, :, CLAMPS]
            )
            == array_digest(expected_actions[:, 50:]),
            "disabled_readout_returns_original": hard_boundary_readout(
                incumbent, archived, hard_native
            )
            is incumbent,
            "zero_innovation_returns_hard_baseline": noop is hard_base,
            "archived_gpu_replay_max_pass": bool(np.max(np.abs(replay)) <= 0.002),
            "archived_gpu_replay_rmse_pass": bool(
                np.sqrt(np.mean(replay**2)) <= 0.0002
            ),
        }
        controls = custody._write(
            output / "controls.json",
            {
                "schema": SCHEMA + "-controls",
                "checks": checks,
                "passed": all(checks.values()),
                "upstream": upstream,
                "archived_gpu_replay_max_error_m": float(np.max(np.abs(replay))),
                "archived_gpu_replay_coordinate_rmse_m": float(
                    np.sqrt(np.mean(replay**2))
                ),
                "hard_boundary_maximum_coordinate_error_m": float(
                    np.abs(
                        hard_native[:, :, CLAMPS].astype(float)
                        - expected_actions.astype(float)
                    ).max()
                ),
                "prescribed_segment_inextensibility_claimed": False,
            },
        )
        print(
            json.dumps(
                {"stage": "controls", "passed": controls["passed"], "checks": checks}
            ),
            flush=True,
        )
        if not controls["passed"]:
            raise ValueError("native qualification failed; no retry or scoring")
        arrays = {
            "names": np.array(plan["names"]),
            "incumbent": base,
            "paired": paired,
            "hard_baseline": hard_base,
            "hard_paired": hard_paired,
            "incumbent_full": incumbent,
            "archived_native": archived,
            "native": nominal,
            "native_updated": updated,
            "hard_native": hard_native,
            "hard_updated": hard_updated,
            "hard_zero": hard_zero,
            "known_actions": actions,
            "sparse_observations": observations,
            "old_dx": dx,
            "old_dv": dv,
            "hard_dx": hx,
            "hard_dv": hv,
            "hard_endpoint_positions": hard_states[49].positions.numpy().copy(),
            "hard_endpoint_velocity": hard_states[49].velocity.numpy().copy(),
            "hard_updated_endpoint_positions": hard_endpoint.positions.numpy().copy(),
            "hard_updated_endpoint_velocity": hard_endpoint.velocity.numpy().copy(),
        }
        for name in ("m_restWprev", "m_restWnext", "learned_pmass"):
            arrays["native_rest__" + name] = (
                getattr(original.model, name).detach().numpy().copy()
            )
            arrays["hard_rest__" + name] = (
                getattr(hard.model, name).detach().numpy().copy()
            )
        if any(not np.isfinite(v).all() for k, v in arrays.items() if k != "names"):
            raise ValueError("nonfinite sealed prediction or trace")
        binding = custody._save_arrays(output / "predictions.npz", arrays)
    seal = custody._write(
        output / "prediction_seal.json",
        {
            "schema": SCHEMA + "-prediction-seal",
            "lock_id": lock["artifact_id"],
            "names": plan["names"],
            "predictions": binding,
            "controls_id": controls["artifact_id"],
            "controls_sha256": file_digest(output / "controls.json"),
            "locked_case_count": 14,
            "ordinary_successes": 14,
            "retained_technical_failures": 0,
            "unsealable": 0,
            "replacements": 0,
            "complete": True,
            "source_future_scoring_authorized": True,
            "future_truth_scored": False,
            "future_truth_in_prediction": False,
            "target_access": False,
            "held_v8_access": False,
            "fresh_transfer_authorized": False,
            "runtime": runtime,
            "elapsed_s": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                "stage": "sealed",
                "ordinary_successes": 14,
                "seal_id": seal["artifact_id"],
                "seal_sha256": file_digest(output / "prediction_seal.json"),
                "future_truth_scored": False,
            }
        ),
        flush=True,
    )


def validated_barrier(
    output: Path, lock: dict[str, Any], plan: dict[str, Any], seal_digest: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if (output / "technical_failure.json").exists():
        raise ValueError("technical failure is terminal")
    seal = custody._read(output / "prediction_seal.json", seal_digest)
    controls = custody._read(output / "controls.json", seal["controls_sha256"])
    if (
        seal.get("schema") != SCHEMA + "-prediction-seal"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("names") != plan["names"]
        or seal.get("complete") is not True
        or seal.get("ordinary_successes") != 14
        or seal.get("locked_case_count") != 14
        or any(
            seal.get(k) != 0
            for k in ("retained_technical_failures", "unsealable", "replacements")
        )
        or seal.get("source_future_scoring_authorized") is not True
        or any(
            seal.get(k) is not False
            for k in (
                "future_truth_scored",
                "future_truth_in_prediction",
                "target_access",
                "held_v8_access",
                "fresh_transfer_authorized",
            )
        )
        or controls.get("artifact_id") != seal.get("controls_id")
        or controls.get("schema") != SCHEMA + "-controls"
        or controls.get("passed") is not True
        or set(controls.get("checks", {})) != CHECKS
        or not all(v is True for v in controls["checks"].values())
    ):
        raise ValueError("complete qualified source prediction barrier required")
    arrays = custody._load_arrays(output / "predictions.npz", seal["predictions"])
    if arrays["names"].tolist() != plan["names"]:
        raise ValueError("prediction roster differs")
    validate_prediction_invariants(arrays)
    with np.load(plan["paired_archive"]["path"], allow_pickle=False) as parent:
        if array_digest(arrays["incumbent"]) != array_digest(
            parent["incumbent"]
        ) or array_digest(arrays["paired"]) != array_digest(
            parent["incumbent_propagated_pose_velocity"]
        ):
            raise ValueError("unchanged reference predictions differ")
    return seal, arrays


def validate_prediction_invariants(arrays: dict[str, np.ndarray]) -> None:
    def same(a: np.ndarray, b: np.ndarray) -> None:
        if array_digest(a) != array_digest(b):
            raise ValueError("sealed prediction invariant differs")

    if any(not np.isfinite(v).all() for k, v in arrays.items() if k != "names"):
        raise ValueError("nonfinite prediction artifact")
    same(arrays["incumbent"], arrays["incumbent_full"][:, 50:])
    same(
        arrays["hard_native"][:, :, CLAMPS], arrays["known_actions"].astype(np.float32)
    )
    same(
        arrays["hard_updated"][:, :, CLAMPS],
        arrays["known_actions"][:, 50:].astype(np.float32),
    )
    same(arrays["hard_zero"], arrays["hard_native"][:, 50:])
    hard_full = hard_boundary_readout(
        arrays["incumbent_full"],
        arrays["archived_native"],
        arrays["hard_native"],
        enabled=True,
    )
    same(arrays["hard_baseline"], hard_full[:, 50:])
    same(
        arrays["hard_paired"],
        paired_physical_readout(
            arrays["hard_baseline"],
            arrays["hard_native"][:, 50:],
            arrays["hard_updated"],
        ),
    )
    same(
        arrays["paired"],
        paired_physical_readout(
            arrays["incumbent"], arrays["native"][:, 50:], arrays["native_updated"]
        ),
    )
    for label, prefix in (
        ("old", arrays["incumbent_full"][:, :50]),
        ("hard", hard_full[:, :50]),
    ):
        dx, dv = sparse_state_increments(
            prefix, arrays["sparse_observations"], config_for_source()
        )
        same(arrays[label + "_dx"], dx)
        same(arrays[label + "_dv"], dv)
    same(arrays["hard_endpoint_positions"], arrays["hard_native"][:, 49])
    same(
        arrays["hard_updated_endpoint_positions"],
        arrays["hard_endpoint_positions"] + arrays["hard_dx"].astype(np.float32),
    )
    same(
        arrays["hard_updated_endpoint_velocity"],
        arrays["hard_endpoint_velocity"] + arrays["hard_dv"].astype(np.float32),
    )
    for name in ("m_restWprev", "m_restWnext", "learned_pmass"):
        same(arrays["native_rest__" + name], arrays["hard_rest__" + name])


def score(output: Path, lock_digest: str, seal_digest: str) -> None:
    if (output / "result.json").exists():
        raise ValueError("source result already exists; do not rescore")
    lock, plan = verify_lock(output, lock_digest)
    seal, arrays = validated_barrier(output, lock, plan, seal_digest)
    with np.load(plan["archive"]["path"], allow_pickle=False) as data:
        truth = data["targets"][:, 50:170].copy()
    scores = score_predictions(plan["names"], {k: arrays[k] for k in ARMS}, truth)
    result = custody._write(
        output / "result.json",
        {
            "schema": SCHEMA + "-result",
            "lock_id": lock["artifact_id"],
            "source_implementation_revision": lock["revision"],
            "prediction_seal_id": seal["artifact_id"],
            "prediction_seal_sha256": seal_digest,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "predict", "score"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lock-sha256")
    parser.add_argument("--prediction-seal-sha256")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output_root)
    elif not args.lock_sha256:
        parser.error("predict/score requires the exact lock file SHA-256")
    elif args.stage == "predict":
        predict(args.output_root, args.lock_sha256)
    elif not args.prediction_seal_sha256:
        parser.error("score requires the exact complete prediction seal SHA-256")
    else:
        score(args.output_root, args.lock_sha256, args.prediction_seal_sha256)


if __name__ == "__main__":
    main()
