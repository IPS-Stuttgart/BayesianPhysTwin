"""Qualify native reset and clamp controls before any decision study."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.dlolab_native import (
    STATE_FIELDS,
    DloLabConfig,
    DloLabRuntime,
    native_state_digests,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-bank", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip():
        raise ValueError("qualification requires clean committed source")
    config = DloLabConfig()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_paths = (
        "src/bayesian_phystwin_experiments/dlolab_native.py",
        "scripts/remote/qualify_dlolab_native.py",
        "tests/test_dlolab_native.py",
        "docs/dlolab_native_qualification_v1.md",
    )
    attempt = {
        "schema": "dlolab-native-qualification-attempt-v1",
        "source_revision": revision,
        "source_sha256": {name: file_digest(ROOT / name) for name in source_paths},
        "config": dataclasses.asdict(config),
        "config_id": config.identity,
        "synthetic_only": True,
        "protected_data_read": False,
        "method_outcomes_read": False,
        "physical_execution": False,
        "world_bank": args.world_bank,
    }
    write_json_once(args.output_dir / "attempt.json", attempt)
    start = time.monotonic()
    runtime = None
    try:
        runtime = DloLabRuntime(
            args.upstream,
            config,
            batch_size=3 if args.world_bank else 1,
            bending_moduli=(config.bending_modulus * np.array([0.5, 1.0, 2.0]))
            if args.world_bank
            else None,
            lateral_velocities=np.array([-0.15, 0.0, 0.15])
            if args.world_bank
            else None,
        )
        initial = runtime.capture()
        clamps = runtime.initial_positions[:, :2].copy()
        prefix = np.broadcast_to(clamps, (25, *clamps.shape)).copy()
        prefix_prediction = runtime.rollout(prefix)
        branch = runtime.capture()
        commands = np.broadcast_to(clamps, (40, *clamps.shape)).copy()
        phase = np.linspace(0.0, 1.0, len(commands))
        commands[:, :, :, 1] += (0.02 * (3 * phase**2 - 2 * phase**3))[:, None, None]
        future = runtime.rollout(commands)
        future_state = runtime.capture()
        runtime.restore(branch)
        replay = runtime.rollout(commands)
        replay_state = runtime.capture()
        runtime.restore(initial)
        monolithic = runtime.rollout(np.concatenate([prefix, commands]))
        monolithic_state = runtime.capture()
        runtime.restore(branch)
        reverse_commands = commands.copy()
        reverse_commands[:, :, :, 1] *= -1.0
        alternative = runtime.rollout(reverse_commands)
        branch.validate(config, runtime.model_id)
        initial.validate(config, runtime.model_id)
        max_clamp_error = float(np.max(np.abs(future[:, :, :2] - commands)))
        action_effect = float(
            np.max(
                np.linalg.norm(
                    future[:, :, 2:] - alternative[:, :, 2:],
                    axis=-1,
                )
            )
        )
        lengths = np.linalg.norm(np.diff(monolithic, axis=2), axis=-1)
        length_error = float(np.max(np.abs(lengths / config.interval_m - 1.0)))
        checks = {
            "finite_native_trajectories": bool(
                all(
                    np.isfinite(x).all()
                    for x in (
                        prefix_prediction,
                        future,
                        replay,
                        monolithic,
                        alternative,
                    )
                )
            ),
            "replay_trajectory_byte_identity": array_digest(future)
            == array_digest(replay),
            "replay_all_memory_byte_identity": future_state.field_digests
            == replay_state.field_digests,
            "monolithic_trajectory_byte_identity": array_digest(monolithic[25:])
            == array_digest(future),
            "monolithic_all_memory_byte_identity": monolithic_state.field_digests
            == future_state.field_digests,
            "clamp_tracking_error_at_most_1e_minus_10_m": max_clamp_error <= 1e-10,
            "alternative_action_moves_free_nodes_above_1e_minus_5_m": action_effect
            > 1e-5,
            "segment_length_relative_error_at_most_10pct": length_error <= 0.1,
            "snapshot_unmodified": branch.field_digests
            == native_state_digests(branch.native_state),
        }
        with (args.output_dir / "trajectories.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                prefix=prefix_prediction,
                future=future,
                replay=replay,
                monolithic=monolithic,
                alternative=alternative,
                commands=commands,
            )
        result = {
            **attempt,
            "schema": "dlolab-native-qualification-result-v1",
            "upstream": runtime.provenance,
            "runtime": {
                "python": platform.python_version(),
                "torch": importlib.metadata.version("torch"),
                "numpy": np.__version__,
                "quadrants": importlib.metadata.version("quadrants"),
                "device": "cpu",
                "precision": "float64",
                "torch_threads": 1,
            },
            "state_fields": list(STATE_FIELDS),
            "checks": checks,
            "qualification_passed": all(checks.values()),
            "maximum_clamp_error_m": max_clamp_error,
            "alternative_action_effect_m": action_effect,
            "maximum_relative_segment_length_error": length_error,
            "wall_seconds": time.monotonic() - start,
            "trajectories_sha256": file_digest(args.output_dir / "trajectories.npz"),
            "decision_study_authorized": False,
            "official_benchmark_result": False,
        }
        write_json_once(args.output_dir / "result.json", result)
        print(
            json.dumps(
                {
                    "checks": checks,
                    "qualification_passed": result["qualification_passed"],
                },
                sort_keys=True,
            )
        )
    except Exception as error:
        write_json_once(
            args.output_dir / "failure.json",
            {
                **attempt,
                "schema": "dlolab-native-qualification-failure-v1",
                "error_type": type(error).__name__,
                "message": str(error),
                "qualification_passed": False,
                "decision_study_authorized": False,
            },
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
