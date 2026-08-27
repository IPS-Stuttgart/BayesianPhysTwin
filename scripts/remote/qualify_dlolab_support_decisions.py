"""One frozen native support-task qualification, not a Bayesian method result."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_native import verify_upstream
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    load_bundle,
    read_record,
    runtime_identity,
    write_bundle,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_support_task import (
    NativeSupportRuntime,
    SupportTaskConfig,
    action_commands,
    contact_clearance,
    qualification_protocol,
    sensitivity_gate,
    source_task_losses,
    source_worlds,
    task_actions,
    task_goals,
)

ROOT = Path(__file__).resolve().parents[2]
NATIVE_CHECKS = (
    "finite_native_trajectories",
    "root_error_at_most_1e_minus_10_m",
    "support_position_unchanged",
    "length_error_at_most_10pct",
    "support_penetration_at_most_3mm",
    "at_least_two_worlds_make_support_contact",
    "native_snapshots_unmodified",
    "action_0_positions_byte_identical",
    "action_0_all_memory_byte_identical",
    "action_11_positions_byte_identical",
    "action_11_all_memory_byte_identical",
    "monolithic_positions_byte_identical",
    "monolithic_all_memory_byte_identical",
)
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_support_task.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_study.py",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "src/bayesian_phystwin/guard_harm_risk.py",
    "src/bayesian_phystwin/_canonical_contracts.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "scripts/remote/qualify_dlolab_support_decisions.py",
    "tests/test_dlolab_support_task.py",
    "docs/dlolab_support_decision_qualification_v1.md",
)


def freeze(output: Path, upstream: Path) -> dict[str, Any]:
    revision = clean_revision(ROOT)
    identity = verify_upstream(upstream)
    runtime = runtime_identity()
    sources = {path: file_digest(ROOT / path) for path in SOURCES}
    output.mkdir(parents=True, exist_ok=False)
    return write_record(
        output / "lock.json",
        {
            "schema": "dlolab-support-decision-lock-v1",
            "source_revision": revision,
            "source_sha256": sources,
            "upstream": identity,
            "runtime": runtime,
            "output_root": str(output.resolve()),
            "upstream_root": str(upstream.resolve()),
            "protocol": qualification_protocol(),
            "native_executions": 1,
            "method_evaluation_authorized": False,
            "protected_data_read": False,
        },
    )


def validate_lock(output: Path) -> dict[str, Any]:
    lock = read_record(output / "lock.json")
    if (
        lock.get("schema") != "dlolab-support-decision-lock-v1"
        or lock.get("protocol") != qualification_protocol()
        or lock.get("native_executions") != 1
        or lock.get("method_evaluation_authorized") is not False
        or lock.get("protected_data_read") is not False
    ):
        raise ValueError("support qualification protocol changed")
    if (
        str(output.resolve()) != lock["output_root"]
        or clean_revision(ROOT) != lock["source_revision"]
    ):
        raise ValueError("registered root or execution revision changed")
    if lock["source_sha256"] != {path: file_digest(ROOT / path) for path in SOURCES}:
        raise ValueError("registered implementation changed")
    if (
        runtime_identity() != lock["runtime"]
        or verify_upstream(Path(lock["upstream_root"])) != lock["upstream"]
    ):
        raise ValueError("native runtime changed")
    return lock


def generate(output: Path) -> dict[str, Any]:
    lock = validate_lock(output)
    directory = output / "native"
    directory.mkdir(exist_ok=False)
    attempt = write_record(
        directory / "attempt.json",
        {
            "schema": "dlolab-support-native-attempt-v1",
            "lock_id": lock["artifact_id"],
            "retry_authorized": False,
            "source_only": True,
            "protected_data_read": False,
        },
    )
    runtime = None
    start = time.monotonic()
    try:
        config = SupportTaskConfig()
        bending, support_x = source_worlds()
        runtime = NativeSupportRuntime(
            Path(lock["upstream_root"]), config, bending, support_x
        )
        initial = runtime.capture()
        hold = np.broadcast_to(
            runtime.initial_positions[:, :2], (config.prefix_steps, 6, 2, 3)
        ).copy()
        prefix = runtime.rollout(hold)
        endpoint = runtime.capture()
        futures = []
        final_states = []
        root_error = float(np.max(np.abs(prefix[:, :, :2] - hold)))
        support_unchanged = runtime.support_unchanged()
        for index in range(12):
            runtime.restore(endpoint)
            commands = action_commands(config, runtime.initial_positions, index)
            value = runtime.rollout(commands)
            root_error = max(
                root_error, float(np.max(np.abs(value[:, :, :2] - commands)))
            )
            support_unchanged = support_unchanged and runtime.support_unchanged()
            futures.append(value)
            final_states.append(runtime.capture().field_digests)
            print(
                f"generated registered support-task action {index + 1}/12", flush=True
            )
        futures_array = np.stack(futures)
        replay_checks = {}
        replay_arrays = {}
        replay_states = {}
        for index in (0, 11):
            runtime.restore(endpoint)
            replay = runtime.rollout(
                action_commands(config, runtime.initial_positions, index)
            )
            replay_arrays[f"replay_{index}"] = replay
            replay_checks[f"action_{index}_positions_byte_identical"] = array_digest(
                replay
            ) == array_digest(futures[index])
            replay_states[str(index)] = runtime.capture().field_digests
            replay_checks[f"action_{index}_all_memory_byte_identical"] = (
                replay_states[str(index)] == final_states[index]
            )
        runtime.restore(initial)
        monolithic = runtime.rollout(
            np.concatenate(
                [hold, action_commands(config, runtime.initial_positions, 11)]
            )
        )
        replay_checks["monolithic_positions_byte_identical"] = array_digest(
            monolithic[config.prefix_steps :]
        ) == array_digest(futures[11])
        monolithic_state = runtime.capture().field_digests
        replay_checks["monolithic_all_memory_byte_identical"] = (
            monolithic_state == final_states[11]
        )
        endpoint.validate(runtime.config, runtime.model_id)
        initial.validate(runtime.config, runtime.model_id)
        full = np.concatenate([prefix, *futures], axis=0)
        lengths = np.linalg.norm(np.diff(full, axis=2), axis=-1)
        length_error = float(np.max(np.abs(lengths / config.rod.interval_m - 1)))
        clearance = contact_clearance(full, runtime.initial_support, config)
        contact_worlds = int(np.any(clearance <= 0.002, axis=0).sum())
        penetration = float(max(0.0, -clearance.min()))
        checks = {
            "finite_native_trajectories": bool(np.isfinite(full).all()),
            "root_error_at_most_1e_minus_10_m": root_error <= 1e-10,
            "support_position_unchanged": bool(
                support_unchanged and runtime.support_unchanged()
            ),
            "length_error_at_most_10pct": length_error <= 0.10,
            "support_penetration_at_most_3mm": penetration <= 0.003,
            "at_least_two_worlds_make_support_contact": contact_worlds >= 2,
            "native_snapshots_unmodified": True,
            **replay_checks,
        }
        arrays = {
            "prefix": prefix,
            "future": futures_array,
            "monolithic": monolithic,
            "support": runtime.initial_support,
            "bending": bending,
            "support_x": support_x,
            "contact_clearance": clearance,
            **replay_arrays,
        }
        model_id = runtime.model_id
        runtime.close()
        runtime = None
        bundle = write_bundle(directory, arrays)
        result = write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-support-native-seal-v1",
                "lock_id": lock["artifact_id"],
                "attempt_id": attempt["artifact_id"],
                "world_count": 6,
                "action_count": 12,
                "bundle": bundle,
                "model_id": model_id,
                "endpoint_state_sha256": endpoint.field_digests,
                "final_state_sha256": final_states,
                "replay_state_sha256": replay_states,
                "monolithic_state_sha256": monolithic_state,
                "checks": checks,
                "native_qualification_passed": all(checks.values()),
                "maximum_relative_length_error": length_error,
                "maximum_root_error_m": root_error,
                "maximum_support_penetration_m": penetration,
                "contact_worlds": contact_worlds,
                "wall_seconds": time.monotonic() - start,
                "protected_data_read": False,
                "method_comparison": False,
                "method_evaluation_authorized": False,
            },
        )
        print(
            f"native qualification={result['native_qualification_passed']}; id={result['artifact_id']}"
        )
        return result
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-support-native-failure-v1",
                "lock_id": lock["artifact_id"],
                "attempt_id": attempt["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "method_evaluation_authorized": False,
                "protected_data_read": False,
            },
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()


def analyze(output: Path) -> dict[str, Any]:
    lock = validate_lock(output)
    seal = read_record(output / "native" / "seal.json")
    if (
        seal.get("schema") != "dlolab-support-native-seal-v1"
        or seal.get("lock_id") != lock["artifact_id"]
        or seal.get("protected_data_read") is not False
        or seal.get("method_comparison") is not False
        or seal.get("method_evaluation_authorized") is not False
    ):
        raise ValueError("native seal identity changed")
    checks = seal.get("checks", {})
    if (
        seal.get("native_qualification_passed") is not True
        or set(checks) != set(NATIVE_CHECKS)
        or not all(v is True for v in checks.values())
    ):
        raise ValueError("native contract failed; decision analysis prohibited")
    if seal.get("world_count") != 6 or seal.get("action_count") != 12:
        raise ValueError("source denominator changed")
    arrays = load_bundle(output / "native", seal["bundle"])
    for name, expected in zip(("bending", "support_x"), source_worlds(), strict=True):
        np.testing.assert_array_equal(arrays[name], expected)
    for index in (0, 11):
        if array_digest(arrays[f"replay_{index}"]) != array_digest(
            arrays["future"][index]
        ):
            raise ValueError("native trajectory replay changed")
        if seal["replay_state_sha256"][str(index)] != seal["final_state_sha256"][index]:
            raise ValueError("native memory replay changed")
    if array_digest(
        arrays["monolithic"][SupportTaskConfig().prefix_steps :]
    ) != array_digest(arrays["future"][11]):
        raise ValueError("native monolithic replay changed")
    if seal["monolithic_state_sha256"] != seal["final_state_sha256"][11]:
        raise ValueError("native monolithic memory changed")
    loss = source_task_losses(arrays["future"], SupportTaskConfig())
    # Independent loop checks metric units, goal/world/action alignment, and averaging.
    direct = np.empty_like(loss)
    for goal_index, goal in enumerate(task_goals()):
        for world in range(6):
            for action, displacement in enumerate(task_actions()):
                positions = arrays["future"][action, -50:, world, -1]
                direct[goal_index, world, action] = sum(
                    float(np.dot(x - goal, x - goal)) for x in positions
                ) / 50 + 0.02 * float(displacement @ displacement)
    np.testing.assert_allclose(loss, direct, rtol=1e-12, atol=1e-14)
    outcome = sensitivity_gate(loss)
    result = write_record(
        output / "sensitivity.json",
        {
            "schema": "dlolab-support-decision-sensitivity-result-v1",
            "lock_id": lock["artifact_id"],
            "native_seal_id": seal["artifact_id"],
            "losses_sha256": array_digest(loss),
            "losses_goal_world_action_m2": loss.tolist(),
            "independent_loss_checks": int(loss.size),
            "native_qualification_passed": True,
            "source_only": True,
            "protected_data_read": False,
            **outcome,
        },
    )
    print(
        f"sensitivity={result['decision_sensitivity_passed']}; passing goals={result['passing_goals']}/9; id={result['artifact_id']}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "generate", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.upstream is None:
            parser.error("freeze requires --upstream")
        print(freeze(args.output, args.upstream)["artifact_id"])
    elif args.upstream is not None:
        parser.error("native source path comes only from the lock")
    elif args.stage == "generate":
        generate(args.output)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
