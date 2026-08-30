#!/usr/bin/env python3
"""Run the reward-aligned stochastic-execution Slingshot certificate."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin_experiments import (
    dlolab_slingshot_policy_certificate_source_v4 as method,
)
from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[2]
V3_RUNNER_PATH = (
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "slingshot_policy_certificate_source_v4_base", V3_RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

OUTPUT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v4"
)
ATTEMPT_LEDGER = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v4.attempt.json"
)
QUALIFICATION_SUMMARY = (
    ROOT
    / "results/source/dlolab_slingshot_reward_aligned_native_qualification_v1/summary.json"
)
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_source_v4.py",
    "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v4.py",
    "tests/test_dlolab_slingshot_policy_certificate_source_v4.py",
    "tests/test_dlolab_slingshot_policy_certificate_source_v4_custody.py",
    "docs/dlolab_slingshot_policy_certificate_source_v4.md",
    "docs/dlolab_slingshot_reward_aligned_native_qualification_v1.md",
    "results/source/dlolab_slingshot_reward_aligned_native_qualification_v1/summary.json",
    *runner.SOURCES,
)


def load_qualification() -> dict[str, Any]:
    """Load the exact source-only reward-aligned qualification."""

    if (
        QUALIFICATION_SUMMARY.is_symlink()
        or file_digest(QUALIFICATION_SUMMARY)
        != method.QUALIFICATION_RESULT_SHA256
    ):
        raise ValueError("reward-aligned native qualification file changed")
    value = read_record(QUALIFICATION_SUMMARY)
    denominator = value.get("denominator", {})
    contract = value.get("registered_reward_aligned_contract", {})
    if (
        value.get("artifact_id") != method.QUALIFICATION_RESULT_ID
        or value.get("status") != "passed_on_opened_v3_calibration_source"
        or denominator.get("worlds") != 128
        or denominator.get("ordinary_action_processes") != 1024
        or denominator.get("reward_aligned_qualified_worlds") != 128
        or contract.get("duplicate_reward_error_at_most") != 0.001
        or contract.get("duplicate_position_is_reported_process_variability_not_admission")
        is not True
        or contract.get("incumbent_reward_estimator")
        != "mean_of_independent_action_slots_5_and_7"
        or value.get("v4_protocol_freeze_authorized") is not True
        or value.get("v4_scientific_execution_authorized") is not False
        or value.get("retry_authorized") is not False
        or value.get("replacement_authorized") is not False
        or value.get("protected_data_read") is not False
        or value.get("held_v8_read") is not False
        or value.get("dlo4_dlo5_read") is not False
    ):
        raise ValueError("passing reward-aligned native qualification required")
    return cast(dict[str, Any], value)


def configure() -> None:
    """Configure the reviewed v3 custody engine for the v4 estimand."""

    runner.OUTPUT_ROOT = OUTPUT_ROOT
    runner.WORKER_RUNNER_PATH = Path(__file__).resolve()
    runner.ATTEMPT_LEDGER = ATTEMPT_LEDGER
    runner.QUALIFICATION_SUMMARY = QUALIFICATION_SUMMARY
    runner.QUALIFICATION_RESULT_ID = method.QUALIFICATION_RESULT_ID
    runner.QUALIFICATION_RESULT_SHA256 = method.QUALIFICATION_RESULT_SHA256
    runner.STUDY_LABEL = "Slingshot v4"
    runner.LOCK_SCHEMA = "dlolab-slingshot-policy-certificate-lock-v4"
    runner.ATTEMPT_SCHEMA = "dlolab-slingshot-policy-certificate-attempt-v4"
    runner.CANDIDATE_SCHEMA = "dlolab-slingshot-policy-candidates-v4"
    runner.CALIBRATION_SCHEMA = "dlolab-slingshot-policy-calibration-v4"
    runner.DECISION_SCHEMA = "dlolab-slingshot-policy-decisions-v4"
    runner.BARRIER_SCHEMA = "dlolab-slingshot-policy-barrier-v4"
    runner.TASK_CLAIM_SCHEMA = "dlolab-slingshot-policy-task-claim-v4"
    runner.TASK_SEAL_SCHEMA = "dlolab-slingshot-policy-task-seal-v4"
    runner.TASK_FAILURE_SCHEMA = "dlolab-slingshot-policy-task-failure-v4"
    runner.WORLD_QUALIFICATION_SCHEMA = (
        "dlolab-slingshot-policy-world-qualification-v4"
    )
    runner.RUN_FAILURE_SCHEMA = "dlolab-slingshot-policy-run-failure-v4"
    runner.SOURCES = SOURCES
    runner.load_qualification = load_qualification
    runner.protocol = method.protocol
    runner.continuous_worlds = method.continuous_worlds
    runner.prefix_batch_count = method.prefix_batch_count
    runner.prefix_task = method.prefix_task
    runner.future_action_task = method.future_action_task
    runner.candidate_predictions = method.candidate_predictions
    runner.independent_world_qa = method.reward_aligned_world_qa
    runner.world_rewards = method.reward_aligned_world_rewards
    runner.score = method.score


configure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-role", choices=("calibration", "evaluation"))
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-action", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    worker_values = (args.worker_role, args.worker_kind, args.worker_index)
    if args.verify_only and any(value is not None for value in worker_values):
        parser.error("verification cannot be combined with worker execution")
    if args.verify_only:
        verified = runner.verify_result(args.output)
        print(f"verified Slingshot v4 result {verified['artifact_id']}", flush=True)
    elif all(value is not None for value in worker_values):
        runner.worker(
            args.output,
            args.worker_role,
            args.worker_kind,
            args.worker_index,
            args.worker_action,
        )
    elif any(value is not None for value in (*worker_values, args.worker_action)):
        parser.error("all registered worker arguments are required")
    else:
        runner.run(args.output)
