#!/usr/bin/env python3
"""Build the source-only posterior-aware Slingshot certificate diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record
from bayesian_phystwin_experiments.dlolab_slingshot_batch import split_batch
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    BASELINE,
    infer,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import task_metrics
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v1 import (
    bias_invariant_features,
    posterior_policy_action,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v2 import (
    combined_competence_features,
    descriptive_prefix_capacity,
    posterior_diagnostic_features,
    repeated_rotation_diagnostic,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
POLICY_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v1"
)
PARENT_LOCK_ID = "015e6d84aa68a2a4310552ef4880752b972890f02d3e09e333ff575c92b8df25"
PARENT_RESULT_ID = "9b8ff0817744392e0584c9b59936dd1b0e9331d3b0fa2d021f5a361947d32ee9"
POLICY_LOCK_ID = "9401705f7d11f2acae32b4307eeff4e044aeba3e3e2a6403a568a999ee33a550"
POLICY_FAILURE_ID = "e2806339d7f83081140c74d6f9e15eb3605f6aec1798b5931cdf227353e1f76f"
POLICY_RESULT_ID = "f0ac1753c92630bcc738db30f466f0745ec726d7aff74b99a0198e5aca6fb25b"
REFERENCE_ROLES = (("evaluation", 32), ("calibration", 19))
BOUND_SOURCES = (
    "src/bayesian_phystwin/policy_gain_certificate.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_policy_certificate_v2.py",
    "scripts/audit_dlolab_slingshot_policy_certificate_development_v2.py",
    "scripts/verify_dlolab_slingshot_policy_certificate_development_v2.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "artifact_id"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_runner() -> ModuleType:
    path = ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v1.py"
    spec = importlib.util.spec_from_file_location("policy_certificate_v1_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen policy-certificate runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent_rows() -> tuple[
    list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]
]:
    lock = read_record(PARENT_ROOT / "lock.json")
    result = read_record(PARENT_ROOT / "result.json")
    if (
        lock.get("artifact_id") != PARENT_LOCK_ID
        or result.get("artifact_id") != PARENT_RESULT_ID
        or lock.get("protected_data_read") is not False
    ):
        raise ValueError("registered public-simulator parent changed")
    ids: list[str] = []
    geometry: list[np.ndarray] = []
    posterior: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    actions: list[int] = []
    for role, count in REFERENCE_ROLES:
        for index in range(count):
            case_id = f"parent-{role}-{index:02d}"
            with np.load(
                PARENT_ROOT
                / f"{role}-predictions"
                / f"case-{index:02d}"
                / "arrays.npz",
                allow_pickle=False,
            ) as archive:
                inference = {
                    name: np.array(archive[name], copy=True)
                    for name in (
                        "weights",
                        "iid_weights",
                        "expected_losses",
                        "iid_expected_losses",
                        "map_losses",
                        "nominal_losses",
                        "prior_losses",
                        "raw_upper",
                    )
                }
                observation = np.array(archive["observation"], copy=True)
            with np.load(
                PARENT_ROOT / f"{role}-future-{index:02d}" / "arrays.npz",
                allow_pickle=False,
            ) as archive:
                future = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            reward = np.asarray(
                [task_metrics(row)["native_reward"] for row in split_batch(future, 8)[:7]],
                dtype=np.float64,
            )
            ids.append(case_id)
            geometry.append(bias_invariant_features(observation))
            posterior.append(posterior_diagnostic_features(inference))
            gains.append(reward - reward[BASELINE])
            actions.append(int(posterior_policy_action(inference["expected_losses"])))
    return ids, geometry, posterior, gains, actions


def _policy_rows(
    runner: ModuleType,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    np.ndarray,
    np.ndarray,
]:
    lock = read_record(POLICY_ROOT / "lock.json")
    failure = read_record(POLICY_ROOT / "failure.json")
    result = read_record(
        ROOT / "results/source/dlolab_slingshot_policy_certificate_source_v1/summary.json"
    )
    if (
        lock.get("artifact_id") != POLICY_LOCK_ID
        or failure.get("artifact_id") != POLICY_FAILURE_ID
        or result.get("artifact_id") != POLICY_RESULT_ID
        or failure.get("terminal_stage") != "evaluation-decision-barrier"
        or result.get("ordinary_evaluation_futures") != 0
        or list(POLICY_ROOT.glob("evaluation-future-*"))
    ):
        raise ValueError("terminal policy-certificate source boundary changed")

    _, bank, _ = runner.load_parent()
    _, candidate = runner.load_candidates(POLICY_ROOT, lock, "calibration")
    rewards, _, all_qa = runner._future_rewards(POLICY_ROOT, lock, "calibration")
    _, prefix = runner.load_candidates(POLICY_ROOT, lock, "evaluation")
    if not all_qa or rewards.shape != (96, 7):
        raise ValueError("complete qualified opened calibration futures required")

    ids: list[str] = []
    geometry: list[np.ndarray] = []
    posterior: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    actions: list[int] = []
    for index, observation in enumerate(candidate["observation_m"]):
        inferred = infer(observation, bank["prefix"], bank["reward"])
        action = int(candidate["candidate_actions"][index])
        if action != int(posterior_policy_action(inferred["expected_losses"])):
            raise ValueError("opened calibration candidate action changed")
        ids.append(f"policy-calibration-{index:03d}")
        geometry.append(bias_invariant_features(observation))
        posterior.append(posterior_diagnostic_features(inferred))
        gains.append(rewards[index] - rewards[index, BASELINE])
        actions.append(action)

    prefix_features = np.stack(
        [
            combined_competence_features(
                observation, infer(observation, bank["prefix"], bank["reward"])
            )
            for observation in prefix["observation_m"]
        ]
    )
    prefix_actions = np.asarray(prefix["candidate_actions"], dtype=np.int64)
    return (
        ids,
        geometry,
        posterior,
        gains,
        actions,
        prefix_features,
        prefix_actions,
    )


def build() -> dict[str, Any]:
    runner = _load_runner()
    parent = _parent_rows()
    policy = _policy_rows(runner)
    ids = tuple(parent[0] + policy[0])
    geometry = np.stack(parent[1] + policy[1])
    posterior = np.stack(parent[2] + policy[2])
    combined = np.concatenate((geometry, posterior), axis=1)
    gains = np.stack(parent[3] + policy[3])
    actions = np.asarray(parent[4] + policy[4], dtype=np.int64)

    specifications = {
        "geometry_uniform_k5": (geometry, 5, False),
        "combined_distance_k5": (combined, 5, True),
        "combined_distance_k7": (combined, 7, True),
        "combined_distance_k10": (combined, 10, True),
    }
    models = {
        name: repeated_rotation_diagnostic(
            case_ids=ids,
            features=feature,
            candidate_actions=actions,
            action_gains=gains,
            neighbor_count=neighbors,
            distance_weighted=weighted,
        )
        for name, (feature, neighbors, weighted) in specifications.items()
    }
    eligible = [
        name
        for name, value in models.items()
        if value["marginal_lower_bound_coverage_quantiles"][2] >= 0.85
        and value["harmful_accepted_count_quantiles"][2] <= 1.0
    ]
    if not eligible:
        selected_name: str | None = None
    else:
        selected_name = sorted(
            eligible,
            key=lambda name: (
                models[name]["harmful_accepted_count_quantiles"][2],
                -models[name]["mean_guarded_gain_quantiles"][2],
                name,
            ),
        )[0]
    selected = models[selected_name] if selected_name is not None else None
    capacity = (
        descriptive_prefix_capacity(
            case_ids=ids,
            features=combined,
            candidate_actions=actions,
            action_gains=gains,
            prefix_features=policy[5],
            prefix_actions=policy[6],
            neighbor_count=7,
        )
        if selected_name == "combined_distance_k7"
        else None
    )
    predecessor_gain = models["geometry_uniform_k5"]["mean_guarded_gain_quantiles"][2]
    checks = {
        "selected_combined_distance_k7": selected_name == "combined_distance_k7",
        "median_guarded_gain_at_least_0_003": (
            selected is not None and selected["mean_guarded_gain_quantiles"][2] >= 0.003
        ),
        "minimum_guarded_gain_positive": (
            selected is not None and selected["mean_guarded_gain_quantiles"][0] > 0.0
        ),
        "median_coverage_at_least_0_85": (
            selected is not None
            and selected["marginal_lower_bound_coverage_quantiles"][2] >= 0.85
        ),
        "median_harmful_accepted_at_most_1": (
            selected is not None
            and selected["harmful_accepted_count_quantiles"][2] <= 1.0
        ),
        "median_gain_beats_geometry_predecessor_by_0_001": (
            selected is not None
            and selected["mean_guarded_gain_quantiles"][2] - predecessor_gain >= 0.001
        ),
        "prefix_capacity_at_least_24": (
            capacity is not None and capacity["accepted_prefix_count"] >= 24
        ),
    }
    result: dict[str, Any] = {
        "schema": "dlolab-slingshot-policy-certificate-development-v2",
        "status": "opened_source_model_selection_and_capacity_diagnostic_only",
        "source_world_count": len(ids),
        "source_groups": {"parent_opened": 51, "policy_calibration_opened": 96},
        "prefix_only_capacity_world_count": len(policy[6]),
        "feature_dimensions": {
            "bias_invariant_geometry": geometry.shape[1],
            "posterior_diagnostics": posterior.shape[1],
            "combined": combined.shape[1],
        },
        "candidate_models": models,
        "selection_rule": (
            "among median-coverage>=0.85 and median-harm<=1 candidates, minimize "
            "median harm, then maximize median guarded gain, then lexical tie break"
        ),
        "selected_model": selected_name,
        "selected_prefix_capacity": capacity,
        "advancement_checks": checks,
        "advancement_gate_passed": bool(all(checks.values())),
        "parent_lock_id": PARENT_LOCK_ID,
        "parent_result_id": PARENT_RESULT_ID,
        "policy_lock_id": POLICY_LOCK_ID,
        "policy_failure_id": POLICY_FAILURE_ID,
        "policy_result_id": POLICY_RESULT_ID,
        "source_sha256": {name: _sha256(ROOT / name) for name in BOUND_SOURCES},
        "source_outcomes_previously_opened": True,
        "prefix_panel_outcomes_read": False,
        "prefix_panel_futures_generated": False,
        "new_simulation_executed": False,
        "prospective_coverage_claim": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
    }
    result["artifact_id"] = _canonical_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "artifact_id",
                    "selected_model",
                    "selected_prefix_capacity",
                    "advancement_gate_passed",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
