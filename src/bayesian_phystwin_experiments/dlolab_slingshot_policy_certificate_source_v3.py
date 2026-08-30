"""Fresh independent-action Slingshot policy-gain certification."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.policy_gain_certificate import (
    LocalPolicyGainPredictor,
    PolicyGainCalibration,
    predict_distance_weighted_local_policy_gain,
)

from .coupled_action_regret import RegretCalibration
from .dlolab_slingshot_belief import (
    BASELINE,
    BIAS_STD_M,
    NOISE_STD_M,
    REWARD_MARGIN,
    infer,
)
from .dlolab_slingshot_independent_native_v3 import qualification_worlds
from .dlolab_slingshot_policy_certificate_source_v1 import MISCOVERAGE
from .dlolab_slingshot_policy_certificate_source_v2 import (
    ARM_NAMES,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    HARM_RISK_BUDGET,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    CALIBRATION_RANK as V2_CALIBRATION_RANK,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    calibrate as calibrate,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    calibrate_simultaneous_guard as calibrate_simultaneous_guard,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    continuous_worlds as policy_v2_worlds,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    guarded_decisions as guarded_decisions,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    opened_world_keys as prior_opened_world_keys,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    pre_future_checks as pre_future_checks,
)
from .dlolab_slingshot_policy_certificate_source_v2 import (
    score as score_v2,
)
from .dlolab_slingshot_policy_certificate_v1 import posterior_policy_action
from .dlolab_slingshot_policy_certificate_v2 import (
    NEIGHBOR_COUNT,
    combined_competence_features,
)

Array: TypeAlias = NDArray[Any]
CALIBRATION_RANK = V2_CALIBRATION_RANK
COUNTS = {"calibration": 128, "evaluation": 288}
WORLD_SEEDS = {"calibration": 262070, "evaluation": 262071}
SENSOR_SEEDS = {"calibration": 262072, "evaluation": 262073}
ACTION_COUNT = 8
FUTURE_PROCESS_COUNT = ACTION_COUNT * sum(COUNTS.values())
QUALIFICATION_RESULT_ID = (
    "14a33a58c5f37992da517c4114eb959b6a174fc1101a992aa5ae668a9b3cd096"
)
QUALIFICATION_RESULT_SHA256 = (
    "016ce7038d62bb7babc1cde3aeaffe7d411e2710bbb21327d3201840102df498"
)


def _world_key(world: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(world["x_offset_m"]),
        float(world["bending_E"]),
        float(world["stretching_K"]),
    )


def continuous_worlds(role: str) -> list[dict[str, Any]]:
    """Return the fixed fresh v3 roster for one statistical partition."""

    if role not in COUNTS:
        raise ValueError("unknown policy-certificate partition")
    count = COUNTS[role]
    rng = np.random.default_rng(WORLD_SEEDS[role])
    x = rng.uniform(-0.02, 0.02, count)
    bending = 1e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), count))
    stretching = 8e5 * np.exp(rng.uniform(np.log(0.5), np.log(2.0), count))
    return [
        {
            "index": index,
            "x_offset_m": float(x[index]),
            "bending_E": float(bending[index]),
            "stretching_K": float(stretching[index]),
        }
        for index in range(count)
    ]


def opened_world_keys() -> set[tuple[float, float, float]]:
    """Return every source or development world opened before v3."""

    prior = set(
        cast(set[tuple[float, float, float]], prior_opened_world_keys())
    )
    prior.update(
        _world_key(world)
        for role in COUNTS
        for world in policy_v2_worlds(role)
    )
    prior.update(_world_key(world) for world in qualification_worlds())
    return prior


def validate_rosters() -> None:
    calibration = {_world_key(world) for world in continuous_worlds("calibration")}
    evaluation = {_world_key(world) for world in continuous_worlds("evaluation")}
    if (
        len(calibration) != COUNTS["calibration"]
        or len(evaluation) != COUNTS["evaluation"]
        or calibration & evaluation
        or (calibration | evaluation) & opened_world_keys()
    ):
        raise ValueError("fresh disjoint v3 policy-certificate rosters required")


def prefix_batch_count(role: str) -> int:
    if role not in COUNTS or COUNTS[role] % 8:
        raise ValueError("registered eight-world prefix batching required")
    return COUNTS[role] // 8


def prefix_task(role: str, batch: int) -> dict[str, Any]:
    count = prefix_batch_count(role)
    if type(batch) is not int or batch not in range(count):
        raise ValueError("registered prefix batch required")
    indices = list(range(8 * batch, 8 * batch + 8))
    return {
        "kind": "prefix_only",
        "name": f"{role}-prefix-{batch:02d}",
        "role": role,
        "batch": batch,
        "world_indices": indices,
    }


def future_action_task(role: str, world_index: int, action_index: int) -> dict[str, Any]:
    if (
        role not in COUNTS
        or type(world_index) is not int
        or world_index not in range(COUNTS[role])
        or type(action_index) is not int
        or action_index not in range(ACTION_COUNT)
    ):
        raise ValueError("registered independent future action required")
    return {
        "kind": "independent_action_future",
        "name": f"{role}-future-{world_index:03d}-action-{action_index:02d}",
        "role": role,
        "world_index": world_index,
        "action_index": action_index,
    }


def sensor_errors(role: str) -> NDArray[np.float64]:
    """Generate one new shared-bias plus iid-noise draw per v3 world."""

    if role not in COUNTS:
        raise ValueError("unknown sensor partition")
    rng = np.random.default_rng(SENSOR_SEEDS[role])
    bias = rng.normal(0.0, BIAS_STD_M, (COUNTS[role], 1, 1, 3))
    noise = rng.normal(0.0, NOISE_STD_M, (COUNTS[role], 3, 4, 3))
    return bias + noise


def candidate_predictions(
    role: str,
    truth_prefix: object,
    bank_prefix: object,
    bank_reward: object,
    predictor: LocalPolicyGainPredictor,
) -> dict[str, Array]:
    """Apply the unchanged posterior policy to the fresh v3 observations."""

    truth = np.asarray(truth_prefix, dtype=np.float64)
    model_prefix = np.asarray(bank_prefix, dtype=np.float64)
    model_reward = np.asarray(bank_reward, dtype=np.float64)
    if (
        role not in COUNTS
        or truth.shape != (COUNTS[role], 3, 4, 3)
        or model_prefix.shape != (27, 3, 4, 3)
        or model_reward.shape != (27, 7)
        or any(
            not np.all(np.isfinite(value))
            for value in (truth, model_prefix, model_reward)
        )
    ):
        raise ValueError("complete finite v3 prediction inputs required")
    observation = truth + sensor_errors(role)
    inferred = [infer(row, model_prefix, model_reward) for row in observation]
    expected = np.stack([value["expected_losses"] for value in inferred])
    mean_raw_upper = np.stack([value["raw_upper"][0] for value in inferred])
    actions = posterior_policy_action(expected)
    features = np.stack(
        [
            combined_competence_features(row, value)
            for row, value in zip(observation, inferred, strict=True)
        ]
    )
    prediction = predict_distance_weighted_local_policy_gain(
        predictor,
        query_features=features,
        candidate_actions=actions,
    )
    return {
        "truth_prefix_m": truth,
        "observation_m": observation,
        "features": features,
        "expected_losses": expected,
        "mean_raw_upper": mean_raw_upper,
        "candidate_actions": actions,
        "predicted_gain": prediction.predicted_gain,
        "neighbor_indices": prediction.neighbor_indices,
        "neighbor_squared_distances": prediction.neighbor_squared_distances,
    }


def score(
    candidate: dict[str, Array],
    guarded: dict[str, Array],
    rewards: object,
    calibration: PolicyGainCalibration,
    simultaneous_calibration: RegretCalibration,
    *,
    all_native_qa: bool,
    pre_future_gate_passed: bool,
) -> dict[str, Any]:
    """Apply the unchanged v2 statistical scorer to the fresh v3 denominator."""

    result = score_v2(
        candidate,
        guarded,
        rewards,
        calibration,
        simultaneous_calibration,
        all_native_qa=all_native_qa,
        pre_future_gate_passed=pre_future_gate_passed,
    )
    result["schema"] = "dlolab-slingshot-policy-certificate-result-v3"
    return cast(dict[str, Any], result)


def protocol() -> dict[str, Any]:
    validate_rosters()
    return {
        "schema": "dlolab-slingshot-policy-certificate-source-v3",
        "role": "prospective_public_simulator_policy_certificate",
        "qualification": {
            "result_id": QUALIFICATION_RESULT_ID,
            "summary_sha256": QUALIFICATION_RESULT_SHA256,
            "ordinary_processes": 64,
            "qualified_worlds": 8,
            "passed": True,
        },
        "parent_v2": {
            "result_id": (
                "e00de6c8b7a82fa13ee1076e05e52ac1ea472de6a042e395a8f3490e8703c016"
            ),
            "status": "retained_evaluation_native_qa_failure",
            "ordinary_futures": 286,
            "technical_failures": 2,
            "partial_score_authorized": False,
            "retry_authorized": False,
        },
        "reference": {
            "world_count": 147,
            "status": "unchanged_opened_source_training_only",
            "feature": (
                "shared_bias_invariant_geometry_plus_residual_independent_"
                "bayesian_posterior_diagnostics"
            ),
            "neighbor_count": NEIGHBOR_COUNT,
        },
        "partitions": {
            role: {
                "worlds": continuous_worlds(role),
                "world_seed": WORLD_SEEDS[role],
                "sensor_seed": SENSOR_SEEDS[role],
                "count": COUNTS[role],
            }
            for role in COUNTS
        },
        "all_rosters_mutually_disjoint": True,
        "world_distribution": {
            "x_offset_m": "uniform[-0.02,0.02]",
            "bending_E": "log_uniform[50000,200000]",
            "stretching_K": "log_uniform[400000,1600000]",
        },
        "statistical_unit": "one_fresh_continuous_world_and_one_sensor_draw",
        "candidate_policy": "unchanged_frozen_posterior_predictive_mean_action",
        "calibration": {
            "primary_kind": "one_sided_split_conformal_selected_policy_gain",
            "comparator_kind": "simultaneous_split_conformal_mean_action_regret",
            "miscoverage": MISCOVERAGE,
            "count": COUNTS["calibration"],
            "rank": CALIBRATION_RANK,
        },
        "execution": {
            "prefix": "qualified_eight_world_causal_batch",
            "future": "one_world_one_action_per_fresh_python_process",
            "actions_per_world": ACTION_COUNT,
            "future_process_count": FUTURE_PROCESS_COUNT,
            "world_qa_before_calibration_or_scoring": True,
            "complete_world_requires_all_eight_actions": True,
        },
        "fallback_action": BASELINE,
        "harm_margin": REWARD_MARGIN,
        "arms": list(ARM_NAMES),
        "primary_arm": "policy_gain_guard",
        "matched_comparator_arm": "simultaneous_mean_regret_guard",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "harm_risk_budget": HARM_RISK_BUDGET,
        "pre_evaluation_future_gate": {
            "all_calibration_and_evaluation_prefixes_qualified": True,
            "calibration_complete": True,
            "accepted_evaluation_worlds_at_least": 24,
            "fallback_evaluation_worlds_at_least": 24,
        },
        "source_gate": {
            "complete_288_world_denominator": True,
            "mean_guarded_gain_at_least": 0.002,
            "paired_bootstrap_gain_ci95_lower_above": 0.0,
            "mean_gain_over_simultaneous_guard_at_least": 0.001,
            "paired_bootstrap_gain_vs_simultaneous_ci95_lower_above": 0.0,
            "marginal_policy_gain_coverage_at_least": 0.85,
            "simultaneous_action_coverage_at_least": 0.85,
            "harm_probability_upper95_at_most": HARM_RISK_BUDGET,
            "retained_posterior_gain_fraction_at_least": 0.10,
            "posterior_harmed_worlds_at_least": 10,
            "guard_removes_harmed_worlds_at_least": 5,
            "oracle_headroom_fraction_at_least": 0.05,
        },
        "stage_order": [
            "calibration_prefixes_and_candidate_predictions",
            "calibration_independent_action_futures",
            "calibration_world_qualification",
            "single_conformal_offset",
            "evaluation_prefixes_and_guarded_decisions",
            "evaluation_decision_barrier",
            "evaluation_independent_action_futures",
            "evaluation_world_qualification",
            "score",
        ],
        "technical_failure_policy": (
            "retain_failure_and_stop_without_partial_score_retry_or_replacement"
        ),
        "evaluation_future_before_decision_barrier": False,
        "retry_authorized": False,
        "replacement_authorized": False,
        "v2_world_retry_authorized": False,
        "v2_partial_outcome_scoring_authorized": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
        "gpu_work": False,
        "push_or_merge": False,
    }
