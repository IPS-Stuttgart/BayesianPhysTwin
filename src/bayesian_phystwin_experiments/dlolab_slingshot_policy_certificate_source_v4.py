"""Reward-aligned stochastic-execution Slingshot policy certificate."""

from __future__ import annotations

import copy
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.policy_gain_certificate import (
    LocalPolicyGainPredictor,
    PolicyGainCalibration,
    predict_distance_weighted_local_policy_gain,
)

from .coupled_action_regret import RegretCalibration
from .dlolab_slingshot_belief import BASELINE, BIAS_STD_M, NOISE_STD_M, infer
from .dlolab_slingshot_cmaes import task_metrics
from .dlolab_slingshot_independent_native_v3 import independent_world_qa
from .dlolab_slingshot_policy_certificate_source_v3 import (
    ACTION_COUNT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CALIBRATION_RANK,
    COUNTS,
    HARM_RISK_BUDGET,
)
from .dlolab_slingshot_policy_certificate_source_v3 import (
    calibrate as calibrate,
)
from .dlolab_slingshot_policy_certificate_source_v3 import (
    calibrate_simultaneous_guard as calibrate_simultaneous_guard,
)
from .dlolab_slingshot_policy_certificate_source_v3 import (
    continuous_worlds as policy_v3_worlds,
)
from .dlolab_slingshot_policy_certificate_source_v3 import (
    guarded_decisions as guarded_decisions,
)
from .dlolab_slingshot_policy_certificate_source_v3 import (
    opened_world_keys as prior_opened_world_keys,
)
from .dlolab_slingshot_policy_certificate_source_v3 import (
    pre_future_checks as pre_future_checks,
)
from .dlolab_slingshot_policy_certificate_source_v3 import protocol as protocol_v3
from .dlolab_slingshot_policy_certificate_source_v3 import (
    score as score_v3,
)
from .dlolab_slingshot_policy_certificate_v1 import posterior_policy_action
from .dlolab_slingshot_policy_certificate_v2 import (
    NEIGHBOR_COUNT,
    combined_competence_features,
)

Array: TypeAlias = NDArray[Any]
WORLD_SEEDS = {"calibration": 262080, "evaluation": 262081}
SENSOR_SEEDS = {"calibration": 262082, "evaluation": 262083}
QUALIFICATION_RESULT_ID = (
    "35d06883b7f5192127e59db7c6da8693eb38ee2083af00e24158b0a079fa998b"
)
QUALIFICATION_RESULT_SHA256 = (
    "711d0808acd56d710128cd74b9189b47092dec7fcf600b20aaf3f7f8bfc54901"
)


def _world_key(world: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(world["x_offset_m"]),
        float(world["bending_E"]),
        float(world["stretching_K"]),
    )


def continuous_worlds(role: str) -> list[dict[str, Any]]:
    """Return the fixed fresh v4 roster."""

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
    prior = set(
        cast(set[tuple[float, float, float]], prior_opened_world_keys())
    )
    prior.update(
        _world_key(world)
        for role in COUNTS
        for world in policy_v3_worlds(role)
    )
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
        raise ValueError("fresh disjoint v4 policy-certificate rosters required")


def prefix_batch_count(role: str) -> int:
    if role not in COUNTS or COUNTS[role] % 8:
        raise ValueError("registered eight-world prefix batching required")
    return int(COUNTS[role] // 8)


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
        raise ValueError("complete finite v4 prediction inputs required")
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


def reward_aligned_world_qa(
    rows: list[dict[str, Array]],
    reports: list[dict[str, Any]],
    expected_controls: Array,
    world: dict[str, Any],
    *,
    world_count: int,
) -> dict[str, Any]:
    """Require reward repeatability while reporting state-process divergence."""

    base = independent_world_qa(
        rows,
        reports,
        expected_controls,
        world,
        world_count=world_count,
    )
    deterministic_position = bool(base["checks"]["duplicate_positions"])
    checks = {
        name: bool(value)
        for name, value in base["checks"].items()
        if name != "duplicate_positions"
    }
    checks["duplicate_position_reported"] = True
    return {
        **base,
        "checks": checks,
        "qa_passed": bool(all(checks.values())),
        "duplicate_position_deterministic": deterministic_position,
        "execution_estimand": "reward_under_one_native_process_realization",
    }


def reward_aligned_world_rewards(rows: list[dict[str, Array]]) -> Array:
    """Average the two independent incumbent replicas in the score input."""

    if len(rows) != ACTION_COUNT:
        raise ValueError("all eight action rows are required for world rewards")
    rewards = np.asarray(
        [task_metrics(row)["native_reward"] for row in rows[:7]],
        dtype=np.float64,
    )
    rewards[BASELINE] = np.mean(
        [
            task_metrics(rows[BASELINE])["native_reward"],
            task_metrics(rows[7])["native_reward"],
        ]
    )
    return rewards


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
    result = score_v3(
        candidate,
        guarded,
        rewards,
        calibration,
        simultaneous_calibration,
        all_native_qa=all_native_qa,
        pre_future_gate_passed=pre_future_gate_passed,
    )
    result["schema"] = "dlolab-slingshot-policy-certificate-result-v4"
    result["execution_estimand"] = "reward_under_registered_native_process_draw"
    result["incumbent_reward_estimator"] = "mean_of_action_slots_5_and_7"
    return cast(dict[str, Any], result)


def protocol() -> dict[str, Any]:
    validate_rosters()
    value = copy.deepcopy(protocol_v3())
    value["schema"] = "dlolab-slingshot-policy-certificate-source-v4"
    value["role"] = "prospective_reward_aligned_stochastic_execution_certificate"
    value["qualification"] = {
        "result_id": QUALIFICATION_RESULT_ID,
        "summary_sha256": QUALIFICATION_RESULT_SHA256,
        "source_worlds": 128,
        "ordinary_action_processes": 1024,
        "reward_aligned_worlds": 128,
        "passed": True,
    }
    value["parent_v3"] = {
        "result_id": (
            "89e7a37827fadd0e3a25d81fd67a7a1685447e2240d1f045bb20b5fc709a26d4"
        ),
        "status": "retained_calibration_world_qa_failure",
        "rescored": False,
        "roster_reused": False,
    }
    value["partitions"] = {
        role: {
            "worlds": continuous_worlds(role),
            "world_seed": WORLD_SEEDS[role],
            "sensor_seed": SENSOR_SEEDS[role],
            "count": COUNTS[role],
        }
        for role in COUNTS
    }
    value["statistical_unit"] = (
        "one_fresh_continuous_world_one_sensor_draw_and_native_process_draw"
    )
    value["execution"].update(
        {
            "world_qa": "reward_aligned_stochastic_execution",
            "duplicate_reward_error_at_most": 0.001,
            "duplicate_position": "reported_not_admission",
            "incumbent_reward_estimator": "mean_of_action_slots_5_and_7",
            "state_process_variability_reported": True,
        }
    )
    value["technical_failure_policy"] = (
        "retain_missing_malformed_or_reward_unrepeatable_execution_without_partial_score"
    )
    value["bootstrap_seed"] = BOOTSTRAP_SEED
    value["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    value["harm_risk_budget"] = HARM_RISK_BUDGET
    value["calibration"]["rank"] = CALIBRATION_RANK
    value["reference"]["neighbor_count"] = NEIGHBOR_COUNT
    return cast(dict[str, Any], value)
