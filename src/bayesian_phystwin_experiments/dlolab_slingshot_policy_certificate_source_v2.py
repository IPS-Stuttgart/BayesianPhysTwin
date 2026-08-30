"""Prospective public-simulator Slingshot policy-gain certification."""

from __future__ import annotations

import dataclasses
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound
from bayesian_phystwin.policy_gain_certificate import (
    LocalPolicyGainPredictor,
    PolicyGainCalibration,
    apply_policy_gain_guard,
    calibrate_policy_gain_lower_bound,
    predict_distance_weighted_local_policy_gain,
)

from .coupled_action_regret import (
    RegretCalibration,
    calibrate_simultaneous_regret,
    guarded_action,
)
from .dlolab_slingshot_active_bayes import continuous_worlds as active_v1_worlds
from .dlolab_slingshot_active_bayes_v2 import continuous_worlds as active_v2_worlds
from .dlolab_slingshot_belief import (
    BASELINE,
    BIAS_STD_M,
    NOISE_STD_M,
    ORDER,
    REWARD_MARGIN,
    infer,
    particle_worlds,
    sample_worlds,
)
from .dlolab_slingshot_certified_guard_v2 import continuous_worlds as guard_v2_worlds
from .dlolab_slingshot_policy_certificate_source_v1 import (
    continuous_worlds as policy_v1_worlds,
)
from .dlolab_slingshot_policy_certificate_v1 import (
    MISCOVERAGE,
    posterior_policy_action,
)
from .dlolab_slingshot_policy_certificate_v2 import (
    NEIGHBOR_COUNT,
    combined_competence_features,
)
from .dlolab_slingshot_value import worlds as source_worlds

Array: TypeAlias = NDArray[Any]
COUNTS = {"calibration": 128, "evaluation": 288}
WORLD_SEEDS = {"calibration": 262050, "evaluation": 262051}
SENSOR_SEEDS = {"calibration": 262052, "evaluation": 262053}
BOOTSTRAP_SEED = 262054
BOOTSTRAP_REPLICATES = 20_000
HARM_RISK_BUDGET = 0.05
CALIBRATION_RANK = 117
ARM_NAMES = (
    "incumbent",
    "posterior_predictive_mean",
    "simultaneous_mean_regret_guard",
    "policy_gain_guard",
)


def continuous_worlds(role: str) -> list[dict[str, Any]]:
    """Return the fixed role-specific continuous-world roster."""

    if role not in COUNTS:
        raise ValueError("unknown policy-certificate partition")
    rng = np.random.default_rng(WORLD_SEEDS[role])
    count = COUNTS[role]
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


def _world_key(world: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(world["x_offset_m"]),
        float(world["bending_E"]),
        float(world["stretching_K"]),
    )


def opened_world_keys() -> set[tuple[float, float, float]]:
    """Return every registered Slingshot world that predates this protocol."""

    opened = [
        *source_worlds(),
        *particle_worlds(),
        *sample_worlds("calibration"),
        *sample_worlds("evaluation"),
        *active_v1_worlds(),
        *active_v2_worlds(),
        *guard_v2_worlds(),
        *policy_v1_worlds("calibration"),
        *policy_v1_worlds("evaluation"),
    ]
    return {_world_key(world) for world in opened}


def validate_rosters() -> None:
    calibration = {_world_key(world) for world in continuous_worlds("calibration")}
    evaluation = {_world_key(world) for world in continuous_worlds("evaluation")}
    if (
        len(calibration) != COUNTS["calibration"]
        or len(evaluation) != COUNTS["evaluation"]
        or calibration & evaluation
        or (calibration | evaluation) & opened_world_keys()
    ):
        raise ValueError("fresh disjoint policy-certificate rosters required")


def prefix_batch_count(role: str) -> int:
    if role not in COUNTS or COUNTS[role] % 8:
        raise ValueError("registered eight-world batching required")
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
        "native_world_indices": indices,
    }


def future_task(role: str, index: int) -> dict[str, Any]:
    if role not in COUNTS or type(index) is not int or index not in range(COUNTS[role]):
        raise ValueError("registered policy-certificate future required")
    return {
        "kind": "all_action_future",
        "name": f"{role}-future-{index:03d}",
        "role": role,
        "world_index": index,
    }


def sensor_errors(role: str) -> NDArray[np.float64]:
    """Generate one registered shared-bias plus iid-noise draw per world."""

    if role not in COUNTS:
        raise ValueError("unknown sensor partition")
    rng = np.random.default_rng(SENSOR_SEEDS[role])
    bias = rng.normal(0.0, BIAS_STD_M, (COUNTS[role], 1, 1, 3))
    noise = rng.normal(0.0, NOISE_STD_M, (COUNTS[role], 3, 4, 3))
    return bias + noise


def protocol() -> dict[str, Any]:
    validate_rosters()
    return {
        "schema": "dlolab-slingshot-policy-certificate-source-v2",
        "role": "prospective_public_simulator_posterior_aware_policy_certificate",
        "reference": {
            "world_count": 147,
            "groups": {"parent_opened": 51, "policy_v1_calibration_opened": 96},
            "status": "previously_opened_source_training_only",
            "development_artifact_id": (
                "5b8e50986f1f7dc7785389fa840a2e0993cc8bcaa5a5c3d8095567ff4c81e682"
            ),
            "outcomes_used_for_local_prediction_only": True,
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
        "candidate_policy": "frozen_posterior_predictive_mean_action",
        "feature": (
            "shared_bias_invariant_geometry_plus_residual_independent_"
            "bayesian_posterior_diagnostics"
        ),
        "local_predictor": {
            "kind": "standardized_inverse_distance_nearest_neighbor_gain",
            "neighbor_count": NEIGHBOR_COUNT,
            "reference_scaling_only": True,
            "canonical_reference_order": True,
            "exact_match_rule": "average_only_zero_distance_reference_rows",
        },
        "calibration": {
            "primary_kind": "one_sided_split_conformal_selected_policy_gain",
            "comparator_kind": "simultaneous_split_conformal_mean_action_regret",
            "miscoverage": MISCOVERAGE,
            "count": COUNTS["calibration"],
            "rank": CALIBRATION_RANK,
            "same_worlds_and_all_action_futures_for_both_calibrators": True,
            "calibration_can_change_policy": False,
            "calibration_can_change_predictor": False,
        },
        "fallback_action": BASELINE,
        "harm_margin": REWARD_MARGIN,
        "harm_event": "guarded_gain_below_negative_0_002",
        "arms": list(ARM_NAMES),
        "primary_arm": "policy_gain_guard",
        "matched_comparator_arm": "simultaneous_mean_regret_guard",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "harm_risk_interval": "one_sided_exact_clopper_pearson_over_all_worlds",
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
            "paired_bootstrap_gain_vs_simultaneous_guard_ci95_lower_above": 0.0,
            "mean_gain_over_simultaneous_guard_at_least": 0.001,
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
            "calibration_futures",
            "single_conformal_offset",
            "evaluation_prefixes_and_guarded_decisions",
            "evaluation_decision_barrier",
            "evaluation_futures",
            "score",
        ],
        "evaluation_future_before_decision_barrier": False,
        "retry_authorized": False,
        "replacement_authorized": False,
        "policy_v1_unopened_evaluation_futures_used": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
        "gpu_work": False,
        "push_or_merge": False,
    }


def candidate_predictions(
    role: str,
    truth_prefix: object,
    bank_prefix: object,
    bank_reward: object,
    predictor: LocalPolicyGainPredictor,
) -> dict[str, Array]:
    """Infer the fixed posterior policy and its source-local gain prediction."""

    truth = np.asarray(truth_prefix, dtype=np.float64)
    model_prefix = np.asarray(bank_prefix, dtype=np.float64)
    model_reward = np.asarray(bank_reward, dtype=np.float64)
    if (
        role not in COUNTS
        or truth.shape != (COUNTS[role], 3, 4, 3)
        or model_prefix.shape != (27, 3, 4, 3)
        or model_reward.shape != (27, 7)
        or any(not np.all(np.isfinite(value)) for value in (truth, model_prefix, model_reward))
    ):
        raise ValueError("complete finite policy-certificate prediction inputs required")
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


def calibrate(
    candidate: dict[str, Array], rewards: object
) -> tuple[PolicyGainCalibration, NDArray[np.float64]]:
    """Fit the only permitted scalar from the calibration futures."""

    reward = np.asarray(rewards, dtype=np.float64)
    actions = np.asarray(candidate["candidate_actions"], dtype=np.int64)
    predicted = np.asarray(candidate["predicted_gain"], dtype=np.float64)
    if reward.shape != (COUNTS["calibration"], 7) or actions.shape != predicted.shape:
        raise ValueError("complete aligned calibration futures required")
    realized = reward[np.arange(len(reward)), actions] - reward[:, BASELINE]
    result = calibrate_policy_gain_lower_bound(
        predicted_gain=predicted,
        realized_gain=realized,
        miscoverage=MISCOVERAGE,
    )
    if (result.calibration_count, result.rank) != (
        COUNTS["calibration"],
        CALIBRATION_RANK,
    ):
        raise ValueError("registered conformal calibration arithmetic changed")
    return result, realized


def calibrate_simultaneous_guard(
    candidate: dict[str, Array], rewards: object
) -> RegretCalibration:
    """Fit the matched maximum-over-actions comparator on the same futures."""

    reward = np.asarray(rewards, dtype=np.float64)
    raw_upper = np.asarray(candidate["mean_raw_upper"], dtype=np.float64)
    if (
        reward.shape != (COUNTS["calibration"], 7)
        or raw_upper.shape != (COUNTS["calibration"], len(ORDER))
    ):
        raise ValueError("complete aligned simultaneous-regret calibration required")
    result = calibrate_simultaneous_regret(
        raw_upper,
        -reward[:, ORDER],
        coverage=1.0 - MISCOVERAGE,
    )
    if (result.count, result.rank, result.offset is None) != (
        COUNTS["calibration"],
        CALIBRATION_RANK,
        False,
    ):
        raise ValueError("registered simultaneous calibration arithmetic changed")
    return result


def guarded_decisions(
    candidate: dict[str, Array],
    calibration: PolicyGainCalibration,
    simultaneous_calibration: RegretCalibration,
) -> dict[str, Array]:
    """Apply the frozen lower-bound admission rule to evaluation candidates."""

    actions = np.asarray(candidate["candidate_actions"], dtype=np.int64)
    predicted = np.asarray(candidate["predicted_gain"], dtype=np.float64)
    expected = np.asarray(candidate["expected_losses"], dtype=np.float64)
    raw_upper = np.asarray(candidate["mean_raw_upper"], dtype=np.float64)
    if (
        len(actions) != COUNTS["evaluation"]
        or predicted.shape != actions.shape
        or expected.shape != (COUNTS["evaluation"], len(ORDER))
        or raw_upper.shape != expected.shape
    ):
        raise ValueError("complete evaluation candidates required")
    guard = apply_policy_gain_guard(
        candidate_actions=actions,
        predicted_gain=predicted,
        calibration=calibration,
        fallback_action=BASELINE,
        harm_margin=REWARD_MARGIN,
    )
    decisions = np.empty((COUNTS["evaluation"], len(ARM_NAMES)), dtype=np.int64)
    order = np.asarray(ORDER, dtype=np.int64)
    simultaneous = np.asarray(
        [
            order[guarded_action(expected[index], raw_upper[index], simultaneous_calibration)]
            for index in range(COUNTS["evaluation"])
        ],
        dtype=np.int64,
    )
    decisions[:, 0] = BASELINE
    decisions[:, 1] = actions
    decisions[:, 2] = simultaneous
    decisions[:, 3] = guard.selected_actions
    return {
        "decisions": decisions,
        "accepted_mask": guard.accepted_mask,
        "simultaneous_accepted_mask": simultaneous != BASELINE,
        "lower_gain_bound": guard.lower_gain_bound,
    }


def pre_future_checks(
    guarded: dict[str, Array], *, all_prefix_qa: bool
) -> dict[str, Any]:
    decision = np.asarray(guarded["decisions"])
    accepted = np.asarray(guarded["accepted_mask"])
    if (
        decision.shape != (COUNTS["evaluation"], len(ARM_NAMES))
        or accepted.shape != (COUNTS["evaluation"],)
        or accepted.dtype.kind != "b"
        or np.any(decision[:, 0] != BASELINE)
        or np.any(decision[~accepted, 3] != BASELINE)
        or np.any(decision[accepted, 3] != decision[accepted, 1])
    ):
        raise ValueError("invalid guarded evaluation decisions")
    accepted_count = int(np.count_nonzero(accepted))
    fallback_count = int(len(accepted) - accepted_count)
    checks = {
        "all_prefixes_qualified": bool(all_prefix_qa),
        "accepted_worlds_at_least_24": accepted_count >= 24,
        "fallback_worlds_at_least_24": fallback_count >= 24,
    }
    return {
        "accepted_worlds": accepted_count,
        "fallback_worlds": fallback_count,
        "checks": checks,
        "pre_future_gate_passed": bool(all(checks.values())),
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
    """Score the complete sealed 288-world evaluation denominator."""

    reward = np.asarray(rewards, dtype=np.float64)
    decision = np.asarray(guarded["decisions"], dtype=np.int64)
    lower = np.asarray(guarded["lower_gain_bound"], dtype=np.float64)
    if (
        reward.shape != (COUNTS["evaluation"], 7)
        or decision.shape != (COUNTS["evaluation"], len(ARM_NAMES))
        or lower.shape != (COUNTS["evaluation"],)
        or not np.all(np.isfinite(reward))
    ):
        raise ValueError("complete finite evaluation denominator required")
    selected = np.take_along_axis(reward, decision, axis=1)
    incumbent = reward[:, BASELINE]
    gain = selected - incumbent[:, None]
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        COUNTS["evaluation"],
        (BOOTSTRAP_REPLICATES, COUNTS["evaluation"]),
    )
    arms: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        harms = int(np.count_nonzero(gain[:, index] < -REWARD_MARGIN))
        arms[name] = {
            "mean_native_reward": float(selected[:, index].mean()),
            "mean_gain_over_incumbent": float(gain[:, index].mean()),
            "mean_gain_ci95": np.quantile(
                gain[bootstrap, index].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "updated_worlds": int(np.count_nonzero(decision[:, index] != BASELINE)),
            "harmful_worlds_beyond_numeric_margin": harms,
            "harm_probability_upper95": one_sided_binomial_upper_bound(
                harms, COUNTS["evaluation"], 0.95
            ),
            "mean_downside_below_incumbent": float(
                np.maximum(-gain[:, index], 0.0).mean()
            ),
        }
    posterior = arms["posterior_predictive_mean"]
    simultaneous = arms["simultaneous_mean_regret_guard"]
    guard = arms["policy_gain_guard"]
    candidate_actions = np.asarray(candidate["candidate_actions"], dtype=np.int64)
    candidate_gain = reward[np.arange(len(reward)), candidate_actions] - incumbent
    coverage = float(np.mean(candidate_gain >= lower))
    raw_upper = np.asarray(candidate["mean_raw_upper"], dtype=np.float64)
    if raw_upper.shape != (COUNTS["evaluation"], len(ORDER)):
        raise ValueError("complete simultaneous-regret evaluation bounds required")
    realized_regret = incumbent[:, None] - reward[:, ORDER]
    if simultaneous_calibration.offset is None:
        raise ValueError("registered simultaneous calibration must have an offset")
    simultaneous_coverage = float(
        np.mean(
            np.all(
                realized_regret[:, 1:]
                <= raw_upper[:, 1:] + simultaneous_calibration.offset,
                axis=1,
            )
        )
    )
    posterior_gain = float(posterior["mean_gain_over_incumbent"])
    simultaneous_gain = float(simultaneous["mean_gain_over_incumbent"])
    guard_gain = float(guard["mean_gain_over_incumbent"])
    oracle_gain = float(reward.max(axis=1).mean() - incumbent.mean())
    retained = guard_gain / posterior_gain if posterior_gain > 0.0 else 0.0
    headroom = guard_gain / oracle_gain if oracle_gain > 0.0 else 0.0
    harm_reduction = int(
        posterior["harmful_worlds_beyond_numeric_margin"]
        - guard["harmful_worlds_beyond_numeric_margin"]
    )
    arm_index = {name: index for index, name in enumerate(ARM_NAMES)}
    paired_vs_simultaneous = (
        gain[:, arm_index["policy_gain_guard"]]
        - gain[:, arm_index["simultaneous_mean_regret_guard"]]
    )
    paired_vs_simultaneous_ci = np.quantile(
        paired_vs_simultaneous[bootstrap].mean(axis=1), [0.025, 0.975]
    ).tolist()
    gain_over_simultaneous = guard_gain - simultaneous_gain
    checks = {
        "complete_denominator": True,
        "all_native_qa": bool(all_native_qa),
        "pre_future_gate_passed": bool(pre_future_gate_passed),
        "guard_gain_at_least_0_002": guard_gain >= 0.002,
        "positive_paired_ci95_vs_incumbent": guard["mean_gain_ci95"][0] > 0.0,
        "gain_over_simultaneous_guard_at_least_0_001": (
            gain_over_simultaneous >= 0.001
        ),
        "positive_paired_ci95_vs_simultaneous_guard": (
            paired_vs_simultaneous_ci[0] > 0.0
        ),
        "policy_gain_coverage_at_least_0_85": coverage >= 0.85,
        "simultaneous_action_coverage_at_least_0_85": (
            simultaneous_coverage >= 0.85
        ),
        "guard_harm_upper_at_most_0_05": (
            guard["harm_probability_upper95"] <= HARM_RISK_BUDGET
        ),
        "guard_retains_at_least_10pct_posterior_gain": retained >= 0.10,
        "posterior_harmed_worlds_at_least_10": (
            posterior["harmful_worlds_beyond_numeric_margin"] >= 10
        ),
        "guard_removes_at_least_5_harmed_worlds": harm_reduction >= 5,
        "guard_captures_at_least_5pct_oracle_headroom": headroom >= 0.05,
    }
    return {
        "schema": "dlolab-slingshot-policy-certificate-result-v2",
        "arms": arms,
        "calibration": dataclasses.asdict(calibration),
        "simultaneous_calibration": dataclasses.asdict(simultaneous_calibration),
        "marginal_policy_gain_coverage": coverage,
        "simultaneous_action_coverage": simultaneous_coverage,
        "policy_guard_gain_over_simultaneous_guard": gain_over_simultaneous,
        "policy_guard_paired_gain_vs_simultaneous_guard_ci95": (
            paired_vs_simultaneous_ci
        ),
        "harm_reduction_vs_posterior": harm_reduction,
        "guard_fraction_of_posterior_gain": retained,
        "guard_fraction_of_oracle_headroom": headroom,
        "oracle_gain_over_incumbent": oracle_gain,
        "checks": checks,
        "source_gate_passed": bool(all(checks.values())),
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
    }
