"""Prospective joint certificate for two public-simulator decision queries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._portable_contracts import content_id
from .guard_harm_risk import one_sided_binomial_upper_bound

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

SCHEMA = "bayesian_phystwin.query_portfolio_replication"
VERSION = 1
QUERY_IDS = ("dlolab_wrapping_v9", "dlolab_slingshot_v4")
WORLD_COUNT = 320
REWARD_MARGIN = 0.002
OVERALL_ALPHA = 0.05
GAIN_FAMILY_ALPHA = 0.01
HARM_FAMILY_ALPHA = 0.04
GAIN_QUERY_ALPHA = GAIN_FAMILY_ALPHA / len(QUERY_IDS)
HARM_QUERY_ALPHA = HARM_FAMILY_ALPHA / len(QUERY_IDS)
HARM_BUDGET = 0.05
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 263_100
WORLD_SEEDS = {
    "dlolab_wrapping_v9": 263_101,
    "dlolab_slingshot_v4": 263_102,
}
SENSOR_SEEDS = {
    "dlolab_wrapping_v9": 263_103,
    "dlolab_slingshot_v4": 263_104,
}


@dataclass(frozen=True, slots=True)
class QueryOutcomeV1:
    """Complete outcome for one frozen decision query."""

    query_id: str
    gain: FloatArray
    candidate_deployed: BoolArray
    ordinary_success: BoolArray

    def __post_init__(self) -> None:
        if self.query_id not in QUERY_IDS:
            raise ValueError("query_id is not registered")
        gain = np.ascontiguousarray(self.gain, dtype=np.float64)
        deployed = np.ascontiguousarray(self.candidate_deployed, dtype=np.bool_)
        success = np.ascontiguousarray(self.ordinary_success, dtype=np.bool_)
        if gain.shape != (WORLD_COUNT,):
            raise ValueError("one gain per registered world required")
        if deployed.shape != gain.shape or success.shape != gain.shape:
            raise ValueError("deployment and custody vectors must match gain")
        if not np.isfinite(gain).all():
            raise ValueError("all registered gains must be finite")
        if np.any(~success):
            raise ValueError("complete ordinary-success denominator required")
        if np.any(~deployed & (gain != 0.0)):
            raise ValueError("fallback worlds must have exact zero gain")
        gain.setflags(write=False)
        deployed.setflags(write=False)
        success.setflags(write=False)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "candidate_deployed", deployed)
        object.__setattr__(self, "ordinary_success", success)


def protocol() -> dict[str, Any]:
    """Return the outcome-blind registered design."""

    value: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "role": "prospective_two_query_public_simulator_replication",
        "queries": list(QUERY_IDS),
        "worlds_per_query": WORLD_COUNT,
        "fresh_worlds": True,
        "frozen_component_policies": {
            "dlolab_wrapping_v9": {
                "source_evidence_id": (
                    "d3c577ce1ec215c6d56c4d405e7f9d886f38b7e6d021bb6d62f37da6bd4784b9"
                ),
                "policy": "posterior_975_guard",
            },
            "dlolab_slingshot_v4": {
                "source_evidence_id": (
                    "2882809b7265714a93be2d3f1455eeac527adbe681cc990cde762777fcaf3a85"
                ),
                "policy": "policy_gain_guard",
            },
        },
        "world_seeds": WORLD_SEEDS,
        "sensor_seeds": SENSOR_SEEDS,
        "reward_margin": REWARD_MARGIN,
        "overall_alpha": OVERALL_ALPHA,
        "allocation": {
            "gain_family_alpha": GAIN_FAMILY_ALPHA,
            "gain_per_query_alpha": GAIN_QUERY_ALPHA,
            "harm_family_alpha": HARM_FAMILY_ALPHA,
            "harm_per_query_alpha": HARM_QUERY_ALPHA,
        },
        "gain_test": {
            "estimand": "query_specific_mean_reward_gain_over_exact_fallback",
            "bootstrap": "world_level_percentile_lower_bound",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "pass": "lower_bound_strictly_greater_than_zero_for_both_queries",
        },
        "harm_test": {
            "event": "gain_strictly_below_negative_reward_margin",
            "upper_bound": "one_sided_clopper_pearson",
            "budget": HARM_BUDGET,
            "pass": "upper_bound_at_most_budget_for_both_queries",
        },
        "joint_claim_confidence": 1.0 - OVERALL_ALPHA,
        "aggregation": "separate_equal_world_query_specific_estimands",
        "cross_task_reward_pooling": False,
        "technical_failure_policy": "terminal_failure_no_replacement_no_partial_claim",
        "fallback": "exact_registered_incumbent",
        "design_diagnostics": {
            "maximum_harmful_worlds_per_query": maximum_harms_allowed(),
            "diagnostic_only": True,
        },
        "outcomes_opened": False,
    }
    value["protocol_id"] = content_id(value)
    return value


def _gain_lower(gain: FloatArray, *, query_index: int) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED + query_index)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, WORLD_COUNT, size=(stop - start, WORLD_COUNT))
        means[start:stop] = gain[indices].mean(axis=1)
    return float(np.quantile(means, GAIN_QUERY_ALPHA))


def score(outcomes: Mapping[str, QueryOutcomeV1]) -> dict[str, Any]:
    """Score the complete registered portfolio after the joint barrier."""

    if set(outcomes) != set(QUERY_IDS):
        raise ValueError("exact registered query set required")
    rows: dict[str, Any] = {}
    all_gain = True
    all_harm = True
    for query_index, query_id in enumerate(QUERY_IDS):
        outcome = outcomes[query_id]
        if outcome.query_id != query_id:
            raise ValueError("outcome key and query_id disagree")
        harmful = int(np.count_nonzero(outcome.gain < -REWARD_MARGIN))
        lower = _gain_lower(outcome.gain, query_index=query_index)
        harm_upper = one_sided_binomial_upper_bound(
            harmful,
            WORLD_COUNT,
            1.0 - HARM_QUERY_ALPHA,
        )
        gain_passed = lower > 0.0
        harm_passed = harm_upper <= HARM_BUDGET
        all_gain &= gain_passed
        all_harm &= harm_passed
        rows[query_id] = {
            "worlds": WORLD_COUNT,
            "candidate_deployed_worlds": int(outcome.candidate_deployed.sum()),
            "exact_fallback_worlds": int((~outcome.candidate_deployed).sum()),
            "mean_gain": float(outcome.gain.mean()),
            "gain_lower_bound": lower,
            "gain_confidence": 1.0 - GAIN_QUERY_ALPHA,
            "harmful_worlds": harmful,
            "harm_upper_bound": harm_upper,
            "harm_confidence": 1.0 - HARM_QUERY_ALPHA,
            "gain_gate_passed": gain_passed,
            "harm_gate_passed": harm_passed,
        }
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}.result",
        "version": VERSION,
        "protocol_id": protocol()["protocol_id"],
        "queries": rows,
        "simultaneous_positive_value_passed": all_gain,
        "simultaneous_harm_control_passed": all_harm,
        "joint_portfolio_claim_passed": all_gain and all_harm,
        "joint_claim_confidence": 1.0 - OVERALL_ALPHA,
        "cross_task_reward_pooling": False,
        "complete_denominator": True,
    }
    result["artifact_id"] = content_id(result)
    return result


def maximum_harms_allowed() -> int:
    """Largest registered harm count whose adjusted upper bound passes."""

    allowed = -1
    for harms in range(WORLD_COUNT + 1):
        upper = one_sided_binomial_upper_bound(
            harms,
            WORLD_COUNT,
            1.0 - HARM_QUERY_ALPHA,
        )
        if upper <= HARM_BUDGET:
            allowed = harms
    if allowed < 0 or not math.isfinite(float(allowed)):
        raise RuntimeError("registered harm design has no feasible count")
    return allowed
