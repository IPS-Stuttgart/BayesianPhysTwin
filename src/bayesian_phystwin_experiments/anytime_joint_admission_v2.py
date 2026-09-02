"""Controlled power and type-I study for joint anytime admission version 2."""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np

SCHEMA: Final = "bayesian-phystwin.anytime-joint-admission-study-v2"
SCHEMA_VERSION: Final = 2


def _probability(value: object, *, label: str) -> float:
    result = float(cast(Any, value))
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    result = int(cast(Any, value))
    if result < 1 or result != float(cast(Any, value)):
        raise ValueError(f"{label} must be a positive integer")
    return result


def _finite_vector(value: object, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty finite vector")
    return result


def _log_mixture(log_wealth: np.ndarray) -> np.ndarray:
    if log_wealth.ndim != 2 or log_wealth.shape[1] == 0:
        raise ValueError("log wealth must be a nonempty matrix")
    terms = log_wealth - math.log(log_wealth.shape[1])
    maximum = np.max(terms, axis=1)
    return maximum + np.log(np.sum(np.exp(terms - maximum[:, None]), axis=1))


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    """Return a Wilson score interval for a binomial proportion."""

    if successes < 0 or total < 1 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _crossing_summary(first_crossing: np.ndarray) -> dict[str, object]:
    crossed = first_crossing >= 0
    count = int(np.sum(crossed))
    total = len(first_crossing)
    observed = first_crossing[crossed]
    return {
        "replication_count": total,
        "crossing_count": count,
        "crossing_probability": count / total,
        "wilson_95_interval": wilson_interval(count, total),
        "median_first_crossing": (
            None if count == 0 else float(np.median(observed))
        ),
        "first_crossing_quantiles_10_90": (
            None
            if count == 0
            else [float(value) for value in np.quantile(observed, (0.10, 0.90))]
        ),
    }


def simulate_discrete_admission_scenario(
    *,
    probabilities: np.ndarray,
    gain_scores: np.ndarray,
    harmful: np.ndarray,
    replication_count: int,
    horizon: int,
    minimum_resolved_trials: int,
    shared_epoch_alpha: float,
    gain_bet_fractions: np.ndarray,
    maximum_harm_rate: float,
    harm_alternative_fractions: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Compare shared-alpha IUT and Bonferroni-split admission on one stream."""

    probabilities = _finite_vector(probabilities, label="probabilities")
    gain_scores = _finite_vector(gain_scores, label="gain_scores")
    harmful = np.asarray(harmful, dtype=np.bool_)
    if harmful.ndim != 1 or not (
        len(probabilities) == len(gain_scores) == len(harmful)
    ):
        raise ValueError("scenario atoms must have equal nonzero length")
    if np.any(probabilities <= 0.0) or not math.isclose(
        float(np.sum(probabilities)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("probabilities must be positive and sum to one")
    if np.any(gain_scores < -1.0) or np.any(gain_scores > 1.0):
        raise ValueError("gain scores must lie in [-1, 1]")
    replications = _positive_integer(replication_count, label="replication_count")
    steps = _positive_integer(horizon, label="horizon")
    minimum = _positive_integer(
        minimum_resolved_trials,
        label="minimum_resolved_trials",
    )
    alpha = _probability(shared_epoch_alpha, label="shared_epoch_alpha")
    ceiling = _probability(maximum_harm_rate, label="maximum_harm_rate")
    gain_bets = _finite_vector(gain_bet_fractions, label="gain_bet_fractions")
    if np.any(gain_bets <= 0.0) or np.any(gain_bets >= 1.0):
        raise ValueError("gain bet fractions must lie in (0, 1)")
    harm_fractions = _finite_vector(
        harm_alternative_fractions,
        label="harm_alternative_fractions",
    )
    if np.any(harm_fractions <= 0.0) or np.any(harm_fractions >= 1.0):
        raise ValueError("harm alternative fractions must lie in (0, 1)")
    harm_alternatives = ceiling * harm_fractions

    rng = np.random.default_rng(seed)
    gain_log_wealth = np.zeros((replications, len(gain_bets)), dtype=np.float64)
    harm_log_wealth = np.zeros(
        (replications, len(harm_alternatives)),
        dtype=np.float64,
    )
    shared_gain_crossed = np.zeros(replications, dtype=np.bool_)
    shared_harm_crossed = np.zeros(replications, dtype=np.bool_)
    split_gain_crossed = np.zeros(replications, dtype=np.bool_)
    split_harm_crossed = np.zeros(replications, dtype=np.bool_)
    shared_first = np.full(replications, -1, dtype=np.int64)
    split_first = np.full(replications, -1, dtype=np.int64)
    shared_log_threshold = -math.log(alpha)
    split_log_threshold = -math.log(alpha / 2.0)

    for observation in range(1, steps + 1):
        atom = rng.choice(len(probabilities), size=replications, p=probabilities)
        scores = gain_scores[atom]
        harms = harmful[atom]
        gain_log_wealth += np.log1p(scores[:, None] * gain_bets[None, :])
        harm_factors = np.where(
            harms[:, None],
            harm_alternatives[None, :] / ceiling,
            (1.0 - harm_alternatives[None, :]) / (1.0 - ceiling),
        )
        harm_log_wealth += np.log(harm_factors)
        gain_log_e = _log_mixture(gain_log_wealth)
        harm_log_e = _log_mixture(harm_log_wealth)

        shared_gain_crossed |= gain_log_e >= shared_log_threshold
        shared_harm_crossed |= harm_log_e >= shared_log_threshold
        split_gain_crossed |= gain_log_e >= split_log_threshold
        split_harm_crossed |= harm_log_e >= split_log_threshold
        if observation >= minimum:
            new_shared = (
                (shared_first < 0)
                & shared_gain_crossed
                & shared_harm_crossed
            )
            new_split = (
                (split_first < 0)
                & split_gain_crossed
                & split_harm_crossed
            )
            shared_first[new_shared] = observation
            split_first[new_split] = observation

    expected_gain = float(np.sum(probabilities * gain_scores))
    expected_harm_rate = float(np.sum(probabilities * harmful.astype(np.float64)))
    return {
        "expected_gain_score": expected_gain,
        "expected_harm_rate": expected_harm_rate,
        "shared_alpha_iut": {
            **_crossing_summary(shared_first),
            "component_alpha": alpha,
            "e_value_threshold": 1.0 / alpha,
        },
        "bonferroni_split": {
            **_crossing_summary(split_first),
            "component_alpha": alpha / 2.0,
            "e_value_threshold": 2.0 / alpha,
        },
    }


def run_joint_admission_study(protocol: dict[str, object]) -> dict[str, object]:
    """Execute the frozen controlled comparison specified by ``protocol``."""

    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported joint-admission study schema")
    if protocol.get("contract") != "anytime-joint-admission-power-v2":
        raise ValueError("unsupported joint-admission study contract")
    design = cast(dict[str, object], protocol["design"])
    total_alpha = _probability(design["total_alpha"], label="total_alpha")
    continuation = _probability(
        design["epoch_alpha_continuation"],
        label="epoch_alpha_continuation",
    )
    shared_epoch_alpha = total_alpha * (1.0 - continuation)
    replications = cast(dict[str, object], design["replication_count"])
    gain_bets = _finite_vector(
        design["gain_bet_fractions"],
        label="gain_bet_fractions",
    )
    harm_fractions = _finite_vector(
        design["harm_alternative_fractions"],
        label="harm_alternative_fractions",
    )
    scenarios = cast(dict[str, object], protocol["scenarios"])
    seed_base = int(cast(Any, design["seed_base"]))
    results: dict[str, object] = {}
    for index, (name, raw_scenario) in enumerate(scenarios.items()):
        scenario = cast(dict[str, object], raw_scenario)
        group = str(scenario["group"])
        result = simulate_discrete_admission_scenario(
            probabilities=np.asarray(scenario["probabilities"], dtype=np.float64),
            gain_scores=np.asarray(scenario["gain_scores"], dtype=np.float64),
            harmful=np.asarray(scenario["harmful"], dtype=np.bool_),
            replication_count=_positive_integer(
                replications[group],
                label=f"replication_count.{group}",
            ),
            horizon=_positive_integer(design["horizon"], label="horizon"),
            minimum_resolved_trials=_positive_integer(
                design["minimum_resolved_trials"],
                label="minimum_resolved_trials",
            ),
            shared_epoch_alpha=shared_epoch_alpha,
            gain_bet_fractions=gain_bets,
            maximum_harm_rate=_probability(
                design["maximum_harm_rate"],
                label="maximum_harm_rate",
            ),
            harm_alternative_fractions=harm_fractions,
            seed=seed_base + index,
        )
        result["group"] = group
        result["null_component"] = scenario.get("null_component")
        results[name] = result

    gates = cast(dict[str, object], protocol["mechanism_gate"])
    null_names = [
        name
        for name, value in results.items()
        if cast(dict[str, object], value)["group"] == "null"
    ]
    maximum_null_upper = max(
        float(
            cast(
                list[float],
                cast(dict[str, object], cast(dict[str, object], results[name])["shared_alpha_iut"])[
                    "wilson_95_interval"
                ],
            )[1]
        )
        for name in null_names
    )
    moderate = cast(dict[str, object], results["moderate_safe_benefit"])
    moderate_shared = cast(dict[str, object], moderate["shared_alpha_iut"])
    moderate_split = cast(dict[str, object], moderate["bonferroni_split"])
    strong = cast(dict[str, object], results["strong_safe_benefit"])
    strong_shared = cast(dict[str, object], strong["shared_alpha_iut"])
    power_gain = float(moderate_shared["crossing_probability"]) - float(
        moderate_split["crossing_probability"]
    )
    median_ratio = float(moderate_shared["median_first_crossing"]) / float(
        moderate_split["median_first_crossing"]
    )
    gate_results = {
        "null_wilson_upper": maximum_null_upper
        <= float(cast(Any, gates["maximum_null_wilson_upper"])),
        "moderate_power_gain": power_gain
        >= float(cast(Any, gates["minimum_moderate_power_gain"])),
        "moderate_median_crossing_ratio": median_ratio
        <= float(cast(Any, gates["maximum_moderate_median_crossing_ratio"])),
        "strong_power": float(strong_shared["crossing_probability"])
        >= float(cast(Any, gates["minimum_strong_power"])),
    }
    passed = all(gate_results.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": "anytime-joint-admission-power-result-v2",
        "decision": (
            "shared-alpha-iut-efficiency-supported"
            if passed
            else "shared-alpha-iut-efficiency-not-supported"
        ),
        "design": {
            **design,
            "shared_first_epoch_alpha": shared_epoch_alpha,
            "shared_component_threshold": 1.0 / shared_epoch_alpha,
            "split_component_threshold": 2.0 / shared_epoch_alpha,
        },
        "scenarios": results,
        "derived_comparison": {
            "maximum_null_wilson_upper": maximum_null_upper,
            "moderate_power_gain_shared_minus_split": power_gain,
            "moderate_median_crossing_ratio_shared_over_split": median_ratio,
        },
        "mechanism_gate": {
            "passed": passed,
            "requirements": gates,
            "results": gate_results,
        },
        "theorem_boundary": {
            "invalid_candidate_null": (
                "insufficient conditional mean bounded gain OR conditional harm "
                "rate at or above the registered ceiling"
            ),
            "shared_alpha_justification": (
                "joint admission implies rejection of whichever fixed component "
                "null is true; dependence between component e-processes is allowed"
            ),
            "latched_crossing": (
                "each component may cross at a different stopping time within the "
                "same epoch"
            ),
            "not_covered": (
                "epochs in which neither fixed component null holds throughout, "
                "unregistered score changes, physical safety, or real-data freshness"
            ),
        },
        "claim_boundary": (
            "Controlled Monte Carlo evidence for the shared-alpha intersection--union "
            "admission mechanism. It is not fresh real-world validation, a universal "
            "power guarantee, or a physical-safety certificate."
        ),
    }
