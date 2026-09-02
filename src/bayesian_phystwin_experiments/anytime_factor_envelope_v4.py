"""Controlled development study for factor-envelope admission version 4."""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin_experiments.anytime_joint_admission_v2 import wilson_interval

SCHEMA: Final = "bayesian-phystwin.anytime-factor-envelope-study-v4"
SCHEMA_VERSION: Final = 4


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
    if log_wealth.ndim < 2 or any(size == 0 for size in log_wealth.shape[1:]):
        raise ValueError("log wealth must contain replications and components")
    flattened = log_wealth.reshape(log_wealth.shape[0], -1)
    terms = flattened - math.log(flattened.shape[1])
    maximum = np.max(terms, axis=1)
    return maximum + np.log(np.sum(np.exp(terms - maximum[:, None]), axis=1))


def _crossing_summary(first_crossing: np.ndarray) -> dict[str, object]:
    crossed = first_crossing >= 0
    count = int(np.sum(crossed))
    total = len(first_crossing)
    values = first_crossing[crossed]
    return {
        "replication_count": total,
        "crossing_count": count,
        "crossing_probability": count / total,
        "wilson_95_interval": wilson_interval(count, total),
        "median_first_crossing": (None if count == 0 else float(np.median(values))),
        "first_crossing_quantiles_10_90": (
            None
            if count == 0
            else [float(value) for value in np.quantile(values, (0.10, 0.90))]
        ),
    }


def _phase_arrays(
    phase: dict[str, object],
    *,
    maximum_harm_rate: float,
    gain_bets: np.ndarray,
    harm_fractions: np.ndarray,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    duration = _positive_integer(phase["duration"], label="phase duration")
    probabilities = _finite_vector(phase["probabilities"], label="probabilities")
    gain_scores = _finite_vector(phase["gain_scores"], label="gain_scores")
    harmful = np.asarray(phase["harmful"], dtype=np.bool_)
    if harmful.ndim != 1 or not (
        len(probabilities) == len(gain_scores) == len(harmful)
    ):
        raise ValueError("phase atoms must have equal nonzero length")
    if np.any(probabilities <= 0.0) or not math.isclose(
        float(np.sum(probabilities)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("phase probabilities must be positive and sum to one")
    if np.any(gain_scores < -1.0) or np.any(gain_scores > 1.0):
        raise ValueError("gain scores must lie in [-1, 1]")

    scale = max(maximum_harm_rate, 1.0 - maximum_harm_rate)
    harm_scores = (maximum_harm_rate - harmful.astype(np.float64)) / scale
    robust_scores = np.minimum(gain_scores, harm_scores)

    alternatives = maximum_harm_rate * harm_fractions
    gain_factors = 1.0 + gain_scores[:, None] * gain_bets[None, :]
    harm_factors = np.where(
        harmful[:, None],
        alternatives[None, :] / maximum_harm_rate,
        (1.0 - alternatives[None, :]) / (1.0 - maximum_harm_rate),
    )
    envelope = np.minimum(
        gain_factors[:, :, None],
        harm_factors[:, None, :],
    )
    expected_envelope = np.sum(
        probabilities[:, None, None] * envelope,
        axis=0,
    )
    diagnostics = {
        "expected_gain_score": float(np.sum(probabilities * gain_scores)),
        "expected_harm_rate": float(np.sum(probabilities * harmful.astype(np.float64))),
        "expected_min_score": float(np.sum(probabilities * robust_scores)),
        "maximum_expected_envelope_factor": float(np.max(expected_envelope)),
        "minimum_expected_envelope_factor": float(np.min(expected_envelope)),
    }
    return (
        duration,
        probabilities,
        gain_scores,
        harmful,
        robust_scores,
        diagnostics,
    )


def simulate_factor_envelope_scenario(
    *,
    phases: list[dict[str, object]],
    replication_count: int,
    minimum_resolved_trials: int,
    shared_epoch_alpha: float,
    gain_bet_fractions: np.ndarray,
    maximum_harm_rate: float,
    harm_alternative_fractions: np.ndarray,
    robust_bet_fractions: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Compare version-3 scalarization with version-4 factor envelopes."""

    if not phases:
        raise ValueError("phases must not be empty")
    replications = _positive_integer(replication_count, label="replication_count")
    minimum = _positive_integer(
        minimum_resolved_trials,
        label="minimum_resolved_trials",
    )
    alpha = _probability(shared_epoch_alpha, label="shared_epoch_alpha")
    ceiling = _probability(maximum_harm_rate, label="maximum_harm_rate")
    gain_bets = _finite_vector(
        gain_bet_fractions,
        label="gain_bet_fractions",
    )
    harm_fractions = _finite_vector(
        harm_alternative_fractions,
        label="harm_alternative_fractions",
    )
    robust_bets = _finite_vector(
        robust_bet_fractions,
        label="robust_bet_fractions",
    )
    for label, values in (
        ("gain_bet_fractions", gain_bets),
        ("harm_alternative_fractions", harm_fractions),
        ("robust_bet_fractions", robust_bets),
    ):
        if np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError(f"{label} must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    robust_log_wealth = np.zeros(
        (replications, len(robust_bets)),
        dtype=np.float64,
    )
    envelope_log_wealth = np.zeros(
        (replications, len(gain_bets), len(harm_fractions)),
        dtype=np.float64,
    )
    robust_first = np.full(replications, -1, dtype=np.int64)
    envelope_first = np.full(replications, -1, dtype=np.int64)
    threshold = -math.log(alpha)
    observation = 0
    phase_summaries: list[dict[str, object]] = []
    alternatives = ceiling * harm_fractions

    for index, phase in enumerate(phases):
        (
            duration,
            probabilities,
            gain_scores,
            harmful,
            robust_scores,
            diagnostics,
        ) = _phase_arrays(
            phase,
            maximum_harm_rate=ceiling,
            gain_bets=gain_bets,
            harm_fractions=harm_fractions,
        )
        phase_summaries.append(
            {
                "phase_index": index,
                "name": str(phase.get("name", f"phase-{index}")),
                "duration": duration,
                "active_null_component": phase.get("active_null_component"),
                **diagnostics,
            }
        )
        for _ in range(duration):
            observation += 1
            atoms = rng.choice(
                len(probabilities),
                size=replications,
                p=probabilities,
            )
            gains = gain_scores[atoms]
            harms = harmful[atoms]
            robust = robust_scores[atoms]

            robust_log_wealth += np.log1p(robust[:, None] * robust_bets[None, :])
            gain_factors = 1.0 + gains[:, None] * gain_bets[None, :]
            harm_factors = np.where(
                harms[:, None],
                alternatives[None, :] / ceiling,
                (1.0 - alternatives[None, :]) / (1.0 - ceiling),
            )
            envelope_factors = np.minimum(
                gain_factors[:, :, None],
                harm_factors[:, None, :],
            )
            envelope_log_wealth += np.log(envelope_factors)

            robust_log_e = _log_mixture(robust_log_wealth)
            envelope_log_e = _log_mixture(envelope_log_wealth)
            if observation >= minimum:
                new_robust = (robust_first < 0) & (robust_log_e >= threshold)
                new_envelope = (envelope_first < 0) & (envelope_log_e >= threshold)
                robust_first[new_robust] = observation
                envelope_first[new_envelope] = observation

    return {
        "phase_count": len(phases),
        "horizon": observation,
        "phase_expectations": phase_summaries,
        "switching_union_min_score_v3": {
            **_crossing_summary(robust_first),
            "component_alpha": alpha,
            "e_value_threshold": 1.0 / alpha,
            "factor_family": ("min(1 + lambda G, 1 + lambda S) with one shared lambda"),
        },
        "switching_union_factor_envelope_v4": {
            **_crossing_summary(envelope_first),
            "component_alpha": alpha,
            "e_value_threshold": 1.0 / alpha,
            "factor_family": (
                "min(1 + lambda_g G, BernoulliLR(q_h; rho)) mixed over "
                "the Cartesian product of independently fixed parameters"
            ),
            "component_count": int(len(gain_bets) * len(harm_fractions)),
        },
    }


def run_factor_envelope_study(
    protocol: dict[str, object],
) -> dict[str, object]:
    """Execute the frozen version-4 controlled development protocol."""

    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported factor-envelope study schema")
    if protocol.get("contract") != "anytime-factor-envelope-development-v4":
        raise ValueError("unsupported factor-envelope study contract")
    design = cast(dict[str, object], protocol["design"])
    total_alpha = _probability(design["total_alpha"], label="total_alpha")
    continuation = _probability(
        design["epoch_alpha_continuation"],
        label="epoch_alpha_continuation",
    )
    epoch_alpha = total_alpha * (1.0 - continuation)
    replication_counts = cast(dict[str, object], design["replication_count"])
    scenarios = cast(dict[str, object], protocol["scenarios"])
    gain_bets = np.asarray(design["gain_bet_fractions"], dtype=np.float64)
    harm_fractions = np.asarray(
        design["harm_alternative_fractions"],
        dtype=np.float64,
    )
    robust_bets = np.asarray(
        design["robust_bet_fractions"],
        dtype=np.float64,
    )
    ceiling = _probability(
        design["maximum_harm_rate"],
        label="maximum_harm_rate",
    )
    seed_base = int(cast(Any, design["seed_base"]))
    results: dict[str, object] = {}
    for index, (name, raw_scenario) in enumerate(scenarios.items()):
        scenario = cast(dict[str, object], raw_scenario)
        group = str(scenario["group"])
        result = simulate_factor_envelope_scenario(
            phases=cast(list[dict[str, object]], scenario["phases"]),
            replication_count=_positive_integer(
                replication_counts[group],
                label=f"replication_count.{group}",
            ),
            minimum_resolved_trials=_positive_integer(
                design["minimum_resolved_trials"],
                label="minimum_resolved_trials",
            ),
            shared_epoch_alpha=epoch_alpha,
            gain_bet_fractions=gain_bets,
            maximum_harm_rate=ceiling,
            harm_alternative_fractions=harm_fractions,
            robust_bet_fractions=robust_bets,
            seed=seed_base + index,
        )
        result["group"] = group
        result["registered_null"] = scenario.get("registered_null")
        results[name] = result

    null_names = [
        name
        for name, raw in results.items()
        if cast(dict[str, object], raw)["group"] == "null"
    ]

    def method(name: str, method_name: str) -> dict[str, object]:
        scenario = cast(dict[str, object], results[name])
        return cast(dict[str, object], scenario[method_name])

    envelope_name = "switching_union_factor_envelope_v4"
    min_score_name = "switching_union_min_score_v3"
    maximum_null_upper = max(
        float(
            cast(
                list[float],
                method(name, envelope_name)["wilson_95_interval"],
            )[1]
        )
        for name in null_names
    )
    switching_envelope = method("switching_invalidity", envelope_name)
    moderate_envelope = method("moderate_safe_benefit", envelope_name)
    moderate_min_score = method("moderate_safe_benefit", min_score_name)
    strong_envelope = method("strong_safe_benefit", envelope_name)

    moderate_power_gain = float(
        cast(Any, moderate_envelope["crossing_probability"])
    ) - float(cast(Any, moderate_min_score["crossing_probability"]))
    envelope_median = float(cast(Any, moderate_envelope["median_first_crossing"]))
    min_score_median = float(cast(Any, moderate_min_score["median_first_crossing"]))
    moderate_median_ratio = envelope_median / min_score_median

    requirements = cast(dict[str, object], protocol["mechanism_gate"])
    gate_results = {
        "factor_envelope_null_control": maximum_null_upper
        <= float(cast(Any, requirements["maximum_envelope_null_wilson_upper"])),
        "switching_null_control": float(
            cast(Any, switching_envelope["crossing_probability"])
        )
        <= float(
            cast(
                Any,
                requirements["maximum_switching_null_envelope_crossing"],
            )
        ),
        "moderate_power": float(cast(Any, moderate_envelope["crossing_probability"]))
        >= float(cast(Any, requirements["minimum_moderate_envelope_power"])),
        "moderate_power_gain": moderate_power_gain
        >= float(cast(Any, requirements["minimum_moderate_power_gain"])),
        "moderate_crossing_speed": moderate_median_ratio
        <= float(
            cast(
                Any,
                requirements["maximum_moderate_median_crossing_ratio"],
            )
        ),
        "strong_power": float(cast(Any, strong_envelope["crossing_probability"]))
        >= float(cast(Any, requirements["minimum_strong_envelope_power"])),
    }
    passed = all(gate_results.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": "anytime-factor-envelope-development-result-v4",
        "decision": (
            "factor-envelope-efficiency-supported"
            if passed
            else "factor-envelope-efficiency-not-supported"
        ),
        "design": {
            **design,
            "first_epoch_alpha": epoch_alpha,
            "e_value_threshold": 1.0 / epoch_alpha,
            "factor_envelope_component_count": int(
                len(gain_bets) * len(harm_fractions)
            ),
        },
        "scenarios": results,
        "derived_comparison": {
            "maximum_envelope_null_wilson_upper": maximum_null_upper,
            "switching_null_envelope_crossing_probability": switching_envelope[
                "crossing_probability"
            ],
            "moderate_envelope_power": moderate_envelope["crossing_probability"],
            "moderate_min_score_power": moderate_min_score["crossing_probability"],
            "moderate_power_gain_envelope_minus_min_score": moderate_power_gain,
            "moderate_median_crossing_ratio_envelope_over_min_score": (
                moderate_median_ratio
            ),
            "strong_envelope_power": strong_envelope["crossing_probability"],
        },
        "mechanism_gate": {
            "passed": passed,
            "requirements": requirements,
            "results": gate_results,
        },
        "theorem_boundary": {
            "general_composition": (
                "under a pointwise union of component nulls, the minimum of "
                "independently fixed component e-factors is dominated by the "
                "factor associated with whichever null is active"
            ),
            "time_product": (
                "products remain test supermartingales even when the active "
                "component changes arbitrarily with the past"
            ),
            "parameter_mixture": (
                "an outcome-independent mixture over fixed component-parameter "
                "tuples preserves the e-process property"
            ),
            "relation_to_v3": (
                "version 3 is the shared-fraction diagonal restriction after "
                "expressing the Bernoulli factor as a linear harm-score factor"
            ),
            "not_covered": (
                "outcome-dependent factor-grid redesign, unregistered candidate "
                "or fallback changes, physical safety, or real-world validity"
            ),
        },
        "claim_boundary": (
            "Controlled confirmation evidence for the independently tuned "
            "factor-envelope construction. Scenario families were inherited "
            "from the already observed version-3 controlled study. A separate "
            "version-4 pilot roster was used only to choose the frozen gates; "
            "the retained confirmation seed roster was not opened before this "
            "protocol was committed. No real outcomes are used."
        ),
    }
