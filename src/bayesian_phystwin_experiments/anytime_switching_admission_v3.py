"""Controlled stress study for switching-null anytime admission version 3."""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin_experiments.anytime_joint_admission_v2 import wilson_interval

SCHEMA: Final = "bayesian-phystwin.anytime-switching-admission-study-v3"
SCHEMA_VERSION: Final = 3


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
        "median_first_crossing": (
            None if count == 0 else float(np.median(values))
        ),
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
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    harm_scores = (
        maximum_harm_rate - harmful.astype(np.float64)
    ) / scale
    robust_scores = np.minimum(gain_scores, harm_scores)
    return duration, probabilities, gain_scores, harmful, robust_scores


def simulate_switching_admission_scenario(
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
    """Compare stable-null IUT and switching-null robust admission."""

    if not phases:
        raise ValueError("phases must not be empty")
    replications = _positive_integer(replication_count, label="replication_count")
    minimum = _positive_integer(
        minimum_resolved_trials,
        label="minimum_resolved_trials",
    )
    alpha = _probability(shared_epoch_alpha, label="shared_epoch_alpha")
    ceiling = _probability(maximum_harm_rate, label="maximum_harm_rate")
    gain_bets = _finite_vector(gain_bet_fractions, label="gain_bet_fractions")
    robust_bets = _finite_vector(
        robust_bet_fractions,
        label="robust_bet_fractions",
    )
    if (
        np.any(gain_bets <= 0.0)
        or np.any(gain_bets >= 1.0)
        or np.any(robust_bets <= 0.0)
        or np.any(robust_bets >= 1.0)
    ):
        raise ValueError("bet fractions must lie in (0, 1)")
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
    robust_log_wealth = np.zeros(
        (replications, len(robust_bets)),
        dtype=np.float64,
    )
    gain_crossed = np.zeros(replications, dtype=np.bool_)
    harm_crossed = np.zeros(replications, dtype=np.bool_)
    iut_first = np.full(replications, -1, dtype=np.int64)
    robust_first = np.full(replications, -1, dtype=np.int64)
    threshold = -math.log(alpha)
    observation = 0
    phase_summaries: list[dict[str, object]] = []

    for index, phase in enumerate(phases):
        duration, probabilities, gain_scores, harmful, robust_scores = _phase_arrays(
            phase,
            maximum_harm_rate=ceiling,
        )
        phase_summaries.append(
            {
                "phase_index": index,
                "name": str(phase.get("name", f"phase-{index}")),
                "duration": duration,
                "active_null_component": phase.get("active_null_component"),
                "expected_gain_score": float(
                    np.sum(probabilities * gain_scores)
                ),
                "expected_harm_rate": float(
                    np.sum(probabilities * harmful.astype(np.float64))
                ),
                "expected_robust_score": float(
                    np.sum(probabilities * robust_scores)
                ),
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
            gain_log_wealth += np.log1p(
                gains[:, None] * gain_bets[None, :]
            )
            harm_factors = np.where(
                harms[:, None],
                harm_alternatives[None, :] / ceiling,
                (1.0 - harm_alternatives[None, :]) / (1.0 - ceiling),
            )
            harm_log_wealth += np.log(harm_factors)
            robust_log_wealth += np.log1p(
                robust[:, None] * robust_bets[None, :]
            )
            gain_log_e = _log_mixture(gain_log_wealth)
            harm_log_e = _log_mixture(harm_log_wealth)
            robust_log_e = _log_mixture(robust_log_wealth)
            gain_crossed |= gain_log_e >= threshold
            harm_crossed |= harm_log_e >= threshold
            if observation >= minimum:
                new_iut = (
                    (iut_first < 0)
                    & gain_crossed
                    & harm_crossed
                )
                new_robust = (
                    (robust_first < 0)
                    & (robust_log_e >= threshold)
                )
                iut_first[new_iut] = observation
                robust_first[new_robust] = observation

    return {
        "phase_count": len(phases),
        "horizon": observation,
        "phase_expectations": phase_summaries,
        "latched_shared_alpha_iut": {
            **_crossing_summary(iut_first),
            "component_alpha": alpha,
            "e_value_threshold": 1.0 / alpha,
            "assumption": (
                "one fixed component null must hold throughout the epoch"
            ),
        },
        "switching_union_min_score": {
            **_crossing_summary(robust_first),
            "component_alpha": alpha,
            "e_value_threshold": 1.0 / alpha,
            "assumption": (
                "at each reveal at least one component null holds; the active "
                "component may switch predictably or adversarially"
            ),
        },
    }


def run_switching_admission_study(
    protocol: dict[str, object],
) -> dict[str, object]:
    """Execute the frozen switching-null stress protocol."""

    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported switching-admission study schema")
    if protocol.get("contract") != "anytime-switching-admission-stress-v3":
        raise ValueError("unsupported switching-admission study contract")
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
    seed_base = int(cast(Any, design["seed_base"]))
    results: dict[str, object] = {}
    for index, (name, raw_scenario) in enumerate(scenarios.items()):
        scenario = cast(dict[str, object], raw_scenario)
        group = str(scenario["group"])
        phases = cast(list[dict[str, object]], scenario["phases"])
        result = simulate_switching_admission_scenario(
            phases=phases,
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
            maximum_harm_rate=_probability(
                design["maximum_harm_rate"],
                label="maximum_harm_rate",
            ),
            harm_alternative_fractions=harm_fractions,
            robust_bet_fractions=robust_bets,
            seed=seed_base + index,
        )
        result["group"] = group
        result["registered_null"] = scenario.get("registered_null")
        results[name] = result

    switching = cast(dict[str, object], results["switching_invalidity"])
    switching_iut = cast(
        dict[str, object],
        switching["latched_shared_alpha_iut"],
    )
    switching_robust = cast(
        dict[str, object],
        switching["switching_union_min_score"],
    )
    null_names = [
        name
        for name, raw in results.items()
        if cast(dict[str, object], raw)["group"] == "null"
    ]
    maximum_robust_null_upper = max(
        float(
            cast(
                list[float],
                cast(
                    dict[str, object],
                    cast(dict[str, object], results[name])[
                        "switching_union_min_score"
                    ],
                )["wilson_95_interval"],
            )[1]
        )
        for name in null_names
    )
    moderate = cast(dict[str, object], results["moderate_safe_benefit"])
    strong = cast(dict[str, object], results["strong_safe_benefit"])
    moderate_robust = cast(
        dict[str, object],
        moderate["switching_union_min_score"],
    )
    strong_robust = cast(
        dict[str, object],
        strong["switching_union_min_score"],
    )
    requirements = cast(dict[str, object], protocol["mechanism_gate"])
    gate_results = {
        "robust_null_control": maximum_robust_null_upper
        <= float(cast(Any, requirements["maximum_robust_null_wilson_upper"])),
        "latched_iut_failure_outside_assumptions": float(
            cast(Any, switching_iut["crossing_probability"])
        )
        >= float(
            cast(Any, requirements["minimum_switching_null_iut_crossing"])
        ),
        "switching_robust_control": float(
            cast(Any, switching_robust["crossing_probability"])
        )
        <= float(
            cast(Any, requirements["maximum_switching_null_robust_crossing"])
        ),
        "moderate_robust_power": float(
            cast(Any, moderate_robust["crossing_probability"])
        )
        >= float(cast(Any, requirements["minimum_moderate_robust_power"])),
        "strong_robust_power": float(
            cast(Any, strong_robust["crossing_probability"])
        )
        >= float(cast(Any, requirements["minimum_strong_robust_power"])),
    }
    passed = all(gate_results.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": "anytime-switching-admission-stress-result-v3",
        "decision": (
            "switching-null-robustness-supported"
            if passed
            else "switching-null-robustness-not-supported"
        ),
        "design": {
            **design,
            "first_epoch_alpha": epoch_alpha,
            "e_value_threshold": 1.0 / epoch_alpha,
        },
        "scenarios": results,
        "derived_comparison": {
            "maximum_robust_null_wilson_upper": maximum_robust_null_upper,
            "switching_null_latched_iut_crossing_probability": switching_iut[
                "crossing_probability"
            ],
            "switching_null_robust_crossing_probability": switching_robust[
                "crossing_probability"
            ],
            "moderate_robust_power": moderate_robust["crossing_probability"],
            "strong_robust_power": strong_robust["crossing_probability"],
        },
        "mechanism_gate": {
            "passed": passed,
            "requirements": requirements,
            "results": gate_results,
        },
        "theorem_boundary": {
            "stable_component_iut": (
                "efficient but requires one component null to hold throughout "
                "the admission epoch"
            ),
            "switching_union_certificate": (
                "valid when at every reveal at least one component null holds, "
                "even if the active component changes over time"
            ),
            "key_inequality": (
                "min(gain_score, harm_score) is no larger than whichever "
                "component score has nonpositive conditional expectation"
            ),
            "not_covered": (
                "unregistered adaptation of candidate, fallback, score, harm "
                "definition, information set, or betting fractions"
            ),
        },
        "claim_boundary": (
            "Controlled stress evidence for the switching-null certificate and "
            "for the necessity of the stable-null assumption behind the more "
            "efficient latched intersection--union rule. It is not fresh real-world "
            "validation or a physical-safety guarantee."
        ),
    }
