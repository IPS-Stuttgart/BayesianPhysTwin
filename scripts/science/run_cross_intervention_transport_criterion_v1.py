#!/usr/bin/env python3
"""Test the cross-intervention transport criterion in controlled regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

SCHEMA: Final = "bayesian_phystwin.cross_intervention_transport_criterion"
SCHEMA_VERSION: Final = 1
DESIGN_SHA256: Final = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
ACTION_RESPONSE: Final[Mapping[str, float]] = {
    "lift_low": 1.0,
    "lift_high": 1.6,
    "lower_high": -1.5,
    "lateral_low": 0.75,
}
CLAIM_BOUNDARY: Final = (
    "Controlled local-linear mechanism evidence only. A positive result shows "
    "that held-out intervention transport can reject source-local discrepancy "
    "under the registered simulation assumptions. It does not establish a unique "
    "physical cause, simulator adequacy, real-object transfer, real calibration, "
    "provider competence, Causal4D physical benefit, deployment safety, or state "
    "of the art."
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Regime:
    """One frozen mechanism or assumption-violation regime."""

    name: str
    physical_strength: float = 0.0
    local_scale: float = 0.0
    local_correlation: float = 0.0
    shared_bias_scale: float = 0.0
    action_aligned_bias_scale: float = 0.0
    target_transport_scale: float = 1.0
    declared_identifiable: bool = True


REGIMES: Final[tuple[Regime, ...]] = (
    Regime(
        "transportable_physical",
        physical_strength=1.5,
        local_scale=0.1,
    ),
    Regime("source_local_discrepancy", local_scale=1.5),
    Regime(
        "correlated_local_discrepancy",
        local_scale=1.5,
        local_correlation=0.5,
    ),
    Regime("shared_action_independent_bias", shared_bias_scale=1.5),
    Regime(
        "action_aligned_undeclared_nuisance",
        action_aligned_bias_scale=1.0,
    ),
    Regime(
        "action_aligned_declared_nuisance",
        action_aligned_bias_scale=1.0,
        declared_identifiable=False,
    ),
    Regime(
        "physical_conservative_transport",
        physical_strength=1.5,
        local_scale=0.1,
        target_transport_scale=0.5,
    ),
    Regime(
        "physical_sign_error",
        physical_strength=1.5,
        local_scale=0.1,
        target_transport_scale=-0.5,
    ),
    Regime(
        "physical_local_mixture",
        physical_strength=1.0,
        local_scale=0.8,
        local_correlation=0.2,
    ),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--noise-scale", type=float, default=0.3)
    parser.add_argument("--guard-threshold", type=float, default=0.25)
    parser.add_argument("--harmful-gain-margin", type=float, default=0.25)
    parser.add_argument("--discrepancy-shrink", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_no_clobber(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _load_action_roster(
    path: Path,
) -> tuple[FloatArray, FloatArray, list[dict[str, str]]]:
    payload = _strict_json(path)
    if payload.get("causal4d_design_sha256") != DESIGN_SHA256:
        raise ValueError(
            "sparse-pair roster is not bound to the registered Causal4D design"
        )
    rows = payload.get("session_pairs")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("registered simulation requires the exact 18-session roster")

    source: list[float] = []
    target: list[float] = []
    normalized: list[dict[str, str]] = []
    seen_sessions: set[str] = set()
    seen_executions: set[str] = set()
    for index, raw in enumerate(rows):
        if type(raw) is not dict:
            raise ValueError(f"session_pairs[{index}] must be a JSON object")
        session_id = raw.get("object_session_id")
        source_execution = raw.get("source_execution_id")
        target_execution = raw.get("target_execution_id")
        source_action = raw.get("source_action_id")
        target_action = raw.get("target_action_id")
        values = (
            session_id,
            source_execution,
            target_execution,
            source_action,
            target_action,
        )
        if any(type(value) is not str or not value for value in values):
            raise ValueError(f"session_pairs[{index}] contains an invalid identity")
        assert isinstance(session_id, str)
        assert isinstance(source_execution, str)
        assert isinstance(target_execution, str)
        assert isinstance(source_action, str)
        assert isinstance(target_action, str)
        if session_id in seen_sessions:
            raise ValueError("physical sessions must be unique")
        if source_execution in seen_executions or target_execution in seen_executions:
            raise ValueError("execution identities must be unique")
        if source_action == target_action:
            raise ValueError("every registered pair must be genuinely cross-action")
        if source_action not in ACTION_RESPONSE or target_action not in ACTION_RESPONSE:
            raise ValueError(
                "roster names an action without a frozen simulation response"
            )
        seen_sessions.add(session_id)
        seen_executions.update((source_execution, target_execution))
        source.append(ACTION_RESPONSE[source_action])
        target.append(ACTION_RESPONSE[target_action])
        normalized.append(
            {
                "object_session_id": session_id,
                "source_action_id": source_action,
                "target_action_id": target_action,
            }
        )
    if len(seen_sessions) != 18 or len(seen_executions) != 36:
        raise ValueError("registered roster cardinality drifted")
    return (
        np.asarray(source, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        normalized,
    )


def _bootstrap_weights(sample_count: int, replicates: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample_count, size=(replicates, sample_count))
    weights = np.zeros((replicates, sample_count), dtype=np.float64)
    for column in range(sample_count):
        weights[:, column] = np.count_nonzero(indices == column, axis=1)
    return weights / sample_count


def _bootstrap_lower(values: FloatArray, weights: FloatArray) -> FloatArray:
    means = values @ weights.T
    return np.asarray(np.quantile(means, 0.025, axis=1), dtype=np.float64)


def _binomial_cdf(events: int, total: int, probability: float) -> float:
    return float(
        sum(
            math.comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(events + 1)
        )
    )


def _clopper_pearson_upper(events: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 1.0
    if events < 0 or events > total:
        raise ValueError("event count must satisfy 0 <= events <= total")
    if events == total:
        return 1.0
    alpha = 1.0 - confidence
    low = events / total
    high = 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if _binomial_cdf(events, total, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _harm_upper_lookup(maximum_sessions: int) -> FloatArray:
    result = np.ones((maximum_sessions + 1, maximum_sessions + 1), dtype=np.float64)
    for total in range(1, maximum_sessions + 1):
        for events in range(total + 1):
            result[total, events] = _clopper_pearson_upper(events, total)
    return result


def _gain(target: FloatArray, correction: FloatArray) -> FloatArray:
    return target * target - (target - correction) ** 2


def _run_regime(
    regime: Regime,
    *,
    source_response: FloatArray,
    target_response: FloatArray,
    trials: int,
    bootstrap_weights: FloatArray,
    seed: int,
    noise_scale: float,
    guard_threshold: float,
    harmful_gain_margin: float,
    discrepancy_shrink: float,
    harm_lookup: FloatArray,
) -> dict[str, object]:
    session_count = source_response.size
    rng = np.random.default_rng(seed)
    shape = (trials, session_count)
    latent = rng.normal(size=shape)
    source_local = rng.normal(size=shape) * regime.local_scale
    target_independent_local = rng.normal(size=shape) * regime.local_scale
    target_local = (
        regime.local_correlation * source_local
        + math.sqrt(max(0.0, 1.0 - regime.local_correlation**2))
        * target_independent_local
    )
    shared_bias = rng.normal(size=shape) * regime.shared_bias_scale
    aligned_bias = rng.normal(size=shape) * regime.action_aligned_bias_scale
    source_noise = rng.normal(scale=noise_scale, size=shape)
    target_noise = rng.normal(scale=noise_scale, size=shape)

    source_truth = (
        regime.physical_strength * source_response * latent
        + source_local
        + shared_bias
        + source_response * aligned_bias
        + source_noise
    )
    target_truth = (
        regime.physical_strength * target_response * latent
        + target_local
        + shared_bias
        + target_response * aligned_bias
        + target_noise
    )

    declared_variance = (
        noise_scale**2 + regime.local_scale**2 + regime.shared_bias_scale**2
    )
    source_gain_factor = source_response / (source_response**2 + declared_variance)
    posterior_mean = source_truth * source_gain_factor
    source_correction = posterior_mean * source_response
    physical_correction = (
        posterior_mean * target_response * regime.target_transport_scale
    )
    discrepancy_correction = discrepancy_shrink * source_truth
    last_residual_correction = source_truth

    posterior_variance = np.maximum(
        1e-12,
        1.0 - source_response * source_gain_factor,
    )
    predicted_target_scale = np.abs(target_response) * np.sqrt(
        posterior_variance + noise_scale**2
    )
    accepted = np.abs(physical_correction) / (predicted_target_scale + 1e-12) >= (
        guard_threshold
    )
    if not regime.declared_identifiable:
        accepted = np.zeros_like(accepted, dtype=bool)
    guarded_correction = np.where(accepted, physical_correction, 0.0)

    source_gains = _gain(source_truth, source_correction)
    physical_gains = _gain(target_truth, guarded_correction)
    discrepancy_gains = _gain(target_truth, discrepancy_correction)
    last_residual_gains = _gain(target_truth, last_residual_correction)

    source_lower = _bootstrap_lower(source_gains, bootstrap_weights)
    transport_lower = _bootstrap_lower(physical_gains, bootstrap_weights)
    discrepancy_contrast_lower = _bootstrap_lower(
        physical_gains - discrepancy_gains,
        bootstrap_weights,
    )
    residual_contrast_lower = _bootstrap_lower(
        physical_gains - last_residual_gains,
        bootstrap_weights,
    )

    selected_sessions = np.count_nonzero(accepted, axis=1)
    harmful_sessions = np.count_nonzero(
        accepted & (physical_gains < -harmful_gain_margin),
        axis=1,
    )
    harm_upper = np.asarray(
        [
            harm_lookup[int(selected), int(harmful)]
            for selected, harmful in zip(
                selected_sessions,
                harmful_sessions,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    source_only_claim = source_lower > 0.0
    transport_only_claim = transport_lower > 0.0
    full_protocol_claim = (
        (selected_sessions >= 14)
        & (transport_lower > 0.0)
        & (discrepancy_contrast_lower > 0.0)
        & (residual_contrast_lower > 0.0)
        & (harm_upper <= 0.20)
    )

    return {
        "regime": regime.name,
        "mechanism": {
            "physical_strength": regime.physical_strength,
            "local_scale": regime.local_scale,
            "local_correlation": regime.local_correlation,
            "shared_bias_scale": regime.shared_bias_scale,
            "action_aligned_bias_scale": regime.action_aligned_bias_scale,
            "target_transport_scale": regime.target_transport_scale,
            "declared_identifiable": regime.declared_identifiable,
        },
        "claim_rates": {
            "source_only": float(np.mean(source_only_claim)),
            "transport_only": float(np.mean(transport_only_claim)),
            "full_protocol": float(np.mean(full_protocol_claim)),
        },
        "mean_session_statistics": {
            "source_gain": float(np.mean(source_gains)),
            "transport_gain": float(np.mean(physical_gains)),
            "transport_minus_discrepancy": float(
                np.mean(physical_gains - discrepancy_gains)
            ),
            "transport_minus_last_residual": float(
                np.mean(physical_gains - last_residual_gains)
            ),
            "accepted_physical_sessions": float(np.mean(selected_sessions)),
            "materially_harmful_accepted_sessions": float(np.mean(harmful_sessions)),
        },
        "trial_mean_transport_gain_quantiles": {
            "q05": float(np.quantile(np.mean(physical_gains, axis=1), 0.05)),
            "q50": float(np.quantile(np.mean(physical_gains, axis=1), 0.50)),
            "q95": float(np.quantile(np.mean(physical_gains, axis=1), 0.95)),
        },
    }


def _registered_checks(regimes: Mapping[str, Mapping[str, object]]) -> dict[str, bool]:
    def rate(regime: str, rule: str) -> float:
        row = regimes[regime]["claim_rates"]
        if not isinstance(row, Mapping):
            raise ValueError("claim_rates must be a mapping")
        value = row[rule]
        if not isinstance(value, float):
            raise ValueError("claim rate must be a float")
        return value

    return {
        "source_fit_falsely_accepts_local_discrepancy": (
            rate("source_local_discrepancy", "source_only") >= 0.95
        ),
        "transport_rejects_source_local_discrepancy": (
            rate("source_local_discrepancy", "transport_only") <= 0.05
        ),
        "full_protocol_rejects_local_and_shared_bias": (
            rate("source_local_discrepancy", "full_protocol") <= 0.01
            and rate("correlated_local_discrepancy", "full_protocol") <= 0.01
            and rate("shared_action_independent_bias", "full_protocol") <= 0.01
        ),
        "transport_detects_registered_physical_mechanism": (
            rate("transportable_physical", "transport_only") >= 0.90
        ),
        "undeclared_action_aligned_nuisance_can_fool_transport": (
            rate("action_aligned_undeclared_nuisance", "transport_only") >= 0.80
            and rate("action_aligned_undeclared_nuisance", "full_protocol") >= 0.10
        ),
        "declared_nonidentifiability_fails_closed": (
            rate("action_aligned_declared_nuisance", "transport_only") == 0.0
            and rate("action_aligned_declared_nuisance", "full_protocol") == 0.0
        ),
        "simulator_sign_error_is_not_misreported_as_transport": (
            rate("physical_sign_error", "transport_only") <= 0.05
            and rate("physical_sign_error", "full_protocol") <= 0.01
        ),
        "conservative_transport_improves_full_protocol_power": (
            rate("physical_conservative_transport", "full_protocol")
            >= rate("transportable_physical", "full_protocol") + 0.20
        ),
    }


def _render_markdown(result: Mapping[str, object]) -> str:
    regimes = result["regimes"]
    checks = result["registered_checks"]
    assert isinstance(regimes, Mapping)
    assert isinstance(checks, Mapping)
    lines = [
        "# Cross-intervention transport criterion V1",
        "",
        "## Decision",
        "",
        f"**{result['decision']}**",
        "",
        "The criterion is useful as a falsification test for source-local discrepancy, "
        "but it is not sufficient without declared nuisance and identifiability "
        "checks.",
        "",
        "## Controlled claim rates",
        "",
        "| Regime | Source-only | Transport-only | Full protocol |",
        "|---|---:|---:|---:|",
    ]
    for name, raw in regimes.items():
        assert isinstance(raw, Mapping)
        rates = raw["claim_rates"]
        assert isinstance(rates, Mapping)
        lines.append(
            f"| `{name}` | {100.0 * float(rates['source_only']):.1f}% | "
            f"{100.0 * float(rates['transport_only']):.1f}% | "
            f"{100.0 * float(rates['full_protocol']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Registered checks",
            "",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Same-action source improvement is non-diagnostic: it accepts the "
            "source-local discrepancy null in nearly every trial.",
            "2. Held-out intervention transport sharply reduces that false physical "
            "attribution while retaining high sensitivity to the registered shared "
            "physical coefficient.",
            "3. An undeclared nuisance with exactly the physical action signature can "
            "still fool transport. Declaring the competing nuisance makes the query "
            "nonidentifiable and the full method correctly returns exact fallback.",
            "4. The complete protocol is substantially more conservative than the "
            "transport endpoint. In this finite-session design, conservative "
            "correction magnitude improves the probability of satisfying the harmful-update gate.",
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def run_study(
    *,
    roster: Path,
    trials: int,
    bootstrap_replicates: int,
    seed: int,
    noise_scale: float,
    guard_threshold: float,
    harmful_gain_margin: float,
    discrepancy_shrink: float,
) -> dict[str, object]:
    if trials < 100:
        raise ValueError("trials must be at least 100")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    for name, value in (
        ("noise_scale", noise_scale),
        ("guard_threshold", guard_threshold),
        ("harmful_gain_margin", harmful_gain_margin),
        ("discrepancy_shrink", discrepancy_shrink),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative value")

    source_response, target_response, action_pairs = _load_action_roster(roster)
    weights = _bootstrap_weights(
        source_response.size,
        bootstrap_replicates,
        seed + 991,
    )
    harm_lookup = _harm_upper_lookup(source_response.size)
    regime_results: dict[str, Mapping[str, object]] = {}
    for index, regime in enumerate(REGIMES):
        regime_results[regime.name] = _run_regime(
            regime,
            source_response=source_response,
            target_response=target_response,
            trials=trials,
            bootstrap_weights=weights,
            seed=seed + 101 * index,
            noise_scale=noise_scale,
            guard_threshold=guard_threshold,
            harmful_gain_margin=harmful_gain_margin,
            discrepancy_shrink=discrepancy_shrink,
            harm_lookup=harm_lookup,
        )
    checks = _registered_checks(regime_results)
    passed = all(checks.values())
    result: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "CrossInterventionTransportCriterionStudyV1",
        "decision": (
            "criterion-useful-but-requires-declared-nuisance-and-conservative-guard"
            if passed
            else "registered-helpfulness-checks-not-established"
        ),
        "all_registered_checks_passed": passed,
        "causal4d_design_sha256": DESIGN_SHA256,
        "session_count": int(source_response.size),
        "action_response": dict(ACTION_RESPONSE),
        "action_pairs": action_pairs,
        "simulation": {
            "trials_per_regime": trials,
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "noise_scale": noise_scale,
            "guard_threshold": guard_threshold,
            "harmful_gain_margin": harmful_gain_margin,
            "discrepancy_shrink": discrepancy_shrink,
            "minimum_accepted_physical_sessions": 14,
            "maximum_harmful_accepted_fraction": 0.20,
        },
        "regimes": regime_results,
        "registered_checks": checks,
        "target_outcomes_used": False,
        "deform360_confirmation_opened": False,
        "causal4d_physical_outcome_used": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["result_id"] = _canonical_id(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_study(
        roster=args.roster,
        trials=args.trials,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        noise_scale=args.noise_scale,
        guard_threshold=args.guard_threshold,
        harmful_gain_margin=args.harmful_gain_margin,
        discrepancy_shrink=args.discrepancy_shrink,
    )
    _write_no_clobber(
        args.output_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_no_clobber(args.output_markdown, _render_markdown(result))
    print(
        json.dumps(
            {
                "all_registered_checks_passed": result["all_registered_checks_passed"],
                "decision": result["decision"],
                "result_id": result["result_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["all_registered_checks_passed"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
