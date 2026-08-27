"""Controlled evaluation of the cross-intervention transport criterion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

DESIGN_SHA256: Final = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
PROTOCOL_SCHEMA: Final = "bayesian_phystwin.cross_intervention_criterion_benchmark"
RESULT_SCHEMA: Final = "bayesian_phystwin.cross_intervention_criterion_result"
RULES: Final = (
    "source_fit_only",
    "held_out_transport",
    "held_out_transport_with_controls",
)
SCENARIOS: Final = (
    "shared_physical_correct_transport",
    "source_local_discrepancy",
    "stationary_shared_bias",
    "partially_shared_discrepancy",
    "mixed_transport_and_residual_persistence",
    "shared_physical_misspecified_transport",
    "action_locked_nuisance",
)
LABELS: Final = {
    "shared_physical_correct_transport": "Shared physical, correct transport",
    "source_local_discrepancy": "Source-local discrepancy",
    "stationary_shared_bias": "Stationary shared bias",
    "partially_shared_discrepancy": "Partially shared discrepancy",
    "mixed_transport_and_residual_persistence": "Mixed transport + persistence",
    "shared_physical_misspecified_transport": "Physical, wrong transport",
    "action_locked_nuisance": "Action-locked nuisance",
}

FloatArray = NDArray[np.float64]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def content_id(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _number(value: object, *, name: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its admissible range")
    return result


def validate_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("unexpected cross-intervention protocol")
    if value.get("causal4d_design_sha256") != DESIGN_SHA256:
        raise ValueError("protocol does not bind the frozen Causal4D design")
    for name, minimum in (
        ("study_count", 100),
        ("bootstrap_replicates", 100),
        ("identity_draw_count", 10000),
        ("seed", 0),
    ):
        item = value.get(name)
        if type(item) is not int or item < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in (
        "canonical_noise_sd",
        "confidence_level",
        "identity_noise_sd",
        "misspecified_target_scale",
        "mixed_persistence_strength",
        "mixed_transport_strength",
        "normalized_action_unit_m",
        "partial_shared_correlation",
        "prior_variance",
    ):
        value[name] = _number(value.get(name), name=name)
    noise = value.get("noise_sd_grid")
    if type(noise) is not list:
        raise ValueError("noise_sd_grid must be a JSON array")
    normalized = tuple(_number(item, name="noise", minimum=1e-12) for item in noise)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("noise_sd_grid must be unique and increasing")
    if value["canonical_noise_sd"] not in normalized:
        raise ValueError("canonical_noise_sd must occur in noise_sd_grid")
    value["noise_sd_grid"] = list(normalized)
    vectors = value.get("action_vectors")
    if type(vectors) is not dict or set(vectors) != {
        "lift_low",
        "lift_high",
        "lower_high",
        "lateral_low",
    }:
        raise ValueError("action_vectors must bind the four frozen actions")
    for action, vector in vectors.items():
        if type(vector) is not list or len(vector) != 2:
            raise ValueError(f"{action} must have one two-dimensional vector")
        if not any(_number(item, name=action) != 0.0 for item in vector):
            raise ValueError(f"{action} must be nonzero")
    return value


def load_pairs(
    roster: Mapping[str, Any],
    *,
    actions: set[str],
) -> tuple[dict[str, str], ...]:
    if roster.get("causal4d_design_sha256") != DESIGN_SHA256:
        raise ValueError("roster design identity differs")
    if roster.get("schema_version") != 2:
        raise ValueError("unsupported sparse-pair roster")
    if roster.get("reverse_same_session_reuse_allowed") is not False:
        raise ValueError("reverse reuse must remain forbidden")
    rows = roster.get("session_pairs")
    if type(rows) is not list or len(rows) != 18:
        raise ValueError("the frozen roster must contain 18 pairs")
    pairs: list[dict[str, str]] = []
    sessions: set[str] = set()
    executions: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            raise ValueError("each roster row must be an object")
        pair = cast(dict[str, Any], row)
        normalized: dict[str, str] = {}
        for name in (
            "object_session_id",
            "source_execution_id",
            "target_execution_id",
            "source_action_id",
            "target_action_id",
        ):
            item = pair.get(name)
            if type(item) is not str or not item:
                raise ValueError(f"invalid {name}")
            normalized[name] = item
        if {normalized["source_action_id"], normalized["target_action_id"]} - actions:
            raise ValueError("roster names an unknown action")
        if normalized["source_action_id"] == normalized["target_action_id"]:
            raise ValueError("every row must be cross-action")
        if normalized["object_session_id"] in sessions:
            raise ValueError("physical sessions repeat")
        sessions.add(normalized["object_session_id"])
        for name in ("source_execution_id", "target_execution_id"):
            if normalized[name] in executions:
                raise ValueError("execution IDs repeat")
            executions.add(normalized[name])
        pairs.append(normalized)
    return tuple(sorted(pairs, key=lambda pair: pair["object_session_id"]))


def _seed(base: int, *stream: int) -> int:
    return int(np.random.SeedSequence([base, *stream]).generate_state(1)[0])


def _lower_bounds(
    values: FloatArray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> FloatArray:
    studies, statistics, sessions = values.shape
    output = np.empty((studies, statistics), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, studies, 40):
        stop = min(studies, start + 40)
        count = stop - start
        index = rng.integers(0, sessions, size=(count, replicates, sessions))
        sampled = np.take_along_axis(
            values[start:stop, :, None, :], index[:, None, :, :], axis=3
        )
        output[start:stop] = np.quantile(
            sampled.mean(axis=3), 1.0 - confidence, axis=2, method="linear"
        )
    return output


def _gain(error: FloatArray, correction: FloatArray) -> FloatArray:
    return np.sum(error**2, axis=2) - np.sum((error - correction) ** 2, axis=2)


def _simulate(
    scenario: str,
    *,
    source_action: FloatArray,
    target_action: FloatArray,
    noise: float,
    protocol: Mapping[str, Any],
    simulation_seed: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(simulation_seed)
    studies = cast(int, protocol["study_count"])
    sessions, dimension = source_action.shape
    coefficient = rng.normal(size=(studies, sessions))
    source_noise = rng.normal(scale=noise, size=(studies, sessions, dimension))
    target_noise = rng.normal(scale=noise, size=(studies, sessions, dimension))
    if scenario == SCENARIOS[0]:
        source_error = source_action * coefficient[:, :, None] + source_noise
        target_error = target_action * coefficient[:, :, None] + target_noise
    elif scenario == SCENARIOS[1]:
        source_error = rng.normal(size=source_noise.shape) + source_noise
        target_error = rng.normal(size=target_noise.shape) + target_noise
    elif scenario == SCENARIOS[2]:
        bias = rng.normal(size=source_noise.shape)
        source_error, target_error = bias + source_noise, bias + target_noise
    elif scenario == SCENARIOS[3]:
        source_term = rng.normal(size=source_noise.shape)
        independent = rng.normal(size=target_noise.shape)
        rho = cast(float, protocol["partial_shared_correlation"])
        source_error = source_term + source_noise
        target_error = (
            rho * source_term + np.sqrt(1.0 - rho**2) * independent + target_noise
        )
    elif scenario == SCENARIOS[4]:
        source_error = rng.normal(size=source_noise.shape) + source_noise
        prior = cast(float, protocol["prior_variance"])
        norm = np.sum(source_action**2, axis=1)
        provisional = np.einsum("nd,snd->sn", source_action, source_error)
        provisional /= noise**2 / prior + norm[None, :]
        action_aligned = target_action * provisional[:, :, None]
        target_error = (
            cast(float, protocol["mixed_transport_strength"]) * action_aligned
            + cast(float, protocol["mixed_persistence_strength"]) * source_error
            + target_noise
        )
    elif scenario == SCENARIOS[5]:
        source_error = source_action * coefficient[:, :, None] + source_noise
        scale = cast(float, protocol["misspecified_target_scale"])
        target_error = scale * target_action * coefficient[:, :, None] + target_noise
    else:
        nuisance = rng.normal(size=(studies, sessions))
        source_error = source_action * nuisance[:, :, None] + source_noise
        target_error = target_action * nuisance[:, :, None] + target_noise

    prior = cast(float, protocol["prior_variance"])
    norm = np.sum(source_action**2, axis=1)
    posterior = np.einsum("nd,snd->sn", source_action, source_error)
    posterior /= noise**2 / prior + norm[None, :]
    source_candidate = source_action * posterior[:, :, None]
    physical_candidate = target_action * posterior[:, :, None]
    physical_gain = _gain(target_error, physical_candidate)
    discrepancy_gain = _gain(target_error, source_candidate)
    residual_gain = _gain(target_error, source_error)
    source_gain = np.sum(source_error**2, axis=2)
    source_gain -= np.sum((source_error - source_candidate) ** 2, axis=2)
    statistics = np.stack(
        (
            source_gain,
            physical_gain,
            physical_gain - discrepancy_gain,
            physical_gain - residual_gain,
        ),
        axis=1,
    )
    lower = _lower_bounds(
        statistics,
        replicates=cast(int, protocol["bootstrap_replicates"]),
        confidence=cast(float, protocol["confidence_level"]),
        seed=bootstrap_seed,
    )
    passed = (
        lower[:, 0] > 0,
        lower[:, 1] > 0,
        np.all(lower[:, 1:] > 0, axis=1),
    )
    return {
        "source_fit_pass_count": int(np.count_nonzero(passed[0])),
        "held_out_pass_count": int(np.count_nonzero(passed[1])),
        "controlled_pass_count": int(np.count_nonzero(passed[2])),
        "study_count": studies,
        "mean_physical_gain": float(np.mean(physical_gain)),
        "mean_discrepancy_gain": float(np.mean(discrepancy_gain)),
        "mean_last_residual_gain": float(np.mean(residual_gain)),
    }


def _rate(record: Mapping[str, Any], name: str) -> float:
    return int(record[name]) / int(record["study_count"])


def _identity_check(
    target_action: FloatArray,
    protocol: Mapping[str, Any],
) -> dict[str, float]:
    rng = np.random.default_rng(_seed(cast(int, protocol["seed"]), 9001))
    draws = cast(int, protocol["identity_draw_count"])
    noise = cast(float, protocol["identity_noise_sd"])
    physical_error = 0.0
    local_error = 0.0
    for action in np.unique(target_action, axis=0):
        for posterior in (-1.5, -0.5, 0.5, 1.5):
            correction = action * posterior
            expected = float(correction @ correction)
            physical_target = correction + rng.normal(
                scale=noise, size=(draws, correction.size)
            )
            local_target = rng.normal(scale=noise, size=(draws, correction.size))
            physical_gain = np.sum(physical_target**2, axis=1)
            physical_gain -= np.sum((physical_target - correction) ** 2, axis=1)
            local_gain = np.sum(local_target**2, axis=1)
            local_gain -= np.sum((local_target - correction) ** 2, axis=1)
            physical_error = max(
                physical_error,
                abs(float(physical_gain.mean()) - expected),
            )
            local_error = max(
                local_error,
                abs(float(local_gain.mean()) + expected),
            )
    return {
        "maximum_physical_absolute_error": physical_error,
        "maximum_local_absolute_error": local_error,
    }


def run_benchmark(
    *,
    protocol: Mapping[str, Any],
    roster: Mapping[str, Any],
    protocol_sha256: str,
    roster_sha256: str,
) -> dict[str, object]:
    config = validate_protocol(protocol)
    vectors = cast(Mapping[str, Sequence[float]], config["action_vectors"])
    pairs = load_pairs(roster, actions=set(vectors))
    source_action = np.asarray([vectors[p["source_action_id"]] for p in pairs])
    target_action = np.asarray([vectors[p["target_action_id"]] for p in pairs])
    canonical: dict[str, object] = {}
    sensitivity: list[dict[str, object]] = []
    noise_grid = cast(Sequence[float], config["noise_sd_grid"])
    for noise_index, noise in enumerate(noise_grid):
        records: dict[str, dict[str, object]] = {}
        for scenario_index, scenario in enumerate(SCENARIOS):
            records[scenario] = _simulate(
                scenario,
                source_action=source_action,
                target_action=target_action,
                noise=float(noise),
                protocol=config,
                simulation_seed=_seed(
                    config["seed"], noise_index, scenario_index, 0
                ),
                bootstrap_seed=_seed(
                    config["seed"], noise_index, scenario_index, 1
                ),
            )
        sensitivity.append(
            {
                "noise_sd": float(noise),
                "physical_controlled_pass_rate": _rate(
                    records[SCENARIOS[0]], "controlled_pass_count"
                ),
                "local_controlled_pass_rate": _rate(
                    records[SCENARIOS[1]], "controlled_pass_count"
                ),
                "action_locked_controlled_pass_rate": _rate(
                    records[SCENARIOS[-1]], "controlled_pass_count"
                ),
            }
        )
        if float(noise) == config["canonical_noise_sd"]:
            canonical = records
    if not canonical:
        raise RuntimeError("canonical noise cell was not evaluated")

    summary = {
        "physical_controlled_pass_rate": _rate(
            canonical[SCENARIOS[0]], "controlled_pass_count"
        ),
        "local_source_fit_pass_rate": _rate(
            canonical[SCENARIOS[1]], "source_fit_pass_count"
        ),
        "local_controlled_pass_rate": _rate(
            canonical[SCENARIOS[1]], "controlled_pass_count"
        ),
        "mixed_held_out_pass_rate": _rate(
            canonical[SCENARIOS[4]], "held_out_pass_count"
        ),
        "mixed_controlled_pass_rate": _rate(
            canonical[SCENARIOS[4]], "controlled_pass_count"
        ),
        "wrong_transport_controlled_pass_rate": _rate(
            canonical[SCENARIOS[5]], "controlled_pass_count"
        ),
        "action_locked_controlled_pass_rate": _rate(
            canonical[SCENARIOS[6]], "controlled_pass_count"
        ),
    }
    identity = _identity_check(target_action, config)
    criteria = {
        "gain_identities_reproduced": max(identity.values()) < 0.03,
        "source_fit_accepts_local_discrepancy": (
            summary["local_source_fit_pass_rate"] >= 0.95
        ),
        "controlled_transport_rejects_local_discrepancy": (
            summary["local_controlled_pass_rate"] <= 0.02
        ),
        "moderate_noise_physical_power": (
            summary["physical_controlled_pass_rate"] >= 0.90
        ),
        "held_out_gain_accepts_persistence_stress": (
            summary["mixed_held_out_pass_rate"] >= 0.80
        ),
        "matched_controls_reject_persistence_stress": (
            summary["mixed_controlled_pass_rate"] <= 0.10
        ),
        "action_locked_nuisance_remains_nonidentifiable": (
            summary["action_locked_controlled_pass_rate"] >= 0.90
        ),
        "wrong_transport_exposes_false_negative_boundary": (
            summary["wrong_transport_controlled_pass_rate"] <= 0.10
        ),
    }
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": content_id(config),
        "protocol_file_sha256": protocol_sha256,
        "roster_file_sha256": roster_sha256,
        "causal4d_design_sha256": DESIGN_SHA256,
        "session_pair_count": len(pairs),
        "study_count_per_cell": config["study_count"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "canonical_noise_sd": config["canonical_noise_sd"],
        "identity_check": identity,
        "canonical_scenarios": canonical,
        "noise_sensitivity": sensitivity,
        "canonical_summary": summary,
        "helpfulness_criteria": criteria,
        "decision": (
            "helpful_as_a_bounded_falsification_criterion"
            if all(criteria.values())
            else "helpfulness_not_demonstrated"
        ),
        "interpretation": (
            "Held-out transport rejects source-local corrections that source "
            "fit accepts, and matched residual controls reject improvement "
            "driven mainly by persistence. The criterion remains non-identifying "
            "for an action-locked nuisance and can reject a shared physical "
            "effect when simulator transport is wrong."
        ),
        "access_boundary": (
            "Synthetic draws only; no physical or protected outcome was read."
        ),
        "claim_boundary": config["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    scenarios = cast(
        Mapping[str, Mapping[str, Any]],
        result["canonical_scenarios"],
    )
    lines = [
        "# Cross-intervention criterion V1 result",
        "",
        "## Decision",
        "",
        f"**`{result['decision']}`**",
        "",
        str(result["interpretation"]),
        "",
        (
            "The benchmark uses the exact frozen 18-session Causal4D "
            "action-pair roster. Each cell contains "
            f"{result['study_count_per_cell']} simulated studies."
        ),
        "",
        f"## Canonical normalized noise SD = {float(result['canonical_noise_sd']):g}",
        "",
        (
            "| Scenario | Source fit | Held-out gain | Held-out + controls | "
            "Mean physical gain |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        record = scenarios[scenario]
        total = int(record["study_count"])
        source_rate = int(record["source_fit_pass_count"]) / total
        held_out_rate = int(record["held_out_pass_count"]) / total
        controlled_rate = int(record["controlled_pass_count"]) / total
        lines.append(
            f"| {LABELS[scenario]} | {source_rate:.1%} | "
            f"{held_out_rate:.1%} | {controlled_rate:.1%} | "
            f"{float(record['mean_physical_gain']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Noise sensitivity",
            "",
            (
                "| Noise SD | Correct physical | Local discrepancy | "
                "Action-locked nuisance |"
            ),
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cast(Sequence[Mapping[str, Any]], result["noise_sensitivity"]):
        lines.append(
            f"| {float(row['noise_sd']):g} | "
            f"{float(row['physical_controlled_pass_rate']):.1%} | "
            f"{float(row['local_controlled_pass_rate']):.1%} | "
            f"{float(row['action_locked_controlled_pass_rate']):.1%} |"
        )
    summary = cast(Mapping[str, Any], result["canonical_summary"])
    identity = cast(Mapping[str, Any], result["identity_check"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Source fit accepts the local discrepancy in "
                f"{float(summary['local_source_fit_pass_rate']):.1%} of studies; "
                "the controlled cross-action rule accepts it in "
                f"{float(summary['local_controlled_pass_rate']):.1%}. The "
                "correct physical control passes in "
                f"{float(summary['physical_controlled_pass_rate']):.1%}."
            ),
            "",
            (
                "The persistence stress passes held-out gain alone in "
                f"{float(summary['mixed_held_out_pass_rate']):.1%}, but only "
                f"{float(summary['mixed_controlled_pass_rate']):.1%} after "
                "requiring superiority over both residual controls."
            ),
            "",
            (
                "The proposition identities are reproduced with maximum errors "
                f"{float(identity['maximum_physical_absolute_error']):.6f} "
                "(physical) and "
                f"{float(identity['maximum_local_absolute_error']):.6f} (local)."
            ),
            "",
            (
                "The action-locked nuisance passes at approximately the "
                "physical-control rate, while wrong simulator transport rejects "
                "a genuinely shared coefficient. Those are the intended claim "
                "boundaries."
            ),
            "",
            "## Reproducibility",
            "",
            f"- Result ID: `{result['result_id']}`",
            f"- Protocol ID: `{result['protocol_id']}`",
            f"- Causal4D design SHA-256: `{result['causal4d_design_sha256']}`",
            "- Access boundary: synthetic draws only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
