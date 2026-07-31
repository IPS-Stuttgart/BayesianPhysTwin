"""Protocol and weighting helpers for the frozen two-seed DEFORM ensemble."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

DEFORM_DLO_DEEP_ENSEMBLE_SCHEMA_VERSION = 1
DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT = "deform-dlo-deep-ensemble-v1"


def _identity(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if (
        not str(value.get("repository_path", ""))
        or len(str(value.get("sha256", ""))) != 64
    ):
        raise ValueError(f"{label} identity is invalid")
    return value


def load_deform_dlo1_deep_ensemble_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the standalone two-seed evaluator protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO_DEEP_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError("unsupported DLO deep-ensemble schema")
    if payload.get("contract") != DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT:
        raise ValueError("unsupported DLO deep-ensemble contract")
    upstream = payload.get("upstream")
    parents = payload.get("parents")
    evaluation = payload.get("evaluation")
    policy = payload.get("policy")
    if not all(
        isinstance(value, Mapping) for value in (upstream, parents, evaluation, policy)
    ):
        raise ValueError("DLO deep-ensemble protocol is incomplete")
    if (
        upstream.get("repository") != "https://github.com/roahmlab/DEFORM"
        or len(str(upstream.get("commit", ""))) != 40
    ):
        raise ValueError("DLO deep-ensemble upstream differs")
    _identity(parents.get("seed42_longrun_protocol"), label="seed42 protocol")
    _identity(parents.get("seed42_source_manifest"), label="seed42 manifest")
    _identity(parents.get("seed43_source_protocol"), label="seed43 protocol")
    if (
        evaluation.get("dlo_type") != "DLO1"
        or int(evaluation.get("frame_count", -1)) != 500
        or int(evaluation.get("node_count", -1)) != 13
        or evaluation.get("cublas_workspace_config") != ":4096:8"
        or float(evaluation.get("source_replay_tolerance_m", math.nan)) != 1e-6
        or evaluation.get("official_eval_read") is not False
        or policy.get("member_checkpoint_selection")
        != "minimum-validation-l1-tie-earliest-v1"
        or policy.get("comparison_baseline")
        != "lower-validation-error-single-member-v1"
        or policy.get("fallback") != "comparison-baseline-exact"
    ):
        raise ValueError("DLO deep-ensemble evaluation policy differs")
    operators = policy.get("operators")
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)):
        raise ValueError("DLO deep-ensemble operators are malformed")
    normalized_operators = []
    expected = (
        ("equal_weight_predictive_mean", "uniform"),
        ("validation_softmax_predictive_mean", "validation-softmax"),
    )
    for raw_operator, (expected_name, expected_weighting) in zip(
        operators,
        expected,
        strict=True,
    ):
        if not isinstance(raw_operator, Mapping):
            raise ValueError("DLO deep-ensemble operator is malformed")
        name = str(raw_operator.get("name", ""))
        weighting = str(raw_operator.get("weighting", ""))
        if name != expected_name or weighting != expected_weighting:
            raise ValueError("DLO deep-ensemble operator bank differs")
        normalized = dict(raw_operator)
        if weighting == "validation-softmax":
            temperature = float(raw_operator.get("temperature_m", math.nan))
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("DLO deep-ensemble temperature is invalid")
            normalized["temperature_m"] = temperature
        normalized_operators.append(normalized)
    if len(normalized_operators) != len(expected):
        raise ValueError("DLO deep-ensemble operator count differs")
    for key in ("validation_improvement_min", "source_transfer_improvement_min"):
        value = float(policy.get(key, math.nan))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("DLO deep-ensemble gate is invalid")
    if int(policy.get("source_transfer_minimum_case_wins", -1)) != 5:
        raise ValueError("DLO deep-ensemble win gate differs")
    if (
        float(policy.get("coordinate_variance_floor_m2", math.nan)) != 0.000025
        or float(policy.get("coordinate_interval_nominal_coverage", math.nan)) != 0.9
    ):
        raise ValueError("DLO deep-ensemble uncertainty policy differs")
    fresh = policy.get("fresh_confirmation")
    if (
        not isinstance(fresh, Mapping)
        or fresh.get("dlo_type") != "DLO2"
        or fresh.get("seeds") != [42, 43]
        or fresh.get("same_split_horizon_batch_updates_and_operators") is not True
        or fresh.get("no_dlo1_retuning") is not True
        or fresh.get("official_eval_remains_closed_until_fresh_source_gate") is not True
    ):
        raise ValueError("DLO deep-ensemble fresh policy differs")

    result = dict(payload)
    result["policy"] = {**policy, "operators": normalized_operators}
    result["protocol_path"] = str(source)
    return result


def build_deform_two_seed_weights(
    validation_l1_m: Mapping[int, float],
    policy: Mapping[str, object],
) -> dict[str, dict[int, float]]:
    """Resolve the registered two-seed weights from validation errors only."""

    errors = {int(seed): float(error) for seed, error in validation_l1_m.items()}
    if set(errors) != {42, 43} or any(
        not math.isfinite(error) or error < 0.0 for error in errors.values()
    ):
        raise ValueError("two-seed validation errors are invalid")
    operators = policy.get("operators")
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)):
        raise ValueError("two-seed operator policy is malformed")
    arms: dict[str, dict[int, float]] = {}
    for operator in operators:
        if not isinstance(operator, Mapping):
            raise ValueError("two-seed operator is malformed")
        name = str(operator.get("name", ""))
        weighting = str(operator.get("weighting", ""))
        if weighting == "uniform":
            arms[name] = {42: 0.5, 43: 0.5}
        elif weighting == "validation-softmax":
            temperature = float(operator.get("temperature_m", math.nan))
            logits = np.asarray(
                [
                    -(errors[seed] - min(errors.values())) / temperature
                    for seed in (42, 43)
                ],
                dtype=np.float64,
            )
            unnormalized = np.exp(logits - np.max(logits))
            normalized = unnormalized / np.sum(unnormalized)
            arms[name] = {
                seed: float(weight)
                for seed, weight in zip((42, 43), normalized, strict=True)
            }
        else:
            raise ValueError("two-seed weighting is unsupported")
    return arms


def validate_deform_two_seed_manifests(
    seed42: Mapping[str, object],
    seed43: Mapping[str, object],
) -> None:
    """Require identical source bytes and partitions across the two seeds."""

    for manifest in (seed42, seed43):
        if (
            manifest.get("contract") != "deform-dlo-source-reproduction-v1"
            or manifest.get("dlo_type") != "DLO1"
            or manifest.get("official_eval_read") is not False
        ):
            raise ValueError("two-seed source manifest differs")
    if seed42.get("split") != seed43.get("split"):
        raise ValueError("two-seed source partitions differ")

    def identities(manifest: Mapping[str, object]) -> dict[str, tuple[object, object]]:
        trajectories = manifest.get("trajectories")
        if not isinstance(trajectories, Mapping):
            raise ValueError("two-seed source trajectories are malformed")
        result = {}
        for name, raw_identity in trajectories.items():
            if not isinstance(raw_identity, Mapping):
                raise ValueError("two-seed trajectory identity is malformed")
            result[str(name)] = (
                raw_identity.get("sha256"),
                raw_identity.get("size_bytes"),
            )
        return result

    if identities(seed42) != identities(seed43):
        raise ValueError("two-seed source trajectory bytes differ")
