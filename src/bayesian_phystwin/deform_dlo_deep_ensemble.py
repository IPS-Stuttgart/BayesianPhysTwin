"""Protocol and weighting helpers for the frozen two-seed DEFORM ensemble."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

DEFORM_DLO_DEEP_ENSEMBLE_SCHEMA_VERSION = 1
DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT = "deform-dlo-deep-ensemble-v1"
DEFORM_DLO_DEEP_ENSEMBLE_RESULT_CONTRACT = (
    "deform-dlo-deep-ensemble-result-v1"
)
DEFORM_DLO2_DEEP_ENSEMBLE_CONTRACT = "deform-dlo2-deep-ensemble-v1"
DEFORM_DLO2_DEEP_ENSEMBLE_RESULT_CONTRACT = (
    "deform-dlo2-deep-ensemble-result-v1"
)
DEFORM_DLO2_DEEP_SEED_AUTHORIZATION_CONTRACT = (
    "deform-dlo2-deep-seed-authorization-v1"
)


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


def load_deform_dlo2_deep_ensemble_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the fresh DLO2 two-seed confirmation protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO_DEEP_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError("unsupported DLO2 deep-ensemble schema")
    if payload.get("contract") != DEFORM_DLO2_DEEP_ENSEMBLE_CONTRACT:
        raise ValueError("unsupported DLO2 deep-ensemble contract")
    upstream = payload.get("upstream")
    parents = payload.get("parents")
    evaluation = payload.get("evaluation")
    policy = payload.get("policy")
    if not all(
        isinstance(value, Mapping) for value in (upstream, parents, evaluation, policy)
    ):
        raise ValueError("DLO2 deep-ensemble protocol is incomplete")
    if (
        upstream.get("repository") != "https://github.com/roahmlab/DEFORM"
        or len(str(upstream.get("commit", ""))) != 40
    ):
        raise ValueError("DLO2 deep-ensemble upstream differs")
    for label in (
        "dlo1_ensemble_protocol",
        "seed42_source_protocol",
        "seed43_source_protocol",
    ):
        _identity(parents.get(label), label=label)
    if (
        evaluation.get("dlo_type") != "DLO2"
        or int(evaluation.get("frame_count", -1)) != 500
        or int(evaluation.get("node_count", -1)) != 12
        or evaluation.get("cublas_workspace_config") != ":4096:8"
        or float(evaluation.get("source_replay_tolerance_m", math.nan)) != 1e-6
        or evaluation.get("official_eval_read") is not False
        or policy.get("member_checkpoint_selection")
        != "minimum-validation-l1-tie-earliest-v1"
        or policy.get("comparison_baseline")
        != "lower-validation-error-single-member-v1"
        or policy.get("fallback") != "comparison-baseline-exact"
    ):
        raise ValueError("DLO2 deep-ensemble evaluation policy differs")
    operators = policy.get("operators")
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)):
        raise ValueError("DLO2 deep-ensemble operators are malformed")
    expected = (
        ("equal_weight_predictive_mean", "uniform"),
        ("validation_softmax_predictive_mean", "validation-softmax"),
    )
    normalized_operators = []
    for raw_operator, (expected_name, expected_weighting) in zip(
        operators,
        expected,
        strict=True,
    ):
        if not isinstance(raw_operator, Mapping):
            raise ValueError("DLO2 deep-ensemble operator is malformed")
        normalized = dict(raw_operator)
        if (
            raw_operator.get("name") != expected_name
            or raw_operator.get("weighting") != expected_weighting
        ):
            raise ValueError("DLO2 deep-ensemble operator bank differs")
        if expected_weighting == "validation-softmax":
            temperature = float(raw_operator.get("temperature_m", math.nan))
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("DLO2 deep-ensemble temperature is invalid")
            normalized["temperature_m"] = temperature
        normalized_operators.append(normalized)
    if len(normalized_operators) != len(expected):
        raise ValueError("DLO2 deep-ensemble operator count differs")
    for key in ("validation_improvement_min", "source_transfer_improvement_min"):
        value = float(policy.get(key, math.nan))
        if not math.isfinite(value) or value != 0.01:
            raise ValueError("DLO2 deep-ensemble gate differs")
    if (
        int(policy.get("source_transfer_minimum_case_wins", -1)) != 5
        or float(policy.get("candidate_published_reference_l1_m", math.nan))
        != 0.0097
        or float(policy.get("candidate_published_error_multiplier_max", math.nan))
        != 1.1
        or int(policy.get("candidate_minimum_persistence_wins", -1)) != 6
        or float(policy.get("coordinate_variance_floor_m2", math.nan)) != 0.000025
        or float(policy.get("coordinate_interval_nominal_coverage", math.nan))
        != 0.9
    ):
        raise ValueError("DLO2 deep-ensemble confirmation gate differs")
    alltrain = policy.get("alltrain_authorization")
    if (
        not isinstance(alltrain, Mapping)
        or alltrain.get("dlo_type") != "DLO2"
        or alltrain.get("seeds") != [42, 43]
        or alltrain.get("same_operator_and_weights") is not True
        or alltrain.get("same_total_update_budget") is not True
        or alltrain.get("no_source_retuning") is not True
        or alltrain.get("official_eval_remains_closed_until_alltrain_refit")
        is not True
    ):
        raise ValueError("DLO2 deep-ensemble all-train policy differs")

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
    *,
    dlo_type: str = "DLO1",
) -> None:
    """Require identical source bytes and partitions across the two seeds."""

    for manifest in (seed42, seed43):
        if (
            manifest.get("contract") != "deform-dlo-source-reproduction-v1"
            or manifest.get("dlo_type") != dlo_type
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


def validate_deform_dlo2_deep_ensemble_parent(
    source_protocol: Mapping[str, object],
    parent_protocol: Mapping[str, object],
    parent_result: Mapping[str, object],
    selection_seal: Mapping[str, object],
    *,
    parent_protocol_sha256: str,
    selection_seal_sha256: str,
) -> dict[str, object]:
    """Authorize one fresh DLO2 seed only after the frozen DLO1 ensemble wins."""

    authorization = source_protocol.get("authorization")
    required = (
        authorization.get("required_parent_deep_ensemble")
        if isinstance(authorization, Mapping)
        else None
    )
    policy = parent_protocol.get("policy")
    selection = parent_result.get("selection")
    sealed_selection = selection_seal.get("selection")
    selected_spec = parent_result.get("selected_spec")
    candidate_specs = selection_seal.get("candidate_specs")
    source_test = parent_result.get("source_test")
    transfer = source_test.get("transfer") if isinstance(source_test, Mapping) else None
    uncertainty = parent_result.get("uncertainty")
    result_seal = parent_result.get("selection_seal")
    sealed_protocol = selection_seal.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (
            required,
            policy,
            selection,
            sealed_selection,
            selected_spec,
            candidate_specs,
            transfer,
            uncertainty,
            result_seal,
            sealed_protocol,
        )
    ):
        raise ValueError("DLO1 ensemble parent is incomplete")
    required = cast(Mapping[str, object], required)
    policy = cast(Mapping[str, object], policy)
    selection = cast(Mapping[str, object], selection)
    sealed_selection = cast(Mapping[str, object], sealed_selection)
    selected_spec = cast(Mapping[str, object], selected_spec)
    candidate_specs = cast(Mapping[str, object], candidate_specs)
    transfer = cast(Mapping[str, object], transfer)
    uncertainty = cast(Mapping[str, object], uncertainty)
    result_seal = cast(Mapping[str, object], result_seal)
    sealed_protocol = cast(Mapping[str, object], sealed_protocol)

    selected_arm = str(parent_result.get("selected_arm", ""))
    weights = selected_spec.get("weights")
    normalized_weights = (
        {int(seed): float(weight) for seed, weight in weights.items()}
        if isinstance(weights, Mapping)
        else {}
    )
    variance_scale = float(
        str(uncertainty.get("validation_fitted_variance_scale", math.nan))
    )
    validation_gain = float(str(selection.get("relative_improvement", -math.inf)))
    source_gain = float(str(transfer.get("relative_improvement", -math.inf)))
    source_wins = int(str(transfer.get("wins", -1)))
    if (
        source_protocol.get("dlo_types") != ("DLO2",)
        or required.get("protocol_sha256") != parent_protocol_sha256
        or required.get("result_contract")
        != DEFORM_DLO_DEEP_ENSEMBLE_RESULT_CONTRACT
        or required.get("selection_contract")
        != DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT
        or required.get("exact_fallback") is not False
        or required.get("fresh_dlo2_deep_ensemble_authorized") is not True
        or parent_protocol.get("contract") != DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT
        or parent_result.get("contract")
        != DEFORM_DLO_DEEP_ENSEMBLE_RESULT_CONTRACT
        or parent_result.get("official_eval_read") is not False
        or parent_result.get("exact_fallback") is not False
        or parent_result.get("fresh_dlo2_deep_ensemble_authorized") is not True
        or selection_seal.get("contract") != DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT
        or selection_seal.get("official_eval_read") is not False
        or selection_seal.get("source_test_evaluated_by_this_stage") is not False
        or sealed_protocol.get("sha256") != parent_protocol_sha256
        or result_seal.get("sha256") != selection_seal_sha256
        or selection != sealed_selection
        or selection.get("fallback_used") is not False
        or not selected_arm
        or selection.get("selected_arm") != selected_arm
        or candidate_specs.get(selected_arm) != selected_spec
        or selected_spec.get("operator") != "predictive_mean"
        or set(normalized_weights) != {42, 43}
        or not math.isclose(sum(normalized_weights.values()), 1.0, abs_tol=1e-12)
        or any(
            not math.isfinite(weight) or weight < 0.0
            for weight in normalized_weights.values()
        )
        or not math.isfinite(validation_gain)
        or validation_gain
        < float(str(policy.get("validation_improvement_min", math.inf)))
        or not math.isfinite(source_gain)
        or source_gain
        < float(str(policy.get("source_transfer_improvement_min", math.inf)))
        or source_wins
        < int(str(policy.get("source_transfer_minimum_case_wins", 10**9)))
        or not math.isfinite(variance_scale)
        or variance_scale < 1.0
        or parent_result.get("fresh_confirmation_contract")
        != policy.get("fresh_confirmation")
    ):
        raise ValueError("DLO1 ensemble parent did not authorize fresh DLO2")

    return {
        "contract": DEFORM_DLO_DEEP_ENSEMBLE_RESULT_CONTRACT,
        "selection_contract": DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT,
        "selected_arm": selected_arm,
        "selected_spec": dict(selected_spec),
        "validation_relative_improvement": validation_gain,
        "source_transfer_relative_improvement": source_gain,
        "source_transfer_wins": source_wins,
        "validation_fitted_variance_scale": variance_scale,
        "fresh_dlo2_deep_ensemble_authorized": True,
    }
