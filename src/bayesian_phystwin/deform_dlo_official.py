"""Authorization and metrics for the one-shot official DEFORM DLO2 evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .deform_dlo_checkpoint_belief import evaluate_deform_coordinate_uncertainty

DEFORM_DLO2_OFFICIAL_SCHEMA_VERSION = 2
DEFORM_DLO2_OFFICIAL_CONTRACT = "deform-dlo2-official-eval-v2"
DEFORM_DLO2_DEEP_OFFICIAL_SCHEMA_VERSION = 1
DEFORM_DLO2_DEEP_OFFICIAL_CONTRACT = "deform-dlo2-deep-official-eval-v1"
DEFORM_CANONICAL_REFERENCE_DRAW = (1, 7, 9, 7, 11, 7, 13, 8, 8, 6, 8, 5, 8, 4)
DEFORM_UPSTREAM_TRAIN_SCRIPT_SHA256 = (
    "d45abe23a22b0f01fa266833844c4f9b71a2b7e375f8e955e3278b9e969acc55"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    number = int(str(value))
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _weights(value: object, *, label: str) -> dict[int, float]:
    raw = _mapping(value, label=label)
    result = {int(str(update)): float(str(weight)) for update, weight in raw.items()}
    if (
        not result
        or any(
            update < 0 or not math.isfinite(weight) or weight <= 0.0
            for update, weight in result.items()
        )
        or not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-10)
    ):
        raise ValueError(f"{label} is invalid")
    return result


def _reference_draw_indices(
    value: object,
    *,
    expected_case_count: int,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("canonical reference draw must be an array")
    indices = tuple(int(str(item)) for item in value)
    if (
        indices != DEFORM_CANONICAL_REFERENCE_DRAW
        or len(indices) != expected_case_count
        or any(index < 0 or index >= expected_case_count for index in indices)
    ):
        raise ValueError("canonical reference draw differs")
    return indices


def load_deform_dlo2_official_protocol(path: str | Path) -> dict[str, object]:
    """Load the immutable one-shot DLO2 evaluation protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO2_OFFICIAL_SCHEMA_VERSION:
        raise ValueError("unsupported DLO2 official-evaluation schema")
    if payload.get("contract") != DEFORM_DLO2_OFFICIAL_CONTRACT:
        raise ValueError("unsupported DLO2 official-evaluation contract")
    if payload.get("model_initialization") != "official-deform-dlo-initialization-v1":
        raise ValueError("DLO2 official initialization contract differs")

    parent = _mapping(
        payload.get("parent_alltrain_protocol"), label="parent_alltrain_protocol"
    )
    required = _mapping(payload.get("required_parent"), label="required_parent")
    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    methods = _mapping(payload.get("methods"), label="methods")
    gate = _mapping(payload.get("claim_gate"), label="claim_gate")
    uncertainty = _mapping(payload.get("uncertainty"), label="uncertainty")
    if (
        not str(parent.get("repository_path", ""))
        or len(str(parent.get("sha256", ""))) != 64
    ):
        raise ValueError("DLO2 official parent identity is invalid")
    if (
        required.get("result_contract") != "deform-dlo2-alltrain-result-v1"
        or required.get("official_eval_read") is not False
        or required.get("official_eval_execution_authorized") is not True
        or required.get("final_method_contract")
        != "deform-dlo2-alltrain-final-method-v1"
    ):
        raise ValueError("DLO2 official parent gate differs")
    reference = float(str(evaluation.get("published_reference_l1_m", math.nan)))
    reference_operator = _mapping(
        evaluation.get("published_reference_operator"),
        label="published_reference_operator",
    )
    if (
        evaluation.get("dlo_type") != "DLO2"
        or evaluation.get("partition") != "eval"
        or _positive_int(
            evaluation.get("expected_trajectory_count"),
            label="expected_trajectory_count",
        )
        != 14
        or _positive_int(
            evaluation.get("expected_frame_count"), label="expected_frame_count"
        )
        != 500
        or _positive_int(
            evaluation.get("expected_node_count"), label="expected_node_count"
        )
        != 12
        or evaluation.get("trajectory_policy")
        != "all-eval-files-sorted-once-plus-canonical-reference-draw-v2"
        or evaluation.get("failure_policy") != "seal-failure-no-retry-v1"
        or evaluation.get("metric") != "mean-coordinate-l1-m"
        or evaluation.get("horizon_breakdown") != "equal-frame-thirds-v1"
        or not math.isfinite(reference)
        or reference <= 0.0
    ):
        raise ValueError("DLO2 official evaluation contract differs")
    reference_draw = _reference_draw_indices(
        reference_operator.get("canonical_eval_indices"),
        expected_case_count=14,
    )
    if (
        reference_operator.get("loader")
        != "upstream-random-choices-with-replacement-v1"
        or reference_operator.get("upstream_train_script_sha256")
        != DEFORM_UPSTREAM_TRAIN_SCRIPT_SHA256
        or int(str(reference_operator.get("python_random_seed", -1))) != 0
        or int(str(reference_operator.get("preceding_train_population", -1))) != 56
        or int(str(reference_operator.get("preceding_train_draw_count", -1))) != 56
        or int(str(reference_operator.get("eval_population", -1))) != 14
        or int(str(reference_operator.get("eval_draw_count", -1))) != 14
        or reference_operator.get("canonical_filename_order") != "sorted-by-name-v1"
        or int(str(reference_operator.get("canonical_unique_index_count", -1)))
        != len(set(reference_draw))
        or reference_operator.get("upstream_glob_order") != "unspecified"
    ):
        raise ValueError("DLO2 published-reference operator differs")
    if (
        methods.get("candidate") != "preselected-alltrain-posterior"
        or methods.get("comparison_baseline")
        != "preselected-alltrain-single-checkpoint"
        or methods.get("action_aware_persistence") is not True
        or methods.get("target_selection") is not False
        or methods.get("target_calibration") is not False
        or methods.get("target_retries") is not False
        or methods.get("case_replacement") is not False
    ):
        raise ValueError("DLO2 official method policy differs")
    relative_improvement = float(
        str(gate.get("bayesian_relative_improvement_min", math.nan))
    )
    if (
        gate.get("published_reference_all_unique_strictly_better") is not True
        or gate.get("published_reference_canonical_draw_strictly_better") is not True
        or not math.isfinite(relative_improvement)
        or not 0.0 < relative_improvement < 1.0
        or _positive_int(
            gate.get("bayesian_minimum_case_wins"),
            label="bayesian_minimum_case_wins",
        )
        > 14
        or gate.get("require_all_expected_cases") is not True
    ):
        raise ValueError("DLO2 official claim gate differs")
    if (
        uncertainty.get("use_source_validation_scale_unchanged") is not True
        or uncertainty.get("report_coordinate_marginal_coverage") is not True
        or uncertainty.get("report_interval_width") is not True
        or uncertainty.get("report_gaussian_nll") is not True
        or uncertainty.get("report_coordinate_nees") is not True
        or uncertainty.get("report_horizon_breakdown") is not True
    ):
        raise ValueError("DLO2 official uncertainty policy differs")

    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def _checkpoint_records(value: object) -> dict[int, Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("all-train checkpoint records are malformed")
    indexed: dict[int, Mapping[str, object]] = {}
    for raw in value:
        record = _mapping(raw, label="all-train checkpoint")
        update = int(str(record.get("update", -1)))
        if update < 0 or update in indexed:
            raise ValueError("all-train checkpoint updates are invalid")
        indexed[update] = record
    return indexed


def validate_deform_dlo2_official_authorization(
    protocol: Mapping[str, object],
    alltrain_protocol: Mapping[str, object],
    alltrain_result: Mapping[str, object],
    final_method: Mapping[str, object],
    method_spec: Mapping[str, object],
    *,
    alltrain_protocol_sha256: str,
    alltrain_result_sha256: str,
    final_method_sha256: str,
    method_spec_sha256: str,
) -> dict[str, object]:
    """Return the exact frozen method only after all pre-target checks pass."""

    parent = _mapping(
        protocol.get("parent_alltrain_protocol"), label="parent_alltrain_protocol"
    )
    required = _mapping(protocol.get("required_parent"), label="required_parent")
    if (
        parent.get("sha256") != alltrain_protocol_sha256
        or alltrain_protocol.get("contract") != "deform-dlo2-alltrain-refit-v1"
        or alltrain_result.get("contract") != required.get("result_contract")
        or alltrain_result.get("official_eval_read")
        is not required.get("official_eval_read")
        or alltrain_result.get("official_eval_execution_authorized")
        is not required.get("official_eval_execution_authorized")
    ):
        raise ValueError("all-train result did not authorize official evaluation")
    result_protocol = _mapping(
        alltrain_result.get("protocol"), label="all-train result protocol"
    )
    result_final = _mapping(
        alltrain_result.get("final_method"), label="all-train final method identity"
    )
    result_spec = _mapping(
        alltrain_result.get("method_spec"), label="all-train method-spec identity"
    )
    if (
        result_protocol.get("sha256") != alltrain_protocol_sha256
        or result_final.get("sha256") != final_method_sha256
        or result_spec.get("sha256") != method_spec_sha256
        or len(alltrain_result_sha256) != 64
    ):
        raise ValueError("all-train result lineage differs")
    if (
        final_method.get("contract") != required.get("final_method_contract")
        or final_method.get("official_eval_read") is not False
        or method_spec.get("contract") != "deform-dlo2-alltrain-method-spec-v1"
        or method_spec.get("official_eval_read") is not False
    ):
        raise ValueError("all-train final method contract differs")
    final_spec = _mapping(
        final_method.get("method_spec"), label="final-method method-spec identity"
    )
    if final_spec.get("sha256") != method_spec_sha256:
        raise ValueError("final method points to a different method specification")

    operator = str(final_method.get("operator", ""))
    if operator not in (
        "parameter_mean",
        "predictive_mean",
        "predictive_median",
    ):
        raise ValueError("all-train final operator is invalid")
    final_weights = _weights(
        final_method.get("checkpoint_weights"), label="final checkpoint weights"
    )
    spec_weights = _weights(
        method_spec.get("checkpoint_weights"), label="method-spec checkpoint weights"
    )
    if operator != method_spec.get("operator") or final_weights != spec_weights:
        raise ValueError("all-train final method differs from its frozen spec")

    baseline = _mapping(
        final_method.get("comparison_baseline_checkpoint"),
        label="comparison baseline checkpoint",
    )
    baseline_update = int(str(baseline.get("update", -1)))
    if baseline_update != int(str(method_spec.get("comparison_baseline_update", -2))):
        raise ValueError("comparison checkpoint differs from its frozen update")
    raw_members = _mapping(
        final_method.get("member_checkpoints"), label="member checkpoints"
    )
    members = {
        int(str(update)): _mapping(identity, label="member checkpoint")
        for update, identity in raw_members.items()
    }
    if set(members) != set(final_weights) or any(
        int(str(identity.get("update", -1))) != update
        for update, identity in members.items()
    ):
        raise ValueError("final posterior member checkpoints differ")
    checkpoints = _checkpoint_records(alltrain_result.get("checkpoints"))
    if baseline_update not in checkpoints or checkpoints[baseline_update] != baseline:
        raise ValueError("comparison checkpoint is outside the all-train run")
    if any(checkpoints.get(update) != identity for update, identity in members.items()):
        raise ValueError("posterior member is outside the all-train run")

    parameter_mean = final_method.get("parameter_mean_checkpoint")
    if operator == "parameter_mean":
        parameter_mean = _mapping(parameter_mean, label="parameter-mean checkpoint")
    elif parameter_mean is not None:
        raise ValueError("predictive method unexpectedly has a parameter mean")

    calibration = _mapping(
        final_method.get("variance_calibration"), label="variance calibration"
    )
    scale = float(str(calibration.get("scale", math.nan)))
    floor = float(str(calibration.get("floor_m2", math.nan)))
    coverage = float(str(calibration.get("nominal_coordinate_coverage", math.nan)))
    if (
        not math.isfinite(scale)
        or scale < 1.0
        or not math.isfinite(floor)
        or floor <= 0.0
        or not math.isfinite(coverage)
        or not 0.0 < coverage < 1.0
        or scale
        != float(str(method_spec.get("validation_fitted_variance_scale", math.nan)))
        or floor != float(str(method_spec.get("variance_floor_m2", math.nan)))
        or coverage
        != float(str(method_spec.get("nominal_coordinate_coverage", math.nan)))
    ):
        raise ValueError("all-train uncertainty calibration differs")

    runtime = _mapping(alltrain_result.get("runtime"), label="all-train runtime")
    return {
        "operator": operator,
        "weights": final_weights,
        "comparison_baseline_checkpoint": dict(baseline),
        "member_checkpoints": {
            update: dict(value) for update, value in members.items()
        },
        "parameter_mean_checkpoint": (
            dict(parameter_mean) if isinstance(parameter_mean, Mapping) else None
        ),
        "variance_scale": scale,
        "variance_floor_m2": floor,
        "nominal_coordinate_coverage": coverage,
        "runtime": dict(runtime),
        "lineage": {
            "alltrain_protocol_sha256": alltrain_protocol_sha256,
            "alltrain_result_sha256": alltrain_result_sha256,
            "final_method_sha256": final_method_sha256,
            "method_spec_sha256": method_spec_sha256,
        },
    }


def load_deform_dlo2_deep_official_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the immutable one-shot two-seed DLO2 evaluation protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO2_DEEP_OFFICIAL_SCHEMA_VERSION:
        raise ValueError("unsupported DLO2 deep official-evaluation schema")
    if payload.get("contract") != DEFORM_DLO2_DEEP_OFFICIAL_CONTRACT:
        raise ValueError("unsupported DLO2 deep official-evaluation contract")
    if payload.get("model_initialization") != "official-deform-dlo-initialization-v1":
        raise ValueError("DLO2 deep official initialization differs")
    parent = _mapping(
        payload.get("parent_alltrain_protocol"), label="parent_alltrain_protocol"
    )
    required = _mapping(payload.get("required_parent"), label="required_parent")
    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    methods = _mapping(payload.get("methods"), label="methods")
    gate = _mapping(payload.get("claim_gate"), label="claim_gate")
    uncertainty = _mapping(payload.get("uncertainty"), label="uncertainty")
    if (
        not str(parent.get("repository_path", ""))
        or len(str(parent.get("sha256", ""))) != 64
        or required.get("result_contract")
        != "deform-dlo2-deep-alltrain-result-v1"
        or required.get("official_eval_read") is not False
        or required.get("official_eval_execution_authorized") is not True
        or required.get("final_method_contract")
        != "deform-dlo2-deep-alltrain-final-method-v1"
    ):
        raise ValueError("DLO2 deep official parent gate differs")
    reference = float(str(evaluation.get("published_reference_l1_m", math.nan)))
    reference_operator = _mapping(
        evaluation.get("published_reference_operator"),
        label="published_reference_operator",
    )
    reference_draw = _reference_draw_indices(
        reference_operator.get("canonical_eval_indices"),
        expected_case_count=14,
    )
    if (
        evaluation.get("dlo_type") != "DLO2"
        or evaluation.get("partition") != "eval"
        or _positive_int(
            evaluation.get("expected_trajectory_count"),
            label="expected_trajectory_count",
        )
        != 14
        or _positive_int(
            evaluation.get("expected_frame_count"), label="expected_frame_count"
        )
        != 500
        or _positive_int(
            evaluation.get("expected_node_count"), label="expected_node_count"
        )
        != 12
        or evaluation.get("trajectory_policy")
        != "all-eval-files-sorted-once-plus-canonical-reference-draw-v2"
        or evaluation.get("failure_policy") != "seal-failure-no-retry-v1"
        or evaluation.get("metric") != "mean-coordinate-l1-m"
        or evaluation.get("horizon_breakdown") != "equal-frame-thirds-v1"
        or not math.isfinite(reference)
        or reference != 0.0097
        or reference_operator.get("loader")
        != "upstream-random-choices-with-replacement-v1"
        or reference_operator.get("upstream_train_script_sha256")
        != DEFORM_UPSTREAM_TRAIN_SCRIPT_SHA256
        or int(str(reference_operator.get("python_random_seed", -1))) != 0
        or int(str(reference_operator.get("preceding_train_population", -1))) != 56
        or int(str(reference_operator.get("preceding_train_draw_count", -1))) != 56
        or int(str(reference_operator.get("eval_population", -1))) != 14
        or int(str(reference_operator.get("eval_draw_count", -1))) != 14
        or reference_operator.get("canonical_filename_order") != "sorted-by-name-v1"
        or int(str(reference_operator.get("canonical_unique_index_count", -1)))
        != len(set(reference_draw))
        or reference_operator.get("upstream_glob_order") != "unspecified"
    ):
        raise ValueError("DLO2 deep official evaluation contract differs")
    if (
        methods.get("candidate")
        != "preselected-alltrain-two-seed-predictive-mean"
        or methods.get("comparison_baseline")
        != "preselected-lower-validation-seed"
        or methods.get("action_aware_persistence") is not True
        or methods.get("target_selection") is not False
        or methods.get("target_calibration") is not False
        or methods.get("target_retries") is not False
        or methods.get("case_replacement") is not False
    ):
        raise ValueError("DLO2 deep official method policy differs")
    relative_improvement = float(
        str(gate.get("ensemble_relative_improvement_min", math.nan))
    )
    if (
        gate.get("published_reference_all_unique_strictly_better") is not True
        or gate.get("published_reference_canonical_draw_strictly_better") is not True
        or relative_improvement != 0.01
        or _positive_int(
            gate.get("ensemble_minimum_case_wins"),
            label="ensemble_minimum_case_wins",
        )
        != 8
        or gate.get("require_all_expected_cases") is not True
        or uncertainty.get("use_source_validation_scale_unchanged") is not True
        or uncertainty.get("report_coordinate_marginal_coverage") is not True
        or uncertainty.get("report_interval_width") is not True
        or uncertainty.get("report_gaussian_nll") is not True
        or uncertainty.get("report_coordinate_nees") is not True
        or uncertainty.get("report_horizon_breakdown") is not True
    ):
        raise ValueError("DLO2 deep official claim or uncertainty gate differs")
    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def validate_deform_dlo2_deep_official_authorization(
    protocol: Mapping[str, object],
    alltrain_protocol: Mapping[str, object],
    alltrain_result: Mapping[str, object],
    final_method: Mapping[str, object],
    *,
    alltrain_protocol_sha256: str,
    alltrain_result_sha256: str,
    final_method_sha256: str,
) -> dict[str, object]:
    """Return the exact two-seed method after every pre-target check passes."""

    parent = _mapping(
        protocol.get("parent_alltrain_protocol"), label="parent_alltrain_protocol"
    )
    required = _mapping(protocol.get("required_parent"), label="required_parent")
    if (
        parent.get("sha256") != alltrain_protocol_sha256
        or alltrain_protocol.get("contract")
        != "deform-dlo2-deep-alltrain-refit-v1"
        or alltrain_result.get("contract") != required.get("result_contract")
        or alltrain_result.get("official_eval_read")
        is not required.get("official_eval_read")
        or alltrain_result.get("official_eval_execution_authorized")
        is not required.get("official_eval_execution_authorized")
        or len(alltrain_result_sha256) != 64
    ):
        raise ValueError("deep all-train result did not authorize official evaluation")
    result_protocol = _mapping(
        alltrain_result.get("protocol"), label="deep all-train result protocol"
    )
    result_final = _mapping(
        alltrain_result.get("final_method"), label="deep final-method identity"
    )
    if (
        result_protocol.get("sha256") != alltrain_protocol_sha256
        or result_final.get("sha256") != final_method_sha256
        or final_method.get("contract") != required.get("final_method_contract")
        or final_method.get("official_eval_read") is not False
        or final_method.get("operator") != "predictive_mean"
    ):
        raise ValueError("deep all-train final method lineage differs")
    weights = _weights(final_method.get("seed_weights"), label="seed weights")
    updates_raw = _mapping(final_method.get("member_updates"), label="member updates")
    updates = {int(str(seed)): int(str(update)) for seed, update in updates_raw.items()}
    members_raw = _mapping(
        final_method.get("member_checkpoints"), label="member checkpoints"
    )
    members = {
        int(str(seed)): _mapping(identity, label="member checkpoint")
        for seed, identity in members_raw.items()
    }
    baseline_seed = int(str(final_method.get("comparison_baseline_seed", -1)))
    baseline = _mapping(
        final_method.get("comparison_baseline_checkpoint"),
        label="comparison baseline checkpoint",
    )
    if (
        set(weights) != {42, 43}
        or set(updates) != {42, 43}
        or set(members) != {42, 43}
        or any(
            int(str(members[seed].get("update", -1))) != updates[seed]
            for seed in (42, 43)
        )
        or baseline_seed not in (42, 43)
        or members[baseline_seed] != baseline
    ):
        raise ValueError("deep all-train member bank differs")
    result_seed_runs = _mapping(
        alltrain_result.get("seed_results"), label="result seed runs"
    )
    method_seed_runs = _mapping(
        final_method.get("seed_results"), label="method seed runs"
    )
    if result_seed_runs != method_seed_runs or set(map(int, result_seed_runs)) != {
        42,
        43,
    }:
        raise ValueError("deep all-train seed lineage differs")
    calibration = _mapping(
        final_method.get("variance_calibration"), label="variance calibration"
    )
    scale = float(str(calibration.get("scale", math.nan)))
    floor = float(str(calibration.get("floor_m2", math.nan)))
    coverage = float(str(calibration.get("nominal_coordinate_coverage", math.nan)))
    if (
        not math.isfinite(scale)
        or scale < 1.0
        or not math.isfinite(floor)
        or floor <= 0.0
        or not math.isfinite(coverage)
        or not 0.0 < coverage < 1.0
    ):
        raise ValueError("deep all-train uncertainty calibration differs")
    selected_method = _mapping(
        alltrain_result.get("selected_method"), label="selected method"
    )
    selected_weights = _weights(
        selected_method.get("seed_weights"), label="selected seed weights"
    )
    selected_updates_raw = _mapping(
        selected_method.get("member_updates"), label="selected member updates"
    )
    selected_updates = {
        int(str(seed)): int(str(update))
        for seed, update in selected_updates_raw.items()
    }
    selected_calibration = _mapping(
        selected_method.get("variance_calibration"),
        label="selected variance calibration",
    )
    if (
        selected_method.get("operator") != "predictive_mean"
        or selected_weights != weights
        or selected_updates != updates
        or int(str(selected_method.get("comparison_baseline_seed", -1)))
        != baseline_seed
        or selected_calibration != calibration
    ):
        raise ValueError("deep all-train member bank differs from frozen selection")
    runtime = _mapping(alltrain_result.get("runtime"), label="deep all-train runtime")
    return {
        "operator": "predictive_mean",
        "weights": weights,
        "member_updates": updates,
        "comparison_baseline_seed": baseline_seed,
        "comparison_baseline_checkpoint": dict(baseline),
        "member_checkpoints": {
            seed: dict(identity) for seed, identity in members.items()
        },
        "variance_scale": scale,
        "variance_floor_m2": floor,
        "nominal_coordinate_coverage": coverage,
        "seed_results": {
            int(str(seed)): dict(_mapping(identity, label="seed result"))
            for seed, identity in result_seed_runs.items()
        },
        "runtime": dict(runtime),
        "lineage": {
            "alltrain_protocol_sha256": alltrain_protocol_sha256,
            "alltrain_result_sha256": alltrain_result_sha256,
            "final_method_sha256": final_method_sha256,
        },
    }


def summarize_deform_dlo2_official_records(
    candidate_records: Sequence[Mapping[str, object]],
    baseline_records: Sequence[Mapping[str, object]],
    *,
    expected_case_count: int,
    published_reference_l1_m: float,
    minimum_relative_improvement: float,
    minimum_case_wins: int,
    canonical_reference_draw_indices: Sequence[int] = DEFORM_CANONICAL_REFERENCE_DRAW,
) -> dict[str, object]:
    """Compute the frozen aggregate, paired comparison, and claim gate."""

    def indexed(
        records: Sequence[Mapping[str, object]], *, label: str
    ) -> dict[str, Mapping[str, object]]:
        result: dict[str, Mapping[str, object]] = {}
        for record in records:
            name = str(record.get("name", ""))
            if not name or name in result:
                raise ValueError(f"{label} official cases are not unique")
            for key in (
                "model_l1_m",
                "persistence_l1_m",
                "early_l1_m",
                "middle_l1_m",
                "late_l1_m",
            ):
                value = float(str(record.get(key, math.nan)))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{label} official metric is invalid")
            result[name] = record
        return result

    candidate = indexed(candidate_records, label="candidate")
    baseline = indexed(baseline_records, label="baseline")
    if (
        len(candidate) != expected_case_count
        or set(candidate) != set(baseline)
        or minimum_case_wins > expected_case_count
    ):
        raise ValueError("official evaluation does not cover the frozen cohort")
    names = tuple(sorted(candidate))
    reference_draw = _reference_draw_indices(
        canonical_reference_draw_indices,
        expected_case_count=expected_case_count,
    )
    for name in names:
        if float(str(candidate[name]["persistence_l1_m"])) != float(
            str(baseline[name]["persistence_l1_m"])
        ):
            raise ValueError("official persistence comparator differs between arms")

    def mean(records: Mapping[str, Mapping[str, object]], key: str) -> float:
        return float(np.mean([float(str(records[name][key])) for name in names]))

    candidate_mean = mean(candidate, "model_l1_m")
    baseline_mean = mean(baseline, "model_l1_m")
    persistence_mean = mean(candidate, "persistence_l1_m")

    def draw_mean(records: Mapping[str, Mapping[str, object]], key: str) -> float:
        return float(
            np.mean(
                [float(str(records[names[index]][key])) for index in reference_draw]
            )
        )

    candidate_reference_mean = draw_mean(candidate, "model_l1_m")
    baseline_reference_mean = draw_mean(baseline, "model_l1_m")
    persistence_reference_mean = draw_mean(candidate, "persistence_l1_m")
    relative_improvement = (
        (baseline_mean - candidate_mean) / baseline_mean if baseline_mean > 0.0 else 0.0
    )
    wins = sum(
        float(str(candidate[name]["model_l1_m"]))
        < float(str(baseline[name]["model_l1_m"]))
        for name in names
    )
    published_all_unique_passed = candidate_mean < published_reference_l1_m
    published_canonical_draw_passed = (
        candidate_reference_mean < published_reference_l1_m
    )
    improvement_passed = relative_improvement >= minimum_relative_improvement
    wins_passed = wins >= minimum_case_wins
    return {
        "case_count": len(names),
        "candidate_mean_l1_m": candidate_mean,
        "comparison_baseline_mean_l1_m": baseline_mean,
        "action_aware_persistence_mean_l1_m": persistence_mean,
        "candidate_horizon_l1_m": {
            "early": mean(candidate, "early_l1_m"),
            "middle": mean(candidate, "middle_l1_m"),
            "late": mean(candidate, "late_l1_m"),
        },
        "comparison_baseline_horizon_l1_m": {
            "early": mean(baseline, "early_l1_m"),
            "middle": mean(baseline, "middle_l1_m"),
            "late": mean(baseline, "late_l1_m"),
        },
        "bayesian_relative_improvement": relative_improvement,
        "bayesian_case_wins": wins,
        "published_reference_l1_m": float(published_reference_l1_m),
        "published_reference_compatibility": {
            "operator": "canonical-with-replacement-draw-v2",
            "canonical_eval_indices": list(reference_draw),
            "canonical_unique_index_count": len(set(reference_draw)),
            "candidate_mean_l1_m": candidate_reference_mean,
            "comparison_baseline_mean_l1_m": baseline_reference_mean,
            "action_aware_persistence_mean_l1_m": persistence_reference_mean,
            "upstream_glob_order": "unspecified",
        },
        "claim_gate": {
            "all_expected_cases_present": True,
            "published_reference_all_unique_strictly_better": (
                published_all_unique_passed
            ),
            "published_reference_canonical_draw_strictly_better": (
                published_canonical_draw_passed
            ),
            "bayesian_relative_improvement_passed": improvement_passed,
            "bayesian_case_wins_passed": wins_passed,
            "passed": (
                published_all_unique_passed
                and published_canonical_draw_passed
                and improvement_passed
                and wins_passed
            ),
        },
    }


def evaluate_deform_dlo2_official_uncertainty(
    predictions: np.ndarray,
    targets: np.ndarray,
    raw_variance_m2: np.ndarray,
    *,
    variance_floor_m2: float,
    variance_scale: float,
    nominal_coverage: float,
) -> dict[str, object]:
    """Evaluate fixed source-calibrated uncertainty overall and by horizon."""

    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    variance = np.asarray(raw_variance_m2, dtype=np.float64)
    overall = evaluate_deform_coordinate_uncertainty(
        predicted,
        observed,
        variance,
        variance_floor_m2=variance_floor_m2,
        variance_scale=variance_scale,
        nominal_coverage=nominal_coverage,
    )
    effective = variance_scale * np.maximum(variance, variance_floor_m2)
    overall["coordinate_nees"] = float(
        np.mean(np.square(predicted - observed) / effective)
    )
    horizon = predicted.shape[1]
    thirds: dict[str, object] = {}
    for third, label in enumerate(("early", "middle", "late")):
        indices = [
            frame for frame in range(horizon) if min(2, (3 * frame) // horizon) == third
        ]
        values = evaluate_deform_coordinate_uncertainty(
            predicted[:, indices],
            observed[:, indices],
            variance[:, indices],
            variance_floor_m2=variance_floor_m2,
            variance_scale=variance_scale,
            nominal_coverage=nominal_coverage,
        )
        values["coordinate_nees"] = float(
            np.mean(
                np.square(predicted[:, indices] - observed[:, indices])
                / effective[:, indices]
            )
        )
        thirds[label] = values
    return {"overall": overall, "horizon": thirds}
