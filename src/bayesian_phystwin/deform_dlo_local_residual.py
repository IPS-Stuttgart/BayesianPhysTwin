"""Trajectory-grouped Bayesian local residual dynamics for DEFORM."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

DEFORM_LOCAL_RESIDUAL_PROTOCOL_SCHEMA_VERSION = 1
DEFORM_LOCAL_RESIDUAL_PROTOCOL_CONTRACT = "deform-dlo-local-residual-source-v4"


def load_deform_local_residual_protocol(path: str | Path) -> dict[str, object]:
    """Load and validate the DLO1-only local-residual source protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_LOCAL_RESIDUAL_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported DEFORM local-residual protocol schema")
    if payload.get("contract") != DEFORM_LOCAL_RESIDUAL_PROTOCOL_CONTRACT:
        raise ValueError("unsupported DEFORM local-residual protocol contract")
    if (
        payload.get("dlo_type") != "DLO1"
        or payload.get("source_test_status") != "post-open-development-only"
        or payload.get("official_eval_policy") != "forbidden"
        or payload.get("fresh_confirmation_dlo") != "DLO2"
    ):
        raise ValueError("local-residual development must remain DLO1-only")
    boundary = payload.get("information_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("dlo1_source_test_read_only_after_validation_seal") is not True
        or boundary.get("dlo2_training_read") is not False
        or boundary.get("dlo2_source_outcome_read") is not False
        or boundary.get("official_eval_read") is not False
    ):
        raise ValueError("local-residual protocol does not seal future data")
    for key in ("longrun_protocol", "longrun_result", "source_manifest"):
        identity = payload.get(key)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", identity.get("path", "")))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError("local-residual parent identity is invalid")
    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("local-residual protocol omits its baseline")
    for key in ("validation_l1_m", "source_test_l1_m"):
        value = float(cast(Any, baseline.get(key, math.nan)))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("local-residual baseline metric is invalid")
    tolerance = float(cast(Any, baseline.get("reproduction_tolerance_m", math.nan)))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("local-residual baseline tolerance is invalid")
    feature_contract = payload.get("features")
    if (
        not isinstance(feature_contract, Mapping)
        or feature_contract.get("query_evidence")
        != "two-observed-states-known-future-clamped-action-and-baseline-rollout-only"
        or feature_contract.get("frame") != "initial-action-gravity-frame-v1"
        or feature_contract.get("future_free_node_truth") != "fit-and-score-only"
    ):
        raise ValueError("local-residual feature contract is invalid")
    posterior = payload.get("posterior")
    if (
        not isinstance(posterior, Mapping)
        or posterior.get("operator") != "per-node-trajectory-grouped-bayesian-ridge-v1"
        or posterior.get("duplicate_policy") != "collapse-exact-causal-query-clusters"
        or posterior.get("uncertainty")
        != "trajectory-cluster-sandwich-plus-fit-residual-plus-unresolved-mean"
        or posterior.get("clamped_node_policy") != "baseline-exact"
    ):
        raise ValueError("local-residual posterior contract is invalid")
    floor = float(cast(Any, posterior.get("coordinate_variance_floor_m2", math.nan)))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("local-residual variance floor is invalid")
    bank = payload.get("candidate_bank")
    if not isinstance(bank, Mapping):
        raise ValueError("local-residual protocol omits its candidate bank")
    ridges = tuple(float(value) for value in bank.get("ridges", ()))
    shrinkages = tuple(float(value) for value in bank.get("shrinkages", ()))
    if (
        not ridges
        or tuple(sorted(set(ridges))) != ridges
        or any(not math.isfinite(value) or value <= 0.0 for value in ridges)
        or not shrinkages
        or tuple(sorted(set(shrinkages))) != shrinkages
        or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0 for value in shrinkages
        )
        or bank.get("fallback") != "selected-update-6400-byte-exact"
    ):
        raise ValueError("local-residual candidate bank is invalid")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("local-residual protocol omits its gates")
    for stage in ("validation", "source_test"):
        gate = gates.get(stage)
        if not isinstance(gate, Mapping):
            raise ValueError("local-residual gate is invalid")
        improvement = float(
            cast(Any, gate.get("minimum_relative_improvement", math.nan))
        )
        ratio = float(cast(Any, gate.get("maximum_case_ratio", math.nan)))
        wins = int(cast(Any, gate.get("minimum_case_wins", -1)))
        if (
            not math.isfinite(improvement)
            or not 0.0 < improvement < 1.0
            or not math.isfinite(ratio)
            or ratio < 1.0
            or wins < 1
        ):
            raise ValueError("local-residual gate threshold is invalid")
    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def load_deform_dlo2_local_residual_protocol(path: str | Path) -> dict[str, object]:
    """Load the fixed-arm fresh DLO2 transfer protocol."""

    from bayesian_phystwin.deform_dlo_source import load_deform_dlo_source_protocol

    protocol = load_deform_dlo_source_protocol(path)
    if protocol.get("dlo_types") != ("DLO2",):
        raise ValueError("local-residual transfer protocol must remain DLO2-only")
    local = protocol.get("local_residual")
    if not isinstance(local, Mapping):
        raise ValueError("DLO2 protocol omits the local-residual contract")
    for key in ("parent_protocol", "parent_result"):
        identity = local.get(key)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", ""))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError("DLO2 local-residual parent identity is invalid")
    fixed = local.get("fixed_arm")
    if (
        not isinstance(fixed, Mapping)
        or fixed.get("name") != "r1_s0p5"
        or float(cast(Any, fixed.get("ridge", math.nan))) != 1.0
        or float(cast(Any, fixed.get("shrinkage", math.nan))) != 0.5
        or fixed.get("selection_source") != "frozen-dlo1-v4"
        or local.get("query_evidence")
        != "two-observed-states-known-future-clamped-action-and-baseline-rollout-only"
        or local.get("future_free_node_truth") != "fit-and-score-only"
        or local.get("operator") != "per-node-trajectory-grouped-bayesian-ridge-v1"
        or local.get("duplicate_policy") != "collapse-exact-causal-query-clusters"
        or local.get("clamped_node_policy") != "baseline-exact"
        or local.get("fallback") != "selected-dlo2-checkpoint-exact"
    ):
        raise ValueError("DLO2 local-residual fixed arm differs from DLO1")
    floor = float(cast(Any, local.get("coordinate_variance_floor_m2", math.nan)))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("DLO2 local-residual variance floor is invalid")
    for stage in ("validation_gate", "source_transfer_gate"):
        gate = local.get(stage)
        if not isinstance(gate, Mapping):
            raise ValueError("DLO2 local-residual gate is invalid")
        improvement = float(
            cast(Any, gate.get("minimum_relative_improvement", math.nan))
        )
        ratio = float(cast(Any, gate.get("maximum_case_ratio", math.nan)))
        wins = int(cast(Any, gate.get("minimum_case_wins", -1)))
        if (
            not math.isfinite(improvement)
            or not 0.0 < improvement < 1.0
            or not math.isfinite(ratio)
            or ratio < 1.0
            or wins < 1
        ):
            raise ValueError("DLO2 local-residual gate threshold is invalid")
    source_gate = cast(Mapping[str, object], local["source_transfer_gate"])
    maximum = float(cast(Any, source_gate.get("maximum_candidate_l1_m", math.nan)))
    if not math.isfinite(maximum) or maximum != 0.0097:
        raise ValueError("DLO2 local-residual published-reference gate changed")
    authorization = protocol.get("authorization")
    parent_result = cast(Mapping[str, object], local["parent_result"])
    parent_protocol = cast(Mapping[str, object], local["parent_protocol"])
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("required_parent_contract")
        != "deform-dlo-local-residual-result-v4"
        or authorization.get("required_parent_result_sha256")
        != parent_result.get("sha256")
        or authorization.get("required_parent_protocol_sha256")
        != parent_protocol.get("sha256")
        or authorization.get("required_parent_selected_arm") != fixed.get("name")
        or authorization.get("required_parent_source_gate_passed") is not True
        or authorization.get("required_parent_fresh_dlo2_local_residual_authorized")
        is not True
    ):
        raise ValueError("DLO2 local-residual authorization contract is invalid")
    return protocol


def load_deform_dlo2_local_residual_v6_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the validation-selected DLO2 source-transfer protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DLO2 local-residual v6 schema")
    if payload.get("contract") != "deform-dlo2-local-residual-source-v6":
        raise ValueError("unsupported DLO2 local-residual v6 contract")
    upstream = payload.get("upstream")
    boundary = payload.get("information_boundary")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("repository") != "https://github.com/roahmlab/DEFORM"
        or len(str(upstream.get("commit", ""))) != 40
        or not isinstance(boundary, Mapping)
        or boundary.get("development_partition") != "DLO2/train"
        or boundary.get("source_read_only_after_opening_seal") is not True
        or boundary.get("official_eval_read") is not False
        or boundary.get("prob4d_used") is not False
    ):
        raise ValueError("DLO2 local-residual v6 information boundary is invalid")
    for key in (
        "parent_protocol",
        "parent_result",
        "development_selection",
        "training_result",
        "source_manifest",
        "selected_checkpoint",
    ):
        identity = payload.get(key)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", identity.get("path", "")))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError(f"DLO2 local-residual v6 {key} identity is invalid")
    local = payload.get("local_residual")
    fixed = local.get("fixed_arm") if isinstance(local, Mapping) else None
    if (
        not isinstance(local, Mapping)
        or not isinstance(fixed, Mapping)
        or fixed.get("name") != "r1_s0p25"
        or float(cast(Any, fixed.get("ridge", math.nan))) != 1.0
        or float(cast(Any, fixed.get("shrinkage", math.nan))) != 0.25
        or fixed.get("selection_source") != "dlo2-fit-validation-v6"
        or local.get("query_evidence")
        != "two-observed-states-known-future-clamped-action-and-baseline-rollout-only"
        or local.get("future_free_node_truth") != "fit-and-score-only"
        or local.get("operator") != "per-node-trajectory-grouped-bayesian-ridge-v1"
        or local.get("duplicate_policy") != "collapse-exact-causal-query-clusters"
        or local.get("clamped_node_policy") != "baseline-exact"
        or local.get("fallback") != "selected-dlo2-checkpoint-exact"
    ):
        raise ValueError("DLO2 local-residual v6 fixed arm is invalid")
    floor = float(cast(Any, local.get("coordinate_variance_floor_m2", math.nan)))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("DLO2 local-residual v6 variance floor is invalid")
    gate = payload.get("source_transfer_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("DLO2 local-residual v6 source gate is missing")
    improvement = float(cast(Any, gate.get("minimum_relative_improvement", math.nan)))
    wins = int(cast(Any, gate.get("minimum_case_wins", -1)))
    ratio = float(cast(Any, gate.get("maximum_case_ratio", math.nan)))
    maximum = float(cast(Any, gate.get("maximum_candidate_l1_m", math.nan)))
    if (
        not math.isfinite(improvement)
        or improvement != 0.01
        or wins != 6
        or not math.isfinite(ratio)
        or ratio != 1.10
        or not math.isfinite(maximum)
        or maximum != 0.0097
    ):
        raise ValueError("DLO2 local-residual v6 source gate changed")
    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def validate_deform_dlo2_local_residual_v6_parents(
    protocol: Mapping[str, object],
    parent_result: Mapping[str, object],
    development_selection: Mapping[str, object],
) -> dict[str, object]:
    """Verify that v5 stayed closed and v6 selected only on validation."""

    parent_protocol = cast(Mapping[str, object], protocol["parent_protocol"])
    parent_identity = cast(Mapping[str, object], protocol["parent_result"])
    selection_identity = cast(Mapping[str, object], protocol["development_selection"])
    training_identity = cast(Mapping[str, object], protocol["training_result"])
    checkpoint_identity = cast(Mapping[str, object], protocol["selected_checkpoint"])
    parent_validation = parent_result.get("validation_gate")
    parent_source = parent_result.get("source_gate")
    selected = development_selection.get("selected_arm")
    selected_summary = (
        selected.get("summary") if isinstance(selected, Mapping) else None
    )
    if (
        parent_result.get("contract") != "deform-dlo2-local-residual-result-v5"
        or parent_result.get("protocol_sha256") != parent_protocol.get("sha256")
        or parent_result.get("source_test_opened") is not False
        or parent_result.get("official_eval_read") is not False
        or parent_result.get("fallback_used") is not True
        or parent_result.get("alltrain_and_official_evaluation_authorized") is not False
        or not isinstance(parent_validation, Mapping)
        or parent_validation.get("passed") is not False
        or not isinstance(parent_source, Mapping)
        or parent_source.get("reason") != "validation-gate-failed"
        or development_selection.get("contract")
        != "deform-dlo2-local-residual-development-v6"
        or development_selection.get("protocol_sha256") != parent_protocol.get("sha256")
        or development_selection.get("training_result_sha256")
        != training_identity.get("sha256")
        or development_selection.get("selected_checkpoint_sha256")
        != checkpoint_identity.get("sha256")
        or development_selection.get("source_test_opened") is not False
        or development_selection.get("official_eval_read") is not False
        or not isinstance(selected, Mapping)
        or selected.get("name") != "shrinkage-0.25"
        or float(cast(Any, selected.get("shrinkage", math.nan))) != 0.25
        or not isinstance(selected_summary, Mapping)
        or float(cast(Any, selected_summary.get("relative_improvement", -math.inf)))
        < 0.01
        or int(cast(Any, selected_summary.get("wins", -1))) < 6
        or float(cast(Any, selected_summary.get("maximum_case_ratio", math.inf))) > 1.05
    ):
        raise ValueError("DLO2 local-residual v6 parents do not authorize source")
    return {
        "parent_result_contract": str(parent_result["contract"]),
        "parent_result_sha256": str(parent_identity["sha256"]),
        "development_selection_contract": str(development_selection["contract"]),
        "development_selection_sha256": str(selection_identity["sha256"]),
        "selected_arm": "r1_s0p25",
        "source_test_opened": False,
        "official_eval_read": False,
    }


def load_deform_dlo2_local_residual_alltrain_v7_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the target-blind all-56 refit of the source-confirmed method."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DLO2 local-residual all-train v7 schema")
    if payload.get("contract") != "deform-dlo2-local-residual-alltrain-v7":
        raise ValueError("unsupported DLO2 local-residual all-train v7 contract")
    if payload.get("prob4d_used") is not False:
        raise ValueError("DLO2 local-residual all-train must keep Prob4D unused")
    upstream = payload.get("upstream")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("repository") != "https://github.com/roahmlab/DEFORM"
        or len(str(upstream.get("commit", ""))) != 40
    ):
        raise ValueError("DLO2 local-residual all-train upstream is invalid")
    for key in ("parent_source_protocol", "parent_source_result", "source_manifest"):
        identity = payload.get(key)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", identity.get("path", "")))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError(f"DLO2 local-residual all-train {key} is invalid")
    data = payload.get("data")
    if (
        not isinstance(data, Mapping)
        or data.get("dlo_type") != "DLO2"
        or data.get("partition") != "train"
        or int(cast(Any, data.get("trajectory_count", -1))) != 56
        or int(cast(Any, data.get("frame_count", -1))) != 500
        or int(cast(Any, data.get("node_count", -1))) != 12
        or data.get("use_every_train_trajectory") is not True
        or data.get("official_eval_read") is not False
    ):
        raise ValueError("DLO2 local-residual all-train data contract differs")
    training = payload.get("training")
    checkpoints = (
        tuple(int(value) for value in training.get("checkpoint_updates", ()))
        if isinstance(training, Mapping)
        else ()
    )
    if (
        not isinstance(training, Mapping)
        or int(cast(Any, training.get("random_seed", -1))) != 42
        or int(cast(Any, training.get("unroll_horizon_frames", -1))) != 50
        or int(cast(Any, training.get("batch_size", -1))) != 32
        or int(cast(Any, training.get("total_updates", -1))) != 6400
        or checkpoints != (0, 280, 640, 1280, 2560, 4000, 5200, 6040, 6400)
        or training.get("optimizer") != "official-sgd-parameter-groups-v1"
        or training.get("cublas_workspace_config") != ":4096:8"
        or training.get("known_action_nodes") != [0, 1, -2, -1]
        or training.get("window_sampling") != "frozen-uniform-all-56-train-v1"
        or int(cast(Any, training.get("physical_checkpoint_update", -1))) != 6400
    ):
        raise ValueError("DLO2 local-residual all-train training contract differs")
    local = payload.get("local_residual")
    fixed = local.get("fixed_arm") if isinstance(local, Mapping) else None
    if (
        not isinstance(local, Mapping)
        or not isinstance(fixed, Mapping)
        or fixed.get("name") != "r1_s0p25"
        or float(cast(Any, fixed.get("ridge", math.nan))) != 1.0
        or float(cast(Any, fixed.get("shrinkage", math.nan))) != 0.25
        or fixed.get("selection_source") != "passed-dlo2-source-v6"
        or local.get("query_evidence")
        != "two-observed-states-known-future-clamped-action-and-baseline-rollout-only"
        or local.get("future_free_node_truth") != "fit-and-score-only"
        or local.get("operator") != "per-node-trajectory-grouped-bayesian-ridge-v1"
        or local.get("duplicate_policy") != "collapse-exact-causal-query-clusters"
        or local.get("clamped_node_policy") != "baseline-exact"
        or local.get("fit_trajectory_policy") != "all-56-official-train"
        or local.get("validation_reselection") is not False
        or local.get("source_reselection") is not False
        or local.get("target_reselection") is not False
        or local.get("fallback") != "alltrain-physical-checkpoint-exact"
    ):
        raise ValueError("DLO2 local-residual all-train method contract differs")
    floor = float(cast(Any, local.get("coordinate_variance_floor_m2", math.nan)))
    output = payload.get("output")
    if (
        not math.isfinite(floor)
        or floor != 1e-6
        or not isinstance(output, Mapping)
        or output.get("official_eval_authorized_by_refit") is not True
        or output.get("official_eval_read") is not False
        or output.get("preserve_physical_checkpoint") is not True
        or output.get("preserve_local_residual_model") is not True
    ):
        raise ValueError("DLO2 local-residual all-train output contract differs")
    result = dict(payload)
    result["checkpoint_updates"] = checkpoints
    result["protocol_path"] = str(source)
    return result


def validate_deform_dlo2_local_residual_alltrain_v7_authorization(
    protocol: Mapping[str, object],
    source_protocol: Mapping[str, object],
    source_result: Mapping[str, object],
    *,
    source_protocol_sha256: str,
    source_result_sha256: str,
) -> dict[str, object]:
    """Require the passed untouched DLO2 source gate before the all-56 refit."""

    protocol_identity = protocol.get("parent_source_protocol")
    result_identity = protocol.get("parent_source_result")
    local = protocol.get("local_residual")
    source_local = source_protocol.get("local_residual")
    gate = source_result.get("source_gate")
    fixed = source_result.get("fixed_arm")
    if (
        not isinstance(protocol_identity, Mapping)
        or not isinstance(result_identity, Mapping)
        or protocol_identity.get("sha256") != source_protocol_sha256
        or result_identity.get("sha256") != source_result_sha256
        or source_protocol.get("contract") != "deform-dlo2-local-residual-source-v6"
        or source_result.get("contract") != "deform-dlo2-local-residual-result-v6"
        or source_result.get("protocol_sha256") != source_protocol_sha256
        or source_result.get("source_test_opened") is not True
        or source_result.get("official_eval_read") is not False
        or source_result.get("fallback_used") is not False
        or source_result.get("fallback_byte_exact") is not True
        or source_result.get("alltrain_and_official_evaluation_authorized") is not True
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or float(cast(Any, gate.get("relative_improvement", -math.inf))) < 0.01
        or int(cast(Any, gate.get("wins", -1))) < 6
        or float(cast(Any, gate.get("maximum_case_ratio", math.inf))) > 1.10
        or float(cast(Any, gate.get("candidate_mean_l1_m", math.inf))) > 0.0097
        or not isinstance(local, Mapping)
        or not isinstance(source_local, Mapping)
        or not isinstance(fixed, Mapping)
        or fixed.get("name") != "r1_s0p25"
        or float(cast(Any, fixed.get("ridge", math.nan))) != 1.0
        or float(cast(Any, fixed.get("shrinkage", math.nan))) != 0.25
        or source_result.get("fitted_model_sha256")
        != source_local.get("expected_fitted_model_sha256")
    ):
        raise ValueError("DLO2 v6 source result does not authorize all-train refit")
    return {
        "source_protocol_sha256": source_protocol_sha256,
        "source_result_sha256": source_result_sha256,
        "source_gate_passed": True,
        "source_test_opened": True,
        "official_eval_read": False,
        "selected_arm": "r1_s0p25",
        "shrinkage": 0.25,
    }


def load_deform_dlo2_local_residual_official_v7_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the exactly-once official evaluation of the frozen all-56 method."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DLO2 local-residual official v7 schema")
    if payload.get("contract") != "deform-dlo2-local-residual-official-v7":
        raise ValueError("unsupported DLO2 local-residual official v7 contract")
    if payload.get("prob4d_used") is not False:
        raise ValueError("DLO2 local-residual official eval must keep Prob4D unused")
    parent = payload.get("parent_alltrain_protocol")
    required = payload.get("required_parent")
    evaluation = payload.get("evaluation")
    methods = payload.get("methods")
    gate = payload.get("claim_gate")
    uncertainty = payload.get("uncertainty")
    if not all(
        isinstance(value, Mapping)
        for value in (parent, required, evaluation, methods, gate, uncertainty)
    ):
        raise ValueError("DLO2 local-residual official protocol is incomplete")
    if (
        not str(parent.get("repository_path", ""))
        or len(str(parent.get("sha256", ""))) != 64
        or required.get("result_contract")
        != "deform-dlo2-local-residual-alltrain-result-v7"
        or required.get("final_method_contract")
        != "deform-dlo2-local-residual-final-method-v7"
        or required.get("official_eval_read") is not False
        or required.get("official_eval_execution_authorized") is not True
    ):
        raise ValueError("DLO2 local-residual official parent contract differs")
    reference = evaluation.get("published_reference_operator")
    expected_draw = [1, 7, 9, 7, 11, 7, 13, 8, 8, 6, 8, 5, 8, 4]
    if (
        evaluation.get("dlo_type") != "DLO2"
        or evaluation.get("partition") != "eval"
        or int(cast(Any, evaluation.get("expected_trajectory_count", -1))) != 14
        or int(cast(Any, evaluation.get("expected_frame_count", -1))) != 500
        or int(cast(Any, evaluation.get("expected_node_count", -1))) != 12
        or evaluation.get("trajectory_policy")
        != "all-eval-files-sorted-once-plus-canonical-reference-draw-v2"
        or evaluation.get("failure_policy") != "seal-failure-no-retry-v1"
        or evaluation.get("metric") != "mean-coordinate-l1-m"
        or float(cast(Any, evaluation.get("published_reference_l1_m", math.nan)))
        != 0.0097
        or not isinstance(reference, Mapping)
        or reference.get("canonical_eval_indices") != expected_draw
        or int(cast(Any, reference.get("canonical_unique_index_count", -1))) != 9
    ):
        raise ValueError("DLO2 local-residual official evaluation contract differs")
    if (
        methods.get("candidate") != "alltrain-physical-plus-r1-s0p25-local-residual"
        or methods.get("comparison_baseline")
        != "identically-trained-alltrain-physical-checkpoint"
        or methods.get("action_aware_persistence") is not True
        or methods.get("target_selection") is not False
        or methods.get("target_calibration") is not False
        or methods.get("target_retries") is not False
        or methods.get("case_replacement") is not False
    ):
        raise ValueError("DLO2 local-residual official method policy differs")
    if (
        gate.get("published_reference_all_unique_strictly_better") is not True
        or gate.get("published_reference_canonical_draw_strictly_better") is not True
        or float(cast(Any, gate.get("candidate_relative_improvement_min", math.nan)))
        != 0.01
        or int(cast(Any, gate.get("candidate_minimum_case_wins", -1))) != 8
        or float(cast(Any, gate.get("maximum_case_ratio", math.nan))) != 1.10
        or gate.get("require_all_expected_cases") is not True
    ):
        raise ValueError("DLO2 local-residual official claim gate differs")
    if (
        uncertainty.get("use_alltrain_fit_covariance_unchanged") is not True
        or float(cast(Any, uncertainty.get("variance_scale", math.nan))) != 1.0
        or float(cast(Any, uncertainty.get("variance_floor_m2", math.nan))) != 1e-6
        or float(cast(Any, uncertainty.get("nominal_coordinate_coverage", math.nan)))
        != 0.90
    ):
        raise ValueError("DLO2 local-residual official uncertainty differs")
    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def validate_deform_dlo2_local_residual_official_v7_authorization(
    protocol: Mapping[str, object],
    alltrain_protocol: Mapping[str, object],
    alltrain_result: Mapping[str, object],
    final_method: Mapping[str, object],
    *,
    alltrain_protocol_sha256: str,
    alltrain_result_sha256: str,
    final_method_sha256: str,
) -> dict[str, object]:
    """Verify the immutable all-56 method before target enumeration."""

    parent = protocol.get("parent_alltrain_protocol")
    required = protocol.get("required_parent")
    result_protocol = alltrain_result.get("protocol")
    final_identity = alltrain_result.get("final_method")
    checkpoint = final_method.get("physical_checkpoint")
    model = final_method.get("local_residual_model")
    fixed = final_method.get("fixed_arm")
    if (
        not isinstance(parent, Mapping)
        or parent.get("sha256") != alltrain_protocol_sha256
        or alltrain_protocol.get("contract") != "deform-dlo2-local-residual-alltrain-v7"
        or not isinstance(required, Mapping)
        or alltrain_result.get("contract") != required.get("result_contract")
        or alltrain_result.get("official_eval_read")
        is not required.get("official_eval_read")
        or alltrain_result.get("official_eval_execution_authorized")
        is not required.get("official_eval_execution_authorized")
        or not isinstance(result_protocol, Mapping)
        or result_protocol.get("sha256") != alltrain_protocol_sha256
        or not isinstance(final_identity, Mapping)
        or final_identity.get("sha256") != final_method_sha256
        or final_method.get("contract") != required.get("final_method_contract")
        or final_method.get("official_eval_read") is not False
        or not isinstance(checkpoint, Mapping)
        or int(cast(Any, checkpoint.get("update", -1))) != 6400
        or len(str(checkpoint.get("sha256", ""))) != 64
        or not isinstance(model, Mapping)
        or len(str(model.get("sha256", ""))) != 64
        or not isinstance(fixed, Mapping)
        or fixed.get("name") != "r1_s0p25"
        or float(cast(Any, fixed.get("ridge", math.nan))) != 1.0
        or float(cast(Any, fixed.get("shrinkage", math.nan))) != 0.25
        or final_method.get("validation_reselection") is not False
        or final_method.get("source_reselection") is not False
        or final_method.get("target_reselection") is not False
    ):
        raise ValueError(
            "all-train local-residual method does not authorize official eval"
        )
    runtime = alltrain_result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("all-train local-residual runtime is missing")
    return {
        "alltrain_protocol_sha256": alltrain_protocol_sha256,
        "alltrain_result_sha256": alltrain_result_sha256,
        "final_method_sha256": final_method_sha256,
        "physical_checkpoint": dict(checkpoint),
        "local_residual_model": dict(model),
        "fixed_arm": dict(fixed),
        "runtime": dict(runtime),
        "official_eval_read": False,
    }


def validate_deform_dlo2_local_residual_parent(
    protocol: Mapping[str, object],
    parent: Mapping[str, object],
) -> dict[str, object]:
    """Verify that the frozen DLO1 v4 result authorizes DLO2 access."""

    authorization = protocol.get("authorization")
    local = protocol.get("local_residual")
    source_gate = parent.get("source_gate")
    selected_spec = parent.get("selected_spec")
    if (
        not isinstance(authorization, Mapping)
        or not isinstance(local, Mapping)
        or not isinstance(source_gate, Mapping)
        or not isinstance(selected_spec, Mapping)
        or parent.get("contract") != authorization.get("required_parent_contract")
        or parent.get("protocol_sha256")
        != authorization.get("required_parent_protocol_sha256")
        or parent.get("selected_arm")
        != authorization.get("required_parent_selected_arm")
        or source_gate.get("passed")
        is not authorization.get("required_parent_source_gate_passed")
        or parent.get("fresh_dlo2_local_residual_authorized")
        is not authorization.get("required_parent_fresh_dlo2_local_residual_authorized")
        or parent.get("dlo2_read") is not False
        or parent.get("official_eval_read") is not False
        or float(cast(Any, selected_spec.get("ridge", math.nan))) != 1.0
        or float(cast(Any, selected_spec.get("shrinkage", math.nan))) != 0.5
    ):
        raise ValueError("DLO1 local-residual result did not authorize DLO2")
    return {
        "contract": str(parent["contract"]),
        "protocol_sha256": str(parent["protocol_sha256"]),
        "selected_arm": str(parent["selected_arm"]),
        "selected_spec": {
            "ridge": float(cast(Any, selected_spec["ridge"])),
            "shrinkage": float(cast(Any, selected_spec["shrinkage"])),
        },
        "source_gate_passed": bool(source_gate["passed"]),
        "fresh_dlo2_local_residual_authorized": bool(
            parent["fresh_dlo2_local_residual_authorized"]
        ),
        "dlo2_read": False,
        "official_eval_read": False,
    }


def _finite_array(values: np.ndarray, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {ndim}-D array")
    return array


def _clamped_indices(node_count: int) -> np.ndarray:
    if node_count < 5:
        raise ValueError("DEFORM local residual requires at least five nodes")
    return np.asarray((0, 1, node_count - 2, node_count - 1), dtype=np.int64)


def _unit(vector: np.ndarray, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"cannot construct {label} from degenerate geometry")
    return vector / norm


def _initial_action_frames(initial_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_count = initial_states.shape[2]
    clamped = _clamped_indices(node_count)
    points_at_zero = initial_states[:, 0]
    centers = np.mean(points_at_zero[:, clamped], axis=1)
    frames = []
    for points in points_at_zero:
        left = np.mean(points[clamped[:2]], axis=0)
        right = np.mean(points[clamped[2:]], axis=0)
        x_axis = _unit(right - left, label="primary action frame")
        secondary = 0.5 * (
            (points[clamped[1]] - points[clamped[0]])
            + (points[clamped[3]] - points[clamped[2]])
        )
        secondary = secondary - np.dot(secondary, x_axis) * x_axis
        if np.linalg.norm(secondary) <= 1e-10:
            secondary = np.asarray((0.0, 0.0, 1.0))
            if abs(float(np.dot(secondary, x_axis))) > 1.0 - 1e-8:
                secondary = np.asarray((0.0, 1.0, 0.0))
            secondary = secondary - np.dot(secondary, x_axis) * x_axis
        y_axis = _unit(secondary, label="secondary action frame")
        z_axis = _unit(np.cross(x_axis, y_axis), label="action-frame normal")
        y_axis = np.cross(z_axis, x_axis)
        frames.append(np.stack((x_axis, y_axis, z_axis), axis=1))
    return centers, np.stack(frames)


def deform_causal_inputs(
    trajectories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the only trajectory values admissible to the query predictor."""

    full = _finite_array(trajectories, ndim=4, label="trajectories")
    if full.shape[1] < 3 or full.shape[3] != 3:
        raise ValueError("DEFORM trajectories have an invalid shape")
    clamped = _clamped_indices(full.shape[2])
    return full[:, :2].copy(), full[:, 2:, clamped].copy()


def build_deform_local_residual_features(
    initial_states: np.ndarray,
    clamped_action: np.ndarray,
    baseline_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build gravity-frame local dynamics features without outcome innovations."""

    initial = _finite_array(initial_states, ndim=4, label="initial states")
    action = _finite_array(clamped_action, ndim=4, label="clamped action")
    baseline = _finite_array(
        baseline_predictions,
        ndim=4,
        label="baseline predictions",
    )
    if (
        initial.shape[1] != 2
        or initial.shape[0] != baseline.shape[0]
        or initial.shape[2:] != baseline.shape[2:]
        or action.shape != (baseline.shape[0], baseline.shape[1], 4, 3)
    ):
        raise ValueError("DEFORM local residual inputs do not align")
    node_count = baseline.shape[2]
    internal: np.ndarray = np.arange(2, node_count - 2, dtype=np.int64)
    centers, frames = _initial_action_frames(initial)
    action_centers = np.mean(action, axis=2)
    baseline_relative = baseline - action_centers[:, :, None, :]
    baseline_canonical = np.einsum("ntvi,nij->ntvj", baseline_relative, frames)
    action_relative = action - action_centers[:, :, None, :]
    action_canonical = np.einsum("ntai,nij->ntaj", action_relative, frames)
    initial_relative = initial - centers[:, None, None, :]
    initial_canonical = np.einsum("nsvi,nij->nsvj", initial_relative, frames)

    initial_velocity = initial[:, 1] - initial[:, 0]
    first_velocity = baseline[:, 0] - initial[:, 1]
    baseline_velocity = np.diff(baseline, axis=1, prepend=initial[:, 1:2])
    baseline_velocity[:, 0] = first_velocity
    baseline_velocity = np.einsum("ntvi,nij->ntvj", baseline_velocity, frames)
    baseline_acceleration = np.diff(
        baseline_velocity,
        axis=1,
        prepend=baseline_velocity[:, :1],
    )
    initial_velocity_canonical = np.einsum("nvi,nij->nvj", initial_velocity, frames)

    initial_clamped = initial[:, 1, _clamped_indices(node_count)]
    action_with_initial = np.concatenate((initial_clamped[:, None], action), axis=1)
    action_velocity = np.diff(action_with_initial, axis=1)
    action_velocity = np.einsum("ntai,nij->ntaj", action_velocity, frames)
    action_acceleration = np.diff(
        action_velocity,
        axis=1,
        prepend=action_velocity[:, :1],
    )

    left = baseline_canonical[:, :, internal - 1]
    current = baseline_canonical[:, :, internal]
    right = baseline_canonical[:, :, internal + 1]
    curvature = 0.5 * (left + right) - current
    left_action = np.mean(action_canonical[:, :, :2], axis=2)
    right_action = np.mean(action_canonical[:, :, 2:], axis=2)
    relative_left = current - left_action[:, :, None]
    relative_right = current - right_action[:, :, None]

    batch_size, horizon, internal_count, _ = current.shape
    normalized_time = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    time_features = np.stack(
        (
            normalized_time,
            np.square(normalized_time),
            np.sin(np.pi * normalized_time),
            np.cos(np.pi * normalized_time),
        ),
        axis=1,
    )
    time_features = np.broadcast_to(
        time_features[None, :, None],
        (batch_size, horizon, internal_count, 4),
    )
    arc = np.linspace(-1.0, 1.0, node_count, dtype=np.float64)[internal]
    arc_features = np.stack((arc, np.square(arc)), axis=1)
    arc_features = np.broadcast_to(
        arc_features[None, None],
        (batch_size, horizon, internal_count, 2),
    )
    initial_position = np.broadcast_to(
        initial_canonical[:, None, 1, internal],
        (batch_size, horizon, internal_count, 3),
    )
    initial_velocity_feature = np.broadcast_to(
        initial_velocity_canonical[:, None, internal],
        (batch_size, horizon, internal_count, 3),
    )
    action_flat = np.broadcast_to(
        action_canonical.reshape(batch_size, horizon, 1, 12),
        (batch_size, horizon, internal_count, 12),
    )
    action_velocity_flat = np.broadcast_to(
        action_velocity.reshape(batch_size, horizon, 1, 12),
        (batch_size, horizon, internal_count, 12),
    )
    action_acceleration_flat = np.broadcast_to(
        action_acceleration.reshape(batch_size, horizon, 1, 12),
        (batch_size, horizon, internal_count, 12),
    )
    scalar_norms = np.stack(
        (
            np.linalg.norm(baseline_velocity[:, :, internal], axis=3),
            np.linalg.norm(baseline_acceleration[:, :, internal], axis=3),
            np.linalg.norm(curvature, axis=3),
            np.linalg.norm(relative_left, axis=3),
            np.linalg.norm(relative_right, axis=3),
        ),
        axis=3,
    )
    dynamic_interactions = (
        np.concatenate(
            (
                current,
                baseline_velocity[:, :, internal],
                curvature,
                action_velocity_flat,
            ),
            axis=3,
        )
        * normalized_time[None, :, None, None]
    )
    features = np.concatenate(
        (
            time_features,
            arc_features,
            initial_position,
            initial_velocity_feature,
            current,
            baseline_velocity[:, :, internal],
            baseline_acceleration[:, :, internal],
            curvature,
            action_flat,
            action_velocity_flat,
            action_acceleration_flat,
            relative_left,
            relative_right,
            scalar_norms,
            dynamic_interactions,
        ),
        axis=3,
    )
    return features, frames


def _collapse_duplicate_queries(
    initial: np.ndarray,
    action: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    groups: dict[bytes, list[int]] = {}
    for index in range(initial.shape[0]):
        key = b"".join(
            (
                np.ascontiguousarray(initial[index]).tobytes(),
                np.ascontiguousarray(action[index]).tobytes(),
                np.ascontiguousarray(baseline[index]).tobytes(),
            )
        )
        groups.setdefault(key, []).append(index)
    grouped_initial = []
    grouped_action = []
    grouped_baseline = []
    grouped_targets = []
    grouped_names = []
    for indices in groups.values():
        grouped_initial.append(initial[indices[0]])
        grouped_action.append(action[indices[0]])
        grouped_baseline.append(baseline[indices[0]])
        grouped_targets.append(np.mean(targets[indices], axis=0))
        grouped_names.append(tuple(str(names[index]) for index in indices))
    return (
        np.stack(grouped_initial),
        np.stack(grouped_action),
        np.stack(grouped_baseline),
        np.stack(grouped_targets),
        tuple(grouped_names),
    )


def fit_deform_local_residual(
    initial_states: np.ndarray,
    clamped_action: np.ndarray,
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    names: Sequence[str],
    *,
    ridge: float,
    variance_floor_m2: float,
) -> dict[str, object]:
    """Fit trajectory-clustered Bayesian ridge residual dynamics."""

    initial = _finite_array(initial_states, ndim=4, label="fit initial states")
    action = _finite_array(clamped_action, ndim=4, label="fit clamped action")
    baseline = _finite_array(
        baseline_predictions,
        ndim=4,
        label="fit baseline predictions",
    )
    observed = _finite_array(targets, ndim=4, label="fit targets")
    if (
        baseline.shape != observed.shape
        or initial.shape[0] != baseline.shape[0]
        or action.shape[:2] != baseline.shape[:2]
        or len(names) != baseline.shape[0]
        or len(set(names)) != len(names)
    ):
        raise ValueError("DEFORM local residual fit arrays do not align")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("DEFORM local residual ridge must be positive")
    if not math.isfinite(variance_floor_m2) or variance_floor_m2 <= 0.0:
        raise ValueError("DEFORM local residual variance floor must be positive")
    initial, action, baseline, observed, grouped_names = _collapse_duplicate_queries(
        initial,
        action,
        baseline,
        observed,
        names,
    )
    features, frames = build_deform_local_residual_features(
        initial,
        action,
        baseline,
    )
    residual_global = observed - baseline
    residual_canonical = np.einsum("ntvi,nij->ntvj", residual_global, frames)
    internal: np.ndarray = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    residual_canonical = residual_canonical[:, :, internal]
    trajectory_count, horizon, internal_count, feature_count = features.shape
    feature_location = np.zeros((internal_count, feature_count), dtype=np.float64)
    feature_scale = np.ones_like(feature_location)
    coefficients = np.zeros((internal_count, feature_count + 1, 3), dtype=np.float64)
    coefficient_covariance = np.zeros(
        (internal_count, 3, feature_count + 1, feature_count + 1),
        dtype=np.float64,
    )
    residual_variance = np.zeros((internal_count, 3), dtype=np.float64)
    penalty = np.eye(feature_count + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for node in range(internal_count):
        raw_x = features[:, :, node]
        location = np.mean(raw_x, axis=(0, 1))
        scale = np.std(raw_x, axis=(0, 1))
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (raw_x - location) / scale
        design = np.concatenate(
            (
                np.ones((trajectory_count, horizon, 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        flat_design = design.reshape(-1, feature_count + 1)
        response = residual_canonical[:, :, node].reshape(-1, 3)
        bread = np.linalg.inv(flat_design.T @ flat_design + penalty)
        coefficient = bread @ flat_design.T @ response
        fit_residual = residual_canonical[:, :, node] - np.einsum(
            "ntd,dc->ntc", design, coefficient
        )
        correction = trajectory_count / max(1, trajectory_count - 1)
        for coordinate in range(3):
            scores = np.einsum(
                "ntd,nt->nd",
                design,
                fit_residual[:, :, coordinate],
            )
            meat = scores.T @ scores * correction
            coefficient_covariance[node, coordinate] = bread @ meat @ bread
        feature_location[node] = location
        feature_scale[node] = scale
        coefficients[node] = coefficient
        residual_variance[node] = np.mean(np.square(fit_residual), axis=(0, 1))
    return {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-model-v1",
        "node_count": baseline.shape[2],
        "prediction_horizon": baseline.shape[1],
        "feature_count": feature_count,
        "trajectory_clusters": grouped_names,
        "feature_location": feature_location,
        "feature_scale": feature_scale,
        "coefficients": coefficients,
        "coefficient_covariance": coefficient_covariance,
        "residual_variance": residual_variance,
        "ridge": ridge,
        "variance_floor_m2": variance_floor_m2,
    }


def predict_deform_local_residual(
    model: dict[str, object],
    initial_states: np.ndarray,
    clamped_action: np.ndarray,
    baseline_predictions: np.ndarray,
    *,
    shrinkage: float,
) -> dict[str, np.ndarray]:
    """Predict local residuals from causal state/action features only."""

    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("DEFORM local residual shrinkage is invalid")
    baseline = _finite_array(
        baseline_predictions,
        ndim=4,
        label="query baseline predictions",
    )
    if (
        int(cast(Any, model.get("node_count", -1))) != baseline.shape[2]
        or int(cast(Any, model.get("prediction_horizon", -1))) != baseline.shape[1]
    ):
        raise ValueError("DEFORM local residual model shape differs")
    features, frames = build_deform_local_residual_features(
        initial_states,
        clamped_action,
        baseline,
    )
    location = _finite_array(
        np.asarray(model.get("feature_location")),
        ndim=2,
        label="feature location",
    )
    scale = _finite_array(
        np.asarray(model.get("feature_scale")),
        ndim=2,
        label="feature scale",
    )
    coefficients = _finite_array(
        np.asarray(model.get("coefficients")),
        ndim=3,
        label="local residual coefficients",
    )
    covariance = _finite_array(
        np.asarray(model.get("coefficient_covariance")),
        ndim=4,
        label="local residual coefficient covariance",
    )
    residual_variance = _finite_array(
        np.asarray(model.get("residual_variance")),
        ndim=2,
        label="local residual variance",
    )
    internal_count = baseline.shape[2] - 4
    feature_count = features.shape[3]
    if (
        location.shape != (internal_count, feature_count)
        or scale.shape != location.shape
        or coefficients.shape != (internal_count, feature_count + 1, 3)
        or covariance.shape != (internal_count, 3, feature_count + 1, feature_count + 1)
        or residual_variance.shape != (internal_count, 3)
    ):
        raise ValueError("DEFORM local residual model arrays do not align")
    means = []
    variances = []
    for node in range(internal_count):
        standardized = (features[:, :, node] - location[node]) / scale[node]
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        mean = np.einsum("ntd,dc->ntc", design, coefficients[node])
        epistemic = np.stack(
            [
                np.einsum(
                    "ntd,de,nte->nt",
                    design,
                    covariance[node, coordinate],
                    design,
                )
                for coordinate in range(3)
            ],
            axis=2,
        )
        means.append(mean)
        variances.append(epistemic + residual_variance[node])
    correction_canonical = np.stack(means, axis=2)
    variance_canonical = np.maximum(np.stack(variances, axis=2), 0.0)
    correction_global = np.einsum("ntvj,nij->ntvi", correction_canonical, frames)
    variance_global = np.einsum("ntvj,nij->ntvi", variance_canonical, np.square(frames))
    floor = float(cast(Any, model.get("variance_floor_m2", math.nan)))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("DEFORM local residual model variance floor is invalid")
    unresolved = np.square((1.0 - shrinkage) * correction_global)
    internal: np.ndarray = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    candidate = baseline.copy()
    candidate[:, :, internal] += shrinkage * correction_global
    coordinate_variance = np.zeros_like(baseline)
    coordinate_variance[:, :, internal] = variance_global + unresolved + floor
    return {
        "predictions": candidate,
        "coordinate_variance_m2": coordinate_variance,
        "correction_l2_m": np.sqrt(
            np.mean(np.square(shrinkage * correction_global), axis=(1, 2, 3))
        ),
    }


def serialize_deform_local_residual_model(
    model: dict[str, object],
) -> dict[str, Any]:
    """Return a pickle-free representation for immutable NPZ storage."""

    clusters = model.get("trajectory_clusters")
    if not isinstance(clusters, Sequence) or isinstance(clusters, (str, bytes)):
        raise ValueError("DEFORM local residual clusters are invalid")
    return {
        "schema_version": np.asarray(
            [int(cast(Any, model["schema_version"]))], dtype=np.int64
        ),
        "node_count": np.asarray([int(cast(Any, model["node_count"]))], dtype=np.int64),
        "prediction_horizon": np.asarray(
            [int(cast(Any, model["prediction_horizon"]))], dtype=np.int64
        ),
        "feature_count": np.asarray(
            [int(cast(Any, model["feature_count"]))], dtype=np.int64
        ),
        "feature_location": np.asarray(model["feature_location"]),
        "feature_scale": np.asarray(model["feature_scale"]),
        "coefficients": np.asarray(model["coefficients"]),
        "coefficient_covariance": np.asarray(model["coefficient_covariance"]),
        "residual_variance": np.asarray(model["residual_variance"]),
        "ridge": np.asarray([float(cast(Any, model["ridge"]))], dtype=np.float64),
        "variance_floor_m2": np.asarray(
            [float(cast(Any, model["variance_floor_m2"]))], dtype=np.float64
        ),
        "trajectory_cluster_count": np.asarray([len(clusters)], dtype=np.int64),
        "trajectory_clusters_json": np.asarray(
            [json.dumps([list(group) for group in clusters], separators=(",", ":"))]
        ),
    }


def deserialize_deform_local_residual_model(
    archive: Mapping[str, Any],
) -> dict[str, object]:
    """Reconstruct and validate a local-residual model from a pickle-free NPZ."""

    def scalar(key: str, dtype: type[int] | type[float]) -> int | float:
        value = np.asarray(archive[key])
        if value.shape != (1,):
            raise ValueError(f"DEFORM local residual {key} is not scalar")
        return dtype(value[0])

    schema_version = int(scalar("schema_version", int))
    node_count = int(scalar("node_count", int))
    prediction_horizon = int(scalar("prediction_horizon", int))
    feature_count = int(scalar("feature_count", int))
    ridge = float(scalar("ridge", float))
    variance_floor_m2 = float(scalar("variance_floor_m2", float))
    cluster_count = int(scalar("trajectory_cluster_count", int))
    raw_clusters = np.asarray(archive["trajectory_clusters_json"])
    if raw_clusters.shape != (1,):
        raise ValueError("DEFORM local residual clusters are not scalar JSON")
    try:
        decoded = json.loads(str(raw_clusters[0]))
    except (TypeError, ValueError) as error:
        raise ValueError("DEFORM local residual clusters are invalid JSON") from error
    if (
        schema_version != 1
        or node_count < 5
        or prediction_horizon < 1
        or feature_count < 1
        or not math.isfinite(ridge)
        or ridge <= 0.0
        or not math.isfinite(variance_floor_m2)
        or variance_floor_m2 <= 0.0
        or not isinstance(decoded, list)
        or len(decoded) != cluster_count
        or any(
            not isinstance(group, list)
            or not group
            or not all(isinstance(name, str) and name for name in group)
            for group in decoded
        )
    ):
        raise ValueError("DEFORM local residual serialized metadata is invalid")
    internal_count = node_count - 4
    location = _finite_array(
        np.asarray(archive["feature_location"]), ndim=2, label="feature location"
    ).copy()
    scale = _finite_array(
        np.asarray(archive["feature_scale"]), ndim=2, label="feature scale"
    ).copy()
    coefficients = _finite_array(
        np.asarray(archive["coefficients"]),
        ndim=3,
        label="local residual coefficients",
    ).copy()
    covariance = _finite_array(
        np.asarray(archive["coefficient_covariance"]),
        ndim=4,
        label="local residual coefficient covariance",
    ).copy()
    residual_variance = _finite_array(
        np.asarray(archive["residual_variance"]),
        ndim=2,
        label="local residual variance",
    ).copy()
    if (
        location.shape != (internal_count, feature_count)
        or scale.shape != location.shape
        or np.any(scale <= 0.0)
        or coefficients.shape != (internal_count, feature_count + 1, 3)
        or covariance.shape != (internal_count, 3, feature_count + 1, feature_count + 1)
        or residual_variance.shape != (internal_count, 3)
        or np.any(residual_variance < 0.0)
    ):
        raise ValueError("DEFORM local residual serialized arrays do not align")
    return {
        "schema_version": schema_version,
        "contract": "deform-dlo-local-residual-model-v1",
        "node_count": node_count,
        "prediction_horizon": prediction_horizon,
        "feature_count": feature_count,
        "trajectory_clusters": tuple(tuple(group) for group in decoded),
        "feature_location": location,
        "feature_scale": scale,
        "coefficients": coefficients,
        "coefficient_covariance": covariance,
        "residual_variance": residual_variance,
        "ridge": ridge,
        "variance_floor_m2": variance_floor_m2,
    }
