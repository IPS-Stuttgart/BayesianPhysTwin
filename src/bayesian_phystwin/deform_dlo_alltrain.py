"""Authorization helpers for the final all-training-data DEFORM refit."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFORM_DLO2_ALLTRAIN_SCHEMA_VERSION = 1
DEFORM_DLO2_ALLTRAIN_CONTRACT = "deform-dlo2-alltrain-refit-v1"
DEFORM_DLO2_DEEP_ALLTRAIN_CONTRACT = "deform-dlo2-deep-alltrain-refit-v1"
DEFORM_DLO2_DEEP_ALLTRAIN_RESULT_CONTRACT = (
    "deform-dlo2-deep-alltrain-result-v1"
)


def load_deform_dlo2_alltrain_protocol(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the target-blind all-56 refit protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO2_ALLTRAIN_SCHEMA_VERSION:
        raise ValueError("unsupported DLO2 all-train schema")
    if payload.get("contract") != DEFORM_DLO2_ALLTRAIN_CONTRACT:
        raise ValueError("unsupported DLO2 all-train contract")
    if payload.get("model_initialization") != "official-deform-dlo-initialization-v1":
        raise ValueError("DLO2 all-train initialization contract differs")

    parent = payload.get("parent_source_protocol")
    required = payload.get("required_parent")
    data = payload.get("data")
    training = payload.get("training")
    transfer = payload.get("method_transfer")
    output = payload.get("output")
    if not all(
        isinstance(value, Mapping)
        for value in (parent, required, data, training, transfer, output)
    ):
        raise ValueError("DLO2 all-train protocol is incomplete")
    if (
        not str(parent.get("repository_path", ""))
        or len(str(parent.get("sha256", ""))) != 64
    ):
        raise ValueError("DLO2 all-train parent identity is invalid")
    if (
        required.get("source_result_contract")
        != "deform-dlo-source-reproduction-result-v1"
        or required.get("source_gate_passed") is not True
        or required.get("posterior_result_contract")
        != "deform-dlo2-posterior-result-v1"
        or required.get("posterior_exact_fallback") is not False
        or required.get("posterior_official_eval_authorized") is not True
    ):
        raise ValueError("DLO2 all-train parent gate differs")
    if (
        data.get("dlo_type") != "DLO2"
        or data.get("partition") != "train"
        or int(data.get("trajectory_count", -1)) != 56
        or int(data.get("frame_count", -1)) != 500
        or int(data.get("node_count", -1)) != 12
        or data.get("use_every_train_trajectory") is not True
        or data.get("official_eval_read") is not False
    ):
        raise ValueError("DLO2 all-train data contract differs")
    checkpoints = tuple(int(value) for value in training.get("checkpoint_updates", ()))
    if (
        int(training.get("random_seed", -1)) != 42
        or int(training.get("unroll_horizon_frames", -1)) != 50
        or int(training.get("batch_size", -1)) != 32
        or int(training.get("total_updates", -1)) != 6400
        or not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] != 6400
        or tuple(sorted(set(checkpoints))) != checkpoints
        or training.get("optimizer") != "official-sgd-parameter-groups-v1"
        or training.get("cublas_workspace_config") != ":4096:8"
        or training.get("known_action_nodes") != [0, 1, -2, -1]
    ):
        raise ValueError("DLO2 all-train training contract differs")
    if (
        transfer.get("selection_source") != "fresh-dlo2-posterior-result"
        or transfer.get("operator") != "copy-selected-spec-exactly"
        or transfer.get("checkpoint_weights") != "copy-selected-spec-exactly"
        or transfer.get("validation_reselection") is not False
        or transfer.get("source_reselection") is not False
        or transfer.get("target_reselection") is not False
        or transfer.get("variance_scale") != "copy-validation-fitted-scale-exactly"
        or output.get("official_eval_authorized_by_refit_alone") is not False
    ):
        raise ValueError("DLO2 all-train transfer contract differs")

    result = dict(payload)
    result["checkpoint_updates"] = checkpoints
    result["protocol_path"] = str(source)
    return result


def validate_deform_dlo2_alltrain_authorization(
    protocol: Mapping[str, object],
    source_result: Mapping[str, object],
    posterior_result: Mapping[str, object],
    selection_seal: Mapping[str, object],
    *,
    source_protocol_sha256: str,
    source_result_sha256: str,
) -> dict[str, object]:
    """Return the immutable posterior spec only when every source gate passed."""

    required = protocol.get("required_parent")
    if not isinstance(required, Mapping):
        raise ValueError("DLO2 all-train protocol omits parent gates")
    source_gate = source_result.get("source_gate")
    selected_checkpoint = source_result.get("selected_checkpoint")
    if (
        source_result.get("contract") != required.get("source_result_contract")
        or source_result.get("official_eval_read") is not False
        or source_result.get("advancement_authorized") is not True
        or not isinstance(source_gate, Mapping)
        or source_gate.get("passed") is not required.get("source_gate_passed")
    ):
        raise ValueError("DLO2 source result did not authorize all-train refit")
    if not isinstance(selected_checkpoint, Mapping):
        raise ValueError("DLO2 source result omits its selected checkpoint")
    baseline_update = int(selected_checkpoint.get("update", -1))
    if (
        posterior_result.get("contract") != required.get("posterior_result_contract")
        or posterior_result.get("official_eval_read") is not False
        or posterior_result.get("exact_fallback")
        is not required.get("posterior_exact_fallback")
        or posterior_result.get("identical_information_official_eval_authorized")
        is not required.get("posterior_official_eval_authorized")
    ):
        raise ValueError("DLO2 posterior did not authorize all-train refit")
    if (
        selection_seal.get("contract") != "deform-dlo2-posterior-selection-v1"
        or selection_seal.get("official_eval_read") is not False
    ):
        raise ValueError("DLO2 posterior selection seal differs")
    selected_spec = posterior_result.get("selected_spec")
    selected_arm = str(posterior_result.get("selected_arm", ""))
    posterior_selection = posterior_result.get("selection")
    sealed_selection = selection_seal.get("selection")
    candidate_specs = selection_seal.get("candidate_specs")
    if not isinstance(selected_spec, Mapping):
        raise ValueError("DLO2 posterior omits its selected method spec")
    if (
        not selected_arm
        or not isinstance(posterior_selection, Mapping)
        or not isinstance(sealed_selection, Mapping)
        or posterior_selection != sealed_selection
        or posterior_selection.get("selected_arm") != selected_arm
        or not isinstance(candidate_specs, Mapping)
        or candidate_specs.get(selected_arm) != selected_spec
    ):
        raise ValueError("DLO2 posterior selected method does not match its seal")
    operator = str(selected_spec.get("operator", ""))
    raw_weights = selected_spec.get("weights")
    if operator not in (
        "parameter_mean",
        "predictive_mean",
        "predictive_median",
    ) or not isinstance(raw_weights, Mapping):
        raise ValueError("DLO2 posterior selected method is invalid")
    weights = {int(update): float(weight) for update, weight in raw_weights.items()}
    raw_checkpoint_updates = protocol.get("checkpoint_updates")
    if not isinstance(raw_checkpoint_updates, Sequence) or isinstance(
        raw_checkpoint_updates, (str, bytes)
    ):
        raise ValueError("DLO2 all-train checkpoint schedule is invalid")
    checkpoint_updates = {int(update) for update in raw_checkpoint_updates}
    if (
        not weights
        or baseline_update not in checkpoint_updates
        or not set(weights).issubset(checkpoint_updates)
        or any(
            not math.isfinite(weight) or weight <= 0.0 for weight in weights.values()
        )
        or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-10)
    ):
        raise ValueError("DLO2 posterior checkpoint weights are invalid")
    source_identity = selection_seal.get("source_result")
    protocol_identity = selection_seal.get("protocol")
    if (
        not isinstance(source_identity, Mapping)
        or source_identity.get("sha256") != source_result_sha256
        or not isinstance(protocol_identity, Mapping)
        or protocol_identity.get("sha256") != source_protocol_sha256
    ):
        raise ValueError("DLO2 posterior selection lineage differs")
    uncertainty = posterior_result.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise ValueError("DLO2 posterior omits uncertainty calibration")
    variance_scale = float(
        uncertainty.get("validation_fitted_variance_scale", math.nan)
    )
    variance_floor = float(uncertainty.get("variance_floor_m2", math.nan))
    nominal_coverage = float(uncertainty.get("nominal_coordinate_coverage", math.nan))
    if (
        not math.isfinite(variance_scale)
        or variance_scale < 1.0
        or not math.isfinite(variance_floor)
        or variance_floor <= 0.0
        or not math.isfinite(nominal_coverage)
        or not 0.0 < nominal_coverage < 1.0
    ):
        raise ValueError("DLO2 posterior variance scale is invalid")
    return {
        "operator": operator,
        "weights": weights,
        "comparison_baseline_update": baseline_update,
        "validation_fitted_variance_scale": variance_scale,
        "variance_floor_m2": variance_floor,
        "nominal_coordinate_coverage": nominal_coverage,
    }


def load_deform_dlo2_deep_alltrain_protocol(
    path: str | Path,
) -> dict[str, object]:
    """Load the frozen two-seed all-56 DLO2 refit protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO2_ALLTRAIN_SCHEMA_VERSION:
        raise ValueError("unsupported DLO2 deep all-train schema")
    if payload.get("contract") != DEFORM_DLO2_DEEP_ALLTRAIN_CONTRACT:
        raise ValueError("unsupported DLO2 deep all-train contract")
    if payload.get("model_initialization") != "official-deform-dlo-initialization-v1":
        raise ValueError("DLO2 deep all-train initialization differs")
    parents = payload.get("parents")
    required = payload.get("required_parent")
    data = payload.get("data")
    training = payload.get("training")
    transfer = payload.get("method_transfer")
    output = payload.get("output")
    if not all(
        isinstance(value, Mapping)
        for value in (parents, required, data, training, transfer, output)
    ):
        raise ValueError("DLO2 deep all-train protocol is incomplete")
    for label in (
        "ensemble_protocol",
        "seed42_source_protocol",
        "seed43_source_protocol",
    ):
        identity = parents.get(label)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", ""))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError(f"DLO2 deep all-train {label} identity is invalid")
    if (
        required.get("source_result_contract")
        != "deform-dlo-source-reproduction-result-v1"
        or required.get("source_gate_passed") is not True
        or required.get("ensemble_result_contract")
        != "deform-dlo2-deep-ensemble-result-v1"
        or required.get("ensemble_selection_contract")
        != "deform-dlo2-deep-ensemble-v1"
        or required.get("ensemble_exact_fallback") is not False
        or required.get("ensemble_alltrain_authorized") is not True
    ):
        raise ValueError("DLO2 deep all-train parent gate differs")
    if (
        data.get("dlo_type") != "DLO2"
        or data.get("partition") != "train"
        or int(data.get("trajectory_count", -1)) != 56
        or int(data.get("frame_count", -1)) != 500
        or int(data.get("node_count", -1)) != 12
        or data.get("use_every_train_trajectory") is not True
        or data.get("official_eval_read") is not False
    ):
        raise ValueError("DLO2 deep all-train data contract differs")
    checkpoints = tuple(int(value) for value in training.get("checkpoint_updates", ()))
    if (
        training.get("random_seeds") != [42, 43]
        or int(training.get("unroll_horizon_frames", -1)) != 50
        or int(training.get("batch_size", -1)) != 32
        or int(training.get("total_updates", -1)) != 6400
        or not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] != 6400
        or tuple(sorted(set(checkpoints))) != checkpoints
        or training.get("optimizer") != "official-sgd-parameter-groups-v1"
        or training.get("cublas_workspace_config") != ":4096:8"
        or training.get("known_action_nodes") != [0, 1, -2, -1]
        or training.get("window_sampling")
        != "frozen-uniform-all-56-train-v1"
    ):
        raise ValueError("DLO2 deep all-train training contract differs")
    if (
        transfer.get("selection_source")
        != "fresh-dlo2-deep-ensemble-result"
        or transfer.get("operator")
        != "copy-selected-predictive-mean-exactly"
        or transfer.get("seed_weights") != "copy-selected-spec-exactly"
        or transfer.get("member_updates") != "copy-selected-spec-exactly"
        or transfer.get("validation_reselection") is not False
        or transfer.get("source_reselection") is not False
        or transfer.get("target_reselection") is not False
        or transfer.get("variance_scale")
        != "copy-validation-fitted-scale-exactly"
        or output.get("preserve_seed_member_checkpoints") is not True
        or output.get("assemble_only_after_both_seed_runs_verify") is not True
        or output.get("official_eval_authorized_by_one_seed_alone") is not False
        or output.get("official_eval_authorized_by_assembly_alone") is not False
    ):
        raise ValueError("DLO2 deep all-train transfer contract differs")
    result = dict(payload)
    result["checkpoint_updates"] = checkpoints
    result["protocol_path"] = str(source)
    return result


def validate_deform_dlo2_deep_alltrain_authorization(
    protocol: Mapping[str, object],
    source_results: Mapping[int, Mapping[str, object]],
    ensemble_result: Mapping[str, object],
    selection_seal: Mapping[str, object],
    *,
    source_protocol_sha256s: Mapping[int, str],
    source_result_sha256s: Mapping[int, str],
    ensemble_protocol_sha256: str,
    selection_seal_sha256: str,
) -> dict[str, object]:
    """Return the immutable two-seed method after every fresh-source gate passes."""

    required = protocol.get("required_parent")
    parents = protocol.get("parents")
    if not isinstance(required, Mapping) or not isinstance(parents, Mapping):
        raise ValueError("DLO2 deep all-train protocol omits parent gates")
    if set(source_results) != {42, 43} or set(source_protocol_sha256s) != {
        42,
        43,
    } or set(source_result_sha256s) != {42, 43}:
        raise ValueError("DLO2 deep all-train requires seeds 42 and 43")
    selected_updates = {}
    for seed in (42, 43):
        result = source_results[seed]
        source_gate = result.get("source_gate")
        selected = result.get("selected_checkpoint")
        stage = result.get("stage_authorization")
        parent_identity = parents.get(f"seed{seed}_source_protocol")
        if (
            result.get("contract") != required.get("source_result_contract")
            or result.get("official_eval_read") is not False
            or result.get("advancement_authorized") is not True
            or not isinstance(source_gate, Mapping)
            or source_gate.get("passed") is not required.get("source_gate_passed")
            or not isinstance(selected, Mapping)
            or not isinstance(stage, Mapping)
            or stage.get("contract")
            != "deform-dlo2-deep-seed-authorization-v1"
            or int(stage.get("seed", -1)) != seed
            or not isinstance(parent_identity, Mapping)
            or parent_identity.get("sha256") != source_protocol_sha256s[seed]
            or len(source_result_sha256s[seed]) != 64
        ):
            raise ValueError(f"DLO2 seed-{seed} did not authorize all-train refit")
        selected_updates[seed] = int(selected.get("update", -1))

    selection_identity = ensemble_result.get("selection_seal")
    selection = ensemble_result.get("selection")
    sealed_selection = selection_seal.get("selection")
    selected_spec = ensemble_result.get("selected_spec")
    candidate_specs = selection_seal.get("candidate_specs")
    sealed_protocol = selection_seal.get("protocol")
    source_test = ensemble_result.get("source_test")
    transfer = source_test.get("transfer") if isinstance(source_test, Mapping) else None
    candidate_gate = (
        source_test.get("candidate_gate") if isinstance(source_test, Mapping) else None
    )
    uncertainty = ensemble_result.get("uncertainty")
    alltrain_contract = ensemble_result.get("alltrain_authorization_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            selection_identity,
            selection,
            sealed_selection,
            selected_spec,
            candidate_specs,
            sealed_protocol,
            transfer,
            candidate_gate,
            uncertainty,
            alltrain_contract,
        )
    ):
        raise ValueError("DLO2 ensemble result is incomplete")
    selected_arm = str(ensemble_result.get("selected_arm", ""))
    raw_weights = selected_spec.get("weights")
    raw_updates = selected_spec.get("selected_member_updates")
    if not isinstance(raw_weights, Mapping) or not isinstance(raw_updates, Mapping):
        raise ValueError("DLO2 ensemble selected spec is malformed")
    weights = {int(seed): float(weight) for seed, weight in raw_weights.items()}
    member_updates = {int(seed): int(update) for seed, update in raw_updates.items()}
    source_identities = {
        seed: selection_seal.get(f"seed{seed}_source_result") for seed in (42, 43)
    }
    variance_scale = float(
        uncertainty.get("validation_fitted_variance_scale", math.nan)
    )
    variance_floor = float(uncertainty.get("variance_floor_m2", math.nan))
    nominal_coverage = float(uncertainty.get("nominal_coordinate_coverage", math.nan))
    checkpoint_updates = {int(value) for value in protocol.get("checkpoint_updates", ())}
    parent_ensemble = parents.get("ensemble_protocol")
    if (
        ensemble_result.get("contract")
        != required.get("ensemble_result_contract")
        or ensemble_result.get("official_eval_read") is not False
        or ensemble_result.get("exact_fallback")
        is not required.get("ensemble_exact_fallback")
        or ensemble_result.get("alltrain_deep_ensemble_authorized")
        is not required.get("ensemble_alltrain_authorized")
        or selection_seal.get("contract")
        != required.get("ensemble_selection_contract")
        or selection_seal.get("official_eval_read") is not False
        or selection_seal.get("source_test_evaluated_by_this_stage") is not False
        or not isinstance(parent_ensemble, Mapping)
        or parent_ensemble.get("sha256") != ensemble_protocol_sha256
        or sealed_protocol.get("sha256") != ensemble_protocol_sha256
        or selection_identity.get("sha256") != selection_seal_sha256
        or selection != sealed_selection
        or selection.get("fallback_used") is not False
        or not selected_arm
        or selection.get("selected_arm") != selected_arm
        or candidate_specs.get(selected_arm) != selected_spec
        or selected_spec.get("operator") != "predictive_mean"
        or set(weights) != {42, 43}
        or any(
            not math.isfinite(weight) or weight <= 0.0 for weight in weights.values()
        )
        or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-10)
        or member_updates != selected_updates
        or not set(member_updates.values()).issubset(checkpoint_updates)
        or int(ensemble_result.get("comparison_baseline_seed", -1)) not in (42, 43)
        or float(transfer.get("relative_improvement", -math.inf)) < 0.01
        or int(transfer.get("wins", -1)) < 5
        or candidate_gate.get("passed") is not True
        or any(
            not isinstance(source_identities[seed], Mapping)
            or source_identities[seed].get("sha256")
            != source_result_sha256s[seed]
            for seed in (42, 43)
        )
        or not math.isfinite(variance_scale)
        or variance_scale < 1.0
        or not math.isfinite(variance_floor)
        or variance_floor <= 0.0
        or not math.isfinite(nominal_coverage)
        or not 0.0 < nominal_coverage < 1.0
        or alltrain_contract.get("seeds") != [42, 43]
        or alltrain_contract.get("same_operator_and_weights") is not True
        or alltrain_contract.get("no_source_retuning") is not True
    ):
        raise ValueError("DLO2 ensemble did not authorize all-train refit")
    return {
        "operator": "predictive_mean",
        "weights": weights,
        "member_updates": member_updates,
        "comparison_baseline_seed": int(
            ensemble_result["comparison_baseline_seed"]
        ),
        "validation_fitted_variance_scale": variance_scale,
        "variance_floor_m2": variance_floor,
        "nominal_coordinate_coverage": nominal_coverage,
        "selected_arm": selected_arm,
    }
