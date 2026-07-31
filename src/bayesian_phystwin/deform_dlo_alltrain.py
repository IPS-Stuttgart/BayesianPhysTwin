"""Authorization helpers for the final all-training-data DEFORM refit."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFORM_DLO2_ALLTRAIN_SCHEMA_VERSION = 1
DEFORM_DLO2_ALLTRAIN_CONTRACT = "deform-dlo2-alltrain-refit-v1"


def load_deform_dlo2_alltrain_protocol(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the target-blind all-56 refit protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO2_ALLTRAIN_SCHEMA_VERSION:
        raise ValueError("unsupported DLO2 all-train schema")
    if payload.get("contract") != DEFORM_DLO2_ALLTRAIN_CONTRACT:
        raise ValueError("unsupported DLO2 all-train contract")
    if (
        payload.get("model_initialization")
        != "official-deform-dlo-initialization-v1"
    ):
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
    if operator not in ("parameter_mean", "predictive_mean") or not isinstance(
        raw_weights, Mapping
    ):
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
