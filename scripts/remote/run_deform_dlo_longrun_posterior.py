#!/usr/bin/env python3
"""Evaluate a frozen checkpoint posterior after the long-run source gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    average_deform_checkpoint_states,
    build_deform_checkpoint_belief_arms,
    calibrate_deform_coordinate_variance,
    combine_deform_checkpoint_predictions,
    deform_prediction_records,
    evaluate_deform_checkpoint_belief_transfer,
    evaluate_deform_coordinate_uncertainty,
    load_deform_longrun_posterior_protocol,
    select_deform_checkpoint_belief_arm,
    weighted_deform_prediction_median,
)
from bayesian_phystwin.deform_dlo_longrun import load_deform_dlo_longrun_protocol
from bayesian_phystwin.deform_dlo_source import (
    choose_deform_validation_checkpoint,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--longrun-protocol", type=Path, required=True)
    parser.add_argument("--longrun-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _validate_longrun_result(
    result: dict[str, object],
    *,
    protocol_sha256: str,
) -> None:
    source_gate = result.get("source_gate")
    if (
        result.get("contract") != "deform-dlo-longrun-result-v2"
        or result.get("official_eval_read") is not False
        or not isinstance(source_gate, dict)
        or source_gate.get("passed") is not True
        or result.get("checkpoint_posterior_authorized") is not True
    ):
        raise ValueError("long-run result did not authorize checkpoint posterior")
    result_protocol = result.get("protocol")
    if (
        not isinstance(result_protocol, dict)
        or result_protocol.get("sha256") != protocol_sha256
    ):
        raise ValueError("long-run result uses a different protocol")
    validation = result.get("validation")
    checkpoints = result.get("checkpoints")
    if not isinstance(validation, list) or not isinstance(checkpoints, list):
        raise ValueError("long-run result omits validation or checkpoints")
    if choose_deform_validation_checkpoint(validation) != result.get(
        "selected_checkpoint"
    ):
        raise ValueError("long-run selected checkpoint is inconsistent")


def _checkpoint_states(
    result: dict[str, object],
    required_updates: set[int],
    *,
    torch: Any,
) -> dict[int, dict[str, Any]]:
    records = result["checkpoints"]
    indexed = {int(record["update"]): record for record in records}
    if len(indexed) != len(records) or not required_updates.issubset(indexed):
        raise ValueError("long-run result omits required posterior checkpoints")
    states = {}
    for update in sorted(required_updates):
        identity = indexed[update]
        path = Path(str(identity["path"])).resolve()
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError("long-run checkpoint identity does not verify")
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if (
            int(bundle.get("global_update", -1)) != update
            or bundle.get("longrun_protocol_sha256") != result["protocol"]["sha256"]
            or bundle.get("continuation_schedule_sha256")
            != result["continuation_schedule"]["sha256"]
        ):
            raise ValueError("long-run checkpoint payload differs")
        state = bundle.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("long-run checkpoint omits its model state")
        states[update] = state
    return states


def _rollout_arrays(
    trajectories: dict[str, np.ndarray],
    *,
    modules: Any,
    model_function: Any,
    model: Any,
    torch: Any,
    device: str,
) -> dict[str, object]:
    names = list(trajectories)
    arrays = np.stack([trajectories[name] for name in names])
    values = torch.from_numpy(arrays).to(device=device)
    previous = values[:, :-2]
    vertices = values[:, 1:-1]
    targets = values[:, 2:]
    batch_size, horizon, node_count, _ = targets.shape
    clamped_selection = torch.tensor((0, 1, -2, -1), device=device)
    clamped_index = torch.zeros(node_count, device=device)
    clamped_index[clamped_selection] = 1.0
    initial = source_runtime._initial_direction(torch, device).repeat(
        batch_size,
        1,
        1,
    )
    inputs = targets[:, :, clamped_selection]
    theta_full = torch.zeros(batch_size, node_count - 1, device=device)
    model.eval()
    predictions = []
    with torch.no_grad():
        rest_edges = modules.computeEdges(vertices[:, 0])
        material_u0 = model_function.compute_u0(
            rest_edges[:, 0].float(),
            initial[:, 0],
        )
        current_velocity = (vertices[:, 0] - previous[:, 0]).div(model.dt)
        rest_lengths = model.m_restEdgeL.repeat(batch_size, 1)
        model.m_restWprev, model.m_restWnext, model.learned_pmass = model.Rod_Init(
            batch_size,
            initial,
            rest_lengths,
            clamped_index,
        )
        predicted_vertices = None
        propagated_vertices = None
        for frame in range(horizon):
            if frame == 0:
                predicted_vertices, current_velocity, theta_full = model(
                    vertices[:, frame],
                    current_velocity,
                    initial,
                    clamped_index,
                    material_u0,
                    inputs[:, frame],
                    clamped_selection,
                    theta_full,
                    mode="evaluation",
                )
            else:
                previous_vertices = (
                    previous[:, frame] if frame == 1 else propagated_vertices
                )
                propagated_vertices = predicted_vertices
                previous_edges = modules.computeEdges(previous_vertices)
                current_edges = modules.computeEdges(propagated_vertices)
                material_u0 = model_function.parallelTransportFrame(
                    previous_edges[:, 0],
                    current_edges[:, 0],
                    material_u0,
                )
                predicted_vertices, current_velocity, theta_full = model(
                    propagated_vertices.clone(),
                    current_velocity.clone(),
                    initial,
                    clamped_index,
                    material_u0,
                    inputs[:, frame],
                    clamped_selection,
                    theta_full,
                    mode="evaluation",
                )
            predictions.append(predicted_vertices.detach().cpu())
    predicted_array = torch.stack(predictions, dim=1).numpy()
    target_array = targets.detach().cpu().numpy()
    persistence = vertices[:, 0].unsqueeze(1).repeat(1, horizon, 1, 1)
    persistence[:, :, clamped_selection] = targets[:, :, clamped_selection]
    return {
        "names": names,
        "predictions": predicted_array,
        "targets": target_array,
        "persistence": persistence.detach().cpu().numpy(),
    }


def _evaluate_state(
    state: dict[str, Any],
    trajectories: dict[str, np.ndarray],
    *,
    modules: Any,
    torch: Any,
    device: str,
    dlo_type: str = "DLO1",
    node_count: int = 13,
) -> dict[str, object]:
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        device,
        dlo_type=dlo_type,
        node_count=node_count,
    )
    model.load_state_dict(state, strict=True)
    return _rollout_arrays(
        trajectories,
        modules=modules,
        model_function=model_function,
        model=model,
        torch=torch,
        device=device,
    )


def _assert_common_rollout(
    reference: dict[str, object],
    candidate: dict[str, object],
) -> None:
    if reference["names"] != candidate["names"]:
        raise ValueError("checkpoint posterior rollout names differ")
    for key in ("targets", "persistence"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"checkpoint posterior rollout {key} differs")


def _records(rollout: dict[str, object]) -> list[dict[str, object]]:
    return deform_prediction_records(
        rollout["predictions"],
        rollout["targets"],
        rollout["persistence"],
        rollout["names"],
    )


def _mean_error(records: list[dict[str, object]]) -> float:
    errors = np.asarray([float(record["model_l1_m"]) for record in records])
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("checkpoint-posterior errors are invalid")
    return float(np.mean(errors))


def main() -> int:
    args = _parse_args()
    posterior = load_deform_longrun_posterior_protocol(args.protocol)
    load_deform_dlo_longrun_protocol(args.longrun_protocol)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    result_path = args.longrun_result.resolve()
    result = _read_json(result_path)
    posterior_protocol_sha256 = sha256_file(args.protocol)
    longrun_protocol_sha256 = sha256_file(args.longrun_protocol)
    if longrun_protocol_sha256 != posterior["parent_longrun_protocol"]["sha256"]:
        raise ValueError("posterior policy binds a different long-run protocol")
    _validate_longrun_result(
        result,
        protocol_sha256=longrun_protocol_sha256,
    )
    manifest_path = args.source_manifest.resolve()
    manifest = _read_json(manifest_path)
    parent_source_result = _read_json(
        Path(posterior["parent_source_result"]["repository_path"])
    )
    if (
        sha256_file(Path(posterior["parent_source_result"]["repository_path"]))
        != posterior["parent_source_result"]["sha256"]
        or sha256_file(manifest_path)
        != parent_source_result["source_manifest"]["sha256"]
        or manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("official_eval_read") is not False
        or manifest.get("dlo_type") != "DLO1"
    ):
        raise ValueError("checkpoint posterior source lineage differs")

    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        result["upstream"]["commit"],
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    cublas_config = str(result["runtime"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    if (
        torch.__version__ != result["runtime"]["torch"]
        or torch.version.cuda != result["runtime"]["cuda"]
    ):
        raise RuntimeError("checkpoint-posterior runtime differs from long run")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, 20260731)

    arm_protocol = {
        "arms": posterior["arms"],
        "validation_gate": {
            "softmax_temperature_m": posterior["softmax_temperature_m"],
        },
    }
    validation_records = result["validation"]
    weights_by_base_arm = build_deform_checkpoint_belief_arms(
        validation_records,
        arm_protocol,
    )
    required_updates = {
        update
        for name, weights in weights_by_base_arm.items()
        if name != "selected_single"
        for update in weights
    }
    states = _checkpoint_states(
        result,
        required_updates,
        torch=torch,
    )

    frame_count = 500
    node_count = 13
    validation_names = list(manifest["split"]["validation"])
    validation_trajectories = source_runtime._load_named_trajectories(
        manifest,
        validation_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    member_validation: dict[int, dict[str, object]] = {}
    reference_validation = None
    for update in sorted(required_updates):
        rollout = _evaluate_state(
            states[update],
            validation_trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
        )
        if reference_validation is None:
            reference_validation = rollout
        else:
            _assert_common_rollout(reference_validation, rollout)
        member_validation[update] = rollout
    if reference_validation is None:
        raise ValueError("checkpoint posterior has no validation members")

    selected_single = choose_deform_validation_checkpoint(validation_records)
    validation_errors = {"selected_single": float(selected_single["validation_l1_m"])}
    candidate_specs: dict[str, dict[str, object]] = {}
    candidate_validation: dict[str, dict[str, object]] = {}
    candidate_variance: dict[str, np.ndarray] = {}
    for base_name, weights in weights_by_base_arm.items():
        if base_name == "selected_single":
            continue
        member_predictions = {
            update: member_validation[update]["predictions"] for update in weights
        }
        predictive_mean, predictive_variance = combine_deform_checkpoint_predictions(
            member_predictions, weights
        )
        predictive_median = weighted_deform_prediction_median(
            member_predictions,
            weights,
        )
        for operator in posterior["operators"]:
            candidate_name = f"{operator}::{base_name}"
            if operator == "predictive_mean":
                point_prediction = predictive_mean
            elif operator == "predictive_median":
                point_prediction = predictive_median
            elif operator == "parameter_mean":
                averaged_state = average_deform_checkpoint_states(
                    {update: states[update] for update in weights},
                    weights,
                )
                parameter_rollout = _evaluate_state(
                    averaged_state,
                    validation_trajectories,
                    modules=modules,
                    torch=torch,
                    device=args.device,
                )
                _assert_common_rollout(reference_validation, parameter_rollout)
                point_prediction = parameter_rollout["predictions"]
            else:
                raise ValueError("unsupported checkpoint-posterior operator")
            rollout = {
                "names": reference_validation["names"],
                "predictions": point_prediction,
                "targets": reference_validation["targets"],
                "persistence": reference_validation["persistence"],
            }
            records = _records(rollout)
            validation_errors[candidate_name] = _mean_error(records)
            candidate_specs[candidate_name] = {
                "operator": operator,
                "base_arm": base_name,
                "weights": weights,
            }
            candidate_validation[candidate_name] = rollout
            candidate_variance[candidate_name] = predictive_variance

    selection = select_deform_checkpoint_belief_arm(
        validation_errors,
        minimum_relative_improvement=float(posterior["validation_improvement_min"]),
    )
    selected_arm = str(selection["selected_arm"])
    selection_seal = {
        "schema_version": 1,
        "contract": "deform-dlo-longrun-posterior-selection-v1",
        "claim_boundary": posterior["claim_boundary"],
        "official_eval_read": False,
        "longrun_result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": posterior_protocol_sha256,
        },
        "longrun_protocol": {
            "path": str(args.longrun_protocol.resolve()),
            "sha256": longrun_protocol_sha256,
        },
        "upstream": upstream,
        "candidate_specs": candidate_specs,
        "validation_errors_l1_m": validation_errors,
        "selection": selection,
        "source_test_evaluated_by_this_stage": False,
    }
    selection_path = output_root / "selection_seal.json"
    _write_json(selection_path, selection_seal)

    baseline_source_records = list(result["source_test"])
    uncertainty = None
    if selection["fallback_used"]:
        candidate_source_records = baseline_source_records
        exact_fallback = True
    else:
        exact_fallback = False
        spec = candidate_specs[selected_arm]
        weights = spec["weights"]
        source_names = list(manifest["split"]["source_test"])
        source_trajectories = source_runtime._load_named_trajectories(
            manifest,
            source_names,
            frame_count=frame_count,
            node_count=node_count,
        )
        member_source = {}
        reference_source = None
        for update in sorted(weights):
            rollout = _evaluate_state(
                states[update],
                source_trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
            )
            if reference_source is None:
                reference_source = rollout
            else:
                _assert_common_rollout(reference_source, rollout)
            member_source[update] = rollout
        source_mean, source_variance = combine_deform_checkpoint_predictions(
            {update: member_source[update]["predictions"] for update in weights},
            weights,
        )
        if spec["operator"] == "predictive_mean":
            source_point_prediction = source_mean
        elif spec["operator"] == "predictive_median":
            source_point_prediction = weighted_deform_prediction_median(
                {update: member_source[update]["predictions"] for update in weights},
                weights,
            )
        else:
            averaged_state = average_deform_checkpoint_states(
                {update: states[update] for update in weights},
                weights,
            )
            parameter_source = _evaluate_state(
                averaged_state,
                source_trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
            )
            _assert_common_rollout(reference_source, parameter_source)
            source_point_prediction = parameter_source["predictions"]
        source_rollout = {
            "names": reference_source["names"],
            "predictions": source_point_prediction,
            "targets": reference_source["targets"],
            "persistence": reference_source["persistence"],
        }
        candidate_source_records = _records(source_rollout)
        validation_rollout = candidate_validation[selected_arm]
        variance_floor = float(posterior["coordinate_variance_floor_m2"])
        variance_scale = calibrate_deform_coordinate_variance(
            validation_rollout["predictions"],
            validation_rollout["targets"],
            candidate_variance[selected_arm],
            variance_floor_m2=variance_floor,
        )
        nominal_coverage = float(posterior["coordinate_interval_nominal_coverage"])
        uncertainty = {
            "variance_floor_m2": variance_floor,
            "validation_fitted_variance_scale": variance_scale,
            "nominal_coordinate_coverage": nominal_coverage,
            "validation": evaluate_deform_coordinate_uncertainty(
                validation_rollout["predictions"],
                validation_rollout["targets"],
                candidate_variance[selected_arm],
                variance_floor_m2=variance_floor,
                variance_scale=variance_scale,
                nominal_coverage=nominal_coverage,
            ),
            "source_test": evaluate_deform_coordinate_uncertainty(
                source_point_prediction,
                reference_source["targets"],
                source_variance,
                variance_floor_m2=variance_floor,
                variance_scale=variance_scale,
                nominal_coverage=nominal_coverage,
            ),
            "claim_boundary": "coordinate-marginal diagnostic; not independent calibration",
        }

    transfer = evaluate_deform_checkpoint_belief_transfer(
        candidate_source_records,
        baseline_source_records,
    )
    continuation_passed = (
        not exact_fallback
        and float(transfer["relative_improvement"])
        >= float(posterior["source_transfer_improvement_min"])
        and int(transfer["wins"]) >= int(posterior["source_transfer_minimum_case_wins"])
    )
    output = {
        "schema_version": 1,
        "contract": "deform-dlo-longrun-posterior-result-v1",
        "claim_boundary": posterior["claim_boundary"],
        "official_eval_read": False,
        "selection_seal": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "selection": selection,
        "selected_arm": selected_arm,
        "selected_spec": candidate_specs.get(selected_arm),
        "exact_fallback": exact_fallback,
        "source_test": {
            "candidate": candidate_source_records,
            "baseline": baseline_source_records,
            "transfer": transfer,
        },
        "uncertainty": uncertainty,
        "fresh_dlo2_checkpoint_posterior_authorized": continuation_passed,
        "fresh_confirmation_contract": posterior["fresh_confirmation"],
    }
    output_path = output_root / "posterior_result.json"
    _write_json(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
