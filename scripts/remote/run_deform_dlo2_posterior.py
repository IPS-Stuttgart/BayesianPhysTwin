#!/usr/bin/env python3
"""Evaluate the frozen Bayesian checkpoint posterior on fresh DLO2 source data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_deform_dlo_checkpoint_belief as source_belief_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    average_deform_checkpoint_states,
    build_deform_checkpoint_belief_arms,
    calibrate_deform_coordinate_variance,
    combine_deform_checkpoint_predictions,
    evaluate_deform_checkpoint_belief_transfer,
    evaluate_deform_coordinate_uncertainty,
    select_deform_checkpoint_belief_arm,
    validate_deform_dlo2_checkpoint_posterior,
)
from bayesian_phystwin.deform_dlo_source import (
    choose_deform_validation_checkpoint,
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
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


def main() -> int:
    args = _parse_args()
    protocol = load_deform_dlo_source_protocol(args.protocol)
    policy = validate_deform_dlo2_checkpoint_posterior(protocol)
    protocol_sha256 = sha256_file(args.protocol)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_result_path = args.source_result.resolve()
    source_result = _read_json(source_result_path)
    upstream_commit = str(protocol["upstream"]["commit"])
    source_belief_runtime._validate_source_result(
        source_result,
        source_protocol_sha256=protocol_sha256,
        upstream_commit=upstream_commit,
    )
    upstream = source_runtime._assert_upstream(args.upstream_root, upstream_commit)

    manifest_identity = source_result.get("source_manifest")
    if not isinstance(manifest_identity, dict):
        raise ValueError("DLO2 source result omits its manifest identity")
    manifest_path = Path(str(manifest_identity.get("path", ""))).resolve()
    if not manifest_path.is_file() or sha256_file(
        manifest_path
    ) != manifest_identity.get("sha256"):
        raise ValueError("DLO2 source manifest identity does not verify")
    manifest = _read_json(manifest_path)
    manifest_protocol = manifest.get("protocol")
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO2"
        or manifest.get("official_eval_read") is not False
        or not isinstance(manifest_protocol, dict)
        or manifest_protocol.get("sha256") != protocol_sha256
    ):
        raise ValueError("DLO2 posterior source lineage differs")

    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")
    cublas_config = str(protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    runtime = source_result.get("runtime")
    if (
        not isinstance(runtime, dict)
        or torch.__version__ != runtime.get("torch")
        or torch.version.cuda != runtime.get("cuda")
    ):
        raise RuntimeError("DLO2 posterior runtime differs from its source run")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(
        torch,
        int(protocol["training"]["random_seed"]),
    )

    validation_records = source_result.get("validation")
    if not isinstance(validation_records, list):
        raise ValueError("DLO2 source result omits validation records")
    arm_protocol = {
        "arms": policy["arms"],
        "validation_gate": {
            "softmax_temperature_m": policy["softmax_temperature_m"],
        },
    }
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
    states = source_belief_runtime._checkpoint_states(
        source_result,
        required_updates,
    )

    frame_count = int(protocol["data"]["expected_frames_per_trajectory"])
    node_count = int(protocol["data"]["expected_node_count"]["DLO2"])
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
        rollout = posterior_runtime._evaluate_state(
            states[update],
            validation_trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
            node_count=node_count,
        )
        if reference_validation is None:
            reference_validation = rollout
        else:
            posterior_runtime._assert_common_rollout(reference_validation, rollout)
        member_validation[update] = rollout
    if reference_validation is None:
        raise ValueError("DLO2 posterior has no validation members")

    selected_single = choose_deform_validation_checkpoint(validation_records)
    validation_errors = {"selected_single": float(selected_single["validation_l1_m"])}
    candidate_specs: dict[str, dict[str, object]] = {}
    candidate_validation: dict[str, dict[str, object]] = {}
    candidate_variance = {}
    for base_name, weights in weights_by_base_arm.items():
        if base_name == "selected_single":
            continue
        predictive_mean, predictive_variance = combine_deform_checkpoint_predictions(
            {update: member_validation[update]["predictions"] for update in weights},
            weights,
        )
        for operator in policy["operators"]:
            candidate_name = f"{operator}::{base_name}"
            if operator == "predictive_mean":
                point_prediction = predictive_mean
            elif operator == "parameter_mean":
                averaged_state = average_deform_checkpoint_states(
                    {update: states[update] for update in weights},
                    weights,
                )
                parameter_rollout = posterior_runtime._evaluate_state(
                    averaged_state,
                    validation_trajectories,
                    modules=modules,
                    torch=torch,
                    device=args.device,
                    node_count=node_count,
                )
                posterior_runtime._assert_common_rollout(
                    reference_validation,
                    parameter_rollout,
                )
                point_prediction = parameter_rollout["predictions"]
            else:
                raise ValueError("unsupported DLO2 posterior operator")
            candidate_rollout = {
                "names": reference_validation["names"],
                "predictions": point_prediction,
                "targets": reference_validation["targets"],
                "persistence": reference_validation["persistence"],
            }
            records = posterior_runtime._records(candidate_rollout)
            validation_errors[candidate_name] = posterior_runtime._mean_error(records)
            candidate_specs[candidate_name] = {
                "operator": operator,
                "base_arm": base_name,
                "weights": weights,
            }
            candidate_validation[candidate_name] = candidate_rollout
            candidate_variance[candidate_name] = predictive_variance

    selection = select_deform_checkpoint_belief_arm(
        validation_errors,
        minimum_relative_improvement=float(policy["validation_improvement_min"]),
    )
    selected_arm = str(selection["selected_arm"])
    selection_seal = {
        "schema_version": 1,
        "contract": "deform-dlo2-posterior-selection-v1",
        "claim_boundary": "fresh DLO2 source selection; official evaluation unopened",
        "official_eval_read": False,
        "source_result": {
            "path": str(source_result_path),
            "sha256": sha256_file(source_result_path),
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": protocol_sha256,
        },
        "upstream": upstream,
        "candidate_specs": candidate_specs,
        "validation_errors_l1_m": validation_errors,
        "selection": selection,
        "source_test_evaluated_by_this_stage": False,
    }
    selection_path = output_root / "selection_seal.json"
    _write_json(selection_path, selection_seal)

    baseline_source_records = list(source_result["source_test"])
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
            rollout = posterior_runtime._evaluate_state(
                states[update],
                source_trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
                node_count=node_count,
            )
            if reference_source is None:
                reference_source = rollout
            else:
                posterior_runtime._assert_common_rollout(reference_source, rollout)
            member_source[update] = rollout
        source_mean, source_variance = combine_deform_checkpoint_predictions(
            {update: member_source[update]["predictions"] for update in weights},
            weights,
        )
        if spec["operator"] == "predictive_mean":
            source_point_prediction = source_mean
        else:
            averaged_state = average_deform_checkpoint_states(
                {update: states[update] for update in weights},
                weights,
            )
            parameter_source = posterior_runtime._evaluate_state(
                averaged_state,
                source_trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
                node_count=node_count,
            )
            posterior_runtime._assert_common_rollout(
                reference_source,
                parameter_source,
            )
            source_point_prediction = parameter_source["predictions"]
        source_rollout = {
            "names": reference_source["names"],
            "predictions": source_point_prediction,
            "targets": reference_source["targets"],
            "persistence": reference_source["persistence"],
        }
        candidate_source_records = posterior_runtime._records(source_rollout)
        validation_rollout = candidate_validation[selected_arm]
        variance_floor = float(policy["coordinate_variance_floor_m2"])
        variance_scale = calibrate_deform_coordinate_variance(
            validation_rollout["predictions"],
            validation_rollout["targets"],
            candidate_variance[selected_arm],
            variance_floor_m2=variance_floor,
        )
        nominal_coverage = float(policy["coordinate_interval_nominal_coverage"])
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
            "claim_boundary": (
                "coordinate-marginal fresh-source diagnostic; "
                "official evaluation unopened"
            ),
        }

    transfer = evaluate_deform_checkpoint_belief_transfer(
        candidate_source_records,
        baseline_source_records,
        claim_boundary="fresh DLO2 source transfer; official evaluation unopened",
    )
    continuation_passed = (
        not exact_fallback
        and float(transfer["relative_improvement"])
        >= float(policy["source_transfer_improvement_min"])
        and int(transfer["wins"]) >= int(policy["source_transfer_minimum_case_wins"])
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo2-posterior-result-v1",
        "claim_boundary": "fresh DLO2 source result; not an official evaluation",
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
        "identical_information_official_eval_authorized": continuation_passed,
    }
    result_path = output_root / "posterior_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
