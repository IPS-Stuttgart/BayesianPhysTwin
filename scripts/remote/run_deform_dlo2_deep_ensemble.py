#!/usr/bin/env python3
"""Evaluate the frozen two-seed ensemble on fresh DLO2 source data."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import run_deform_dlo_checkpoint_belief as source_belief_runtime
import run_deform_dlo_deep_ensemble as ensemble_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    calibrate_deform_coordinate_variance,
    combine_deform_checkpoint_predictions,
    evaluate_deform_checkpoint_belief_transfer,
    evaluate_deform_coordinate_uncertainty,
    select_deform_checkpoint_belief_arm,
)
from bayesian_phystwin.deform_dlo_deep_ensemble import (
    DEFORM_DLO2_DEEP_ENSEMBLE_CONTRACT,
    DEFORM_DLO2_DEEP_ENSEMBLE_RESULT_CONTRACT,
    build_deform_two_seed_weights,
    load_deform_dlo1_deep_ensemble_protocol,
    load_deform_dlo2_deep_ensemble_protocol,
    validate_deform_two_seed_manifests,
)
from bayesian_phystwin.deform_dlo_source import (
    evaluate_deform_source_gate,
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed42-source-protocol", type=Path, required=True)
    parser.add_argument("--seed42-source-result", type=Path, required=True)
    parser.add_argument("--seed43-source-protocol", type=Path, required=True)
    parser.add_argument("--seed43-source-result", type=Path, required=True)
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


def _verify_parent_policy(
    protocol: Mapping[str, object],
    *,
    repository_root: Path,
) -> None:
    parents = protocol["parents"]
    identity = parents["dlo1_ensemble_protocol"]
    path = (repository_root / str(identity["repository_path"])).resolve()
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError("DLO1 ensemble policy identity does not verify")
    parent = load_deform_dlo1_deep_ensemble_protocol(path)
    parent_policy = parent["policy"]
    policy = protocol["policy"]
    for key in (
        "member_checkpoint_selection",
        "comparison_baseline",
        "fallback",
        "operators",
        "validation_improvement_min",
        "source_transfer_improvement_min",
        "source_transfer_minimum_case_wins",
        "coordinate_variance_floor_m2",
        "coordinate_interval_nominal_coverage",
    ):
        if policy[key] != parent_policy[key]:
            raise ValueError("DLO2 ensemble retuned the DLO1 operator policy")


def _verified_source(
    *,
    seed: int,
    protocol_path: Path,
    protocol_identity: Mapping[str, object],
    result_path: Path,
    upstream_commit: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if (
        sha256_file(protocol_path) != protocol_identity.get("sha256")
        or str(protocol_identity.get("repository_path", ""))
        != f"configs/sota/deform_dlo2_deep_seed{seed}_v1.json"
    ):
        raise ValueError(f"DLO2 seed-{seed} protocol identity differs")
    source_protocol = load_deform_dlo_source_protocol(protocol_path)
    if (
        source_protocol["dlo_types"] != ("DLO2",)
        or int(source_protocol["training"]["random_seed"]) != seed
        or int(source_protocol["deep_ensemble_role"]["seed"]) != seed
    ):
        raise ValueError(f"DLO2 seed-{seed} source protocol differs")
    result = _read_json(result_path)
    source_belief_runtime._validate_source_result(
        result,
        source_protocol_sha256=sha256_file(protocol_path),
        upstream_commit=upstream_commit,
    )
    stage = result.get("stage_authorization")
    if (
        not isinstance(stage, Mapping)
        or stage.get("contract") != "deform-dlo2-deep-seed-authorization-v1"
        or stage.get("protocol_sha256") != sha256_file(protocol_path)
        or int(stage.get("seed", -1)) != seed
        or int(stage.get("peer_seed", -1)) != ({42, 43} - {seed}).pop()
        or not str(stage.get("parent_deep_ensemble_result_sha256", ""))
    ):
        raise ValueError(f"DLO2 seed-{seed} source authorization differs")
    manifest_identity = result.get("source_manifest")
    if not isinstance(manifest_identity, Mapping):
        raise ValueError(f"DLO2 seed-{seed} source manifest is missing")
    manifest_path = Path(str(manifest_identity.get("path", ""))).resolve()
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != manifest_identity.get("sha256")
    ):
        raise ValueError(f"DLO2 seed-{seed} source manifest does not verify")
    return source_protocol, result, _read_json(manifest_path)


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo2_deep_ensemble_protocol(protocol_path)
    policy = protocol["policy"]
    parents = protocol["parents"]
    evaluation = protocol["evaluation"]
    upstream_policy = protocol["upstream"]
    if not all(
        isinstance(value, Mapping)
        for value in (policy, parents, evaluation, upstream_policy)
    ):
        raise ValueError("DLO2 ensemble protocol sections are malformed")
    repository_root = Path(__file__).resolve().parents[2]
    _verify_parent_policy(protocol, repository_root=repository_root)

    source_protocols = {}
    source_results = {}
    manifests = {}
    result_paths = {
        42: args.seed42_source_result.resolve(),
        43: args.seed43_source_result.resolve(),
    }
    protocol_paths = {
        42: args.seed42_source_protocol.resolve(),
        43: args.seed43_source_protocol.resolve(),
    }
    for seed in (42, 43):
        identity = parents[f"seed{seed}_source_protocol"]
        if not isinstance(identity, Mapping):
            raise ValueError(f"DLO2 seed-{seed} protocol lock is malformed")
        source_protocols[seed], source_results[seed], manifests[seed] = (
            _verified_source(
                seed=seed,
                protocol_path=protocol_paths[seed],
                protocol_identity=identity,
                result_path=result_paths[seed],
                upstream_commit=str(upstream_policy["commit"]),
            )
        )
    validate_deform_two_seed_manifests(
        manifests[42],
        manifests[43],
        dlo_type="DLO2",
    )
    stages = [source_results[seed]["stage_authorization"] for seed in (42, 43)]
    if (
        stages[0]["parent_deep_ensemble_result_sha256"]
        != stages[1]["parent_deep_ensemble_result_sha256"]
        or stages[0]["selected_arm"] != stages[1]["selected_arm"]
        or ensemble_runtime._runtime_identity(source_results[42])
        != ensemble_runtime._runtime_identity(source_results[43])
    ):
        raise ValueError("DLO2 ensemble members do not share one frozen lineage")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")
    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        str(upstream_policy["commit"]),
    )
    cublas_config = str(evaluation["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("DLO2 ensemble cuBLAS configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    runtime_torch, runtime_cuda = ensemble_runtime._runtime_identity(
        source_results[42]
    )
    if torch.__version__ != runtime_torch or torch.version.cuda != runtime_cuda:
        raise RuntimeError("DLO2 ensemble evaluator runtime differs")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, 20260731)
    selected_updates = {}
    states = {}
    for seed in (42, 43):
        selected_updates[seed], states[seed] = ensemble_runtime._selected_state(
            source_results[seed],
            torch=torch,
        )

    frame_count = int(evaluation["frame_count"])
    node_count = int(evaluation["node_count"])
    validation_names = list(manifests[42]["split"]["validation"])
    validation_trajectories = source_runtime._load_named_trajectories(
        manifests[42],
        validation_names,
        frame_count=frame_count,
        node_count=node_count,
    )
    validation_rollouts = {}
    reference_validation = None
    for seed in (42, 43):
        rollout = posterior_runtime._evaluate_state(
            states[seed],
            validation_trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
            dlo_type="DLO2",
            node_count=node_count,
        )
        if reference_validation is None:
            reference_validation = rollout
        else:
            posterior_runtime._assert_common_rollout(reference_validation, rollout)
        validation_rollouts[seed] = rollout
    if reference_validation is None:
        raise RuntimeError("DLO2 ensemble validation rollout is empty")

    member_validation_errors = {
        seed: ensemble_runtime._mean_error(
            posterior_runtime._records(validation_rollouts[seed])
        )
        for seed in (42, 43)
    }
    baseline_seed = min(
        member_validation_errors,
        key=lambda seed: (member_validation_errors[seed], seed),
    )
    validation_errors = {"selected_single": member_validation_errors[baseline_seed]}
    candidate_specs = {}
    candidate_validation = {}
    candidate_variance = {}
    weights_by_arm = build_deform_two_seed_weights(member_validation_errors, policy)
    member_predictions = {
        seed: validation_rollouts[seed]["predictions"] for seed in (42, 43)
    }
    for name, weights in weights_by_arm.items():
        prediction, variance = combine_deform_checkpoint_predictions(
            member_predictions,
            weights,
        )
        candidate = {
            "names": reference_validation["names"],
            "predictions": prediction,
            "targets": reference_validation["targets"],
            "persistence": reference_validation["persistence"],
        }
        validation_errors[name] = ensemble_runtime._mean_error(
            posterior_runtime._records(candidate)
        )
        candidate_specs[name] = {
            "operator": "predictive_mean",
            "weights": weights,
            "selected_member_updates": selected_updates,
        }
        candidate_validation[name] = candidate
        candidate_variance[name] = variance

    selection = select_deform_checkpoint_belief_arm(
        validation_errors,
        minimum_relative_improvement=float(policy["validation_improvement_min"]),
    )
    selected_arm = str(selection["selected_arm"])
    selection_seal = {
        "schema_version": 1,
        "contract": DEFORM_DLO2_DEEP_ENSEMBLE_CONTRACT,
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "seed42_source_result": {
            "path": str(result_paths[42]),
            "sha256": sha256_file(result_paths[42]),
        },
        "seed43_source_result": {
            "path": str(result_paths[43]),
            "sha256": sha256_file(result_paths[43]),
        },
        "upstream": upstream,
        "baseline_seed": baseline_seed,
        "candidate_specs": candidate_specs,
        "validation_errors_l1_m": validation_errors,
        "selection": selection,
        "source_test_evaluated_by_this_stage": False,
    }
    selection_path = output_root / "selection_seal.json"
    _write_json(selection_path, selection_seal)

    baseline_source_records = ensemble_runtime._stored_source_records(
        source_results[baseline_seed]
    )
    uncertainty = None
    if selection["fallback_used"]:
        candidate_source_records = baseline_source_records
        exact_fallback = True
    else:
        exact_fallback = False
        source_names = list(manifests[42]["split"]["source_test"])
        source_trajectories = source_runtime._load_named_trajectories(
            manifests[42],
            source_names,
            frame_count=frame_count,
            node_count=node_count,
        )
        source_rollouts = {}
        reference_source = None
        for seed in (42, 43):
            rollout = posterior_runtime._evaluate_state(
                states[seed],
                source_trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
                dlo_type="DLO2",
                node_count=node_count,
            )
            if reference_source is None:
                reference_source = rollout
            else:
                posterior_runtime._assert_common_rollout(reference_source, rollout)
            source_rollouts[seed] = rollout
        if reference_source is None:
            raise RuntimeError("DLO2 ensemble source rollout is empty")
        weights = candidate_specs[selected_arm]["weights"]
        source_prediction, source_variance = combine_deform_checkpoint_predictions(
            {seed: source_rollouts[seed]["predictions"] for seed in (42, 43)},
            weights,
        )
        candidate_source_records = posterior_runtime._records(
            {
                "names": reference_source["names"],
                "predictions": source_prediction,
                "targets": reference_source["targets"],
                "persistence": reference_source["persistence"],
            }
        )
        rerun_baseline_records = posterior_runtime._records(
            source_rollouts[baseline_seed]
        )
        for stored, rerun in zip(
            baseline_source_records,
            rerun_baseline_records,
            strict=True,
        ):
            if stored["name"] != rerun["name"] or not math.isclose(
                float(stored["model_l1_m"]),
                float(rerun["model_l1_m"]),
                rel_tol=0.0,
                abs_tol=float(evaluation["source_replay_tolerance_m"]),
            ):
                raise RuntimeError("DLO2 ensemble baseline source replay differs")
        baseline_source_records = rerun_baseline_records
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
                source_prediction,
                reference_source["targets"],
                source_variance,
                variance_floor_m2=variance_floor,
                variance_scale=variance_scale,
                nominal_coverage=nominal_coverage,
            ),
            "claim_boundary": "coordinate-marginal fresh-source diagnostic",
        }

    transfer = evaluate_deform_checkpoint_belief_transfer(
        candidate_source_records,
        baseline_source_records,
        claim_boundary="fresh DLO2 source confirmation; official eval unopened",
    )
    candidate_gate = evaluate_deform_source_gate(
        candidate_source_records,
        published_reference_l1_m=float(
            policy["candidate_published_reference_l1_m"]
        ),
        published_error_multiplier_max=float(
            policy["candidate_published_error_multiplier_max"]
        ),
        minimum_persistence_wins=int(policy["candidate_minimum_persistence_wins"]),
    )
    authorized = (
        not exact_fallback
        and float(transfer["relative_improvement"])
        >= float(policy["source_transfer_improvement_min"])
        and int(transfer["wins"])
        >= int(policy["source_transfer_minimum_case_wins"])
        and candidate_gate["passed"] is True
    )
    result = {
        "schema_version": 1,
        "contract": DEFORM_DLO2_DEEP_ENSEMBLE_RESULT_CONTRACT,
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "selection_seal": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "selection": selection,
        "selected_arm": selected_arm,
        "selected_spec": candidate_specs.get(selected_arm),
        "comparison_baseline_seed": baseline_seed,
        "exact_fallback": exact_fallback,
        "source_test": {
            "candidate": candidate_source_records,
            "baseline": baseline_source_records,
            "transfer": transfer,
            "candidate_gate": candidate_gate,
        },
        "uncertainty": uncertainty,
        "alltrain_deep_ensemble_authorized": authorized,
        "alltrain_authorization_contract": policy["alltrain_authorization"],
    }
    result_path = output_root / "ensemble_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
