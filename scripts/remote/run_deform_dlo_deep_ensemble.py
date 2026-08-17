#!/usr/bin/env python3
"""Evaluate the frozen two-seed DEFORM ensemble on DLO1 source data."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_dlo_checkpoint_belief as source_belief_runtime
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
    DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT,
    build_deform_two_seed_weights,
    load_deform_dlo1_deep_ensemble_protocol,
    validate_deform_two_seed_manifests,
)
from bayesian_phystwin.deform_dlo_longrun import load_deform_dlo_longrun_protocol
from bayesian_phystwin.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed42-longrun-protocol", type=Path, required=True)
    parser.add_argument("--seed42-longrun-result", type=Path, required=True)
    parser.add_argument("--seed42-source-manifest", type=Path, required=True)
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


def _verified_manifest(path: Path, *, expected_sha256: object) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise ValueError("two-seed source manifest identity does not verify")
    return _read_json(resolved)


def _selected_state(
    result: Mapping[str, object],
    *,
    torch: Any,
) -> tuple[int, dict[str, Any]]:
    selected = result.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise ValueError("two-seed result omits its selected checkpoint")
    checkpoint = selected.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("two-seed selected checkpoint identity is malformed")
    path = Path(str(checkpoint.get("path", ""))).resolve()
    update = int(str(selected.get("update", -1)))
    if (
        update < 0
        or int(str(checkpoint.get("update", -2))) != update
        or not path.is_file()
        or sha256_file(path) != checkpoint.get("sha256")
    ):
        raise ValueError("two-seed selected checkpoint identity does not verify")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    state = bundle.get("model_state_dict") if isinstance(bundle, Mapping) else None
    if not isinstance(state, dict):
        raise ValueError("two-seed selected checkpoint omits model state")
    return update, state


def _runtime_identity(result: Mapping[str, object]) -> tuple[object, object]:
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("two-seed result omits runtime identity")
    return runtime.get("torch"), runtime.get("cuda")


def _mean_error(records: list[dict[str, object]]) -> float:
    values = np.asarray([float(record["model_l1_m"]) for record in records])
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("two-seed prediction errors are invalid")
    return float(np.mean(values))


def _stored_source_records(result: Mapping[str, object]) -> list[dict[str, object]]:
    records = result.get("source_test")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("two-seed result omits source records")
    return list(records)


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo1_deep_ensemble_protocol(protocol_path)
    policy = protocol["policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("two-seed policy is malformed")
    parents = protocol["parents"]
    evaluation = protocol["evaluation"]
    upstream_policy = protocol["upstream"]
    if not all(
        isinstance(value, Mapping) for value in (parents, evaluation, upstream_policy)
    ):
        raise ValueError("two-seed protocol sections are malformed")
    longrun_protocol_path = args.seed42_longrun_protocol.resolve()
    load_deform_dlo_longrun_protocol(longrun_protocol_path)
    companion = parents["seed42_longrun_protocol"]
    if not isinstance(companion, Mapping) or sha256_file(
        longrun_protocol_path
    ) != companion.get("sha256"):
        raise ValueError("two-seed companion protocol differs")

    seed42_result_path = args.seed42_longrun_result.resolve()
    seed42_result = _read_json(seed42_result_path)
    posterior_runtime._validate_longrun_result(
        seed42_result,
        protocol_sha256=sha256_file(longrun_protocol_path),
    )
    seed43_result_path = args.seed43_source_result.resolve()
    seed43_result = _read_json(seed43_result_path)
    seed43_protocol_path = args.seed43_source_protocol.resolve()
    seed43_protocol = source_runtime.load_deform_dlo_source_protocol(
        seed43_protocol_path
    )
    seed43_protocol_identity = parents["seed43_source_protocol"]
    seed43_candidate = seed43_protocol.get("deep_ensemble_candidate")
    seed43_policy = (
        {
            key: value
            for key, value in seed43_candidate.items()
            if key != "companion_seed42_protocol"
        }
        if isinstance(seed43_candidate, Mapping)
        else None
    )
    if (
        not isinstance(seed43_protocol_identity, Mapping)
        or sha256_file(seed43_protocol_path) != seed43_protocol_identity.get("sha256")
        or seed43_policy != policy
    ):
        raise ValueError("seed-43 source protocol differs from ensemble lock")
    source_belief_runtime._validate_source_result(
        seed43_result,
        source_protocol_sha256=sha256_file(seed43_protocol_path),
        upstream_commit=str(upstream_policy["commit"]),
    )
    if _runtime_identity(seed42_result) != _runtime_identity(seed43_result):
        raise ValueError("two-seed runtimes differ")

    seed42_manifest_path = args.seed42_source_manifest.resolve()
    seed42_manifest_identity = parents["seed42_source_manifest"]
    if not isinstance(seed42_manifest_identity, Mapping):
        raise ValueError("seed-42 source manifest lock is malformed")
    seed42_manifest = _verified_manifest(
        seed42_manifest_path,
        expected_sha256=seed42_manifest_identity.get("sha256"),
    )
    seed43_manifest_identity = seed43_result.get("source_manifest")
    if not isinstance(seed43_manifest_identity, Mapping):
        raise ValueError("seed-43 result omits source manifest")
    seed43_manifest_path = Path(str(seed43_manifest_identity.get("path", ""))).resolve()
    seed43_manifest = _verified_manifest(
        seed43_manifest_path,
        expected_sha256=seed43_manifest_identity.get("sha256"),
    )
    validate_deform_two_seed_manifests(seed42_manifest, seed43_manifest)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        str(upstream_policy["commit"]),
    )

    cublas_config = str(evaluation["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("two-seed cuBLAS configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    runtime_torch, runtime_cuda = _runtime_identity(seed42_result)
    if torch.__version__ != runtime_torch or torch.version.cuda != runtime_cuda:
        raise RuntimeError("two-seed evaluator runtime differs")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, 20260731)
    selected_updates = {}
    states = {}
    for seed, result in ((42, seed42_result), (43, seed43_result)):
        selected_updates[seed], states[seed] = _selected_state(result, torch=torch)

    frame_count = int(evaluation["frame_count"])
    node_count = int(evaluation["node_count"])
    validation_names = list(seed42_manifest["split"]["validation"])
    validation_trajectories = source_runtime._load_named_trajectories(
        seed42_manifest,
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
            dlo_type="DLO1",
            node_count=node_count,
        )
        if reference_validation is None:
            reference_validation = rollout
        else:
            posterior_runtime._assert_common_rollout(reference_validation, rollout)
        validation_rollouts[seed] = rollout
    if reference_validation is None:
        raise RuntimeError("two-seed validation rollout is empty")

    member_validation_records = {
        seed: posterior_runtime._records(validation_rollouts[seed]) for seed in (42, 43)
    }
    member_validation_errors = {
        seed: _mean_error(member_validation_records[seed]) for seed in (42, 43)
    }
    baseline_seed = min(
        member_validation_errors,
        key=lambda seed: (member_validation_errors[seed], seed),
    )
    validation_errors = {
        "selected_single": member_validation_errors[baseline_seed],
    }
    candidate_specs = {}
    candidate_validation = {}
    candidate_variance = {}
    weights_by_arm = build_deform_two_seed_weights(member_validation_errors, policy)
    member_validation_predictions = {
        seed: validation_rollouts[seed]["predictions"] for seed in (42, 43)
    }
    for name, weights in weights_by_arm.items():
        prediction, variance = combine_deform_checkpoint_predictions(
            member_validation_predictions,
            weights,
        )
        rollout = {
            "names": reference_validation["names"],
            "predictions": prediction,
            "targets": reference_validation["targets"],
            "persistence": reference_validation["persistence"],
        }
        validation_errors[name] = _mean_error(posterior_runtime._records(rollout))
        candidate_specs[name] = {
            "operator": "predictive_mean",
            "weights": weights,
            "selected_member_updates": selected_updates,
        }
        candidate_validation[name] = rollout
        candidate_variance[name] = variance

    selection = select_deform_checkpoint_belief_arm(
        validation_errors,
        minimum_relative_improvement=float(policy["validation_improvement_min"]),
    )
    selected_arm = str(selection["selected_arm"])
    selection_seal = {
        "schema_version": 1,
        "contract": DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT,
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "seed42_longrun_result": {
            "path": str(seed42_result_path),
            "sha256": sha256_file(seed42_result_path),
        },
        "seed43_source_result": {
            "path": str(seed43_result_path),
            "sha256": sha256_file(seed43_result_path),
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

    baseline_source_records = _stored_source_records(
        seed42_result if baseline_seed == 42 else seed43_result
    )
    uncertainty = None
    if selection["fallback_used"]:
        candidate_source_records = baseline_source_records
        exact_fallback = True
    else:
        exact_fallback = False
        source_names = list(seed42_manifest["split"]["source_test"])
        source_trajectories = source_runtime._load_named_trajectories(
            seed42_manifest,
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
                dlo_type="DLO1",
                node_count=node_count,
            )
            if reference_source is None:
                reference_source = rollout
            else:
                posterior_runtime._assert_common_rollout(reference_source, rollout)
            source_rollouts[seed] = rollout
        if reference_source is None:
            raise RuntimeError("two-seed source rollout is empty")
        weights = candidate_specs[selected_arm]["weights"]
        source_prediction, source_variance = combine_deform_checkpoint_predictions(
            {seed: source_rollouts[seed]["predictions"] for seed in (42, 43)},
            weights,
        )
        combined_source = {
            "names": reference_source["names"],
            "predictions": source_prediction,
            "targets": reference_source["targets"],
            "persistence": reference_source["persistence"],
        }
        candidate_source_records = posterior_runtime._records(combined_source)
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
                raise RuntimeError("two-seed baseline source replay differs")
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
            "claim_boundary": "coordinate-marginal exploratory source diagnostic",
        }

    transfer = evaluate_deform_checkpoint_belief_transfer(
        candidate_source_records,
        baseline_source_records,
        claim_boundary="post-open DLO1 exploratory; fresh DLO2 required",
    )
    authorized = (
        not exact_fallback
        and float(transfer["relative_improvement"])
        >= float(policy["source_transfer_improvement_min"])
        and int(transfer["wins"]) >= int(policy["source_transfer_minimum_case_wins"])
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo-deep-ensemble-result-v1",
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
        },
        "uncertainty": uncertainty,
        "fresh_dlo2_deep_ensemble_authorized": authorized,
        "fresh_confirmation_contract": policy["fresh_confirmation"],
    }
    result_path = output_root / "ensemble_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
