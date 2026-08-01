#!/usr/bin/env python3
"""Run the authorized one-shot two-seed DEFORM DLO2 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_dlo2_official as official_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_deep_alltrain_protocol,
    validate_deform_dlo2_deep_alltrain_authorization,
)
from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    combine_deform_checkpoint_predictions,
)
from bayesian_phystwin.deform_dlo_official import (
    evaluate_deform_dlo2_official_uncertainty,
    load_deform_dlo2_deep_official_protocol,
    summarize_deform_dlo2_official_records,
    validate_deform_dlo2_deep_official_authorization,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--alltrain-protocol", type=Path, required=True)
    parser.add_argument("--seed42-source-protocol", type=Path, required=True)
    parser.add_argument("--seed43-source-protocol", type=Path, required=True)
    parser.add_argument("--alltrain-result", type=Path, required=True)
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


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _verified_json(
    identity: Mapping[str, object], *, label: str
) -> tuple[Path, dict[str, object]]:
    path = Path(str(identity.get("path", ""))).resolve()
    default_size = path.stat().st_size if path.is_file() else -1
    if (
        not path.is_file()
        or path.stat().st_size
        != int(str(identity.get("size_bytes", default_size)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity does not verify")
    return path, _read_json(path)


def _verified_file(identity: Mapping[str, object], *, label: str) -> Path:
    path = Path(str(identity.get("path", ""))).resolve()
    default_size = path.stat().st_size if path.is_file() else -1
    if (
        not path.is_file()
        or path.stat().st_size
        != int(str(identity.get("size_bytes", default_size)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity does not verify")
    return path


def _verified_deep_checkpoint(
    identity: Mapping[str, object],
    *,
    seed: int,
    update: int,
    alltrain_protocol_sha256: str,
    schedule_sha256: str,
    method_spec_sha256: str,
    torch: Any,
) -> dict[str, Any]:
    path = _verified_file(identity, label=f"seed-{seed} checkpoint")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(bundle, dict)
        or not isinstance(bundle.get("model_state_dict"), dict)
        or int(bundle.get("seed", -1)) != seed
        or int(bundle.get("update", -1)) != update
        or bundle.get("deep_alltrain_protocol_sha256")
        != alltrain_protocol_sha256
        or bundle.get("schedule_sha256") != schedule_sha256
        or bundle.get("method_spec_sha256") != method_spec_sha256
    ):
        raise ValueError(f"seed-{seed} checkpoint lineage differs")
    return bundle


def _failure_payload(*, stage: str, error: BaseException) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-official-eval-failure-v1",
        "official_eval_read": True,
        "retry_authorized": False,
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
    }


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    alltrain_protocol_path = args.alltrain_protocol.resolve()
    alltrain_result_path = args.alltrain_result.resolve()
    source_protocol_paths = {
        42: args.seed42_source_protocol.resolve(),
        43: args.seed43_source_protocol.resolve(),
    }
    protocol = load_deform_dlo2_deep_official_protocol(protocol_path)
    alltrain_protocol = load_deform_dlo2_deep_alltrain_protocol(
        alltrain_protocol_path
    )
    alltrain_protocol_sha256 = sha256_file(alltrain_protocol_path)
    if (
        alltrain_protocol_sha256
        != protocol["parent_alltrain_protocol"]["sha256"]
    ):
        raise ValueError("deep official evaluator binds another all-train protocol")

    source_protocols = {}
    source_protocol_sha256s = {}
    for seed in (42, 43):
        source_protocols[seed] = load_deform_dlo_source_protocol(
            source_protocol_paths[seed]
        )
        source_protocol_sha256s[seed] = sha256_file(source_protocol_paths[seed])
        if (
            source_protocol_sha256s[seed]
            != alltrain_protocol["parents"][f"seed{seed}_source_protocol"][
                "sha256"
            ]
            or source_protocols[seed]["dlo_types"] != ("DLO2",)
            or int(source_protocols[seed]["training"]["random_seed"]) != seed
        ):
            raise ValueError(f"seed-{seed} source protocol lineage differs")
    upstream_commits = {
        str(source_protocols[seed]["upstream"]["commit"]) for seed in (42, 43)
    }
    if len(upstream_commits) != 1:
        raise ValueError("deep source members bind different upstream commits")

    alltrain_result = _read_json(alltrain_result_path)
    final_identity = _mapping(
        alltrain_result.get("final_method"), label="final-method identity"
    )
    final_method_path, final_method = _verified_json(
        final_identity, label="final method"
    )
    selected = validate_deform_dlo2_deep_official_authorization(
        protocol,
        alltrain_protocol,
        alltrain_result,
        final_method,
        alltrain_protocol_sha256=alltrain_protocol_sha256,
        alltrain_result_sha256=sha256_file(alltrain_result_path),
        final_method_sha256=sha256_file(final_method_path),
    )

    seed_result_paths = {}
    seed_results = {}
    method_specs = {}
    final_members = {}
    schedules = {}
    ensemble_identity = None
    source_result_identities = None
    for seed in (42, 43):
        seed_identity = _mapping(
            selected["seed_results"][seed], label=f"seed-{seed} result identity"
        )
        seed_result_paths[seed], seed_result = _verified_json(
            seed_identity, label=f"seed-{seed} all-train result"
        )
        seed_results[seed] = seed_result
        protocol_identity = _mapping(
            seed_result.get("protocol"), label=f"seed-{seed} protocol identity"
        )
        method_identity = _mapping(
            seed_result.get("method_spec"), label=f"seed-{seed} method identity"
        )
        member_identity = _mapping(
            seed_result.get("final_member"), label=f"seed-{seed} member identity"
        )
        schedule_identity = _mapping(
            seed_result.get("window_schedule"),
            label=f"seed-{seed} schedule identity",
        )
        if (
            seed_result.get("contract")
            != "deform-dlo2-deep-alltrain-seed-result-v1"
            or seed_result.get("official_eval_read") is not False
            or seed_result.get("assembly_authorized") is not True
            or int(seed_result.get("seed", -1)) != seed
            or protocol_identity.get("sha256") != alltrain_protocol_sha256
        ):
            raise ValueError(f"seed-{seed} all-train result differs")
        method_path, method_spec = _verified_json(
            method_identity, label=f"seed-{seed} method specification"
        )
        _, final_member = _verified_json(
            member_identity, label=f"seed-{seed} final member"
        )
        schedule_path = _verified_file(
            schedule_identity, label=f"seed-{seed} schedule"
        )
        method_specs[seed] = (method_path, method_spec)
        final_members[seed] = final_member
        schedules[seed] = schedule_path
        checkpoint = _mapping(
            final_member.get("selected_checkpoint"),
            label=f"seed-{seed} selected checkpoint",
        )
        if (
            method_spec.get("contract")
            != "deform-dlo2-deep-alltrain-seed-method-v1"
            or method_spec.get("official_eval_read") is not False
            or int(method_spec.get("seed", -1)) != seed
            or method_spec.get("operator") != selected["operator"]
            or float(method_spec.get("seed_weight", -1.0))
            != selected["weights"][seed]
            or int(method_spec.get("selected_update", -1))
            != selected["member_updates"][seed]
            or int(method_spec.get("comparison_baseline_seed", -1))
            != selected["comparison_baseline_seed"]
            or final_member.get("contract")
            != "deform-dlo2-deep-alltrain-seed-final-v1"
            or final_member.get("official_eval_read") is not False
            or int(final_member.get("seed", -1)) != seed
            or final_member.get("operator") != selected["operator"]
            or float(final_member.get("weight", -1.0))
            != selected["weights"][seed]
            or int(final_member.get("selected_update", -1))
            != selected["member_updates"][seed]
            or checkpoint != selected["member_checkpoints"][seed]
            or _mapping(
                final_member.get("method_spec"),
                label=f"seed-{seed} final member method",
            ).get("sha256")
            != sha256_file(method_path)
            or _mapping(
                final_member.get("window_schedule"),
                label=f"seed-{seed} final member schedule",
            ).get("sha256")
            != sha256_file(schedule_path)
        ):
            raise ValueError(f"seed-{seed} frozen method differs")
        candidate_ensemble = _mapping(
            method_spec.get("ensemble_result"),
            label=f"seed-{seed} ensemble identity",
        )
        seed_ensemble = _mapping(
            seed_result.get("ensemble_result"),
            label=f"seed-{seed} result ensemble identity",
        )
        method_selection = _mapping(
            method_spec.get("selection_seal"),
            label=f"seed-{seed} selection identity",
        )
        method_calibration = _mapping(
            method_spec.get("variance_calibration"),
            label=f"seed-{seed} variance calibration",
        )
        expected_calibration = {
            "scale": selected["variance_scale"],
            "floor_m2": selected["variance_floor_m2"],
            "nominal_coordinate_coverage": selected[
                "nominal_coordinate_coverage"
            ],
        }
        if (
            dict(seed_ensemble) != dict(candidate_ensemble)
            or method_calibration != expected_calibration
            or len(str(method_selection.get("sha256", ""))) != 64
        ):
            raise ValueError(f"seed-{seed} source-selection lineage differs")
        candidate_sources = _mapping(
            method_spec.get("source_results"),
            label=f"seed-{seed} source identities",
        )
        if ensemble_identity is None:
            ensemble_identity = dict(candidate_ensemble)
            source_result_identities = dict(candidate_sources)
        elif (
            dict(candidate_ensemble) != ensemble_identity
            or dict(candidate_sources) != source_result_identities
        ):
            raise ValueError("deep all-train members bind different source selection")

    if ensemble_identity is None or source_result_identities is None:
        raise RuntimeError("deep all-train source selection is absent")
    ensemble_result_path, ensemble_result = _verified_json(
        ensemble_identity, label="DLO2 ensemble result"
    )
    assembled_ensemble = _mapping(
        alltrain_result.get("ensemble_result"),
        label="assembled ensemble identity",
    )
    if dict(assembled_ensemble) != ensemble_identity:
        raise ValueError("assembled result binds another ensemble selection")
    selection_identity = _mapping(
        ensemble_result.get("selection_seal"), label="ensemble selection seal"
    )
    selection_path, selection_seal = _verified_json(
        selection_identity, label="ensemble selection seal"
    )
    for seed in (42, 43):
        method_selection = _mapping(
            method_specs[seed][1].get("selection_seal"),
            label=f"seed-{seed} selection identity",
        )
        if dict(method_selection) != dict(selection_identity):
            raise ValueError("deep all-train members bind another selection seal")
    source_results = {}
    source_result_sha256s = {}
    for seed in (42, 43):
        identity = _mapping(
            source_result_identities.get(str(seed)),
            label=f"source seed-{seed} result identity",
        )
        _, source_results[seed] = _verified_json(
            identity, label=f"source seed-{seed} result"
        )
        source_result_sha256s[seed] = str(identity.get("sha256", ""))
    source_selected = validate_deform_dlo2_deep_alltrain_authorization(
        alltrain_protocol,
        source_results,
        ensemble_result,
        selection_seal,
        source_protocol_sha256s=source_protocol_sha256s,
        source_result_sha256s=source_result_sha256s,
        ensemble_protocol_sha256=str(
            alltrain_protocol["parents"]["ensemble_protocol"]["sha256"]
        ),
        selection_seal_sha256=sha256_file(selection_path),
    )
    if (
        source_selected["operator"] != selected["operator"]
        or source_selected["weights"] != selected["weights"]
        or source_selected["member_updates"] != selected["member_updates"]
        or source_selected["comparison_baseline_seed"]
        != selected["comparison_baseline_seed"]
        or source_selected["validation_fitted_variance_scale"]
        != selected["variance_scale"]
        or source_selected["variance_floor_m2"]
        != selected["variance_floor_m2"]
        or source_selected["nominal_coordinate_coverage"]
        != selected["nominal_coordinate_coverage"]
    ):
        raise ValueError("assembled method differs from fresh source selection")

    upstream = source_runtime._assert_upstream(
        args.upstream_root, upstream_commits.pop()
    )
    cublas_config = str(alltrain_protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("deep official cuBLAS configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    runtime = _mapping(selected["runtime"], label="selected runtime")
    if torch.__version__ != runtime.get("torch") or torch.version.cuda != runtime.get(
        "cuda"
    ):
        raise RuntimeError("deep official runtime differs from all-train refit")
    bundles = {}
    for seed in (42, 43):
        method_path, _ = method_specs[seed]
        identity = _mapping(
            selected["member_checkpoints"][seed],
            label=f"seed-{seed} checkpoint identity",
        )
        bundles[seed] = _verified_deep_checkpoint(
            identity,
            seed=seed,
            update=selected["member_updates"][seed],
            alltrain_protocol_sha256=alltrain_protocol_sha256,
            schedule_sha256=sha256_file(schedules[seed]),
            method_spec_sha256=sha256_file(method_path),
            torch=torch,
        )

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"one-shot output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    authorization = {
        "schema_version": 1,
        "contract": "deform-dlo2-deep-official-eval-authorization-v1",
        "official_eval_read": False,
        "one_shot_execution_authorized": True,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "alltrain_protocol": {
            "path": str(alltrain_protocol_path),
            "sha256": alltrain_protocol_sha256,
        },
        "alltrain_result": {
            "path": str(alltrain_result_path),
            "sha256": sha256_file(alltrain_result_path),
        },
        "final_method": {
            "path": str(final_method_path),
            "sha256": sha256_file(final_method_path),
        },
        "ensemble_result": {
            "path": str(ensemble_result_path),
            "sha256": sha256_file(ensemble_result_path),
        },
        "source_protocols": {
            str(seed): {
                "path": str(source_protocol_paths[seed]),
                "sha256": source_protocol_sha256s[seed],
            }
            for seed in (42, 43)
        },
        "selected_operator": selected["operator"],
        "seed_weights": selected["weights"],
        "member_updates": selected["member_updates"],
        "comparison_baseline_seed": selected["comparison_baseline_seed"],
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "upstream": upstream,
    }
    authorization_path = output_root / "authorization.json"
    _write_json(authorization_path, authorization)

    evaluation = _mapping(protocol["evaluation"], label="evaluation")
    gate = _mapping(protocol["claim_gate"], label="claim gate")
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(torch, 20260801)
    eval_root = args.upstream_root.resolve() / "data_set" / "DLO2" / "eval"
    stage = "target-manifest"
    started = time.perf_counter()
    try:
        reference_operator = _mapping(
            evaluation["published_reference_operator"],
            label="published reference operator",
        )
        reference_draw = official_runtime._integer_list(
            reference_operator["canonical_eval_indices"],
            label="canonical reference draw",
        )
        manifest = official_runtime._build_eval_manifest(
            eval_root,
            expected_count=int(str(evaluation["expected_trajectory_count"])),
            canonical_reference_draw_indices=reference_draw,
            protocol_path=protocol_path,
            alltrain_result_path=alltrain_result_path,
        )
        manifest_path = output_root / "evaluation_manifest.json"
        _write_json(manifest_path, manifest)
        stage = "target-load"
        names = official_runtime._string_list(
            manifest["ordered_names"], label="evaluation names"
        )
        trajectories = source_runtime._load_named_trajectories(
            manifest,
            names,
            frame_count=int(str(evaluation["expected_frame_count"])),
            node_count=int(str(evaluation["expected_node_count"])),
        )

        stage = "fixed-rollouts"
        rollouts = {}
        reference_rollout = None
        for seed in (42, 43):
            rollout = posterior_runtime._evaluate_state(
                bundles[seed]["model_state_dict"],
                trajectories,
                modules=modules,
                torch=torch,
                device=args.device,
                dlo_type="DLO2",
                node_count=int(str(evaluation["expected_node_count"])),
            )
            if reference_rollout is None:
                reference_rollout = rollout
            else:
                posterior_runtime._assert_common_rollout(reference_rollout, rollout)
            rollouts[seed] = rollout
        if reference_rollout is None:
            raise RuntimeError("deep official rollout is empty")
        candidate_prediction, raw_variance = (
            combine_deform_checkpoint_predictions(
                {seed: rollouts[seed]["predictions"] for seed in (42, 43)},
                selected["weights"],
            )
        )
        candidate_records = posterior_runtime._records(
            {
                "names": reference_rollout["names"],
                "predictions": candidate_prediction,
                "targets": reference_rollout["targets"],
                "persistence": reference_rollout["persistence"],
            }
        )
        baseline_rollout = rollouts[selected["comparison_baseline_seed"]]
        baseline_records = posterior_runtime._records(baseline_rollout)
        summary = summarize_deform_dlo2_official_records(
            candidate_records,
            baseline_records,
            expected_case_count=int(str(evaluation["expected_trajectory_count"])),
            published_reference_l1_m=float(
                str(evaluation["published_reference_l1_m"])
            ),
            minimum_relative_improvement=float(
                str(gate["ensemble_relative_improvement_min"])
            ),
            minimum_case_wins=int(str(gate["ensemble_minimum_case_wins"])),
            canonical_reference_draw_indices=reference_draw,
        )
        uncertainty = evaluate_deform_dlo2_official_uncertainty(
            candidate_prediction,
            np.asarray(reference_rollout["targets"]),
            raw_variance,
            variance_floor_m2=float(str(selected["variance_floor_m2"])),
            variance_scale=float(str(selected["variance_scale"])),
            nominal_coverage=float(
                str(selected["nominal_coordinate_coverage"])
            ),
        )
        result = {
            "schema_version": 1,
            "contract": "deform-dlo2-deep-official-eval-result-v1",
            "claim_boundary": protocol["claim_boundary"],
            "official_eval_read": True,
            "target_selection_performed": False,
            "target_calibration_performed": False,
            "target_retry_performed": False,
            "all_expected_cases_evaluated_once": True,
            "protocol": authorization["protocol"],
            "authorization": {
                "path": str(authorization_path),
                "sha256": sha256_file(authorization_path),
            },
            "evaluation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "operator": selected["operator"],
            "seed_weights": selected["weights"],
            "member_updates": selected["member_updates"],
            "comparison_baseline_seed": selected["comparison_baseline_seed"],
            "comparison": summary,
            "candidate_cases": candidate_records,
            "comparison_baseline_cases": baseline_records,
            "uncertainty": {
                "source_validation_scale_reused_unchanged": True,
                "variance_scale": selected["variance_scale"],
                "variance_floor_m2": selected["variance_floor_m2"],
                "nominal_coordinate_coverage": selected[
                    "nominal_coordinate_coverage"
                ],
                "metrics": uncertainty,
            },
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
        result_path = output_root / "official_result.json"
        _write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        _write_json(
            output_root / "official_failure.json",
            _failure_payload(stage=stage, error=error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
