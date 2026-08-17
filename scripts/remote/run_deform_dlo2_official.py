#!/usr/bin/env python3
"""Run the authorized one-shot official DEFORM DLO2 evaluation."""

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
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_alltrain_protocol,
)
from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    combine_deform_checkpoint_predictions,
    weighted_deform_prediction_median,
)
from bayesian_phystwin.deform_dlo_official import (
    DEFORM_CANONICAL_REFERENCE_DRAW,
    evaluate_deform_dlo2_official_uncertainty,
    load_deform_dlo2_official_protocol,
    summarize_deform_dlo2_official_records,
    validate_deform_dlo2_official_authorization,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--alltrain-protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
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


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integer_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) for item in value):
        raise ValueError(f"{label} must be an integer array")
    try:
        return [int(str(item)) for item in value]
    except ValueError as error:
        raise ValueError(f"{label} must be an integer array") from error


def _string_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string array")
    return list(value)


def _verified_json_identity(
    identity: Mapping[str, object], *, label: str
) -> tuple[Path, dict[str, object]]:
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or sha256_file(path) != identity.get("sha256")
        or int(str(identity.get("size_bytes", path.stat().st_size)))
        != path.stat().st_size
    ):
        raise ValueError(f"{label} identity does not verify")
    return path, _read_json(path)


def _verified_file_identity(identity: Mapping[str, object], *, label: str) -> Path:
    path = Path(str(identity.get("path", ""))).resolve()
    default_size = path.stat().st_size if path.exists() else -1
    expected_size = int(str(identity.get("size_bytes", default_size)))
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity does not verify")
    return path


def _verified_checkpoint_bundle(
    identity: Mapping[str, object],
    *,
    torch: Any,
    alltrain_protocol_sha256: str,
    schedule_sha256: str,
    method_spec_sha256: str,
    expected_update: int | None,
) -> dict[str, Any]:
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(str(identity.get("size_bytes", -1)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError("all-train checkpoint identity does not verify")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or not isinstance(
        bundle.get("model_state_dict"), dict
    ):
        raise ValueError("all-train checkpoint omits its model state")
    if (
        bundle.get("alltrain_protocol_sha256") != alltrain_protocol_sha256
        or bundle.get("schedule_sha256") != schedule_sha256
        or bundle.get("method_spec_sha256") != method_spec_sha256
    ):
        raise ValueError("all-train checkpoint lineage differs")
    if expected_update is not None and int(bundle.get("update", -1)) != expected_update:
        raise ValueError("all-train checkpoint update differs")
    return bundle


def _build_eval_manifest(
    eval_root: Path,
    *,
    expected_count: int,
    canonical_reference_draw_indices: list[int],
    protocol_path: Path,
    alltrain_result_path: Path,
) -> dict[str, object]:
    paths = tuple(sorted(eval_root.glob("*.pkl"), key=lambda path: path.name))
    if len(paths) != expected_count:
        raise ValueError(
            f"official DLO2 evaluation expected {expected_count} trajectories, "
            f"got {len(paths)}"
        )
    identities = {
        path.name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    ordered_names = list(identities)
    if (
        tuple(canonical_reference_draw_indices) != DEFORM_CANONICAL_REFERENCE_DRAW
        or len(canonical_reference_draw_indices) != expected_count
        or any(
            index < 0 or index >= expected_count
            for index in canonical_reference_draw_indices
        )
    ):
        raise ValueError("canonical reference draw is invalid")
    return {
        "schema_version": 2,
        "contract": "deform-dlo2-official-eval-manifest-v2",
        "official_eval_read": True,
        "outcomes_evaluated": False,
        "partition": "eval",
        "trajectory_policy": (
            "all-eval-files-sorted-once-plus-canonical-reference-draw-v2"
        ),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "alltrain_result": {
            "path": str(alltrain_result_path.resolve()),
            "sha256": sha256_file(alltrain_result_path),
        },
        "trajectories": identities,
        "ordered_names": ordered_names,
        "canonical_reference_draw_indices": canonical_reference_draw_indices,
        "canonical_reference_draw_names": [
            ordered_names[index] for index in canonical_reference_draw_indices
        ],
        "canonical_reference_unique_count": len(set(canonical_reference_draw_indices)),
    }


def _failure_payload(*, stage: str, error: BaseException) -> dict[str, object]:
    return {
        "schema_version": 2,
        "contract": "deform-dlo2-official-eval-failure-v2",
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
    source_protocol_path = args.source_protocol.resolve()
    alltrain_result_path = args.alltrain_result.resolve()
    protocol = load_deform_dlo2_official_protocol(protocol_path)
    alltrain_protocol = load_deform_dlo2_alltrain_protocol(alltrain_protocol_path)
    source_protocol = load_deform_dlo_source_protocol(source_protocol_path)
    alltrain_protocol_sha256 = sha256_file(alltrain_protocol_path)
    source_protocol_sha256 = sha256_file(source_protocol_path)
    official_parent = _mapping(
        protocol["parent_alltrain_protocol"], label="official parent protocol"
    )
    alltrain_parent = _mapping(
        alltrain_protocol["parent_source_protocol"], label="all-train parent protocol"
    )
    evaluation = _mapping(protocol["evaluation"], label="evaluation")
    if (
        alltrain_protocol_sha256 != official_parent["sha256"]
        or source_protocol_sha256 != alltrain_parent["sha256"]
        or source_protocol["dlo_types"] != ("DLO2",)
    ):
        raise ValueError("official evaluator protocol lineage differs")

    alltrain_result = _read_json(alltrain_result_path)
    final_identity = _mapping(
        alltrain_result.get("final_method"), label="final-method identity"
    )
    method_spec_identity = _mapping(
        alltrain_result.get("method_spec"), label="method-spec identity"
    )
    final_method_path, final_method = _verified_json_identity(
        final_identity, label="final method"
    )
    method_spec_path, method_spec = _verified_json_identity(
        method_spec_identity, label="method specification"
    )
    selected = validate_deform_dlo2_official_authorization(
        protocol,
        alltrain_protocol,
        alltrain_result,
        final_method,
        method_spec,
        alltrain_protocol_sha256=alltrain_protocol_sha256,
        alltrain_result_sha256=sha256_file(alltrain_result_path),
        final_method_sha256=sha256_file(final_method_path),
        method_spec_sha256=sha256_file(method_spec_path),
    )
    upstream_commit = str(source_protocol["upstream"]["commit"])
    upstream = source_runtime._assert_upstream(args.upstream_root, upstream_commit)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"one-shot output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    authorization = {
        "schema_version": 2,
        "contract": "deform-dlo2-official-eval-authorization-v2",
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
        "source_protocol": {
            "path": str(source_protocol_path),
            "sha256": source_protocol_sha256,
        },
        "alltrain_result": {
            "path": str(alltrain_result_path),
            "sha256": sha256_file(alltrain_result_path),
        },
        "final_method": {
            "path": str(final_method_path),
            "sha256": sha256_file(final_method_path),
        },
        "method_spec": {
            "path": str(method_spec_path),
            "sha256": sha256_file(method_spec_path),
        },
        "selected_operator": selected["operator"],
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "upstream": upstream,
    }
    _write_json(output_root / "authorization.json", authorization)

    cublas_config = str(alltrain_protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    runtime = _mapping(selected["runtime"], label="selected runtime")
    if torch.__version__ != runtime.get("torch") or torch.version.cuda != runtime.get(
        "cuda"
    ):
        raise RuntimeError("official evaluator runtime differs from all-train refit")
    source_runtime._seed_everything(
        torch, int(alltrain_protocol["training"]["random_seed"])
    )
    schedule = _mapping(
        alltrain_result.get("window_schedule"), label="all-train schedule"
    )
    _verified_file_identity(schedule, label="all-train schedule")
    schedule_sha256 = str(schedule.get("sha256", ""))
    method_spec_sha256 = sha256_file(method_spec_path)
    bundles: dict[str, dict[str, Any]] = {}

    def verify_checkpoint(
        identity: Mapping[str, object], *, expected_update: int | None
    ) -> None:
        digest = str(identity.get("sha256", ""))
        if digest in bundles:
            return
        bundles[digest] = _verified_checkpoint_bundle(
            identity,
            torch=torch,
            alltrain_protocol_sha256=alltrain_protocol_sha256,
            schedule_sha256=schedule_sha256,
            method_spec_sha256=method_spec_sha256,
            expected_update=expected_update,
        )

    baseline_identity = _mapping(
        selected["comparison_baseline_checkpoint"], label="comparison checkpoint"
    )
    verify_checkpoint(
        baseline_identity,
        expected_update=int(str(baseline_identity["update"])),
    )
    member_identities = _mapping(
        selected["member_checkpoints"], label="member checkpoints"
    )
    for update_text, identity in member_identities.items():
        verify_checkpoint(
            _mapping(identity, label="member checkpoint"),
            expected_update=int(update_text),
        )
    parameter_identity = selected["parameter_mean_checkpoint"]
    if isinstance(parameter_identity, dict):
        verify_checkpoint(parameter_identity, expected_update=None)

    modules = source_runtime._load_upstream(args.upstream_root)
    eval_root = args.upstream_root.resolve() / "data_set" / "DLO2" / "eval"
    stage = "target-manifest"
    started = time.perf_counter()
    try:
        reference_operator = _mapping(
            evaluation["published_reference_operator"],
            label="published reference operator",
        )
        reference_draw = _integer_list(
            reference_operator["canonical_eval_indices"],
            label="canonical reference draw",
        )
        manifest = _build_eval_manifest(
            eval_root,
            expected_count=int(str(evaluation["expected_trajectory_count"])),
            canonical_reference_draw_indices=reference_draw,
            protocol_path=protocol_path,
            alltrain_result_path=alltrain_result_path,
        )
        manifest_path = output_root / "evaluation_manifest.json"
        _write_json(manifest_path, manifest)
        stage = "target-load"
        names = _string_list(manifest["ordered_names"], label="evaluation names")
        trajectories = source_runtime._load_named_trajectories(
            manifest,
            names,
            frame_count=int(str(evaluation["expected_frame_count"])),
            node_count=int(str(evaluation["expected_node_count"])),
        )

        stage = "fixed-rollouts"
        rollout_cache: dict[str, dict[str, object]] = {}

        def rollout(identity: Mapping[str, object]) -> dict[str, object]:
            digest = str(identity["sha256"])
            if digest not in rollout_cache:
                rollout_cache[digest] = posterior_runtime._evaluate_state(
                    bundles[digest]["model_state_dict"],
                    trajectories,
                    modules=modules,
                    torch=torch,
                    device=args.device,
                    dlo_type="DLO2",
                    node_count=int(str(evaluation["expected_node_count"])),
                )
            return rollout_cache[digest]

        baseline_rollout = rollout(baseline_identity)
        member_rollouts: dict[int, dict[str, object]] = {}
        for member_update, identity in sorted(
            (
                (int(key), _mapping(value, label="member checkpoint"))
                for key, value in member_identities.items()
            )
        ):
            member_rollout = rollout(identity)
            posterior_runtime._assert_common_rollout(baseline_rollout, member_rollout)
            member_rollouts[member_update] = member_rollout
        posterior_mean, posterior_variance = combine_deform_checkpoint_predictions(
            {
                update: member_rollouts[update]["predictions"]
                for update in member_rollouts
            },
            selected["weights"],
        )
        if selected["operator"] == "predictive_mean":
            candidate_prediction = posterior_mean
        elif selected["operator"] == "predictive_median":
            candidate_prediction = weighted_deform_prediction_median(
                {
                    update: member_rollouts[update]["predictions"]
                    for update in member_rollouts
                },
                selected["weights"],
            )
        else:
            if not isinstance(parameter_identity, dict):
                raise RuntimeError("parameter-mean checkpoint disappeared")
            parameter_rollout = rollout(parameter_identity)
            posterior_runtime._assert_common_rollout(
                baseline_rollout, parameter_rollout
            )
            candidate_prediction = parameter_rollout["predictions"]

        candidate_rollout = {
            "names": baseline_rollout["names"],
            "predictions": candidate_prediction,
            "targets": baseline_rollout["targets"],
            "persistence": baseline_rollout["persistence"],
        }
        candidate_records = posterior_runtime._records(candidate_rollout)
        baseline_records = posterior_runtime._records(baseline_rollout)
        gate_config = _mapping(protocol["claim_gate"], label="claim gate")
        summary = summarize_deform_dlo2_official_records(
            candidate_records,
            baseline_records,
            expected_case_count=int(str(evaluation["expected_trajectory_count"])),
            published_reference_l1_m=float(str(evaluation["published_reference_l1_m"])),
            minimum_relative_improvement=float(
                str(gate_config["bayesian_relative_improvement_min"])
            ),
            minimum_case_wins=int(str(gate_config["bayesian_minimum_case_wins"])),
            canonical_reference_draw_indices=reference_draw,
        )
        uncertainty = evaluate_deform_dlo2_official_uncertainty(
            candidate_prediction,
            np.asarray(baseline_rollout["targets"]),
            posterior_variance,
            variance_floor_m2=float(str(selected["variance_floor_m2"])),
            variance_scale=float(str(selected["variance_scale"])),
            nominal_coverage=float(str(selected["nominal_coordinate_coverage"])),
        )
        result = {
            "schema_version": 2,
            "contract": "deform-dlo2-official-eval-result-v2",
            "claim_boundary": protocol["claim_boundary"],
            "official_eval_read": True,
            "target_selection_performed": False,
            "target_calibration_performed": False,
            "target_retry_performed": False,
            "all_expected_cases_evaluated_once": True,
            "protocol": authorization["protocol"],
            "authorization": {
                "path": str((output_root / "authorization.json").resolve()),
                "sha256": sha256_file(output_root / "authorization.json"),
            },
            "evaluation_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "operator": selected["operator"],
            "checkpoint_weights": selected["weights"],
            "comparison": summary,
            "candidate_cases": candidate_records,
            "comparison_baseline_cases": baseline_records,
            "uncertainty": {
                "source_validation_scale_reused_unchanged": True,
                "variance_scale": selected["variance_scale"],
                "variance_floor_m2": selected["variance_floor_m2"],
                "nominal_coordinate_coverage": selected["nominal_coordinate_coverage"],
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
        failure_path = output_root / "official_failure.json"
        _write_json(failure_path, _failure_payload(stage=stage, error=error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
