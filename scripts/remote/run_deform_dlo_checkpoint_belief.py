#!/usr/bin/env python3
"""Evaluate frozen DEFORM checkpoint averaging without opening official eval."""

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
    evaluate_deform_checkpoint_belief_transfer,
    load_deform_checkpoint_belief_protocol,
    select_deform_checkpoint_belief_arm,
)
from bayesian_phystwin.deform_dlo_source import (
    choose_deform_validation_checkpoint,
    load_deform_dlo_source_protocol,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
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


def _validate_source_result(
    result: dict[str, object],
    *,
    source_protocol_sha256: str,
    upstream_commit: str,
) -> None:
    if result.get("contract") != "deform-dlo-source-reproduction-result-v1":
        raise ValueError("checkpoint belief requires the frozen source result")
    if result.get("official_eval_read") is not False:
        raise ValueError("source result crossed the official-eval boundary")
    gate = result.get("source_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or result.get("advancement_authorized") is not True
    ):
        raise ValueError("source reproduction did not authorize advancement")
    upstream = result.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != upstream_commit:
        raise ValueError("source result uses a different DEFORM commit")
    validation = result.get("validation")
    checkpoints = result.get("checkpoints")
    if not isinstance(validation, list) or not isinstance(checkpoints, list):
        raise ValueError("source result omits validation or checkpoints")
    selected = choose_deform_validation_checkpoint(validation)
    if selected != result.get("selected_checkpoint"):
        raise ValueError("source result selected checkpoint is inconsistent")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("source checkpoint identity is malformed")
        path = Path(str(checkpoint.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != checkpoint.get("sha256"):
            raise ValueError("source checkpoint identity does not verify")
        bundle = _torch_load(path, map_location="cpu")
        if bundle.get("protocol_sha256") != source_protocol_sha256:
            raise ValueError("source checkpoint protocol identity differs")
        if bundle.get("schedule_sha256") != result["window_schedule"]["sha256"]:
            raise ValueError("source checkpoint schedule identity differs")
        if int(bundle.get("update", -1)) != int(checkpoint.get("update", -2)):
            raise ValueError("source checkpoint update identity differs")


def _torch_load(path: Path, *, map_location: str) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _checkpoint_states(
    source_result: dict[str, object],
    required_updates: set[int],
) -> dict[int, dict[str, Any]]:
    records = source_result["checkpoints"]
    indexed = {int(record["update"]): record for record in records}
    if len(indexed) != len(records) or not required_updates.issubset(indexed):
        raise ValueError("source result does not contain all checkpoint updates")
    result = {}
    for update in sorted(required_updates):
        bundle = _torch_load(Path(indexed[update]["path"]), map_location="cpu")
        state = bundle.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("source checkpoint omits its model state")
        result[update] = state
    return result


def _evaluate_state(
    state: dict[str, Any],
    trajectories: dict[str, np.ndarray],
    *,
    modules: Any,
    torch: Any,
    device: str,
) -> list[dict[str, object]]:
    model_function, model = source_runtime._build_dlo1_model(modules, torch, device)
    model.load_state_dict(state, strict=True)
    return source_runtime._rollout_records(
        trajectories,
        modules=modules,
        model_function=model_function,
        model=model,
        torch=torch,
        device=device,
    )


def _mean_model_l1(records: list[dict[str, object]]) -> float:
    errors = np.asarray([float(record["model_l1_m"]) for record in records])
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("checkpoint-belief rollout errors are invalid")
    return float(np.mean(errors))


def main() -> int:
    args = _parse_args()
    protocol = load_deform_checkpoint_belief_protocol(args.protocol)
    source_protocol = load_deform_dlo_source_protocol(args.source_protocol)
    if (
        protocol["source_reproduction_commit"]
        != "4ef71f16b909a8db7b60f08047010250f0b765b1"
    ):
        raise ValueError("checkpoint-belief source commit is not frozen v1")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_result_path = args.source_result.resolve()
    source_result = _read_json(source_result_path)
    source_protocol_sha256 = sha256_file(args.source_protocol)
    upstream_commit = str(source_protocol["upstream"]["commit"])
    _validate_source_result(
        source_result,
        source_protocol_sha256=source_protocol_sha256,
        upstream_commit=upstream_commit,
    )
    upstream = source_runtime._assert_upstream(args.upstream_root, upstream_commit)

    manifest_identity = source_result["source_manifest"]
    manifest_path = Path(str(manifest_identity["path"])).resolve()
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != manifest_identity["sha256"]
    ):
        raise ValueError("source manifest identity does not verify")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO1"
        or manifest.get("official_eval_read") is not False
    ):
        raise ValueError("checkpoint belief requires the frozen DLO1 manifest")

    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    cublas_config = str(source_protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    import torch

    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(
        torch,
        int(source_protocol["training"]["random_seed"]),
    )
    frame_count = int(source_protocol["data"]["expected_frames_per_trajectory"])
    node_count = int(source_protocol["data"]["expected_node_count"]["DLO1"])
    validation_names = list(manifest["split"]["validation"])
    validation_trajectories = source_runtime._load_named_trajectories(
        manifest,
        validation_names,
        frame_count=frame_count,
        node_count=node_count,
    )

    validation_records = source_result["validation"]
    arms = build_deform_checkpoint_belief_arms(validation_records, protocol)
    required_updates = {
        update for arm_weights in arms.values() for update in arm_weights
    }
    states = _checkpoint_states(source_result, required_updates)
    selected_record = choose_deform_validation_checkpoint(validation_records)
    validation_errors = {"selected_single": float(selected_record["validation_l1_m"])}
    validation_cases: dict[str, list[dict[str, object]]] = {
        "selected_single": list(selected_record["cases"])
    }
    for name, weights in arms.items():
        if name == "selected_single":
            continue
        averaged = average_deform_checkpoint_states(
            {update: states[update] for update in weights},
            weights,
        )
        records = _evaluate_state(
            averaged,
            validation_trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
        )
        validation_cases[name] = records
        validation_errors[name] = _mean_model_l1(records)

    gate = protocol["validation_gate"]
    selection = select_deform_checkpoint_belief_arm(
        validation_errors,
        minimum_relative_improvement=float(gate["minimum_relative_improvement"]),
    )
    selected_arm = str(selection["selected_arm"])
    selection_seal = {
        "schema_version": 1,
        "contract": "deform-dlo-checkpoint-belief-selection-v1",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "source_result": {
            "path": str(source_result_path),
            "sha256": sha256_file(source_result_path),
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "source_protocol": {
            "path": str(args.source_protocol.resolve()),
            "sha256": source_protocol_sha256,
        },
        "upstream": upstream,
        "arms": arms,
        "validation_errors_l1_m": validation_errors,
        "validation_cases": validation_cases,
        "selection": selection,
        "selected_arm_weights": arms[selected_arm],
        "source_test_evaluated_by_this_stage": False,
    }
    selection_path = output_root / "selection_seal.json"
    _write_json(selection_path, selection_seal)

    baseline_source_records = list(source_result["source_test"])
    if selection["fallback_used"]:
        candidate_source_records = baseline_source_records
        exact_fallback = True
    else:
        selected_weights = arms[selected_arm]
        selected_state = average_deform_checkpoint_states(
            {update: states[update] for update in selected_weights},
            selected_weights,
        )
        source_names = list(manifest["split"]["source_test"])
        source_trajectories = source_runtime._load_named_trajectories(
            manifest,
            source_names,
            frame_count=frame_count,
            node_count=node_count,
        )
        candidate_source_records = _evaluate_state(
            selected_state,
            source_trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
        )
        exact_fallback = False

    transfer = evaluate_deform_checkpoint_belief_transfer(
        candidate_source_records,
        baseline_source_records,
    )
    continuation = protocol["source_transfer_report"][
        "required_for_method_continuation"
    ]
    continuation_passed = (
        not exact_fallback
        and float(transfer["relative_improvement"])
        >= float(continuation["relative_improvement_min"])
        and int(transfer["wins"]) >= int(continuation["minimum_case_wins"])
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo-checkpoint-belief-result-v1",
        "claim_boundary": protocol["claim_boundary"],
        "official_eval_read": False,
        "selection_seal": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "selection": selection,
        "selected_arm": selected_arm,
        "selected_arm_weights": arms[selected_arm],
        "exact_fallback": exact_fallback,
        "source_test": {
            "candidate": candidate_source_records,
            "baseline": baseline_source_records,
            "transfer": transfer,
        },
        "fresh_dlo2_method_confirmation_authorized": continuation_passed,
        "fresh_confirmation_contract": protocol["fresh_confirmation"],
    }
    result_path = output_root / "checkpoint_belief_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
