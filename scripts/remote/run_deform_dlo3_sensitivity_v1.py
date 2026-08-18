#!/usr/bin/env python3
"""Run the frozen DLO3 solver/material audit without selecting an arm."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deserialize_deform_local_residual_model,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    evaluate_deform_dlo3_source_gate,
    load_deform_dlo_robustness_v1_protocol,
    validate_deform_dlo3_source_manifest,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

Array = np.ndarray[Any, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verified_file(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(cast(Any, identity.get("size_bytes", -1)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity changed")
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked sensitivity output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _rollout_variant(
    state: dict[str, Any],
    trajectories: dict[str, Array],
    *,
    modules: Any,
    torch: Any,
    device: str,
    pbd_iterations: int,
    stiffness_multiplier: float,
) -> dict[str, object]:
    model_function, model = source_runtime._build_dlo_model(
        modules,
        torch,
        device,
        dlo_type="DLO3",
        node_count=12,
        pbd_iterations=pbd_iterations,
    )
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        model.DEFORM_func.bend_stiffness.mul_(stiffness_multiplier)
        model.DEFORM_func.twist_stiffness.mul_(stiffness_multiplier)
    return cast(
        dict[str, object],
        posterior_runtime._rollout_arrays(
            trajectories,
            modules=modules,
            model_function=model_function,
            model=model,
            torch=torch,
            device=device,
        ),
    )


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    result_path = args.seed_result.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    manifest = _read_json(manifest_path)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    seed_result = _read_json(result_path)
    training = _mapping(protocol.get("physical_training"), label="physical training")
    sensitivity = _mapping(
        protocol.get("physics_solver_sensitivity"), label="sensitivity"
    )
    if (
        seed_result.get("contract") != "deform-dlo3-robustness-seed-result-v1"
        or seed_result.get("seed") != training.get("primary_seed")
        or seed_result.get("source_test_opened") is not True
        or seed_result.get("primary_eval_enumerated") is not False
        or seed_result.get("primary_eval_read") is not False
        or seed_result.get("target_authorized") is not False
        or seed_result.get("held_v8_access") is not False
    ):
        raise ValueError("DLO3 sensitivity seed-result custody differs")
    protocol_identity = _mapping(seed_result.get("protocol"), label="seed protocol")
    manifest_identity = _mapping(
        seed_result.get("source_manifest"), label="seed manifest"
    )
    if protocol_identity.get("sha256") != sha256_file(
        protocol_path
    ) or manifest_identity.get("sha256") != sha256_file(manifest_path):
        raise ValueError("DLO3 sensitivity lineage differs")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"sensitivity output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        str(_mapping(protocol["upstream"], label="upstream")["commit"]),
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO4")
    source_runtime._install_eval_read_guard(data_root / "DLO5")
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-sensitivity-preflight-v1",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "seed_result": _identity(result_path),
        "upstream": upstream,
        "source_test_opened_by_parent": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "preflight.json", preflight)

    cublas_config = str(training["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    source_runtime._seed_everything(torch, int(cast(Any, training["primary_seed"])))
    modules = source_runtime._load_upstream(args.upstream_root)
    names = list(partitions["source_test"])
    trajectories = source_runtime._load_named_trajectories(
        manifest,
        names,
        frame_count=500,
        node_count=12,
    )
    method_seal_path = _verified_file(
        seed_result.get("method_seal"), label="method seal"
    )
    method_seal = _read_json(method_seal_path)
    checkpoint_path = _verified_file(
        method_seal.get("physical_checkpoint"), label="physical checkpoint"
    )
    local_model_path = _verified_file(
        method_seal.get("local_residual_model"), label="local residual model"
    )
    prediction_seal_path = _verified_file(
        seed_result.get("prediction_seal"), label="prediction seal"
    )
    prediction_seal = _read_json(prediction_seal_path)
    source_predictions_path = _verified_file(
        prediction_seal.get("predictions"), label="source predictions"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    with np.load(local_model_path, allow_pickle=False) as archive:
        local_model = deserialize_deform_local_residual_model(archive)
    with np.load(source_predictions_path, allow_pickle=False) as archive:
        parent_names = tuple(str(value) for value in archive["names"])
        parent_candidate = np.asarray(archive["candidate"])
    if parent_names != tuple(names):
        raise ValueError("DLO3 sensitivity source prediction names differ")
    initial, action = local_runtime._causal_inputs(trajectories, names)
    shrinkage = float(
        cast(
            Any,
            _mapping(protocol["local_residual"], label="local residual")["shrinkage"],
        )
    )

    variants: list[tuple[str, int, float]] = [
        (f"pbd-{value}", value, 1.0)
        for value in cast(Sequence[int], sensitivity["pbd_iteration_values"])
    ] + [
        (f"stiffness-{value:.1f}", 10, value)
        for value in cast(Sequence[float], sensitivity["joint_bend_twist_multipliers"])
    ]
    physical_predictions: dict[str, Array] = {}
    corrected_predictions: dict[str, Array] = {}
    targets: Array | None = None
    for label, pbd_iterations, multiplier in variants:
        rollout = _rollout_variant(
            state,
            trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
            pbd_iterations=pbd_iterations,
            stiffness_multiplier=multiplier,
        )
        if tuple(cast(Sequence[str], rollout["names"])) != tuple(names):
            raise ValueError("DLO3 sensitivity rollout names differ")
        current_targets = np.asarray(rollout["targets"])
        if targets is None:
            targets = current_targets
        elif not np.array_equal(targets, current_targets):
            raise ValueError("DLO3 sensitivity rollout targets differ")
        physical = np.asarray(rollout["predictions"])
        corrected = predict_deform_local_residual(
            local_model,
            initial,
            action,
            physical,
            shrinkage=shrinkage,
        )["predictions"]
        physical_predictions[label] = physical
        corrected_predictions[label] = np.asarray(corrected)
    if targets is None:
        raise RuntimeError("DLO3 sensitivity produced no variants")
    if not np.array_equal(corrected_predictions["pbd-10"], parent_candidate):
        raise RuntimeError(
            "nominal sensitivity replay differs from sealed source prediction"
        )
    if not np.array_equal(corrected_predictions["stiffness-1.0"], parent_candidate):
        raise RuntimeError(
            "nominal stiffness replay differs from sealed source prediction"
        )

    predictions_path = output_root / "sensitivity_predictions.npz"
    payload: dict[str, Array] = {"names": np.asarray(names)}
    payload.update(
        {f"physical_{key}": value for key, value in physical_predictions.items()}
    )
    payload.update(
        {f"candidate_{key}": value for key, value in corrected_predictions.items()}
    )
    np.savez_compressed(predictions_path, **cast(dict[str, Any], payload))
    seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-sensitivity-prediction-seal-v1",
        "predictions": _identity(predictions_path),
        "variant_count": len(variants),
        "source_outcomes_scored": False,
        "primary_eval_read": False,
    }
    seal_path = output_root / "prediction_seal.json"
    _write_json(seal_path, seal)

    nominal_physical = physical_predictions["pbd-10"]
    scores = {
        label: evaluate_deform_dlo3_source_gate(
            corrected_predictions[label], nominal_physical, targets, names, protocol
        )
        for label, _, _ in variants
    }
    result = {
        "schema_version": 1,
        "contract": "deform-dlo3-physics-solver-sensitivity-result-v1",
        "claim_boundary": "DLO3 train source-test sensitivity only; no arm selection.",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "seed_result": _identity(result_path),
        "prediction_seal": _identity(seal_path),
        "variants": scores,
        "selection_effect": "none",
        "nominal_replay_exact": True,
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    result_path_out = output_root / "sensitivity_result.json"
    _write_json(result_path_out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
