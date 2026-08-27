#!/usr/bin/env python3
"""Freeze and run one opened-data weak-constraint DEFORM belief experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_forecast_aware_sensing as previous
import run_deform_multiobject_state_restart as multi

from bayesian_phystwin_experiments.deform_forecast_sensing import (
    SensingConfig,
    query_pairs,
)
from bayesian_phystwin_experiments.deform_multiobject_restart import (
    config_for_object,
    summarize_predictions,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    RodState,
    array_digest,
    file_digest,
    paired_physical_readout,
    update_rod_state,
    write_json_once,
)
from bayesian_phystwin_experiments.deform_weak_constraint_belief import (
    ARMS,
    CALIBRATION_FAMILIES,
    EXPERIMENT,
    NATIVE_ARMS,
    PROTOCOL,
    BeliefConfig,
    arm_columns,
    calibration_scales,
    impulse_basis,
    infer_prefix,
    load_protocol,
    marginal_covariance,
    ols_endpoint,
    physical_impulses,
    primary_decision,
    scaled_covariance,
    summarize_uq,
    validate_response,
)

ROOT = Path(__file__).resolve().parents[2]


def freeze(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("source must be committed and clean before freezing")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "configs/sota/deform_*",
            "scripts/remote/run_deform*",
            "scripts/verify_deform*",
            "tests/test_deform*",
            "docs/deform_*",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deform-state-restart-source-receipt-v1",
            "experiment": EXPERIMENT,
            "revision": revision,
            "git_clean": True,
            "files": {p: file_digest(ROOT / p) for p in paths},
            "new_outcomes_read": False,
            "protected_data_access": False,
        },
    )
    print(
        json.dumps({"source_revision": revision, "bound_files": len(paths)}), flush=True
    )


def response_parent(
    item: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, np.ndarray]:
    root = Path(protocol["response_parent_root"])
    barrier_path = root / "prediction_barrier.json"
    if (
        file_digest(barrier_path)
        != protocol["response_parent_prediction_barrier_sha256"]
    ):
        raise ValueError("sealed parent response barrier changed")
    barrier = json.loads(barrier_path.read_text())
    if barrier["source_revision"] != protocol["response_parent_source_revision"]:
        raise ValueError("response implementation differs")
    directory = root / item["object"]
    seal_path = directory / "prediction_seal.json"
    if file_digest(seal_path) != barrier["objects"][item["object"]]["seal_sha256"]:
        raise ValueError("parent response seal changed")
    seal = json.loads(seal_path.read_text())
    if seal["names"] != item["names"]:
        raise ValueError("parent response case order changed")
    model = previous.verified_arrays(
        directory / "model.npz", seal["files"]["model.npz"]
    )
    previous.verify_incumbent_source(item, model["incumbent"])
    return model


@dataclass
class Prepared:
    config: RestartConfig
    rod: Any
    states: dict[int, RodState]
    actions: np.ndarray
    incumbent: np.ndarray
    nominal: np.ndarray
    response: np.ndarray
    old: dict[str, np.ndarray]
    model_record: dict[str, Any]


def inject_and_roll(
    prepared: Prepared, pose: np.ndarray, velocity: np.ndarray, torch: Any
) -> np.ndarray:
    state = prepared.states[25].clone()
    for step, frame in enumerate((25, 33, 41, 49)):
        if step:
            _, state = prepared.rod.rollout(
                state, prepared.actions[:, (25, 33, 41)[step - 1] + 1 : frame + 1]
            )
        state = update_rod_state(
            state,
            torch.tensor(pose[:, step], dtype=torch.float32),
            torch.tensor(velocity[:, step], dtype=torch.float32),
            gain=1,
            clamped_nodes=prepared.config.clamped_nodes,
        )
    future, _ = prepared.rod.rollout(state, prepared.actions[:, 50:])
    return paired_physical_readout(
        prepared.incumbent[:, 50:], prepared.nominal[:, 25:], future
    )


def prepare_object(
    item: dict[str, Any],
    protocol: dict[str, Any],
    parent: dict[str, Any],
    manifest: dict[str, Any],
    modules: Any,
    torch: Any,
    output: Path,
) -> Prepared:
    import run_deform_dlo_source as source

    output.mkdir(exist_ok=False)
    config, belief = config_for_object(parent, item), BeliefConfig()
    raw = source._load_named_trajectories(
        manifest, item["names"], frame_count=500, node_count=config.node_count
    )
    initial, actions = multi.native.causal_model_inputs(
        np.stack([raw[n] for n in item["names"]]), config
    )
    del raw
    old_model = response_parent(item, protocol)
    old_predictions = previous.parent_prediction(item, protocol)
    checkpoint = torch.load(
        item["checkpoint"]["path"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    rod = multi.MultiObjectRod(modules, torch, checkpoint, config, item["object"])
    first, state = rod.rollout(rod.initialize(initial), actions[:, :26])
    states = {25: state.clone()}
    pieces = [first[:, -1:]]
    for start, end in ((26, 34), (34, 42), (42, 50)):
        piece, state = rod.rollout(state, actions[:, start:end])
        pieces.append(piece)
        states[end - 1] = state.clone()
    future, _ = rod.rollout(state, actions[:, 50:])
    pieces.append(future)
    nominal = np.concatenate(pieces, axis=1)
    incumbent = old_model["incumbent"]
    if array_digest(nominal) != array_digest(old_model["nominal_from_anchor"]):
        raise ValueError("native replay differs from the sealed source model")
    if array_digest(incumbent[:, 50:]) != array_digest(old_predictions["incumbent"]):
        raise ValueError("incumbent mean changed")
    if array_digest(future) != array_digest(old_predictions["physical_nominal"]):
        raise ValueError("previous physical continuation changed")
    response = np.zeros((*nominal.shape, 60))
    response[..., :24] = old_model["response"]
    pose, velocity = impulse_basis(config, belief)
    asymmetry = []
    for mode in range(24, 60):
        step = mode // 12 - 1
        frame = belief.observation_frames[step]
        values = []
        for sign in (-1, 1):
            dx = np.broadcast_to(
                sign * belief.finite_difference_fraction * pose[step, mode],
                (len(item["names"]), config.node_count, 3),
            ).copy()
            dv = np.broadcast_to(
                sign * belief.finite_difference_fraction * velocity[step, mode],
                dx.shape,
            ).copy()
            perturbed = update_rod_state(
                states[frame],
                torch.tensor(dx, dtype=torch.float32),
                torch.tensor(dv, dtype=torch.float32),
                gain=1,
                clamped_nodes=config.clamped_nodes,
            )
            continuation, _ = rod.rollout(perturbed, actions[:, frame + 1 :])
            values.append(
                np.concatenate(
                    (perturbed.positions.detach().cpu().numpy()[:, None], continuation),
                    axis=1,
                ).astype(np.float64)
            )
        response[:, frame - 25 :, :, :, mode] = (values[1] - values[0]) / (
            2 * belief.finite_difference_fraction
        )
        asymmetry.append(
            float(
                np.max(np.abs((values[1] + values[0]) / 2 - nominal[:, frame - 25 :]))
            )
        )
        if (mode + 1) % 4 == 0:
            print(
                json.dumps(
                    {
                        "stage": "process-responses",
                        "object": item["object"],
                        "completed": mode - 23,
                        "total": 36,
                    }
                ),
                flush=True,
            )
    for case in response:
        validate_response(case, config)
    record = previous.save_arrays(
        output / "model.npz",
        {
            "names": np.asarray(item["names"]),
            "incumbent": incumbent,
            "nominal_from_anchor": nominal,
            "response": response,
            "initial": initial,
            "actions": actions,
        },
    )
    prepared = Prepared(
        config,
        rod,
        states,
        actions,
        incumbent,
        nominal,
        response,
        old_predictions,
        record,
    )
    zero = np.zeros((len(item["names"]), 4, config.node_count, 3))
    unchanged = inject_and_roll(prepared, zero, zero, torch)
    if array_digest(unchanged) != array_digest(incumbent[:, 50:]):
        raise ValueError("zero-increment native/readout fallback changed")
    write_json_once(
        output / "controls.json",
        {
            "parent_incumbent_byte_identical": True,
            "parent_native_future_byte_identical": True,
            "parent_initial_response_byte_identical": array_digest(response[..., :24])
            == array_digest(old_model["response"]),
            "zero_increment_mean_byte_identical": True,
            "clamp_and_pre_impulse_response_exactly_zero": True,
            "maximum_process_response_asymmetry_m": max(asymmetry),
            "asymmetry_is_diagnostic_not_selection": True,
            "prefix_measurements_revealed": False,
            "new_metrics_computed": False,
        },
    )
    return prepared


def periodic_pose(prepared: Prepared, points: np.ndarray, torch: Any) -> np.ndarray:
    state = prepared.states[25].clone()
    for step, frame in enumerate((25, 33, 41, 49)):
        if step:
            _, state = prepared.rod.rollout(
                state, prepared.actions[:, (25, 33, 41)[step - 1] + 1 : frame + 1]
            )
        current = (
            prepared.incumbent[:, frame].astype(np.float64)
            + state.positions.detach().cpu().numpy().astype(np.float64)
            - prepared.nominal[:, frame - 25]
        )
        dx = np.zeros_like(current)
        dx[:, prepared.config.observed_nodes] = (
            points[:, step * 4 : (step + 1) * 4]
            - current[:, prepared.config.observed_nodes]
        )
        state = update_rod_state(
            state,
            torch.tensor(dx, dtype=torch.float32),
            torch.zeros_like(state.velocity),
            gain=1,
            clamped_nodes=prepared.config.clamped_nodes,
        )
    future, _ = prepared.rod.rollout(state, prepared.actions[:, 50:])
    return paired_physical_readout(
        prepared.incumbent[:, 50:], prepared.nominal[:, 25:], future
    )


def predict_object(
    prepared: Prepared,
    item: dict[str, Any],
    manifest: dict[str, Any],
    torch: Any,
    output: Path,
) -> dict[str, Any]:
    import run_deform_dlo_source as source

    config, belief = prepared.config, BeliefConfig()
    raw = source._load_named_trajectories(
        manifest, item["names"], frame_count=500, node_count=config.node_count
    )
    pairs = query_pairs(config, SensingConfig())
    points = np.stack(
        [np.stack([raw[n][t + 2, node] for t, node in pairs]) for n in item["names"]]
    ).astype(np.float64)
    del raw
    reference = np.stack([prepared.incumbent[:, t, n] for t, n in pairs], axis=1)
    predictions = {
        "incumbent": prepared.incumbent[:, 50:],
        "previous_paired_8": prepared.old["incumbent_propagated_pose_velocity"],
    }
    dx, dv, ols_gain = ols_endpoint(prepared.incumbent, points, config, belief)
    endpoint = update_rod_state(
        prepared.states[49],
        torch.tensor(dx, dtype=torch.float32),
        torch.tensor(dv, dtype=torch.float32),
        gain=1,
        clamped_nodes=config.clamped_nodes,
    )
    physical, _ = prepared.rod.rollout(endpoint, prepared.actions[:, 50:])
    predictions["ols_physical_16"] = paired_physical_readout(
        predictions["incumbent"], prepared.nominal[:, 25:], physical
    )
    t = np.arange(1, 121)[None, :, None, None] * config.dt_s
    predictions["ols_readout_16"] = (
        predictions["incumbent"] + dx[:, None] + t * dv[:, None]
    )
    predictions["periodic_pose_16"] = periodic_pose(prepared, points, torch)
    fits = {
        "observations": points,
        "reference": reference,
        "ols_pose": dx,
        "ols_velocity": dv,
        "ols_gain": ols_gain,
    }
    covariances = {}
    for arm in NATIVE_ARMS:
        estimates = [
            infer_prefix(
                prepared.response[i], reference[i], points[i], arm, config, belief
            )
            for i in range(len(points))
        ]
        coefficients, posterior = (
            np.stack([pair[k] for pair in estimates]) for k in range(2)
        )
        pose, velocity, gain = physical_impulses(coefficients, arm, config, belief)
        predictions[arm] = inject_and_roll(prepared, pose, velocity, torch)
        covariances[arm] = marginal_covariance(
            prepared.response, posterior, gain, arm, belief
        )
        fits.update(
            {
                f"{arm}__coefficients": coefficients,
                f"{arm}__posterior": posterior,
                f"{arm}__gain": gain,
                f"{arm}__pose": pose,
                f"{arm}__velocity": velocity,
            }
        )
        print(
            json.dumps({"stage": "prediction", "object": item["object"], "arm": arm}),
            flush=True,
        )
    if tuple(predictions) != ARMS:
        raise ValueError("prediction arm set/order changed")
    files = {
        "model.npz": prepared.model_record,
        "predictions.npz": previous.save_arrays(
            output / "predictions.npz",
            {"names": np.asarray(item["names"]), **predictions},
        ),
        "fits.npz": previous.save_arrays(output / "fits.npz", fits),
        "covariances.npz": previous.save_arrays(
            output / "covariances.npz", covariances
        ),
    }
    write_json_once(
        output / "prediction_seal.json",
        {
            "schema": EXPERIMENT + "-object-seal",
            "names": item["names"],
            "object": item["object"],
            "files": files,
            "controls_sha256": file_digest(output / "controls.json"),
            "new_metrics_computed": False,
            "protected_data_access": False,
            "future_free_node_truth_used": False,
            "previous_paired_prediction_byte_identical": True,
        },
    )
    return {
        "seal_sha256": file_digest(output / "prediction_seal.json"),
        "ordinary_success": len(item["names"]),
    }


def predict(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    import run_deform_dlo_source as source
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    stage, completed, start = "preflight", {}, time.perf_counter()
    try:
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or str(torch.__version__) != parent["runtime"]["torch"]
        ):
            raise ValueError("exact frozen CPU runtime required")
        torch.set_num_threads(1)
        source._seed_everything(torch, parent["bootstrap_seed"])
        upstream = source._assert_upstream(
            Path(parent["upstream_root"]), parent["upstream_commit"]
        )
        manifests = {
            item["object"]: multi.verify_input_files(item) for item in parent["objects"]
        }
        modules = source._load_upstream(Path(parent["upstream_root"]))
        write_json_once(
            args.output / "preflight.json",
            {
                "schema": EXPERIMENT + "-preflight",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "threads": 1,
                },
                "upstream": upstream,
                "protected_data_access": False,
                "new_metrics_computed": False,
            },
        )
        prepared = {}
        with torch.no_grad():
            for item in parent["objects"]:
                stage = "responses-" + item["object"]
                prepared[item["object"]] = prepare_object(
                    item,
                    protocol,
                    parent,
                    manifests[item["object"]],
                    modules,
                    torch,
                    args.output / item["object"],
                )
            write_json_once(
                args.output / "response_barrier.json",
                {
                    "schema": EXPERIMENT + "-response-barrier",
                    "source_receipt_sha256": file_digest(args.source_receipt),
                    "protocol_sha256": file_digest(ROOT / PROTOCOL),
                    "objects": {
                        name: value.model_record for name, value in prepared.items()
                    },
                    "case_count": 30,
                    "prefix_measurements_revealed": False,
                    "new_metrics_computed": False,
                    "protected_data_access": False,
                },
            )
            for item in parent["objects"]:
                stage = "predict-" + item["object"]
                completed[item["object"]] = predict_object(
                    prepared[item["object"]],
                    item,
                    manifests[item["object"]],
                    torch,
                    args.output / item["object"],
                )
        write_json_once(
            args.output / "prediction_barrier.json",
            {
                "schema": EXPERIMENT + "-prediction-barrier",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "preflight_sha256": file_digest(args.output / "preflight.json"),
                "response_barrier_sha256": file_digest(
                    args.output / "response_barrier.json"
                ),
                "objects": completed,
                "ordinary_success": 30,
                "analysis_case_count": 29,
                "retained_technical_failure": 0,
                "unsealable": 0,
                "no_replacement": True,
                "new_metrics_computed": False,
                "protected_data_access": False,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        print(
            json.dumps({"stage": "all-predictions-sealed", "ordinary_success": 30}),
            flush=True,
        )
    except Exception as error:
        write_json_once(
            args.output / "failure.json",
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "completed_objects": completed,
                "no_automatic_retry": True,
                "protected_data_access": False,
            },
        )
        raise


def validate_barrier(
    output: Path,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
    receipt_digest: str,
) -> dict[str, Any]:
    if (output / "failure.json").exists():
        raise ValueError("retained technical failure blocks advancement")
    barrier = json.loads((output / "prediction_barrier.json").read_text())
    expected = {
        "schema": EXPERIMENT + "-prediction-barrier",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": receipt_digest,
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "ordinary_success": 30,
        "analysis_case_count": 29,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "no_replacement": True,
        "new_metrics_computed": False,
        "protected_data_access": False,
    }
    if any(barrier.get(k) != v for k, v in expected.items()) or set(
        barrier["objects"]
    ) != {"DLO1", "DLO2", "DLO3"}:
        raise ValueError("incomplete denominator or changed prediction boundary")
    for name in ("preflight", "response_barrier"):
        if file_digest(output / f"{name}.json") != barrier[name + "_sha256"]:
            raise ValueError("pre-measurement source provenance changed")
    preflight = json.loads((output / "preflight.json").read_text())
    if any(
        preflight.get(k) != v
        for k, v in {
            "schema": EXPERIMENT + "-preflight",
            "source_revision": receipt["revision"],
            "source_receipt_sha256": receipt_digest,
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "protected_data_access": False,
            "new_metrics_computed": False,
        }.items()
    ):
        raise ValueError("preflight provenance or information boundary differs")
    responses = json.loads((output / "response_barrier.json").read_text())
    if any(
        responses.get(k) != v
        for k, v in {
            "schema": EXPERIMENT + "-response-barrier",
            "source_receipt_sha256": receipt_digest,
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "case_count": 30,
            "prefix_measurements_revealed": False,
            "new_metrics_computed": False,
            "protected_data_access": False,
        }.items()
    ):
        raise ValueError("responses were not frozen before observations")
    if set(responses["objects"]) != {"DLO1", "DLO2", "DLO3"}:
        raise ValueError("incomplete response objects")
    for item in parent["objects"]:
        directory = output / item["object"]
        record = barrier["objects"][item["object"]]
        if (
            record["ordinary_success"] != len(item["names"])
            or file_digest(directory / "prediction_seal.json") != record["seal_sha256"]
        ):
            raise ValueError("object count or seal changed")
        seal = json.loads((directory / "prediction_seal.json").read_text())
        if any(
            seal.get(k) != v
            for k, v in {
                "schema": EXPERIMENT + "-object-seal",
                "object": item["object"],
                "names": item["names"],
                "new_metrics_computed": False,
                "protected_data_access": False,
                "future_free_node_truth_used": False,
                "previous_paired_prediction_byte_identical": True,
            }.items()
        ):
            raise ValueError("object identity or information boundary differs")
        if set(seal["files"]) != {
            "model.npz",
            "predictions.npz",
            "fits.npz",
            "covariances.npz",
        }:
            raise ValueError("required prediction/covariance/inference files missing")
        arrays = {
            name: previous.verified_arrays(directory / name, spec)
            for name, spec in seal["files"].items()
        }
        if responses["objects"][item["object"]] != seal["files"]["model.npz"]:
            raise ValueError("response changed after observations")
        if file_digest(directory / "controls.json") != seal["controls_sha256"]:
            raise ValueError("control record changed")
        controls = json.loads((directory / "controls.json").read_text())
        for key in (
            "parent_incumbent_byte_identical",
            "parent_native_future_byte_identical",
            "parent_initial_response_byte_identical",
            "zero_increment_mean_byte_identical",
            "clamp_and_pre_impulse_response_exactly_zero",
        ):
            if controls.get(key) is not True:
                raise ValueError("required identity or causal response control failed")
        config = config_for_object(parent, item)
        batch, nodes = len(item["names"]), config.node_count
        model, predictions, fits, cov = (
            arrays[key]
            for key in ("model.npz", "predictions.npz", "fits.npz", "covariances.npz")
        )
        model_shapes = {
            "incumbent": (batch, 170, nodes, 3),
            "nominal_from_anchor": (batch, 145, nodes, 3),
            "response": (batch, 145, nodes, 3, 60),
            "initial": (batch, 2, nodes, 3),
            "actions": (batch, 170, 4, 3),
        }
        if set(model) != {"names", *model_shapes} or any(
            model[k].shape != shape for k, shape in model_shapes.items()
        ):
            raise ValueError("model carrier shapes/fields changed")
        if (
            model["names"].tolist() != item["names"]
            or predictions["names"].tolist() != item["names"]
        ):
            raise ValueError("trajectory ordering changed")
        previous.verify_incumbent_source(item, model["incumbent"])
        parent_model = response_parent(item, protocol)
        old = previous.parent_prediction(item, protocol)
        for first, second in (
            (model["response"][..., :24], parent_model["response"]),
            (model["nominal_from_anchor"], parent_model["nominal_from_anchor"]),
            (predictions["incumbent"], old["incumbent"]),
            (
                predictions["previous_paired_8"],
                old["incumbent_propagated_pose_velocity"],
            ),
            (predictions["incumbent"], model["incumbent"][:, 50:]),
        ):
            if array_digest(first) != array_digest(second):
                raise ValueError(
                    "immutable incumbent, prior update, or response changed"
                )
        for row in model["response"]:
            validate_response(row, config)
        if set(predictions) != {"names", *ARMS} or any(
            predictions[k].shape != (batch, 120, nodes, 3) for k in ARMS
        ):
            raise ValueError("point arm or horizon denominator changed")
        if set(cov) != set(NATIVE_ARMS):
            raise ValueError("missing marginal covariance arm")
        expected_fits = {
            "observations",
            "reference",
            "ols_pose",
            "ols_velocity",
            "ols_gain",
        } | {
            f"{arm}__{k}"
            for arm in NATIVE_ARMS
            for k in ("coefficients", "posterior", "gain", "pose", "velocity")
        }
        if (
            set(fits) != expected_fits
            or fits["observations"].shape != (batch, 16, 3)
            or fits["reference"].shape != (batch, 16, 3)
            or fits["ols_pose"].shape != (batch, nodes, 3)
            or fits["ols_velocity"].shape != (batch, nodes, 3)
            or fits["ols_gain"].shape != (batch,)
        ):
            raise ValueError("inference evidence field set differs")
        for arm in NATIVE_ARMS:
            rank = len(arm_columns(arm)) + 3
            shapes = {
                "coefficients": (batch, rank),
                "posterior": (batch, rank, rank),
                "gain": (batch,),
                "pose": (batch, 4, nodes, 3),
                "velocity": (batch, 4, nodes, 3),
            }
            if any(fits[f"{arm}__{k}"].shape != shape for k, shape in shapes.items()):
                raise ValueError("latent or physical posterior dimensions changed")
            gain = fits[f"{arm}__gain"]
            if (
                np.any((gain <= 0) | (gain > 1))
                or np.linalg.eigvalsh(fits[f"{arm}__posterior"]).min() < -1e-9
            ):
                raise ValueError("invalid guard gain or non-PSD latent posterior")
            for key in ("pose", "velocity"):
                if np.any(fits[f"{arm}__{key}"][:, :, config.clamped_nodes]):
                    raise ValueError("inferred increments move prescribed clamps")
            if cov[arm].shape != (batch, 120, nodes, 3, 3):
                raise ValueError("marginal covariance dimensions changed")
            np.linalg.cholesky(cov[arm])
            if not np.allclose(cov[arm], cov[arm].swapaxes(-2, -1), atol=1e-12, rtol=0):
                raise ValueError("marginal covariance is not symmetric")
    return barrier


def truth_for(item: dict[str, Any], parent: dict[str, Any]) -> np.ndarray:
    import run_deform_dlo_source as source

    config = config_for_object(parent, item)
    raw = source._load_named_trajectories(
        multi.verify_input_files(item),
        item["names"],
        frame_count=500,
        node_count=config.node_count,
    )
    return np.stack([raw[n][52:172] for n in item["names"]])


def load_object(
    output: Path, item: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = output / item["object"]
    seal = json.loads((directory / "prediction_seal.json").read_text())
    prediction = previous.verified_arrays(
        directory / "predictions.npz", seal["files"]["predictions.npz"]
    )
    prediction.pop("names")
    cov = previous.verified_arrays(
        directory / "covariances.npz", seal["files"]["covariances.npz"]
    )
    return prediction, cov


def calibration_inputs(
    predictions: dict[str, np.ndarray],
    covariance: dict[str, np.ndarray],
    truth: np.ndarray,
    hidden: tuple[int, ...],
    keep: list[int],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    if family not in CALIBRATION_FAMILIES:
        raise ValueError("unregistered calibration family")
    error = predictions[CALIBRATION_FAMILIES[family]][keep][:, :, hidden].astype(
        np.float64
    ) - truth[keep][:, :, hidden].astype(np.float64)
    if family.endswith("shaped"):
        cov = covariance["weak_16"][keep][:, :, hidden].copy()
    else:
        cov = np.broadcast_to(
            BeliefConfig().covariance_floor_std_m ** 2 * np.eye(3), (*error.shape, 3)
        ).copy()
    return error, cov


def calibrate(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    validate_barrier(
        args.output, protocol, parent, receipt, file_digest(args.source_receipt)
    )
    item = next(item for item in parent["objects"] if item["object"] == "DLO2")
    truth = truth_for(item, parent)
    prediction, covariance = load_object(args.output, item)
    config = config_for_object(parent, item)
    keep = [i for i, name in enumerate(item["names"]) if name != config.design_case]
    scales = {}
    for family in CALIBRATION_FAMILIES:
        error, cov = calibration_inputs(
            prediction, covariance, truth, config.hidden_nodes, keep, family
        )
        scales[family] = calibration_scales(error, cov, object_name="DLO2")
    write_json_once(
        args.output / "calibration.json",
        {
            "schema": EXPERIMENT + "-source-calibration",
            "source_revision": receipt["revision"],
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "prediction_barrier_sha256": file_digest(
                args.output / "prediction_barrier.json"
            ),
            "calibration_object": "DLO2",
            "calibration_case_count": 13,
            "excluded_design_case": "103.pkl",
            "source_truth_sha256": array_digest(truth),
            "scales": scales,
            "transfer_metrics_computed": False,
            "protected_data_access": False,
            "same_mean_isotropic_comparator": True,
            "population_coverage_guarantee": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "source-calibration-sealed",
                "case_count": 13,
                "sha256": file_digest(args.output / "calibration.json"),
            }
        ),
        flush=True,
    )


def validate_calibration(
    path: Path, output: Path, expected_digest: str, receipt: dict[str, Any]
) -> dict[str, Any]:
    if not expected_digest or file_digest(path) != expected_digest:
        raise ValueError(
            "explicit source-calibration digest required before transfer scoring"
        )
    record = json.loads(path.read_text())
    expected = {
        "schema": EXPERIMENT + "-source-calibration",
        "source_revision": receipt["revision"],
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "prediction_barrier_sha256": file_digest(output / "prediction_barrier.json"),
        "calibration_object": "DLO2",
        "calibration_case_count": 13,
        "excluded_design_case": "103.pkl",
        "transfer_metrics_computed": False,
        "protected_data_access": False,
        "same_mean_isotropic_comparator": True,
        "population_coverage_guarantee": False,
    }
    if any(record.get(k) != v for k, v in expected.items()) or set(
        record["scales"]
    ) != set(CALIBRATION_FAMILIES):
        raise ValueError("calibration boundary or family set differs")
    for family in record["scales"].values():
        if set(family) != {"moment", "conformal"}:
            raise ValueError("calibration variant changed")
        for scales in family.values():
            if len(scales) != 3 or any(not np.isfinite(x) or x <= 0 for x in scales):
                raise ValueError("invalid horizon scales")
    return record


def score(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    barrier = validate_barrier(
        args.output, protocol, parent, receipt, file_digest(args.source_receipt)
    )
    calibration = validate_calibration(
        args.output / "calibration.json", args.output, args.calibration_sha256, receipt
    )
    results = {}
    for item in parent["objects"]:
        truth = truth_for(item, parent)
        prediction, covariance = load_object(args.output, item)
        config = config_for_object(parent, item)
        keep = [i for i, name in enumerate(item["names"]) if name != config.design_case]
        point = summarize_predictions(prediction, truth, item["names"], config)
        uq = {}
        for family in CALIBRATION_FAMILIES:
            error, cov = calibration_inputs(
                prediction, covariance, truth, config.hidden_nodes, keep, family
            )
            for variant, scales in calibration["scales"][family].items():
                uq[family + "__" + variant] = summarize_uq(
                    error, scaled_covariance(cov, scales)
                )
        results[item["object"]] = {"point": point, "uq": uq}
    decision = primary_decision(results, BeliefConfig())
    aggregate = previous.summarize_aggregates(
        {name: {"clean": value["point"]} for name, value in results.items()}
    )
    record = {
        "schema": EXPERIMENT + "-result",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": file_digest(args.source_receipt),
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "prediction_barrier_sha256": file_digest(
            args.output / "prediction_barrier.json"
        ),
        "calibration_sha256": file_digest(args.output / "calibration.json"),
        "objects": results,
        "aggregate": aggregate,
        "decision": decision,
        "ordinary_success": barrier["ordinary_success"],
        "analysis_case_count": 29,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "protected_data_access": False,
        "original_results_modified": False,
        "population_confirmation_or_sota_claim": False,
    }
    write_json_once(args.output / "result.json", record)
    print(
        json.dumps(
            {
                "stage": "scored",
                "decision": decision,
                "transfer_clean": aggregate["transfer_only"]["clean"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "predict", "validate", "calibrate", "score")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-receipt-sha256")
    parser.add_argument("--calibration-sha256", default="")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
        return
    if args.source_receipt is None or not args.source_receipt_sha256:
        parser.error("source receipt and expected digest are required")
    receipt = multi.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    if receipt.get("experiment") != EXPERIMENT:
        raise ValueError("receipt belongs to another experiment")
    protocol, parent = load_protocol(ROOT / PROTOCOL, ROOT)
    if args.command == "predict":
        predict(args, protocol, parent, receipt)
    elif args.command == "calibrate":
        calibrate(args, protocol, parent, receipt)
    elif args.command == "score":
        score(args, protocol, parent, receipt)
    else:
        validate_barrier(
            args.output, protocol, parent, receipt, file_digest(args.source_receipt)
        )
        print(
            json.dumps({"barrier_valid": True, "new_metrics_computed": False}),
            flush=True,
        )


if __name__ == "__main__":
    main()
