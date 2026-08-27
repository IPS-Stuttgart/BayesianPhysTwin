#!/usr/bin/env python3
"""Freeze, predict, and score a matched opened-object DEFORM coupling test."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_sparse_state_restart as native
import run_deform_state_restart_noise as noise_runtime

from bayesian_phystwin_experiments.deform_multiobject_restart import (
    config_for_object,
    load_protocol,
    summarize_predictions,
    transfer_assessment,
    validate_manifest,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    array_digest,
    file_digest,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_multiobject_state_restart_v1.json"


def build_cpu_model(
    modules: Any, torch: Any, state: dict[str, Any], dlo: str
) -> tuple[Any, Any]:
    if dlo == "DLO2":
        return native.build_cpu_model(modules, torch, state)
    from bayesian_phystwin_experiments.deform_dlo_upstream import (
        load_deform_dlo_initialization,
    )

    initialization = load_deform_dlo_initialization(modules.train_deform_path, dlo)
    n = initialization.node_count
    if dlo not in ("DLO1", "DLO3") or n != (13 if dlo == "DLO1" else 12):
        raise ValueError("unregistered physical object")
    function = modules.DEFORM_func(n_vert=n, n_edge=n - 1, device="cpu")
    model = modules.DEFORM_sim(n_vert=n, n_edge=n - 1, pbd_iter=10, device="cpu")
    rest = torch.tensor(initialization.rest_vertices_m, dtype=torch.float32).unsqueeze(
        0
    )
    model.rest_vert = torch.nn.Parameter(rest)
    model.m_restEdgeL, model.m_restRegionL = modules.computeLengths(
        modules.computeEdges(rest.clone())
    )
    model.DEFORM_func.bend_stiffness = torch.nn.Parameter(
        initialization.bend_stiffness * torch.ones((1, n - 1))
    )
    model.DEFORM_func.twist_stiffness = torch.nn.Parameter(
        initialization.twist_stiffness * torch.ones((1, n - 1))
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return function, model


class MultiObjectRod(native.NativeRod):
    """Reuse the verified full-state transition; only initialization is object-specific."""

    def __init__(
        self,
        modules: Any,
        torch: Any,
        checkpoint: dict[str, Any],
        config: RestartConfig,
        dlo: str,
    ):
        self.modules, self.torch, self.config = modules, torch, config
        self.function, self.model = build_cpu_model(modules, torch, checkpoint, dlo)
        self.clamped_selection = torch.tensor(config.clamped_nodes)
        self.clamped_index = torch.zeros(config.node_count)
        self.clamped_index[self.clamped_selection] = 1


def freeze(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit the experiment before freezing source")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            PROTOCOL,
            "scripts/remote/run_deform_multiobject_state_restart.py",
            "scripts/remote/run_deform_sparse_state_restart.py",
            "scripts/remote/run_deform_state_restart_noise.py",
            "scripts/remote/run_deform_dlo_source.py",
            "scripts/remote/run_deform_dlo_longrun_posterior.py",
            "scripts/verify_deform_multiobject_state_restart.py",
            "tests/test_deform_multiobject_state_restart.py",
            "docs/deform_multiobject_state_restart_v1.md",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deform-state-restart-source-receipt-v1",
            "experiment": "deform-multiobject-state-restart-v1",
            "revision": revision,
            "git_clean": True,
            "files": {p: file_digest(ROOT / p) for p in paths},
            "new_outcomes_read": False,
            "protected_data_access": False,
        },
    )
    print(json.dumps({"source_revision": revision, "bound_files": len(paths)}))


def control_gate(controls: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    bounded = (
        "native_adapter_max_error_m",
        "archived_gpu_replay_max_error_m",
        "archived_gpu_replay_coordinate_rmse_m",
    )
    return bool(
        all(
            np.isfinite(controls[k]) and 0 <= controls[k] <= thresholds[k]
            for k in bounded
        )
        and controls["zero_update_continuation_byte_identical"] is True
        and controls["incumbent_zero_update_returns_original_object"] is True
        and np.isfinite(controls["synthetic_recovery_fraction"])
        and thresholds["synthetic_minimum_recovery_fraction"]
        <= controls["synthetic_recovery_fraction"]
        <= 1.0
        and controls["synthetic_error_before_l2_m"] > 1e-12
    )


def verify_input_files(item: dict[str, Any]) -> dict[str, Any]:
    for label in ("checkpoint", "archive"):
        spec = item[label]
        if file_digest(Path(spec["path"])) != spec["sha256"]:
            raise ValueError(f"{item['object']} frozen {label} changed")
    return validate_manifest(item)


def save_predictions(
    path: Path, names: list[str], predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("cannot seal a nonfinite forecast as an ordinary success")
    payload: dict[str, Any] = {"names": np.asarray(names), **predictions}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return {
        "sha256": file_digest(path),
        "arrays": {k: array_digest(v) for k, v in payload.items()},
    }


def run_object(
    item: dict[str, Any],
    protocol: dict[str, Any],
    modules: Any,
    torch: Any,
    manifest: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    import run_deform_dlo_longrun_posterior as legacy
    import run_deform_dlo_source as source

    config = config_for_object(protocol, item)
    names = item["names"]
    output.mkdir(exist_ok=False)
    data = source._load_named_trajectories(
        manifest, names, frame_count=500, node_count=config.node_count
    )
    raw = np.stack([data[name] for name in names])
    initial, actions = native.causal_model_inputs(raw, config)
    observation_indices = (
        np.asarray(config.observation_frames) + protocol["dataset_frame_offset"]
    )
    observed = raw[:, observation_indices][:, :, config.observed_nodes].copy()
    archive_spec = item["archive"]
    with np.load(archive_spec["path"], allow_pickle=False) as archive:
        if archive["names"].tolist() != names:
            raise ValueError("prediction archive identity order differs")
        incumbent = archive[archive_spec["incumbent_key"]][
            :, : config.forecast_end
        ].copy()
        archived_physical = archive[archive_spec["physical_key"]][
            :, : config.forecast_end
        ].copy()
    expected_shape = (len(names), config.forecast_end, config.node_count, 3)
    if incumbent.shape != expected_shape or archived_physical.shape != expected_shape:
        raise ValueError("frozen archive time/identity contract differs")
    checkpoint = torch.load(
        item["checkpoint"]["path"], map_location="cpu", weights_only=True
    )["model_state_dict"]
    with torch.no_grad():
        rod = MultiObjectRod(modules, torch, checkpoint, config, item["object"])
        prefix, snapshot = rod.rollout(
            rod.initialize(initial), actions[:, : config.prefix_length]
        )
        snapshot = snapshot.clone()
        future_actions = actions[:, config.prefix_length :]
        nominal, _ = rod.rollout(snapshot.clone(), future_actions)
        nominal_all = np.concatenate((prefix, nominal), axis=1)
        base = incumbent[:, config.prefix_length :]
        zero = torch.zeros_like(snapshot.positions)
        unchanged = update_rod_state(
            snapshot, zero, zero, gain=0, clamped_nodes=config.clamped_nodes
        )
        zero_future, _ = rod.rollout(unchanged, future_actions)
        function, model = build_cpu_model(modules, torch, checkpoint, item["object"])
        reference = legacy._rollout_arrays(
            {name: data[name][: config.forecast_end + 2] for name in names},
            modules=modules,
            model=model,
            model_function=function,
            torch=torch,
            device="cpu",
        )
        replay_error = nominal_all.astype(np.float64) - archived_physical
        controls: dict[str, Any] = {
            "native_adapter_max_error_m": float(
                np.max(np.abs(nominal_all - np.asarray(reference["predictions"])))
            ),
            "archived_gpu_replay_max_error_m": float(np.max(np.abs(replay_error))),
            "archived_gpu_replay_coordinate_rmse_m": float(
                np.sqrt(np.mean(replay_error**2))
            ),
            "zero_update_continuation_byte_identical": array_digest(nominal)
            == array_digest(zero_future),
            "incumbent_zero_update_returns_original_object": paired_physical_readout(
                base, nominal, zero_future
            )
            is base,
        }
        thresholds = protocol["controls"]
        wave = np.sin(np.linspace(0, np.pi, config.node_count))[None, :, None]
        direction = np.array([0.6, -0.3, 0.74161985])[None, None, :]
        perturbation = np.repeat(wave * direction, len(names), axis=0)
        perturbation[:, config.clamped_nodes] = 0
        delta_x = torch.tensor(
            perturbation * thresholds["synthetic_pose_perturbation_m"],
            dtype=torch.float32,
        )
        delta_v = torch.tensor(
            perturbation * thresholds["synthetic_velocity_perturbation_m_s"],
            dtype=torch.float32,
        )
        perturbed = update_rod_state(
            snapshot, delta_x, delta_v, gain=1, clamped_nodes=config.clamped_nodes
        )
        short = future_actions[:, : thresholds["synthetic_horizon"]]
        damaged, _ = rod.rollout(perturbed, short)
        repaired_state = update_rod_state(
            perturbed,
            snapshot.positions - perturbed.positions,
            snapshot.velocity - perturbed.velocity,
            gain=1,
            clamped_nodes=config.clamped_nodes,
        )
        repaired, _ = rod.rollout(repaired_state, short)
        synthetic_reference = nominal[:, : short.shape[1]]
        before = float(np.linalg.norm(damaged.astype(np.float64) - synthetic_reference))
        after = float(np.linalg.norm(repaired.astype(np.float64) - synthetic_reference))
        controls.update(
            {
                "synthetic_error_before_l2_m": before,
                "synthetic_error_after_l2_m": after,
                "synthetic_recovery_fraction": 1 - after / before
                if before > 1e-12
                else 0.0,
            }
        )
        controls["passed"] = control_gate(controls, thresholds)
        write_json_once(output / "controls.json", controls)
        print(
            json.dumps({"stage": "controls", "object": item["object"], **controls}),
            flush=True,
        )
        if not controls["passed"]:
            raise ValueError(
                "native parity/no-op/synthetic control failed; no automatic retry"
            )

        def propagated(
            dx: np.ndarray, dv: np.ndarray, gain: float, paired: bool
        ) -> np.ndarray:
            state = update_rod_state(
                snapshot,
                torch.tensor(dx, dtype=torch.float32),
                torch.tensor(dv, dtype=torch.float32),
                gain=gain,
                clamped_nodes=config.clamped_nodes,
            )
            points, _ = rod.rollout(state, future_actions)
            return paired_physical_readout(base, nominal, points) if paired else points

        dx, dv = sparse_state_increments(
            incumbent[:, : config.prefix_length], observed, config
        )
        pdx, pdv = sparse_state_increments(prefix, observed, config)
        predictions = {
            "incumbent": base,
            "physical_nominal": nominal,
            "readout_sparse_pose": base + dx[:, None],
        }
        predictions["physical_sparse_pose"] = propagated(
            pdx, np.zeros_like(pdv), 1.0, False
        )
        predictions["physical_sparse_pose_velocity"] = propagated(pdx, pdv, 1.0, False)
        predictions["incumbent_propagated_pose"] = propagated(
            dx, np.zeros_like(dv), 1.0, True
        )
        predictions[protocol["primary_arm"]] = propagated(
            dx, dv, protocol["gains"]["primary"], True
        )
        predictions[protocol["secondary_arm"]] = propagated(
            dx, dv, protocol["gains"]["secondary"], True
        )
        if list(predictions) != protocol["clean_arms"]:
            raise ValueError("clean arm contract differs")
        files = {"clean": save_predictions(output / "clean.npz", names, predictions)}
        del predictions
        noise = protocol["noise"]
        for condition_index, condition in enumerate(noise["conditions"]):
            draws: dict[str, list[np.ndarray]] = {
                arm: [] for arm in protocol["noise_arms"]
            }
            for repetition in range(noise["repetitions"]):
                measurement_error = noise_runtime.observation_noise(
                    observed.shape,
                    seed=noise["seed"] + item["noise_seed_offset"] + repetition,
                    independent_std_m=noise["independent_std_m"],
                    shared_std_m=noise["shared_std_m"],
                )[condition_index]
                dx, dv = sparse_state_increments(
                    incumbent[:, : config.prefix_length],
                    observed + measurement_error,
                    config,
                )
                draws["incumbent"].append(base)
                draws["readout_sparse_pose"].append(base + dx[:, None])
                draws[protocol["primary_arm"]].append(
                    propagated(dx, dv, protocol["gains"]["primary"], True)
                )
                draws[protocol["secondary_arm"]].append(
                    propagated(dx, dv, protocol["gains"]["secondary"], True)
                )
                print(
                    json.dumps(
                        {
                            "stage": "noise-prediction",
                            "object": item["object"],
                            "condition": condition,
                            "repetition": repetition + 1,
                        }
                    ),
                    flush=True,
                )
            files[condition] = save_predictions(
                output / f"{condition}.npz",
                names,
                {k: np.stack(v) for k, v in draws.items()},
            )
            del draws
    seal = {
        "schema": "deform-multiobject-object-prediction-seal-v1",
        "object": item["object"],
        "names": names,
        "case_count": len(names),
        "files": files,
        "controls_sha256": file_digest(output / "controls.json"),
        "incumbent_array_sha256": array_digest(base),
        "input_sha256s": {
            key: item[key]["sha256"] for key in ("archive", "checkpoint", "manifest")
        },
        "new_metrics_computed": False,
        "protected_data_access": False,
    }
    write_json_once(output / "prediction_seal.json", seal)
    return {
        "seal_sha256": file_digest(output / "prediction_seal.json"),
        "case_count": len(names),
        "ordinary_success": len(names),
    }


def predict(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    import run_deform_dlo_source as source
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stage = "runtime-and-input-checks"
    completed: dict[str, Any] = {}
    try:
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or str(torch.__version__) != protocol["runtime"]["torch"]
        ):
            raise ValueError("exact CPU-only Torch runtime is required")
        torch.set_num_threads(1)
        source._seed_everything(torch, protocol["bootstrap_seed"])
        upstream = source._assert_upstream(
            Path(protocol["upstream_root"]), protocol["upstream_commit"]
        )
        manifests = {
            item["object"]: verify_input_files(item) for item in protocol["objects"]
        }
        modules = source._load_upstream(Path(protocol["upstream_root"]))
        write_json_once(
            args.output / "preflight.json",
            {
                "schema": "deform-multiobject-preflight-v1",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "upstream": upstream,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "threads": 1,
                },
                "prediction_case_count": protocol["prediction_case_count"],
                "analysis_case_count": protocol["analysis_case_count"],
                "new_metrics_computed": False,
                "protected_data_access": False,
            },
        )
        for item in protocol["objects"]:
            stage = "predict-" + item["object"]
            completed[item["object"]] = run_object(
                item,
                protocol,
                modules,
                torch,
                manifests[item["object"]],
                args.output / item["object"],
            )
        write_json_once(
            args.output / "prediction_barrier.json",
            {
                "schema": "deform-multiobject-prediction-barrier-v1",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "preflight_sha256": file_digest(args.output / "preflight.json"),
                "objects": completed,
                "ordinary_success": protocol["prediction_case_count"],
                "retained_technical_failure": 0,
                "unsealable": 0,
                "analysis_case_count": protocol["analysis_case_count"],
                "new_metrics_computed": False,
                "protected_data_access": False,
                "no_replacement": True,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "all-predictions-sealed",
                    "case_count": protocol["prediction_case_count"],
                    "new_metrics_computed": False,
                }
            ),
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
                "protected_data_access": False,
                "no_automatic_retry": True,
            },
        )
        raise


def validate_barrier(
    output: Path, protocol: dict[str, Any], receipt_digest: str
) -> dict[str, Any]:
    if (output / "failure.json").exists():
        raise ValueError("a retained technical failure blocks scoring")
    barrier = json.loads((output / "prediction_barrier.json").read_text())
    required = {
        "schema": "deform-multiobject-prediction-barrier-v1",
        "source_receipt_sha256": receipt_digest,
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "ordinary_success": protocol["prediction_case_count"],
        "retained_technical_failure": 0,
        "unsealable": 0,
        "analysis_case_count": protocol["analysis_case_count"],
        "new_metrics_computed": False,
        "protected_data_access": False,
        "no_replacement": True,
    }
    if any(barrier.get(k) != v for k, v in required.items()):
        raise ValueError("global prediction barrier or information boundary differs")
    if file_digest(output / "preflight.json") != barrier["preflight_sha256"]:
        raise ValueError("preflight changed after prediction")
    if set(barrier["objects"]) != {item["object"] for item in protocol["objects"]}:
        raise ValueError("incomplete multi-object denominator")
    preflight = json.loads((output / "preflight.json").read_text())
    if (
        preflight["source_receipt_sha256"] != receipt_digest
        or preflight["source_revision"] != barrier["source_revision"]
        or preflight["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or preflight["new_metrics_computed"] is not False
        or preflight["protected_data_access"] is not False
    ):
        raise ValueError("preflight source or boundary contract differs")
    for item in protocol["objects"]:
        directory = output / item["object"]
        record = barrier["objects"][item["object"]]
        if record["case_count"] != len(item["names"]) or record[
            "ordinary_success"
        ] != len(item["names"]):
            raise ValueError("object case accounting differs")
        if file_digest(directory / "prediction_seal.json") != record["seal_sha256"]:
            raise ValueError("object prediction seal changed")
        seal = json.loads((directory / "prediction_seal.json").read_text())
        if (
            seal["schema"] != "deform-multiobject-object-prediction-seal-v1"
            or seal["object"] != item["object"]
            or seal["case_count"] != len(item["names"])
            or seal["names"] != item["names"]
            or seal["new_metrics_computed"] is not False
            or seal["protected_data_access"] is not False
            or seal["input_sha256s"]
            != {
                key: item[key]["sha256"]
                for key in ("archive", "checkpoint", "manifest")
            }
        ):
            raise ValueError("object identity or pre-score boundary differs")
        if file_digest(Path(item["archive"]["path"])) != item["archive"]["sha256"]:
            raise ValueError("registered incumbent archive changed")
        with np.load(item["archive"]["path"], allow_pickle=False) as registered:
            mean = registered[item["archive"]["incumbent_key"]][
                :, protocol["prefix_length"] : protocol["forecast_end"]
            ]
            if array_digest(mean) != seal["incumbent_array_sha256"]:
                raise ValueError("sealed incumbent is not the registered mean")
        if file_digest(directory / "controls.json") != seal["controls_sha256"]:
            raise ValueError("object controls changed")
        controls = json.loads((directory / "controls.json").read_text())
        if (
            not control_gate(controls, protocol["controls"])
            or controls["passed"] is not True
        ):
            raise ValueError("object control authorization does not rederive")
        if set(seal["files"]) != {"clean", *protocol["noise"]["conditions"]}:
            raise ValueError("missing registered noise condition")
        for label, spec in seal["files"].items():
            path = directory / f"{label}.npz"
            if file_digest(path) != spec["sha256"]:
                raise ValueError("prediction archive changed after sealing")
            with np.load(path, allow_pickle=False) as archive:
                arms = (
                    protocol["clean_arms"]
                    if label == "clean"
                    else protocol["noise_arms"]
                )
                if (
                    set(archive.files) != {"names", *arms}
                    or archive["names"].tolist() != item["names"]
                ):
                    raise ValueError("sealed arm or identity set differs")
                if {key: array_digest(archive[key]) for key in archive.files} != spec[
                    "arrays"
                ]:
                    raise ValueError("sealed array bytes differ")
                base = archive["incumbent"]
                if label != "clean":
                    if base.shape[0] != protocol["noise"]["repetitions"] or any(
                        array_digest(v) != seal["incumbent_array_sha256"] for v in base
                    ):
                        raise ValueError(
                            "noise repeats must retain the identical incumbent"
                        )
                elif array_digest(base) != seal["incumbent_array_sha256"]:
                    raise ValueError("incumbent bytes changed")
    return barrier


def score(
    args: argparse.Namespace, protocol: dict[str, Any], receipt: dict[str, Any]
) -> None:
    import run_deform_dlo_source as source

    if (args.output / "result.json").exists():
        raise ValueError("result is write-once")
    validate_barrier(args.output, protocol, file_digest(args.source_receipt))
    results = {}
    extra_hidden = {}
    for item in protocol["objects"]:
        manifest = verify_input_files(item)
        config = config_for_object(protocol, item)
        data = source._load_named_trajectories(
            manifest, item["names"], frame_count=500, node_count=config.node_count
        )
        truth = np.stack([data[n] for n in item["names"]])[
            :, config.prefix_length + 2 : config.forecast_end + 2
        ]
        conditions = {}
        for label in ("clean", *protocol["noise"]["conditions"]):
            with np.load(
                args.output / item["object"] / f"{label}.npz", allow_pickle=False
            ) as archive:
                predictions = {k: archive[k] for k in archive.files if k != "names"}
            conditions[label] = summarize_predictions(
                predictions, truth, item["names"], config
            )
            if item["object"] == "DLO1":
                all_hidden = tuple(
                    i
                    for i in range(config.node_count)
                    if i not in (*config.observed_nodes, *config.clamped_nodes)
                )
                extra_hidden[label] = summarize_predictions(
                    predictions,
                    truth,
                    item["names"],
                    dataclasses.replace(config, hidden_nodes=all_hidden),
                )
        results[item["object"]] = conditions
    result = {
        "schema": "deform-multiobject-state-restart-result-v1",
        "source_revision": receipt["revision"],
        "prediction_barrier_sha256": file_digest(
            args.output / "prediction_barrier.json"
        ),
        "scope": protocol["scope"],
        "objects": results,
        "assessment": transfer_assessment(protocol, results),
        "dlo1_all_hidden_secondary": extra_hidden,
        "ordinary_success": 30,
        "analyzed_trajectories": 29,
        "transfer_trajectories": 16,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "protected_data_access": False,
        "original_results_modified": False,
        "sota_or_fresh_confirmation_claim": False,
    }
    write_json_once(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "stage": "scored",
                "primary_transfer_gate_passed": result["assessment"][
                    "primary_transfer_gate_passed"
                ],
                "analyzed_trajectories": 29,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "predict", "score"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-receipt-sha256")
    args = parser.parse_args()
    protocol = load_protocol(ROOT / PROTOCOL)
    if args.mode == "freeze":
        freeze(args.output)
        return
    if args.source_receipt is None or args.source_receipt_sha256 is None:
        parser.error("the exact source receipt and its SHA-256 are required")
    receipt = native.verify_source(args.source_receipt, args.source_receipt_sha256)
    if receipt.get("experiment") != protocol["schema"]:
        raise ValueError("source receipt belongs to a different experiment")
    (predict if args.mode == "predict" else score)(args, protocol, receipt)


if __name__ == "__main__":
    main()
