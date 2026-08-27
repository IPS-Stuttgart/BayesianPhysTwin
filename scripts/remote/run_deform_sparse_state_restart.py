#!/usr/bin/env python3
"""Run the isolated opened-DLO2 physical state-restart experiment on CPU."""

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

from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    RodState,
    aggregate_paired_metrics,
    array_digest,
    file_digest,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_sparse_state_restart_dev_v1.json"


def freeze_source(output: Path) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if status:
        raise ValueError("commit the source experiment before freezing it")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            PROTOCOL,
            "scripts/remote/run_deform_sparse_state_restart.py",
            "scripts/remote/run_deform_dlo_source.py",
            "scripts/remote/run_deform_dlo_longrun_posterior.py",
            "tests/test_deform_state_restart.py",
            "docs/deform_sparse_state_restart_dev_v1.md",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deform-state-restart-source-receipt-v1",
            "revision": revision,
            "git_clean": True,
            "files": {p: file_digest(ROOT / p) for p in paths},
            "data_read": False,
        },
    )
    print(
        json.dumps(
            {"source_revision": revision, "bound_files": len(paths)}, sort_keys=True
        )
    )


def verify_source(receipt_path: Path, expected_digest: str) -> dict[str, Any]:
    if file_digest(receipt_path) != expected_digest:
        raise ValueError("source receipt digest differs")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "deform-state-restart-source-receipt-v1"
        or receipt.get("git_clean") is not True
    ):
        raise ValueError("source receipt is not a clean frozen implementation")
    for name, digest in receipt["files"].items():
        if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("source identity escapes checkout")
        if file_digest(ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    return receipt


def build_cpu_model(modules: Any, torch: Any, state: dict[str, Any]) -> tuple[Any, Any]:
    """Same initialization as the CUDA-only legacy builder, without editing it."""
    from bayesian_phystwin_experiments.deform_dlo_upstream import (
        load_deform_dlo_initialization,
    )

    initialization = load_deform_dlo_initialization(modules.train_deform_path, "DLO2")
    node_count = initialization.node_count
    if node_count != 12:
        raise ValueError("frozen DLO2 node count changed")
    function = modules.DEFORM_func(
        n_vert=node_count, n_edge=node_count - 1, device="cpu"
    )
    model = modules.DEFORM_sim(
        n_vert=node_count, n_edge=node_count - 1, pbd_iter=10, device="cpu"
    )
    rest = torch.tensor(initialization.rest_vertices_m, dtype=torch.float32).unsqueeze(
        0
    )
    model.rest_vert = torch.nn.Parameter(rest)
    model.m_restEdgeL, model.m_restRegionL = modules.computeLengths(
        modules.computeEdges(rest.clone())
    )
    model.DEFORM_func.bend_stiffness = torch.nn.Parameter(
        initialization.bend_stiffness * torch.ones((1, node_count - 1)),
    )
    model.DEFORM_func.twist_stiffness = torch.nn.Parameter(
        initialization.twist_stiffness * torch.ones((1, node_count - 1)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return function, model


class NativeRod:
    def __init__(
        self,
        modules: Any,
        torch: Any,
        checkpoint: dict[str, Any],
        config: RestartConfig,
    ):
        self.modules, self.torch, self.config = modules, torch, config
        self.function, self.model = build_cpu_model(modules, torch, checkpoint)
        self.clamped_selection = torch.tensor(config.clamped_nodes)
        self.clamped_index = torch.zeros(config.node_count)
        self.clamped_index[self.clamped_selection] = 1

    def initialize(self, first_two_frames: np.ndarray) -> RodState:
        torch = self.torch
        if first_two_frames.ndim != 4 or first_two_frames.shape[1:] != (
            2,
            self.config.node_count,
            3,
        ):
            raise ValueError("initial state requires exactly two frames")
        values = torch.tensor(first_two_frames, dtype=torch.float32)
        batch = len(values)
        self.initial = (
            torch.tensor(((0.0, 0.6, 0.8), (0.0, 0.0, 1.0)))
            .unsqueeze(0)
            .repeat(batch, 1, 1)
        )
        rest = self.model.m_restEdgeL.repeat(batch, 1)
        self.model.m_restWprev, self.model.m_restWnext, self.model.learned_pmass = (
            self.model.Rod_Init(
                batch,
                self.initial,
                rest,
                self.clamped_index,
            )
        )
        positions = values[:, 1]
        material = self.function.compute_u0(
            self.modules.computeEdges(positions)[:, 0].float(), self.initial[:, 0]
        )
        return RodState(
            positions,
            (positions - values[:, 0]) / self.model.dt,
            values[:, 0],
            material,
            torch.zeros(batch, self.config.node_count - 1),
            -1,
        )

    def advance(self, state: RodState, clamped_positions: np.ndarray) -> RodState:
        material = state.material_u0
        if state.prediction_index >= 0:
            previous_edges = self.modules.computeEdges(state.previous_positions)
            current_edges = self.modules.computeEdges(state.positions)
            material = self.function.parallelTransportFrame(
                previous_edges[:, 0], current_edges[:, 0], material
            )
        points, velocity, theta = self.model(
            state.positions.clone(),
            state.velocity.clone(),
            self.initial,
            self.clamped_index,
            material,
            self.torch.tensor(clamped_positions, dtype=self.torch.float32),
            self.clamped_selection,
            state.theta.clone(),
            mode="evaluation",
        )
        if not all(x.isfinite().all() for x in (points, velocity, theta, material)):
            raise ValueError("native state became nonfinite")
        return RodState(
            points,
            velocity,
            state.positions.clone(),
            material,
            theta,
            state.prediction_index + 1,
        )

    def rollout(
        self, state: RodState, clamped_positions: np.ndarray
    ) -> tuple[np.ndarray, RodState]:
        points = []
        for frame in range(clamped_positions.shape[1]):
            state = self.advance(state, clamped_positions[:, frame])
            points.append(state.positions.detach().cpu().numpy().copy())
        return np.stack(points, axis=1), state


def causal_model_inputs(
    trajectories: np.ndarray, config: RestartConfig
) -> tuple[np.ndarray, np.ndarray]:
    if trajectories.ndim != 4 or trajectories.shape[1] < config.forecast_end + 2:
        raise ValueError("raw trajectory is too short")
    initial = trajectories[:, :2].copy()
    actions = trajectories[:, 2 : config.forecast_end + 2, config.clamped_nodes].copy()
    return initial, actions


def _save_predictions(path: Path, predictions: dict[str, np.ndarray]) -> dict[str, str]:
    payload: dict[str, Any] = dict(predictions)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return {name: array_digest(array) for name, array in predictions.items()}


def run(args: argparse.Namespace) -> None:
    receipt = verify_source(args.source_receipt, args.source_receipt_sha256)
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    fields = {f.name for f in dataclasses.fields(RestartConfig)}
    values = {k: v for k, v in protocol.items() if k in fields}
    for key in (
        "observation_frames",
        "observed_nodes",
        "hidden_nodes",
        "clamped_nodes",
    ):
        values[key] = tuple(values[key])
    config = RestartConfig(**values)
    if file_digest(args.archive) != protocol["source_archive_sha256"]:
        raise ValueError("opened prediction archive identity differs")
    if file_digest(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("physical checkpoint identity differs")
    if file_digest(args.evaluation_manifest) != args.evaluation_manifest_sha256:
        raise ValueError("opened raw-trajectory manifest identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stage = "runtime-initialization"
    try:
        import run_deform_dlo_longrun_posterior as legacy
        import run_deform_dlo_source as source
        import torch

        if str(torch.__version__) != protocol["runtime"]["torch"]:
            raise ValueError("Torch version differs from frozen runtime")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("this CPU experiment must hide all CUDA devices")
        torch.set_num_threads(1)
        source._seed_everything(torch, config.seed)
        upstream = source._assert_upstream(
            args.upstream_root, protocol["upstream_commit"]
        )
        modules = source._load_upstream(args.upstream_root)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        manifest = json.loads(args.evaluation_manifest.read_text())
        names = manifest["ordered_names"]
        if names != protocol["expected_names"] or set(manifest["trajectories"]) != set(
            names
        ):
            raise ValueError("only the fixed already-open roster may be loaded")
        preflight: dict[str, Any] = {
            "source_revision": receipt["revision"],
            "source_receipt_sha256": file_digest(args.source_receipt),
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "upstream": upstream,
            "archive_sha256": file_digest(args.archive),
            "checkpoint_sha256": file_digest(args.checkpoint),
            "evaluation_manifest_sha256": file_digest(args.evaluation_manifest),
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": "cpu",
                "threads": 1,
            },
            "case_count": len(names),
            "protected_cohort_read": False,
            "new_official_evaluation": False,
            "original_results_modified": False,
        }
        write_json_once(args.output / "preflight.json", preflight)
        stage = "opened-input-preparation"
        trajectories = source._load_named_trajectories(
            manifest, names, frame_count=500, node_count=12
        )
        raw = np.stack([trajectories[name] for name in names])
        initial, actions = causal_model_inputs(raw, config)
        with np.load(args.archive, allow_pickle=False) as archive:
            if archive["names"].tolist() != names:
                raise ValueError("archive identity order differs")
            archived_physical = archive["baseline_predictions"][
                :, : config.forecast_end
            ].copy()
            incumbent = archive["candidate_predictions"][
                :, : config.forecast_end
            ].copy()
            observed = archive["targets"][:, config.observation_frames][
                :, :, config.observed_nodes
            ].copy()
            full_prefix = archive["targets"][:, : config.prefix_length].copy()
        # All new inference receives only initial states, permitted prefix, and clamped actions.
        stage = "native-replay-and-controls"
        with torch.no_grad():
            rod = NativeRod(modules, torch, checkpoint["model_state_dict"], config)
            state = rod.initialize(initial)
            prefix, state = rod.rollout(state, actions[:, : config.prefix_length])
            snapshot = state.clone()
            future_actions = actions[:, config.prefix_length :]
            nominal, _ = rod.rollout(state, future_actions)
            nominal_all = np.concatenate((prefix, nominal), axis=1)
            zero = torch.zeros_like(snapshot.positions)
            zero_state = update_rod_state(
                snapshot, zero, zero, gain=0, clamped_nodes=config.clamped_nodes
            )
            zero_future, _ = rod.rollout(zero_state, future_actions)
            function, model = build_cpu_model(
                modules, torch, checkpoint["model_state_dict"]
            )
            legacy_result = legacy._rollout_arrays(
                {name: trajectories[name][: config.forecast_end + 2] for name in names},
                modules=modules,
                model=model,
                model_function=function,
                torch=torch,
                device="cpu",
            )
            legacy_points = np.asarray(legacy_result["predictions"])
            archive_error = nominal_all.astype(np.float64) - archived_physical
            incumbent_future = incumbent[:, config.prefix_length :]
            controls: dict[str, Any] = {
                "native_adapter_max_error_m": float(
                    np.max(np.abs(nominal_all - legacy_points))
                ),
                "archived_gpu_replay_max_error_m": float(np.max(np.abs(archive_error))),
                "archived_gpu_replay_coordinate_rmse_m": float(
                    np.sqrt(np.mean(archive_error**2))
                ),
                "zero_update_continuation_byte_identical": array_digest(nominal)
                == array_digest(zero_future),
                "incumbent_zero_update_returns_original_object": paired_physical_readout(
                    incumbent_future,
                    nominal,
                    zero_future,
                )
                is incumbent_future,
            }
            # A known smooth state perturbation is repaired without any real future labels.
            wave = np.sin(np.linspace(0, np.pi, config.node_count))[None, :, None]
            direction = np.array([0.6, -0.3, 0.74161985])[None, None, :]
            pose = np.repeat(wave * direction, len(names), axis=0)
            pose[:, config.clamped_nodes] = 0
            delta_x = torch.tensor(
                pose * protocol["controls"]["synthetic_pose_perturbation_m"],
                dtype=torch.float32,
            )
            delta_v = torch.tensor(
                pose * protocol["controls"]["synthetic_velocity_perturbation_m_s"],
                dtype=torch.float32,
            )
            perturbed = update_rod_state(
                snapshot, delta_x, delta_v, gain=1, clamped_nodes=config.clamped_nodes
            )
            short_actions = future_actions[
                :, : protocol["controls"]["synthetic_horizon"]
            ]
            damaged, _ = rod.rollout(perturbed, short_actions)
            repaired_state = update_rod_state(
                perturbed,
                snapshot.positions - perturbed.positions,
                snapshot.velocity - perturbed.velocity,
                gain=1,
                clamped_nodes=config.clamped_nodes,
            )
            repaired, _ = rod.rollout(repaired_state, short_actions)
            synthetic_reference = nominal[:, : len(short_actions[0])]
            before = float(
                np.linalg.norm(damaged.astype(np.float64) - synthetic_reference)
            )
            after = float(
                np.linalg.norm(repaired.astype(np.float64) - synthetic_reference)
            )
            controls.update(
                {
                    "synthetic_error_before_l2_m": before,
                    "synthetic_error_after_l2_m": after,
                    "synthetic_recovery_fraction": 1 - after / before
                    if before > 1e-12
                    else 0.0,
                }
            )
            thresholds = protocol["controls"]
            controls["passed"] = bool(
                controls["native_adapter_max_error_m"]
                <= thresholds["native_adapter_max_error_m"]
                and controls["archived_gpu_replay_max_error_m"]
                <= thresholds["archived_gpu_replay_max_error_m"]
                and controls["archived_gpu_replay_coordinate_rmse_m"]
                <= thresholds["archived_gpu_replay_coordinate_rmse_m"]
                and controls["zero_update_continuation_byte_identical"]
                and controls["incumbent_zero_update_returns_original_object"]
                and controls["synthetic_recovery_fraction"]
                >= thresholds["synthetic_minimum_recovery_fraction"]
            )
            write_json_once(args.output / "controls.json", controls)
            print(json.dumps({"stage": stage, **controls}), flush=True)
            if not controls["passed"]:
                raise ValueError("native state-restart control gate failed")
            stage = "matched-arm-predictions"
            base = incumbent[:, config.prefix_length :]
            predictions = {"incumbent": base, "physical_nominal": nominal}
            physical_dx, physical_dv = sparse_state_increments(prefix, observed, config)
            extra_dx, extra_dv = sparse_state_increments(
                incumbent[:, : config.prefix_length], observed, config
            )
            predictions["readout_sparse_pose"] = base + extra_dx[:, None]
            full_dx = full_prefix[:, -1].astype(np.float64) - prefix[:, -1]
            duration = (
                config.observation_frames[1] - config.observation_frames[0]
            ) * config.dt_s
            full_residual = full_prefix.astype(np.float64) - prefix
            full_dv = (
                full_residual[:, -1] - full_residual[:, config.observation_frames[0]]
            ) / duration
            full_dx[:, config.clamped_nodes] = 0
            full_dv[:, config.clamped_nodes] = 0
            specs = (
                (
                    "physical_sparse_pose",
                    physical_dx,
                    np.zeros_like(physical_dv),
                    1.0,
                    False,
                ),
                ("physical_sparse_pose_velocity", physical_dx, physical_dv, 1.0, False),
                (
                    "physical_full_pose_reference",
                    full_dx,
                    np.zeros_like(full_dv),
                    1.0,
                    False,
                ),
                ("physical_full_pose_velocity_reference", full_dx, full_dv, 1.0, False),
                (
                    "incumbent_propagated_pose",
                    extra_dx,
                    np.zeros_like(extra_dv),
                    1.0,
                    True,
                ),
                ("incumbent_propagated_pose_velocity", extra_dx, extra_dv, 1.0, True),
                (
                    "incumbent_propagated_pose_velocity_quarter",
                    extra_dx,
                    extra_dv,
                    0.25,
                    True,
                ),
            )
            for name, dx, dv, gain, paired in specs:
                updated = update_rod_state(
                    snapshot,
                    torch.tensor(dx, dtype=torch.float32),
                    torch.tensor(dv, dtype=torch.float32),
                    gain=gain,
                    clamped_nodes=config.clamped_nodes,
                )
                points, _ = rod.rollout(updated, future_actions)
                predictions[name] = (
                    paired_physical_readout(base, nominal, points) if paired else points
                )
                print(
                    json.dumps({"stage": stage, "arm": name, "cases": len(names)}),
                    flush=True,
                )
        if list(predictions) != protocol["arms"]:
            raise ValueError("prediction arms differ from frozen protocol")
        hashes = _save_predictions(args.output / "predictions.npz", predictions)
        barrier = {
            "schema": "deform-state-restart-prediction-barrier-v1",
            "case_count": len(names),
            "arms": list(predictions),
            "array_digests": hashes,
            "predictions_sha256": file_digest(args.output / "predictions.npz"),
            "controls_sha256": file_digest(args.output / "controls.json"),
            "preflight_sha256": file_digest(args.output / "preflight.json"),
            "source_revision": receipt["revision"],
            "new_metrics_computed": False,
            "protected_cohort_read": False,
            "no_replacement": True,
        }
        write_json_once(args.output / "prediction_barrier.json", barrier)
        stage = "exploratory-metrics"
        with np.load(args.archive, allow_pickle=False) as archive:
            truth = archive["targets"][
                :, config.prefix_length : config.forecast_end
            ].copy()
        if not np.array_equal(
            raw[:, config.prefix_length + 2 : config.forecast_end + 2], truth
        ):
            raise ValueError(
                "raw trajectory and frozen archive identity alignment differ"
            )
        result = aggregate_paired_metrics(predictions, truth, names, config)
        result.update(
            {
                "controls": controls,
                "prediction_barrier_sha256": file_digest(
                    args.output / "prediction_barrier.json"
                ),
                "source_revision": receipt["revision"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json_once(args.output / "result.json", result)
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "case_count": len(names) - 1,
                    "elapsed_seconds": result["elapsed_seconds"],
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
                "elapsed_seconds": time.perf_counter() - started,
                "protected_cohort_read": False,
                "original_results_modified": False,
                "retained_failure": True,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-source-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-receipt-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--evaluation-manifest-sha256")
    args = parser.parse_args()
    if args.freeze_source_only:
        freeze_source(args.output)
    else:
        for field in (
            "source_receipt",
            "source_receipt_sha256",
            "archive",
            "checkpoint",
            "upstream_root",
            "evaluation_manifest",
            "evaluation_manifest_sha256",
        ):
            if getattr(args, field) is None:
                parser.error(f"--{field.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
