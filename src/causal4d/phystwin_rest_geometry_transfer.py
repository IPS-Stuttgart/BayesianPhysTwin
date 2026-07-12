"""Execute a locked source-to-target rest-geometry transfer in PhysTwin Warp."""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.phystwin_comparison import official_metrics_by_frame
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_residual_dynamics import _load_pickle, _sha256
from bayesian_phystwin.phystwin_state_injection import (
    _initialize_simulator,
    _released_self_collision_for_case,
    estimate_endpoint_velocity_delta,
)
from causal4d.phystwin_rest_geometry import _run_configured_restart
from causal4d.rest_geometry import (
    apply_frame_correction,
    reattach_controller_rest_lengths,
    rotate_vectors,
)
from causal4d.rest_geometry_cross_action import (
    build_rest_geometry_transfer_result_record,
)
from causal4d.rest_geometry_transfer import (
    attach_target_controller_to_canonical_graph,
    canonical_material_graph_sha256,
    load_canonical_material_graph,
    load_source_rest_geometry_correction,
    prepare_target_rest_geometry_configuration,
)


TRANSFER_METHODS = (
    "released",
    "endpoint_restart",
    "output_frame_graph",
    "frame_state_original_rest",
    "graph_state_original_rest",
    "rest_geometry_only",
    "frame_rest_geometry",
    "frame_rest_geometry_reattached",
    "selected_frame_rest_geometry",
)


def _require_plan_record(
    plan_record: Mapping[str, Any],
    *,
    source_execution_id: str,
    target_execution_id: str,
    selected_candidate_id: str,
) -> None:
    if plan_record.get("source_execution_id") != source_execution_id:
        raise ValueError("source correction does not match the transfer plan")
    if plan_record.get("target_execution_id") != target_execution_id:
        raise ValueError("target execution does not match the transfer plan")
    if plan_record.get("selected_candidate_id") != selected_candidate_id:
        raise ValueError("source correction does not use the frozen candidate")
    if plan_record.get("canonical_material_graph_required") is not True:
        raise ValueError("transfer plan did not require canonical material identity")
    target_prefix_allowed = bool(plan_record.get("target_response_prefix_allowed"))
    policy = plan_record.get("correction_evidence_policy")
    if target_prefix_allowed != (policy == "target_pre_holdout_only"):
        raise ValueError("transfer plan response-prefix policy is inconsistent")


def _metrics_by_frame(
    trajectory: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    gt_track: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    stop_frame: int,
) -> dict[str, np.ndarray]:
    official = official_metrics_by_frame(
        trajectory,
        observed,
        visible,
        gt_track,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=stop_frame,
    )
    return {
        "future_chamfer_distance_m": np.asarray(
            official["chamfer_distance_m"], dtype=float
        ),
        "future_track_error_m": np.asarray(
            official["track_error_m"], dtype=float
        ),
    }


def evaluate_phystwin_rest_geometry_transfer_case(
    official_repo: str | Path,
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    gt_track_path: str | Path,
    source_correction_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    plan_record: Mapping[str, Any],
    target_execution_id: str,
    rollout_start_frame: int,
    evaluation_start_frame: int | None = None,
    evaluation_stop_frame: int | None = None,
    velocity_history_frames: int = 3,
    dt: float = 5e-5,
    num_substeps: int = 667,
    self_collision: bool | None = None,
    deterministic_spring_forces: bool = True,
    device: str = "cuda:0",
    expected_protocol_id: str | None = None,
    expected_protocol_design_sha256: str | None = None,
    canonical_material_graph_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rerun one target with a correction inferred only on its paired source."""

    if velocity_history_frames < 2 or dt <= 0.0 or num_substeps < 1:
        raise ValueError("transfer rollout time settings are invalid")
    if device != "cuda:0":
        raise ValueError(
            "the pinned official simulator selects cuda:0; use CUDA_VISIBLE_DEVICES"
        )
    source = load_source_rest_geometry_correction(
        source_correction_manifest_path
    )
    if expected_protocol_id is not None and source.protocol_id != expected_protocol_id:
        raise ValueError("source correction belongs to a different protocol")
    if (
        expected_protocol_design_sha256 is not None
        and source.protocol_design_sha256 != expected_protocol_design_sha256
    ):
        raise ValueError("source correction protocol design digest changed")
    _require_plan_record(
        plan_record,
        source_execution_id=source.source_execution_id,
        target_execution_id=target_execution_id,
        selected_candidate_id=source.selected_candidate_id,
    )
    contact_policy = str(plan_record["contact_policy"])
    data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_params_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    controller = np.asarray(data["controller_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    stop_frame = frame_count if evaluation_stop_frame is None else evaluation_stop_frame
    score_start = (
        rollout_start_frame
        if evaluation_start_frame is None
        else evaluation_start_frame
    )
    if not velocity_history_frames <= rollout_start_frame < frame_count:
        raise ValueError("target rollout start cannot support state initialization")
    if not rollout_start_frame <= score_start < stop_frame <= frame_count:
        raise ValueError("target evaluation window is invalid")
    if baseline.shape[0] < frame_count:
        raise ValueError("target baseline trajectory has too few frames")
    baseline = baseline[:frame_count]
    structure_points = np.concatenate((observed[0], surface, interior), axis=0)
    if baseline.shape[1] != len(structure_points):
        raise ValueError("target baseline and reconstructed object disagree")
    if self_collision is None:
        self_collision = _released_self_collision_for_case(
            Path(final_data_path).resolve().parent.name
        )
    graph_config = PhysTwinSpringGraphConfig(
        object_radius=float(optimal["object_radius"]),
        object_max_neighbours=int(optimal["object_max_neighbours"]),
        controller_radius=float(optimal["controller_radius"]),
        controller_max_neighbours=int(optimal["controller_max_neighbours"]),
    )
    if canonical_material_graph_path is None:
        graph = build_phystwin_spring_graph(
            structure_points,
            controller[0],
            config=graph_config,
        )
    else:
        canonical_graph = load_canonical_material_graph(
            canonical_material_graph_path
        )
        if len(canonical_graph.vertices) != len(structure_points):
            raise ValueError("canonical graph and target object size disagree")
        graph = attach_target_controller_to_canonical_graph(
            canonical_graph,
            controller[0],
            config=graph_config,
        )
    material_digest = canonical_material_graph_sha256(
        graph.vertices[: len(structure_points)],
        graph.springs,
        graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
    )
    frame_dt = dt * num_substeps
    start_index = rollout_start_frame - 1
    target_velocity = estimate_endpoint_velocity_delta(
        baseline[
            rollout_start_frame
            - velocity_history_frames : rollout_start_frame
        ],
        frame_dt=frame_dt,
    )
    selected_configuration = prepare_target_rest_geometry_configuration(
        source,
        target_material_graph_sha256=material_digest,
        target_position=baseline[start_index],
        target_velocity=target_velocity,
        target_controller_points=controller,
        target_springs=graph.springs,
        target_released_rest_lengths=graph.rest_lengths,
        num_object_springs=graph.num_object_springs,
        contact_policy=contact_policy,
    )
    corrected_controller = apply_frame_correction(controller, source.frame)
    frame_position = apply_frame_correction(baseline[start_index], source.frame)
    full_position = frame_position + (
        source.hyperparameters["rest_geometry_scale"] * source.nonrigid_field
    )
    rotated_velocity = rotate_vectors(target_velocity, source.frame)
    preserve_rest = np.asarray(graph.rest_lengths, dtype=float).copy()
    preserve_rest[: graph.num_object_springs] = (
        source.corrected_object_rest_lengths
    )
    reattached_rest, _, _ = reattach_controller_rest_lengths(
        source.corrected_reference_vertices,
        corrected_controller[0],
        graph.springs,
        preserve_rest,
        num_object_springs=graph.num_object_springs,
        maximum_log_ratio=float(
            np.log(source.hyperparameters["rest_length_ratio_bound"])
        ),
    )

    warnings.filterwarnings(
        "ignore",
        message=(
            "Running the tape backwards may produce incorrect gradients because "
            "recorded kernel set_control_points.*"
        ),
    )
    simulator, torch, wp, _ = _initialize_simulator(
        official_repo,
        data,
        optimal,
        checkpoint_path,
        graph,
        num_surface_points=original_count + len(surface),
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        device=device,
    )
    candidates = {"released": baseline.copy()}

    def add_rollout(
        method: str,
        *,
        rest_lengths: np.ndarray,
        controller_points: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        candidate = baseline.copy()
        candidate[start_index] = position
        candidate[rollout_start_frame:] = _run_configured_restart(
            simulator,
            torch,
            wp,
            rest_lengths=rest_lengths,
            controller_points=controller_points,
            position=position,
            velocity=velocity,
            start_frame=rollout_start_frame,
            stop_frame=frame_count,
            device=device,
        )
        candidates[method] = candidate

    add_rollout(
        "endpoint_restart",
        rest_lengths=graph.rest_lengths,
        controller_points=controller,
        position=baseline[start_index],
        velocity=target_velocity,
    )
    output_candidate = baseline.copy()
    output_candidate[start_index:] = apply_frame_correction(
        baseline[start_index:], source.frame
    ) + source.hyperparameters["rest_geometry_scale"] * source.nonrigid_field[None]
    candidates["output_frame_graph"] = output_candidate
    add_rollout(
        "frame_state_original_rest",
        rest_lengths=graph.rest_lengths,
        controller_points=corrected_controller,
        position=frame_position,
        velocity=rotated_velocity,
    )
    add_rollout(
        "graph_state_original_rest",
        rest_lengths=graph.rest_lengths,
        controller_points=corrected_controller,
        position=full_position,
        velocity=rotated_velocity,
    )
    add_rollout(
        "rest_geometry_only",
        rest_lengths=preserve_rest,
        controller_points=controller,
        position=baseline[start_index],
        velocity=target_velocity,
    )
    add_rollout(
        "frame_rest_geometry",
        rest_lengths=preserve_rest,
        controller_points=corrected_controller,
        position=full_position,
        velocity=rotated_velocity,
    )
    add_rollout(
        "frame_rest_geometry_reattached",
        rest_lengths=reattached_rest,
        controller_points=corrected_controller,
        position=full_position,
        velocity=rotated_velocity,
    )
    selected_method = (
        "frame_rest_geometry_reattached"
        if selected_configuration.controller_attachment_policy
        == "rebuild_on_corrected_target_contact"
        else "frame_rest_geometry"
    )
    candidates["selected_frame_rest_geometry"] = candidates[selected_method].copy()
    if set(candidates) != set(TRANSFER_METHODS):
        raise RuntimeError("transfer rollout method bank is incomplete")

    num_surface_points = original_count + len(surface)
    metrics_by_method = {
        method: _metrics_by_frame(
            trajectory,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=score_start,
            stop_frame=stop_frame,
        )
        for method, trajectory in candidates.items()
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rollout_path = output / "target_transfer_rollouts.npz"
    np.savez_compressed(
        rollout_path,
        **{
            f"trajectory__{method}": trajectory[rollout_start_frame:]
            for method, trajectory in candidates.items()
        },
    )
    result_record = build_rest_geometry_transfer_result_record(
        plan_record,
        metrics_by_method,
        canonical_material_graph_sha256=material_digest,
        source_correction_sha256=source.source_manifest_sha256,
        target_rollout_bundle_sha256=_sha256(rollout_path),
    )
    record_path = output / "target_transfer_result_record.json"
    record_path.write_text(
        json.dumps(result_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "method": "source-to-target graph/rest-geometry Warp transfer",
        "target_execution_id": target_execution_id,
        "source_execution_id": source.source_execution_id,
        "plan_record_id": plan_record["record_id"],
        "selected_candidate_id": source.selected_candidate_id,
        "contact_policy": contact_policy,
        "controller_attachment_policy": (
            selected_configuration.controller_attachment_policy
        ),
        "canonical_material_graph_sha256": material_digest,
        "rollout_start_frame": rollout_start_frame,
        "evaluation_frames": [score_start, stop_frame],
        "information_boundary": {
            "source_correction_evidence": plan_record[
                "correction_evidence_policy"
            ],
            "target_response_prefix_used": bool(
                plan_record["target_response_prefix_allowed"]
            ),
            "target_holdout_frames_used_for_correction": False,
            "future_controls": "recorded target controller trajectory",
            "future_observations": "none",
        },
        "inputs": {
            "final_data_sha256": _sha256(final_data_path),
            "baseline_trajectory_sha256": _sha256(baseline_trajectory_path),
            "optimal_params_sha256": _sha256(optimal_params_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "gt_track_sha256": _sha256(gt_track_path),
            "source_correction_manifest_sha256": source.source_manifest_sha256,
            "canonical_material_graph_file_sha256": (
                None
                if canonical_material_graph_path is None
                else _sha256(canonical_material_graph_path)
            ),
        },
        "mean_metrics_by_method": {
            method: {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            for method, metrics in metrics_by_method.items()
        },
        "outputs": {
            "rollouts": str(rollout_path.resolve()),
            "result_record": str(record_path.resolve()),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    simulator = None
    gc.collect()
    torch.cuda.empty_cache()
    return {**summary, "summary_path": str(summary_path.resolve())}
