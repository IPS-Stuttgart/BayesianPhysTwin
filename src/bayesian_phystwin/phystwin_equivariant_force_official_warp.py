"""Mechanical source-case evaluator for the equivariant-force Stage-2 gate."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_comparison import official_metrics_by_frame
from .phystwin_equivariant_force import canonicalize_force_edges
from .phystwin_equivariant_force_artifact import (
    load_equivariant_force_artifact,
)
from .phystwin_equivariant_force_data import (
    load_equivariant_force_episode,
    validate_force_episode_model_compatibility,
)
from .phystwin_equivariant_force_source import (
    load_equivariant_force_source_protocol,
)
from .phystwin_equivariant_force_stage2 import (
    EQUIVARIANT_FORCE_STAGE2_CONTRACT,
    fit_prefix_graph_persistence,
    load_equivariant_force_stage2_protocol,
    readout_correction_shrinkage,
    stage2_frame_intervals,
)
from .phystwin_equivariant_force_warp import (
    controller_attachment_matrix,
    rollout_equivariant_force_ensemble_segment,
)
from .phystwin_discrepancy_localization import _rollout_state_segment
from .phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from .phystwin_horizon_analysis import split_future_horizon
from .phystwin_residual_dynamics import (
    _load_pickle,
    _sha256,
    _target_validity,
)
from .phystwin_state_injection import (
    _initialize_simulator,
    _released_self_collision_for_case,
)


SEED_AGGREGATION = (
    "arithmetic_mean_force_field_per_frame_float64_then_float32"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def _case_members(
    competence: Mapping[str, Any],
    case_id: str,
    torch: Any,
    *,
    expected_seeds: tuple[int, ...],
    expected_model_config: Any,
    device: str,
) -> tuple[list[Any], list[np.ndarray], list[dict[str, Any]]]:
    matches = []
    for fold in competence.get("folds", ()):
        if case_id in fold.get("held_out_cases", ()):
            matches.append(fold)
    if len(matches) != 1:
        raise ValueError(f"{case_id}: competence record has no unique fold")
    fold = matches[0]
    seeds = tuple(int(item["seed"]) for item in fold.get("seeds", ()))
    if seeds != expected_seeds:
        raise ValueError(f"{case_id}: Stage-1 seed order changed")

    models = []
    latents = []
    provenance = []
    for seed_record in fold["seeds"]:
        seed = int(seed_record["seed"])
        held_matches = [
            value
            for value in seed_record.get("held_out", ())
            if value.get("case") == case_id
        ]
        if len(held_matches) != 1:
            raise ValueError(f"{case_id}: seed {seed} has no unique latent")
        held = held_matches[0]
        if held.get("adaptation", {}).get("future_frames_used") is not False:
            raise ValueError(f"{case_id}: seed {seed} latent crossed the prefix")
        latent_path = Path(held["latent_path"])
        if _sha256(latent_path) != held.get("latent_sha256"):
            raise ValueError(f"{case_id}: seed {seed} latent hash changed")
        with np.load(latent_path, allow_pickle=False) as archive:
            latent = np.asarray(archive["latent"], dtype=np.float32)
        if latent.shape != (expected_model_config.latent_dim,) or not np.all(
            np.isfinite(latent)
        ):
            raise ValueError(f"{case_id}: seed {seed} latent is invalid")

        model_record = seed_record["model_artifact"]
        manifest_path = Path(model_record["manifest_path"])
        if _sha256(manifest_path) != model_record.get("manifest_sha256"):
            raise ValueError(f"{case_id}: seed {seed} model manifest changed")
        artifact = load_equivariant_force_artifact(manifest_path)
        if artifact.config != expected_model_config:
            raise ValueError(f"{case_id}: seed {seed} model config changed")
        if f"source_episode_{case_id}" in artifact.source_checksums:
            raise ValueError(f"{case_id}: seed {seed} model trained on holdout")
        if case_id not in artifact.training_summary.get(
            "held_out_cases",
            (),
        ):
            raise ValueError(f"{case_id}: seed {seed} holdout provenance changed")
        model = artifact.instantiate(torch).to(device)
        model.eval()
        models.append(model)
        latents.append(latent)
        provenance.append(
            {
                "seed": seed,
                "model_artifact_id": artifact.artifact_id,
                "model_manifest_path": str(manifest_path.resolve()),
                "model_manifest_sha256": model_record["manifest_sha256"],
                "latent_path": str(latent_path.resolve()),
                "latent_sha256": held["latent_sha256"],
            }
        )
    return models, latents, provenance


def summarize_stage2_metrics(
    by_frame: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Reduce official per-frame arrays with the existing horizon thirds."""

    chamfer = np.asarray(by_frame["chamfer_distance_m"], dtype=float)
    track = np.asarray(by_frame["track_error_m"], dtype=float)
    if (
        chamfer.ndim != 1
        or track.shape != chamfer.shape
        or len(track) < 3
        or not np.all(np.isfinite(chamfer))
        or not np.all(np.isfinite(track))
    ):
        raise ValueError("Stage-2 metric arrays must be finite matching vectors")
    late = split_future_horizon(len(track))["late"]
    return {
        "chamfer_distance_m": float(np.mean(chamfer)),
        "track_error_m": float(np.mean(track)),
        "late_track_error_m": float(np.mean(track[late])),
    }


def _write_case_artifacts(
    output_dir: str | Path,
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "stage2_case_arrays.npz"
    with archive_path.open("wb") as handle:
        np.savez_compressed(handle, **dict(arrays))
    payload = {
        **dict(record),
        "array_archive": {
            "path": str(archive_path),
            "sha256": _sha256(archive_path),
        },
    }
    record_path = output / "stage2_case_record.json"
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **payload,
        "record_path": str(record_path),
        "record_sha256": _sha256(record_path),
    }


def evaluate_equivariant_force_official_warp_case(
    official_repo: str | Path,
    data_root: str | Path,
    episode_root: str | Path,
    competence_record_path: str | Path,
    source_protocol_path: str | Path,
    stage2_protocol_path: str | Path,
    case_id: str,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Evaluate one source case after, and only after, a Stage-1 pass."""

    source = load_equivariant_force_source_protocol(
        source_protocol_path,
        device=device,
    )
    stage2 = load_equivariant_force_stage2_protocol(
        stage2_protocol_path,
        source_protocol_path=source_protocol_path,
    )
    if case_id not in source.payload["source_cases"]:
        raise ValueError("Stage 2 accepts registered source cases only")
    competence = _load_json(competence_record_path)
    if competence.get("force_target_competence_passed") is not True:
        raise ValueError("Stage 1 did not pass; Stage 2 is blocked")
    if competence.get("target_artifacts_opened") is not False:
        raise ValueError("Stage-1 record crossed the target boundary")
    if competence.get("protocol_sha256") != stage2.source_protocol_sha256:
        raise ValueError("Stage-1 record used another source protocol")
    execution = competence.get("stage1_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Stage-1 execution provenance is missing")
    implementation_sha256 = execution.get("stage1_implementation_sha256")
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in implementation_sha256)
    ):
        raise ValueError("Stage-1 implementation identity is invalid")
    if execution.get("mode") not in {"serial", "registered_fold_merge"}:
        raise ValueError("Stage-1 execution mode is invalid")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("official-Warp Stage 2 requires PyTorch") from error
    models, latents, model_provenance = _case_members(
        competence,
        case_id,
        torch,
        expected_seeds=stage2.seeds,
        expected_model_config=source.model,
        device=device,
    )
    episode_path = Path(episode_root) / case_id / "force_episode"
    episode = load_equivariant_force_episode(episode_path)
    validate_force_episode_model_compatibility(episode, source.model)

    case_root = Path(data_root).resolve() / case_id
    episode_sources = {
        "baseline_trajectory": _sha256(case_root / "inference.pkl"),
        "final_data": _sha256(case_root / "final_data.pkl"),
        "optimal_params": _sha256(case_root / "optimal_params.pkl"),
    }
    if episode.source_checksums != episode_sources:
        raise ValueError("Stage-2 files differ from the force episode sources")
    data = _load_pickle(case_root / "final_data.pkl")
    optimal = _load_pickle(case_root / "optimal_params.pkl")
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    controllers = np.asarray(data["controller_points"], dtype=float)
    surface = np.asarray(data["surface_points"], dtype=float)
    interior = np.asarray(data["interior_points"], dtype=float)
    gt_track = np.asarray(
        _load_pickle(case_root / "gt_track_3d.pkl"),
        dtype=float,
    )
    split = _load_json(case_root / "split.json")
    train_start, train_end = (int(value) for value in split["train"])
    if train_start != 0 or train_end != episode.validation_end_frame:
        raise ValueError("released split differs from the force episode")
    expected_frame_dt = float(source.payload["official_warp"]["dt"]) * int(
        source.payload["official_warp"]["num_substeps"]
    )
    if not np.isclose(
        episode.frame_dt_s,
        expected_frame_dt,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Stage-2 frame timing differs from the force episode")
    fit_end = episode.fit_end_frame
    frames = stage2_frame_intervals(fit_end, train_end)
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    graph = build_phystwin_spring_graph(
        structure,
        controllers[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(
                optimal["controller_max_neighbours"]
            ),
        ),
    )
    if graph.num_object_points is None:
        raise RuntimeError("released graph omits its object boundary")
    object_count = int(graph.num_object_points)
    object_edges = canonicalize_force_edges(
        graph.springs[: graph.num_object_springs],
        num_nodes=object_count,
    )
    if not np.array_equal(object_edges, episode.object_edges):
        raise ValueError("Stage-2 graph differs from the force episode")
    if episode.positions_m.shape[1:] != (object_count, 3):
        raise ValueError("Stage-2 graph and episode state disagree")

    simulator, torch, wp, _ = _initialize_simulator(
        official_repo,
        data,
        optimal,
        case_root / "checkpoint.pth",
        graph,
        num_surface_points=observed.shape[1] + len(surface),
        original_count=observed.shape[1],
        dt=float(source.payload["official_warp"]["dt"]),
        num_substeps=int(source.payload["official_warp"]["num_substeps"]),
        self_collision=_released_self_collision_for_case(case_id),
        deterministic_spring_forces=True,
        spring_parameterization="grouped",
        device=device,
    )
    attachment, attachment_support = controller_attachment_matrix(
        graph.springs,
        num_object_nodes=object_count,
        num_control_nodes=controllers.shape[1],
    )
    common_rollout = {
        "start_frame": frames.simulator_step_frames[0],
        "stop_frame": train_end,
        "rest_positions_m": episode.rest_positions_m,
        "object_edges": episode.object_edges,
        "rest_lengths_m": episode.rest_lengths_m,
        "controller_points_m": controllers,
        "attachment_matrix": attachment,
        "support_prior": attachment_support,
        "regime_probabilities": episode.regime_probabilities,
        "latents": latents,
        "gravity_mps2": episode.gravity_mps2,
        "force_scale_sim": episode.force_scale_sim,
        "frame_dt_s": episode.frame_dt_s,
        "activity_speed_mps": float(
            source.payload["official_warp"]["activity_speed_mps"]
        ),
        "device": device,
    }
    reference, _, zero_history, zero_diagnostics = (
        rollout_equivariant_force_ensemble_segment(
            simulator,
            torch,
            wp,
            models,
            episode.positions_m[0],
            episode.velocities_mps[0],
            admission_weight=0.0,
            **common_rollout,
        )
    )
    if np.any(zero_history):
        raise AssertionError("zero admission emitted a nonzero force")
    simulator.clear_external_forces()
    wp.synchronize()
    direct_reference, _ = _rollout_state_segment(
        simulator,
        torch,
        wp,
        episode.positions_m[0],
        episode.velocities_mps[0],
        start_frame=frames.simulator_step_frames[0],
        stop_frame=train_end,
        device=device,
    )
    zero_force_parity = bool(np.array_equal(reference, direct_reference))
    if not zero_force_parity:
        raise AssertionError("repeated exact-zero official-Warp rollout changed")
    candidate, _, force_history, force_diagnostics = (
        rollout_equivariant_force_ensemble_segment(
            simulator,
            torch,
            wp,
            models,
            episode.positions_m[0],
            episode.velocities_mps[0],
            admission_weight=float(
                source.payload["official_warp"]["admission_weight"]
            ),
            **common_rollout,
        )
    )
    if len(reference) != train_end or len(candidate) != train_end:
        raise RuntimeError("Stage-2 rollout frame contract was violated")
    if not np.array_equal(reference[0], candidate[0]) or not np.array_equal(
        reference[0],
        episode.positions_m[0],
    ):
        raise AssertionError("candidate and reference initial states differ")
    maximum_allowed_force = (
        source.model.maximum_normalized_force
        * episode.force_scale_sim
        * float(source.payload["official_warp"]["admission_weight"])
    )
    if force_diagnostics.maximum_force_sim > maximum_allowed_force + 1.0e-6:
        raise AssertionError("Stage-2 force exceeded its frozen per-node bound")

    valid = _target_validity(visible, motion_valid)
    readout = stage2.payload["readout_refit"]
    fit_kwargs = {
        "graph_prior_strength": float(readout["graph_prior_strength"]),
        "maximum_residual_m": float(readout["maximum_residual_m"]),
    }
    reference_fit = fit_prefix_graph_persistence(
        observed[:fit_end],
        reference[:fit_end],
        valid[:fit_end],
        object_edges,
        **fit_kwargs,
    )
    candidate_fit = fit_prefix_graph_persistence(
        observed[:fit_end],
        candidate[:fit_end],
        valid[:fit_end],
        object_edges,
        **fit_kwargs,
    )
    shrinkage = readout_correction_shrinkage(
        candidate_fit.correction_m,
        reference_fit.correction_m,
        minimum_reference_rms_m=float(
            readout["minimum_reference_rms_m"]
        ),
    )
    reference_readout = reference.copy()
    candidate_readout = candidate.copy()
    reference_readout[fit_end:] += reference_fit.correction_m[None]
    candidate_readout[fit_end:] += candidate_fit.correction_m[None]
    metric_kwargs = {
        "object_points": observed,
        "object_visibilities": visible,
        "gt_track_3d": gt_track,
        "num_surface_points": observed.shape[1] + len(surface),
        "start_frame": fit_end,
        "end_frame": train_end,
    }
    reference_by_frame = official_metrics_by_frame(
        reference_readout,
        **metric_kwargs,
    )
    candidate_by_frame = official_metrics_by_frame(
        candidate_readout,
        **metric_kwargs,
    )
    reference_metrics = summarize_stage2_metrics(reference_by_frame)
    candidate_metrics = summarize_stage2_metrics(candidate_by_frame)

    record = {
        "schema_version": 1,
        "case_id": case_id,
        "stage2_execution_contract": EQUIVARIANT_FORCE_STAGE2_CONTRACT,
        "seed_aggregation": SEED_AGGREGATION,
        "frame_contract": {
            "initial_state_frame": frames.initial_state_frame,
            "first_simulator_step_frame": frames.simulator_step_frames[0],
            "fit_end_is_exclusive": True,
            "fit_end_frame": fit_end,
            "train_end_frame": train_end,
            "score_interval": "[fit_end_frame, train_end_frame)",
        },
        "target_artifacts_opened": False,
        "zero_force_bitwise_parity": zero_force_parity,
        "readout_correction_reference_supported": bool(
            shrinkage["reference_supported"]
        ),
        "readout_correction_shrinkage": float(
            shrinkage["readout_correction_shrinkage"]
        ),
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "readout_correction": {
            **shrinkage,
            "reference_laplacian_energy_m2": (
                reference_fit.laplacian_energy_m2
            ),
            "candidate_laplacian_energy_m2": (
                candidate_fit.laplacian_energy_m2
            ),
        },
        "force_rollout": asdict(force_diagnostics),
        "zero_force_rollout": asdict(zero_diagnostics),
        "model_members": model_provenance,
        "source_checksums": {
            "competence_record": _sha256(competence_record_path),
            "source_protocol": _sha256(source_protocol_path),
            "stage2_protocol": _sha256(stage2_protocol_path),
            "force_episode_manifest": _sha256(
                Path(episode_path).with_suffix(".json")
            ),
            "final_data": _sha256(case_root / "final_data.pkl"),
            "optimal_params": _sha256(case_root / "optimal_params.pkl"),
            "checkpoint": _sha256(case_root / "checkpoint.pth"),
            "gt_track_3d": _sha256(case_root / "gt_track_3d.pkl"),
        },
    }
    arrays = {
        "reference_physical_m": reference,
        "candidate_physical_m": candidate,
        "reference_readout_m": reference_readout,
        "candidate_readout_m": candidate_readout,
        "reference_correction_m": reference_fit.correction_m,
        "candidate_correction_m": candidate_fit.correction_m,
        "zero_force_history_sim": zero_history,
        "candidate_force_history_sim": force_history,
        "reference_chamfer_by_frame_m": reference_by_frame[
            "chamfer_distance_m"
        ],
        "candidate_chamfer_by_frame_m": candidate_by_frame[
            "chamfer_distance_m"
        ],
        "reference_track_by_frame_m": reference_by_frame["track_error_m"],
        "candidate_track_by_frame_m": candidate_by_frame["track_error_m"],
    }
    return _write_case_artifacts(output_dir, record, arrays)
