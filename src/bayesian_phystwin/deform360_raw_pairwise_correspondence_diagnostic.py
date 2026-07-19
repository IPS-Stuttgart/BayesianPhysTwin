"""Frozen pairwise-correspondence gate on raw Deform360 AllTracker data.

The raw multiview measurement artifact is checksum-validated before an
already-open development target is loaded.  The correspondence detector and
its thresholds are imported unchanged from the synthetic open-27 diagnostic.
No target error is used by the selector, detector, or routing policy.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cpd_registration import NonrigidCpdConfig, fit_nonrigid_cpd
from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_online_belief_evaluation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _physical_object_cluster_bootstrap,
    _post_update_scored_frames,
    _relative_change,
    score_deform360_hidden_trajectory,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    _load_measurement_artifact,
    _load_open_case_for_evaluation,
    _sha256,
    _validate_prediction_seal,
    expected_open_case_names,
)
from .phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
    detect_pairwise_consensus_correspondences,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


PROTOCOL_ID = "deform360-open27-raw-alltracker-pairwise-gate-v1-development"
MINIMUM_SELECTOR_SUPPORT = 3
PHYSICAL_ARM = "physical_prior"
PERSISTENCE_ARM = "persistence"
SELECTED_RAW_ARM = "selected_raw_backbone_persistence_insufficient_default"
LEGACY_PHYSICAL_DEFAULT_ARM = "selected_raw_backbone_legacy_physical_default"
UNGATED_RBF_ARM = "raw_selected_backbone_full_blend_rbf_support_gated"
CLIQUE_RBF_ARM = "raw_selected_backbone_full_blend_rbf_pairwise_clique"
CPD_ARM = "raw_independent_cpd_selected_backbone"
ASSOCIATION_ADAPTIVE_ARM = "raw_association_adaptive_rbf_or_cpd"
ARMS = (
    PHYSICAL_ARM,
    PERSISTENCE_ARM,
    SELECTED_RAW_ARM,
    LEGACY_PHYSICAL_DEFAULT_ARM,
    UNGATED_RBF_ARM,
    CLIQUE_RBF_ARM,
    CPD_ARM,
    ASSOCIATION_ADAPTIVE_ARM,
)


def _corrected_frame(
    backbone_frame_m: np.ndarray,
    correction_m: np.ndarray,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    return (
        np.asarray(backbone_frame_m, dtype=float)
        + np.asarray(correction_m, dtype=float)
    ).astype(dtype, copy=False)


def evaluate_raw_pairwise_correspondence_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: Sequence[int],
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
    cpd_config: NonrigidCpdConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Apply the frozen clique gate and explicit routing controls."""

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence = np.asarray(persistence_input, dtype=float)
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    measurement = np.asarray(measurement_m, dtype=float)
    measurement_visible = np.asarray(measurement_visibility, dtype=bool)
    measurement_valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    frames = tuple(int(frame) for frame in scored_frames)
    if prior.shape != persistence.shape or prior.shape != target.shape:
        raise ValueError("prior, persistence, and target shapes differ")
    if prior.ndim != 3 or prior.shape[2] != 3:
        raise ValueError("trajectories must have shape (T, N, 3)")
    if measurement.shape != prior.shape:
        raise ValueError("measurement must match the trajectory shape")
    for name, mask in (
        ("visibility", visible),
        ("validity", valid),
        ("measurement_visibility", measurement_visible),
        ("measurement_validity", measurement_valid),
    ):
        if mask.shape != prior.shape[:2]:
            raise ValueError(f"{name} must have shape (T, N)")
    if centers.ndim != 1 or len(centers) != len(np.unique(centers)):
        raise ValueError("center_ids must be a unique vector")
    if np.any(centers < 0) or np.any(centers >= prior.shape[1]):
        raise ValueError("centre ID exceeds trajectory")
    if not np.array_equal(
        prior_input[0].astype(np.float32), target[0].astype(np.float32)
    ):
        raise ValueError("frame-zero material identities differ")

    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    cpd_cfg = cpd_config or NonrigidCpdConfig()
    backbones = {PHYSICAL_ARM: prior, PERSISTENCE_ARM: persistence}

    def initial_belief(backbone: np.ndarray):
        return initialize_recursive_rbf_belief(
            centers,
            backbone[0, centers],
            backbone[0],
            config=belief_cfg,
        )

    ungated_states = {
        name: initial_belief(backbone) for name, backbone in backbones.items()
    }
    clique_states = {
        name: initial_belief(backbone) for name, backbone in backbones.items()
    }
    dynamic_arms = (
        SELECTED_RAW_ARM,
        LEGACY_PHYSICAL_DEFAULT_ARM,
        UNGATED_RBF_ARM,
        CLIQUE_RBF_ARM,
        CPD_ARM,
        ASSOCIATION_ADAPTIVE_ARM,
    )
    trajectories = {arm: prior_input.copy() for arm in dynamic_arms}
    output_dtype = prior_input.dtype
    updates: list[dict[str, Any]] = []

    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(prior)
        )
        available = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        observed = measurement[update, available_ids]
        selector_support = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        if selector_support:
            chamfer = {
                name: _symmetric_set_chamfer_m(
                    backbone[update, available_ids], observed
                )
                for name, backbone in backbones.items()
            }
            selected_name = min(
                (PHYSICAL_ARM, PERSISTENCE_ARM),
                key=lambda name: (
                    chamfer[name],
                    0 if name == PHYSICAL_ARM else 1,
                ),
            )
        else:
            chamfer = (
                {
                    name: _symmetric_set_chamfer_m(
                        backbone[update, available_ids], observed
                    )
                    for name, backbone in backbones.items()
                }
                if len(available_ids)
                else {PHYSICAL_ARM: None, PERSISTENCE_ARM: None}
            )
            selected_name = PERSISTENCE_ARM
        selected = backbones[selected_name]
        legacy = backbones[PHYSICAL_ARM] if not selector_support else selected
        for arm in dynamic_arms:
            trajectories[arm][update + 1 : stop] = selected[update + 1 : stop]
        trajectories[LEGACY_PHYSICAL_DEFAULT_ARM][update + 1 : stop] = legacy[
            update + 1 : stop
        ]

        residuals: dict[str, np.ndarray] = {}
        gates = {}
        for backbone_name, backbone in backbones.items():
            residual = np.full((len(centers), 3), np.nan, dtype=float)
            residual[available] = observed - backbone[update, available_ids]
            residuals[backbone_name] = residual
            gates[backbone_name] = detect_pairwise_consensus_correspondences(
                backbone[update, centers],
                measurement[update, centers],
                available,
                material_ids=centers,
                config=gate_cfg,
            )

        if selector_support:
            for backbone_name, backbone in backbones.items():
                ungated_states[backbone_name], _ = update_recursive_rbf_belief(
                    ungated_states[backbone_name],
                    update,
                    backbone[update, centers],
                    residuals[backbone_name],
                    available.copy(),
                    config=belief_cfg,
                )
                gate = gates[backbone_name]
                if gate.accepted:
                    clique_states[backbone_name], _ = update_recursive_rbf_belief(
                        clique_states[backbone_name],
                        update,
                        backbone[update, centers],
                        residuals[backbone_name],
                        gate.inlier_mask.copy(),
                        config=belief_cfg,
                    )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    ungated_states[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                trajectories[UNGATED_RBF_ARM][frame] = _corrected_frame(
                    selected[frame], decoded.mean_m, dtype=output_dtype
                )

        selected_gate = gates[selected_name]
        if selector_support and selected_gate.accepted:
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    clique_states[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                trajectories[CLIQUE_RBF_ARM][frame] = _corrected_frame(
                    selected[frame], decoded.mean_m, dtype=output_dtype
                )
        elif not np.array_equal(
            trajectories[CLIQUE_RBF_ARM][update + 1 : stop],
            selected[update + 1 : stop],
        ):
            raise AssertionError("clique abstention did not preserve selected backbone")

        transform = None
        fit_error = None
        if selector_support:
            try:
                transform = fit_nonrigid_cpd(
                    selected[update, available_ids],
                    observed,
                    config=cpd_cfg,
                )
                for frame in range(update + 1, stop):
                    trajectories[CPD_ARM][frame] = transform.transform(
                        selected[frame]
                    ).astype(output_dtype, copy=False)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                fit_error = f"{type(error).__name__}: {error}"
        if transform is None and not np.array_equal(
            trajectories[CPD_ARM][update + 1 : stop],
            selected[update + 1 : stop],
        ):
            raise AssertionError("CPD failure did not preserve selected backbone")

        if selector_support and selected_gate.accepted:
            adaptive_route = "pairwise_consensus_rbf"
            routed = trajectories[CLIQUE_RBF_ARM]
        elif transform is not None:
            adaptive_route = "unordered_cpd"
            routed = trajectories[CPD_ARM]
        else:
            adaptive_route = "selected_raw_backbone"
            routed = selected
        trajectories[ASSOCIATION_ADAPTIVE_ARM][update + 1 : stop] = routed[
            update + 1 : stop
        ]
        if not np.array_equal(
            trajectories[ASSOCIATION_ADAPTIVE_ARM][update + 1 : stop],
            routed[update + 1 : stop],
        ):
            raise AssertionError("association-adaptive route is not bit-exact")

        updates.append(
            {
                "frame": int(update),
                "interval_end_exclusive": int(stop),
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": selector_support,
                "selected_backbone": selected_name,
                "selector_decision": (
                    "current_observation_chamfer"
                    if selector_support
                    else "insufficient_support_persistence_default"
                ),
                "legacy_physical_default_backbone": (
                    PHYSICAL_ARM if not selector_support else selected_name
                ),
                "current_observation_chamfer_m": chamfer,
                "selected_pairwise_gate": {
                    "accepted": bool(selector_support and selected_gate.accepted),
                    "decision": (
                        selected_gate.decision
                        if selector_support
                        else "insufficient_selector_support"
                    ),
                    "inlier_count": selected_gate.inlier_count,
                    "inlier_fraction": selected_gate.inlier_fraction,
                    "compatible_pair_fraction": (
                        selected_gate.compatible_pair_fraction
                    ),
                    "bit_exact_raw_fallback": bool(
                        not (selector_support and selected_gate.accepted)
                        and np.array_equal(
                            trajectories[CLIQUE_RBF_ARM][update + 1 : stop],
                            selected[update + 1 : stop],
                        )
                    ),
                },
                "cpd": {
                    "fit_performed": transform is not None,
                    "fit_error": fit_error,
                    "effective_correspondence_count": (
                        None
                        if transform is None
                        else transform.effective_correspondence_count
                    ),
                },
                "association_adaptive": {
                    "route": adaptive_route,
                    "bit_exact_selected_route": True,
                },
            }
        )

    all_trajectories = {
        PHYSICAL_ARM: prior_input.copy(),
        PERSISTENCE_ARM: persistence_input.copy(),
        **trajectories,
    }
    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visible,
            valid,
            center_ids=centers,
            scored_frames=frames,
        )
        for arm, trajectory in all_trajectories.items()
    }
    measurement_errors: list[float] = []
    for update in UPDATE_FRAMES:
        supported = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & visible[update, centers]
            & valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(target[update, centers]), axis=1)
        )
        measurement_errors.extend(
            np.linalg.norm(
                measurement[update, centers[supported]]
                - target[update, centers[supported]],
                axis=1,
            ).tolist()
        )
    error = np.asarray(measurement_errors, dtype=float)
    report = {
        "protocol_id": PROTOCOL_ID,
        "center_ids": centers.tolist(),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": list(frames),
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "cpd_config": asdict(cpd_cfg),
        "updates": updates,
        "scores": scores,
        "raw_measurement_target_open_audit": {
            "count": len(error),
            "mean_error_m": None if not len(error) else float(np.mean(error)),
            "median_error_m": None if not len(error) else float(np.median(error)),
            "p90_error_m": None if not len(error) else float(np.quantile(error, 0.9)),
            "maximum_error_m": None if not len(error) else float(np.max(error)),
            "used_for_threshold_or_method_selection": False,
        },
        "method_contract": {
            "selector_minimum_support": MINIMUM_SELECTOR_SUPPORT,
            "selector_insufficient_default": PERSISTENCE_ARM,
            "legacy_physical_default_arm": LEGACY_PHYSICAL_DEFAULT_ARM,
            "clique_gate": asdict(gate_cfg),
            "clique_threshold_frozen_before_raw_target_scoring": True,
            "backbone_states": "separate physical and persistence RBF states",
            "clique_rejection": "bit-exact selected raw backbone",
            "association_adaptive": (
                "clique RBF if accepted; otherwise independent unordered CPD; "
                "otherwise bit-exact selected raw backbone"
            ),
        },
    }
    return report, all_trajectories


def evaluate_raw_pairwise_correspondence_case(
    panel_case_dir: str | Path,
    measurement_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one causal measurement artifact, then score its open outcome."""

    case_dir = Path(panel_case_dir).resolve()
    measurement_path = Path(measurement_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    measurement_manifest, measurement_arrays = _load_measurement_artifact(
        case_dir,
        measurement_path,
        seal,
    )
    boundary = measurement_manifest.get("information_boundary", {})
    if (
        boundary.get("target_data_read") is not False
        or boundary.get("outcome_manifest_read") is not False
    ):
        raise ValueError("raw measurement artifact crossed the target boundary")
    open_seal, prior, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(case_dir)
    )
    if open_seal != seal:
        raise ValueError("prediction seal changed while opening the outcome")
    scored_frames = _post_update_scored_frames(len(target))
    report, trajectories = evaluate_raw_pairwise_correspondence_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
        measurement_arrays["measurement_m"],
        measurement_arrays["measurement_visibility"],
        measurement_arrays["measurement_validity"],
        center_ids=measurement_arrays["center_ids"],
        scored_frames=scored_frames,
    )
    report.update(
        {
            "case": case_dir.name,
            "object_id": str(seal["object_id"]),
            "episode_id": int(seal["episode_id"]),
            "measurement_manifest_sha256": _sha256(
                measurement_path / MANIFEST_FILENAME
            ),
            "measurement_result_sha256": measurement_manifest["result_sha256"],
            "information_boundary": {
                "measurement_verified_before_target_open": True,
                "measurement_builder_target_read": False,
                "target_role": "scoring audit only; no tuning",
            },
        }
    )
    return report, trajectories


def _comparison(
    reports: Sequence[Mapping[str, Any]],
    groups: Mapping[str, str],
    aggregate: Mapping[str, Mapping[str, float]],
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        differences = {
            str(report["case"]): float(
                report["scores"][candidate][metric]
                - report["scores"][comparator][metric]
            )
            for report in reports
        }
        result = _physical_object_cluster_bootstrap(differences, groups)
        result["relative_change"] = _relative_change(
            aggregate[candidate][metric], aggregate[comparator][metric]
        )
        result["episode_wins"] = int(
            np.sum(np.asarray(list(differences.values())) < 0.0)
        )
        output[metric] = result
    return output


def evaluate_raw_pairwise_correspondence_cohort(
    panel_root: str | Path,
    measurement_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate exactly the open-27 raw AllTracker measurement artifacts."""

    panel = Path(panel_root).resolve()
    measurements = Path(measurement_root).resolve()
    output = Path(output_dir).resolve()
    expected = expected_open_case_names()
    missing = [
        case
        for case in expected
        if not (panel / case).is_dir()
        or not (measurements / case / MANIFEST_FILENAME).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing open raw measurement cases: {missing}")
    output.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, str]] = []
    for case in expected:
        report, trajectories = evaluate_raw_pairwise_correspondence_case(
            panel / case, measurements / case
        )
        report_path = output / f"{case}.json"
        archive_path = output / f"{case}.npz"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(archive_path, **trajectories)
        reports.append(report)
        groups[case] = str(report["object_id"])
        artifacts.append(
            {
                "case": case,
                "report_sha256": _sha256(report_path),
                "archive_sha256": _sha256(archive_path),
            }
        )

    aggregate = {
        arm: {
            metric: float(
                np.mean([report["scores"][arm][metric] for report in reports])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in ARMS
    }
    pairs = (
        (CLIQUE_RBF_ARM, UNGATED_RBF_ARM),
        (CLIQUE_RBF_ARM, CPD_ARM),
        (CLIQUE_RBF_ARM, SELECTED_RAW_ARM),
        (CLIQUE_RBF_ARM, PHYSICAL_ARM),
        (CLIQUE_RBF_ARM, PERSISTENCE_ARM),
        (ASSOCIATION_ADAPTIVE_ARM, UNGATED_RBF_ARM),
        (ASSOCIATION_ADAPTIVE_ARM, CPD_ARM),
        (ASSOCIATION_ADAPTIVE_ARM, CLIQUE_RBF_ARM),
        (ASSOCIATION_ADAPTIVE_ARM, SELECTED_RAW_ARM),
        (UNGATED_RBF_ARM, CPD_ARM),
        (UNGATED_RBF_ARM, SELECTED_RAW_ARM),
        (UNGATED_RBF_ARM, PHYSICAL_ARM),
        (UNGATED_RBF_ARM, PERSISTENCE_ARM),
        (CPD_ARM, SELECTED_RAW_ARM),
        (CPD_ARM, PHYSICAL_ARM),
        (CPD_ARM, PERSISTENCE_ARM),
        (SELECTED_RAW_ARM, LEGACY_PHYSICAL_DEFAULT_ARM),
    )
    comparisons = {
        f"{candidate}_vs_{comparator}": _comparison(
            reports, groups, aggregate, candidate, comparator
        )
        for candidate, comparator in pairs
    }
    updates = [update for report in reports for update in report["updates"]]
    accepted = [update["selected_pairwise_gate"]["accepted"] for update in updates]
    route_counts = {
        route: int(
            sum(update["association_adaptive"]["route"] == route for update in updates)
        )
        for route in (
            "pairwise_consensus_rbf",
            "unordered_cpd",
            "selected_raw_backbone",
        )
    }
    error_counts = np.asarray(
        [report["raw_measurement_target_open_audit"]["count"] for report in reports],
        dtype=float,
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "gate_config": reports[0]["gate_config"],
        "belief_config": reports[0]["belief_config"],
        "cpd_config": reports[0]["cpd_config"],
        "aggregate": aggregate,
        "comparisons": comparisons,
        "coverage": {
            "accepted_update_count": int(np.sum(accepted)),
            "update_count": len(updates),
            "clique_rbf_correction_coverage": float(np.mean(accepted)),
            "insufficient_selector_support_count": int(
                sum(not update["selector_support_sufficient"] for update in updates)
            ),
            "association_adaptive_routes": route_counts,
            "clique_rejected_exact_raw_fallback_count": int(
                sum(
                    update["selected_pairwise_gate"]["bit_exact_raw_fallback"]
                    for update in updates
                    if not update["selected_pairwise_gate"]["accepted"]
                )
            ),
        },
        "selector_counts": {
            name: int(sum(update["selected_backbone"] == name for update in updates))
            for name in (PHYSICAL_ARM, PERSISTENCE_ARM)
        },
        "raw_measurement_target_open_audit": {
            "pooled_count": int(np.sum(error_counts)),
            "count_weighted_mean_error_m": float(
                np.sum(
                    error_counts
                    * np.asarray(
                        [
                            report["raw_measurement_target_open_audit"]["mean_error_m"]
                            for report in reports
                        ]
                    )
                )
                / np.sum(error_counts)
            ),
            "used_for_threshold_or_method_selection": False,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "physical object",
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "post-hoc raw-camera diagnostic on the already-open Deform360 27; "
            "clique thresholds were frozen on synthetic source-only geometry before "
            "this raw target scoring; not held-target or official benchmark evidence"
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "ARMS",
    "ASSOCIATION_ADAPTIVE_ARM",
    "CLIQUE_RBF_ARM",
    "CPD_ARM",
    "LEGACY_PHYSICAL_DEFAULT_ARM",
    "PROTOCOL_ID",
    "SELECTED_RAW_ARM",
    "UNGATED_RBF_ARM",
    "evaluate_raw_pairwise_correspondence_arrays",
    "evaluate_raw_pairwise_correspondence_case",
    "evaluate_raw_pairwise_correspondence_cohort",
]
