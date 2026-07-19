"""Robust-correspondence development diagnostic on the open Deform360 27.

This module evaluates a fixed, outcome-free pairwise-strain consensus gate for
the selected-backbone, full-blend Euclidean RBF arm.  Only the current 16
assimilation centres are corrupted or inspected.  Those identities remain
permanently excluded from hidden-point scoring.  A rejected update copies the
current-observation-selected physical or persistence trajectory bit-for-bit.

The cohort is hard-coded through the audited online-belief loader; this module
does not discover or inspect any held calibration or target episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from .cpd_registration import NonrigidCpdConfig, fit_nonrigid_cpd
from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_online_belief_evaluation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CENTER_COUNT,
    EXPECTED_SOURCE_EPISODES,
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _expected_episode_directories,
    _physical_object_cluster_bootstrap,
    _relative_change,
    _sha256,
    evaluate_deform360_online_belief_case,
    score_deform360_hidden_trajectory,
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


PROTOCOL_ID = "deform360-open27-pairwise-correspondence-gate-v1-development"
CORRUPTION_SEEDS = tuple(range(8))

PHYSICAL_ARM = "physical_prior"
PERSISTENCE_ARM = "persistence"
SELECTED_RAW_ARM = "selected_raw_backbone"
UNGATED_RBF_ARM = "selected_backbone_full_blend_euclidean_rbf_ungated"
LEGACY_MIXED_UNGATED_RBF_ARM = (
    "legacy_mixed_state_selected_backbone_full_blend_euclidean_rbf_ungated"
)
ROBUST_RBF_ARM = "selected_backbone_full_blend_euclidean_rbf_pairwise_consensus"
CPD_ARM = "independent_cpd_selected_backbone"
ARMS = (
    PHYSICAL_ARM,
    PERSISTENCE_ARM,
    SELECTED_RAW_ARM,
    UNGATED_RBF_ARM,
    LEGACY_MIXED_UNGATED_RBF_ARM,
    ROBUST_RBF_ARM,
    CPD_ARM,
)


@dataclass(frozen=True)
class MatchedObservationStress:
    """Fixed synthetic stress on assigned current-centre observations."""

    name: str
    gaussian_sigma_m: float = 0.0
    mismatch_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stress name must be nonempty")
        if not np.isfinite(self.gaussian_sigma_m) or self.gaussian_sigma_m < 0.0:
            raise ValueError("gaussian_sigma_m must be nonnegative")
        if not np.isfinite(self.mismatch_fraction) or not (
            0.0 <= self.mismatch_fraction <= 1.0
        ):
            raise ValueError("mismatch_fraction must lie in [0, 1]")
        if self.gaussian_sigma_m > 0.0 and self.mismatch_fraction > 0.0:
            raise ValueError("this bounded diagnostic varies one stress at a time")


OBSERVATION_STRESSES = (
    MatchedObservationStress(name="clean"),
    MatchedObservationStress(name="gaussian_5mm", gaussian_sigma_m=0.005),
    MatchedObservationStress(name="mismatch_12p5pct", mismatch_fraction=0.125),
    MatchedObservationStress(name="mismatch_25pct", mismatch_fraction=0.25),
    MatchedObservationStress(name="mismatch_50pct", mismatch_fraction=0.50),
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _stable_rng(*parts: object) -> np.random.Generator:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return np.random.default_rng(seed)


def corrupt_matched_current_observation(
    clean_observation_m: np.ndarray,
    available: np.ndarray,
    *,
    case_name: str,
    frame: int,
    stress: MatchedObservationStress,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply deterministic noise or a within-set identity derangement.

    A mismatch rotates values among a selected destination subset, preserving
    the unordered observed set exactly.  CPD therefore receives the same set,
    while a matched RBF arm receives explicitly wrong material identities.
    """

    clean = np.asarray(clean_observation_m, dtype=float)
    mask = np.asarray(available, dtype=bool)
    if clean.ndim != 2 or clean.shape[1] != 3 or mask.shape != (len(clean),):
        raise ValueError("clean observation and availability must have (K, 3)/(K,)")
    output = clean.copy()
    effective = mask & np.all(np.isfinite(clean), axis=1)
    mismatch = np.zeros(len(clean), dtype=bool)
    rng = _stable_rng(
        "deform360-pairwise-correspondence-gate-v1",
        case_name,
        frame,
        stress.name,
        seed,
    )
    if stress.gaussian_sigma_m > 0.0:
        noise = rng.normal(0.0, stress.gaussian_sigma_m, size=clean.shape)
        output[effective] += noise[effective]
    if stress.mismatch_fraction > 0.0:
        candidates = np.flatnonzero(effective)
        if len(candidates) >= 2:
            mismatch_count = min(
                len(candidates),
                max(2, int(np.floor(stress.mismatch_fraction * len(candidates)))),
            )
            destinations = np.sort(
                rng.choice(candidates, size=mismatch_count, replace=False)
            )
            sources = np.roll(destinations, 1)
            output[destinations] = clean[sources]
            mismatch[destinations] = True
    return (
        output,
        mismatch,
        {
            "available_count": int(np.sum(effective)),
            "gaussian_sigma_m": stress.gaussian_sigma_m,
            "mismatch_count": int(np.sum(mismatch)),
            "realized_mismatch_fraction": (
                0.0
                if not np.any(effective)
                else float(np.sum(mismatch) / np.sum(effective))
            ),
            "unordered_set_preserved": stress.mismatch_fraction > 0.0,
        },
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


def _score_interval(
    trajectories: Mapping[str, np.ndarray],
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    frames: Sequence[int],
) -> dict[str, dict[str, float]]:
    return {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target_m,
            visibility,
            validity,
            center_ids=center_ids,
            scored_frames=frames,
        )
        for arm, trajectory in trajectories.items()
    }


def evaluate_robust_correspondence_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: Sequence[int],
    case_name: str,
    stress: MatchedObservationStress,
    seed: int,
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
    cpd_config: NonrigidCpdConfig | None = None,
    cpd_trajectory_override_m: np.ndarray | None = None,
    score_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate one fixed stress using current measurements only."""

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence = np.asarray(persistence_input, dtype=float)
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    frames = tuple(int(frame) for frame in scored_frames)
    if (
        prior.shape != persistence.shape
        or prior.shape != target.shape
        or prior.ndim != 3
        or prior.shape[2] != 3
    ):
        raise ValueError("physical prior, persistence, target must share (T, N, 3)")
    if visible.shape != target.shape[:2] or valid.shape != target.shape[:2]:
        raise ValueError("visibility and validity must have shape (T, N)")
    if centers.shape != (CENTER_COUNT,) or len(np.unique(centers)) != len(centers):
        raise ValueError(f"center_ids must contain {CENTER_COUNT} unique IDs")
    if np.any(centers < 0) or np.any(centers >= target.shape[1]):
        raise ValueError("centre ID exceeds the trajectory")
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

    def initial_belief():
        return initialize_recursive_rbf_belief(
            centers,
            prior[0, centers],
            prior[0],
            config=belief_cfg,
        )

    robust_beliefs = {name: initial_belief() for name in backbones}
    ungated_beliefs = {name: initial_belief() for name in backbones}
    legacy_mixed_ungated_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=belief_cfg,
    )
    dynamic_arms = (
        SELECTED_RAW_ARM,
        UNGATED_RBF_ARM,
        LEGACY_MIXED_UNGATED_RBF_ARM,
        ROBUST_RBF_ARM,
        CPD_ARM,
    )
    trajectories = {arm: prior_input.copy() for arm in dynamic_arms}
    if cpd_trajectory_override_m is not None:
        cpd_override = np.asarray(cpd_trajectory_override_m)
        if cpd_override.shape != prior_input.shape:
            raise ValueError("cpd_trajectory_override_m must match the trajectory")
        trajectories[CPD_ARM] = cpd_override.copy()
    reuse_cpd_control = cpd_trajectory_override_m is not None
    output_dtype = prior_input.dtype
    update_reports: list[dict[str, object]] = []

    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(target)
        )
        available = (
            visible[update, centers]
            & valid[update, centers]
            & np.all(np.isfinite(target[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        clean_observation = target[update, centers]
        observation, mismatch_mask, corruption = corrupt_matched_current_observation(
            clean_observation,
            available,
            case_name=case_name,
            frame=update,
            stress=stress,
            seed=seed,
        )
        observed_ids = centers[available]
        observed_set = observation[available]
        if len(observed_ids):
            current_chamfer = {
                PHYSICAL_ARM: _symmetric_set_chamfer_m(
                    prior[update, observed_ids], observed_set
                ),
                PERSISTENCE_ARM: _symmetric_set_chamfer_m(
                    persistence[update, observed_ids], observed_set
                ),
            }
            selected_name = min(
                (PHYSICAL_ARM, PERSISTENCE_ARM),
                key=lambda name: (
                    current_chamfer[name],
                    0 if name == PHYSICAL_ARM else 1,
                ),
            )
        else:
            current_chamfer = {}
            selected_name = PHYSICAL_ARM
        selected = backbones[selected_name]
        for arm in dynamic_arms:
            if arm == CPD_ARM and reuse_cpd_control:
                continue
            trajectories[arm][update + 1 : stop] = selected[update + 1 : stop]

        residuals: dict[str, np.ndarray] = {}
        gates = {}
        for backbone_name, backbone in backbones.items():
            residual = np.full((CENTER_COUNT, 3), np.nan, dtype=float)
            residual[available] = (
                observation[available] - backbone[update, centers[available]]
            )
            residuals[backbone_name] = residual
            gates[backbone_name] = detect_pairwise_consensus_correspondences(
                backbone[update, centers],
                observation,
                available,
                material_ids=centers,
                config=gate_cfg,
            )
        if np.any(available):
            for backbone_name, backbone in backbones.items():
                ungated_beliefs[backbone_name], _ = update_recursive_rbf_belief(
                    ungated_beliefs[backbone_name],
                    update,
                    backbone[update, centers],
                    residuals[backbone_name],
                    available,
                    config=belief_cfg,
                )
                backbone_gate = gates[backbone_name]
                if backbone_gate.accepted:
                    robust_beliefs[backbone_name], _ = update_recursive_rbf_belief(
                        robust_beliefs[backbone_name],
                        update,
                        backbone[update, centers],
                        residuals[backbone_name],
                        backbone_gate.inlier_mask.copy(),
                        config=belief_cfg,
                    )
            legacy_mixed_ungated_belief, _ = update_recursive_rbf_belief(
                legacy_mixed_ungated_belief,
                update,
                selected[update, centers],
                residuals[selected_name],
                available,
                config=belief_cfg,
            )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    ungated_beliefs[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                trajectories[UNGATED_RBF_ARM][frame] = _corrected_frame(
                    selected[frame], decoded.mean_m, dtype=output_dtype
                )
                mixed_decoded = decode_recursive_rbf_belief(
                    legacy_mixed_ungated_belief,
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                trajectories[LEGACY_MIXED_UNGATED_RBF_ARM][frame] = _corrected_frame(
                    selected[frame], mixed_decoded.mean_m, dtype=output_dtype
                )

        gate = gates[selected_name]
        if gate.accepted:
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    robust_beliefs[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                trajectories[ROBUST_RBF_ARM][frame] = _corrected_frame(
                    selected[frame], decoded.mean_m, dtype=output_dtype
                )
        elif not np.array_equal(
            trajectories[ROBUST_RBF_ARM][update + 1 : stop],
            selected[update + 1 : stop],
        ):
            raise AssertionError("rejected RBF update did not preserve raw backbone")

        cpd_fit = None
        cpd_error = None
        cpd_reused = reuse_cpd_control
        if len(observed_ids) >= 3 and not cpd_reused:
            try:
                cpd_fit = fit_nonrigid_cpd(
                    selected[update, observed_ids],
                    observed_set,
                    config=cpd_cfg,
                )
                for frame in range(update + 1, stop):
                    trajectories[CPD_ARM][frame] = cpd_fit.transform(
                        selected[frame]
                    ).astype(output_dtype, copy=False)
            except (RuntimeError, np.linalg.LinAlgError) as error:
                cpd_error = f"{type(error).__name__}: {error}"
        if (
            not cpd_reused
            and cpd_fit is None
            and not np.array_equal(
                trajectories[CPD_ARM][update + 1 : stop],
                selected[update + 1 : stop],
            )
        ):
            raise AssertionError("failed CPD update did not preserve raw backbone")

        interval_frames = tuple(range(update + 1, stop))
        retained_clean = gate.inlier_mask & available & ~mismatch_mask
        retained_bad = gate.inlier_mask & mismatch_mask
        clean_count = int(np.sum(available & ~mismatch_mask))
        bad_count = int(np.sum(mismatch_mask))
        update_reports.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "interval_scored_frame_count": len(interval_frames),
                "corruption": corruption,
                "selected_backbone": {
                    "metric": "current observed-centre symmetric set Chamfer",
                    "physical_prior_m": current_chamfer.get(PHYSICAL_ARM),
                    "persistence_m": current_chamfer.get(PERSISTENCE_ARM),
                    "tie_break": PHYSICAL_ARM,
                    "selected": selected_name,
                },
                "pairwise_gate": {
                    "accepted": gate.accepted,
                    "decision": gate.decision,
                    "available_count": gate.available_count,
                    "inlier_count": gate.inlier_count,
                    "inlier_fraction": gate.inlier_fraction,
                    "pair_count": gate.pair_count,
                    "compatible_pair_fraction": gate.compatible_pair_fraction,
                    "median_inlier_normalized_strain": (
                        gate.median_inlier_normalized_strain
                    ),
                    "maximum_inlier_normalized_strain": (
                        gate.maximum_inlier_normalized_strain
                    ),
                    "inlier_center_ids": centers[gate.inlier_mask].tolist(),
                    "known_stress_diagnostic_only": {
                        "clean_correspondence_count": clean_count,
                        "mismatched_correspondence_count": bad_count,
                        "retained_clean_count": int(np.sum(retained_clean)),
                        "retained_mismatched_count": int(np.sum(retained_bad)),
                    },
                    "rejected_exact_selected_backbone_fallback": bool(
                        not gate.accepted
                        and np.array_equal(
                            trajectories[ROBUST_RBF_ARM][update + 1 : stop],
                            selected[update + 1 : stop],
                        )
                    ),
                    "by_backbone": {
                        name: {
                            "accepted": backbone_gate.accepted,
                            "decision": backbone_gate.decision,
                            "inlier_count": backbone_gate.inlier_count,
                            "inlier_fraction": backbone_gate.inlier_fraction,
                        }
                        for name, backbone_gate in gates.items()
                    },
                },
                "cpd": {
                    "fit_performed": cpd_fit is not None,
                    "reused_set_invariant_control": cpd_reused,
                    "fit_error": cpd_error,
                    "iterations": None if cpd_fit is None else cpd_fit.iterations,
                    "converged": None if cpd_fit is None else cpd_fit.converged,
                    "effective_correspondence_count": (
                        None
                        if cpd_fit is None
                        else cpd_fit.effective_correspondence_count
                    ),
                },
            }
        )

    all_trajectories = {
        PHYSICAL_ARM: prior_input,
        PERSISTENCE_ARM: persistence_input,
        **trajectories,
    }
    overrides = dict(score_overrides or {})
    unknown_overrides = set(overrides) - set(all_trajectories)
    if unknown_overrides:
        raise ValueError(f"unknown score override arms: {sorted(unknown_overrides)}")
    scores = _score_interval(
        {
            arm: trajectory
            for arm, trajectory in all_trajectories.items()
            if arm not in overrides
        },
        target,
        visible,
        valid,
        centers,
        frames,
    )
    scores.update(overrides)
    frame_to_score_index = {frame: index for index, frame in enumerate(frames)}
    for update_report in update_reports:
        interval_frames = tuple(
            range(
                int(update_report["frame"]) + 1,
                int(update_report["interval_end_exclusive"]),
            )
        )
        indices = [frame_to_score_index[frame] for frame in interval_frames]
        update_report["interval_scores"] = {
            arm: {
                "post_update_hidden_identity_rmse_m": float(
                    np.mean(
                        np.asarray(score["by_frame"]["hidden_identity_rmse_m"])[indices]
                    )
                ),
                "post_update_hidden_symmetric_chamfer_m": float(
                    np.mean(
                        np.asarray(score["by_frame"]["hidden_symmetric_chamfer_m"])[
                            indices
                        ]
                    )
                ),
            }
            for arm, score in scores.items()
        }
    result: dict[str, object] = {
        "stress": asdict(stress),
        "seed": seed,
        "updates": update_reports,
        "scores": scores,
    }
    return result, all_trajectories


def evaluate_deform360_robust_correspondence_case(
    episode_dir: str | Path,
    *,
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
    cpd_config: NonrigidCpdConfig | None = None,
) -> dict[str, object]:
    """Evaluate every frozen stress on one already-open audited episode."""

    baseline_report, baseline_arrays = evaluate_deform360_online_belief_case(
        episode_dir
    )
    prior = np.asarray(baseline_arrays["physical_prior_m"])
    persistence = np.asarray(baseline_arrays["persistence_m"])
    centers = np.asarray(baseline_arrays["center_ids"], dtype=np.int64)
    target_path = Path(str(baseline_report["inputs"]["target_data"]["path"])).resolve()
    if _sha256(target_path) != baseline_report["inputs"]["target_data"]["sha256"]:
        raise ValueError("audited target payload changed after baseline validation")
    target_data = _load_pickle(target_path)
    target = np.asarray(target_data["object_points"], dtype=float)
    visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
    validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if target.shape != prior.shape:
        raise ValueError("audited target no longer matches physical prior")
    scored_frames = tuple(int(value) for value in baseline_report["scored_frames"])

    fixed_scores = _score_interval(
        {PHYSICAL_ARM: prior, PERSISTENCE_ARM: persistence},
        target,
        visibility,
        validity,
        centers,
        scored_frames,
    )
    clean_stress = OBSERVATION_STRESSES[0]
    if clean_stress.name != "clean":
        raise AssertionError("the first fixed stress must be clean")
    clean_result, clean_trajectories = evaluate_robust_correspondence_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
        center_ids=centers,
        scored_frames=scored_frames,
        case_name=str(baseline_report["case"]),
        stress=clean_stress,
        seed=0,
        gate_config=gate_config,
        belief_config=belief_config,
        cpd_config=cpd_config,
        score_overrides=fixed_scores,
    )
    stress_reports: dict[str, list[dict[str, object]]] = {"clean": [clean_result]}
    for stress in OBSERVATION_STRESSES[1:]:
        records: list[dict[str, object]] = []
        for seed in CORRUPTION_SEEDS:
            set_invariant = stress.mismatch_fraction > 0.0
            score_overrides: dict[str, Mapping[str, object]] = dict(fixed_scores)
            if set_invariant:
                score_overrides.update(
                    {
                        SELECTED_RAW_ARM: clean_result["scores"][SELECTED_RAW_ARM],
                        CPD_ARM: clean_result["scores"][CPD_ARM],
                    }
                )
            result, trajectories = evaluate_robust_correspondence_arrays(
                prior,
                persistence,
                target,
                visibility,
                validity,
                center_ids=centers,
                scored_frames=scored_frames,
                case_name=str(baseline_report["case"]),
                stress=stress,
                seed=seed,
                gate_config=gate_config,
                belief_config=belief_config,
                cpd_config=cpd_config,
                cpd_trajectory_override_m=(
                    clean_trajectories[CPD_ARM] if set_invariant else None
                ),
                score_overrides=score_overrides,
            )
            if set_invariant:
                if not np.array_equal(
                    trajectories[SELECTED_RAW_ARM],
                    clean_trajectories[SELECTED_RAW_ARM],
                ):
                    raise AssertionError(
                        "identity derangement changed the set-based raw selector"
                    )
                if not np.array_equal(
                    trajectories[CPD_ARM], clean_trajectories[CPD_ARM]
                ):
                    raise AssertionError(
                        "identity derangement changed the unordered CPD control"
                    )
            records.append(result)
        stress_reports[stress.name] = records
    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    cpd_cfg = cpd_config or NonrigidCpdConfig()
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case": baseline_report["case"],
        "object_id": baseline_report["object_id"],
        "episode_id": baseline_report["episode_id"],
        "center_ids": centers.tolist(),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": list(scored_frames),
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "cpd_config": asdict(cpd_cfg),
        "stresses": stress_reports,
        "inputs": baseline_report["inputs"],
        "information_boundary": {
            **baseline_report["information_boundary"],
            "detector_inputs": (
                "selected raw backbone and assigned sparse observations at the "
                "current update only"
            ),
            "detector_target_error_input": False,
            "synthetic_corruption_scope": "current assimilation centres only",
            "scoring_centres_permanently_excluded": True,
            "held_calibration_or_target_access": False,
        },
    }


def _case_condition_score(
    report: Mapping[str, Any],
    condition: str,
    arm: str,
    metric: str,
) -> float:
    records = report["stresses"][condition]
    return float(np.mean([record["scores"][arm][metric] for record in records]))


def _comparison(
    reports: Sequence[Mapping[str, Any]],
    groups: Mapping[str, str],
    condition: str,
    candidate: str,
    comparator: str,
    aggregate: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for metric in PRIMARY_METRICS:
        differences = {
            str(report["case"]): _case_condition_score(
                report, condition, candidate, metric
            )
            - _case_condition_score(report, condition, comparator, metric)
            for report in reports
        }
        result = _physical_object_cluster_bootstrap(differences, groups)
        result["relative_change"] = _relative_change(
            aggregate[candidate][metric], aggregate[comparator][metric]
        )
        result["episode_wins"] = int(
            np.sum(np.asarray(list(differences.values())) < 0.0)
        )
        metrics[metric] = result
    return {
        "metrics": metrics,
        "joint_two_metric_episode_wins": int(
            sum(
                all(
                    _case_condition_score(report, condition, candidate, metric)
                    < _case_condition_score(report, condition, comparator, metric)
                    for metric in PRIMARY_METRICS
                )
                for report in reports
            )
        ),
    }


def _coverage_and_risk(
    reports: Sequence[Mapping[str, Any]],
    condition: str,
) -> dict[str, object]:
    records = [record for report in reports for record in report["stresses"][condition]]
    updates = [update for record in records for update in record["updates"]]
    accepted = [update for update in updates if update["pairwise_gate"]["accepted"]]
    rejected = [update for update in updates if not update["pairwise_gate"]["accepted"]]
    total_frames = int(sum(update["interval_scored_frame_count"] for update in updates))
    accepted_frames = int(
        sum(update["interval_scored_frame_count"] for update in accepted)
    )

    def conditional(values: Sequence[Mapping[str, Any]]) -> dict[str, object] | None:
        if not values:
            return None
        return {
            arm: {
                metric: float(
                    np.mean(
                        [update["interval_scores"][arm][metric] for update in values]
                    )
                )
                for metric in PRIMARY_METRICS
            }
            for arm in (
                ROBUST_RBF_ARM,
                SELECTED_RAW_ARM,
                UNGATED_RBF_ARM,
                LEGACY_MIXED_UNGATED_RBF_ARM,
                CPD_ARM,
            )
        }

    known = [
        update["pairwise_gate"]["known_stress_diagnostic_only"] for update in updates
    ]
    retained = sum(int(value["retained_clean_count"]) for value in known)
    retained_bad = sum(int(value["retained_mismatched_count"]) for value in known)
    clean = sum(int(value["clean_correspondence_count"]) for value in known)
    bad = sum(int(value["mismatched_correspondence_count"]) for value in known)
    return {
        "record_count": len(records),
        "update_count": len(updates),
        "accepted_update_count": len(accepted),
        "rejected_update_count": len(rejected),
        "accepted_update_fraction": len(accepted) / len(updates),
        "corrected_scored_frame_fraction": accepted_frames / total_frames,
        "mean_retained_fraction": float(
            np.mean([update["pairwise_gate"]["inlier_fraction"] for update in updates])
        ),
        "rejected_exact_fallback_count": int(
            sum(
                update["pairwise_gate"]["rejected_exact_selected_backbone_fallback"]
                for update in rejected
            )
        ),
        "known_stress_detector_diagnostic": {
            "retained_correspondence_precision": (
                None
                if retained + retained_bad == 0
                else retained / (retained + retained_bad)
            ),
            "clean_correspondence_recall": None if clean == 0 else retained / clean,
            "mismatch_rejection_recall": None
            if bad == 0
            else (bad - retained_bad) / bad,
        },
        "accepted_interval_risk": conditional(accepted),
        "rejected_interval_risk": conditional(rejected),
    }


def evaluate_deform360_robust_correspondence_cohort(
    root: str | Path,
    output: str | Path,
    *,
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
    cpd_config: NonrigidCpdConfig | None = None,
) -> dict[str, object]:
    """Persist the fixed robustness diagnostic for exactly the open 27."""

    cohort_root = Path(root).resolve()
    output_dir = Path(output).resolve()
    expected = _expected_episode_directories()
    if len(expected) != 27:
        raise AssertionError("fixed source panel no longer contains 27 episodes")
    missing = [case for case in expected if not (cohort_root / case).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing fixed Deform360 episodes: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)

    reports: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, str]] = []
    for case_name in expected:
        report = evaluate_deform360_robust_correspondence_case(
            cohort_root / case_name,
            gate_config=gate_config,
            belief_config=belief_config,
            cpd_config=cpd_config,
        )
        report_path = output_dir / f"{case_name}.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        groups[case_name] = str(report["object_id"])
        artifacts.append({"case": case_name, "report_sha256": _sha256(report_path)})

    condition_names = tuple(stress.name for stress in OBSERVATION_STRESSES)
    aggregate = {
        condition: {
            arm: {
                metric: float(
                    np.mean(
                        [
                            _case_condition_score(report, condition, arm, metric)
                            for report in reports
                        ]
                    )
                )
                for metric in PRIMARY_METRICS
            }
            for arm in ARMS
        }
        for condition in condition_names
    }
    pairs = (
        (ROBUST_RBF_ARM, UNGATED_RBF_ARM),
        (UNGATED_RBF_ARM, LEGACY_MIXED_UNGATED_RBF_ARM),
        (ROBUST_RBF_ARM, LEGACY_MIXED_UNGATED_RBF_ARM),
        (ROBUST_RBF_ARM, CPD_ARM),
        (ROBUST_RBF_ARM, SELECTED_RAW_ARM),
        (ROBUST_RBF_ARM, PHYSICAL_ARM),
        (ROBUST_RBF_ARM, PERSISTENCE_ARM),
        (CPD_ARM, SELECTED_RAW_ARM),
    )
    comparisons = {
        condition: {
            f"{candidate}_vs_{comparator}": _comparison(
                reports,
                groups,
                condition,
                candidate,
                comparator,
                aggregate[condition],
            )
            for candidate, comparator in pairs
        }
        for condition in condition_names
    }
    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    cpd_cfg = cpd_config or NonrigidCpdConfig()
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_root": str(cohort_root),
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "physical_objects": {
            key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
        },
        "corruption_seeds": list(CORRUPTION_SEEDS),
        "stresses": [asdict(stress) for stress in OBSERVATION_STRESSES],
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "cpd_config": asdict(cpd_cfg),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "coverage_and_risk": {
            condition: _coverage_and_risk(reports, condition)
            for condition in condition_names
        },
        "selection_contract": {
            "detector": (
                "exact maximum clique under current pair-distance strain <= "
                "max(30 mm, 10 percent source pair distance)"
            ),
            "threshold_rationale": (
                "source-only geometry; conservative envelope for 5 mm coordinate "
                "noise plus modest material strain; no hidden target error used"
            ),
            "acceptance": "at least 9 centres and at least 70 percent consensus",
            "backbone": (
                "current-observation symmetric-set-Chamfer selector between physical "
                "and persistence, with physical tie break"
            ),
            "state_semantics": (
                "independent causal RBF posterior per physical/persistence backbone; "
                "both are updated from the current observation and only the selected "
                "state is decoded; posterior state never crosses a backbone switch"
            ),
            "legacy_diagnostic": (
                "a separately reported mixed-state ungated arm reproduces the old "
                "cross-backbone state semantics but is never the candidate"
            ),
            "rejection": (
                "bit-exact selected raw physical or persistence trajectory for the "
                "whole following interval"
            ),
            "mismatch": (
                "within-observed-set value derangement; assigned RBF identities are "
                "wrong while unordered CPD input is exactly preserved"
            ),
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "physical object after averaging fixed seeds within episode",
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "post-hoc detector development on the already-open independent-source "
            "Deform360 27; fused material-track pseudo-measurements; not held-target "
            "or official benchmark evidence"
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ARMS",
    "CORRUPTION_SEEDS",
    "CPD_ARM",
    "LEGACY_MIXED_UNGATED_RBF_ARM",
    "MatchedObservationStress",
    "OBSERVATION_STRESSES",
    "PROTOCOL_ID",
    "ROBUST_RBF_ARM",
    "SELECTED_RAW_ARM",
    "UNGATED_RBF_ARM",
    "corrupt_matched_current_observation",
    "evaluate_deform360_robust_correspondence_case",
    "evaluate_deform360_robust_correspondence_cohort",
    "evaluate_robust_correspondence_arrays",
]
