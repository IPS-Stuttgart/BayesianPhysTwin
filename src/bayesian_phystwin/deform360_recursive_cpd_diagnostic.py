"""Temporal belief controls on top of unordered Deform360 CPD updates.

This development diagnostic asks a deliberately narrow question: once a
strong independent coherent-point-drift (CPD) registration baseline is used,
does carrying a deformation belief across update times add robustness?  Every
arm receives the same unordered set at the current update, uses the same
current-observation physical/persistence backbone decision, and fits the same
CPD transformations.  The only candidate difference is an exponentially
tempered mixture of the current and earlier CPD displacement fields.

The cohort is the fixed, already-open 27-episode independent-source panel.
Assimilation centres are permanently excluded from scoring.  Synthetic
corruptions are fixed below, deterministic, and applied only to the current
centre set; the hidden scoring identities are never corrupted or observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from .cpd_registration import (
    NonrigidCpdConfig,
    NonrigidCpdTransform,
    fit_nonrigid_cpd,
)
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


PROTOCOL_ID = "deform360-open27-recursive-tempered-cpd-v2-posthoc-development"
STRESS_SEED = 20_260_719
TEMPERING_GAINS = (0.50, 0.75, 0.90)
ADAPTIVE_MINIMUM_GAIN = 0.75
INDEPENDENT_ARM = "independent_cpd_observed_backbone"
SELECTED_RAW_BACKBONE_ARM = "selected_raw_backbone"
FROZEN_TRACKER_ARM = "cpd_tracker_frozen_current"
CONSTANT_VELOCITY_TRACKER_ARM = "cpd_tracker_constant_velocity"
ADAPTIVE_ARM = "recursive_tempered_cpd_effective_support"


@dataclass(frozen=True)
class CpdObservationStress:
    """A fixed causal corruption applied to the current unordered point set."""

    name: str
    gaussian_sigma_m: float = 0.0
    outlier_fraction: float = 0.0
    outlier_magnitude_m: float = 0.0
    retained_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stress name must be nonempty")
        if not np.isfinite(self.gaussian_sigma_m) or self.gaussian_sigma_m < 0.0:
            raise ValueError("gaussian_sigma_m must be nonnegative")
        if not np.isfinite(self.outlier_fraction) or not (
            0.0 <= self.outlier_fraction <= 1.0
        ):
            raise ValueError("outlier_fraction must lie in [0, 1]")
        if not np.isfinite(self.outlier_magnitude_m) or self.outlier_magnitude_m < 0.0:
            raise ValueError("outlier_magnitude_m must be nonnegative")
        if not np.isfinite(self.retained_fraction) or not (
            0.0 < self.retained_fraction <= 1.0
        ):
            raise ValueError("retained_fraction must lie in (0, 1]")
        if (self.outlier_fraction == 0.0) != (self.outlier_magnitude_m == 0.0):
            raise ValueError(
                "outlier_fraction and outlier_magnitude_m must both be zero or positive"
            )


OBSERVATION_STRESSES = (
    CpdObservationStress(name="clean"),
    CpdObservationStress(name="gaussian_2mm", gaussian_sigma_m=0.002),
    CpdObservationStress(name="gaussian_3mm", gaussian_sigma_m=0.003),
    CpdObservationStress(
        name="outlier_12p5pct_10mm",
        outlier_fraction=0.125,
        outlier_magnitude_m=0.010,
    ),
    CpdObservationStress(
        name="outlier_12p5pct_20mm",
        outlier_fraction=0.125,
        outlier_magnitude_m=0.020,
    ),
    CpdObservationStress(
        name="outlier_25pct_30mm",
        outlier_fraction=0.25,
        outlier_magnitude_m=0.030,
    ),
    CpdObservationStress(name="occlusion_50pct", retained_fraction=0.50),
    CpdObservationStress(
        name="combined_2mm_12p5pct_20mm",
        gaussian_sigma_m=0.002,
        outlier_fraction=0.125,
        outlier_magnitude_m=0.020,
    ),
    CpdObservationStress(
        name="combined_2mm_25pct_30mm_50pct",
        gaussian_sigma_m=0.002,
        outlier_fraction=0.25,
        outlier_magnitude_m=0.030,
        retained_fraction=0.50,
    ),
)


def _gain_arm(gain: float) -> str:
    return f"recursive_tempered_cpd_gain_{int(round(100 * gain)):03d}"


RECURSIVE_ARMS = tuple(_gain_arm(gain) for gain in TEMPERING_GAINS)
TRACKER_ARMS = (FROZEN_TRACKER_ARM, CONSTANT_VELOCITY_TRACKER_ARM)
ARMS = (
    (
        INDEPENDENT_ARM,
        SELECTED_RAW_BACKBONE_ARM,
    )
    + TRACKER_ARMS
    + RECURSIVE_ARMS
    + (ADAPTIVE_ARM,)
)


@dataclass(frozen=True)
class TemperedCpdFieldState:
    """Finite mixture representation of a recursively tempered CPD field."""

    transforms: tuple[NonrigidCpdTransform, ...]
    weights: np.ndarray
    update_index: int

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float).copy()
        if not self.transforms or weights.shape != (len(self.transforms),):
            raise ValueError("one weight is required for every CPD transform")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("tempered CPD weights must be finite and nonnegative")
        if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-12):
            raise ValueError("tempered CPD weights must sum to one")
        if self.update_index < 0:
            raise ValueError("update_index must be nonnegative")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)


def update_tempered_cpd_field(
    state: TemperedCpdFieldState | None,
    transform: NonrigidCpdTransform,
    *,
    gain: float,
    update_index: int,
) -> TemperedCpdFieldState:
    """Assimilate one raw CPD field with a fixed exponential posterior gain."""

    if not np.isfinite(gain) or not 0.0 < gain <= 1.0:
        raise ValueError("gain must lie in (0, 1]")
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    if state is None:
        return TemperedCpdFieldState(
            transforms=(transform,),
            weights=np.ones(1, dtype=float),
            update_index=update_index,
        )
    if update_index <= state.update_index:
        raise ValueError("CPD field updates must have strictly increasing indices")
    gap = update_index - state.update_index
    retention = (1.0 - gain) ** gap
    old_weights = state.weights * retention
    weights = np.concatenate((old_weights, np.asarray([1.0 - retention])))
    positive = weights > np.finfo(float).eps
    return TemperedCpdFieldState(
        transforms=tuple(
            transform_value
            for transform_value, keep in zip(
                state.transforms + (transform,), positive, strict=True
            )
            if keep
        ),
        weights=weights[positive] / np.sum(weights[positive]),
        update_index=update_index,
    )


def decode_tempered_cpd_field(
    state: TemperedCpdFieldState,
    query_points_m: np.ndarray,
) -> np.ndarray:
    """Apply the posterior-mean displacement field to arbitrary query points."""

    query = np.asarray(query_points_m, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query_points_m must have shape (Q, 3)")
    if not np.all(np.isfinite(query)):
        raise ValueError("query_points_m must be finite")
    correction = np.zeros_like(query)
    for weight, transform in zip(state.weights, state.transforms, strict=True):
        correction += weight * (transform.transform(query) - query)
    return query + correction


def effective_support_tempering_gain(
    effective_correspondence_count: float,
    *,
    nominal_center_count: int = CENTER_COUNT,
    minimum_gain: float = ADAPTIVE_MINIMUM_GAIN,
) -> float:
    """Map CPD effective support to the frozen post-hoc development gain.

    This rule was defined after inspecting the fixed-gain open-panel diagnostic
    and before any raw-camera or held-target evaluation.  It has no tunable
    dependence on outcome error: gain is simply ``effective / 16`` clipped to
    ``[0.75, 1]``.
    """

    if nominal_center_count < 1:
        raise ValueError("nominal_center_count must be positive")
    if not np.isfinite(minimum_gain) or not 0.0 < minimum_gain <= 1.0:
        raise ValueError("minimum_gain must lie in (0, 1]")
    if (
        not np.isfinite(effective_correspondence_count)
        or effective_correspondence_count <= 0.0
    ):
        raise ValueError("effective_correspondence_count must be positive")
    return float(
        np.clip(
            effective_correspondence_count / nominal_center_count,
            minimum_gain,
            1.0,
        )
    )


def _stress_rng(case_name: str, frame: int, stress_name: str) -> np.random.Generator:
    payload = f"{STRESS_SEED}:{case_name}:{frame}:{stress_name}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def corrupt_current_unordered_set(
    point_ids: np.ndarray,
    points_m: np.ndarray,
    *,
    case_name: str,
    frame: int,
    stress: CpdObservationStress,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return a deterministic corrupted subset without exposing correspondences."""

    ids = np.asarray(point_ids, dtype=np.int64)
    points = np.asarray(points_m, dtype=float)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("point_ids must be a unique vector")
    if points.shape != (len(ids), 3) or not np.all(np.isfinite(points)):
        raise ValueError("points_m must have finite shape (len(point_ids), 3)")
    if not len(ids):
        return (
            ids.copy(),
            points.copy(),
            {
                "input_count": 0,
                "retained_count": 0,
                "outlier_count": 0,
            },
        )

    rng = _stress_rng(case_name, frame, stress.name)
    retained_count = max(1, int(np.ceil(stress.retained_fraction * len(ids))))
    retained_local = np.sort(rng.choice(len(ids), size=retained_count, replace=False))
    retained_ids = ids[retained_local]
    corrupted = points[retained_local].copy()
    if stress.gaussian_sigma_m > 0.0:
        corrupted += rng.normal(
            loc=0.0,
            scale=stress.gaussian_sigma_m,
            size=corrupted.shape,
        )

    outlier_count = (
        0
        if stress.outlier_fraction == 0.0
        else max(1, int(np.ceil(stress.outlier_fraction * retained_count)))
    )
    if outlier_count:
        outlier_local = rng.choice(retained_count, size=outlier_count, replace=False)
        directions = rng.normal(size=(outlier_count, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.maximum(norms, np.finfo(float).tiny)
        corrupted[outlier_local] += stress.outlier_magnitude_m * directions

    permutation = rng.permutation(retained_count)
    return (
        retained_ids,
        corrupted[permutation],
        {
            "input_count": int(len(ids)),
            "retained_count": int(retained_count),
            "outlier_count": int(outlier_count),
            "target_order_permuted": True,
            "returned_ids_are_support_metadata_only": True,
        },
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _evaluate_stress(
    *,
    case_name: str,
    prior: np.ndarray,
    persistence: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    centers: np.ndarray,
    scored_frames: tuple[int, ...],
    stress: CpdObservationStress,
    config: NonrigidCpdConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    trajectories = {arm: prior.copy() for arm in ARMS}
    states: dict[float, dict[str, TemperedCpdFieldState | None]] = {
        gain: {"physical_prior": None, "persistence": None} for gain in TEMPERING_GAINS
    }
    adaptive_states: dict[str, TemperedCpdFieldState | None] = {
        "physical_prior": None,
        "persistence": None,
    }
    updates: list[dict[str, object]] = []
    backbones = {"physical_prior": prior, "persistence": persistence}
    last_registered_frame = 0
    last_registered_state = prior[0].astype(float, copy=True)

    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(target)
        )
        available = (
            visibility[update, centers]
            & validity[update, centers]
            & np.all(np.isfinite(target[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        clean_ids = centers[available]
        observed_ids, observed_set, corruption = corrupt_current_unordered_set(
            clean_ids,
            target[update, clean_ids],
            case_name=case_name,
            frame=update,
            stress=stress,
        )
        current_chamfer: dict[str, float] = {}
        if len(observed_ids):
            for backbone_name, backbone in backbones.items():
                current_chamfer[backbone_name] = _symmetric_set_chamfer_m(
                    backbone[update, observed_ids],
                    observed_set,
                )
            selected_backbone = min(
                ("physical_prior", "persistence"),
                key=lambda name: (
                    current_chamfer[name],
                    0 if name == "physical_prior" else 1,
                ),
            )
        else:
            selected_backbone = "physical_prior"
        selected_trajectory = backbones[selected_backbone]
        for arm in ARMS:
            trajectories[arm][update + 1 : stop] = selected_trajectory[
                update + 1 : stop
            ]

        can_fit = len(observed_ids) >= 3
        transforms: dict[str, NonrigidCpdTransform] = {}
        fit_error = None
        fallback_reason = (
            "no_observation"
            if not len(observed_ids)
            else "insufficient_support"
            if not can_fit
            else None
        )
        if can_fit:
            try:
                for backbone_name, backbone in backbones.items():
                    source_set = backbone[update, observed_ids]
                    transforms[backbone_name] = fit_nonrigid_cpd(
                        source_set,
                        observed_set,
                        config=config,
                    )
            except (RuntimeError, np.linalg.LinAlgError) as error:
                transforms.clear()
                fit_error = f"{type(error).__name__}: {error}"
                fallback_reason = "fit_failure"

        fit_performed = len(transforms) == len(backbones)
        adaptive_gains: dict[str, float] = {}
        adaptive_exact_independent = False
        adaptive_exact_reason = None
        if fit_performed:
            selected_transform = transforms[selected_backbone]
            registered_current = selected_transform.transform(
                selected_trajectory[update]
            )
            elapsed = update - last_registered_frame
            if elapsed <= 0:
                raise AssertionError(
                    "tracker update frames must be strictly increasing"
                )
            registered_velocity_per_frame = (
                registered_current - last_registered_state
            ) / elapsed
            for frame in range(update + 1, stop):
                trajectories[INDEPENDENT_ARM][frame] = selected_transform.transform(
                    selected_trajectory[frame]
                ).astype(prior.dtype, copy=False)
                trajectories[FROZEN_TRACKER_ARM][frame] = registered_current.astype(
                    prior.dtype,
                    copy=False,
                )
                trajectories[CONSTANT_VELOCITY_TRACKER_ARM][frame] = (
                    registered_current
                    + (frame - update) * registered_velocity_per_frame
                ).astype(prior.dtype, copy=False)

            for gain in TEMPERING_GAINS:
                for backbone_name, transform in transforms.items():
                    states[gain][backbone_name] = update_tempered_cpd_field(
                        states[gain][backbone_name],
                        transform,
                        gain=gain,
                        update_index=update_index,
                    )
                state = states[gain][selected_backbone]
                if state is None:
                    raise AssertionError("selected recursive CPD state was not updated")
                arm = _gain_arm(gain)
                for frame in range(update + 1, stop):
                    if len(state.transforms) == 1:
                        trajectories[arm][frame] = trajectories[INDEPENDENT_ARM][frame]
                    else:
                        trajectories[arm][frame] = decode_tempered_cpd_field(
                            state,
                            selected_trajectory[frame],
                        ).astype(prior.dtype, copy=False)

            adaptive_prior_was_none: dict[str, bool] = {}
            for backbone_name, transform in transforms.items():
                adaptive_prior_was_none[backbone_name] = (
                    adaptive_states[backbone_name] is None
                )
                adaptive_gain = effective_support_tempering_gain(
                    transform.effective_correspondence_count
                )
                adaptive_gains[backbone_name] = adaptive_gain
                adaptive_states[backbone_name] = update_tempered_cpd_field(
                    adaptive_states[backbone_name],
                    transform,
                    gain=adaptive_gain,
                    update_index=update_index,
                )
            adaptive_state = adaptive_states[selected_backbone]
            if adaptive_state is None:
                raise AssertionError("selected adaptive CPD state was not updated")
            selected_adaptive_gain = adaptive_gains[selected_backbone]
            if adaptive_prior_was_none[selected_backbone]:
                adaptive_exact_reason = "first_successful_update"
            elif selected_adaptive_gain == 1.0:
                adaptive_exact_reason = "gain_one"
            adaptive_exact_independent = adaptive_exact_reason is not None
            for frame in range(update + 1, stop):
                if adaptive_exact_independent:
                    trajectories[ADAPTIVE_ARM][frame] = trajectories[INDEPENDENT_ARM][
                        frame
                    ]
                else:
                    trajectories[ADAPTIVE_ARM][frame] = decode_tempered_cpd_field(
                        adaptive_state,
                        selected_trajectory[frame],
                    ).astype(prior.dtype, copy=False)
            if adaptive_exact_independent and not np.array_equal(
                trajectories[ADAPTIVE_ARM][update + 1 : stop],
                trajectories[INDEPENDENT_ARM][update + 1 : stop],
            ):
                raise AssertionError(
                    "adaptive exact-parity interval differs from independent CPD"
                )
            last_registered_frame = update
            last_registered_state = registered_current

        fallback_applied = not fit_performed
        fallback_exact = True
        if fallback_applied:
            fallback_exact = all(
                np.array_equal(
                    trajectories[arm][update + 1 : stop],
                    selected_trajectory[update + 1 : stop],
                )
                for arm in ARMS
            )
            if not fallback_exact:
                raise AssertionError(
                    "failed CPD update did not preserve the selected raw backbone"
                )

        fit_diagnostics = {
            name: {
                "iterations": transform.iterations,
                "converged": transform.converged,
                "source_rms_scale_m": transform.scale_m,
                "final_variance_normalized2": transform.variance_normalized2,
                "effective_correspondence_count": (
                    transform.effective_correspondence_count
                ),
            }
            for name, transform in transforms.items()
        }
        updates.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "clean_available_center_count": int(len(clean_ids)),
                "observed_center_count": int(len(observed_ids)),
                "fit_performed": fit_performed,
                "fit_error": fit_error,
                "fallback": {
                    "applied": fallback_applied,
                    "reason": fallback_reason,
                    "selected_raw_backbone": selected_backbone,
                    "bit_exact_for_all_arms": fallback_exact,
                },
                "corruption": corruption,
                "current_observation_backbone_selection": {
                    "metric": "symmetric set Chamfer on corrupted current set",
                    "physical_prior_m": current_chamfer.get("physical_prior"),
                    "persistence_m": current_chamfer.get("persistence"),
                    "tie_break": "physical_prior",
                    "selected": selected_backbone,
                },
                "fits": fit_diagnostics,
                "adaptive_effective_support_gain": {
                    "formula": "clip(effective_correspondence_count / 16, 0.75, 1)",
                    "by_backbone": adaptive_gains,
                    "selected_gain": adaptive_gains.get(selected_backbone),
                    "exact_independent_cpd_output": adaptive_exact_independent,
                    "exact_independent_reason": adaptive_exact_reason,
                },
                "recursive_state_component_counts": {
                    **{
                        _gain_arm(gain): {
                            name: (
                                None
                                if states[gain][name] is None
                                else len(states[gain][name].transforms)
                            )
                            for name in backbones
                        }
                        for gain in TEMPERING_GAINS
                    },
                    ADAPTIVE_ARM: {
                        name: (
                            None
                            if adaptive_states[name] is None
                            else len(adaptive_states[name].transforms)
                        )
                        for name in backbones
                    },
                },
            }
        )

    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visibility,
            validity,
            center_ids=centers,
            scored_frames=scored_frames,
        )
        for arm, trajectory in trajectories.items()
    }
    return {"updates": updates, "scores": scores}, trajectories


def evaluate_deform360_recursive_cpd_case(
    episode_dir: str | Path,
    *,
    config: NonrigidCpdConfig | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate clean and fixed-stress CPD temporal controls on one open case."""

    baseline_report, baseline_arrays = evaluate_deform360_online_belief_case(
        episode_dir
    )
    cfg = config or NonrigidCpdConfig()
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
        raise ValueError("audited target no longer matches the physical prior")
    scored_frames = tuple(int(value) for value in baseline_report["scored_frames"])

    stress_reports: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {
        "center_ids": centers,
        "physical_prior_m": prior,
        "persistence_m": persistence,
    }
    for stress in OBSERVATION_STRESSES:
        result, trajectories = _evaluate_stress(
            case_name=str(baseline_report["case"]),
            prior=prior,
            persistence=persistence,
            target=target,
            visibility=visibility,
            validity=validity,
            centers=centers,
            scored_frames=scored_frames,
            stress=stress,
            config=cfg,
        )
        stress_reports[stress.name] = result
        arrays.update(
            {
                f"{stress.name}__{arm}_m": trajectory
                for arm, trajectory in trajectories.items()
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case": baseline_report["case"],
        "object_id": baseline_report["object_id"],
        "episode_id": baseline_report["episode_id"],
        "center_ids": centers.tolist(),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": list(scored_frames),
        "cpd_config": asdict(cfg),
        "tempering_gains": list(TEMPERING_GAINS),
        "adaptive_effective_support_rule": {
            "formula": "clip(effective_correspondence_count / 16, 0.75, 1)",
            "minimum_gain": ADAPTIVE_MINIMUM_GAIN,
            "nominal_center_count": CENTER_COUNT,
            "development_status": (
                "defined after inspecting the fixed-gain open-panel v1 diagnostic "
                "and before any raw-camera or held-target evaluation"
            ),
        },
        "stresses": stress_reports,
        "inputs": baseline_report["inputs"],
        "information_boundary": {
            **baseline_report["information_boundary"],
            "registration_observation": (
                "unordered corrupted subset of the current sparse fused material "
                "centres only"
            ),
            "synthetic_corruption_scope": (
                "current assimilation centres only; never hidden scoring identities"
            ),
            "scoring_centres_permanently_excluded": True,
            "held_target_access": False,
        },
        "candidate_contract": {
            "independent_control": (
                "refit CPD independently and query only the selected current "
                "physical/persistence backbone"
            ),
            "tracker_only_controls": {
                FROZEN_TRACKER_ARM: (
                    "repeat the CPD-registered current full state; never query a "
                    "future physical or persistence trajectory"
                ),
                CONSTANT_VELOCITY_TRACKER_ARM: (
                    "linearly extrapolate the CPD-registered current full state "
                    "from the last successful registered update; frame zero is the "
                    "initial accepted state; never query a future physical or "
                    "persistence trajectory"
                ),
            },
            "recursive_difference": (
                "maintain one causal exponentially-tempered CPD displacement-field "
                "belief for each known backbone; use the same current hard selector"
            ),
            "adaptive_candidate": (
                "post-hoc open-panel development rule: clip CPD effective "
                "correspondences divided by the planned 16 centres to [0.75, 1]"
            ),
            "first_update_parity": (
                "all recursive arms equal the independent CPD arm at the first update"
            ),
            "gain_one_parity": (
                "adaptive gain exactly one copies the selected independent CPD "
                "trajectory bit-for-bit for that interval"
            ),
            "failed_fit_fallback": (
                "exact current-observation-selected raw physical or persistence "
                "backbone for that interval; no CPD correction"
            ),
        },
    }
    return report, arrays


def _comparison(
    reports: list[dict[str, object]],
    aggregate: Mapping[str, Mapping[str, Mapping[str, float]]],
    groups: Mapping[str, str],
    stress_name: str,
    candidate: str,
    comparator: str,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for metric in PRIMARY_METRICS:
        differences = {
            str(report["case"]): float(
                report["stresses"][stress_name]["scores"][candidate][metric]
                - report["stresses"][stress_name]["scores"][comparator][metric]
            )
            for report in reports
        }
        result = _physical_object_cluster_bootstrap(differences, groups)
        result["relative_change"] = _relative_change(
            aggregate[stress_name][candidate][metric],
            aggregate[stress_name][comparator][metric],
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
                    report["stresses"][stress_name]["scores"][candidate][metric]
                    < report["stresses"][stress_name]["scores"][comparator][metric]
                    for metric in PRIMARY_METRICS
                )
                for report in reports
            )
        ),
    }


def evaluate_deform360_recursive_cpd_cohort(
    root: str | Path,
    output: str | Path,
    *,
    config: NonrigidCpdConfig | None = None,
) -> dict[str, object]:
    """Persist the bounded recursive-CPD diagnostic for the open 27 only."""

    cohort_root = Path(root).resolve()
    output_dir = Path(output).resolve()
    expected = _expected_episode_directories()
    if len(expected) != 27:
        raise AssertionError("fixed source panel no longer contains 27 episodes")
    missing = [name for name in expected if not (cohort_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing fixed Deform360 episodes: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)

    reports: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, object]] = []
    for case_name in expected:
        report, arrays = evaluate_deform360_recursive_cpd_case(
            cohort_root / case_name,
            config=config,
        )
        report_path = output_dir / f"{case_name}.json"
        arrays_path = output_dir / f"{case_name}.npz"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **arrays)
        reports.append(report)
        groups[case_name] = str(report["object_id"])
        artifacts.append(
            {
                "case": case_name,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )

    stress_names = tuple(stress.name for stress in OBSERVATION_STRESSES)
    aggregate = {
        stress_name: {
            arm: {
                metric: float(
                    np.mean(
                        [
                            report["stresses"][stress_name]["scores"][arm][metric]
                            for report in reports
                        ]
                    )
                )
                for metric in PRIMARY_METRICS
            }
            for arm in ARMS
        }
        for stress_name in stress_names
    }
    comparison_pairs = (
        *((arm, INDEPENDENT_ARM) for arm in RECURSIVE_ARMS),
        (ADAPTIVE_ARM, INDEPENDENT_ARM),
        (INDEPENDENT_ARM, FROZEN_TRACKER_ARM),
        (INDEPENDENT_ARM, CONSTANT_VELOCITY_TRACKER_ARM),
        (INDEPENDENT_ARM, SELECTED_RAW_BACKBONE_ARM),
        (FROZEN_TRACKER_ARM, CONSTANT_VELOCITY_TRACKER_ARM),
    )
    comparisons = {
        stress_name: {
            f"{candidate}_vs_{comparator}": _comparison(
                reports,
                aggregate,
                groups,
                stress_name,
                candidate,
                comparator,
            )
            for candidate, comparator in comparison_pairs
        }
        for stress_name in stress_names
    }
    fit_diagnostics = {
        stress_name: {
            "update_count": len(reports) * len(UPDATE_FRAMES),
            "successful_update_count": int(
                sum(
                    update["fit_performed"]
                    for report in reports
                    for update in report["stresses"][stress_name]["updates"]
                )
            ),
            "selected_backbone_counts": {
                name: int(
                    sum(
                        update["current_observation_backbone_selection"]["selected"]
                        == name
                        for report in reports
                        for update in report["stresses"][stress_name]["updates"]
                    )
                )
                for name in ("physical_prior", "persistence")
            },
        }
        for stress_name in stress_names
    }
    cfg = config or NonrigidCpdConfig()
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_root": str(cohort_root),
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "physical_objects": {
            key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
        },
        "cpd_config": asdict(cfg),
        "stress_seed": STRESS_SEED,
        "observation_stresses": [asdict(stress) for stress in OBSERVATION_STRESSES],
        "tempering_gains": list(TEMPERING_GAINS),
        "adaptive_effective_support_rule": {
            "formula": "clip(effective_correspondence_count / 16, 0.75, 1)",
            "minimum_gain": ADAPTIVE_MINIMUM_GAIN,
            "nominal_center_count": CENTER_COUNT,
            "development_status": (
                "post-hoc after fixed-gain open-panel v1; frozen before raw-camera "
                "or held-target evaluation"
            ),
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "fit_diagnostics": fit_diagnostics,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "physical object",
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "bounded development-only temporal ablation on the already-open 27; "
            "synthetic current-centre corruptions and fused material measurements; "
            "not a held-target, raw-camera, or official Deform360 Table-4 result"
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ADAPTIVE_ARM",
    "ADAPTIVE_MINIMUM_GAIN",
    "ARMS",
    "INDEPENDENT_ARM",
    "CONSTANT_VELOCITY_TRACKER_ARM",
    "FROZEN_TRACKER_ARM",
    "OBSERVATION_STRESSES",
    "PROTOCOL_ID",
    "RECURSIVE_ARMS",
    "SELECTED_RAW_BACKBONE_ARM",
    "TEMPERING_GAINS",
    "TRACKER_ARMS",
    "CpdObservationStress",
    "TemperedCpdFieldState",
    "corrupt_current_unordered_set",
    "decode_tempered_cpd_field",
    "effective_support_tempering_gain",
    "evaluate_deform360_recursive_cpd_case",
    "evaluate_deform360_recursive_cpd_cohort",
    "update_tempered_cpd_field",
]
