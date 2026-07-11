"""Case-held-out validation of regenerated PhysTwin perception reliability cues."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_official_evaluation import _nearest_distances


@dataclass(frozen=True)
class PerceptionCueEvaluationProtocol:
    """Frozen development selection and case-held-out evaluation choices."""

    fit_fraction: float = 0.75
    forward_backward_scale_candidates_px: tuple[float, ...] = (
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    )
    multiview_scale_candidates_px: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    maximum_initial_manual_match_m: float = 0.01
    corruption_thresholds_m: tuple[float, ...] = (0.005, 0.01)
    boundary_scale: float = 0.003
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 20260711
    development_cases: tuple[str, ...] = DEVELOPMENT_CASES


@dataclass(frozen=True)
class ReliabilityTransform:
    """Residual-independent mapping from regenerated cues to reliability."""

    name: str
    forward_backward_scale_px: float | None = None
    multiview_scale_px: float | None = None


@dataclass(frozen=True)
class ManualCueObservations:
    """Manual 3D errors and aligned release/regenerated cue values for one case."""

    error_m: np.ndarray
    hard_valid: np.ndarray
    boundary_distance: np.ndarray
    confidence: np.ndarray
    visibility_probability: np.ndarray
    forward_backward_error_px: np.ndarray
    forward_backward_valid: np.ndarray
    multiview_reprojection_error_px: np.ndarray
    multiview_valid: np.ndarray
    frame_index: np.ndarray
    manual_track_index: np.ndarray
    aligned_object_track_index: np.ndarray
    initial_match_distance_m: np.ndarray


def compose_perception_reliability(
    cues: Mapping[str, np.ndarray],
    transform: ReliabilityTransform,
    *,
    minimum_probability: float = 1e-6,
) -> np.ndarray:
    """Compose network, cycle, and multiview cues with neutral missing values."""

    confidence = np.asarray(cues["confidence"], dtype=float)
    visibility = np.asarray(cues["visibility_probability"], dtype=float)
    if confidence.shape != visibility.shape:
        raise ValueError("confidence and visibility_probability must have equal shapes")
    reliability = np.clip(confidence, 0.0, 1.0) * np.clip(
        visibility, 0.0, 1.0
    )
    if transform.forward_backward_scale_px is not None:
        if transform.forward_backward_scale_px <= 0.0:
            raise ValueError("forward_backward_scale_px must be positive")
        error = np.asarray(cues["forward_backward_error_px"], dtype=float)
        valid = np.asarray(cues["forward_backward_valid"], dtype=bool)
        if error.shape != reliability.shape or valid.shape != reliability.shape:
            raise ValueError("forward/backward cues must match confidence")
        factor = np.ones_like(reliability)
        factor[valid] = np.exp(
            -error[valid] / transform.forward_backward_scale_px
        )
        reliability *= factor
    if transform.multiview_scale_px is not None:
        if transform.multiview_scale_px <= 0.0:
            raise ValueError("multiview_scale_px must be positive")
        error = np.asarray(cues["multiview_reprojection_error_px"], dtype=float)
        valid = np.asarray(cues["multiview_valid"], dtype=bool)
        if error.shape != reliability.shape or valid.shape != reliability.shape:
            raise ValueError("multiview cues must match confidence")
        factor = np.ones_like(reliability)
        factor[valid] = np.exp(-error[valid] / transform.multiview_scale_px)
        reliability *= factor
    return np.clip(reliability, minimum_probability, 1.0)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _hard_valid_by_target_frame(
    visible: np.ndarray, motion_valid: np.ndarray
) -> np.ndarray:
    visible_array = np.asarray(visible, dtype=bool)
    motion = np.asarray(motion_valid, dtype=bool)
    if motion.shape not in {visible_array.shape, (len(visible_array) - 1, visible_array.shape[1])}:
        raise ValueError("object_motions_valid has an incompatible shape")
    hard = np.zeros_like(visible_array)
    hard[0] = visible_array[0]
    hard[1:] = motion[: len(visible_array) - 1]
    return hard


def load_manual_cue_observations(
    case_dir: str | Path,
    cues_path: str | Path,
    *,
    maximum_initial_match_m: float,
) -> ManualCueObservations:
    """Align manual annotations to processed pseudo tracks over training frames."""

    case_path = Path(case_dir)
    data = _load_pickle(case_path / "final_data.pkl")
    manual = np.asarray(_load_pickle(case_path / "gt_track_3d.pkl"), dtype=float)
    split = json.loads((case_path / "split.json").read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    points = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    hard = _hard_valid_by_target_frame(
        visible, np.asarray(data["object_motions_valid"], dtype=bool)
    )
    if manual.shape[0] < train_end or points.shape[0] < train_end:
        raise ValueError("manual and processed tracks must cover the training interval")
    manual_initial_valid = np.isfinite(manual[0]).all(axis=1)
    distances, object_indices = _nearest_distances(
        points[0], manual[0, manual_initial_valid], p=2
    )
    accepted = distances <= maximum_initial_match_m
    manual_indices = np.flatnonzero(manual_initial_valid)[accepted]
    object_indices = object_indices[accepted]
    distances = distances[accepted]
    if len(manual_indices) == 0:
        raise ValueError(f"no manual tracks align within tolerance for {case_path.name}")
    with np.load(cues_path) as archive:
        cue_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "confidence",
        "visibility_probability",
        "forward_backward_error_px",
        "forward_backward_valid",
        "multiview_reprojection_error_px",
        "multiview_valid",
        "boundary_distance",
        "cue_available",
    }
    missing = required - set(cue_arrays)
    if missing:
        raise ValueError(f"cue archive is missing: {', '.join(sorted(missing))}")

    rows: dict[str, list[np.ndarray]] = {
        "error_m": [],
        "hard_valid": [],
        "boundary_distance": [],
        "confidence": [],
        "visibility_probability": [],
        "forward_backward_error_px": [],
        "forward_backward_valid": [],
        "multiview_reprojection_error_px": [],
        "multiview_valid": [],
        "frame_index": [],
        "manual_track_index": [],
        "aligned_object_track_index": [],
    }
    for frame in range(1, train_end):
        frame_manual = manual[frame, manual_indices]
        usable = (
            np.isfinite(frame_manual).all(axis=1)
            & visible[frame, object_indices]
            & np.asarray(cue_arrays["cue_available"][frame, object_indices], dtype=bool)
        )
        if not np.any(usable):
            continue
        selected_manual = manual_indices[usable]
        selected_object = object_indices[usable]
        rows["error_m"].append(
            np.linalg.norm(
                points[frame, selected_object] - manual[frame, selected_manual],
                axis=1,
            )
        )
        rows["hard_valid"].append(hard[frame, selected_object])
        for name in (
            "boundary_distance",
            "confidence",
            "visibility_probability",
            "forward_backward_error_px",
            "forward_backward_valid",
            "multiview_reprojection_error_px",
            "multiview_valid",
        ):
            rows[name].append(cue_arrays[name][frame, selected_object])
        rows["frame_index"].append(
            np.full(len(selected_object), frame, dtype=np.int32)
        )
        rows["manual_track_index"].append(selected_manual.astype(np.int32))
        rows["aligned_object_track_index"].append(selected_object.astype(np.int32))
    if not rows["error_m"]:
        raise ValueError(f"no aligned training observations for {case_path.name}")
    concatenated = {name: np.concatenate(values) for name, values in rows.items()}
    return ManualCueObservations(
        **concatenated,
        initial_match_distance_m=distances,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _rankdata(np.asarray(left, dtype=float))
    right_rank = _rankdata(np.asarray(right, dtype=float))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _auroc(labels: np.ndarray, score: np.ndarray) -> float | None:
    binary = np.asarray(labels, dtype=bool)
    positive = int(np.sum(binary))
    negative = len(binary) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = _rankdata(np.asarray(score, dtype=float))
    rank_sum = float(np.sum(ranks[binary]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _mean_ignoring_nonfinite(
    values: np.ndarray | list[float], *, axis: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0,
    )


def reliability_error_metrics(
    reliability: np.ndarray,
    error_m: np.ndarray,
    *,
    corruption_thresholds_m: tuple[float, ...] = (0.005, 0.01),
) -> dict[str, Any]:
    """Measure ranking and selective error without interpreting scores as calibrated."""

    score = np.asarray(reliability, dtype=float)
    error = np.asarray(error_m, dtype=float)
    if score.shape != error.shape or score.ndim != 1:
        raise ValueError("reliability and error_m must be equal-length vectors")
    if len(score) < 4 or not np.all(np.isfinite(score)) or not np.all(np.isfinite(error)):
        raise ValueError("metrics require at least four finite observations")
    order = np.argsort(score, kind="mergesort")
    quartile_count = max(1, len(score) // 4)
    half_count = max(1, len(score) // 2)
    mean_error = float(np.mean(error))
    return {
        "observation_count": int(len(score)),
        "mean_error_m": mean_error,
        "spearman_reliability_vs_error": _spearman(score, error),
        "lowest_reliability_quartile_error_m": float(
            np.mean(error[order[:quartile_count]])
        ),
        "highest_reliability_quartile_error_m": float(
            np.mean(error[order[-quartile_count:]])
        ),
        "highest_reliability_half_error_m": float(
            np.mean(error[order[-half_count:]])
        ),
        "highest_reliability_half_error_ratio": float(
            np.mean(error[order[-half_count:]]) / mean_error
        ),
        "unreliability_auroc": {
            f"error_at_least_{threshold:g}_m": _auroc(
                error >= threshold, -score
            )
            for threshold in corruption_thresholds_m
        },
    }


def _observation_cues(observations: ManualCueObservations) -> dict[str, np.ndarray]:
    return {
        "confidence": observations.confidence,
        "visibility_probability": observations.visibility_probability,
        "forward_backward_error_px": observations.forward_backward_error_px,
        "forward_backward_valid": observations.forward_backward_valid,
        "multiview_reprojection_error_px": (
            observations.multiview_reprojection_error_px
        ),
        "multiview_valid": observations.multiview_valid,
    }


def _candidate_transforms(
    protocol: PerceptionCueEvaluationProtocol,
) -> tuple[ReliabilityTransform, ...]:
    candidates = [ReliabilityTransform("network")]
    candidates.extend(
        ReliabilityTransform(f"network_fb_{scale:g}", scale, None)
        for scale in protocol.forward_backward_scale_candidates_px
    )
    candidates.extend(
        ReliabilityTransform(f"network_mv_{scale:g}", None, scale)
        for scale in protocol.multiview_scale_candidates_px
    )
    candidates.extend(
        ReliabilityTransform(f"network_fb_{fb:g}_mv_{mv:g}", fb, mv)
        for fb in protocol.forward_backward_scale_candidates_px
        for mv in protocol.multiview_scale_candidates_px
    )
    return tuple(candidates)


def _method_scores(
    observations: ManualCueObservations,
    transforms: Mapping[str, ReliabilityTransform],
    *,
    boundary_scale: float,
) -> dict[str, np.ndarray]:
    cues = _observation_cues(observations)
    scores = {
        "hard_binary": observations.hard_valid.astype(float),
        "boundary_proxy": 1.0
        - np.exp(-np.maximum(observations.boundary_distance, 0.0) / boundary_scale),
        "network": compose_perception_reliability(
            cues, ReliabilityTransform("network")
        ),
    }
    scores.update(
        {
            method: compose_perception_reliability(cues, transform)
            for method, transform in transforms.items()
        }
    )
    return scores


def _aggregate_cases(
    cases: tuple[str, ...],
    case_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    methods = tuple(case_results[cases[0]])
    metrics = (
        "spearman_reliability_vs_error",
        "lowest_reliability_quartile_error_m",
        "highest_reliability_quartile_error_m",
        "highest_reliability_half_error_ratio",
    )
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, len(cases), size=(samples, len(cases)))
    aggregate: dict[str, Any] = {}
    for method in methods:
        method_summary: dict[str, Any] = {}
        for metric in metrics:
            values = np.array(
                [
                    np.nan
                    if case_results[case][method][metric] is None
                    else float(case_results[case][method][metric])
                    for case in cases
                ],
                dtype=float,
            )
            mean = float(_mean_ignoring_nonfinite(values))
            boot = _mean_ignoring_nonfinite(values[bootstrap_indices], axis=1)
            method_summary[metric] = {
                "case_mean": mean,
                "case_bootstrap_95_interval": [
                    float(np.nanquantile(boot, 0.025)),
                    float(np.nanquantile(boot, 0.975)),
                ],
            }
        auroc_summary = {}
        auroc_names = tuple(
            case_results[cases[0]][method]["unreliability_auroc"]
        )
        for name in auroc_names:
            values = np.array(
                [
                    np.nan
                    if case_results[case][method]["unreliability_auroc"][name]
                    is None
                    else float(
                        case_results[case][method]["unreliability_auroc"][name]
                    )
                    for case in cases
                ],
                dtype=float,
            )
            boot = _mean_ignoring_nonfinite(values[bootstrap_indices], axis=1)
            auroc_summary[name] = {
                "case_mean": float(_mean_ignoring_nonfinite(values)),
                "case_bootstrap_95_interval": [
                    float(np.nanquantile(boot, 0.025)),
                    float(np.nanquantile(boot, 0.975)),
                ],
                "defined_case_count": int(np.sum(np.isfinite(values))),
            }
        method_summary["unreliability_auroc"] = auroc_summary
        aggregate[method] = method_summary
    comparisons = {}
    for method in (
        "network_fb_locked",
        "network_mv_locked",
        "network_fb_mv_locked",
        "selected_rich",
    ):
        selected = np.array(
            [case_results[case][method]["highest_reliability_half_error_ratio"] for case in cases]
        )
        for baseline in ("hard_binary", "boundary_proxy", "network"):
            base = np.array(
                [case_results[case][baseline]["highest_reliability_half_error_ratio"] for case in cases]
            )
            difference = selected - base
            boot = np.mean(difference[bootstrap_indices], axis=1)
            comparisons[f"{method}_minus_{baseline}"] = {
                "case_mean_difference": float(np.mean(difference)),
                "case_bootstrap_95_interval": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
                "improved_case_count": int(np.sum(difference < 0.0)),
                "case_count": len(cases),
            }
    return {"methods": aggregate, "comparisons": comparisons}


def run_perception_cue_confirmation(
    data_root: str | Path,
    cue_root: str | Path,
    output_dir: str | Path,
    *,
    protocol: PerceptionCueEvaluationProtocol | None = None,
) -> dict[str, Any]:
    """Select cue transforms on development cases and lock-test other cases."""

    config = protocol or PerceptionCueEvaluationProtocol()
    if not 0.0 < config.fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    if config.maximum_initial_manual_match_m <= 0.0:
        raise ValueError("maximum_initial_manual_match_m must be positive")
    if config.boundary_scale <= 0.0:
        raise ValueError("boundary_scale must be positive")
    root = Path(data_root)
    cues = Path(cue_root)
    output = Path(output_dir)
    source_manifest_path = root / "evaluation_subset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected_cases = tuple(str(case) for case in source_manifest["selected_cases"])
    development = tuple(
        case for case in selected_cases if case in config.development_cases
    )
    confirmation = tuple(
        case for case in selected_cases if case not in config.development_cases
    )
    specification = {
        "method": "case-held-out manual validation of regenerated perception cues",
        "protocol": asdict(config),
        "candidate_transforms": [
            asdict(candidate) for candidate in _candidate_transforms(config)
        ],
        "selection_objective": "case-mean highest-reliability-half error ratio on development fit frames",
        "development_evaluation_interval": "validation frames excluded from cue-scale selection",
        "evaluation_interval": "released training video only; no future frames decoded",
        "cohorts": {
            "development": list(development),
            "confirmation": list(confirmation),
        },
    }
    specification = json.loads(json.dumps(specification))
    locked = _lock_protocol(output, specification)
    observations = {
        case: load_manual_cue_observations(
            root / case,
            cues / case / "cues.npz",
            maximum_initial_match_m=config.maximum_initial_manual_match_m,
        )
        for case in selected_cases
    }
    fit_end_by_case = {}
    for case in selected_cases:
        split = json.loads((root / case / "split.json").read_text(encoding="utf-8"))
        train_start, train_end = (int(value) for value in split["train"])
        if train_start != 0:
            raise ValueError(f"unsupported nonzero training start for {case}")
        fit_end_by_case[case] = math.floor(config.fit_fraction * train_end)
    candidate_results = []
    best: tuple[tuple[float, float, int, str], ReliabilityTransform] | None = None
    ranked_candidates: list[
        tuple[tuple[float, float, int, str], ReliabilityTransform]
    ] = []
    for candidate in _candidate_transforms(config):
        per_case = {}
        for case in development:
            all_reliability = compose_perception_reliability(
                _observation_cues(observations[case]), candidate
            )
            selection = observations[case].frame_index < fit_end_by_case[case]
            per_case[case] = reliability_error_metrics(
                all_reliability[selection],
                observations[case].error_m[selection],
                corruption_thresholds_m=config.corruption_thresholds_m,
            )
        ratio = float(
            np.mean(
                [
                    per_case[case]["highest_reliability_half_error_ratio"]
                    for case in development
                ]
            )
        )
        correlations = [
            per_case[case]["spearman_reliability_vs_error"] for case in development
        ]
        mean_correlation = float(
            _mean_ignoring_nonfinite(
                [np.nan if value is None else value for value in correlations]
            )
        )
        complexity = int(candidate.forward_backward_scale_px is not None) + int(
            candidate.multiview_scale_px is not None
        )
        candidate_results.append(
            {
                "transform": asdict(candidate),
                "development_case_mean_high_reliability_half_error_ratio": ratio,
                "development_case_mean_spearman": mean_correlation,
                "case_metrics": per_case,
            }
        )
        ranking = (ratio, mean_correlation, complexity, candidate.name)
        ranked_candidates.append((ranking, candidate))
        if best is None or ranking < best[0]:
            best = (ranking, candidate)
    assert best is not None
    selected_transform = best[1]
    selected_transforms = {
        "network_fb_locked": min(
            item
            for item in ranked_candidates
            if item[1].forward_backward_scale_px is not None
            and item[1].multiview_scale_px is None
        )[1],
        "network_mv_locked": min(
            item
            for item in ranked_candidates
            if item[1].forward_backward_scale_px is None
            and item[1].multiview_scale_px is not None
        )[1],
        "network_fb_mv_locked": min(
            item
            for item in ranked_candidates
            if item[1].forward_backward_scale_px is not None
            and item[1].multiview_scale_px is not None
        )[1],
        "selected_rich": selected_transform,
    }

    case_results: dict[str, dict[str, dict[str, Any]]] = {}
    case_metadata: dict[str, Any] = {}
    for case in selected_cases:
        observation = observations[case]
        scores = _method_scores(
            observation,
            selected_transforms,
            boundary_scale=config.boundary_scale,
        )
        if case in development:
            evaluation = observation.frame_index >= fit_end_by_case[case]
            evaluation_interval = "development_validation"
        else:
            evaluation = np.ones(len(observation.error_m), dtype=bool)
            evaluation_interval = "case_held_out_training"
        case_results[case] = {
            method: reliability_error_metrics(
                score[evaluation],
                observation.error_m[evaluation],
                corruption_thresholds_m=config.corruption_thresholds_m,
            )
            for method, score in scores.items()
        }
        case_metadata[case] = {
            "manual_track_count": int(len(observation.initial_match_distance_m)),
            "manual_observation_count_total": int(len(observation.error_m)),
            "manual_observation_count_evaluated": int(np.sum(evaluation)),
            "evaluation_interval": evaluation_interval,
            "fit_end_frame": fit_end_by_case[case],
            "initial_match_distance_m": {
                "median": float(np.median(observation.initial_match_distance_m)),
                "maximum": float(np.max(observation.initial_match_distance_m)),
            },
            "multiview_valid_fraction": float(np.mean(observation.multiview_valid)),
        }
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_transform": asdict(selected_transform),
        "selected_component_transforms": {
            method: asdict(transform)
            for method, transform in selected_transforms.items()
        },
        "development_selection": candidate_results,
        "case_metadata": case_metadata,
        "case_results": case_results,
        "development": _aggregate_cases(
            development,
            case_results,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        ),
        "confirmation": _aggregate_cases(
            confirmation,
            case_results,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + 1,
        ),
        "interpretation_boundary": {
            "claim": "cue ranking of manual pseudo-track error on held-out cases",
            "not_claimed": "probability calibration or causal attribution of every tracking error",
            "confirmation_labels": "manual labels are used only for evaluation after development locking",
        },
    }
    result_path = output / "perception_cue_confirmation_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
