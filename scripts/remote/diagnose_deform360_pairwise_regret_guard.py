#!/usr/bin/env python3
"""Cross-fit a source-only regret guard for the exact pairwise RBF candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.bias_aware_belief import fit_source_regret_certificate
from bayesian_phystwin.deform360_cpd_diagnostic import (
    _symmetric_set_chamfer_m,
)
from bayesian_phystwin.deform360_online_belief_evaluation import (
    _load_pickle,
    score_deform360_hidden_trajectory,
)
from bayesian_phystwin.deform360_pairwise_bias_aware_transfer import (
    validate_open27_transfer_bundle,
)
from bayesian_phystwin.deform360_pairwise_regret_guard import (
    DUAL_BACKBONE_ARM,
    GUARDED_ARM,
    PAIRWISE_REGRET_FEATURE_NAMES,
    PRIOR_CORRECTION_AT_UPDATE,
    PRIOR_VARIANCE_AT_UPDATE,
    SELECTED_BACKBONE_ARM,
    SELECTED_BACKBONE_AT_UPDATE,
    apply_pairwise_regret_certificate,
    pairwise_regret_features,
    predict_dual_backbone_pairwise_rbf_arrays,
)

UPDATE_FRAMES = (19, 38, 57)
REGRET_NOMINAL_COVERAGE = 0.90
REGRET_WITHIN_GROUP_COVERAGE = 1.0
REGRET_MINIMUM_IMPROVEMENT_M = 0.000005
REGRET_RIDGE_PENALTY = 10.0
PRIVILEGED_DENSE_ACTION_GUARD_ARM = "privileged_dense_action_guard"
PRIVILEGED_CURRENT_GAIN_M = 0.0001
PRIVILEGED_FUTURE_COSINE = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--opened-stress-root",
        type=Path,
        help="Optional already-open fresh pairwise root for post-open stress only.",
    )
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


def _file_path(
    root: Path,
    record: dict[str, Any],
    role: str,
) -> Path:
    metadata = record["files"][role]
    return root / metadata["root"] / metadata["relative_path"]


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    frames: tuple[int, ...],
) -> dict[str, object]:
    return score_deform360_hidden_trajectory(
        trajectory,
        target,
        visibility,
        validity,
        center_ids=center_ids,
        scored_frames=frames,
    )


def _metric_pair(score: dict[str, object]) -> tuple[float, float]:
    return (
        float(score["post_update_hidden_identity_rmse_m"]),
        float(score["post_update_hidden_symmetric_chamfer_m"]),
    )


def _current_dense_chamfer_diagnostic(
    baseline_m: np.ndarray,
    candidate_m: np.ndarray,
    target_m: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
    *,
    update_frame: int,
) -> dict[str, float | int | None]:
    available = (
        np.asarray(target_visibility[update_frame], dtype=bool)
        & np.asarray(target_validity[update_frame], dtype=bool)
        & np.all(np.isfinite(target_m[update_frame]), axis=1)
    )
    point_count = int(np.sum(available))
    if not point_count:
        return {
            "available_point_count": 0,
            "baseline_chamfer_m": None,
            "candidate_chamfer_m": None,
            "candidate_gain_m": None,
        }
    correction = candidate_m[update_frame + 1] - baseline_m[update_frame + 1]
    baseline_current = baseline_m[update_frame]
    candidate_current = baseline_current + correction
    observed = target_m[update_frame, available]
    baseline_chamfer = _symmetric_set_chamfer_m(baseline_current, observed)
    candidate_chamfer = _symmetric_set_chamfer_m(candidate_current, observed)
    return {
        "available_point_count": point_count,
        "baseline_chamfer_m": baseline_chamfer,
        "candidate_chamfer_m": candidate_chamfer,
        "candidate_gain_m": baseline_chamfer - candidate_chamfer,
    }


def _apply_privileged_dense_action_guard(
    physical_prior_m: np.ndarray,
    baseline_m: np.ndarray,
    candidate_m: np.ndarray,
    target_m: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    guarded = np.asarray(baseline_m).copy()
    decisions = []
    previous = 0
    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(guarded)
        )
        dense = _current_dense_chamfer_diagnostic(
            baseline_m,
            candidate_m,
            target_m,
            target_visibility,
            target_validity,
            update_frame=update,
        )
        context = _known_future_physical_context(
            physical_prior_m,
            baseline_m,
            candidate_m,
            previous_update_frame=previous,
            update_frame=update,
            interval_end_exclusive=stop,
        )
        current_gain = dense["candidate_gain_m"]
        accepted = bool(
            current_gain is not None
            and current_gain > PRIVILEGED_CURRENT_GAIN_M
            and context["correction_future_motion_cosine"]
            > PRIVILEGED_FUTURE_COSINE
        )
        if accepted:
            guarded[update + 1 : stop] = candidate_m[update + 1 : stop]
        elif not np.array_equal(
            guarded[update + 1 : stop],
            baseline_m[update + 1 : stop],
        ):
            raise AssertionError("privileged rejection changed the exact baseline")
        decisions.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_accepted": accepted,
                "current_dense_shape": dense,
                "known_future_physical_context": context,
                "current_gain_threshold_m": PRIVILEGED_CURRENT_GAIN_M,
                "future_cosine_threshold": PRIVILEGED_FUTURE_COSINE,
                "bit_exact_baseline_fallback": bool(
                    not accepted
                    and np.array_equal(
                        guarded[update + 1 : stop],
                        baseline_m[update + 1 : stop],
                    )
                ),
            }
        )
        previous = update
    return guarded, decisions


def _known_future_physical_context(
    physical_prior_m: np.ndarray,
    baseline_m: np.ndarray,
    candidate_m: np.ndarray,
    *,
    previous_update_frame: int,
    update_frame: int,
    interval_end_exclusive: int,
) -> dict[str, float]:
    physical = np.asarray(physical_prior_m, dtype=np.float64)
    baseline = np.asarray(baseline_m, dtype=np.float64)
    candidate = np.asarray(candidate_m, dtype=np.float64)
    center = np.median(physical[0], axis=0)
    object_scale = max(
        1e-6,
        float(2.0 * np.max(np.linalg.norm(physical[0] - center, axis=1))),
    )
    past_delta = physical[update_frame] - physical[previous_update_frame]
    future_delta = (
        physical[interval_end_exclusive - 1] - physical[update_frame]
    )
    first_step = physical[update_frame + 1] - physical[update_frame]
    correction = (
        candidate[update_frame + 1] - baseline[update_frame + 1]
    )

    def radial_rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-12:
            return 0.0
        return float(np.sum(left * right) / denominator)

    step_rms = [
        radial_rms(physical[frame] - physical[frame - 1])
        for frame in range(update_frame + 1, interval_end_exclusive)
    ]
    return {
        "future_net_motion_rms_over_object_scale": (
            radial_rms(future_delta) / object_scale
        ),
        "future_path_length_rms_over_object_scale": (
            float(np.sum(step_rms)) / object_scale
        ),
        "future_to_past_motion_ratio": (
            radial_rms(future_delta) / max(radial_rms(past_delta), 1e-9)
        ),
        "past_future_motion_cosine": cosine(past_delta, future_delta),
        "correction_future_motion_cosine": cosine(correction, future_delta),
        "correction_first_step_cosine": cosine(correction, first_step),
        "first_step_future_motion_cosine": cosine(first_step, future_delta),
    }


def _certificate_payload(certificate: Any) -> dict[str, Any]:
    payload = asdict(certificate)
    for key, value in tuple(payload.items()):
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    return payload


def _object_balanced(
    cases: list[dict[str, Any]],
    arm: str,
) -> dict[str, float]:
    by_object: dict[str, list[tuple[float, float]]] = {}
    for case in cases:
        by_object.setdefault(case["object_id"], []).append(
            _metric_pair(case["scores"][arm])
        )
    object_means = {
        object_id: np.mean(values, axis=0)
        for object_id, values in by_object.items()
    }
    aggregate = np.mean(tuple(object_means.values()), axis=0)
    return {
        "hidden_identity_rmse_m": float(aggregate[0]),
        "hidden_symmetric_chamfer_m": float(aggregate[1]),
    }


def _relative(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] / baseline[key] - 1.0)
        for key in baseline
    }


def _source_cases(bundle_root: Path) -> tuple[list[dict[str, Any]], str]:
    validation = validate_open27_transfer_bundle(bundle_root)
    manifest = json.loads(
        (bundle_root / "transfer_manifest.json").read_text(encoding="utf-8")
    )
    cases = []
    scored_frames = tuple(
        frame
        for frame in range(UPDATE_FRAMES[0] + 1, 76)
        if frame not in UPDATE_FRAMES[1:]
    )
    for record in manifest["cases"]:
        prediction = _load_npz(_file_path(bundle_root, record, "prediction_archive"))
        measurement = _load_npz(_file_path(bundle_root, record, "measurement"))
        target_data = _load_pickle(_file_path(bundle_root, record, "source_target"))
        target = np.asarray(target_data["object_points"])
        visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
        validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
        center_ids = np.asarray(measurement["center_ids"], dtype=np.int64)
        report, arrays = predict_dual_backbone_pairwise_rbf_arrays(
            prediction["prediction_m"],
            prediction["persistence_m"],
            measurement["measurement_m"],
            measurement["measurement_visibility"],
            measurement["measurement_validity"],
            center_ids=center_ids,
            update_frames=UPDATE_FRAMES,
        )
        privileged, privileged_decisions = _apply_privileged_dense_action_guard(
            prediction["prediction_m"],
            arrays[SELECTED_BACKBONE_ARM],
            arrays[DUAL_BACKBONE_ARM],
            target,
            visibility,
            validity,
        )
        features = []
        interval_regret = []
        interval_diagnostics = []
        previous = 0
        for update_index, update in enumerate(UPDATE_FRAMES):
            stop = (
                UPDATE_FRAMES[update_index + 1]
                if update_index + 1 < len(UPDATE_FRAMES)
                else len(target)
            )
            vector, diagnostics = pairwise_regret_features(
                prediction["prediction_m"],
                arrays[SELECTED_BACKBONE_ARM],
                arrays[DUAL_BACKBONE_ARM],
                measurement["measurement_m"],
                measurement["measurement_visibility"],
                measurement["measurement_validity"],
                center_ids=center_ids,
                update_frame=update,
                previous_update_frame=previous,
                interval_end_exclusive=stop,
                inlier_view_count=measurement["triangulation_inlier_view_count"][
                    update_index
                ],
                prior_correction_at_update_m=arrays[
                    PRIOR_CORRECTION_AT_UPDATE
                ][update],
                prior_variance_at_update_m2=arrays[
                    PRIOR_VARIANCE_AT_UPDATE
                ][update],
                selected_backbone_at_update_m=arrays[
                    SELECTED_BACKBONE_AT_UPDATE
                ][update],
            )
            report["updates"][update_index]["regret_features"] = diagnostics
            frames = tuple(range(update + 1, stop))
            baseline_score = _score(
                arrays[SELECTED_BACKBONE_ARM],
                target,
                visibility,
                validity,
                center_ids,
                frames,
            )
            candidate_score = _score(
                arrays[DUAL_BACKBONE_ARM],
                target,
                visibility,
                validity,
                center_ids,
                frames,
            )
            baseline_pair = _metric_pair(baseline_score)
            candidate_pair = _metric_pair(candidate_score)
            features.append(vector)
            interval_regret.append(
                max(
                    candidate_pair[0] - baseline_pair[0],
                    candidate_pair[1] - baseline_pair[1],
                )
            )
            interval_diagnostics.append(
                {
                    "frame": update,
                    "interval_end_exclusive": stop,
                    "features": diagnostics,
                    "baseline_hidden_identity_rmse_m": baseline_pair[0],
                    "baseline_hidden_symmetric_chamfer_m": baseline_pair[1],
                    "candidate_hidden_identity_rmse_m": candidate_pair[0],
                    "candidate_hidden_symmetric_chamfer_m": candidate_pair[1],
                    "maximum_metric_regret_m": interval_regret[-1],
                    "postopen_current_dense_shape": (
                        _current_dense_chamfer_diagnostic(
                            arrays[SELECTED_BACKBONE_ARM],
                            arrays[DUAL_BACKBONE_ARM],
                            target,
                            visibility,
                            validity,
                            update_frame=update,
                        )
                    ),
                    "known_future_physical_context": (
                        _known_future_physical_context(
                            prediction["prediction_m"],
                            arrays[SELECTED_BACKBONE_ARM],
                            arrays[DUAL_BACKBONE_ARM],
                            previous_update_frame=previous,
                            update_frame=update,
                            interval_end_exclusive=stop,
                        )
                    ),
                }
            )
            previous = update
        cases.append(
            {
                "case": record["case"],
                "object_id": record["object_id"],
                "target": target,
                "visibility": visibility,
                "validity": validity,
                "center_ids": center_ids,
                "arrays": arrays,
                "features": np.asarray(features),
                "interval_regret_m": np.asarray(interval_regret),
                "interval_diagnostics": interval_diagnostics,
                "candidate_report": report,
                "privileged_dense_action_guard": privileged_decisions,
                "scores": {
                    SELECTED_BACKBONE_ARM: _score(
                        arrays[SELECTED_BACKBONE_ARM],
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                    DUAL_BACKBONE_ARM: _score(
                        arrays[DUAL_BACKBONE_ARM],
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                    PRIVILEGED_DENSE_ACTION_GUARD_ARM: _score(
                        privileged,
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                },
            }
        )
    return cases, str(validation["manifest_sha256"])


def _fit_certificate(cases: list[dict[str, Any]]) -> Any:
    return fit_source_regret_certificate(
        np.concatenate([case["features"] for case in cases]),
        np.concatenate([case["interval_regret_m"] for case in cases]),
        tuple(
            case["object_id"]
            for case in cases
            for _ in range(len(case["features"]))
        ),
        nominal_coverage=REGRET_NOMINAL_COVERAGE,
        within_group_coverage=REGRET_WITHIN_GROUP_COVERAGE,
        minimum_improvement=REGRET_MINIMUM_IMPROVEMENT_M,
        ridge_penalty=REGRET_RIDGE_PENALTY,
        support_margin_std=0.0,
    )


def _cross_fit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for held_object in sorted({case["object_id"] for case in cases}):
        training = [case for case in cases if case["object_id"] != held_object]
        held = [case for case in cases if case["object_id"] == held_object]
        certificate = _fit_certificate(training)
        for case in held:
            guard_report, guarded = apply_pairwise_regret_certificate(
                case["arrays"][SELECTED_BACKBONE_ARM],
                case["arrays"][DUAL_BACKBONE_ARM],
                case["features"],
                certificate,
                update_frames=UPDATE_FRAMES,
            )
            case["scores"][GUARDED_ARM] = _score(
                guarded,
                case["target"],
                case["visibility"],
                case["validity"],
                case["center_ids"],
                tuple(
                    frame
                    for frame in range(UPDATE_FRAMES[0] + 1, len(guarded))
                    if frame not in UPDATE_FRAMES[1:]
                ),
            )
            records.append(
                {
                    "case": case["case"],
                    "held_object": held_object,
                    "guard": guard_report,
                }
            )
    return {"cases": records}


def _opened_stress_cases(
    root: Path,
    certificate: Any,
) -> list[dict[str, Any]]:
    cases = []
    for outcome_path in sorted((root / "outcome-0a01fb8").glob("*.json")):
        if outcome_path.name == "summary.json":
            continue
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        case_name = str(outcome["case"])
        prediction_dir = root / "predictions" / case_name
        prediction = _load_npz(prediction_dir / "belief_prediction.npz")
        measurement = _load_npz(root / "measurements" / case_name / "measurement.npz")
        candidate_report, candidate_arrays = (
            predict_dual_backbone_pairwise_rbf_arrays(
                prediction["physical_prior_m"],
                prediction["persistence_m"],
                measurement["measurement_m"],
                measurement["measurement_visibility"],
                measurement["measurement_validity"],
                center_ids=np.asarray(prediction["center_ids"], dtype=np.int64),
                update_frames=UPDATE_FRAMES,
            )
        )
        for name, archived in (
            (SELECTED_BACKBONE_ARM, prediction["selected_raw_backbone_m"]),
            (DUAL_BACKBONE_ARM, prediction["candidate_m"]),
        ):
            if not np.array_equal(candidate_arrays[name], archived):
                raise ValueError(
                    f"{case_name}: rebuilt {name} differs from archived prediction"
                )
        target_path = Path(outcome["inputs"]["future_payload"]["path"])
        target_data = _load_pickle(target_path)
        target = np.asarray(target_data["object_points"])
        visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
        validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
        center_ids = np.asarray(prediction["center_ids"], dtype=np.int64)
        privileged, privileged_decisions = _apply_privileged_dense_action_guard(
            prediction["physical_prior_m"],
            prediction["selected_raw_backbone_m"],
            prediction["candidate_m"],
            target,
            visibility,
            validity,
        )
        features = []
        feature_diagnostics = []
        interval_diagnostics = []
        previous = 0
        for update_index, update in enumerate(UPDATE_FRAMES):
            stop = (
                UPDATE_FRAMES[update_index + 1]
                if update_index + 1 < len(UPDATE_FRAMES)
                else len(target)
            )
            vector, diagnostics = pairwise_regret_features(
                prediction["physical_prior_m"],
                prediction["selected_raw_backbone_m"],
                prediction["candidate_m"],
                measurement["measurement_m"],
                measurement["measurement_visibility"],
                measurement["measurement_validity"],
                center_ids=center_ids,
                update_frame=update,
                previous_update_frame=previous,
                interval_end_exclusive=stop,
                inlier_view_count=measurement["triangulation_inlier_view_count"][
                    update_index
                ],
                prior_correction_at_update_m=candidate_arrays[
                    PRIOR_CORRECTION_AT_UPDATE
                ][update],
                prior_variance_at_update_m2=candidate_arrays[
                    PRIOR_VARIANCE_AT_UPDATE
                ][update],
                selected_backbone_at_update_m=candidate_arrays[
                    SELECTED_BACKBONE_AT_UPDATE
                ][update],
            )
            features.append(vector)
            feature_diagnostics.append(diagnostics)
            frames = tuple(range(update + 1, stop))
            baseline_pair = _metric_pair(
                _score(
                    prediction["selected_raw_backbone_m"],
                    target,
                    visibility,
                    validity,
                    center_ids,
                    frames,
                )
            )
            candidate_pair = _metric_pair(
                _score(
                    prediction["candidate_m"],
                    target,
                    visibility,
                    validity,
                    center_ids,
                    frames,
                )
            )
            interval_diagnostics.append(
                {
                    "frame": update,
                    "interval_end_exclusive": stop,
                    "features": diagnostics,
                    "baseline_hidden_identity_rmse_m": baseline_pair[0],
                    "baseline_hidden_symmetric_chamfer_m": baseline_pair[1],
                    "candidate_hidden_identity_rmse_m": candidate_pair[0],
                    "candidate_hidden_symmetric_chamfer_m": candidate_pair[1],
                    "maximum_metric_regret_m": max(
                        candidate_pair[0] - baseline_pair[0],
                        candidate_pair[1] - baseline_pair[1],
                    ),
                    "postopen_current_dense_shape": (
                        _current_dense_chamfer_diagnostic(
                            prediction["selected_raw_backbone_m"],
                            prediction["candidate_m"],
                            target,
                            visibility,
                            validity,
                            update_frame=update,
                        )
                    ),
                    "known_future_physical_context": (
                        _known_future_physical_context(
                            prediction["physical_prior_m"],
                            prediction["selected_raw_backbone_m"],
                            prediction["candidate_m"],
                            previous_update_frame=previous,
                            update_frame=update,
                            interval_end_exclusive=stop,
                        )
                    ),
                }
            )
            previous = update
        guard_report, guarded = apply_pairwise_regret_certificate(
            prediction["selected_raw_backbone_m"],
            prediction["candidate_m"],
            np.asarray(features),
            certificate,
            update_frames=UPDATE_FRAMES,
        )
        scored_frames = tuple(
            frame
            for frame in range(UPDATE_FRAMES[0] + 1, len(target))
            if frame not in UPDATE_FRAMES[1:]
        )
        cases.append(
            {
                "case": case_name,
                "object_id": str(outcome["object_id"]),
                "guard": guard_report,
                "candidate_report": candidate_report,
                "feature_diagnostics": feature_diagnostics,
                "interval_diagnostics": interval_diagnostics,
                "privileged_dense_action_guard": privileged_decisions,
                "scores": {
                    SELECTED_BACKBONE_ARM: _score(
                        prediction["selected_raw_backbone_m"],
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                    DUAL_BACKBONE_ARM: _score(
                        prediction["candidate_m"],
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                    PRIVILEGED_DENSE_ACTION_GUARD_ARM: _score(
                        privileged,
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                    GUARDED_ARM: _score(
                        guarded,
                        target,
                        visibility,
                        validity,
                        center_ids,
                        scored_frames,
                    ),
                },
            }
        )
    return cases


def main() -> None:
    args = _parse_args()
    cases, transfer_sha256 = _source_cases(args.bundle_root.resolve())
    cross_fit = _cross_fit(cases)
    certificate = _fit_certificate(cases)
    source_aggregate = {
        arm: _object_balanced(cases, arm)
        for arm in (
            SELECTED_BACKBONE_ARM,
            DUAL_BACKBONE_ARM,
            GUARDED_ARM,
            PRIVILEGED_DENSE_ACTION_GUARD_ARM,
        )
    }
    payload: dict[str, Any] = {
        "artifact_kind": "Deform360PairwiseRegretGuardSourceDiagnostic",
        "feature_names": list(PAIRWISE_REGRET_FEATURE_NAMES),
        "transfer_manifest_sha256": transfer_sha256,
        "source_case_count": len(cases),
        "source_object_count": len({case["object_id"] for case in cases}),
        "source_cross_fit": cross_fit,
        "source_interval_diagnostics": [
            {
                "case": case["case"],
                "object_id": case["object_id"],
                "intervals": case["interval_diagnostics"],
            }
            for case in cases
        ],
        "source_case_results": [
            {
                "case": case["case"],
                "object_id": case["object_id"],
                "privileged_dense_action_guard": case[
                    "privileged_dense_action_guard"
                ],
                "scores": case["scores"],
            }
            for case in cases
        ],
        "source_object_balanced": source_aggregate,
        "source_relative_to_selected_backbone": {
            arm: _relative(source_aggregate[arm], source_aggregate[SELECTED_BACKBONE_ARM])
            for arm in (
                DUAL_BACKBONE_ARM,
                GUARDED_ARM,
                PRIVILEGED_DENSE_ACTION_GUARD_ARM,
            )
        },
        "full_source_certificate": _certificate_payload(certificate),
        "privileged_capacity_control": {
            "arm": PRIVILEGED_DENSE_ACTION_GUARD_ARM,
            "current_gain_threshold_m": PRIVILEGED_CURRENT_GAIN_M,
            "future_cosine_threshold": PRIVILEGED_FUTURE_COSINE,
            "uses_score_family_current_material_identities": True,
            "deployable_causal_observation": False,
            "purpose": (
                "Post-open capacity check for current material-state validation "
                "plus known-action discrepancy transfer."
            ),
        },
        "claim_boundary": (
            "Opened-source development diagnostic. The privileged control reads "
            "score-family identities only at the current update and is not a "
            "deployable predictor. The optional stress result is post-open. Neither "
            "result can establish prospective accuracy or state of the art."
        ),
    }
    if args.opened_stress_root is not None:
        stress = _opened_stress_cases(args.opened_stress_root.resolve(), certificate)
        stress_aggregate = {
            arm: _object_balanced(stress, arm)
            for arm in (
                SELECTED_BACKBONE_ARM,
                DUAL_BACKBONE_ARM,
                GUARDED_ARM,
                PRIVILEGED_DENSE_ACTION_GUARD_ARM,
            )
        }
        payload["opened_stress"] = {
            "case_count": len(stress),
            "object_count": len({case["object_id"] for case in stress}),
            "cases": stress,
            "object_balanced": stress_aggregate,
            "relative_to_selected_backbone": {
                arm: _relative(
                    stress_aggregate[arm],
                    stress_aggregate[SELECTED_BACKBONE_ARM],
                )
                for arm in (
                    DUAL_BACKBONE_ARM,
                    GUARDED_ARM,
                    PRIVILEGED_DENSE_ACTION_GUARD_ARM,
                )
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output": str(args.output.resolve()),
            "source_object_balanced": source_aggregate,
            "opened_stress_object_balanced": (
                None
                if "opened_stress" not in payload
                else payload["opened_stress"]["object_balanced"]
            ),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
