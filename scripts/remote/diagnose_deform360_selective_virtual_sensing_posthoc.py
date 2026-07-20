#!/usr/bin/env python3
"""Diagnose prospective virtual-sensing failures after outcomes are open."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_online_belief_evaluation import (
    _sha256,
    score_deform360_hidden_trajectory,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    MEASUREMENT_FILENAME,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_ARCHIVE_FILENAME,
    VIRTUAL_SENSING_REPORT_FILENAME,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_evaluation import (
    CASE_EVALUATION_FILENAME,
    PRIMARY_METRICS,
    SCORED_FRAMES,
    TARGET_ARCHIVE_FILENAME,
    _canonical_sha256 as evaluation_sha256,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_prediction import (
    predict_persistence_pairwise_rbf_arrays,
)
from bayesian_phystwin.phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
)
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig


SEALED_PRIMARY = "sealed_primary"
PERSISTENCE = "persistence"
VIEW3 = "posthoc_minimum_3view"
SNR5 = "posthoc_eb_snr5"
VIEW3_SNR5 = "posthoc_minimum_3view_eb_snr5"
ARMS = (SEALED_PRIMARY, PERSISTENCE, VIEW3, SNR5, VIEW3_SNR5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--outcome-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    target_visibility: np.ndarray,
    target_validity: np.ndarray,
    center_ids: np.ndarray,
) -> dict[str, float]:
    report = score_deform360_hidden_trajectory(
        trajectory,
        target,
        target_visibility,
        target_validity,
        center_ids=center_ids,
        scored_frames=SCORED_FRAMES,
    )
    scores = {metric: float(report[metric]) for metric in PRIMARY_METRICS}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("post-hoc trajectory score is not finite")
    return scores


def _posthoc_variant(
    persistence: np.ndarray,
    measurement: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    triangulation_inlier_view_count: np.ndarray,
    center_ids: np.ndarray,
    update_frames: tuple[int, ...],
    gate_config: PairwiseCorrespondenceGateConfig,
    belief_config: RecursiveRbfBeliefConfig,
    *,
    minimum_views: int | None,
    empirical_bayes_shrinkage: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    validity = np.asarray(measurement_validity, dtype=bool).copy()
    adjusted = np.asarray(measurement).copy()
    diagnostics = []
    sigma_squared = float(belief_config.observation_std_m**2)

    for update_index, update in enumerate(update_frames):
        if minimum_views is not None:
            view_row = (
                update_index
                if triangulation_inlier_view_count.shape[0] == len(update_frames)
                else update
            )
            view_columns = (
                np.arange(len(center_ids))
                if triangulation_inlier_view_count.shape[1] == len(center_ids)
                else center_ids
            )
            validity[update, center_ids] &= (
                triangulation_inlier_view_count[view_row, view_columns] >= minimum_views
            )
        supported = (
            measurement_visibility[update, center_ids]
            & validity[update, center_ids]
            & np.all(np.isfinite(measurement[update, center_ids]), axis=1)
            & np.all(np.isfinite(persistence[update, center_ids]), axis=1)
        )
        supported_ids = center_ids[supported]
        residual = (
            measurement[update, supported_ids] - persistence[update, supported_ids]
        )
        residual_energy = (
            None if not len(residual) else float(np.mean(np.square(residual)))
        )
        gain = 1.0
        if empirical_bayes_shrinkage:
            gain = (
                0.0
                if residual_energy is None or residual_energy <= 0.0
                else max(1.0 - sigma_squared / residual_energy, 0.0)
            )
            adjusted[update, supported_ids] = (
                persistence[update, supported_ids] + gain * residual
            )
        diagnostics.append(
            {
                "frame": int(update),
                "supported_center_count": int(len(supported_ids)),
                "residual_mean_coordinate_square_m2": residual_energy,
                "empirical_bayes_gain": float(gain),
                "minimum_inlier_views": minimum_views,
            }
        )

    prediction_report, trajectory = predict_persistence_pairwise_rbf_arrays(
        persistence,
        adjusted,
        measurement_visibility,
        validity,
        center_ids=center_ids,
        update_frames=update_frames,
        gate_config=gate_config,
        belief_config=belief_config,
    )
    return trajectory, diagnostics, prediction_report


def _case_result(
    row: Mapping[str, Any],
    *,
    prediction_root: Path,
    measurement_root: Path,
    outcome_root: Path,
    evaluation_root: Path,
) -> dict[str, Any]:
    case = str(row["case"])
    prediction_dir = prediction_root / case
    measurement_dir = measurement_root / case
    outcome_dir = outcome_root / case
    evaluation_path = evaluation_root / case / CASE_EVALUATION_FILENAME
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("result_sha256") != evaluation_sha256(evaluation):
        raise ValueError(f"case evaluation checksum changed: {case}")

    prediction_path = prediction_dir / VIRTUAL_SENSING_ARCHIVE_FILENAME
    prediction_report_path = prediction_dir / VIRTUAL_SENSING_REPORT_FILENAME
    measurement_path = measurement_dir / MEASUREMENT_FILENAME
    target_path = outcome_dir / TARGET_ARCHIVE_FILENAME
    expected_hashes = evaluation["inputs_sha256"]
    for name, path in (
        ("prediction_archive", prediction_path),
        ("prediction_report", prediction_report_path),
        ("measurement_archive", measurement_path),
        ("target_archive", target_path),
    ):
        if _sha256(path) != expected_hashes[name]:
            raise ValueError(f"{name} changed after evaluation: {case}")

    prediction_report = json.loads(prediction_report_path.read_text(encoding="utf-8"))
    primary_config = prediction_report["method"]["primary"]
    gate_config = PairwiseCorrespondenceGateConfig(**primary_config["gate_config"])
    belief_config = RecursiveRbfBeliefConfig(**primary_config["belief_config"])
    with np.load(prediction_path, allow_pickle=False) as stored:
        sealed_primary = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
        center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
        update_frames = tuple(int(value) for value in stored["update_frames"])
    with np.load(measurement_path, allow_pickle=False) as stored:
        measurement = np.asarray(stored["measurement_m"]).copy()
        measurement_visibility = np.asarray(
            stored["measurement_visibility"], dtype=bool
        )
        measurement_validity = np.asarray(stored["measurement_validity"], dtype=bool)
        inlier_views = np.asarray(
            stored["triangulation_inlier_view_count"], dtype=np.int64
        )
    with np.load(target_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_m"]).copy()
        target_visibility = np.asarray(stored["target_visibility"], dtype=bool)
        target_validity = np.asarray(stored["target_validity"], dtype=bool)

    regenerated, _, _ = _posthoc_variant(
        persistence,
        measurement,
        measurement_visibility,
        measurement_validity,
        inlier_views,
        center_ids,
        update_frames,
        gate_config,
        belief_config,
        minimum_views=None,
        empirical_bayes_shrinkage=False,
    )
    if not np.array_equal(regenerated, sealed_primary):
        raise AssertionError(f"sealed primary is not bit-exactly reproducible: {case}")

    trajectories = {SEALED_PRIMARY: sealed_primary, PERSISTENCE: persistence}
    update_diagnostics: dict[str, Any] = {}
    predictor_reports: dict[str, Any] = {}
    for arm, minimum_views, shrink in (
        (VIEW3, 3, False),
        (SNR5, None, True),
        (VIEW3_SNR5, 3, True),
    ):
        trajectory, diagnostics, report = _posthoc_variant(
            persistence,
            measurement,
            measurement_visibility,
            measurement_validity,
            inlier_views,
            center_ids,
            update_frames,
            gate_config,
            belief_config,
            minimum_views=minimum_views,
            empirical_bayes_shrinkage=shrink,
        )
        trajectories[arm] = trajectory
        update_diagnostics[arm] = diagnostics
        predictor_reports[arm] = {
            "accepted_update_count": int(
                sum(update["accepted"] for update in report["updates"])
            ),
            "updates": report["updates"],
        }

    scores = {
        arm: _score(
            trajectories[arm],
            target,
            target_visibility,
            target_validity,
            center_ids,
        )
        for arm in ARMS
    }
    return {
        "case": case,
        "object_id": row["object_id"],
        "episode_id": row["episode_id"],
        "stratum": row["stratum"],
        "scores": scores,
        "update_diagnostics": update_diagnostics,
        "predictor_reports": predictor_reports,
        "sealed_primary_bit_exactly_regenerated": True,
        "input_sha256": {
            "prediction_archive": _sha256(prediction_path),
            "measurement_archive": _sha256(measurement_path),
            "target_archive": _sha256(target_path),
            "evaluation": _sha256(evaluation_path),
        },
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("post-hoc diagnostic has no evaluated cases")
    by_object: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_object.setdefault(str(case["object_id"]), []).append(case)
    object_rows = []
    for object_id, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_id": object_id,
                "stratum": rows[0]["stratum"],
                "episode_count": len(rows),
                "scores": {
                    arm: {
                        metric: float(
                            np.mean([row["scores"][arm][metric] for row in rows])
                        )
                        for metric in PRIMARY_METRICS
                    }
                    for arm in ARMS
                },
            }
        )
    object_balanced = {
        arm: {
            metric: float(np.mean([row["scores"][arm][metric] for row in object_rows]))
            for metric in PRIMARY_METRICS
        }
        for arm in ARMS
    }
    versus_persistence = {
        arm: {
            metric: {
                "relative_change": float(
                    object_balanced[arm][metric] / object_balanced[PERSISTENCE][metric]
                    - 1.0
                ),
                "improved_object_count": int(
                    sum(
                        row["scores"][arm][metric] < row["scores"][PERSISTENCE][metric]
                        for row in object_rows
                    )
                ),
            }
            for metric in PRIMARY_METRICS
        }
        for arm in ARMS
        if arm != PERSISTENCE
    }
    return {
        "episode_count": len(cases),
        "object_count": len(object_rows),
        "object_results": object_rows,
        "object_balanced_scores": object_balanced,
        "versus_persistence": versus_persistence,
    }


def main() -> int:
    args = _parse_args()
    cohort = json.loads(args.cohort_seal.read_text(encoding="utf-8"))
    prediction_root = args.prediction_root.resolve()
    measurement_root = args.measurement_root.resolve()
    outcome_root = args.outcome_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    sealed_rows = [row for row in cohort["cases"] if row["status"] != "quality-failure"]
    if len(sealed_rows) != 23 or any(
        row["status"] != "prediction-sealed" for row in sealed_rows
    ):
        raise ValueError("expected exactly 23 prediction-sealed cohort cases")
    cases = [
        _case_result(
            row,
            prediction_root=prediction_root,
            measurement_root=measurement_root,
            outcome_root=outcome_root,
            evaluation_root=evaluation_root,
        )
        for row in sealed_rows
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectiveVirtualSensingPosthocMechanismDiagnostic",
        "protocol_id": cohort["protocol_id"],
        "protocol_config_sha256": cohort["protocol_config_sha256"],
        "prediction_cohort_result_sha256": cohort["result_sha256"],
        "status": "post-open exploratory mechanism analysis",
        "claim_boundary": (
            "Outcomes were open before this analysis. These arms cannot replace "
            "the sealed primary, select a method, or support confirmation."
        ),
        "diagnostic_definitions": {
            VIEW3: "require at least three triangulation inlier views per centre",
            SNR5: (
                "positive-part empirical-Bayes residual shrinkage with gain "
                "max(1 - 0.005^2 / mean(residual^2), 0)"
            ),
            VIEW3_SNR5: "apply both post-open diagnostics",
            "undefined_camera_gate_fractions": (
                "encoded as null when an update has zero supported centres"
            ),
        },
        "cases": cases,
        **_aggregate(cases),
    }
    payload = _json_safe(payload)
    payload["result_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
