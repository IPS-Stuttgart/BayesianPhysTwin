#!/usr/bin/env python3
"""Apply the frozen source-v4 guard to opened selective-camera artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_selective_bias_guard_diagnostic import (
    apply_frozen_selective_bias_guard_arrays,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_ARCHIVE_FILENAME,
    VIRTUAL_SENSING_REPORT_FILENAME,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_evaluation import (
    _canonical_sha256 as evaluation_sha256,
)


PRIMARY_CAMERA_ARM = "persistence_full_blend_rbf_pairwise_clique"
PERSISTENCE_ARM = "persistence"
METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--backbone-root", type=Path, action="append", required=True
    )
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


def _find_backbone(case: str, roots: list[Path]) -> Path:
    candidates = [root / case / "backbone_prediction.npz" for root in roots]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected one backbone for {case}, found {len(existing)}: {existing}"
        )
    return existing[0]


def _metric_pair(score: Mapping[str, Any]) -> dict[str, float]:
    result = {metric: float(score[metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("evaluation score is not finite")
    return result


def _case_result(
    evaluation_path: Path,
    *,
    source_lock: Mapping[str, Any],
    source_lock_path: Path,
    measurement_root: Path,
    prediction_root: Path,
    backbone_roots: list[Path],
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("result_sha256") != evaluation_sha256(evaluation):
        raise ValueError(f"evaluation checksum changed: {evaluation_path}")
    case = str(evaluation["case"])
    measurement_path = measurement_root / case / "measurement.npz"
    prediction_path = (
        prediction_root / case / VIRTUAL_SENSING_ARCHIVE_FILENAME
    )
    prediction_report_path = (
        prediction_root / case / VIRTUAL_SENSING_REPORT_FILENAME
    )
    backbone_path = _find_backbone(case, backbone_roots)
    expected = evaluation["inputs_sha256"]
    for name, path in (
        ("measurement_archive", measurement_path),
        ("prediction_archive", prediction_path),
        ("prediction_report", prediction_report_path),
    ):
        if _sha256(path) != expected[name]:
            raise ValueError(f"{name} changed after the sealed evaluation: {case}")
    prediction_report = json.loads(
        prediction_report_path.read_text(encoding="utf-8")
    )
    if _sha256(backbone_path) != prediction_report["inputs_sha256"][
        "backbone_archive"
    ]:
        raise ValueError(f"sealed backbone checksum changed: {case}")

    with np.load(measurement_path, allow_pickle=False) as stored:
        measurement = {name: np.asarray(stored[name]) for name in stored.files}
    with np.load(prediction_path, allow_pickle=False) as stored:
        persistence = np.asarray(stored["persistence_m"])
        sealed_center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
        sealed_updates = np.asarray(stored["update_frames"], dtype=np.int64)
        selected_cameras = np.asarray(stored["selected_cameras"])
    with np.load(backbone_path, allow_pickle=False) as stored:
        driven = np.asarray(stored["prediction_m"])
        backbone_persistence = np.asarray(stored["persistence_m"])
        frame_zero = np.asarray(stored["frame_zero_points_m"])

    if not np.array_equal(persistence, backbone_persistence):
        raise ValueError(f"camera and backbone persistence differ: {case}")
    if not np.array_equal(sealed_center_ids, measurement["center_ids"]):
        raise ValueError(f"camera and measurement center IDs differ: {case}")
    if not np.array_equal(sealed_updates, measurement["update_frames"]):
        raise ValueError(f"camera and measurement update frames differ: {case}")
    report, selected = apply_frozen_selective_bias_guard_arrays(
        persistence,
        driven,
        frame_zero,
        measurement["measurement_m"],
        measurement["measurement_visibility"],
        measurement["measurement_validity"],
        center_ids=sealed_center_ids,
        update_frames=sealed_updates,
        selected_camera_count=len(selected_cameras),
        triangulation_inlier_view_count=measurement[
            "triangulation_inlier_view_count"
        ],
        triangulation_median_reprojection_px=measurement[
            "triangulation_median_reprojection_px"
        ],
        source_lock=source_lock,
    )
    if report["accepted_count"]:
        raise ValueError(
            "opened-outcome score inheritance is valid only for exact fallback"
        )
    if not np.array_equal(selected, persistence):
        raise AssertionError("guarded trajectory is not the sealed persistence")

    return {
        "case": case,
        "object_id": str(evaluation["object_id"]),
        "episode_id": int(evaluation["episode_id"]),
        "stratum": str(evaluation["stratum"]),
        "target_free_guard": report,
        "opened_outcome_join": {
            "guarded_score": _metric_pair(evaluation["scores"][PERSISTENCE_ARM]),
            "persistence_score": _metric_pair(
                evaluation["scores"][PERSISTENCE_ARM]
            ),
            "sealed_camera_score": _metric_pair(
                evaluation["scores"][PRIMARY_CAMERA_ARM]
            ),
            "score_inheritance_justification": (
                "guarded trajectory proved bit-exact sealed persistence before "
                "the opened evaluation score was joined"
            ),
        },
        "input_sha256": {
            "source_lock": _sha256(source_lock_path),
            "measurement_archive": _sha256(measurement_path),
            "prediction_archive": _sha256(prediction_path),
            "prediction_report": _sha256(prediction_report_path),
            "backbone_archive": _sha256(backbone_path),
            "opened_evaluation": _sha256(evaluation_path),
        },
    }


def _object_balanced(
    cases: list[dict[str, Any]], arm: str
) -> dict[str, float]:
    by_object: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_object.setdefault(case["object_id"], []).append(case)
    return {
        metric: float(
            np.mean(
                [
                    np.mean(
                        [
                            member["opened_outcome_join"][arm][metric]
                            for member in members
                        ]
                    )
                    for members in by_object.values()
                ]
            )
        )
        for metric in METRICS
    }


def main() -> None:
    args = _parse_args()
    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    evaluation_paths = sorted(args.evaluation_root.glob("*.json"))
    if not evaluation_paths:
        raise ValueError("no opened selective-camera evaluations found")
    cases = [
        _case_result(
            path,
            source_lock=source_lock,
            source_lock_path=args.source_lock,
            measurement_root=args.measurement_root,
            prediction_root=args.prediction_root,
            backbone_roots=args.backbone_root,
        )
        for path in evaluation_paths
    ]
    guarded = _object_balanced(cases, "guarded_score")
    persistence = _object_balanced(cases, "persistence_score")
    camera = _object_balanced(cases, "sealed_camera_score")
    payload = {
        "artifact_kind": "Deform360SelectiveBiasAwareGuardPostOpenResult",
        "schema_version": 1,
        "case_count": len(cases),
        "object_count": len({case["object_id"] for case in cases}),
        "candidate_available_interval_count": int(
            sum(
                case["target_free_guard"]["candidate_available_count"]
                for case in cases
            )
        ),
        "accepted_interval_count": int(
            sum(
                case["target_free_guard"]["accepted_count"] for case in cases
            )
        ),
        "exact_fallback_interval_count": int(
            sum(
                case["target_free_guard"]["exact_fallback_interval_count"]
                for case in cases
            )
        ),
        "driven_backbone_bit_exact_persistence_case_count": int(
            sum(
                case["target_free_guard"][
                    "driven_backbone_bit_exact_persistence"
                ]
                for case in cases
            )
        ),
        "object_balanced_scores": {
            "guarded_bias_aware_source_v4": guarded,
            "persistence": persistence,
            "sealed_camera_update": camera,
        },
        "guarded_vs_persistence_percent": {
            metric: float(100.0 * (guarded[metric] / persistence[metric] - 1.0))
            for metric in METRICS
        },
        "sealed_camera_vs_persistence_percent": {
            metric: float(100.0 * (camera[metric] / persistence[metric] - 1.0))
            for metric in METRICS
        },
        "cases": cases,
        "information_boundary": {
            "candidate_built_before_opened_score_join": True,
            "opened_outcome_used_to_change_guard": False,
            "source_v4_lock_changed": False,
            "all_selected_updates_proved_exact_fallback": True,
        },
        "claim_boundary": (
            "post-open mechanism evidence on an exhausted cohort; establishes "
            "fallback behavior only, not prospective confirmation or accuracy "
            "improvement"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in (
        "case_count",
        "object_count",
        "candidate_available_interval_count",
        "accepted_interval_count",
        "exact_fallback_interval_count",
        "object_balanced_scores",
        "sealed_camera_vs_persistence_percent",
        "result_sha256",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
