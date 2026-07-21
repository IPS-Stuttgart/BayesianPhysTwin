#!/usr/bin/env python3
"""Audit a translation/contact Deform360 window rule on exhausted source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_selective_virtual_sensing_staging import (
    closure_confidence,
    end_effector_origins,
    select_action_only_window,
    select_translation_contact_window,
)
from causal4d_public.deform360_replication_tactile import (
    build_single_baseline_tactile_overlay,
    write_tactile_overlay_manifest,
)
from deform360.layout import list_tactile_names
from deform360.processing.control_points_stage import _episode_active_frames
from deform360.robot import load_robot_state
from deform360.tactile import process_tactile_episode


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("config_sha256", None)
    unsigned.pop("result_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = dict(payload)
    output["result_sha256"] = _canonical_sha256(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--selection-seal", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "source config must be a JSON object")
    expected = payload.get("config_sha256")
    _require(isinstance(expected, str), "source config hash is missing")
    _require(_canonical_sha256(payload) == expected, "source config hash changed")
    return payload["config"]


def _case_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stratum, objects in protocol["config"]["cohort"]["strata"].items():
        for item in objects:
            for episode_id in item["episode_ids"]:
                rows.append(
                    {
                        "object_id": item["object_id"],
                        "episode_id": int(episode_id),
                        "stratum": stratum,
                    }
                )
    return rows


def _score_at_start(
    actions: np.ndarray,
    openings: np.ndarray,
    contact: np.ndarray,
    start: int,
    *,
    first_update_frame: int,
    prediction_frame_count: int,
    staging_frame_count: int,
) -> dict[str, float]:
    origins = end_effector_origins(actions)[start : start + staging_frame_count]
    closed = closure_confidence(openings)[start : start + staging_frame_count]
    active = contact[start : start + staging_frame_count]
    step = np.linalg.norm(np.diff(origins, axis=0), axis=-1)
    closure_weight = np.minimum(closed[:-1], closed[1:])
    contact_weight = np.minimum(active[:-1], active[1:]).astype(np.float64)[:, None]
    future = slice(first_update_frame, prediction_frame_count - 1)
    weighted = step * closure_weight
    return {
        "future_translation_path_m": float(np.mean(np.sum(weighted[future], axis=0))),
        "contact_supported_future_translation_path_m": float(
            np.mean(np.sum(weighted[future] * contact_weight[future], axis=0))
        ),
        "window_contact_fraction": float(np.mean(active)),
        "future_contact_fraction": float(
            np.mean(active[first_update_frame:prediction_frame_count])
        ),
    }


def main() -> int:
    args = _parse_args()
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    config = _load_config(args.source_config)
    _require(
        source_protocol["config_sha256"]
        == config["source_cohort"]["protocol_config_sha256"],
        "source protocol hash differs from the diagnostic lock",
    )
    selector = config["selector"]
    raw_root = args.raw_root.resolve()
    aligned_root = args.aligned_root.resolve()
    work_root = args.work_root.resolve()
    cases = []
    for record in _case_rows(source_protocol):
        object_id = record["object_id"]
        episode_id = record["episode_id"]
        case = f"{object_id}-ep{episode_id:04d}"
        raw_object = raw_root / object_id
        aligned_episode = aligned_root / object_id / f"episode_{episode_id:04d}"
        overlay = work_root / "overlay" / case
        processed_root = work_root / "processed" / case
        baseline = build_single_baseline_tactile_overlay(
            raw_object, raw_object, overlay, episode_id
        )
        baseline_manifest = work_root / "baseline_manifests" / f"{case}.json"
        write_tactile_overlay_manifest(baseline_manifest, baseline)
        process_tactile_episode(
            overlay,
            aligned_episode,
            episode_id,
            output_dir=processed_root,
            overwrite=False,
        )
        processed_episode = processed_root / f"episode_{episode_id:04d}"
        streams = {
            sensor: np.load(
                processed_episode / sensor / "synced_tactile.npy",
                allow_pickle=False,
            )
            for sensor in list_tactile_names(processed_episode)
        }
        robot = load_robot_state(aligned_episode / "robot" / "robot.npz")
        contact = _episode_active_frames(streams, robot.num_frames)
        legacy = select_action_only_window(
            robot.actions,
            robot.openings,
            protocol_path=str(args.source_protocol),
        )
        selected = select_translation_contact_window(
            robot.actions,
            robot.openings,
            contact,
            staging_frame_count=int(selector["staging_frame_count"]),
            prediction_frame_count=int(selector["prediction_frame_count"]),
            first_update_frame=int(selector["first_update_frame"]),
            candidate_first_frame=int(selector["candidate_first_frame"]),
            candidate_stride_frames=int(selector["candidate_stride_frames"]),
        )
        legacy_start = int(legacy["selected_raw_frame_range_half_open"][0])
        legacy_corrected = _score_at_start(
            robot.actions,
            robot.openings,
            contact,
            legacy_start,
            first_update_frame=int(selector["first_update_frame"]),
            prediction_frame_count=int(selector["prediction_frame_count"]),
            staging_frame_count=int(selector["staging_frame_count"]),
        )
        score = float(selected["contact_supported_future_translation_path_m"])
        old_score = float(
            legacy_corrected["contact_supported_future_translation_path_m"]
        )
        cases.append(
            {
                **record,
                "case": case,
                "legacy_v1": {
                    "selected_raw_frame_range_half_open": legacy[
                        "selected_raw_frame_range_half_open"
                    ],
                    "invalid_signal": "mean of translation, rotation rows, and aperture",
                    **legacy_corrected,
                },
                "translation_contact_v2": selected,
                "selected_start_shift_frames": int(
                    selected["selected_raw_frame_range_half_open"][0] - legacy_start
                ),
                "supported_path_gain_ratio": (
                    score / old_score if old_score > 0.0 else None
                ),
                "action_schema_checks": {
                    "row_zero_matches_transform_translation": bool(
                        np.array_equal(
                            robot.actions[..., 0, :], robot.T_worlds[..., :3, 3]
                        )
                    ),
                    "action_row_count": int(robot.actions.shape[-2]),
                },
                "tactile": {
                    "sensor_count": len(streams),
                    "active_frame_count": int(np.count_nonzero(contact)),
                    "frame_count": len(contact),
                    "baseline_manifest": str(baseline_manifest),
                    "baseline_manifest_sha256": _file_sha256(baseline_manifest),
                },
            }
        )

    _require(
        len(cases) == int(config["source_cohort"]["episode_count"]),
        "source episode count changed",
    )
    _require(
        all(
            row["action_schema_checks"]["row_zero_matches_transform_translation"]
            for row in cases
        ),
        "Deform360 action schema check failed",
    )
    gains = [
        float(row["supported_path_gain_ratio"])
        for row in cases
        if row["supported_path_gain_ratio"] is not None
    ]
    selection_payload = {
        "artifact_kind": "Deform360DynamicWindowSourceSelectionSeal",
        "protocol_id": config["protocol_id"],
        "source_protocol_sha256": _file_sha256(args.source_protocol),
        "source_config_sha256": _file_sha256(args.source_config),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "changed_start_count": sum(
                row["selected_start_shift_frames"] != 0 for row in cases
            ),
            "contact_supported_case_count": sum(
                row["translation_contact_v2"]["has_contact_supported_future_motion"]
                for row in cases
            ),
            "zero_support_quality_failure_count": sum(
                not row["translation_contact_v2"]["has_contact_supported_future_motion"]
                for row in cases
            ),
            "median_supported_path_gain_ratio": float(np.median(gains)),
        },
        "information_boundary": config["information_boundary"],
    }
    _write_json(args.selection_seal, selection_payload)

    outcome_context = []
    if args.evaluation_root is not None:
        evaluation_root = args.evaluation_root.resolve()
        for row in cases:
            evaluation_path = evaluation_root / f"{row['case']}.json"
            if not evaluation_path.is_file():
                outcome_context.append(
                    {"case": row["case"], "status": "preexisting-quality-failure"}
                )
                continue
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            persistence = evaluation["scores"]["persistence"]
            outcome_context.append(
                {
                    "case": row["case"],
                    "status": "opened-v1-outcome",
                    "legacy_window_persistence_identity_rmse_m": persistence[
                        "post_update_hidden_identity_rmse_m"
                    ],
                    "legacy_window_persistence_chamfer_m": persistence[
                        "post_update_hidden_symmetric_chamfer_m"
                    ],
                }
            )
    result = {
        "artifact_kind": "Deform360DynamicWindowSourceAudit",
        "protocol_id": config["protocol_id"],
        "selection_seal": str(args.selection_seal.resolve()),
        "selection_seal_sha256": _file_sha256(args.selection_seal),
        "selection_result_sha256": json.loads(
            args.selection_seal.read_text(encoding="utf-8")
        )["result_sha256"],
        "outcome_context": outcome_context,
        "conclusion": (
            "The frozen v1 action selector averaged pose-encoding rows rather than "
            "using end-effector translation. V2 repairs the signal and moves source "
            "windows toward known future translation under tactile support. New-window "
            "object motion and prediction accuracy remain unevaluated."
        ),
        "information_order": {
            "selection_written_and_hashed_before_outcome_context": True,
            "outcomes_can_modify_selection": False,
            "fresh_objects_or_reserved_targets_read": False,
        },
    }
    _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
