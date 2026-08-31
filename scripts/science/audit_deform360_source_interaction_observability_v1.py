#!/usr/bin/env python3
"""Audit source-only robot/tactile excitation without opening a target episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rotation_log(relative: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        values, vectors = np.linalg.eigh((relative + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return angle * axis
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return angle * axis


def _best_lag_correlation(first: np.ndarray, second: np.ndarray, radius: int) -> dict[str, Any]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    best = {"lag_frames": 0, "correlation": 0.0, "absolute_correlation": 0.0}
    for lag in range(-radius, radius + 1):
        if lag < 0:
            left, right = first[-lag:], second[:lag]
        elif lag > 0:
            left, right = first[:-lag], second[lag:]
        else:
            left, right = first, second
        if len(left) < 8 or np.std(left) <= 0 or np.std(right) <= 0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(correlation) and abs(correlation) > best["absolute_correlation"]:
            best = {
                "lag_frames": lag,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            }
    return best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-episode", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workflow-run-id", required=True, type=int)
    parser.add_argument("--workflow-run-attempt", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.source_episode_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    robot_path = root / "robot" / "robot.npz"
    with np.load(robot_path, allow_pickle=False) as robot:
        transforms = np.asarray(robot["T_worlds"], dtype=np.float64)
        openings = np.asarray(robot["openings"], dtype=np.float64)
        bimanual = bool(np.asarray(robot["bimanual"]).item())
        format_version = int(np.asarray(robot["format_version"]).item())
    if transforms.ndim == 3:
        transforms = transforms[:, None]
        openings = openings[:, None]
    if transforms.ndim != 4 or transforms.shape[-2:] != (4, 4):
        raise ValueError(f"unexpected robot transform shape {transforms.shape}")
    frame_count, gripper_count = transforms.shape[:2]
    if openings.shape != (frame_count, gripper_count):
        raise ValueError("robot opening shape disagrees with transforms")

    sensor_names = (
        "brics-odroid_tactilel_left",
        "brics-odroid_tactilel_right",
        "brics-odroid_tactiler_left",
        "brics-odroid_tactiler_right",
    )
    tactile_arrays = []
    tactile_records = []
    for name in sensor_names:
        path = root / name / "synced_tactile.npy"
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if values.ndim != 3 or values.shape[1:] != (16, 32):
            raise ValueError(f"unexpected tactile shape for {name}: {values.shape}")
        if values.shape[0] != frame_count:
            raise ValueError(
                f"tactile/robot frame mismatch for {name}: {values.shape[0]} != {frame_count}"
            )
        energy = np.asarray(values, dtype=np.float64).sum(axis=(1, 2))
        active_taxels = np.count_nonzero(values > 0, axis=(1, 2))
        tactile_arrays.append(energy)
        tactile_records.append(
            {
                "sensor": name,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "active_frame_count": int(np.count_nonzero(active_taxels)),
                "maximum_active_taxels": int(np.max(active_taxels)),
                "mean_energy": float(np.mean(energy)),
                "maximum_energy": float(np.max(energy)),
            }
        )
    tactile_energy = np.sum(np.stack(tactile_arrays), axis=0)
    contact = tactile_energy > 0.0

    positions = transforms[:, :, :3, 3]
    rotations = transforms[:, :, :3, :3]
    translation_steps = np.diff(positions, axis=0)
    opening_steps = np.diff(openings, axis=0)
    rotation_steps = np.zeros((frame_count - 1, gripper_count, 3), dtype=np.float64)
    for frame in range(frame_count - 1):
        for gripper in range(gripper_count):
            relative = rotations[frame, gripper].T @ rotations[frame + 1, gripper]
            rotation_steps[frame, gripper] = _rotation_log(relative)

    feature_blocks = []
    for gripper in range(gripper_count):
        feature_blocks.extend(
            [
                translation_steps[:, gripper],
                rotation_steps[:, gripper],
                opening_steps[:, gripper, None],
            ]
        )
    features = np.concatenate(feature_blocks, axis=1)
    contact_steps = contact[:-1] | contact[1:]
    selected = features[contact_steps]
    if len(selected) >= 2:
        centered = selected - np.mean(selected, axis=0, keepdims=True)
        scales = np.std(centered, axis=0)
        active_columns = scales > 1e-10
        normalized = centered[:, active_columns] / scales[active_columns]
        singular = (
            np.linalg.svd(normalized, compute_uv=False)
            if normalized.size
            else np.zeros(0, dtype=np.float64)
        )
    else:
        active_columns = np.zeros(features.shape[1], dtype=bool)
        singular = np.zeros(0, dtype=np.float64)
    effective_rank = int(
        np.count_nonzero(singular > (singular[0] * 1e-3 if len(singular) else np.inf))
    )

    action_magnitude = np.linalg.norm(features, axis=1)
    tactile_change = np.abs(np.diff(tactile_energy))
    timing = _best_lag_correlation(action_magnitude, tactile_change, radius=10)
    path_lengths = np.sum(np.linalg.norm(translation_steps, axis=2), axis=0)
    rotation_lengths = np.sum(np.linalg.norm(rotation_steps, axis=2), axis=0)
    opening_ranges = np.ptp(openings, axis=0)
    contact_indices = np.flatnonzero(contact)
    contact_span = (
        int(contact_indices[-1] - contact_indices[0] + 1) if len(contact_indices) else 0
    )

    checks = {
        "at_least_20_contact_frames": int(np.count_nonzero(contact)) >= 20,
        "at_least_10_contact_steps": int(np.count_nonzero(contact_steps)) >= 10,
        "at_least_10mm_total_translation": float(np.sum(path_lengths)) >= 0.010,
        "at_least_two_excitation_dimensions": effective_rank >= 2,
        "nontrivial_actuation_change": bool(
            np.max(opening_ranges) >= 0.002
            or np.sum(rotation_lengths) >= 0.05
            or np.sum(path_lengths) >= 0.020
        ),
    }
    qualified = all(checks.values())
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-interaction-observability-v1",
        "repository": args.repository,
        "revision": args.revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "required_runner_label": "gpuserver4090",
        "source_object": args.source_object,
        "source_episode": args.source_episode,
        "source_episode_root": str(root),
        "robot": {
            "format_version": format_version,
            "bimanual": bimanual,
            "frame_count": frame_count,
            "gripper_count": gripper_count,
            "path_length_m_by_gripper": path_lengths.astype(float).tolist(),
            "rotation_path_rad_by_gripper": rotation_lengths.astype(float).tolist(),
            "opening_range_m_by_gripper": opening_ranges.astype(float).tolist(),
            "maximum_translation_step_m": float(
                np.max(np.linalg.norm(translation_steps, axis=2))
            ),
            "maximum_rotation_step_rad": float(
                np.max(np.linalg.norm(rotation_steps, axis=2))
            ),
            "maximum_opening_step_m": float(np.max(np.abs(opening_steps))),
        },
        "tactile": {
            "sensors": tactile_records,
            "contact_frame_count": int(np.count_nonzero(contact)),
            "contact_fraction": float(np.mean(contact)),
            "contact_span_frames": contact_span,
            "maximum_total_energy": float(np.max(tactile_energy)),
            "mean_total_energy": float(np.mean(tactile_energy)),
        },
        "joint_source_excitation": {
            "feature_dimension": int(features.shape[1]),
            "active_feature_dimension": int(np.count_nonzero(active_columns)),
            "contact_step_count": int(np.count_nonzero(contact_steps)),
            "singular_values": singular[: min(16, len(singular))].astype(float).tolist(),
            "effective_rank_relative_1e_minus_3": effective_rank,
            "motion_tactile_change_best_lag": timing,
        },
        "qualification_checks": checks,
        "source_interaction_observability_qualified": qualified,
        "decision": (
            "source-interaction-observability-qualified"
            if qualified
            else "source-interaction-observability-not-qualified"
        ),
        "information_boundary": {
            "source_robot_payload_opened": True,
            "source_tactile_payloads_opened": True,
            "source_camera_pixels_opened": False,
            "target_directory_contents_listed": False,
            "target_numeric_payload_opened": False,
            "target_scoring_performed": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "This is a source-only excitation and synchronization diagnostic. It does "
            "not establish identifiable physical parameters, geometry quality, target "
            "transport, Prob4D calibration, BayesianPhysTwin benefit, Causal4D value, "
            "safety, fresh confirmation, or a paper claim."
        ),
    }
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _write_json(output / "result.json", result)
    (output / "report.md").write_text(
        "# Deform360 source interaction observability v1\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Contact frames: `{result['tactile']['contact_frame_count']}` / `{frame_count}`\n\n"
        f"Excitation rank: `{effective_rank}` / `{features.shape[1]}`\n\n"
        f"Total translation: `{float(np.sum(path_lengths)):.6f} m`\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
