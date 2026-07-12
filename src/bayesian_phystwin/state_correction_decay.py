"""Post-hoc decay diagnostics for frozen PhysTwin state-correction rollouts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STATE_CORRECTION_DECAY_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_state(values: np.ndarray, name: str) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.ndim != 3 or state.shape[2] != 3:
        raise ValueError(f"{name} must have shape (frame, node, 3)")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be finite")
    return state


def _offset_exponential_fit(
    rms_m: np.ndarray,
    *,
    frame_dt_s: float,
    tail_fraction: float,
    minimum_excess_fraction: float,
) -> dict[str, Any]:
    """Fit decay after the maximum toward a robust empirical tail floor."""

    peak_offset = int(np.argmax(rms_m))
    tail_count = max(3, int(math.ceil(tail_fraction * len(rms_m))))
    tail_count = min(tail_count, len(rms_m))
    tail_floor_m = float(np.median(rms_m[-tail_count:]))
    excess = rms_m[peak_offset:] - tail_floor_m
    peak_excess = float(max(excess[0], 0.0))
    threshold = max(1.0e-12, minimum_excess_fraction * peak_excess)
    selected = np.flatnonzero(excess > threshold)
    result: dict[str, Any] = {
        "model": "tail_floor_plus_exponential_after_peak",
        "peak_frame_offset": peak_offset,
        "tail_fraction": tail_fraction,
        "tail_frame_count": tail_count,
        "tail_floor_m": tail_floor_m,
        "minimum_excess_fraction": minimum_excess_fraction,
        "fit_frame_count": int(len(selected)),
        "time_constant_s": None,
        "half_life_s": None,
        "log_space_r_squared": None,
        "adequate_single_decay": False,
    }
    if len(selected) < 3:
        result["failure_reason"] = "fewer_than_three_frames_above_tail_floor"
        return result

    times_s = frame_dt_s * selected.astype(float)
    log_excess = np.log(excess[selected])
    slope, intercept = np.polyfit(times_s, log_excess, 1)
    if slope >= 0.0:
        result["failure_reason"] = "nondecaying_log_linear_slope"
        return result
    fitted = intercept + slope * times_s
    residual_sum = float(np.sum(np.square(log_excess - fitted)))
    total_sum = float(np.sum(np.square(log_excess - np.mean(log_excess))))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    time_constant_s = float(-1.0 / slope)
    result.update(
        {
            "time_constant_s": time_constant_s,
            "half_life_s": float(math.log(2.0) * time_constant_s),
            "log_space_r_squared": r_squared,
            "adequate_single_decay": bool(r_squared >= 0.80),
        }
    )
    return result


def analyze_state_correction_decay(
    baseline_state_m: np.ndarray,
    corrected_state_m: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    frame_dt_s: float,
    tail_fraction: float = 0.20,
    minimum_excess_fraction: float = 0.05,
) -> dict[str, Any]:
    """Measure how an injected prefix-state perturbation evolves in Warp.

    The interval includes the injection state at ``start_frame`` followed by the
    untouched continuation. The diagnostic reports both total displacement from
    the nominal rollout and retention along the originally injected direction.
    """

    baseline = _finite_state(baseline_state_m, "baseline_state_m")
    corrected = _finite_state(corrected_state_m, "corrected_state_m")
    if baseline.shape != corrected.shape:
        raise ValueError("baseline and corrected states must have matching shapes")
    if not 0 <= start_frame < stop_frame <= len(baseline):
        raise ValueError("state-correction interval is invalid")
    if stop_frame - start_frame < 3:
        raise ValueError("state-correction interval must contain at least three frames")
    if not np.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
        raise ValueError("frame_dt_s must be positive and finite")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if not 0.0 < minimum_excess_fraction < 1.0:
        raise ValueError("minimum_excess_fraction must lie in (0, 1)")

    difference = corrected[start_frame:stop_frame] - baseline[start_frame:stop_frame]
    rms_m = np.sqrt(np.mean(np.sum(np.square(difference), axis=2), axis=1))
    initial = difference[0].reshape(-1)
    initial_energy = float(initial @ initial)
    if initial_energy <= 0.0:
        raise ValueError("the injected state correction has zero position magnitude")
    flattened = difference.reshape(len(difference), -1)
    aligned_retention = flattened @ initial / initial_energy
    orthogonal = flattened - aligned_retention[:, None] * initial[None, :]
    orthogonal_rms_m = np.sqrt(
        np.mean(np.sum(np.square(orthogonal.reshape(difference.shape)), axis=2), axis=1)
    )
    peak_offset = int(np.argmax(rms_m))
    decay_fit = _offset_exponential_fit(
        rms_m,
        frame_dt_s=frame_dt_s,
        tail_fraction=tail_fraction,
        minimum_excess_fraction=minimum_excess_fraction,
    )
    return {
        "schema_version": STATE_CORRECTION_DECAY_SCHEMA_VERSION,
        "analysis_kind": "prefix_state_correction_decay",
        "interval": {
            "start_frame_inclusive": int(start_frame),
            "stop_frame_exclusive": int(stop_frame),
            "frame_count": int(stop_frame - start_frame),
            "frame_dt_s": float(frame_dt_s),
        },
        "summary": {
            "initial_rms_m": float(rms_m[0]),
            "peak_rms_m": float(rms_m[peak_offset]),
            "peak_frame_offset": peak_offset,
            "final_rms_m": float(rms_m[-1]),
            "final_to_initial_ratio": float(rms_m[-1] / rms_m[0]),
            "final_to_peak_ratio": float(rms_m[-1] / rms_m[peak_offset]),
            "final_aligned_retention": float(aligned_retention[-1]),
            "final_orthogonal_rms_m": float(orthogonal_rms_m[-1]),
        },
        "decay_fit": decay_fit,
        "per_frame": {
            "frame": list(range(start_frame, stop_frame)),
            "elapsed_s": (frame_dt_s * np.arange(len(rms_m))).tolist(),
            "rms_m": rms_m.tolist(),
            "aligned_retention": aligned_retention.tolist(),
            "orthogonal_rms_m": orthogonal_rms_m.tolist(),
        },
        "claim_boundary": (
            "Post-hoc trajectory diagnostic only; it does not select a correction "
            "model or identify the physical source of discrepancy."
        ),
    }


def audit_frozen_state_correction_decay(
    summary_json: str | Path,
    rollout_npz: str | Path,
    correction_json: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Audit one frozen localization case and write a checksummed JSON result."""

    summary_path = Path(summary_json)
    rollout_path = Path(rollout_npz)
    correction_path = Path(correction_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    heldout_start, heldout_stop = summary["comparison_contract"][
        "common_heldout_continuation"
    ]
    with np.load(rollout_path, allow_pickle=False) as archive:
        result = analyze_state_correction_decay(
            archive["mean_global__bpt_particle_baseline"],
            archive["mean_global__prefix_state_position_velocity"],
            start_frame=int(heldout_start) - 1,
            stop_frame=int(heldout_stop),
            frame_dt_s=float(correction["frame_dt_s"]),
        )
    result.update(
        {
            "case": summary["case"],
            "experiment": summary["experiment"],
            "source_checksums": {
                "summary_json": _file_sha256(summary_path),
                "rollout_npz": _file_sha256(rollout_path),
                "correction_json": _file_sha256(correction_path),
            },
        }
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
