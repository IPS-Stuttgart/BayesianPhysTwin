"""Capture frozen PokeFlex target, checkpoint, and guarded mesh vertices."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ERROR_REPRODUCTION_ATOL_MM = 5e-4


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(_REPOSITORY_ROOT / "scripts" / "remote"))

import run_pokeflex_checkpoint_registration_independent_depth as runner  # noqa: E402
from bayesian_phystwin.pokeflex_same_object_reporting import (  # noqa: E402
    load_json_object,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _candidate_name,
    run_smoke,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def take_decisions(
    result: dict[str, Any], take_id: str
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return one take's decisions in target-frame order and keyed by frame."""

    current = [
        value for value in result["decisions"] if str(value["take_id"]) == take_id
    ]
    current.sort(key=lambda value: int(value["target_frame"]))
    _require(bool(current), f"take is absent from prospective result: {take_id}")
    return current, {int(value["target_frame"]): value for value in current}


def choose_take(result: dict[str, Any], requested: str | None) -> str:
    """Choose an explicit take or the strongest prospective visual exemplar."""

    takes = result["takes"]
    available = {str(value["take_id"]): value for value in takes}
    if requested is not None:
        _require(requested in available, f"unknown prospective take: {requested}")
        return requested
    return str(max(takes, key=lambda value: value["relative_improvement"])["take_id"])


def capture_frozen_predictions(
    *,
    take_root: Path,
    prospective_result: dict[str, Any],
    take_id: str,
    independent_depth_protocol_path: Path,
    registration_protocol_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """Rerun the frozen evaluator while capturing only paper-render meshes."""

    protocol = load_json_object(independent_depth_protocol_path)
    method = protocol["method_lock"]
    fields = tuple(map(str, method["correction_fields"]))
    scales = tuple(map(float, method["correction_scales"]))
    candidate_order = [
        _candidate_name(field, scale) for field in fields for scale in scales
    ]
    decisions, decision_by_frame = take_decisions(prospective_result, take_id)
    target_frames = [int(value["target_frame"]) for value in decisions]
    calls_per_target = 4 + len(candidate_order)
    captured: dict[int, dict[str, np.ndarray]] = {
        frame: {} for frame in target_frames
    }
    original_surface_sample = runner._surface_sample
    call_count = 0

    def capture_surface_sample(
        vertices_m: np.ndarray,
        faces: np.ndarray,
        sample_count: int,
        seed: int,
    ) -> np.ndarray:
        nonlocal call_count
        sampled = original_surface_sample(vertices_m, faces, sample_count, seed)
        group_index, local_index = divmod(call_count, calls_per_target)
        _require(group_index < len(target_frames), "unexpected surface-sample call")
        target_frame = target_frames[group_index]
        decision = decision_by_frame[target_frame]
        selected_arm = str(decision["selected_arm"])
        selected_index = (
            3
            if selected_arm == "released_checkpoint"
            else 4 + candidate_order.index(selected_arm)
        )
        current = captured[target_frame]
        if local_index == 0:
            current["target_vertices_m"] = np.asarray(
                vertices_m, dtype=np.float32
            ).copy()
        if local_index == 3:
            current["baseline_vertices_m"] = np.asarray(
                vertices_m, dtype=np.float32
            ).copy()
        if local_index == selected_index:
            current["guarded_vertices_m"] = np.asarray(
                vertices_m, dtype=np.float32
            ).copy()
        call_count += 1
        return sampled

    runner._surface_sample = capture_surface_sample
    try:
        reproduced = run_smoke(
            take_root,
            registration_protocol_path,
            upstream_checkout,
            checkpoint_root,
            correction_scales=scales,
            correction_fields=fields,
            residual_geometry="point_to_point",
            maximum_frame=None,
            include_frozen_action_guard=False,
            record_online_observation_regret=False,
            record_independent_anchor_regret=True,
            independent_depth_protocol_path=independent_depth_protocol_path,
            independent_anchor_maximum_template_distance_m=(
                float(method["static_template_support_radius_mm"]) / 1000.0
            ),
        )
    finally:
        runner._surface_sample = original_surface_sample

    _require(
        call_count == len(target_frames) * calls_per_target,
        "surface-sample call inventory changed",
    )
    reproduced_targets = {
        int(value["target_frame"]): value for value in reproduced["targets"]
    }
    checks = []
    for frame in target_frames:
        current = captured[frame]
        _require(
            {
                "target_vertices_m",
                "baseline_vertices_m",
                "guarded_vertices_m",
            }
            <= set(current),
            f"render inventory is incomplete for frame {frame}",
        )
        decision = decision_by_frame[frame]
        target = reproduced_targets[frame]
        baseline = float(target["released_checkpoint_CD_UL1_mm"])
        _require(
            np.isclose(
                baseline,
                float(decision["baseline_error_mm"]),
                atol=ERROR_REPRODUCTION_ATOL_MM,
                rtol=0.0,
            ),
            f"baseline reproduction changed at frame {frame}",
        )
        selected_arm = str(decision["selected_arm"])
        selected = (
            baseline
            if selected_arm == "released_checkpoint"
            else float(target[selected_arm])
        )
        _require(
            np.isclose(
                selected,
                float(decision["selected_error_mm"]),
                atol=ERROR_REPRODUCTION_ATOL_MM,
                rtol=0.0,
            ),
            f"guarded reproduction changed at frame {frame}",
        )
        checks.append(
            {
                "target_frame": frame,
                "selected_arm": selected_arm,
                "baseline_error_mm": baseline,
                "guarded_error_mm": selected,
            }
        )
    return captured, {
        "take_id": take_id,
        "target_frame_count": len(target_frames),
        "surface_sample_call_count": call_count,
        "candidate_order": candidate_order,
        "error_reproduction_atol_mm": ERROR_REPRODUCTION_ATOL_MM,
        "checks": checks,
    }
