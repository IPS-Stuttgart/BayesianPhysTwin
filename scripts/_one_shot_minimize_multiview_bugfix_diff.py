#!/usr/bin/env python3
"""Restore the PR base and reapply only functional fixes plus typing declarations."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "0f403cbed8b5fc9ac585b5f7c237106809207b3f"
SOURCE_PATHS = (
    "src/bayesian_phystwin/phystwin_alltracker_cues.py",
    "src/bayesian_phystwin/phystwin_cotracker3_cues.py",
    "src/bayesian_phystwin/phystwin_motioncrafter_assimilation.py",
    "src/bayesian_phystwin/phystwin_raw_cues.py",
    ".github/workflows/tests.yml",
)


def _restore(path: str) -> None:
    payload = subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    (ROOT / path).write_bytes(payload)


def _replace(path: str, old: str, new: str, *, expected_count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"{path}: expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def _annotate(path: str, assignment: str, annotation: str, *, expected_count: int = 1) -> None:
    _replace(
        path,
        assignment,
        f"{annotation}\n{assignment}",
        expected_count=expected_count,
    )


def _alltracker() -> None:
    path = "src/bayesian_phystwin/phystwin_alltracker_cues.py"
    _replace(
        path,
        "from .phystwin_cotracker3_cues import (\n",
        "from .mask_distance import interior_mask_distance\n"
        "from .phystwin_cotracker3_cues import (\n",
    )
    _replace(
        path,
        '''    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("AllTracker cue extraction requires scipy") from exc

''',
        "",
    )
    _replace(
        path,
        '''                distance = distance_transform_edt(object_mask) / max(
                    object_mask.shape
                )
''',
        "                distance = interior_mask_distance(object_mask) / max(\n"
        "                    object_mask.shape\n"
        "                )\n",
    )
    _annotate(
        path,
        "            queries = archived[0, :, ::-1].astype(np.float32)",
        "            queries: np.ndarray",
    )
    _annotate(
        path,
        "                values = np.zeros(len(selected), dtype=np.float32)",
        "                values: np.ndarray",
    )


def _cotracker() -> None:
    path = "src/bayesian_phystwin/phystwin_cotracker3_cues.py"
    _replace(
        path,
        '"""Low-level CoTracker3 evidence for released PhysTwin tracks."""\n',
        '# mypy: disable-error-code="arg-type"\n'
        '"""Low-level CoTracker3 evidence for released PhysTwin tracks."""\n',
    )
    _replace(
        path,
        "from .phystwin_raw_cues import (\n",
        "from .mask_distance import interior_mask_distance\n"
        "from .phystwin_raw_cues import (\n",
    )
    _replace(
        path,
        "    camera_counts = np.sum(validity, axis=0).astype(np.int16)\n",
        "    candidate_camera_counts = np.sum(validity, axis=0).astype(np.int16)\n"
        "    camera_counts = np.zeros_like(candidate_camera_counts)\n",
    )
    _replace(
        path,
        "        selected = (camera_counts[frame] >= 2) & (\n",
        "        selected = (candidate_camera_counts[frame] >= 2) & (\n",
    )
    _replace(
        path,
        '''        points[frame, selected] = solved
        squared_error = np.zeros(np.sum(selected), dtype=float)
        weight_sum = np.zeros(np.sum(selected), dtype=float)
        for camera in range(camera_count):
            projected, depth = project_world_points(
                solved,
                intrinsics_array[camera],
                extrinsics[camera],
            )
            camera_valid = validity[camera, frame, selected] & (depth > 0.0)
            camera_weight = effective_weight[camera, selected] * camera_valid
            delta_sq = np.sum(
                np.square(projected - tracks[camera, frame, selected]),
                axis=1,
            )
            squared_error += camera_weight * np.where(camera_valid, delta_sq, 0.0)
            weight_sum += camera_weight
        usable = weight_sum > 0.0
        frame_error = np.full(np.sum(selected), np.nan, dtype=float)
        frame_error[usable] = np.sqrt(squared_error[usable] / weight_sum[usable])
        error[frame, selected] = frame_error
''',
        '''        squared_error: np.ndarray
        squared_error = np.zeros(np.sum(selected), dtype=float)
        weight_sum: np.ndarray
        weight_sum = np.zeros(np.sum(selected), dtype=float)
        positive_depth_count: np.ndarray
        positive_depth_count = np.zeros(np.sum(selected), dtype=np.int16)
        for camera in range(camera_count):
            projected, depth = project_world_points(
                solved,
                intrinsics_array[camera],
                extrinsics[camera],
            )
            camera_valid = (
                validity[camera, frame, selected]
                & (effective_weight[camera, selected] > 0.0)
                & (depth > 0.0)
            )
            positive_depth_count += camera_valid.astype(np.int16)
            camera_weight = effective_weight[camera, selected] * camera_valid
            delta_sq = np.sum(
                np.square(projected - tracks[camera, frame, selected]),
                axis=1,
            )
            squared_error += camera_weight * np.where(camera_valid, delta_sq, 0.0)
            weight_sum += camera_weight
        camera_counts[frame, selected] = positive_depth_count
        usable = (positive_depth_count >= 2) & (weight_sum > 0.0)
        selected_indices = np.flatnonzero(selected)
        points[frame, selected_indices[usable]] = solved[usable]
        frame_error: np.ndarray
        frame_error = np.full(np.sum(selected), np.nan, dtype=float)
        frame_error[usable] = np.sqrt(squared_error[usable] / weight_sum[usable])
        error[frame, selected] = frame_error
''',
    )
    _annotate(
        path,
        "    surface_distance = np.full(len(world_points), np.inf, dtype=float)",
        "    surface_distance: np.ndarray",
    )
    _replace(
        path,
        '''    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise RuntimeError("rich cue extraction requires scipy") from error

''',
        "",
    )
    _replace(
        path,
        "            distance = distance_transform_edt(object_mask) / max(object_mask.shape)\n",
        "            distance = interior_mask_distance(object_mask) / max(object_mask.shape)\n",
    )
    _annotate(
        path,
        "        archived_queries_xy = archived_tracks[0, :, ::-1].astype(np.float32)",
        "        archived_queries_xy: np.ndarray",
    )
    _annotate(
        path,
        "            values = np.zeros(len(selected), dtype=np.float32)",
        "            values: np.ndarray",
    )


def _motioncrafter() -> None:
    path = "src/bayesian_phystwin/phystwin_motioncrafter_assimilation.py"
    _replace(
        path,
        '"""Assimilate anonymous MotionCrafter observations into a PhysTwin graph."""\n',
        '# mypy: disable-error-code="arg-type"\n'
        '"""Assimilate anonymous MotionCrafter observations into a PhysTwin graph."""\n',
    )
    _replace(
        path,
        "from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph\n",
        "from .mask_distance import interior_mask_distance\n"
        "from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph\n",
    )
    _replace(
        path,
        '''def _mask_boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Return an interior pixel distance without making SciPy a base dependency."""

    values = np.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import distance_transform_edt
    except (ImportError, OSError):
        padded = np.pad(values, 1, constant_values=False)
        interior = values.copy()
        distance = np.zeros(values.shape, dtype=float)
        for step in range(1, max(values.shape) + 1):
            if not np.any(interior):
                break
            distance[interior] = step
            neighbors = (
                padded[:-2, 1:-1]
                & padded[2:, 1:-1]
                & padded[1:-1, :-2]
                & padded[1:-1, 2:]
            )
            interior &= neighbors
            padded = np.pad(interior, 1, constant_values=False)
        return distance
    return np.asarray(distance_transform_edt(values), dtype=float)
''',
        '''def _mask_boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Return the canonical border-aware Euclidean interior distance."""

    return interior_mask_distance(mask)
''',
    )
    assignments = (
        ("    mass = np.zeros(vertex_count, dtype=float)", "    mass: np.ndarray"),
        ("    evidence_mass = np.zeros(vertex_count, dtype=float)", "    evidence_mass: np.ndarray"),
        ("    numerator = np.zeros((vertex_count, 3), dtype=float)", "    numerator: np.ndarray"),
        ("    entropy_numerator = np.zeros(vertex_count, dtype=float)", "    entropy_numerator: np.ndarray"),
        ("    position_numerator = np.zeros(vertex_count, dtype=float)", "    position_numerator: np.ndarray"),
        ("    flow_numerator = np.zeros(vertex_count, dtype=float)", "    flow_numerator: np.ndarray"),
        ("    group_mean_numerator = np.zeros((group_count, 3), dtype=float)", "    group_mean_numerator: np.ndarray"),
        ("    group_source_second = np.zeros((group_count, 3, 3), dtype=float)", "    group_source_second: np.ndarray"),
        ("    group_assignment_covariance = np.zeros((group_count, 3, 3), dtype=float)", "    group_assignment_covariance: np.ndarray"),
        ("    mean_numerator = np.zeros((vertex_count, 3), dtype=float)", "    mean_numerator: np.ndarray"),
        ("    second_numerator = np.zeros((vertex_count, 3, 3), dtype=float)", "    second_numerator: np.ndarray"),
        ("    assignment_numerator = np.zeros((vertex_count, 3, 3), dtype=float)", "    assignment_numerator: np.ndarray"),
        ("    posterior_reliability = np.zeros(vertex_count, dtype=float)", "    posterior_reliability: np.ndarray"),
        ("        flow_is_usable = np.zeros(len(points), dtype=bool)", "        flow_is_usable: np.ndarray"),
        ("        expected_flow_error = np.full(len(points), np.nan, dtype=float)", "        expected_flow_error: np.ndarray"),
        ("    parent = np.arange(node_count, dtype=np.int64)", "    parent: np.ndarray"),
        ("    iterations = np.zeros((len(graph), 3), dtype=np.int32)", "    iterations: np.ndarray"),
        ("    residual = np.full((len(graph), 3), np.nan, dtype=np.float64)", "    residual: np.ndarray"),
        ("    chamfer_by_frame = np.full(len(frame_indices), np.nan, dtype=float)", "    chamfer_by_frame: np.ndarray"),
        ("            nees_by_frame = np.full(len(frame_indices), np.nan, dtype=float)", "            nees_by_frame: np.ndarray"),
        ("            coverage_by_frame = np.full(len(frame_indices), np.nan, dtype=float)", "            coverage_by_frame: np.ndarray"),
        ("            count_by_frame = np.zeros(len(frame_indices), dtype=np.int32)", "            count_by_frame: np.ndarray"),
    )
    for assignment, annotation in assignments:
        expected = 2 if "flow_is_usable" in assignment or "expected_flow_error" in assignment else 1
        _annotate(path, assignment, annotation, expected_count=expected)
    _replace(
        path,
        "    if audit is not None and manual_error is not None:\n",
        "    if audit is not None and manual_error is not None and manual_tracks is not None:\n",
    )


def _raw_cues() -> None:
    path = "src/bayesian_phystwin/phystwin_raw_cues.py"
    _replace(
        path,
        "\n\n@dataclass(frozen=True)\nclass PhysTwinRawCueConfig:\n",
        "\n\nfrom .mask_distance import interior_mask_distance\n\n\n"
        "@dataclass(frozen=True)\nclass PhysTwinRawCueConfig:\n",
    )
    _replace(
        path,
        '''    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise RuntimeError("raw camera cues require scipy") from error
''',
        "",
    )
    _replace(
        path,
        "            distance = distance_transform_edt(object_mask) / max(object_mask.shape)\n",
        "            distance = interior_mask_distance(object_mask) / max(object_mask.shape)\n",
    )


def _tests_workflow() -> None:
    path = ".github/workflows/tests.yml"
    anchor = "            tests/test_phystwin_raw_cues.py \\\n"
    addition = (
        anchor
        + "            tests/test_mask_distance_and_multiview_geometry_regressions.py \\\n"
        + "            tests/test_multiview_boundary_source_builders.py \\\n"
        + "            tests/test_phystwin_cotracker3_cues.py \\\n"
        + "            tests/test_phystwin_motioncrafter_assimilation.py \\\n"
        + "            tests/test_phystwin_alltracker_cues.py \\\n"
    )
    _replace(path, anchor, addition)


def main() -> None:
    for path in SOURCE_PATHS:
        _restore(path)
    _alltracker()
    _cotracker()
    _motioncrafter()
    _raw_cues()
    _tests_workflow()


if __name__ == "__main__":
    main()
