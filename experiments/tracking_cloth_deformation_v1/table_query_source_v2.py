"""Segment-aware source-only table-query active-probe feasibility study.

The table-collision CSVs contain one isolated setup frame followed by a long
recording gap and a dense 120 Hz motion segment. This wrapper binds the v1
source-only experiment to that documented timestamp structure. Full-lay target
records expose only the setup frame, a causal all-marker prefix, and future
trajectories of the two causally detected grasped corners. No post-prefix
full-lay free-marker coordinate is converted or scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import traceback
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import table_query_source_v1 as base

SCHEMA = "tracking-cloth-table-query-source-v2"


def segmented_rows(
    path: Path, protocol: Mapping[str, Any]
) -> tuple[tuple[float, list[str]], list[tuple[float, list[str]]]]:
    """Return the singleton setup row and dense motion segment."""
    rows = list(base.numeric_rows(path))
    if len(rows) < 3:
        raise ValueError(f"insufficient numeric rows in {path.name}")
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    threshold = float(protocol["gap_threshold_seconds"])
    cuts = np.flatnonzero(np.diff(times) > threshold) + 1
    segments = np.split(np.arange(len(rows)), cuts)
    if len(segments) != 2 or len(segments[0]) != 1:
        raise ValueError(
            f"{path.name} must contain one singleton setup segment "
            "and one motion segment"
        )
    motion_indices = segments[1]
    motion = [rows[int(index)] for index in motion_indices]
    motion_times = times[motion_indices]
    dt = np.diff(motion_times)
    expected_dt = 1.0 / float(protocol["frame_rate_hz"])
    if (
        len(motion) < int(protocol["minimum_motion_segment_rows"])
        or not np.allclose(dt, expected_dt, rtol=0.05, atol=1e-4)
    ):
        raise ValueError(f"{path.name} lacks the frozen dense-motion support")
    duration = float(motion_times[-1] - motion_times[0])
    if duration + 1e-9 < float(
        protocol["minimum_motion_segment_duration_seconds"]
    ):
        raise ValueError(
            f"{path.name} motion segment is shorter than its inventory lock"
        )
    required = float(protocol["prefix_seconds"]) + float(
        protocol["forecast_seconds"]
    )
    if duration + 1e-9 < required:
        raise ValueError(
            f"{path.name} motion segment does not cover the frozen horizon"
        )
    return rows[int(segments[0][0])], motion


def _causal_prefix(
    motion: Sequence[tuple[float, list[str]]],
    order: np.ndarray,
    scale: float,
    prefix_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    start = float(motion[0][0])
    times: list[float] = []
    values: list[np.ndarray] = []
    previous = None
    for timestamp, cells in motion:
        if timestamp > start + prefix_seconds + 1e-9:
            break
        current = base.positions(cells, order) * scale
        if previous is not None:
            current = np.where(np.isfinite(current), current, previous)
        if not np.all(np.isfinite(current)):
            raise ValueError("the first dense-motion prefix row must be complete")
        previous = current.copy()
        times.append(float(timestamp))
        values.append(current)
    if len(values) < 5:
        raise ValueError("insufficient dense-motion prefix for causal edge detection")
    return np.asarray(times), np.asarray(values)


def input_view(
    case: base.TableCase, protocol: Mapping[str, Any]
) -> tuple[base.TableInputs, dict[str, Any]]:
    setup, motion = segmented_rows(case.path, protocol)
    setup_raw = base.positions(setup[1], range(20))
    if setup_raw.shape != (20, 3) or not np.all(np.isfinite(setup_raw)):
        raise ValueError(f"setup frame is incomplete in {case.path.name}")
    scale = base.infer_scale(setup_raw)
    order, layout_diagnostic = base.planar_layout(setup_raw * scale)

    prefix_seconds = float(protocol["prefix_seconds"])
    full_prefix_times, full_prefix = _causal_prefix(
        motion, order, scale, prefix_seconds
    )
    table_z = float(protocol["table_z_m"])
    corners, edge_diagnostic = base.choose_grasped_edge(
        full_prefix_times, full_prefix, table_z
    )
    raw_corners = order[corners]

    motion_start = float(motion[0][0])
    end_time = motion_start + prefix_seconds + float(
        protocol["forecast_seconds"]
    )
    stride = int(protocol["sample_stride"])
    times: list[float] = []
    prefix: list[np.ndarray] = []
    boundary: list[np.ndarray] = []
    last_prefix = None
    last_boundary = None
    for row_index, (timestamp, cells) in enumerate(motion):
        if timestamp > end_time + 1e-8:
            break
        if row_index % stride != 0:
            continue
        corner_values = base.positions(cells, raw_corners) * scale
        if last_boundary is not None:
            corner_values = np.where(
                np.isfinite(corner_values), corner_values, last_boundary
            )
        if not np.all(np.isfinite(corner_values)):
            raise ValueError(f"missing initial driven corner in {case.path.name}")
        last_boundary = corner_values.copy()
        times.append(float(timestamp))
        boundary.append(corner_values)
        if timestamp <= motion_start + prefix_seconds + 1e-8:
            all_values = base.positions(cells, order) * scale
            if last_prefix is not None:
                all_values = np.where(
                    np.isfinite(all_values), all_values, last_prefix
                )
            if not np.all(np.isfinite(all_values)):
                raise ValueError(f"nonfinite causal prefix in {case.path.name}")
            last_prefix = all_values.copy()
            prefix.append(all_values)

    times_array = np.asarray(times, dtype=np.float64)
    prefix_array = np.asarray(prefix, dtype=np.float64)
    boundary_array = np.asarray(boundary, dtype=np.float64)
    if len(prefix_array) < 5 or len(times_array) <= len(prefix_array) + 10:
        raise ValueError(f"insufficient sampled prefix/forecast in {case.path.name}")
    expected_dt = stride / float(protocol["frame_rate_hz"])
    if not np.allclose(
        np.diff(times_array), expected_dt, rtol=0.05, atol=1e-4
    ):
        raise ValueError(
            f"sampled cadence violates the v2 contract in {case.path.name}"
        )
    if times_array[-1] < end_time - 2.0 * expected_dt:
        raise ValueError(f"sampled horizon is incomplete in {case.path.name}")

    inputs = base.TableInputs(
        times=times_array,
        prefix=prefix_array,
        boundary=boundary_array,
        order=order,
        corners=corners,
        cutoff=len(prefix_array) - 1,
        scale=scale,
        table_z=table_z,
    )
    diagnostic = {
        **layout_diagnostic,
        **edge_diagnostic,
        "coordinate_scale_to_m": scale,
        "setup_timestamp_seconds": float(setup[0]),
        "motion_start_timestamp_seconds": motion_start,
        "setup_to_motion_gap_seconds": motion_start - float(setup[0]),
        "motion_segment_rows": len(motion),
        "motion_segment_duration_seconds": float(motion[-1][0] - motion[0][0]),
        "full_rate_prefix_rows": len(full_prefix),
        "sampled_prefix_rows": len(prefix_array),
        "sampled_total_rows": len(times_array),
        "raw_corner_indices": raw_corners.tolist(),
        "full_lay_post_prefix_free_marker_outcomes_read": False,
    }
    return inputs, diagnostic


_ACTIVE_PROTOCOL: Mapping[str, Any] = {}


def scoring_view(case: base.TableCase, inputs: base.TableInputs) -> np.ndarray:
    _, motion = segmented_rows(case.path, _ACTIVE_PROTOCOL)
    rows: list[np.ndarray] = []
    index = 0
    for timestamp, cells in motion:
        if index == len(inputs.times):
            break
        if abs(timestamp - inputs.times[index]) <= 1e-7:
            rows.append(base.positions(cells, inputs.order) * inputs.scale)
            index += 1
    if index != len(inputs.times):
        raise ValueError(f"scoring timestamps do not reproduce {case.path.name}")
    return np.asarray(rows)


def patch_outputs(output: Path, protocol: Mapping[str, Any]) -> None:
    source_fit_path = output / "source_fit.json"
    source_fit = json.loads(source_fit_path.read_text())
    source_fit.update(
        {
            "schema": SCHEMA,
            "segment_contract": {
                "gap_threshold_seconds": protocol["gap_threshold_seconds"],
                "setup_segment_rows": 1,
                "motion_segment_rule": "second and only dense segment",
                "prefix_seconds": protocol["prefix_seconds"],
                "forecast_seconds": protocol["forecast_seconds"],
                "segment_inventory_run_id": protocol["segment_inventory_run_id"],
            },
            "full_lay_post_prefix_free_marker_outcomes_read": False,
        }
    )
    base.write_json(source_fit_path, source_fit)

    source_gate_path = output / "source_gate.json"
    source_gate = json.loads(source_gate_path.read_text())
    source_gate.update(
        {
            "schema": "tracking-cloth-table-query-source-gate-v2",
            "protocol_id": base.object_digest(protocol),
            "full_lay_post_prefix_free_marker_outcomes_read": False,
        }
    )
    base.write_json(source_gate_path, source_gate)

    report_path = output / "report.md"
    report = report_path.read_text().replace(
        "# Tracking Cloth table-query active-probe source feasibility v1",
        "# Tracking Cloth table-query active-probe source feasibility v2",
        1,
    )
    note = (
        "\nThe v2 input contract discards the isolated setup-to-motion timestamp "
        "gap, uses the singleton setup frame only for scale/layout, and evaluates "
        "a 0.5 s causal prefix plus 3.5 s forecast on the dense motion segment.\n"
    )
    report_path.write_text(report + note)

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "schema": "tracking-cloth-table-query-source-run-v2",
            "wrapper_sha256": base.sha256(Path(__file__)),
            "source_fit_sha256": base.sha256(source_fit_path),
            "source_gate_sha256": base.sha256(source_gate_path),
            "segment_inventory_run_id": protocol["segment_inventory_run_id"],
            "full_lay_post_prefix_free_marker_outcomes_read": False,
        }
    )
    base.write_json(manifest_path, manifest)


def run_source(
    root: Path, output: Path, protocol: Mapping[str, Any], workers: int
) -> None:
    global _ACTIVE_PROTOCOL
    _ACTIVE_PROTOCOL = dict(protocol)
    original_input = base.input_view
    original_scoring = base.scoring_view
    try:
        base.input_view = input_view
        base.scoring_view = scoring_view
        base.run_source(root, output, protocol, workers)
    finally:
        base.input_view = original_input
        base.scoring_view = original_scoring
    patch_outputs(output, protocol)


def _csv_row(
    frame: int, timestamp: float, values: Iterable[Sequence[Any]]
) -> list[Any]:
    row: list[Any] = [frame, timestamp]
    for triple in values:
        row.extend(triple)
    return row


def self_test() -> None:
    protocol = {
        "gap_threshold_seconds": 0.05,
        "frame_rate_hz": 120.0,
        "minimum_motion_segment_rows": 500,
        "minimum_motion_segment_duration_seconds": 4.1,
        "prefix_seconds": 0.5,
        "forecast_seconds": 3.5,
        "sample_stride": 4,
        "table_z_m": 0.0,
    }
    grid = np.asarray(
        [
            [0.105 * column, 0.1485 * row, 0.08]
            for row in range(5)
            for column in range(4)
        ],
        dtype=np.float64,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cotton_A2_full_lay_low_friction.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["Frame", "Time", *(f"c{index}" for index in range(60))]
            )
            writer.writerow(_csv_row(1, 1.2, grid))
            for index in range(601):
                timestamp = 20.0 + index / 120.0
                values = grid.copy()
                phase = min(index / 60.0, 1.0)
                values[[0, 3], 0] += 0.03 * phase
                values[[0, 3], 2] += 0.05 * phase
                if index > 60:
                    serialized: list[Sequence[Any]] = []
                    for marker, triple in enumerate(values):
                        if marker in {0, 3, 16, 19}:
                            serialized.append(triple)
                        else:
                            serialized.append(("SEALED", "SEALED", "SEALED"))
                    writer.writerow(_csv_row(index + 2, timestamp, serialized))
                else:
                    writer.writerow(_csv_row(index + 2, timestamp, values))
        case = base.TableCase(path, "cotton", "full_lay", "low_friction")
        setup, motion = segmented_rows(path, protocol)
        assert setup[0] == 1.2
        assert len(motion) == 601
        inputs, diagnostic = input_view(case, protocol)
        assert len(inputs.prefix) >= 5
        assert len(inputs.times) > len(inputs.prefix)
        assert diagnostic["setup_to_motion_gap_seconds"] > 10.0
        assert diagnostic["full_lay_post_prefix_free_marker_outcomes_read"] is False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.dataset_root is None or args.output is None or args.protocol is None:
        parser.error("--dataset-root, --output and --protocol are required")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        protocol = json.loads(args.protocol.read_text())
        run_source(args.dataset_root, args.output, protocol, args.workers)
    except Exception as exc:
        traceback.print_exc()
        if args.output is not None and args.output.is_dir():
            base.write_json(
                args.output / "failure.json",
                {
                    "failed_at": base.now(),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "full_lay_post_prefix_free_marker_outcomes_read": False,
                    "scientific_decision": "incomplete; no target authorization",
                },
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
