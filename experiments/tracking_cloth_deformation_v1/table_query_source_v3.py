"""Strict-parser dense-segment table-query source feasibility study.

The repository's validated CSV reader excludes the dataset's isolated row because
that row has no valid frame index. It therefore exposes one dense 120 Hz motion
segment for every table-collision record. This version uses the first frame of
that strict segment for metric scale and marker layout, then exposes a causal
all-marker prefix and future trajectories of only the detected grasped corners.
No post-prefix full-lay free-marker coordinate is converted or scored.
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
from . import table_query_source_v2 as previous

SCHEMA = "tracking-cloth-table-query-source-v3"


def dense_rows(
    path: Path, protocol: Mapping[str, Any]
) -> tuple[tuple[float, list[str]], list[tuple[float, list[str]]]]:
    """Return the first dense frame and complete strict-parser motion segment."""
    rows = list(base.numeric_rows(path))
    if len(rows) < int(protocol["minimum_motion_segment_rows"]):
        raise ValueError(f"{path.name} lacks the frozen strict-parser row support")
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    dt = np.diff(times)
    expected_dt = 1.0 / float(protocol["frame_rate_hz"])
    if np.any(dt > float(protocol["gap_threshold_seconds"])) or not np.allclose(
        dt, expected_dt, rtol=0.05, atol=1e-4
    ):
        raise ValueError(f"{path.name} is not one dense strict-parser segment")
    duration = float(times[-1] - times[0])
    if duration + 1e-9 < float(protocol["minimum_motion_segment_duration_seconds"]):
        raise ValueError(f"{path.name} is shorter than the strict inventory lock")
    required = float(protocol["prefix_seconds"]) + float(protocol["forecast_seconds"])
    if duration + 1e-9 < required:
        raise ValueError(f"{path.name} does not cover the frozen source horizon")
    return rows[0], rows


def patch_outputs(output: Path, protocol: Mapping[str, Any]) -> None:
    source_fit_path = output / "source_fit.json"
    source_fit = json.loads(source_fit_path.read_text())
    source_fit.update(
        {
            "schema": SCHEMA,
            "segment_contract": {
                "strict_parser_segments": 1,
                "strict_parser_first_frame_used_for_scale_and_layout": True,
                "nonstandard_isolated_time_only_row_used": False,
                "prefix_seconds": protocol["prefix_seconds"],
                "forecast_seconds": protocol["forecast_seconds"],
                "strict_time_inventory_run_id": protocol[
                    "strict_time_inventory_run_id"
                ],
            },
            "full_lay_post_prefix_free_marker_outcomes_read": False,
        }
    )
    base.write_json(source_fit_path, source_fit)

    source_gate_path = output / "source_gate.json"
    source_gate = json.loads(source_gate_path.read_text())
    source_gate.update(
        {
            "schema": "tracking-cloth-table-query-source-gate-v3",
            "protocol_id": base.object_digest(protocol),
            "full_lay_post_prefix_free_marker_outcomes_read": False,
        }
    )
    base.write_json(source_gate_path, source_gate)

    report_path = output / "report.md"
    report = report_path.read_text().replace(
        "# Tracking Cloth table-query active-probe source feasibility v2",
        "# Tracking Cloth table-query active-probe source feasibility v3",
        1,
    )
    report += (
        "\nThe v3 contract follows the repository's strict parser: the "
        "nonstandard isolated row is excluded, and the first frame of the one "
        "dense 120 Hz segment supplies scale/layout and starts the causal prefix.\n"
    )
    report_path.write_text(report)

    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "schema": "tracking-cloth-table-query-source-run-v3",
            "wrapper_sha256": base.sha256(Path(__file__)),
            "source_fit_sha256": base.sha256(source_fit_path),
            "source_gate_sha256": base.sha256(source_gate_path),
            "strict_time_inventory_run_id": protocol["strict_time_inventory_run_id"],
            "full_lay_post_prefix_free_marker_outcomes_read": False,
        }
    )
    base.write_json(manifest_path, manifest)


def run_source(
    root: Path, output: Path, protocol: Mapping[str, Any], workers: int
) -> None:
    original_segmenter = previous.segmented_rows
    original_input = base.input_view
    original_scoring = base.scoring_view
    previous._ACTIVE_PROTOCOL = dict(protocol)
    try:
        previous.segmented_rows = dense_rows
        base.input_view = previous.input_view
        base.scoring_view = previous.scoring_view
        base.run_source(root, output, protocol, workers)
    finally:
        previous.segmented_rows = original_segmenter
        base.input_view = original_input
        base.scoring_view = original_scoring
    previous.patch_outputs(output, protocol)
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
            writer.writerow(["Frame", "Time", *(f"c{index}" for index in range(60))])
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
                    writer.writerow(_csv_row(index + 1, timestamp, serialized))
                else:
                    writer.writerow(_csv_row(index + 1, timestamp, values))
        case = base.TableCase(path, "cotton", "full_lay", "low_friction")
        first, motion = dense_rows(path, protocol)
        assert first[0] == 20.0
        assert len(motion) == 601
        original = previous.segmented_rows
        try:
            previous.segmented_rows = dense_rows
            inputs, diagnostic = previous.input_view(case, protocol)
        finally:
            previous.segmented_rows = original
        assert len(inputs.prefix) >= 5
        assert len(inputs.times) > len(inputs.prefix)
        assert diagnostic["setup_to_motion_gap_seconds"] == 0.0
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
