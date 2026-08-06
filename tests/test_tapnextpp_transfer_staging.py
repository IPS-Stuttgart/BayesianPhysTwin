from __future__ import annotations

import json
import pickle
from importlib import util
from pathlib import Path
from types import ModuleType

import numpy as np

from bayesian_phystwin.tapnextpp_transfer_staging import (
    deterministic_farthest_point_indices,
    plan_transfer_case,
    select_physical_motion_window,
    select_query_identity_ids,
    validate_transfer_protocol,
)


def _protocol_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_tapnextpp_depth_completion_transfer_v1.json"
    )


def _load_protocol() -> dict[str, object]:
    return json.loads(_protocol_path().read_text(encoding="utf-8"))


def _load_staging_runner() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "prepare_phystwin_tapnextpp_depth_transfer.py"
    )
    spec = util.spec_from_file_location("tapnextpp_transfer_runner", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_farthest_point_selection_is_deterministic() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    first = deterministic_farthest_point_indices(points, 3)
    second = deterministic_farthest_point_indices(points.copy(), 3)
    assert np.array_equal(first, second)
    assert first.tolist() == [0, 1, 2]


def test_window_uses_physical_rollout_and_earliest_maximum() -> None:
    trajectory = np.zeros((8, 4, 3), dtype=np.float64)
    trajectory[3, :, 0] = 0.1
    trajectory[6, :, 0] = 0.1
    result = select_physical_motion_window(
        trajectory,
        train_end_frame_exclusive=8,
        window_frames=4,
        sampled_node_count=4,
    )
    assert result.start_frame == 0
    assert result.end_frame_exclusive == 4
    assert np.isclose(result.rms_endpoint_displacement_m, 0.1)


def test_query_selection_ignores_every_frame_after_source() -> None:
    tracks = np.full((6, 5, 3), np.nan)
    tracks[2] = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    selected = select_query_identity_ids(tracks, source_frame=2)
    tracks[3:] = 999.0
    mutated = select_query_identity_ids(tracks, source_frame=2)
    assert np.array_equal(selected, mutated)
    assert selected.tolist() == [0, 3, 1, 2]


def test_frozen_transfer_protocol_builds_dynamic_case_plan() -> None:
    protocol = _load_protocol()
    validate_transfer_protocol(protocol)
    physical = np.zeros((30, 6, 3), dtype=np.float64)
    physical[:, :, 0] = np.arange(30)[:, None] * 0.001
    tracks = np.zeros((30, 5, 3), dtype=np.float64)
    tracks[0, :, 0] = np.arange(5) * 0.01
    plan = plan_transfer_case(
        protocol["fixed_source_cases"][0],
        physical,
        tracks,
        train_end_frame_exclusive=25,
        protocol=protocol,
    )
    assert plan.tracker_config.case_name == protocol["fixed_source_cases"][0]
    assert plan.tracker_config.source_frame_start == 0
    assert plan.tracker_config.source_frame_end_exclusive == 20
    assert len(plan.selected_identity_ids) == 4


def _write_panel_inputs(
    root: Path,
    protocol: dict[str, object],
    *,
    omit_masks_for: str | None = None,
) -> tuple[Path, Path]:
    raw = root / "raw"
    physical = root / "physical"
    for case in protocol["fixed_source_cases"]:
        raw_case = raw / case
        (raw_case / "mask").mkdir(parents=True)
        physical_case = physical / case
        physical_case.mkdir(parents=True)
        tracks = np.zeros((30, 5, 3), dtype=np.float64)
        tracks[:, :, 0] = np.arange(5)[None] * 0.01
        trajectory = np.zeros((30, 6, 3), dtype=np.float64)
        trajectory[:, :, 0] = np.arange(30)[:, None] * 0.001
        with (raw_case / "gt_track_3d.pkl").open("wb") as stream:
            pickle.dump(tracks, stream)
        (raw_case / "split.json").write_text(
            json.dumps({"train": [0, 25], "test": [25, 30]}),
            encoding="utf-8",
        )
        if case != omit_masks_for:
            masks = {
                frame: {
                    camera: {"object": np.ones((4, 5), dtype=bool)}
                    for camera in (0, 1, 2)
                }
                for frame in range(25)
            }
            with (raw_case / "mask" / "processed_masks.pkl").open(
                "wb"
            ) as stream:
                pickle.dump(masks, stream)
        with (physical_case / "inference.pkl").open("wb") as stream:
            pickle.dump(trajectory, stream)
    return raw, physical


def test_transfer_staging_retains_fixed_case_failure_without_replacement(
    tmp_path: Path,
) -> None:
    protocol = _load_protocol()
    failed_case = protocol["fixed_source_cases"][3]
    raw, physical = _write_panel_inputs(
        tmp_path,
        protocol,
        omit_masks_for=failed_case,
    )
    runner = _load_staging_runner()
    result = runner.stage_transfer_panel(
        _protocol_path(),
        raw,
        physical,
        tmp_path / "output",
    )
    assert result["fixed_case_count"] == 8
    assert result["prediction_ready_count"] == 7
    assert result["technical_staging_failure_count"] == 1
    assert [record["case"] for record in result["case_records"]] == protocol[
        "fixed_source_cases"
    ]
    failed = next(
        record for record in result["case_records"] if record["case"] == failed_case
    )
    assert failed["status"] == "technical-staging-failure"
    assert failed["replacement_permitted"] is False
