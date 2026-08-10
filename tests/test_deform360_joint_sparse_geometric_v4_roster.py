from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_geometric_source_v4 import (
    _validate_plan_roster,
)

SUPPORTED_COUNTS = (32, 32, 32, 31, 31, 31, 31, 31, 31, 31)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_record(value: str) -> dict[str, object]:
    return {
        "path": f"{value}.json",
        "sha256": _sha(value),
        "byte_count": 1,
    }


def _fixture() -> tuple[dict[str, Any], dict[tuple[str, int], str]]:
    cases: list[dict[str, object]] = []
    selected: dict[tuple[str, int], str] = {}
    included_pairs: set[tuple[str, str]] = set()
    for object_index, stream_count in enumerate(SUPPORTED_COUNTS):
        object_id = f"object-{object_index:02d}"
        episode_id = object_index
        stratum = "sheet" if object_index < 5 else "volumetric"
        selected[(object_id, episode_id)] = stratum
        streams: list[dict[str, object]] = []
        for camera_index in range(stream_count):
            camera_id = f"camera-{camera_index:03d}"
            job_id = _sha(f"job:{object_id}:{camera_id}")
            included_pairs.add((object_id, camera_id))
            streams.append(
                {
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "prediction_manifest": _file_record(f"prediction-{job_id}"),
                    "metric_prefix": _file_record(f"metric-{job_id}"),
                    "metric_calibration": _file_record(f"calibration-{job_id}"),
                }
            )
        case_identity = {
            "schema": "bayesian-phystwin.deform360-prob4d-metric-case-id-v1",
            "object_id": object_id,
            "episode_id": episode_id,
        }
        cases.append(
            {
                "case_id": content_id(case_identity),
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "causal_frame_range_half_open": [0, 58],
                "streams": streams,
            }
        )

    excluded: list[dict[str, object]] = []
    for index in range(11):
        object_index = index % 10
        object_id = f"object-{object_index:02d}"
        episode_id = object_index
        camera_id = f"excluded-camera-{index:02d}"
        assert (object_id, camera_id) not in included_pairs
        excluded.append(
            {
                "job_id": _sha(f"excluded-job:{index}"),
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": selected[(object_id, episode_id)],
                "camera_id": camera_id,
                "reason": "released-robot-geometry-outside-fixed-camera-prefix",
            }
        )
    return {"cases": cases, "excluded_streams": excluded}, selected


def test_exact_313_included_and_11_excluded_roster_passes() -> None:
    plan, selected = _fixture()
    _validate_plan_roster(plan=plan, selected=selected)


def test_duplicate_included_job_fails_closed() -> None:
    plan, selected = _fixture()
    plan = copy.deepcopy(plan)
    cases = plan["cases"]
    cases[1]["streams"][0]["job_id"] = cases[0]["streams"][0]["job_id"]
    with pytest.raises(ValueError, match="repeats a job"):
        _validate_plan_roster(plan=plan, selected=selected)


def test_missing_supported_stream_cannot_hide_behind_aggregate_counts() -> None:
    plan, selected = _fixture()
    plan = copy.deepcopy(plan)
    plan["cases"][0]["streams"].pop()
    with pytest.raises(ValueError, match="supported-stream count"):
        _validate_plan_roster(plan=plan, selected=selected)


def test_changed_support_negative_reason_fails_closed() -> None:
    plan, selected = _fixture()
    plan = copy.deepcopy(plan)
    plan["excluded_streams"][0]["reason"] = "post-hoc-camera-rejection"
    with pytest.raises(ValueError, match="reason changed"):
        _validate_plan_roster(plan=plan, selected=selected)


def test_included_camera_cannot_reappear_as_excluded() -> None:
    plan, selected = _fixture()
    plan = copy.deepcopy(plan)
    included = plan["cases"][0]["streams"][0]
    plan["excluded_streams"][0]["object_id"] = plan["cases"][0]["object_id"]
    plan["excluded_streams"][0]["episode_id"] = plan["cases"][0]["episode_id"]
    plan["excluded_streams"][0]["stratum"] = plan["cases"][0]["stratum"]
    plan["excluded_streams"][0]["camera_id"] = included["camera_id"]
    with pytest.raises(ValueError, match="repeats an excluded object/camera"):
        _validate_plan_roster(plan=plan, selected=selected)
