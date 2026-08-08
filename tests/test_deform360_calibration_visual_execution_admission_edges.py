from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import test_deform360_calibration_visual_execution_admission as cases

import bayesian_phystwin.deform360_calibration_visual_execution_admission as admission_api
from bayesian_phystwin._portable_contracts import content_id


def _at(value: object, path: tuple[str | int, ...]) -> Any:
    current: Any = value
    for part in path:
        current = current[part]
    return current


def _set_at(
    value: object,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    parent = _at(value, path[:-1])
    parent[path[-1]] = replacement


def _seal(value: dict[str, object], id_field: str) -> None:
    identity = {key: item for key, item in value.items() if key != id_field}
    value[id_field] = content_id(identity)


def _reseal_admission(value: dict[str, object], *, jobs: bool) -> None:
    if jobs:
        for job in _at(value, ("jobs",)):
            _seal(job, "job_id")
    _seal(value, "admission_id")


def test_validation_primitives_and_metadata_loader_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="required"):
        admission_api._require(False, "required")
    with pytest.raises(ValueError, match="JSON object"):
        admission_api._mapping([], name="value")
    with pytest.raises(ValueError, match="JSON array"):
        admission_api._sequence("not-an-array", name="value")
    with pytest.raises(ValueError, match="literal string"):
        admission_api._literal_string(" padded ", name="value")
    with pytest.raises(ValueError, match="integer"):
        admission_api._literal_integer(True, name="value")
    with pytest.raises(ValueError, match="positive finite"):
        admission_api._positive_number(True, name="value")
    with pytest.raises(ValueError, match="positive finite"):
        admission_api._positive_number(float("nan"), name="value")
    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        admission_api._safe_relative_path("../escape.json", name="value")
    with pytest.raises(ValueError, match="two bounds"):
        admission_api._frame_range([0], name="value", expected_count=1)
    with pytest.raises(ValueError, match="non-finite"):
        admission_api._reject_nonfinite("NaN")
    with pytest.raises(ValueError, match="finite JSON"):
        admission_api._json_copy({"value": float("nan")}, name="value")

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="cannot open"):
        admission_api._load_stable_json_object(missing, label="metadata")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        admission_api._load_stable_json_object(directory, label="metadata")

    target = tmp_path / "target.json"
    cases._write_json(target, {})
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        admission_api._load_stable_json_object(symlink, label="metadata")

    with monkeypatch.context() as patch:
        patch.setattr(admission_api, "_MAX_METADATA_BYTES", 0)
        with pytest.raises(ValueError, match="exceeds"):
            admission_api._load_stable_json_object(target, label="metadata")

    with monkeypatch.context() as patch:
        identities = iter(((1, 2, 3, 4, 5), (1, 2, 4, 4, 5)))
        patch.setattr(
            admission_api,
            "_file_identity",
            lambda _value: next(identities),
        )
        with pytest.raises(ValueError, match="changed while being read"):
            admission_api._load_stable_json_object(target, label="metadata")

    with monkeypatch.context() as patch:

        def fail_read(_descriptor: int, _count: int) -> bytes:
            raise OSError("synthetic read failure")

        patch.setattr(admission_api.os, "read", fail_read)
        with pytest.raises(ValueError, match="cannot read"):
            admission_api._load_stable_json_object(target, label="metadata")

    root_array = tmp_path / "root-array.json"
    cases._write_json(root_array, [])
    with pytest.raises(ValueError, match="root must be a JSON object"):
        admission_api._load_stable_json_object(root_array, label="metadata")


def test_inventory_validator_rejects_adversarial_metadata(tmp_path: Path) -> None:
    _plan_path, inventory_path, _plan = cases._inputs(tmp_path)
    base = json.loads(inventory_path.read_text(encoding="utf-8"))

    def reject(
        mutator: Any,
        match: str,
        *,
        reseal: bool = True,
    ) -> None:
        candidate = copy.deepcopy(base)
        mutator(candidate)
        if reseal:
            _seal(candidate, "inventory_id")
        with pytest.raises(ValueError, match=match):
            admission_api.validate_deform360_prepared_source_inventory(candidate)

    reject(lambda value: _set_at(value, ("schema",), "changed"), "schema changed")
    reject(
        lambda value: _set_at(value, ("semantics",), "changed"),
        "semantics changed",
    )
    reject(
        lambda value: _set_at(value, ("status",), "incomplete"),
        "incomplete",
    )
    reject(
        lambda value: _set_at(
            value,
            ("information_boundary", "target_outcomes_used"),
            True,
        ),
        "information boundary changed",
    )
    reject(
        lambda value: _set_at(value, ("claim_boundary",), "changed"),
        "claim boundary changed",
    )
    reject(
        lambda value: _set_at(value, ("source_artifacts",), {}),
        "must not be empty",
    )
    reject(
        lambda value: _at(value, ("source_artifacts",)).pop(
            "sources/calibration-source/result.json"
        ),
        "does not bind the calibration-source result",
    )
    reject(
        lambda value: _set_at(value, ("object_count",), 9),
        "exactly ten calibration objects",
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "synthetic_episode_index"),
            1,
        ),
        "episode index changed",
    )
    reject(
        lambda value: _set_at(value, ("objects", 0, "stratum"), "unknown"),
        "stratum changed",
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "cameras", 0, "video", "byte_count"),
            0,
        ),
        "integer >= 1",
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "cameras", 0, "fps"),
            0.0,
        ),
        "positive finite",
    )

    selected_stop = _at(
        base,
        (
            "objects",
            0,
            "action_window",
            "selected_raw_frame_range_half_open",
            1,
        ),
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "aligned_frame_count"),
            selected_stop - 1,
        ),
        "frame ranges are not nested",
    )
    reject(
        lambda value: _set_at(value, ("objects", 0, "episode_files"), []),
        "episode_files changed",
    )
    reject(
        lambda value: _set_at(value, ("objects", 0, "tactile"), "changed"),
        "tactile records changed",
    )
    reject(
        lambda value: _set_at(value, ("objects", 0, "cameras"), []),
        "camera roster is empty",
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "cameras", 0, "frame_count"),
            _at(base, ("objects", 0, "aligned_frame_count")) + 1,
        ),
        "camera frame count changed",
    )
    reject(
        lambda value: _at(value, ("objects", 0, "cameras")).reverse(),
        "camera roster is not canonical",
    )
    reject(
        lambda value: _at(value, ("objects",)).reverse(),
        "object roster is not canonical",
    )

    first_stratum = _at(base, ("objects", 0, "stratum"))
    replacement_stratum = "volumetric" if first_stratum == "sheet" else "sheet"
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "stratum"),
            replacement_stratum,
        ),
        "five objects per stratum",
    )
    reject(
        lambda value: _set_at(value, ("inventory_id",), "0" * 64),
        "inventory_id does not match",
        reseal=False,
    )


def test_plan_inventory_binding_rejects_cross_artifact_drift(tmp_path: Path) -> None:
    counter = 0

    def reject(mutator: Any, match: str) -> None:
        nonlocal counter
        counter += 1
        plan_path, inventory_path, _plan = cases._inputs(tmp_path / f"case-{counter}")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        mutator(inventory)
        cases._rewrite_inventory(inventory_path, inventory)
        with pytest.raises(ValueError, match=match):
            admission_api.build_deform360_calibration_visual_execution_admission(
                visual_production_plan_path=plan_path,
                prepared_source_inventory_path=inventory_path,
                implementation_revision=cases.IMPLEMENTATION_REVISION,
            )

    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "object_id"),
            "000-replacement",
        ),
        "object cohorts differ",
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "episode_id"),
            _at(value, ("objects", 0, "episode_id")) + 1,
        ),
        "episode_id",
    )

    def swap_strata(value: dict[str, object]) -> None:
        objects = _at(value, ("objects",))
        first = objects[0]
        other = next(
            item for item in objects[1:] if item["stratum"] != first["stratum"]
        )
        first["stratum"], other["stratum"] = other["stratum"], first["stratum"]

    reject(swap_strata, "stratum")

    def shift_ranges(value: dict[str, object]) -> None:
        item = _at(value, ("objects", 0))
        for field in (
            "selected_raw_frame_range_half_open",
            "prediction_raw_frame_range_half_open",
            "prefix_raw_frame_range_half_open",
        ):
            item["action_window"][field] = [
                bound + 1 for bound in item["action_window"][field]
            ]
        item["aligned_frame_count"] += 1
        item["action_window"]["raw_frame_count"] += 1
        for camera in item["cameras"]:
            camera["frame_count"] += 1

    reject(shift_ranges, "frame ranges differ")
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "cameras", 0, "video", "path"),
            "replacement/undistorted.mp4",
        ),
        "video path differs",
    )
    reject(
        lambda value: _set_at(
            value,
            ("objects", 0, "cameras", 0, "timestamps", "path"),
            "replacement/aligned_timestamps.txt",
        ),
        "timestamp path differs",
    )
    reject(
        lambda value: _set_at(
            value,
            ("selection_artifact_sha256",),
            "f" * 64,
        ),
        "selection_artifact_sha256",
    )
    reject(
        lambda value: _set_at(
            value,
            ("calibration_source_run_record_sha256",),
            "e" * 64,
        ),
        "calibration_source_run_record_sha256",
    )
    reject(
        lambda value: _set_at(
            value,
            ("source_artifacts", "sources/calibration-source/result.json"),
            "d" * 64,
        ),
        "calibration-source result",
    )


def test_admission_validator_rejects_global_and_job_drift(tmp_path: Path) -> None:
    base = cases._build(tmp_path)

    def reject(
        mutator: Any,
        match: str,
        *,
        reseal_jobs: bool = False,
        reseal: bool = True,
    ) -> None:
        candidate = copy.deepcopy(base)
        mutator(candidate)
        if reseal:
            _reseal_admission(candidate, jobs=reseal_jobs)
        with pytest.raises(ValueError, match=match):
            admission_api.validate_deform360_calibration_visual_execution_admission(
                candidate
            )

    reject(lambda value: _set_at(value, ("schema",), "changed"), "schema changed")
    reject(lambda value: _set_at(value, ("schema_version",), True), "version changed")
    reject(
        lambda value: _set_at(value, ("semantics",), "changed"),
        "semantics changed",
    )
    reject(
        lambda value: _set_at(
            value,
            ("information_boundary", "target_outcomes_used"),
            True,
        ),
        "information boundary changed",
    )
    reject(
        lambda value: _set_at(value, ("claim_boundary",), "changed"),
        "claim boundary changed",
    )
    reject(
        lambda value: _set_at(value, ("jobs", 0, "stratum"), "unknown"),
        "job stratum changed",
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 0, "aligned_frame_count"),
            _at(
                value,
                ("jobs", 0, "selected_source_frame_range_half_open", 1),
            )
            - 1,
        ),
        "frame ranges are not nested",
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 0, "source_video", "byte_count"),
            0,
        ),
        "integer >= 1",
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 0, "dependence_group_ids"),
            ["0" * 64],
        ),
        "exactly two dependence groups",
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 0, "dependence_group_ids"),
            ["0" * 64, "0" * 64],
        ),
        "dependence groups must be distinct",
    )
    reject(
        lambda value: _set_at(
            value,
            ("camera_view_count",),
            _at(value, ("camera_view_count",)) - 1,
        ),
        "camera_view_count differs",
    )
    reject(
        lambda value: _set_at(value, ("object_count",), 9),
        "exactly ten physical objects",
    )
    reject(
        lambda value: _at(value, ("jobs",)).reverse(),
        "sorted and unique",
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 1, "output_relative_directory"),
            _at(value, ("jobs", 0, "output_relative_directory")),
        ),
        "output path collision",
        reseal_jobs=True,
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 1, "call_namespace"),
            _at(value, ("jobs", 0, "call_namespace")),
        ),
        "call namespace collision",
        reseal_jobs=True,
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 1, "view_root_seed"),
            _at(value, ("jobs", 0, "view_root_seed")),
        ),
        "view seed collision",
        reseal_jobs=True,
    )

    def change_one_view_stratum(value: dict[str, object]) -> None:
        jobs = _at(value, ("jobs",))
        original = jobs[1]["stratum"]
        jobs[1]["stratum"] = "volumetric" if original == "sheet" else "sheet"

    reject(
        change_one_view_stratum,
        "stratum changes within object",
        reseal_jobs=True,
    )
    reject(
        lambda value: _set_at(
            value,
            ("jobs", 1, "aligned_frame_count"),
            _at(value, ("jobs", 1, "aligned_frame_count")) + 1,
        ),
        "object contract changes",
        reseal_jobs=True,
    )

    def change_complete_object_stratum(value: dict[str, object]) -> None:
        jobs = _at(value, ("jobs",))
        object_id = next(job["object_id"] for job in jobs if job["stratum"] == "sheet")
        for job in jobs:
            if job["object_id"] == object_id:
                job["stratum"] = "volumetric"

    reject(
        change_complete_object_stratum,
        "five sheet objects",
        reseal_jobs=True,
    )
    reject(
        lambda value: _set_at(value, ("admission_id",), "0" * 64),
        "admission_id does not match",
        reseal=False,
    )
