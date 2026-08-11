from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "build_deform360_covariance_target_v1.py"
)
_SPEC = importlib.util.spec_from_file_location("_covariance_target_v1", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_selection = _MODULE.build_selection
load_protocol = _MODULE.load_protocol
select_candidate_panel = _MODULE.select_candidate_panel


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metadata() -> dict[str, Any]:
    rows = [
        ("lift corner", "no"),
        ("lift edge", "yes"),
        ("drag", "no"),
        ("push", "yes"),
        ("fold", "no"),
        ("stretch", "yes"),
    ]
    return {
        "sequences": {
            str(index): {
                "action": action,
                "bimanual": bimanual,
                "nonprehensile": "no",
            }
            for index, (action, bimanual) in enumerate(rows)
        }
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    exclusion_path = repository / "protocols" / "locks" / "exclusion.json"
    exclusion = {
        "artifact_kind": "deform360-covariance-only-target-exclusion-v1",
        "schema_version": 1,
        "hash_namespace": "deform360-fresh-object-exclusion-v1",
        "object_hashes": [],
        "object_hash_count": 0,
    }
    exclusion["exclusion_sha256"] = _canonical(exclusion)
    _write(exclusion_path, exclusion)
    protocol_path = repository / "protocols" / "locks" / "protocol.json"
    protocol = {
        "schema": "bayesian-phystwin/deform360-covariance-only-target-protocol-v1",
        "schema_version": 1,
        "protocol_id": "test-covariance-target",
        "status": "locked-before-target-metadata-access",
        "dataset": {
            "repository": "brownu/deform360",
            "revision": "a" * 40,
            "raw_prefix": "raw",
            "processed_prefix": "processed",
        },
        "implementation_revision": "b" * 40,
        "exclusion": {
            "artifact_path": "protocols/locks/exclusion.json",
            "file_sha256": hashlib.sha256(exclusion_path.read_bytes()).hexdigest(),
            "canonical_sha256": exclusion["exclusion_sha256"],
            "hash_namespace": "deform360-fresh-object-exclusion-v1",
            "object_hash_count": 0,
        },
        "information_boundary": {
            "camera_media_decoded": False,
            "robot_or_tactile_arrays_opened": False,
            "geometry_or_track_annotations_opened": False,
            "target_outcomes_opened": False,
        },
        "selection": {
            "seed": "test-covariance-target",
            "roster_size": 24,
            "candidate_objects_per_stratum": 16,
            "metadata_invalid_candidate_policy": (
                "terminate before target payload; do not replace"
            ),
            "action_families": {
                "elevation": ["lift", "wave"],
                "planar_or_contact": ["drag", "push"],
                "shape_change": ["fold", "stretch"],
            },
            "exact_factorial_cells": {
                "object_stratum": ["sheet", "volumetric"],
                "bimanual": ["no", "yes"],
                "sessions_per_cell": 2,
            },
        },
    }
    protocol["protocol_sha256"] = _canonical(protocol)
    _write(protocol_path, protocol)

    sheet = [f"{index:03d}-sheet-{index}-cloth" for index in range(16)]
    volumetric = [f"{index + 100:03d}-volume-{index}" for index in range(16)]
    available = sheet + volumetric
    metadata = {object_id: _metadata() for object_id in available}
    metadata_sha = {
        object_id: _canonical(metadata[object_id]) for object_id in available
    }
    snapshot = {
        "resolved_revision": "a" * 40,
        "raw_objects": available,
        "metadata_by_object": metadata,
        "metadata_sha256_by_object": metadata_sha,
        "opened_paths": [f"raw/{object_id}/metadata.json" for object_id in available],
    }
    return repository, protocol_path, snapshot


def test_factorial_selection_is_deterministic_and_exact(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    first, touched = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )
    shuffled = dict(snapshot)
    shuffled["raw_objects"] = list(reversed(snapshot["raw_objects"]))
    shuffled["metadata_by_object"] = dict(
        reversed(list(snapshot["metadata_by_object"].items()))
    )
    second, _ = build_selection(
        shuffled,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )

    assert first["selection_sha256"] == second["selection_sha256"]
    assert len(first["candidate_panel"]) == 32
    assert len(first["target_roster"]) == 24
    assert len({row["object_id"] for row in first["target_roster"]}) == 24
    cells: dict[tuple[str, str, str], int] = {}
    for row in first["target_roster"]:
        cell = (
            row["stratum"],
            row["bimanual"],
            row["action_family"],
        )
        cells[cell] = cells.get(cell, 0) + 1
    assert len(cells) == 12
    assert set(cells.values()) == {2}
    assert touched["object_hash_count"] == 32
    assert touched["information_boundary"]["target_outcomes_opened"] is False


def test_exclusion_hash_removes_object_before_metadata_panel(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    protocol, _ = load_protocol(protocol_path, repository=repository)
    object_id = snapshot["raw_objects"][0]
    object_hash = _MODULE._object_hash(object_id)

    panel = select_candidate_panel(
        snapshot["raw_objects"] + ["999-extra-sheet-cloth"],
        excluded_hashes={object_hash},
        seed=protocol["selection"]["seed"],
        count_per_stratum=16,
    )

    assert object_id not in {row["object_id"] for row in panel}
    assert len(panel) == 32


def test_malformed_metadata_stops_without_replacement(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    object_id = snapshot["raw_objects"][0]
    snapshot["metadata_by_object"][object_id]["sequences"]["0"]["bimanual"] = (
        "yess"
    )

    with pytest.raises(ValueError, match="bimanual is malformed"):
        build_selection(
            snapshot,
            repository=repository,
            protocol_path=protocol_path,
            implementation_revision="c" * 40,
        )


def test_infeasible_factorial_panel_stops_before_payload(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    for object_id, metadata in snapshot["metadata_by_object"].items():
        if object_id.endswith("-cloth"):
            for sequence in metadata["sequences"].values():
                sequence["bimanual"] = "no"

    with pytest.raises(ValueError, match="factorial cell"):
        build_selection(
            snapshot,
            repository=repository,
            protocol_path=protocol_path,
            implementation_revision="c" * 40,
        )


def test_protocol_tampering_is_rejected(tmp_path: Path) -> None:
    repository, protocol_path, _ = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["selection"]["roster_size"] = 23
    _write(protocol_path, protocol)

    with pytest.raises(ValueError, match="protocol digest changed"):
        load_protocol(protocol_path, repository=repository)
