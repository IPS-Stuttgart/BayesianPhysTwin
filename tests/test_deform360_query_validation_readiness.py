from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "science"
    / "audit_deform360_query_validation_readiness.py"
)
SPEC = importlib.util.spec_from_file_location("deform360_readiness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_fixture(
    tmp_path: Path, *, missing_metadata: bool = False
) -> dict[str, Path]:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw-repository" / "raw"
    objects: list[dict[str, object]] = []
    for index in range(8):
        suffix = "sample-cloth" if index < 4 else "sample-item"
        object_id = f"{index + 1:03d}-{suffix}"
        object_root = raw_root / object_id
        object_root.mkdir(parents=True)
        if not (missing_metadata and index == 0):
            metadata = {
                "sequences": {
                    "0": {
                        "action": "lift object",
                        "bimanual": "yes",
                        "nonprehensile": "no",
                    },
                    "1": {
                        "action": "bend object",
                        "bimanual": "yes",
                        "nonprehensile": "no",
                    },
                }
            }
            (object_root / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        objects.append(
            {
                "object_id": object_id,
                "classification": "candidate_name_only",
                "numeric_path_counts": {".npy": 1},
            }
        )

    inventory = {
        "schema": MODULE.INVENTORY_SCHEMA,
        "content_inventory_sha256": "fixture-inventory",
        "objects": objects,
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    exclusion = {
        "artifact_kind": MODULE.EXCLUSION_KIND,
        "hash_namespace": MODULE.HASH_NAMESPACE,
        "object_hashes": [MODULE._object_hash(objects[0]["object_id"])],
    }
    canonical = MODULE._sha256_json(exclusion)
    exclusion["exclusion_sha256"] = canonical
    exclusion_path = tmp_path / "exclusion.json"
    exclusion_path.write_text(
        json.dumps(exclusion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exclusion_file_sha = hashlib.sha256(exclusion_path.read_bytes()).hexdigest()

    protocol = {
        "schema": MODULE.SCHEMA,
        "schema_version": 1,
        "protocol_id": "fixture",
        "status": "locked-before-development-metadata-access",
        "dataset": {"root": str(data_root.resolve())},
        "runner": {"label": "gpuserver4090"},
        "preflight_binding": {"content_inventory_sha256": "fixture-inventory"},
        "historical_exclusion_binding": {
            "commit": "fixture",
            "file_sha256": exclusion_file_sha,
            "canonical_sha256": canonical,
        },
        "development_metadata_selection": {
            "objects_per_stratum": 4,
            "rank_seed": "fixture-seed",
            "action_families": {
                "elevation": ["lift", "wave"],
                "planar_or_contact": ["move", "press", "pull", "push"],
                "shape_change": ["bend", "fold", "stretch"],
            },
        },
        "information_boundary": {
            "camera_media_decoded": False,
            "robot_or_tactile_arrays_opened": False,
            "geometry_or_track_annotations_opened": False,
            "target_future_opened": False,
            "score_bearing_outcomes_opened": False,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return {
        "data_root": data_root,
        "protocol": protocol_path,
        "inventory": inventory_path,
        "exclusion": exclusion_path,
    }


def test_builds_deterministic_metadata_design_without_target_access(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    first = MODULE.build_readiness(
        data_root=fixture["data_root"].resolve(),
        protocol_path=fixture["protocol"],
        inventory_path=fixture["inventory"],
        historical_exclusion_path=fixture["exclusion"],
    )
    second = MODULE.build_readiness(
        data_root=fixture["data_root"].resolve(),
        protocol_path=fixture["protocol"],
        inventory_path=fixture["inventory"],
        historical_exclusion_path=fixture["exclusion"],
    )

    assert first == second
    assert first["decision"]["development_metadata_design_ready"] is True
    assert first["decision"]["fresh_confirmation_authorized"] is False
    assert first["decision"]["target_payload_access_authorized"] is False
    design = first["development_metadata_design"]
    assert design["metadata_ready_count"] == 8
    assert not design["unsupported"]
    assert {
        row["source_target_pair"]["source"]["action_family"]
        for row in design["selected"]
    } <= {"elevation", "shape_change"}
    for row in design["selected"]:
        pair = row["source_target_pair"]
        assert pair["source"]["action_family"] != pair["target"]["action_family"]
    assert (
        first["historical_exclusion"]["current_complete_cross_project_union_verified"]
        is False
    )
    assert first["information_boundary"]["target_future_opened"] is False


def test_missing_metadata_fails_closed_without_replacement(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, missing_metadata=True)

    result = MODULE.build_readiness(
        data_root=fixture["data_root"].resolve(),
        protocol_path=fixture["protocol"],
        inventory_path=fixture["inventory"],
        historical_exclusion_path=fixture["exclusion"],
    )

    assert result["decision"]["development_metadata_design_ready"] is False
    assert result["development_metadata_design"]["metadata_ready_count"] == 7
    assert len(result["development_metadata_design"]["unsupported"]) == 1
    assert result["decision"]["model_scoring_authorized"] is False


def test_changed_inventory_stops_before_metadata_and_exclusion_tampering_fails(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    inventory = json.loads(fixture["inventory"].read_text(encoding="utf-8"))
    inventory["content_inventory_sha256"] = "changed"
    fixture["inventory"].write_text(json.dumps(inventory), encoding="utf-8")

    changed = MODULE.build_readiness(
        data_root=fixture["data_root"].resolve(),
        protocol_path=fixture["protocol"],
        inventory_path=fixture["inventory"],
        historical_exclusion_path=fixture["exclusion"],
    )
    assert changed["decision"]["status"] == (
        "names-only-inventory-changed-relock-required"
    )
    assert changed["development_metadata_design"]["selected_name_only_count"] == 0
    assert changed["information_boundary"]["selected_metadata_json_opened"] is False

    fixture = _write_fixture(tmp_path / "second")
    exclusion = json.loads(fixture["exclusion"].read_text(encoding="utf-8"))
    exclusion["object_hashes"].append("0" * 64)
    fixture["exclusion"].write_text(json.dumps(exclusion), encoding="utf-8")
    with pytest.raises(ValueError, match="file digest changed"):
        MODULE.build_readiness(
            data_root=fixture["data_root"].resolve(),
            protocol_path=fixture["protocol"],
            inventory_path=fixture["inventory"],
            historical_exclusion_path=fixture["exclusion"],
        )
