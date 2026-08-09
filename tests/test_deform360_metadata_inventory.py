from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "build_deform360_metadata_inventory.py"
)
_SPEC = importlib.util.spec_from_file_location("_deform360_metadata_inventory", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_metadata_inventory = _MODULE.build_metadata_inventory
load_preflight_protocol = _MODULE.load_preflight_protocol
write_inventory = _MODULE.write_inventory


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repo"
    data_root = tmp_path / "data"
    protocol_path = repository / "protocols" / "preflight.json"
    v1_hash = "1" * 64
    v2_hash = "2" * 64
    _write_json(
        repository / "configs" / "v1.json",
        {
            "config_sha256": v1_hash,
            "config": {
                "open_or_reserved_objects": ["001-rope"],
                "candidate_pools": {
                    "sheet": ["010-orange-cloth"],
                    "filament": ["075-leather"],
                },
                "calibration_cohort": {
                    "sheet": [{"object_id": "010-orange-cloth", "episode_ids": [2]}]
                },
                "target_cohort": {
                    "filament": [{"object_id": "075-leather", "episode_ids": [3]}]
                },
            },
        },
    )
    _write_json(
        repository / "configs" / "v2.json",
        {
            "config_sha256": v2_hash,
            "config": {
                "calibration_cohort": {
                    "sheet": [{"object_id": "010-orange-cloth", "episode_ids": [2]}]
                },
                "target_cohort": {
                    "filament": [{"object_id": "075-leather", "episode_ids": [3]}]
                },
            },
        },
    )
    _write_json(
        protocol_path,
        {
            "schema": "bayesian-phystwin/deform360-metadata-preflight-protocol",
            "schema_version": 1,
            "protocol_id": "test-preflight",
            "status": "locked-before-new-dataset-payload-access",
            "prior_protocols": {
                "v1": {
                    "path": "configs/v1.json",
                    "expected_config_sha256": v1_hash,
                },
                "v2": {
                    "path": "configs/v2.json",
                    "expected_config_sha256": v2_hash,
                },
            },
            "numeric_suffixes": [".h5", ".npy", ".npz", ".ply"],
            "sample_path_limit_per_object": 4,
            "information_boundary": {
                "dataset_payload_opened": False,
                "file_contents_hashed": False,
                "names_and_directory_structure_only": True,
                "reserved_target_outcomes_opened": False,
            },
        },
    )
    for relative in (
        "official/001-rope/episode_0001/tracks.npz",
        "prior/010-orange-cloth-ep0002/frame_zero_points.npz",
        "reserved/075-leather-ep0003/trajectory.npz",
        "derived/010-orange-cloth_episode_0004_brics_cam0_tracking/vel.h5",
        "noise/002-016-candidate-sheet.jpg/points.npz",
        "noise/010-orange-cloth-reference-sheet.jpg",
    ):
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return repository, data_root, protocol_path


def test_inventory_uses_exact_known_object_vocabulary(tmp_path: Path) -> None:
    repository, data_root, protocol_path = _fixture(tmp_path)

    result = build_metadata_inventory(
        data_root,
        repository=repository,
        protocol_path=protocol_path,
        revision="revision-test",
    )

    objects = {record["object_id"]: record for record in result["objects"]}
    assert set(objects) == {"001-rope", "010-orange-cloth", "075-leather"}
    assert objects["001-rope"]["classification"] == "prior_open_or_reserved"
    assert objects["010-orange-cloth"]["classification"] == "prior_calibration"
    assert objects["075-leather"]["classification"] == "reserved_target"
    assert objects["010-orange-cloth"]["episode_ids_from_names"] == [2, 4]
    assert result["reserved_target_objects_present_by_name"] == ["075-leather"]
    assert result["information_boundary"]["dataset_payload_opened"] is False
    assert all(
        "002-016-candidate-sheet.jpg" not in record["object_id"]
        for record in result["objects"]
    )


def test_inventory_records_contract_hints_without_opening_files(
    tmp_path: Path,
) -> None:
    repository, data_root, protocol_path = _fixture(tmp_path)

    result = build_metadata_inventory(
        data_root,
        repository=repository,
        protocol_path=protocol_path,
    )

    objects = {record["object_id"]: record for record in result["objects"]}
    assert objects["001-rope"]["contract_hint_counts"]["tracking"] == 1
    assert objects["010-orange-cloth"]["contract_hint_counts"] == {
        "frame_zero_points": 1,
        "tracking": 1,
    }
    assert objects["075-leather"]["contract_hint_counts"]["trajectory"] == 1
    assert result["total_files_named"] == 6


def test_inventory_is_deterministic_and_writable(tmp_path: Path) -> None:
    repository, data_root, protocol_path = _fixture(tmp_path)

    first = build_metadata_inventory(
        data_root,
        repository=repository,
        protocol_path=protocol_path,
        revision="revision-test",
    )
    second = build_metadata_inventory(
        data_root,
        repository=repository,
        protocol_path=protocol_path,
        revision="revision-test-2",
    )

    assert first["content_inventory_sha256"] == second["content_inventory_sha256"]
    assert first["inventory_sha256"] != second["inventory_sha256"]
    output = tmp_path / "inventory.json"
    write_inventory(output, first)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["inventory_sha256"] == first["inventory_sha256"]
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_protocol_rejects_changed_information_boundary(tmp_path: Path) -> None:
    repository, _, protocol_path = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["information_boundary"]["dataset_payload_opened"] = True
    _write_json(protocol_path, protocol)

    try:
        load_preflight_protocol(protocol_path)
    except ValueError as error:
        assert "information boundary" in str(error)
    else:
        raise AssertionError("changed information boundary was accepted")

    assert repository.is_dir()
