from __future__ import annotations

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
    / "build_deform360_hub_selection.py"
)
_SPEC = importlib.util.spec_from_file_location("_deform360_hub_selection", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

bind_episodes = _MODULE.bind_episodes
build_selection = _MODULE.build_selection
load_prior_context = _MODULE.load_prior_context
load_protocol = _MODULE.load_protocol
select_objects = _MODULE.select_objects
write_selection = _MODULE.write_selection


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    protocol_path = repository / "protocols" / "official.json"
    v1_hash = "1" * 64
    v2_hash = "2" * 64
    sheet = [f"s{index}" for index in range(7)]
    volumetric = [f"v{index}" for index in range(7)]
    _write_json(
        repository / "configs" / "v1.json",
        {
            "config_sha256": v1_hash,
            "config": {
                "open_or_reserved_objects": ["s0"],
                "candidate_pools": {
                    "sheet": sheet,
                    "volumetric": volumetric,
                    "filament": ["f0"],
                },
                "calibration_cohort": {
                    "sheet": [{"object_id": "s1", "episode_ids": [0]}]
                },
                "target_cohort": {
                    "volumetric": [{"object_id": "v0", "episode_ids": [1]}]
                },
            },
        },
    )
    _write_json(
        repository / "configs" / "v2.json",
        {
            "config_sha256": v2_hash,
            "config": {
                "calibration_cohort": {},
                "target_cohort": {},
            },
        },
    )
    _write_json(
        protocol_path,
        {
            "schema": "bayesian-phystwin/deform360-official-hub-visuotactile-protocol",
            "schema_version": 1,
            "protocol_id": "test-official",
            "status": "locked-before-official-raw-payload-access",
            "dataset": {
                "repo_id": "brownu/deform360",
                "requested_revision": "main",
                "raw_prefix": "raw",
            },
            "official_processing": {
                "repository": "lhy0807/deform360",
                "revision": "d" * 40,
            },
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
            "cache_preflight": {
                "inventory_sha256": "3" * 64,
                "content_inventory_sha256": "4" * 64,
                "excluded_candidate_objects": ["s2", "v1"],
            },
            "selection": {
                "seed": "test-seed",
                "strata": ["sheet", "volumetric"],
                "calibration_objects_per_stratum": 2,
                "confirmation_objects_per_stratum": 1,
            },
            "information_boundary": {
                "object_directory_names_allowed": True,
                "object_metadata_json_allowed": True,
                "camera_media_opened": False,
                "tactile_arrays_opened": False,
                "target_outcomes_opened": False,
            },
        },
    )
    available = list(reversed(sheet + volumetric + ["unknown-object"]))
    candidate_pools, excluded, _ = load_prior_context(
        repository, load_protocol(protocol_path)
    )
    selected = select_objects(
        available,
        candidate_pools=candidate_pools,
        excluded_objects=excluded,
        selection=load_protocol(protocol_path)["selection"],
    )
    selected_ids = {
        record["object_id"] for records in selected.values() for record in records
    }
    metadata: dict[str, Any] = {}
    metadata_sha: dict[str, str] = {}
    for index, object_id in enumerate(sorted(selected_ids)):
        metadata[object_id] = (
            {"sequences": {"0": {}, "2": {}, "4": {}}}
            if index % 2 == 0
            else {"sequences": [{}, {}, {}]}
        )
        metadata_sha[object_id] = f"{index + 5:x}" * 64
    snapshot = {
        "resolved_revision": "a" * 40,
        "raw_objects": available,
        "metadata_by_object": metadata,
        "metadata_sha256_by_object": metadata_sha,
        "opened_paths": sorted(
            f"raw/{object_id}/metadata.json" for object_id in selected_ids
        ),
    }
    return repository, protocol_path, snapshot


def test_official_hub_selection_is_deterministic_disjoint_and_target_blind(
    tmp_path: Path,
) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)

    first = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="b" * 40,
    )
    shuffled = dict(snapshot)
    shuffled["raw_objects"] = list(reversed(snapshot["raw_objects"]))
    second = build_selection(
        shuffled,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )

    assert first["content_selection_sha256"] == second["content_selection_sha256"]
    assert first["selection_artifact_sha256"] != second["selection_artifact_sha256"]
    calibration = {record["object_id"] for record in first["selection"]["calibration"]}
    confirmation = {
        record["object_id"] for record in first["selection"]["confirmation"]
    }
    assert len(calibration) == 4
    assert len(confirmation) == 2
    assert not calibration & confirmation
    assert not ({"s0", "s1", "s2", "v0", "v1"} & (calibration | confirmation))
    boundary = first["information_boundary"]
    assert boundary["camera_media_opened"] is False
    assert boundary["tactile_arrays_opened"] is False
    assert boundary["robot_arrays_opened"] is False
    assert boundary["geometry_annotations_opened"] is False
    assert boundary["target_outcomes_opened"] is False
    assert all(
        path.endswith("/metadata.json") for path in boundary["opened_metadata_paths"]
    )
    assert first["replacement_allowed_after_payload_access"] is False


def test_selection_binds_mapping_and_list_episode_metadata(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)

    result = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
    )

    episodes = [
        record["episode_id"]
        for records in result["selection"].values()
        for record in records
    ]
    assert episodes
    assert all(episode in {0, 1, 2, 4} for episode in episodes)
    assert len(result["information_boundary"]["opened_metadata_paths"]) == 6


def test_selection_output_is_canonical_and_newline_terminated(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    result = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
    )
    output = tmp_path / "selection.json"

    write_selection(output, result)

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["selection_sha256"] == result["selection_sha256"]
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda protocol, snapshot: protocol["information_boundary"].update(
                {"camera_media_opened": True}
            ),
            "information boundary",
        ),
        (
            lambda protocol, snapshot: protocol["selection"].update(
                {"strata": ["sheet", "filament"]}
            ),
            "sheet and volumetric",
        ),
        (
            lambda protocol, snapshot: protocol["selection"].update(
                {"calibration_objects_per_stratum": True}
            ),
            "positive integer",
        ),
        (
            lambda protocol, snapshot: snapshot.update(
                {"resolved_revision": "not-a-revision"}
            ),
            "resolved revision",
        ),
        (
            lambda protocol, snapshot: snapshot["opened_paths"].append(
                "raw/s3/camera.mp4"
            ),
            "metadata.json",
        ),
        (
            lambda protocol, snapshot: snapshot["opened_paths"].pop(),
            "does not match",
        ),
        (
            lambda protocol, snapshot: snapshot["metadata_by_object"].pop(
                next(iter(snapshot["metadata_by_object"]))
            ),
            "metadata is missing",
        ),
    ],
)
def test_selection_fails_closed_on_boundary_or_provenance_drift(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    mutator(protocol, snapshot)
    _write_json(protocol_path, protocol)

    with pytest.raises(ValueError, match=message):
        build_selection(
            snapshot,
            repository=repository,
            protocol_path=protocol_path,
        )


def test_object_selection_rejects_insufficient_or_duplicate_inventory(
    tmp_path: Path,
) -> None:
    repository, protocol_path, _ = _fixture(tmp_path)
    protocol = load_protocol(protocol_path)
    pools, excluded, _ = load_prior_context(repository, protocol)

    with pytest.raises(ValueError, match="duplicates"):
        select_objects(
            ["s3", "s3", "s4", "s5", "v2", "v3", "v4"],
            candidate_pools=pools,
            excluded_objects=excluded,
            selection=protocol["selection"],
        )
    with pytest.raises(ValueError, match="requires"):
        select_objects(
            ["s3", "s4", "v2", "v3", "v4"],
            candidate_pools=pools,
            excluded_objects=excluded,
            selection=protocol["selection"],
        )


def test_episode_binding_rejects_invalid_sequence_contracts() -> None:
    selection = {
        "calibration": [{"object_id": "s3", "stratum": "sheet"}],
        "confirmation": [],
    }

    with pytest.raises(ValueError, match="no sequences"):
        bind_episodes(
            selection,
            metadata_by_object={"s3": {}},
            metadata_sha256_by_object={"s3": "1" * 64},
            seed="seed",
        )
    with pytest.raises(ValueError, match="Boolean"):
        bind_episodes(
            selection,
            metadata_by_object={"s3": {"sequences": {True: {}}}},
            metadata_sha256_by_object={"s3": "1" * 64},
            seed="seed",
        )
    with pytest.raises(ValueError, match="not an integer"):
        bind_episodes(
            selection,
            metadata_by_object={"s3": {"sequences": {"bad": {}}}},
            metadata_sha256_by_object={"s3": "1" * 64},
            seed="seed",
        )
