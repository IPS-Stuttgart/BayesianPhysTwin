import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    DATASET_REVISION,
    build_metadata_preflight,
    build_staging_queue,
    load_metadata_preflight,
    load_staging_queue,
    morphology_stratum,
)
from bayesian_phystwin.deform360_object_exclusion import (
    canonical_sha256,
    file_sha256,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _catalog(path: Path, object_ids: list[str]) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PublicObjectCatalogSnapshot",
        "objects": [
            {
                "object_id": object_id,
                "oid": hashlib.sha1(object_id.encode()).hexdigest(),
            }
            for object_id in object_ids
        ],
    }
    payload["catalog_sha256"] = "c" * 64
    _write_json(path, payload)


def _object_hash(object_id: str) -> str:
    return hashlib.sha256(
        b"deform360-fresh-object-exclusion-v1\0" + object_id.encode()
    ).hexdigest()


def _exclusion(path: Path, object_id: str) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshObjectExclusionManifest",
        "hash_namespace": "deform360-fresh-object-exclusion-v1",
        "owner": "unit",
        "object_hashes": [_object_hash(object_id)],
        "source_artifact_sha256s": ["a" * 64],
        "information_boundary": {
            "target_artifact_read": False,
            "object_ids_emitted": False,
        },
    }
    payload["exclusion_sha256"] = canonical_sha256(payload)
    _write_json(path, payload)


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sheet = [f"{index:03d}-sheet-{index}-cloth" for index in range(1, 14)]
    compact = [f"{index:03d}-compact-{index}" for index in range(100, 113)]
    complex_rows = [f"{index:03d}-complex-{index}" for index in range(138, 151)]
    excluded = "001-sheet-1-cloth"
    invalid = "100-compact-100"
    object_ids = sorted(sheet + compact + complex_rows)

    catalog = tmp_path / "catalog.json"
    _catalog(catalog, object_ids)
    exclusion = tmp_path / "exclusion.json"
    _exclusion(exclusion, excluded)
    metadata = tmp_path / "metadata"
    for object_id in object_ids:
        bimanual = "maybe" if object_id == invalid else "no"
        _write_json(
            metadata / f"{object_id}.json",
            {
                "object": object_id.removesuffix("-cloth"),
                "sequences": {
                    "0": {
                        "action": "lift",
                        "bimanual": bimanual,
                        "nonprehensile": "no",
                    }
                },
            },
        )

    protocol = tmp_path / "protocol.json"
    _write_json(
        protocol,
        {
            "schema_version": 1,
            "protocol_id": "deform360-dynamic-tapnextpp-provider-v1",
            "data_source": {
                "repository": "brownu/deform360",
                "dataset_revision": DATASET_REVISION,
                "public_catalog_file_sha256": file_sha256(catalog),
                "public_catalog_sha256": "c" * 64,
                "staged_candidate_count": 36,
            },
            "fresh_object_boundary": {
                "exclusion_sha256": json.loads(
                    exclusion.read_text(encoding="utf-8")
                )["exclusion_sha256"],
                "exclusion_file_sha256": file_sha256(exclusion),
            },
        },
    )
    return protocol, catalog, exclusion, metadata


def test_morphology_stratum_is_public_name_only() -> None:
    assert morphology_stratum("025-bag-cloth") == "sheet"
    assert morphology_stratum("097-pillow") == "compact"
    assert morphology_stratum("186-monster") == "complex"
    with pytest.raises(ValueError, match="malformed"):
        morphology_stratum("bad")


def test_preflight_and_queue_are_reproducible_and_outcome_blind(
    tmp_path: Path,
) -> None:
    protocol, catalog, exclusion, metadata = _synthetic_inputs(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    preflight = build_metadata_preflight(
        preflight_path,
        protocol_path=protocol,
        catalog_path=catalog,
        exclusion_path=exclusion,
        metadata_root=metadata,
    )

    assert preflight["counts"] == {
        "nonexcluded": 38,
        "accepted": 37,
        "rejected": 1,
    }
    invalid = next(
        row for row in preflight["objects"] if row["object_id"] == "100-compact-100"
    )
    assert invalid["rejection_reasons"] == ["invalid-bimanual-enum"]
    assert preflight["information_boundary"]["episode_media_read"] is False

    queue_path = tmp_path / "queue.json"
    queue = build_staging_queue(
        queue_path,
        protocol_path=protocol,
        preflight_path=preflight_path,
        implementation_commit="1" * 40,
    )
    assert queue["stratum_counts"] == {
        "sheet": 12,
        "compact": 12,
        "complex": 12,
    }
    assert [row["category"] for row in queue["candidates"][:6]] == [
        "sheet",
        "compact",
        "complex",
        "sheet",
        "compact",
        "complex",
    ]
    assert "001-sheet-1-cloth" not in {
        row["object_id"] for row in queue["candidates"]
    }
    assert "100-compact-100" not in {
        row["object_id"] for row in queue["candidates"]
    }
    load_metadata_preflight(preflight_path)
    load_staging_queue(queue_path, preflight_path=preflight_path)


def test_queue_tampering_is_detected(tmp_path: Path) -> None:
    protocol, catalog, exclusion, metadata = _synthetic_inputs(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    build_metadata_preflight(
        preflight_path,
        protocol_path=protocol,
        catalog_path=catalog,
        exclusion_path=exclusion,
        metadata_root=metadata,
    )
    queue_path = tmp_path / "queue.json"
    build_staging_queue(
        queue_path,
        protocol_path=protocol,
        preflight_path=preflight_path,
        implementation_commit="2" * 40,
    )
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    payload["candidates"][0]["episode_id"] = 1
    queue_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_staging_queue(queue_path, preflight_path=preflight_path)
