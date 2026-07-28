import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    DATASET_REVISION,
    build_dynamic_provider_cohort_lock,
    build_metadata_preflight,
    build_staging_queue,
    build_terminal_disposition,
    dynamic_provider_case_record,
    load_dynamic_provider_cohort_lock,
    load_metadata_preflight,
    load_staging_queue,
    morphology_stratum,
    validate_dynamic_provider_cohort_lock,
)
from bayesian_phystwin.deform360_fresh_source_lock import (
    ADMISSION_KIND,
    UPSTREAM_BINDING,
    FreshSourceAdmissionConfig,
)
from bayesian_phystwin.deform360_object_exclusion import (
    canonical_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PROTOCOL = (
    ROOT / "configs" / "sota" / "deform360_dynamic_tapnextpp_provider_v1.json"
)
SOURCE_EVALUATION_PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_dynamic_tapnextpp_source_evaluation_v1.json"
)
STAGING_QUEUE = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_dynamic_tapnextpp_staging_queue_v1.json"
)
PROCESSING_PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_dynamic_tapnextpp_source_processing_v1.json"
)
RUNTIME_AMENDMENT = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_dynamic_tapnextpp_source_processing_runtime_amendment_v1.json"
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


def _seal(payload: dict[str, object], key: str) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop(key, None)
    payload[key] = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return payload


def _dynamic_admission(path: Path, row: dict[str, object]) -> None:
    config = FreshSourceAdmissionConfig(minimum_camera_count=8)
    object_id = str(row["object_id"])
    episode_id = int(row["episode_id"])
    cameras = [f"camera-{index}" for index in range(8)]
    digest = "d" * 64
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": ADMISSION_KIND,
        "upstream_binding": UPSTREAM_BINDING,
        "case": f"{object_id}-ep{episode_id:04d}",
        "object_id": object_id,
        "episode_id": episode_id,
        "category": row["category"],
        "accepted": True,
        "rejection_reasons": [],
        "config": {
            "minimum_camera_count": config.minimum_camera_count,
            "minimum_point_count": config.minimum_point_count,
            "maximum_point_count": config.maximum_point_count,
            "required_frame_count": config.required_frame_count,
            "update_frames": list(config.update_frames),
            "minimum_test_frame_count": config.minimum_test_frame_count,
        },
        "observed_source_contract": {
            "metadata_parent": object_id,
            "metadata_object": object_id,
            "bimanual": False,
            "camera_count": len(cameras),
            "cameras": cameras,
            "frame_zero_point_count": 128,
            "split_frame_count": 76,
            "active_frame_count": 76,
            "contact_start_frame": 0,
            "contact_end_frame": 75,
            "train_fraction": 0.8,
            "stage_inputs_valid": True,
            "train": [0, 60],
            "test": [60, 76],
        },
        "source_files": {
            name: {"basename": f"{name}.bin", "sha256": digest}
            for name in (
                "metadata",
                "control_meta",
                "split",
                "calibrate",
                "frame_zero",
                "future_payload",
            )
        },
        "information_boundary": {
            "future_object_positions_deserialized": False,
            "future_payload_bytes_hashed": True,
            "future_metrics_read": False,
            "selection_inputs": "unit source contracts only",
        },
    }
    _write_json(path, _seal(payload, "admission_sha256"))


def _complete_dispositions(
    tmp_path: Path,
) -> tuple[list[Path], list[Path]]:
    queue = load_staging_queue(STAGING_QUEUE)
    per_stratum = {name: 0 for name in ("sheet", "compact", "complex")}
    admissions: list[Path] = []
    terminals: list[Path] = []
    for row in queue["candidates"]:
        category = str(row["category"])
        per_stratum[category] += 1
        if per_stratum[category] <= 8:
            path = tmp_path / "admissions" / f"{row['queue_rank']:02d}.json"
            _dynamic_admission(path, row)
            admissions.append(path)
            continue
        evidence = tmp_path / "evidence" / f"{row['queue_rank']:02d}.log"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("source-only terminal failure\n", encoding="utf-8")
        path = tmp_path / "terminals" / f"{row['queue_rank']:02d}.json"
        build_terminal_disposition(
            path,
            queue_path=STAGING_QUEUE,
            queue_rank=int(row["queue_rank"]),
            stage="window_stage",
            reason_code="window-stage-failure",
            evidence_path=evidence,
            producer_commit="c" * 40,
        )
        terminals.append(path)
    return admissions, terminals


def test_complete_disposition_lock_is_balanced_and_outcome_blind(
    tmp_path: Path,
) -> None:
    admissions, terminals = _complete_dispositions(tmp_path)
    output = tmp_path / "cohort.json"
    cohort = build_dynamic_provider_cohort_lock(
        output,
        protocol_path=PROVIDER_PROTOCOL,
        source_evaluation_protocol_path=SOURCE_EVALUATION_PROTOCOL,
        queue_path=STAGING_QUEUE,
        processing_protocol_path=PROCESSING_PROTOCOL,
        runtime_amendment_path=RUNTIME_AMENDMENT,
        admission_paths=admissions,
        terminal_disposition_paths=terminals,
        provider_commit="a" * 40,
        source_processing_commit="b" * 40,
        cohort_lock_builder_commit="e" * 40,
    )

    assert len(cohort["source_cases"]) == 8
    assert len(cohort["sealed_target_cases"]) == 12
    assert cohort["stratum_counts"]["source"] == {
        "sheet": 3,
        "compact": 3,
        "complex": 2,
    }
    assert cohort["stratum_counts"]["target"] == {
        "sheet": 4,
        "compact": 4,
        "complex": 4,
    }
    assert cohort["counts"] == {
        "queued": 36,
        "admitted": 24,
        "source_rejected": 0,
        "technical_failure": 12,
        "selected_source": 8,
        "selected_target": 12,
    }
    assert cohort["information_boundary"]["provider_outcome_or_metric_read"] is False
    assert cohort["bindings"]["source_evaluation_protocol_file_sha256"] == (
        file_sha256(SOURCE_EVALUATION_PROTOCOL)
    )
    load_dynamic_provider_cohort_lock(output)
    source = dynamic_provider_case_record(
        cohort,
        object_id=cohort["source_cases"][0]["object_id"],
        episode_id=0,
        partition="source",
    )
    assert source == cohort["source_cases"][0]
    with pytest.raises(ValueError, match="requested cohort partition"):
        dynamic_provider_case_record(
            cohort,
            object_id=source["object_id"],
            episode_id=0,
            partition="target",
        )


def test_cohort_lock_rejects_incomplete_disposition_ledger(
    tmp_path: Path,
) -> None:
    admissions, terminals = _complete_dispositions(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        build_dynamic_provider_cohort_lock(
            tmp_path / "incomplete.json",
            protocol_path=PROVIDER_PROTOCOL,
            source_evaluation_protocol_path=SOURCE_EVALUATION_PROTOCOL,
            queue_path=STAGING_QUEUE,
            processing_protocol_path=PROCESSING_PROTOCOL,
            runtime_amendment_path=RUNTIME_AMENDMENT,
            admission_paths=admissions,
            terminal_disposition_paths=terminals[:-1],
            provider_commit="a" * 40,
            source_processing_commit="b" * 40,
            cohort_lock_builder_commit="e" * 40,
        )


def test_cohort_lock_tampering_is_detected(tmp_path: Path) -> None:
    admissions, terminals = _complete_dispositions(tmp_path)
    output = tmp_path / "cohort.json"
    cohort = build_dynamic_provider_cohort_lock(
        output,
        protocol_path=PROVIDER_PROTOCOL,
        source_evaluation_protocol_path=SOURCE_EVALUATION_PROTOCOL,
        queue_path=STAGING_QUEUE,
        processing_protocol_path=PROCESSING_PROTOCOL,
        runtime_amendment_path=RUNTIME_AMENDMENT,
        admission_paths=admissions,
        terminal_disposition_paths=terminals,
        provider_commit="a" * 40,
        source_processing_commit="b" * 40,
        cohort_lock_builder_commit="e" * 40,
    )
    cohort["source_cases"][0]["queue_rank"] = 36
    with pytest.raises(ValueError, match="partitions"):
        validate_dynamic_provider_cohort_lock(cohort)
