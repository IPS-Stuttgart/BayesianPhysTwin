from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_official_hub_stage1 import (
    HubFileRecord,
    OfficialHubStage1Lock,
    SelectedEpisode,
    build_official_hub_stage1_preflight,
    download_official_hub_stage1,
    load_official_hub_stage1_lock,
    materialize_official_hub_stage1_processing_view,
    validate_official_hub_stage1_download,
    validate_official_hub_stage1_preflight,
    validate_official_hub_stage1_processing_view,
    write_official_hub_stage1_manifest,
)


def _git_blob(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def _fake_fixture(
    *,
    episode_id: int = 6,
    object_id: str = "900-calibration-object",
    released_object: str | None = None,
) -> tuple[
    OfficialHubStage1Lock,
    dict[str, tuple[HubFileRecord, ...]],
    dict[str, bytes],
    dict[str, bytes],
]:
    metadata = {
        "object": object_id if released_object is None else released_object,
        "sam_prompt": [1, 2],
        "sequences": {
            str(index): {
                "action": f"action {index}",
                "bimanual": "yes" if index >= 5 else "no",
                "nonprehensile": "no",
            }
            for index in range(10)
        },
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    selected = SelectedEpisode(
        object_id=object_id,
        stratum="sheet",
        episode_id=episode_id,
        metadata_path=f"raw/{object_id}/metadata.json",
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
    )
    lock = OfficialHubStage1Lock(
        protocol_id="test-stage1",
        protocol_sha256="1" * 64,
        selection_artifact_sha256="2" * 64,
        selection_file_sha256="3" * 64,
        dataset_repository="brownu/deform360",
        dataset_revision="4" * 40,
        processing_repository="lhy0807/deform360",
        processing_revision="5" * 40,
        calibration=(selected,),
        confirmation_object_ids=("901-confirmation-object",),
    )
    payloads: dict[str, bytes] = {}

    def add(relative: str, payload: bytes | None = None, *, lfs: bool = False) -> None:
        path = f"raw/{object_id}/{relative}"
        value = payload if payload is not None else path.encode()
        payloads[path] = value
        records.append(
            HubFileRecord(
                path=path,
                size=len(value),
                blob_id=_git_blob(value),
                lfs_sha256=hashlib.sha256(value).hexdigest() if lfs else None,
            )
        )

    records: list[HubFileRecord] = []
    add("metadata.json", metadata_bytes)
    for name in ("intrinsics.npy", "extrinsics.npy", "dist.npy"):
        add(f"calibration_refined/{name}", lfs=True)
    for camera_index in range(3):
        camera = f"brics-odroid-{camera_index + 1:03d}_cam0"
        for index in range(10):
            stem = f"{camera}_{1000 + 100 * index}"
            add(f"{camera}/{stem}.mp4", lfs=True)
            add(f"{camera}/{stem}.txt")
    for sensor_name in ("l_left", "l_right", "r_left", "r_right"):
        sensor = f"brics-odroid_tactile{sensor_name}"
        add(f"{sensor}/median_900.npy", lfs=True)
        add(f"{sensor}/median_1450.npy", lfs=True)
        for index in range(10):
            stem = f"{sensor}_{1000 + 100 * index}"
            add(f"{sensor}/{stem}.npy", lfs=True)
            add(f"{sensor}/{stem}.txt")
            add(f"{sensor}/{stem}.wav", lfs=True)
    return (
        lock,
        {object_id: tuple(records)},
        {object_id: metadata_bytes},
        payloads,
    )


def test_repository_stage1_lock_is_self_consistent() -> None:
    repository = Path(__file__).resolve().parents[1]
    lock = load_official_hub_stage1_lock(
        repository,
        "protocols/deform360_official_hub_visuotactile_v1.json",
        "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json",
    )

    assert len(lock.calibration) == 10
    assert len(lock.confirmation_object_ids) == 12
    assert not (
        {item.object_id for item in lock.calibration}
        & set(lock.confirmation_object_ids)
    )


def test_stage1_provenance_correction_preserves_actual_information_order() -> None:
    repository = Path(__file__).resolve().parents[1]
    path = repository / (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_stage1_provenance.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = dict(value)
    artifact_id = canonical.pop("artifact_id")

    assert artifact_id == hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert value["corrected_access_boundary"] == {
        "calibration_payloads_opened": True,
        "calibration_policy_fit": False,
        "calibration_scores_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
    }
    assert (
        value["information_order"]["finite_group_amendment_role"]
        == "post-calibration-payload-pre-calibration-score"
    )
    assert value["finite_group_design_unchanged"] is True


def test_preflight_selects_exact_episode_and_preceding_tactile_baseline() -> None:
    lock, trees, metadata, _ = _fake_fixture(episode_id=6)

    result = build_official_hub_stage1_preflight(
        lock,
        tree_by_object=trees,
        metadata_bytes_by_object=metadata,
    )

    validate_official_hub_stage1_preflight(result, lock=lock)
    row = result["objects"][0]
    assert row["camera_count"] == 3
    assert row["tactile_sensor_count"] == 4
    assert set(row["tactile_baselines"].values()) == {"median_1450.npy"}
    paths = {item["path"] for item in row["files"]}
    assert len(paths) == 22
    assert all(
        "_1600." in path
        or "median_1450" in path
        or "calibration_refined" in path
        or path.endswith("metadata.json")
        for path in paths
    )
    assert not any(path.endswith(".wav") for path in paths)
    assert result["information_boundary"]["confirmation_payload_opened"] is False
    assert result["physical_backend_contract"]["minimum_node_count"] == 128


def test_preflight_rejects_confirmation_tree_and_invalid_enum() -> None:
    lock, trees, metadata, _ = _fake_fixture()
    trees["901-confirmation-object"] = ()
    with pytest.raises(ValueError, match="exactly the locked calibration"):
        build_official_hub_stage1_preflight(
            lock,
            tree_by_object=trees,
            metadata_bytes_by_object=metadata,
        )


def test_preflight_accepts_only_declared_metadata_aliases() -> None:
    lock, trees, metadata, _ = _fake_fixture(
        object_id="026-sock-cloth",
        released_object="026-sock",
    )
    result = build_official_hub_stage1_preflight(
        lock,
        tree_by_object=trees,
        metadata_bytes_by_object=metadata,
    )
    assert result["objects"][0]["metadata"]["released_object"] == "026-sock"

    lock, trees, metadata, _ = _fake_fixture(released_object="silently-renamed")
    with pytest.raises(ValueError, match="metadata object identity changed"):
        build_official_hub_stage1_preflight(
            lock,
            tree_by_object=trees,
            metadata_bytes_by_object=metadata,
        )

    lock, trees, metadata, _ = _fake_fixture()
    decoded = json.loads(next(iter(metadata.values())))
    decoded["sequences"]["6"]["bimanual"] = "yess"
    bad = json.dumps(decoded, sort_keys=True).encode()
    selected = replace(
        lock.calibration[0], metadata_sha256=hashlib.sha256(bad).hexdigest()
    )
    bad_lock = replace(lock, calibration=(selected,))
    with pytest.raises(ValueError, match="invalid bimanual enum"):
        build_official_hub_stage1_preflight(
            bad_lock,
            tree_by_object=trees,
            metadata_bytes_by_object={selected.object_id: bad},
        )


def test_preflight_records_invalid_enum_only_outside_selected_episode() -> None:
    lock, trees, metadata, _ = _fake_fixture()
    decoded = json.loads(next(iter(metadata.values())))
    del decoded["sequences"]["1"]["nonprehensile"]
    decoded["sequences"]["1"]["nonprehensisle"] = "no"
    anomalous = json.dumps(decoded, sort_keys=True).encode()
    selected = replace(
        lock.calibration[0],
        metadata_sha256=hashlib.sha256(anomalous).hexdigest(),
    )
    anomalous_lock = replace(lock, calibration=(selected,))

    result = build_official_hub_stage1_preflight(
        anomalous_lock,
        tree_by_object=trees,
        metadata_bytes_by_object={selected.object_id: anomalous},
    )

    assert result["objects"][0]["metadata"]["nonselected_sequence_anomalies"] == [
        {"episode_id": 1, "field": "nonprehensile", "released_value": None}
    ]


def test_preflight_rejects_missing_sidecar_and_missing_preceding_baseline() -> None:
    lock, trees, metadata, _ = _fake_fixture()
    object_id = lock.calibration[0].object_id
    without_sidecar = tuple(
        item
        for item in trees[object_id]
        if not item.path.endswith("brics-odroid-001_cam0_1600.txt")
    )
    with pytest.raises(ValueError, match="missing timestamp"):
        build_official_hub_stage1_preflight(
            lock,
            tree_by_object={object_id: without_sidecar},
            metadata_bytes_by_object=metadata,
        )

    early_lock = replace(
        lock, calibration=(replace(lock.calibration[0], episode_id=0),)
    )
    without_early_baseline = tuple(
        item for item in trees[object_id] if "median_900.npy" not in item.path
    )
    with pytest.raises(ValueError, match="no preceding baseline"):
        build_official_hub_stage1_preflight(
            early_lock,
            tree_by_object={object_id: without_early_baseline},
            metadata_bytes_by_object=metadata,
        )


def test_preflight_records_harmless_camera_orphan_timestamp() -> None:
    lock, trees, metadata, _ = _fake_fixture()
    object_id = lock.calibration[0].object_id
    camera = "brics-odroid-001_cam0"
    orphan_path = f"raw/{object_id}/{camera}/{camera}_9999.txt"
    orphan_payload = b"orphan timestamp"
    orphan = HubFileRecord(
        path=orphan_path,
        size=len(orphan_payload),
        blob_id=_git_blob(orphan_payload),
    )

    result = build_official_hub_stage1_preflight(
        lock,
        tree_by_object={object_id: (*trees[object_id], orphan)},
        metadata_bytes_by_object=metadata,
    )

    ignored = result["objects"][0]["camera_ignored_orphan_timestamp_stems"]
    assert ignored == {camera: [f"{camera}_9999"]}
    assert orphan_path not in {item["path"] for item in result["objects"][0]["files"]}


def test_download_verifies_content_and_refuses_unplanned_file(tmp_path: Path) -> None:
    lock, trees, metadata, payloads = _fake_fixture()
    preflight = build_official_hub_stage1_preflight(
        lock,
        tree_by_object=trees,
        metadata_bytes_by_object=metadata,
    )
    cache = tmp_path / "cache"
    for path, payload in payloads.items():
        destination = cache / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    result = download_official_hub_stage1(
        preflight,
        tmp_path / "download",
        lock=lock,
        hub_download=lambda path: cache / path,
        max_workers=2,
    )

    assert result["file_count"] == 22
    assert not any(item["reused"] for item in result["files"])
    repeated = download_official_hub_stage1(
        preflight,
        tmp_path / "download",
        lock=lock,
        hub_download=lambda path: cache / path,
        max_workers=2,
    )
    assert all(item["reused"] for item in repeated["files"])

    unauthorized = tmp_path / "download" / "raw" / "other" / "future.npz"
    unauthorized.parent.mkdir(parents=True)
    unauthorized.write_bytes(b"future")
    with pytest.raises(ValueError, match="unauthorized"):
        download_official_hub_stage1(
            preflight,
            tmp_path / "download",
            lock=lock,
            hub_download=lambda path: cache / path,
        )


def test_download_manifest_validation_rejects_inventory_tampering(
    tmp_path: Path,
) -> None:
    lock, trees, metadata, payloads = _fake_fixture()
    preflight = build_official_hub_stage1_preflight(
        lock,
        tree_by_object=trees,
        metadata_bytes_by_object=metadata,
    )
    cache = tmp_path / "cache"
    for path, payload in payloads.items():
        destination = cache / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    download = download_official_hub_stage1(
        preflight,
        tmp_path / "download",
        lock=lock,
        hub_download=lambda path: cache / path,
    )

    digests = validate_official_hub_stage1_download(
        download,
        preflight=preflight,
        lock=lock,
    )
    assert len(digests) == 22

    tampered = dict(download)
    tampered["file_count"] = 21
    with pytest.raises(ValueError, match="download digest changed"):
        validate_official_hub_stage1_download(
            tampered,
            preflight=preflight,
            lock=lock,
        )


def test_processing_view_maps_selected_episode_without_changing_payload(
    tmp_path: Path,
) -> None:
    lock, trees, metadata, payloads = _fake_fixture(episode_id=6)
    preflight = build_official_hub_stage1_preflight(
        lock,
        tree_by_object=trees,
        metadata_bytes_by_object=metadata,
    )
    cache = tmp_path / "cache"
    for path, payload in payloads.items():
        destination = cache / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    payload_root = tmp_path / "payload"
    download = download_official_hub_stage1(
        preflight,
        payload_root,
        lock=lock,
        hub_download=lambda path: cache / path,
    )
    view_root = tmp_path / "processing-view"

    result = materialize_official_hub_stage1_processing_view(
        preflight,
        download,
        payload_root,
        view_root,
        lock=lock,
    )

    object_id = lock.calibration[0].object_id
    source_metadata_path = payload_root / "raw" / object_id / "metadata.json"
    view_metadata_path = view_root / "raw" / object_id / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text())
    view_metadata = json.loads(view_metadata_path.read_text())
    assert set(source_metadata["sequences"]) == {str(index) for index in range(10)}
    assert view_metadata["sequences"] == {"0": source_metadata["sequences"]["6"]}
    assert result["objects"][0]["source_episode_id"] == 6
    assert result["objects"][0]["processing_episode_index"] == 0
    assert result["linked_file_count"] == 21
    linked_video = next((view_root / "raw" / object_id).rglob("*.mp4"))
    assert linked_video.is_symlink()
    assert linked_video.resolve().is_relative_to(payload_root.resolve())
    stored_manifest = json.loads(
        (view_root / "stage1_processing_view.json").read_text()
    )
    assert stored_manifest == result
    validate_official_hub_stage1_processing_view(
        result,
        preflight=preflight,
        download=download,
        view_root=view_root,
        payload_root=payload_root,
        lock=lock,
    )

    with pytest.raises(ValueError, match="processing view already exists"):
        materialize_official_hub_stage1_processing_view(
            preflight,
            download,
            payload_root,
            view_root,
            lock=lock,
        )

    linked_payload = linked_video.read_bytes()
    linked_video.unlink()
    linked_video.write_bytes(linked_payload)
    with pytest.raises(ValueError, match="view media is not a link"):
        validate_official_hub_stage1_processing_view(
            result,
            preflight=preflight,
            download=download,
            view_root=view_root,
            payload_root=payload_root,
            lock=lock,
        )


def test_manifest_writer_is_atomic_and_newline_terminated(tmp_path: Path) -> None:
    lock, trees, metadata, _ = _fake_fixture()
    preflight = build_official_hub_stage1_preflight(
        lock,
        tree_by_object=trees,
        metadata_bytes_by_object=metadata,
    )
    destination = tmp_path / "manifest.json"

    write_official_hub_stage1_manifest(destination, preflight)

    assert json.loads(destination.read_text()) == preflight
    assert destination.read_text().endswith("\n")
