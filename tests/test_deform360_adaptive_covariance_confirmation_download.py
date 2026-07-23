from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import bayesian_phystwin.cli.deform360_adaptive_covariance_confirmation_download as download_cli
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_download import (
    build_confirmation_download_manifest,
    confirmation_download_plan,
    download_confirmation_panel_by_object,
    validate_confirmation_download_root,
    write_confirmation_download_manifest,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_download_manifest,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    build_confirmation_cohort_lock,
)


H1 = "1" * 40
H2 = "2" * 40


def test_download_cli_requires_exact_h2_repository_binding() -> None:
    parser = download_cli.build_parser()
    required = {
        action.dest for action in parser._actions if getattr(action, "required", False)
    }
    assert {
        "adapter_repo",
        "lock",
        "h2_commit",
        "expected_h1",
        "output_root",
        "manifest",
    } <= required


def _write_lock(path: Path) -> dict[str, object]:
    payload = build_confirmation_cohort_lock(H1)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _metadata_bytes(object_id: str) -> bytes:
    return json.dumps(
        {
            "object": f"released label for {object_id}",
            "sequences": {str(index): {} for index in range(10)},
        },
        sort_keys=True,
    ).encode("utf-8")


def _materialize_panel(root: Path, plan: object) -> None:
    for object_id in plan.object_ids:
        destination = root / "raw" / object_id / "metadata.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(_metadata_bytes(object_id))


def _git_blob_id(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _remote_inventory(
    root: Path, plan: object
) -> dict[str, list[dict[str, str | None]]]:
    result: dict[str, list[dict[str, str | None]]] = {}
    for object_id in plan.object_ids:
        records: list[dict[str, str | None]] = []
        object_root = root / "raw" / object_id
        for path in sorted(object_root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                payload = path.read_bytes()
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "blob_id": _git_blob_id(payload),
                        "lfs_sha256": None,
                    }
                )
        result[object_id] = records
    return result


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _value_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _validate_object(
    manifest_path: Path,
    manifest: dict[str, object],
    root: Path,
    object_id: str,
    episode_id: int,
) -> dict[str, object]:
    return validate_confirmation_download_manifest(
        manifest_path,
        protocol_config_sha256=str(manifest["cohort_lock_artifact_sha256"]),
        object_id=object_id,
        episode_id=episode_id,
        metadata_path=root / "raw" / object_id / "metadata.json",
        expected_h1=H1,
        expected_h2=H2,
    )


def test_plan_and_manifest_are_exactly_h1_h2_and_case_bound(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    lock = _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    assert len(plan.object_ids) == 17
    assert len(plan.selected_episodes_by_object) == 17
    assert all(len(episodes) == 2 for _, episodes in plan.selected_episodes_by_object)

    root = tmp_path / "download"
    _materialize_panel(root, plan)
    inventory = _remote_inventory(root, plan)
    manifest = build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=inventory,
    )
    assert manifest["schema_version"] == 2
    assert manifest["implementation_commit_h1"] == H1
    assert manifest["cohort_lock_commit_h2"] == H2
    assert manifest["cohort_lock_artifact_sha256"] == lock["artifact_sha256"]
    assert manifest["object_count"] == 17
    assert manifest["information_boundary"]["target_or_outcome_opened"] is False
    assert all(
        set(record)
        == {
            "path",
            "remote_blob_id",
            "remote_lfs_sha256",
            "size_bytes",
            "sha256",
        }
        for row in manifest["objects"]
        for record in row["files"]
    )
    canonical = deepcopy(manifest)
    declared = canonical.pop("artifact_sha256")
    assert (
        declared
        == hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )


def test_download_is_object_scoped_and_excludes_audio(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    downloaded: list[str] = []

    def list_repo_tree(**kwargs: object) -> list[SimpleNamespace]:
        object_id = str(kwargs["path_in_repo"]).split("/", 1)[1]
        return [
            SimpleNamespace(
                path=f"raw/{object_id}/metadata.json",
                blob_id=_git_blob_id(_metadata_bytes(object_id)),
            ),
            SimpleNamespace(
                path=f"raw/{object_id}/audio.wav",
                blob_id="a" * 40,
            ),
            SimpleNamespace(
                path=f"raw/{object_id}/camera/video.mp4",
                blob_id="c" * 40,
                lfs=SimpleNamespace(
                    sha256=hashlib.sha256(f"video-{object_id}".encode()).hexdigest()
                ),
            ),
            SimpleNamespace(
                path=f"raw/{object_id}/camera",
                blob_id=None,
                tree_id="b" * 40,
            ),
        ]

    root = tmp_path / "download"

    def hub_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        downloaded.append(filename)
        object_id = filename.split("/")[1]
        destination = root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "metadata.json":
            destination.write_bytes(_metadata_bytes(object_id))
        else:
            destination.write_bytes(f"video-{object_id}".encode())
        return str(destination)

    manifest = download_confirmation_panel_by_object(
        lock_path,
        H2,
        root,
        max_workers=2,
        object_delay_seconds=0.0,
        list_repo_tree=list_repo_tree,
        hub_download=hub_download,
        expected_h1=H1,
    )
    assert len(downloaded) == 34
    assert all(not path.endswith("/audio.wav") for path in downloaded)
    assert manifest["object_count"] == 17
    assert tuple(row["object_id"] for row in manifest["objects"]) == plan.object_ids
    assert any(
        record["remote_lfs_sha256"] is not None
        for row in manifest["objects"]
        for record in row["files"]
    )
    first_object, first_episodes = plan.selected_episodes_by_object[0]
    manifest_path = tmp_path / "download-manifest.json"
    _write_manifest(manifest_path, manifest)
    _validate_object(
        manifest_path,
        manifest,
        root,
        first_object,
        first_episodes[0],
    )


def test_download_rejects_repo_tree_escape(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)

    def list_repo_tree(**kwargs: object) -> list[SimpleNamespace]:
        object_id = str(kwargs["path_in_repo"]).split("/", 1)[1]
        return [
            SimpleNamespace(
                path=f"raw/{object_id}/../../other/metadata.json",
                blob_id="evil",
            )
        ]

    with pytest.raises(ValueError, match="escaped"):
        download_confirmation_panel_by_object(
            lock_path,
            H2,
            tmp_path / "download",
            max_workers=1,
            object_delay_seconds=0.0,
            list_repo_tree=list_repo_tree,
            hub_download=lambda **_: "",
            expected_h1=H1,
        )


def test_root_rejects_unlocked_objects_and_manifest_is_noreplace(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    (root / "raw" / "999-unlocked").mkdir()
    with pytest.raises(ValueError, match="unlocked objects"):
        validate_confirmation_download_root(root, plan=plan, require_complete=True)

    (root / "raw" / "999-unlocked").rmdir()
    payload = build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=_remote_inventory(root, plan),
    )
    destination = tmp_path / "manifest.json"
    write_confirmation_download_manifest(destination, payload)
    with pytest.raises(ValueError, match="already exists"):
        write_confirmation_download_manifest(destination, payload)


def test_manifest_builder_requires_and_replays_exact_remote_inventory(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    inventory = _remote_inventory(root, plan)

    with pytest.raises(TypeError):
        build_confirmation_download_manifest(root, plan=plan)

    first = plan.object_ids[0]
    missing = deepcopy(inventory)
    missing[first] = []
    with pytest.raises(ValueError, match="inventory is empty"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=missing,
        )

    wrong_path = deepcopy(inventory)
    wrong_path[first][0]["path"] = f"raw/{plan.object_ids[1]}/metadata.json"
    with pytest.raises(ValueError, match="escaped its object"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=wrong_path,
        )

    wrong_blob = deepcopy(inventory)
    wrong_blob[first][0]["blob_id"] = "not-a-blob"
    with pytest.raises(ValueError, match="record is invalid"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=wrong_blob,
        )


def test_manifest_builder_cryptographically_binds_lfs_content(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    inventory = _remote_inventory(root, plan)
    first = plan.object_ids[0]
    metadata = root / "raw" / first / "metadata.json"
    inventory[first][0]["blob_id"] = "d" * 40
    inventory[first][0]["lfs_sha256"] = hashlib.sha256(
        metadata.read_bytes()
    ).hexdigest()
    build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=inventory,
    )

    inventory[first][0]["lfs_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="LFS content differs"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=inventory,
        )


def test_manifest_and_replay_reject_extra_audio_symlink_special_and_missing(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    inventory = _remote_inventory(root, plan)
    first = plan.object_ids[0]
    first_metadata = root / "raw" / first / "metadata.json"

    extra = first_metadata.parent / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra files"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=inventory,
        )
    extra.unlink()

    audio = first_metadata.parent / "surprise.wav"
    audio.write_bytes(b"audio")
    with pytest.raises(ValueError, match="forbidden audio"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=inventory,
        )
    audio.unlink()

    target = tmp_path / "metadata-target.json"
    target.write_bytes(first_metadata.read_bytes())
    first_metadata.unlink()
    first_metadata.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=inventory,
        )
    first_metadata.unlink()
    first_metadata.write_bytes(target.read_bytes())

    fifo = first_metadata.parent / "special"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="special"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=inventory,
        )
    fifo.unlink()

    first_metadata.unlink()
    with pytest.raises(ValueError, match="metadata is missing"):
        build_confirmation_download_manifest(
            root,
            plan=plan,
            remote_inventory_by_object=inventory,
        )


def test_external_replay_rejects_mutation_swaps_missing_extra_and_symlink(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    for index, object_id in enumerate(plan.object_ids):
        media = root / "raw" / object_id / "camera.mp4"
        media.write_bytes(f"media-{index}".encode())
    manifest = build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=_remote_inventory(root, plan),
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    first, episodes = plan.selected_episodes_by_object[0]
    second = plan.object_ids[1]
    first_media = root / "raw" / first / "camera.mp4"
    second_media = root / "raw" / second / "camera.mp4"
    first_payload = first_media.read_bytes()

    first_media.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="content changed"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    first_media.write_bytes(first_payload)

    second_payload = second_media.read_bytes()
    first_media.write_bytes(second_payload)
    second_media.write_bytes(first_payload)
    with pytest.raises(ValueError, match="content changed"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    first_media.write_bytes(first_payload)
    second_media.write_bytes(second_payload)

    first_metadata = root / "raw" / first / "metadata.json"
    metadata_payload = first_metadata.read_bytes()
    first_metadata.write_bytes(first_payload)
    first_media.write_bytes(metadata_payload)
    with pytest.raises(ValueError, match="content changed"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    first_metadata.write_bytes(metadata_payload)
    first_media.write_bytes(first_payload)

    extra = first_media.parent / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra files"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    extra.unlink()

    audio = first_media.parent / "unexpected.flac"
    audio.write_bytes(b"audio")
    with pytest.raises(ValueError, match="forbidden audio"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    audio.unlink()

    special = first_media.parent / "fifo"
    os.mkfifo(special)
    with pytest.raises(ValueError, match="special"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    special.unlink()

    first_media.unlink()
    with pytest.raises(ValueError, match="missing"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])
    first_media.write_bytes(first_payload)

    target = tmp_path / "media-target"
    target.write_bytes(first_payload)
    first_media.unlink()
    first_media.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("h1", "incompatible"),
        ("h2", "incompatible"),
        ("extra", "incompatible"),
        ("directory_identity", "content boundary changed"),
        ("released_label_identity", "content boundary changed"),
    ],
)
def test_external_replay_rejects_rehashed_authorization_envelope_mutation(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    manifest = build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=_remote_inventory(root, plan),
    )
    first, episodes = plan.selected_episodes_by_object[0]
    if mutation == "h1":
        manifest["implementation_commit_h1"] = "a" * 40
    elif mutation == "h2":
        manifest["cohort_lock_commit_h2"] = "b" * 40
    elif mutation == "extra":
        manifest["unauthorized"] = True
    elif mutation == "directory_identity":
        manifest["information_boundary"]["directory_id_used_as_identity"] = False
    else:
        manifest["information_boundary"][
            "released_metadata_label_used_for_identity"
        ] = True
    manifest["artifact_sha256"] = _value_sha256(
        {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match=expected_message):
        _validate_object(manifest_path, manifest, root, first, episodes[0])


def test_external_replay_binds_remote_blob_to_its_path(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    plan = confirmation_download_plan(lock_path, H2, expected_h1=H1)
    root = tmp_path / "download"
    _materialize_panel(root, plan)
    first, episodes = plan.selected_episodes_by_object[0]
    second_file = root / "raw" / first / "video.mp4"
    second_file.write_bytes(b"video")
    manifest = build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=_remote_inventory(root, plan),
    )
    first_row = next(row for row in manifest["objects"] if row["object_id"] == first)
    first_row["files"][0]["remote_blob_id"], first_row["files"][1]["remote_blob_id"] = (
        first_row["files"][1]["remote_blob_id"],
        first_row["files"][0]["remote_blob_id"],
    )
    manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "artifact_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="remote blob/path binding changed"):
        _validate_object(manifest_path, manifest, root, first, episodes[0])

    fresh = build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=_remote_inventory(root, plan),
    )
    first_row = next(row for row in fresh["objects"] if row["object_id"] == first)
    target_record = next(
        record for record in first_row["files"] if record["remote_lfs_sha256"] is None
    )
    target_record["remote_blob_id"] = "f" * 40
    remote_binding = [
        {
            "path": record["path"],
            "blob_id": record["remote_blob_id"],
            "lfs_sha256": record["remote_lfs_sha256"],
        }
        for record in first_row["files"]
    ]
    first_row["remote_inventory_sha256"] = _value_sha256(remote_binding)
    first_row["content_inventory_sha256"] = _value_sha256(first_row["files"])
    fresh["artifact_sha256"] = _value_sha256(
        {key: value for key, value in fresh.items() if key != "artifact_sha256"}
    )
    _write_manifest(manifest_path, fresh)
    with pytest.raises(ValueError, match="Git blob differs from pinned remote"):
        _validate_object(manifest_path, fresh, root, first, episodes[0])
