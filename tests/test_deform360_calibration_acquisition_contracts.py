# ruff: noqa: F403, F405
from deform360_calibration_acquisition_test_support import *

@pytest.mark.skipif(not STAGE0.is_file(), reason="requires committed protocol locks")
def test_committed_locks_build_exact_calibration_only_plan() -> None:
    plan = build_calibration_acquisition_plan(
        stage0_selection_path=STAGE0,
        visual_provider_lock_path=VISUAL_LOCK,
        implementation_revision="c" * 40,
        protocol_path=PROTOCOL,
    )
    observed = {
        (unit.object_id, unit.episode_id, unit.stratum)
        for unit in plan.calibration_units
    }
    assert observed == {
        ("167-glove-gray-cloth", 0, "sheet"),
        ("198-kneepad-cloth", 2, "sheet"),
        ("026-sock-cloth", 7, "sheet"),
        ("031-cotton-cloth", 0, "sheet"),
        ("036-napkin-cloth", 9, "sheet"),
        ("153-cake", 5, "volumetric"),
        ("152-slime", 8, "volumetric"),
        ("186-monster", 6, "volumetric"),
        ("058-roll-napkin", 1, "volumetric"),
        ("193-frog", 7, "volumetric"),
    }
    assert len(plan.forbidden_confirmation_object_ids) == 12
    assert plan.calibration_payloads_opened is False
    assert plan.confirmation_payloads_opened is False
    assert plan.target_outcomes_used is False


def test_plan_and_case_records_are_content_addressed() -> None:
    plan = _plan()
    assert Deform360CalibrationAcquisitionPlanV1.from_mapping(
        plan.to_record()
    ) == plan

    mutated_plan = plan.to_record()
    mutated_plan["dataset_revision"] = "d" * 40
    with pytest.raises(ValueError, match="plan_id"):
        Deform360CalibrationAcquisitionPlanV1.from_mapping(mutated_plan)

    prepared = _prepared_case(plan, plan.calibration_units[0])
    assert Deform360CalibrationAcquisitionCaseV1.from_mapping(
        prepared.to_record()
    ) == prepared

    failed = _failed_case(plan, plan.calibration_units[1])
    assert failed.output_artifacts == {}
    assert Deform360CalibrationAcquisitionCaseV1.from_mapping(
        failed.to_record()
    ) == failed


def test_selective_paths_open_exactly_one_episode_and_no_audio_or_geometry() -> None:
    object_id = "001-rope"
    prefix = f"raw/{object_id}"
    paths = [
        f"{prefix}/metadata.json",
        f"{prefix}/calibration_refined/dist.npy",
        f"{prefix}/calibration_refined/extrinsics.npy",
        f"{prefix}/calibration_refined/intrinsics.npy",
        f"{prefix}/geometry/points.ply",
        f"{prefix}/reconstruction/mesh.ply",
    ]
    for camera in ("brics-odroid-001_cam0", "brics-odroid-002_cam0"):
        paths.extend(
            [
                f"{prefix}/{camera}/clip_000.mp4",
                f"{prefix}/{camera}/clip_000.txt",
                f"{prefix}/{camera}/clip_001.mp4",
                f"{prefix}/{camera}/clip_001.txt",
            ]
        )
    sensor = "brics-odroid_tactile0"
    paths.extend(
        [
            f"{prefix}/{sensor}/median_reference.npy",
            f"{prefix}/{sensor}/clip_000.npy",
            f"{prefix}/{sensor}/clip_000.txt",
            f"{prefix}/{sensor}/clip_001.npy",
            f"{prefix}/{sensor}/clip_001.txt",
            f"{prefix}/{sensor}/clip_001.wav",
        ]
    )

    selected = select_calibration_object_paths(
        paths,
        object_id=object_id,
        episode_id=1,
    )

    assert f"{prefix}/metadata.json" in selected
    assert all("clip_000" not in path for path in selected)
    assert any("clip_001.mp4" in path for path in selected)
    assert any("clip_001.npy" in path for path in selected)
    assert any("median_reference.npy" in path for path in selected)
    assert all(not path.endswith((".wav", ".flac")) for path in selected)
    assert all("geometry/" not in path for path in selected)
    assert all("reconstruction/" not in path for path in selected)

    without_calibration = [
        path for path in paths if not path.endswith("calibration_refined/dist.npy")
    ]
    with pytest.raises(ValueError, match="required source paths"):
        select_calibration_object_paths(
            without_calibration,
            object_id=object_id,
            episode_id=1,
        )


def test_payload_allowlist_and_download_manifest_are_content_addressed() -> None:
    module = _script_module()
    plan = _plan()
    selected = {
        unit.object_id: (unit.metadata_path,) for unit in plan.calibration_units
    }
    allowlist = module._payload_allowlist(plan, selected)
    allowlist_descriptor = dict(allowlist)
    allowlist_id = allowlist_descriptor.pop("allowlist_id")
    assert allowlist_id == content_id(allowlist_descriptor)
    assert allowlist["calibration_payloads_opened"] is False
    assert allowlist["confirmation_payloads_opened"] is False

    downloads = [
        {
            "object_id": unit.object_id,
            "episode_id": unit.episode_id,
            "selected_path_count": 1,
            "selected_bytes": 2,
            "files": [
                {
                    "path": unit.metadata_path,
                    "bytes": 2,
                    "sha256": unit.metadata_sha256,
                }
            ],
        }
        for unit in reversed(plan.calibration_units)
    ]
    manifest = module._download_manifest(
        plan,
        payload_allowlist_id=str(allowlist_id),
        downloads=downloads,
    )
    descriptor = dict(manifest)
    manifest_id = descriptor.pop("manifest_id")
    assert manifest_id == content_id(descriptor)
    assert manifest["total_file_count"] == 10
    assert manifest["total_bytes"] == 20
    assert [item["object_id"] for item in manifest["units"]] == [
        unit.object_id for unit in plan.calibration_units
    ]
    assert manifest["confirmation_payloads_opened"] is False
    assert manifest["target_outcomes_used"] is False

    with pytest.raises(ValueError, match="locked cohort"):
        module._download_manifest(
            plan,
            payload_allowlist_id=str(allowlist_id),
            downloads=downloads[:-1],
        )

    escaped = dict(selected)
    escaped[plan.calibration_units[0].object_id] = (
        f"raw/{plan.calibration_units[0].object_id}/../forbidden.mp4",
    )
    with pytest.raises(ValueError, match="escapes"):
        module._payload_allowlist(plan, escaped)


def _materialize_allowlist(
    root: Path,
    plan: Deform360CalibrationAcquisitionPlanV1,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for unit in plan.calibration_units:
        path = root / unit.metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        result[unit.object_id] = (unit.metadata_path,)
    return result


def test_download_root_rejects_confirmation_extra_payload_and_symlink(
    tmp_path: Path,
) -> None:
    plan = _plan()
    (tmp_path / "raw").mkdir()
    allowlist = _materialize_allowlist(tmp_path, plan)
    assert len(
        validate_calibration_download_root(
            tmp_path,
            plan,
            require_complete=True,
            expected_paths_by_object=allowlist,
        )
    ) == 10

    extra = tmp_path / "raw" / plan.calibration_units[0].object_id / "other.mp4"
    extra.write_bytes(b"not selected")
    with pytest.raises(ValueError, match="unselected payloads"):
        validate_calibration_download_root(
            tmp_path,
            plan,
            require_complete=True,
            expected_paths_by_object=allowlist,
        )
    extra.unlink()

    confirmation = tmp_path / "raw" / plan.forbidden_confirmation_object_ids[0]
    confirmation.mkdir()
    with pytest.raises(ValueError, match="confirmation-object"):
        validate_calibration_download_root(
            tmp_path,
            plan,
            require_complete=False,
            expected_paths_by_object=allowlist,
        )
    confirmation.rmdir()

    processed_confirmation = (
        tmp_path / "processed" / plan.forbidden_confirmation_object_ids[0]
    )
    processed_confirmation.mkdir(parents=True)
    with pytest.raises(ValueError, match="confirmation-object"):
        validate_calibration_download_root(
            tmp_path,
            plan,
            require_complete=False,
            expected_paths_by_object=allowlist,
        )
    processed_confirmation.rmdir()

    unit = plan.calibration_units[0]
    wrong_episode = (
        tmp_path
        / "processed"
        / unit.object_id
        / f"source_episode_{unit.episode_id + 1:04d}"
    )
    wrong_episode.mkdir(parents=True)
    with pytest.raises(ValueError, match="unselected episode entries"):
        validate_calibration_download_root(
            tmp_path,
            plan,
            require_complete=True,
            expected_paths_by_object=allowlist,
        )
    wrong_episode.rmdir()
    wrong_episode.parent.rmdir()

    target = tmp_path / plan.calibration_units[0].metadata_path
    link = target.with_name("metadata-link.json")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink"):
        validate_calibration_download_root(
            tmp_path,
            plan,
            require_complete=True,
            expected_paths_by_object=allowlist,
        )

