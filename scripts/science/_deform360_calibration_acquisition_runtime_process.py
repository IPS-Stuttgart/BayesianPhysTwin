# ruff: noqa: E402, F403, F405
"""Internal runtime slice for Deform360 calibration acquisition."""

from __future__ import annotations

from _deform360_calibration_acquisition_runtime_common import *

def _require_bayesian_phystwin_import(checkout: Path) -> None:
    import bayesian_phystwin.deform360_calibration_acquisition as acquisition

    expected = (
        checkout
        / "src"
        / "bayesian_phystwin"
        / "deform360_calibration_acquisition.py"
    ).resolve()
    observed = Path(acquisition.__file__).resolve()
    if observed != expected:
        raise ValueError(
            "imported BayesianPhysTwin acquisition module does not come from "
            "the exact checkout"
        )


def _require_deform360_import(checkout: Path) -> None:
    try:
        import deform360
        from deform360.processing import robot_stage
    except ImportError as error:
        raise RuntimeError(
            "calibration acquisition requires the pinned Deform360 package"
        ) from error
    expected = checkout.resolve()
    package_root = Path(deform360.__file__).resolve().parents[1]
    robot_root = Path(robot_stage.__file__).resolve().parents[2]
    if package_root != expected or robot_root != expected:
        raise ValueError(
            "imported Deform360 package does not come from the locked checkout"
        )


def _raw_artifacts_from_download(unit_record: Mapping[str, Any]) -> dict[str, str]:
    files = unit_record.get("files")
    if not isinstance(files, list):
        raise ValueError("download unit has no file inventory")
    result: dict[str, str] = {}
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("download file record must be an object")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("download file record lacks path or sha256")
        result[path] = digest
    if not result:
        raise ValueError("download file inventory is empty")
    return result


def _timeline_details(episode_dir: Path) -> tuple[int, str]:
    manifest = _strict_json(episode_dir / "alignment.json")
    frame_count = manifest.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise ValueError("episode alignment has no positive frame_count")
    camera_dirs = sorted(
        path
        for path in episode_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(_CAMERA_PREFIX)
        and (path / "aligned_timestamps.txt").is_file()
    )
    if not camera_dirs:
        raise ValueError("episode has no aligned camera timeline")
    return frame_count, file_sha256(camera_dirs[0] / "aligned_timestamps.txt")


def _process_case(
    *,
    plan_id: str,
    object_id: str,
    episode_id: int,
    stratum: Literal["sheet", "volumetric"],
    data_root: Path,
    raw_artifacts: Mapping[str, str],
    failure_root: Path,
) -> Deform360CalibrationAcquisitionCaseV1:
    raw_object = data_root / "raw" / object_id
    processed_object = (
        data_root
        / "processed"
        / object_id
        / f"source_episode_{episode_id:04d}"
    )
    metadata_path = raw_object / "metadata.json"
    stage = "metadata"
    before_outputs = set(_all_files(processed_object))
    try:
        metadata = _strict_json(metadata_path)
        bimanual = _metadata_bimanual(metadata, episode_id)

        try:
            from deform360 import tactile, undistort
            from deform360.processing import robot_stage
        except ImportError as error:
            raise RuntimeError(
                "run acquisition inside the pinned official Deform360 environment"
            ) from error

        stage = "undistort"
        episode_dir = undistort.undistort_episode(
            raw_object,
            processed_object,
            _LOCAL_PROCESSING_EPISODE_INDEX,
            cameras=None,
            overwrite=False,
            rebuild_timeline=False,
        )
        stage = "tactile"
        tactile_outputs = tactile.process_tactile_episode(
            raw_object,
            processed_object,
            _LOCAL_PROCESSING_EPISODE_INDEX,
            output_dir=processed_object,
            sensors=None,
            overwrite=False,
        )
        stage = "robot"
        robot_output = robot_stage.process_robot_episode(
            processed_object,
            _LOCAL_PROCESSING_EPISODE_INDEX,
            bimanual=bimanual,
            cameras=None,
            seed=0,
            overwrite=False,
            plot=False,
        )
        stage = "publication"
        frame_count, timeline_sha256 = _timeline_details(episode_dir)
        camera_dirs = sorted(
            path
            for path in episode_dir.iterdir()
            if path.is_dir()
            and path.name.startswith(_CAMERA_PREFIX)
            and (path / "undistorted.mp4").is_file()
        )
        output_files = list(_all_files(episode_dir))
        if robot_output not in output_files:
            output_files.append(robot_output)
        for tactile_output in tactile_outputs.values():
            if tactile_output not in output_files:
                output_files.append(tactile_output)
        output_artifacts = _relative_artifacts(data_root, output_files)
        return Deform360CalibrationAcquisitionCaseV1(
            plan_id=plan_id,
            object_id=object_id,
            episode_id=episode_id,
            stratum=stratum,
            status="prepared",
            raw_factor_artifacts=raw_artifacts,
            output_artifacts=output_artifacts,
            aligned_frame_count=frame_count,
            camera_count=len(camera_dirs),
            tactile_sensor_count=len(tactile_outputs),
            bimanual=bimanual,
            metadata={
                "timeline_sha256": timeline_sha256,
                "source_episode_id": episode_id,
                "official_processing_episode_index": (
                    _LOCAL_PROCESSING_EPISODE_INDEX
                ),
                "official_stages": ["undistort", "tactile", "robot"],
                "camera_policy": "all-official-cameras-sorted-v1",
                "tactile_policy": "all-exact-npy-timestamp-sensors-sorted-v1",
                "robot_seed": 0,
            },
        )
    except Exception as error:
        failure_root.mkdir(parents=True, exist_ok=True)
        rendered = "".join(traceback.format_exception(error)).replace(
            str(data_root),
            "<CALIBRATION_DATA_ROOT>",
        )
        failure_path = failure_root / f"{object_id}-episode-{episode_id:04d}.txt"
        failure_path.write_text(rendered, encoding="utf-8")
        partial_outputs = _relative_artifacts(
            data_root,
            tuple(set(_all_files(processed_object)) - before_outputs),
        )
        redacted_message = str(error).replace(
            str(data_root),
            "<CALIBRATION_DATA_ROOT>",
        )
        normalized = f"{type(error).__name__}:{redacted_message}"
        return Deform360CalibrationAcquisitionCaseV1(
            plan_id=plan_id,
            object_id=object_id,
            episode_id=episode_id,
            stratum=stratum,
            status="technical_failure",
            raw_factor_artifacts=raw_artifacts,
            output_artifacts={},
            failure_stage=stage,
            failure_type=type(error).__name__,
            failure_message_sha256=_sha256_text(normalized),
            metadata={
                "failure_log": failure_path.name,
                "failure_log_sha256": file_sha256(failure_path),
                "partial_output_artifacts": partial_outputs,
                "technical_failure_retained_without_replacement": True,
            },
        )

__all__ = [name for name in globals() if not name.startswith("__")]
