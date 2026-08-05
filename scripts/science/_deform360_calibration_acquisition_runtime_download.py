# ruff: noqa: E402, F403, F405
"""Internal runtime slice for Deform360 calibration acquisition."""

from __future__ import annotations

from _deform360_calibration_acquisition_runtime_common import *

def _tree_paths(
    *,
    repository: str,
    revision: str,
    object_id: str,
    token: str | None,
) -> tuple[str, ...]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "calibration acquisition requires huggingface_hub"
        ) from error
    api = HfApi(token=token or None)
    entries = api.list_repo_tree(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        path_in_repo=f"raw/{object_id}",
        recursive=True,
        expand=False,
    )
    paths = tuple(
        sorted(
            entry.path
            for entry in entries
            if isinstance(getattr(entry, "path", None), str)
        )
    )
    if not paths:
        raise ValueError(f"dataset revision has no raw/{object_id} subtree")
    return paths


def _selected_unit_paths(
    *,
    repository: str,
    revision: str,
    object_id: str,
    episode_id: int,
    token: str | None,
) -> tuple[str, ...]:
    repository_paths = _tree_paths(
        repository=repository,
        revision=revision,
        object_id=object_id,
        token=token,
    )
    return select_calibration_object_paths(
        repository_paths,
        object_id=object_id,
        episode_id=episode_id,
    )


def _download_unit(
    *,
    repository: str,
    revision: str,
    object_id: str,
    episode_id: int,
    selected_paths: Sequence[str],
    expected_metadata_sha256: str,
    data_root: Path,
    token: str | None,
) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "calibration acquisition requires huggingface_hub"
        ) from error

    selected = tuple(selected_paths)
    if not selected:
        raise ValueError(f"payload allowlist is empty for {object_id}")
    root = data_root.resolve()
    records: list[dict[str, object]] = []
    for path in selected:
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != path
        ):
            raise ValueError(f"download path is not canonical: {path}")
        expected = (root / Path(*relative.parts)).absolute()
        if not expected.is_relative_to(root):
            raise ValueError(f"download path escapes data root: {path}")
        downloaded = Path(
            hf_hub_download(
                repo_id=repository,
                repo_type="dataset",
                revision=revision,
                filename=path,
                local_dir=root,
                token=token or None,
            )
        ).expanduser().absolute()
        if downloaded != expected:
            raise ValueError(
                "download escaped requested local path for "
                f"{path}: {downloaded} != {expected}"
            )
        if downloaded.is_symlink() or not downloaded.is_file():
            raise ValueError(
                f"downloaded path is not an ordinary file: {downloaded}"
            )
        if not downloaded.resolve().is_relative_to(root):
            raise ValueError(f"downloaded path resolves outside data root: {path}")
        records.append(
            {
                "path": path,
                "bytes": downloaded.stat().st_size,
                "sha256": file_sha256(downloaded),
            }
        )
    metadata_path = f"raw/{object_id}/metadata.json"
    metadata_records = [
        record for record in records if record.get("path") == metadata_path
    ]
    if len(metadata_records) != 1:
        raise ValueError(f"payload allowlist omitted {metadata_path}")
    observed_metadata = metadata_records[0].get("sha256")
    if observed_metadata != expected_metadata_sha256:
        raise ValueError(
            f"downloaded metadata changed for {object_id}: "
            f"{observed_metadata} != {expected_metadata_sha256}"
        )
    return {
        "object_id": object_id,
        "episode_id": episode_id,
        "selected_path_count": len(records),
        "selected_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }


def _payload_allowlist(
    plan: Deform360CalibrationAcquisitionPlanV1,
    selected_paths_by_object: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    expected_ids = {unit.object_id for unit in plan.calibration_units}
    observed_ids = set(selected_paths_by_object)
    if observed_ids != expected_ids:
        raise ValueError(
            "payload allowlist object set differs from the locked calibration cohort"
        )
    units = []
    for unit in plan.calibration_units:
        raw_paths = selected_paths_by_object[unit.object_id]
        if isinstance(raw_paths, (str, bytes)):
            raise ValueError("payload allowlist paths must be a sequence")
        paths = tuple(sorted(raw_paths))
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("payload allowlist paths must be nonempty and unique")
        prefix = PurePosixPath("raw") / unit.object_id
        invalid = []
        for path_text in paths:
            path = PurePosixPath(path_text) if type(path_text) is str else None
            if (
                path is None
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != path_text
                or path == prefix
                or not path.is_relative_to(prefix)
                or path.suffix.lower() in {".wav", ".flac"}
            ):
                invalid.append(path_text)
        if invalid:
            raise ValueError(
                f"payload allowlist escapes or includes audio for {unit.object_id}"
            )
        units.append(
            {
                "object_id": unit.object_id,
                "episode_id": unit.episode_id,
                "stratum": unit.stratum,
                "metadata_sha256": unit.metadata_sha256,
                "selected_paths": paths,
            }
        )
    descriptor: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-calibration-payload-allowlist",
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "dataset_repository": plan.dataset_repository,
        "dataset_revision": plan.dataset_revision,
        "units": units,
        "calibration_payloads_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
        "claim_boundary": DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY,
    }
    return {**descriptor, "allowlist_id": content_id(descriptor)}


def _download_manifest(
    plan: Deform360CalibrationAcquisitionPlanV1,
    *,
    payload_allowlist_id: str,
    downloads: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    allowlist_id = sha256_digest(
        payload_allowlist_id,
        name="payload_allowlist_id",
    )
    if isinstance(downloads, (str, bytes)):
        raise ValueError("downloads must be a sequence")
    by_object: dict[str, Mapping[str, Any]] = {}
    for item in downloads:
        if not isinstance(item, Mapping):
            raise ValueError("download unit must be a mapping")
        object_id = item.get("object_id")
        if type(object_id) is not str or not object_id:
            raise ValueError("download unit has no object_id")
        if object_id in by_object:
            raise ValueError(f"download manifest repeats object {object_id}")
        by_object[object_id] = item
    expected = {unit.object_id for unit in plan.calibration_units}
    if set(by_object) != expected:
        raise ValueError("download manifest does not cover the locked cohort")
    ordered_downloads: list[dict[str, Any]] = []
    for unit in plan.calibration_units:
        item = by_object[unit.object_id]
        if item.get("episode_id") != unit.episode_id:
            raise ValueError(
                f"download manifest episode changed for {unit.object_id}"
            )
        file_count = item.get("selected_path_count")
        selected_bytes = item.get("selected_bytes")
        files = item.get("files")
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 1
            or isinstance(selected_bytes, bool)
            or not isinstance(selected_bytes, int)
            or selected_bytes < 0
            or not isinstance(files, list)
            or len(files) != file_count
        ):
            raise ValueError(
                f"download manifest inventory changed for {unit.object_id}"
            )
        ordered_downloads.append(dict(item))
    descriptor: dict[str, object] = {
        "schema": "bayesian-phystwin.deform360-calibration-download-manifest",
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "payload_allowlist_id": allowlist_id,
        "dataset_repository": plan.dataset_repository,
        "dataset_revision": plan.dataset_revision,
        "units": ordered_downloads,
        "total_file_count": sum(
            int(item["selected_path_count"]) for item in ordered_downloads
        ),
        "total_bytes": sum(
            int(item["selected_bytes"]) for item in ordered_downloads
        ),
        "calibration_payloads_opened": True,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "replacement_allowed": False,
        "claim_boundary": DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY,
    }
    return {**descriptor, "manifest_id": content_id(descriptor)}

__all__ = [name for name in globals() if not name.startswith("__")]
