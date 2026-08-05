"""Runtime and manifest assembly for Deform360 calibration payload access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from .deform360_calibration_execution import (
    Deform360Stage0SelectionV1,
    load_deform360_stage0_selection,
)
from .deform360_calibration_payload_plan import (
    DATASET_REPOSITORY,
    HubApi,
    UnitPlan,
    build_unit_plan,
    canonical_hub_path,
    list_unit_files,
    require,
)
from .deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
    load_deform360_visual_provider_lock,
)

MATERIALIZATION_SCHEMA = (
    "bayesian-phystwin/deform360-calibration-payload-materialization-v1"
)
MATERIALIZATION_VERSION = 1
MATERIALIZATION_SEMANTICS = (
    "exact-revision-selected-episode-calibration-prefix-materialization-v1"
)
PROCESSING_REPOSITORY = "lhy0807/deform360"
CLAIM_BOUNDARY = (
    "Calibration-input and information-order evidence only. This artifact does "
    "not establish provider competence, tactile informativeness, physical-query "
    "improvement, predictive calibration, material identification, Causal4D "
    "benefit, safety, or state of the art."
)

DownloadFile = Callable[..., str]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_revision(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact lowercase revision")
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be an exact lowercase revision")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git checkout {checkout}") from error
    return completed.stdout.strip()


def require_clean_revision(
    checkout: str | Path,
    *,
    expected_revision: str,
    name: str,
) -> str:
    root = Path(checkout).resolve()
    require((root / ".git").exists(), f"{name} is not a Git checkout: {root}")
    expected = _require_revision(expected_revision, name=f"{name} expected revision")
    observed = _require_revision(
        _run_git(root, "rev-parse", "HEAD"),
        name=f"{name} HEAD",
    )
    require(
        observed == expected,
        f"{name} revision differs from the registered revision",
    )
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    require(not status, f"{name} checkout is dirty")
    return observed


def _safe_download(
    *,
    download_file: DownloadFile,
    path: str,
    revision: str,
    dataset_root: Path,
    token: str | None,
) -> dict[str, object]:
    local = Path(
        download_file(
            repo_id=DATASET_REPOSITORY,
            repo_type="dataset",
            revision=revision,
            filename=path,
            local_dir=dataset_root,
            token=token,
        )
    )
    resolved_root = dataset_root.resolve()
    resolved_local = local.resolve()
    try:
        resolved_local.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"download escaped the dataset root: {path}") from error
    require(
        local.is_file() and not local.is_symlink(),
        f"download is not an ordinary file: {path}",
    )
    return {
        "path": path,
        "status": "downloaded",
        "sha256": file_sha256(local),
        "bytes": local.stat().st_size,
    }


def materialize_paths(
    paths: Sequence[str],
    *,
    revision: str,
    dataset_root: Path,
    token: str | None,
    workers: int,
    download_file: DownloadFile,
) -> tuple[dict[str, object], ...]:
    require(
        type(workers) is int and workers >= 1,
        "workers must be a positive integer",
    )
    canonical_paths = tuple(sorted({canonical_hub_path(path) for path in paths}))
    dataset_root.mkdir(parents=True, exist_ok=True)

    def download(path: str) -> dict[str, object]:
        try:
            return _safe_download(
                download_file=download_file,
                path=path,
                revision=revision,
                dataset_root=dataset_root,
                token=token,
            )
        except Exception as error:
            return {
                "path": path,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(download, canonical_paths))
    return tuple(sorted(records, key=lambda item: cast(str, item["path"])))


def _metadata_integrity_failures(
    plans: Sequence[UnitPlan],
    *,
    downloaded_files: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    records = {str(item["path"]): item for item in downloaded_files}
    failures: list[str] = []
    for plan in plans:
        record = records.get(plan.metadata_path)
        if record is None or record.get("status") != "downloaded":
            failures.append(f"metadata_download_failed:{plan.object_id}")
            continue
        observed = record.get("sha256")
        if observed != plan.expected_metadata_sha256:
            failures.append(f"metadata_sha256_mismatch:{plan.object_id}")
    return tuple(failures)


def _load_processing_calibration_module(processing_checkout: Path) -> Any:
    module_path = processing_checkout / "deform360" / "calibration.py"
    specification = importlib.util.spec_from_file_location(
        "deform360_calibration_for_materialization",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise ValueError("cannot load the pinned Deform360 calibration module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def validate_calibrated_camera_coverage(
    plans: Sequence[UnitPlan],
    *,
    dataset_root: Path,
    processing_checkout: Path,
) -> dict[str, object]:
    module = _load_processing_calibration_module(processing_checkout)
    records: dict[str, object] = {}
    for plan in plans:
        try:
            calibration = module.load_calibration(
                dataset_root / "raw" / plan.object_id
            )
        except Exception as error:
            records[plan.object_id] = {
                "status": "technical_failure",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            continue
        calibrated = tuple(calibration.cameras)
        planned = {camera for camera, _, _ in plan.camera_recordings}
        missing = tuple(sorted(set(calibrated) - planned))
        records[plan.object_id] = {
            "status": "ready" if not missing else "technical_failure",
            "calibrated_camera_count": len(calibrated),
            "calibrated_cameras": list(calibrated),
            "selected_episode_missing_calibrated_cameras": list(missing),
        }
    return records


def build_manifest(
    *,
    selection: Deform360Stage0SelectionV1,
    visual_provider: Deform360VisualProviderLockV1,
    plans: Sequence[UnitPlan],
    implementation_revision: str,
    processing_revision: str,
    opened_payloads: bool,
    downloaded_files: Sequence[Mapping[str, object]] = (),
    calibrated_camera_coverage: Mapping[str, object] | None = None,
    runtime_failures: Sequence[str] = (),
) -> dict[str, object]:
    confirmation_ids = sorted(unit.object_id for unit in selection.confirmation_units)
    confirmation_prefixes = tuple(f"raw/{object_id}/" for object_id in confirmation_ids)
    every_path = [path for plan in plans for path in plan.materialization_paths]
    forbidden = sorted(
        path
        for path in every_path
        if any(path.startswith(prefix) for prefix in confirmation_prefixes)
    )
    require(not forbidden, f"confirmation paths entered materialization: {forbidden}")
    calibration_ids = sorted(unit.object_id for unit in selection.calibration_units)
    require(
        sorted(plan.object_id for plan in plans) == calibration_ids,
        "materialization plans do not cover the exact calibration cohort",
    )

    technical_failure_ids = {
        plan.object_id for plan in plans if plan.status == "technical_failure"
    }
    for object_id, record in (calibrated_camera_coverage or {}).items():
        if isinstance(record, Mapping) and record.get("status") == "technical_failure":
            technical_failure_ids.add(str(object_id))
    if runtime_failures:
        status = "failed"
    elif not opened_payloads:
        status = "planned"
    elif technical_failure_ids:
        status = "complete_with_technical_failures"
    else:
        status = "complete"

    descriptor: dict[str, object] = {
        "schema": MATERIALIZATION_SCHEMA,
        "schema_version": MATERIALIZATION_VERSION,
        "semantics": MATERIALIZATION_SEMANTICS,
        "protocol_id": selection.protocol_id,
        "stage0_snapshot_id": selection.snapshot_id,
        "stage0_source_sha256": selection.source_sha256,
        "selection_artifact_sha256": selection.selection_artifact_sha256,
        "visual_provider_lock_id": visual_provider.artifact_id,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": selection.dataset_revision,
        "processing_repository": PROCESSING_REPOSITORY,
        "processing_revision": processing_revision,
        "implementation_revision": implementation_revision,
        "calibration_object_ids": calibration_ids,
        "confirmation_object_ids": confirmation_ids,
        "status": status,
        "runtime_failures": sorted(set(runtime_failures)),
        "technical_failure_object_count": len(technical_failure_ids),
        "units": [
            plan.to_record()
            for plan in sorted(plans, key=lambda item: item.object_id)
        ],
        "downloaded_files": [dict(item) for item in downloaded_files],
        "calibrated_camera_coverage": dict(calibrated_camera_coverage or {}),
        "information_boundary": {
            "calibration_payloads_opened": opened_payloads,
            "camera_media_opened": False,
            "camera_timestamp_sidecars_opened": opened_payloads,
            "trusted_camera_calibration_opened": opened_payloads,
            "tactile_arrays_opened": opened_payloads,
            "robot_arrays_opened": False,
            "geometry_annotations_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
        "replacement_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return {"manifest_id": _content_id(descriptor), **descriptor}


def write_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    output = Path(path).resolve()
    require(not output.exists(), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    rendered = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)


def execute_materialization(
    *,
    selection_lock: Path,
    protocol_path: Path,
    visual_provider_lock: Path,
    repository_root: Path,
    processing_checkout: Path,
    dataset_root: Path,
    output: Path,
    implementation_revision: str,
    workers: int,
    open_calibration_payloads: bool,
    token: str | None,
    api: HubApi,
    download_file: DownloadFile,
) -> dict[str, object]:
    selection = load_deform360_stage0_selection(
        selection_lock,
        protocol_path=protocol_path,
    )
    visual_provider = load_deform360_visual_provider_lock(visual_provider_lock)
    require(
        visual_provider.protocol_id == selection.protocol_id,
        "visual provider and Stage-0 protocol IDs differ",
    )
    implementation_revision = require_clean_revision(
        repository_root,
        expected_revision=implementation_revision,
        name="BayesianPhysTwin",
    )
    processing_revision = require_clean_revision(
        processing_checkout,
        expected_revision=selection.processing_revision,
        name="Deform360 processing",
    )

    info = api.repo_info(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=selection.dataset_revision,
        token=token,
    )
    resolved_revision = _require_revision(
        info.sha,
        name="resolved dataset revision",
    )
    require(
        resolved_revision == selection.dataset_revision,
        "official Hub revision differs from the Stage-0 lock",
    )

    confirmation_ids = {unit.object_id for unit in selection.confirmation_units}
    plans: list[UnitPlan] = []
    for unit in selection.calibration_units:
        require(
            unit.object_id not in confirmation_ids,
            "calibration/confirmation overlap",
        )
        files = list_unit_files(
            api,
            object_id=unit.object_id,
            revision=selection.dataset_revision,
            token=token,
        )
        plans.append(build_unit_plan(unit, files))

    downloaded: tuple[dict[str, object], ...] = ()
    coverage: Mapping[str, object] = {}
    runtime_failures: list[str] = []
    if open_calibration_payloads:
        paths = [path for plan in plans for path in plan.materialization_paths]
        downloaded = materialize_paths(
            paths,
            revision=selection.dataset_revision,
            dataset_root=dataset_root,
            token=token,
            workers=workers,
            download_file=download_file,
        )
        runtime_failures = [
            f"download_failed:{item['path']}:{item.get('error_type', 'unknown')}"
            for item in downloaded
            if item.get("status") != "downloaded"
        ]
        runtime_failures.extend(
            _metadata_integrity_failures(
                plans,
                downloaded_files=downloaded,
            )
        )
        coverage = validate_calibrated_camera_coverage(
            plans,
            dataset_root=dataset_root,
            processing_checkout=processing_checkout,
        )

    manifest = build_manifest(
        selection=selection,
        visual_provider=visual_provider,
        plans=plans,
        implementation_revision=implementation_revision,
        processing_revision=processing_revision,
        opened_payloads=open_calibration_payloads,
        downloaded_files=downloaded,
        calibrated_camera_coverage=coverage,
        runtime_failures=runtime_failures,
    )
    write_manifest(output, manifest)
    if runtime_failures:
        raise ValueError(
            "calibration payload materialization failed; inspect the persisted manifest"
        )
    return manifest


__all__ = [
    "CLAIM_BOUNDARY",
    "DownloadFile",
    "MATERIALIZATION_SCHEMA",
    "MATERIALIZATION_SEMANTICS",
    "MATERIALIZATION_VERSION",
    "PROCESSING_REPOSITORY",
    "build_manifest",
    "execute_materialization",
    "file_sha256",
    "materialize_paths",
    "require_clean_revision",
    "validate_calibrated_camera_coverage",
    "write_manifest",
]
