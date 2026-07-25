"""Additive execution adapter for the frozen Deform360 prospective v2 study.

The v2 protocol deliberately leaves the source-v4 estimator and every v1
runner checksum unchanged.  This module binds those frozen functions to the
larger v2 calibration cohort in a process-local context.  No outcome loader is
installed here: the adapter is only for target-free prediction construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any

from . import deform360_bias_aware_prospective_artifacts as artifacts
from . import deform360_bias_aware_prospective_physical as physical
from . import deform360_bias_aware_prospective_uncertainty as uncertainty
from .deform360_bias_aware_prospective_artifacts import canonical_sha256, file_sha256
from .deform360_bias_aware_prospective_protocol import EXPECTED_STRATA
from .deform360_bias_aware_prospective_v2_download import (
    bias_aware_prospective_v2_fresh_download_plan,
)
from .deform360_bias_aware_prospective_v2_protocol import (
    DATASET_REVISION,
    PROTOCOL_ID,
    load_bias_aware_prospective_v2_protocol,
)


EXECUTION_LOCK_ARTIFACT_KIND = "Deform360BiasAwareProspectiveV2ExecutionLock"
FRESH_DOWNLOAD_ARTIFACT_KIND = "Deform360BiasAwareProspectiveV2FreshDownload"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_cohort(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    cohort = {stratum: {} for stratum in EXPECTED_STRATA}
    _require(set(records) == set(EXPECTED_STRATA), "unexpected v2 strata")
    for stratum in EXPECTED_STRATA:
        rows = records[stratum]
        _require(isinstance(rows, Sequence), "v2 cohort rows are missing")
        for row in rows:
            _require(isinstance(row, Mapping), "v2 cohort row is invalid")
            object_id = str(row["object_id"])
            episodes = tuple(int(value) for value in row["episode_ids"])
            _require(object_id not in cohort[stratum], "v2 cohort object repeated")
            _require(episodes, "v2 cohort object has no episode")
            cohort[stratum][object_id] = episodes
    return cohort


def load_bias_aware_prospective_v2_execution_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Expose the v2 lock through the normalized interface used by v1 code."""

    payload = load_bias_aware_prospective_v2_protocol(path)
    config = payload["config"]
    calibration = _normalized_cohort(config["calibration_cohort"])
    target = _normalized_cohort(config["target_cohort"])
    return {
        "payload": payload,
        "config": dict(config),
        "config_sha256": str(payload["config_sha256"]),
        "calibration_cohort": calibration,
        "target_cohort": target,
    }


def prospective_v2_case_records(
    protocol_path: str | Path,
    *,
    role: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return canonical v2 cases without weakening the role boundary."""

    protocol = load_bias_aware_prospective_v2_execution_protocol(protocol_path)
    roles = (role,) if role is not None else ("calibration", "target")
    _require(all(value in {"calibration", "target"} for value in roles), "bad role")
    rows: list[dict[str, Any]] = []
    for current_role in roles:
        cohort = protocol[f"{current_role}_cohort"]
        for stratum in EXPECTED_STRATA:
            for object_id, episode_ids in cohort[stratum].items():
                for episode_id in episode_ids:
                    rows.append(
                        {
                            "case": f"{object_id}-ep{episode_id:04d}",
                            "object_id": object_id,
                            "episode_id": int(episode_id),
                            "episode_key": f"{object_id}/{episode_id}",
                            "stratum": stratum,
                            "role": current_role,
                        }
                    )
    expected = 12 if role == "calibration" else 24 if role == "target" else 36
    _require(len(rows) == expected, "prospective v2 case panel is incomplete")
    _require(len({row["case"] for row in rows}) == len(rows), "v2 case repeated")
    return tuple(rows)


def prospective_v2_case_record(
    protocol_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Resolve one locked v2 object/episode pair."""

    matches = [
        row
        for row in prospective_v2_case_records(protocol_path)
        if row["object_id"] == object_id and row["episode_id"] == int(episode_id)
    ]
    _require(len(matches) == 1, "object/episode is outside the prospective v2 lock")
    return matches[0]


def validate_v2_fresh_download_manifest(
    path: Path,
    *,
    protocol_config_sha256: str,
    object_id: str,
    episode_id: int,
    metadata_path: Path,
) -> dict[str, Any]:
    """Validate one fresh-object authorization without reading media."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == FRESH_DOWNLOAD_ARTIFACT_KIND
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("protocol_config_sha256") == protocol_config_sha256
        and payload.get("revision") == DATASET_REVISION,
        "v2 download manifest is incompatible",
    )
    _require(
        payload.get("manifest_sha256")
        == canonical_sha256(payload, digest_key="manifest_sha256"),
        "v2 download manifest checksum changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("fresh_calibration_objects_only") is True
        and boundary.get("fresh_calibration_future_opened") is False
        and boundary.get("reserved_target_downloaded") is False
        and boundary.get("reserved_target_media_read") is False
        and boundary.get("target_metrics_opened") is False,
        "v2 download crossed its information boundary",
    )
    rows = [
        row
        for row in payload.get("objects", [])
        if isinstance(row, Mapping) and row.get("object_id") == object_id
    ]
    _require(len(rows) == 1, "v2 download manifest object changed")
    _require(
        episode_id in rows[0].get("selected_episode_ids", []),
        "v2 download manifest omitted the selected episode",
    )
    _require(
        rows[0].get("metadata_sha256") == file_sha256(metadata_path),
        "v2 object metadata changed after download",
    )
    return payload


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_v2_execution_lock(
    path: str | Path,
    *,
    repository: str | Path,
    require_clean_repository: bool = True,
) -> dict[str, Any]:
    """Validate the additive runner and its frozen v1 implementation inputs."""

    lock_path = Path(path).resolve()
    repo = Path(repository).resolve()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == EXECUTION_LOCK_ARTIFACT_KIND,
        "wrong v2 execution-lock kind",
    )
    _require(payload.get("protocol_id") == PROTOCOL_ID, "execution protocol changed")
    _require(
        payload.get("config_sha256")
        == canonical_sha256(payload, digest_key="config_sha256"),
        "v2 execution-lock checksum changed",
    )
    files = payload.get("files_sha256")
    _require(isinstance(files, Mapping) and files, "execution files are missing")
    for relative, expected in files.items():
        source = repo / str(relative)
        _require(source.is_file(), f"execution file is missing: {relative}")
        _require(file_sha256(source) == expected, f"execution file changed: {relative}")
    if require_clean_repository:
        _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
        lock_commit = str(payload.get("adapter_lock_commit", ""))
        _require(bool(lock_commit), "adapter lock commit is missing")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock_commit, "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        _git_output(repo, "ls-files", "--error-unmatch", str(lock_path.relative_to(repo)))
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("outcome_loader_installed") is False
        and boundary.get("calibration_future_access_authorized") is False
        and boundary.get("target_access_authorized") is False,
        "execution lock authorizes forbidden access",
    )
    return payload


def _set(module: ModuleType, name: str, value: Any, changes: list[tuple[Any, str, Any]]) -> None:
    if hasattr(module, name):
        changes.append((module, name, getattr(module, name)))
        setattr(module, name, value)


@contextmanager
def activate_v2_prediction_runtime() -> Iterator[None]:
    """Bind frozen v1 builders to v2 identities for one isolated process."""

    changes: list[tuple[Any, str, Any]] = []
    for module in (artifacts, physical, uncertainty):
        _set(module, "PROTOCOL_ID", PROTOCOL_ID, changes)
    _set(
        artifacts,
        "load_bias_aware_prospective_protocol",
        load_bias_aware_prospective_v2_execution_protocol,
        changes,
    )
    _set(artifacts, "prospective_case_records", prospective_v2_case_records, changes)
    _set(artifacts, "prospective_case_record", prospective_v2_case_record, changes)
    try:
        yield
    finally:
        for module, name, value in reversed(changes):
            setattr(module, name, value)


def patch_v2_stage_module(
    module: ModuleType,
    *,
    stage: str,
    repository: Path,
    execution_lock: Path,
) -> None:
    """Patch aliases imported by one checksum-bound remote stage."""

    common = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "load_bias_aware_prospective_protocol": (
            load_bias_aware_prospective_v2_execution_protocol
        ),
        "prospective_case_record": prospective_v2_case_record,
        "prospective_case_records": prospective_v2_case_records,
    }
    for name, value in common.items():
        if hasattr(module, name):
            setattr(module, name, value)
    if stage == "prepare-source":
        module.bias_aware_prospective_download_plan = (
            bias_aware_prospective_v2_fresh_download_plan
        )
        module._validate_download_manifest = validate_v2_fresh_download_manifest
    if stage == "physical-prior":
        original_run_logged = module._run_logged

        def run_logged(command: Sequence[str], **kwargs: Any):
            rewritten = list(command)
            if len(rewritten) >= 2 and Path(rewritten[1]).name == (
                "build_deform360_bias_aware_automatic_twin.py"
            ):
                rewritten = [
                    rewritten[0],
                    str(repository / "scripts/remote/run_deform360_bias_aware_v2_stage.py"),
                    "--execution-repo",
                    str(repository),
                    "--execution-lock",
                    str(execution_lock),
                    "--stage",
                    "automatic-twin",
                    *rewritten[2:],
                ]
            return original_run_logged(rewritten, **kwargs)

        module._run_logged = run_logged


__all__ = [
    "EXECUTION_LOCK_ARTIFACT_KIND",
    "activate_v2_prediction_runtime",
    "load_bias_aware_prospective_v2_execution_protocol",
    "patch_v2_stage_module",
    "prospective_v2_case_record",
    "prospective_v2_case_records",
    "validate_v2_execution_lock",
    "validate_v2_fresh_download_manifest",
]
