#!/usr/bin/env python3
"""Open and prepare only the ten locked Deform360 calibration units.

The command requires an explicit ``--open-calibration-payloads`` acknowledgement.
It exposes no confirmation-root or target-outcome argument. It downloads the
minimum exact selected-episode camera/tactile source set, runs the pinned official
undistort, tactile, and robot stages, retains every technical failure without
replacement, and emits a calibration-only evidence ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
    sha256_digest,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_acquisition import (
    DEFORM360_CALIBRATION_ACQUISITION_CLAIM_BOUNDARY,
    Deform360CalibrationAcquisitionCaseV1,
    Deform360CalibrationAcquisitionPlanV1,
    build_calibration_acquisition_plan,
    build_calibration_acquisition_result,
    build_calibration_evidence_ledger,
    file_sha256,
    save_calibration_acquisition_case,
    save_calibration_acquisition_plan,
    save_calibration_acquisition_result,
    save_calibration_evidence_ledger,
    select_calibration_object_paths,
    validate_calibration_download_root,
)


_CAMERA_PREFIX = "brics-odroid-"
_LOCAL_PROCESSING_EPISODE_INDEX = 0


def _run_git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head(checkout: Path, *, name: str) -> str:
    if not (checkout / ".git").exists():
        raise ValueError(f"{name} checkout is not a Git repository: {checkout}")
    revision = _run_git(checkout, "rev-parse", "HEAD")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} checkout has no exact 40-character revision")
    return revision


def _require_clean(
    checkout: Path,
    *,
    name: str,
    allowed_untracked: Sequence[str] = (),
) -> None:
    raw_allowed = tuple(allowed_untracked)
    allowed_values: list[str] = []
    for raw_path in raw_allowed:
        if type(raw_path) is not str or not raw_path:
            raise ValueError(
                "allowed_untracked must contain canonical relative paths"
            )
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or path.as_posix() == "."
            or ".." in path.parts
            or path.as_posix() != raw_path.rstrip("/")
        ):
            raise ValueError(
                "allowed_untracked must contain canonical relative paths"
            )
        allowed_values.append(path.as_posix())
    if len(set(allowed_values)) != len(allowed_values):
        raise ValueError("allowed_untracked must not contain duplicates")
    allowed = tuple(allowed_values)
    status = _run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    violations = []
    for line in status.splitlines():
        if line.startswith("?? "):
            relative = line[3:].rstrip("/")
            if any(
                relative == prefix or relative.startswith(f"{prefix}/")
                for prefix in allowed
            ):
                continue
        violations.append(line)
    if violations:
        raise ValueError(
            f"{name} checkout has tracked or untracked modifications: "
            f"{violations[:5]}"
        )


def _roots_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _require_separate_data_and_output_roots(
    *,
    repository: Path,
    deform360_checkout: Path,
    data_root: Path,
    output: Path,
) -> None:
    for checkout, name in (
        (repository, "BayesianPhysTwin"),
        (deform360_checkout, "Deform360"),
    ):
        if _roots_overlap(data_root, checkout):
            raise ValueError(f"calibration data root overlaps the {name} checkout")
        if _roots_overlap(output, checkout):
            raise ValueError(f"evidence output overlaps the {name} checkout")
    if _roots_overlap(data_root, output):
        raise ValueError("calibration data and compact evidence roots overlap")


def _strict_json(path: Path) -> dict[str, Any]:
    return dict(load_strict_json_object(path, label="Deform360 metadata"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative_artifacts(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = file_sha256(path)
    return artifacts


def _all_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    )


def _metadata_bimanual(metadata: Mapping[str, Any], episode_id: int) -> bool:
    sequences = metadata.get("sequences")
    if isinstance(sequences, list):
        if episode_id >= len(sequences) or not isinstance(
            sequences[episode_id],
            Mapping,
        ):
            raise ValueError(f"metadata has no sequence {episode_id}")
        record = sequences[episode_id]
    elif isinstance(sequences, Mapping):
        record = sequences.get(str(episode_id), sequences.get(episode_id))
        if not isinstance(record, Mapping):
            raise ValueError(f"metadata has no sequence {episode_id}")
    else:
        raise ValueError("metadata sequences must be a list or object")
    value = record.get("bimanual")
    if isinstance(value, bool):
        return value
    if value in {"yes", "no"}:
        return value == "yes"
    raise ValueError("sequence bimanual flag must be Boolean or yes/no")

__all__ = [name for name in globals() if not name.startswith("__")]
