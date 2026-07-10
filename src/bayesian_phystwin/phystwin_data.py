"""Selective, auditable retrieval of the released PhysTwin evaluation data."""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_DATA_ARCHIVE = (
    "https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip"
)
DEFAULT_EXPERIMENTS_ARCHIVE = (
    "https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments.zip"
)
DEFAULT_ADDITIONAL_ARCHIVE = (
    "https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/additional_data.zip"
)
EVALUATION_FILENAMES = ("final_data.pkl", "gt_track_3d.pkl", "split.json")
ADDITIONAL_EVALUATION_FILENAMES = ("final_data.pkl", "split.json")


def _archive_factory(source: str) -> Any:
    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise RuntimeError(
            "selective retrieval requires the 'data' extra: "
            "pip install 'bayesian-phystwin[data]'"
        ) from error
    return RemoteZip(source)


def _available_cases(data_archive: Any, experiments_archive: Any) -> tuple[str, ...]:
    data_names = set(data_archive.namelist())
    experiment_names = set(experiments_archive.namelist())
    prefix = "data/different_types/"
    cases = {
        name[len(prefix) :].split("/", 1)[0]
        for name in data_names
        if name.startswith(prefix) and name.endswith("/split.json")
    }
    complete = []
    for case in sorted(cases):
        required = {
            f"{prefix}{case}/{filename}" for filename in EVALUATION_FILENAMES
        }
        if required <= data_names and f"experiments/{case}/inference.pkl" in experiment_names:
            complete.append(case)
    return tuple(complete)


def _available_additional_cases(archive: Any) -> tuple[str, ...]:
    names = set(archive.namelist())
    prefix = "additional_data/data/different_types/"
    cases = {
        name[len(prefix) :].split("/", 1)[0]
        for name in names
        if name.startswith(prefix) and name.endswith("/split.json")
    }
    complete = []
    for case in sorted(cases):
        required = {
            f"{prefix}{case}/{filename}"
            for filename in ADDITIONAL_EVALUATION_FILENAMES
        }
        inference = f"additional_data/experiments/{case}/inference.pkl"
        if required <= names and inference in names:
            complete.append(case)
    return tuple(complete)


def _digests(path: Path) -> tuple[str, int]:
    sha256 = hashlib.sha256()
    crc32 = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
            crc32 = zlib.crc32(block, crc32)
    return sha256.hexdigest(), crc32 & 0xFFFFFFFF


def _retrieve_member(archive: Any, member: str, destination: Path) -> dict[str, object]:
    info = archive.getinfo(member)
    reused = False
    if destination.is_file() and destination.stat().st_size == info.file_size:
        sha256, crc32 = _digests(destination)
        reused = crc32 == info.CRC
    if not reused:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        sha256_digest = hashlib.sha256()
        crc32 = 0
        try:
            with archive.open(member) as source, temporary.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    sha256_digest.update(block)
                    crc32 = zlib.crc32(block, crc32)
            crc32 &= 0xFFFFFFFF
            if temporary.stat().st_size != info.file_size or crc32 != info.CRC:
                raise OSError(f"archive integrity check failed for {member}")
            os.replace(temporary, destination)
            sha256 = sha256_digest.hexdigest()
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "archive_member": member,
        "bytes": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "sha256": sha256,
        "reused": reused,
    }


def fetch_phystwin_evaluation_subset(
    output_dir: str | Path,
    *,
    cases: Iterable[str] | None = None,
    data_archive_url: str = DEFAULT_DATA_ARCHIVE,
    experiments_archive_url: str = DEFAULT_EXPERIMENTS_ARCHIVE,
    archive_factory: Callable[[str], Any] | None = None,
) -> dict[str, object]:
    """Retrieve only the files needed for released 3D trajectory evaluation."""

    factory = _archive_factory if archive_factory is None else archive_factory
    output = Path(output_dir)
    with ExitStack() as stack:
        data_archive = stack.enter_context(factory(data_archive_url))
        experiments_archive = stack.enter_context(factory(experiments_archive_url))
        available = _available_cases(data_archive, experiments_archive)
        selected = available if cases is None else tuple(dict.fromkeys(cases))
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(
                "unknown or incomplete PhysTwin cases: " + ", ".join(unknown)
            )
        records: dict[str, dict[str, object]] = {}
        for case in selected:
            case_dir = output / case
            files: dict[str, object] = {}
            for filename in EVALUATION_FILENAMES:
                member = f"data/different_types/{case}/{filename}"
                files[filename] = _retrieve_member(
                    data_archive,
                    member,
                    case_dir / filename,
                )
            inference_member = f"experiments/{case}/inference.pkl"
            files["inference.pkl"] = _retrieve_member(
                experiments_archive,
                inference_member,
                case_dir / "inference.pkl",
            )
            records[case] = {"files": files}

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "data": data_archive_url,
            "experiments": experiments_archive_url,
        },
        "available_cases": list(available),
        "selected_cases": list(selected),
        "cases": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "evaluation_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def fetch_phystwin_additional_evaluation_subset(
    output_dir: str | Path,
    *,
    cases: Iterable[str] | None = None,
    archive_url: str = DEFAULT_ADDITIONAL_ARCHIVE,
    archive_factory: Callable[[str], Any] | None = None,
) -> dict[str, object]:
    """Retrieve the label-free additional cloth trajectory evaluation subset."""

    factory = _archive_factory if archive_factory is None else archive_factory
    output = Path(output_dir)
    with factory(archive_url) as archive:
        available = _available_additional_cases(archive)
        selected = available if cases is None else tuple(dict.fromkeys(cases))
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(
                "unknown or incomplete additional PhysTwin cases: "
                + ", ".join(unknown)
            )
        records: dict[str, dict[str, object]] = {}
        for case in selected:
            case_dir = output / case
            files: dict[str, object] = {}
            for filename in ADDITIONAL_EVALUATION_FILENAMES:
                member = (
                    f"additional_data/data/different_types/{case}/{filename}"
                )
                files[filename] = _retrieve_member(
                    archive,
                    member,
                    case_dir / filename,
                )
            inference_member = f"additional_data/experiments/{case}/inference.pkl"
            files["inference.pkl"] = _retrieve_member(
                archive,
                inference_member,
                case_dir / "inference.pkl",
            )
            records[case] = {"files": files}

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": archive_url,
        "available_cases": list(available),
        "selected_cases": list(selected),
        "cases": records,
        "manual_track_labels": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "additional_evaluation_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest
