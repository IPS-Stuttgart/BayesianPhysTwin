#!/usr/bin/env python3
"""Project one registered PokeFlex ZIP to the exact frozen runner inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_public_transfer_audit import (  # noqa: E402
    SOURCE_PROJECTION_RUNNER_FILE_SHA256,
    file_sha256,
    validate_public_transfer_protocol,
)

_ROBOT = re.compile(r"^robot_data\.json$")
_CALIBRATION = re.compile(r"^kinect/(?P<camera>[01])/camera_parameters\.json$")
_DEPTH = re.compile(r"^kinect/(?P<camera>[01])/depth/[0-9]{5}\.png$")
_MESH = re.compile(r"^meshes/mesh-f[0-9]{5}\.obj$")


def _kind(relative: str) -> tuple[str, str | None] | None:
    if _ROBOT.fullmatch(relative):
        return "robot", None
    match = _CALIBRATION.fullmatch(relative)
    if match:
        return "calibration", match.group("camera")
    match = _DEPTH.fullmatch(relative)
    if match:
        return "depth", match.group("camera")
    if _MESH.fullmatch(relative):
        return "mesh", None
    return None


def _manifest_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_public78_retrospective_v6.json"
        ),
    )
    args = parser.parse_args()

    if file_sha256(Path(__file__)) != SOURCE_PROJECTION_RUNNER_FILE_SHA256:
        raise ValueError("source projection runner bytes changed")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validation = validate_public_transfer_protocol(protocol)
    archive = args.archive.resolve()
    take_id = archive.stem
    if take_id not in validation["retrospective_take_ids"]:
        raise ValueError("archive is outside the frozen retrospective cohort")
    expected = protocol["archive_inventory"]["takes"][take_id]
    if archive.stat().st_size != int(expected["bytes"]):
        raise ValueError("source archive size changed")
    if file_sha256(archive) != expected["sha256"]:
        raise ValueError("source archive bytes changed")

    output = args.output.resolve()
    if output.suffixes[-2:] != [".tar", ".zst"]:
        raise ValueError("projected output must end in .tar.zst")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_tar = output.with_suffix("")
    temporary_output = output.with_name(output.name + ".partial")
    manifest_path = output.with_name(output.name + ".manifest.json")
    for path in (temporary_tar, temporary_output):
        path.unlink(missing_ok=True)

    selected = []
    kinds: Counter[tuple[str, str | None]] = Counter()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or len(path.parts) < 2 or path.parts[0] != take_id:
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            kind = _kind(relative)
            if kind is not None:
                selected.append((relative, info, kind))
                kinds[kind] += 1
        selected.sort(key=lambda row: row[0])
        if kinds[("robot", None)] != 1:
            raise ValueError("projected robot inventory changed")
        if kinds[("calibration", "0")] != 1 or kinds[("calibration", "1")] != 1:
            raise ValueError("projected calibration inventory changed")
        if kinds[("depth", "0")] <= 0 or kinds[("depth", "0")] != kinds[("depth", "1")]:
            raise ValueError("projected depth inventory changed")
        if kinds[("mesh", None)] <= 0:
            raise ValueError("projected mesh inventory changed")

        uncompressed_bytes = 0
        with tarfile.open(temporary_tar, "w") as target:
            for relative, info, _ in selected:
                record = tarfile.TarInfo(f"{take_id}/{relative}")
                record.size = info.file_size
                record.mtime = 0
                record.mode = 0o644
                record.uid = record.gid = 0
                record.uname = record.gname = ""
                with source.open(info) as member:
                    target.addfile(record, member)
                uncompressed_bytes += info.file_size

    subprocess.run(
        ["zstd", "-q", "-3", "-f", str(temporary_tar), "-o", str(temporary_output)],
        check=True,
    )
    temporary_tar.unlink()
    if output.exists() and file_sha256(output) != file_sha256(temporary_output):
        temporary_output.unlink()
        raise ValueError("existing projected archive differs")
    temporary_output.replace(output)
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPublicTransferSourceProjection",
        "protocol_sha256": protocol["protocol_sha256"],
        "projection_runner_file_sha256": SOURCE_PROJECTION_RUNNER_FILE_SHA256,
        "take_id": take_id,
        "source_archive_sha256": expected["sha256"],
        "source_archive_bytes": expected["bytes"],
        "member_policy": (
            "robot_data.json, Kinect 0/1 calibration and depth, and mesh OBJ bytes"
        ),
        "member_count": len(selected),
        "uncompressed_member_bytes": uncompressed_bytes,
        "projected_archive_sha256": file_sha256(output),
        "projected_archive_bytes": output.stat().st_size,
        "target_geometry_decoded": False,
        "outcome_metric_computed": False,
    }
    payload["manifest_sha256"] = _manifest_sha256(payload)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered:
        raise ValueError("existing projection manifest differs")
    manifest_path.write_text(rendered, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
