"""Source-only staging for PokeFlex eye-in-hand depth anchors."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from .pokeflex_independent_depth_protocol import (
    load_pokeflex_independent_depth_protocol,
)


_TAKE_PATTERN = re.compile(r"^(?P<object>.+)_T(?P<take>[0-9]+)$")
_DEPTH_PATTERN = re.compile(
    r"^(?P<take>[^/]+)/realsense/(?P<camera>[01])/depth/(?P<frame>[0-9]{5})\.png$"
)
_CALIBRATION_PATTERN = re.compile(
    r"^(?P<take>[^/]+)/realsense/(?P<camera>[01])/camera_parameters\.json$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _take_identity(path: Path) -> tuple[str, str]:
    match = _TAKE_PATTERN.match(path.stem)
    _require(match is not None, "PokeFlex archive name is not an object/take id")
    assert match is not None
    return match.group("object"), f"T{match.group('take')}"


def stage_pokeflex_independent_depth_source(
    archive_path: str | Path,
    destination_root: str | Path,
    protocol_path: str | Path,
) -> dict[str, object]:
    """Extract only RealSense depth and calibration from an outcome-open take."""

    archive = Path(archive_path).resolve()
    destination = Path(destination_root).resolve()
    protocol = load_pokeflex_independent_depth_protocol(protocol_path)
    boundary = protocol["payload"]["evidence_boundary"]
    object_name, take_number = _take_identity(archive)
    _require(
        object_name in boundary["development_objects"],
        "archive object is outside the development cohort",
    )
    _require(
        take_number in boundary["outcome_open_design_takes"],
        "archive take is not outcome-open for source design",
    )
    take_id = archive.stem
    destination_take = destination / take_id

    selected: list[tuple[zipfile.ZipInfo, str, str]] = []
    calibration_count = {"0": 0, "1": 0}
    depth_count = {"0": 0, "1": 0}
    with zipfile.ZipFile(archive) as payload:
        for member in payload.infolist():
            path = PurePosixPath(member.filename)
            _require(".." not in path.parts, "archive contains an unsafe member")
            depth_match = _DEPTH_PATTERN.match(member.filename)
            calibration_match = _CALIBRATION_PATTERN.match(member.filename)
            match = depth_match or calibration_match
            if match is None:
                continue
            _require(match.group("take") == take_id, "archive take root changed")
            relative = PurePosixPath(*path.parts[1:])
            camera = match.group("camera")
            kind = "depth" if depth_match is not None else "calibration"
            selected.append((member, str(relative), kind))
            if kind == "depth":
                depth_count[camera] += 1
            else:
                calibration_count[camera] += 1
        _require(
            calibration_count == {"0": 1, "1": 1},
            "archive RealSense calibration inventory changed",
        )
        _require(
            depth_count["0"] > 0 and depth_count["0"] == depth_count["1"],
            "archive RealSense depth inventory changed",
        )

        file_records = []
        for member, relative, kind in sorted(selected, key=lambda item: item[1]):
            content = payload.read(member)
            target = destination_take / relative
            if target.exists():
                _require(target.read_bytes() == content, f"staged file differs: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            file_records.append(
                {
                    "path": relative,
                    "kind": kind,
                    "size_bytes": len(content),
                    "sha256": _sha256_bytes(content),
                }
            )

    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexIndependentDepthSourceStage",
        "protocol_sha256": protocol["protocol_sha256"],
        "take_id": take_id,
        "archive": str(archive),
        "archive_sha256": _sha256_file(archive),
        "outcome_members_read": False,
        "future_selection_performed": False,
        "camera_depth_frame_count": depth_count,
        "files": file_records,
    }
    manifest = destination_take / "realsense_anchor_stage_manifest.json"
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if manifest.exists():
        _require(
            manifest.read_text(encoding="utf-8") == rendered,
            "existing source-stage manifest differs",
        )
    else:
        manifest.write_text(rendered, encoding="utf-8")
    result["manifest"] = str(manifest)
    return result
