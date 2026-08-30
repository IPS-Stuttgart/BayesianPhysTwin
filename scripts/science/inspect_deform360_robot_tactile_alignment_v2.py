#!/usr/bin/env python3
"""Inspect development-only Deform360 robot/tactile carrier schemas.

This diagnostic opens existing development robot arrays and timing metadata to
identify an executable action-conditioned forecasting panel.  It does not open
reserved objects, camera pixels, geometry, point clouds, or target scores.

The mounted processed tree uses a legacy object-array ``robot.npy`` contract.
Pickle-backed loading is therefore allowed only for that exact trusted dataset
root and only to summarize the released robot structure.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-robot-tactile-alignment-inspection-v2"
TACTILE_RE = re.compile(r"tactile", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{13,})(?!\d)")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def episode_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("sequences", metadata.get("episodes", metadata.get("takes")))
    if isinstance(raw, Mapping):
        items = sorted(
            raw.items(),
            key=lambda item: (
                0 if str(item[0]).isdigit() else 1,
                int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            ),
        )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(enumerate(raw))
    else:
        return []
    result: list[dict[str, Any]] = []
    for raw_id, raw_record in items:
        if not isinstance(raw_record, Mapping):
            continue
        result.append(
            {
                "episode_id": int(raw_id) if str(raw_id).isdigit() else len(result),
                "action": raw_record.get("action"),
                "bimanual": raw_record.get("bimanual"),
                "nonprehensile": raw_record.get("nonprehensile"),
            }
        )
    return result


def _scalar_summary(value: object) -> object:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return type(value).__name__


def _structure(value: object, *, depth: int = 0) -> dict[str, Any]:
    if depth >= 3:
        return {"type": type(value).__name__}
    if isinstance(value, np.ndarray):
        result: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size and value.dtype == object:
            flattened = value.reshape(-1)
            result["first"] = _structure(flattened[0], depth=depth + 1)
            result["last"] = _structure(flattened[-1], depth=depth + 1)
        return result
    if isinstance(value, Mapping):
        keys = sorted(map(str, value.keys()))
        result = {"type": type(value).__name__, "keys": keys}
        summaries: dict[str, Any] = {}
        for key in keys[:24]:
            try:
                summaries[key] = _structure(value[key], depth=depth + 1)  # type: ignore[index]
            except (KeyError, TypeError):
                continue
        result["values"] = summaries
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {"type": type(value).__name__, "length": len(value)}
        if len(value):
            result["first"] = _structure(value[0], depth=depth + 1)
            result["last"] = _structure(value[-1], depth=depth + 1)
        return result
    return {"type": type(value).__name__, "value": _scalar_summary(value)}


def npy_info(path: Path) -> dict[str, Any]:
    pickle_backed = False
    safe_error: str | None = None
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        safe_error = f"{type(error).__name__}: {error}"
        try:
            value = np.load(path, allow_pickle=True)
            pickle_backed = True
        except (OSError, ValueError, ImportError, EOFError) as pickle_error:
            return {
                "path": str(path),
                "readable": False,
                "readable_without_pickle": False,
                "safe_load_error": safe_error,
                "error": f"{type(pickle_error).__name__}: {pickle_error}",
                "size_bytes": int(path.stat().st_size),
            }
    array = np.asarray(value)
    result: dict[str, Any] = {
        "path": str(path),
        "readable": True,
        "readable_without_pickle": not pickle_backed,
        "pickle_backed_official_legacy_contract": pickle_backed,
        "safe_load_error": safe_error,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size_bytes": int(path.stat().st_size),
        "structure": _structure(array),
    }
    if array.dtype.kind in "iuf" and array.size:
        flattened = array.reshape(-1)
        indices = np.linspace(
            0, len(flattened) - 1, min(len(flattened), 4096), dtype=int
        )
        sample = np.asarray(flattened[indices], dtype=np.float64)
        finite = np.isfinite(sample)
        result["sampled_finite_fraction"] = float(np.mean(finite))
        if np.any(finite):
            result["sampled_minimum"] = float(np.min(sample[finite]))
            result["sampled_maximum"] = float(np.max(sample[finite]))
    return result


def text_timing_info(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {
            "path": str(path),
            "readable": False,
            "error": f"{type(error).__name__}: {error}",
            "size_bytes": int(path.stat().st_size),
        }
    values: list[float] = []
    for line in lines:
        match = TIMESTAMP_RE.search(line)
        if match is not None:
            values.append(float(match.group(1)))
            continue
        try:
            values.append(float(line.strip().split()[0]))
        except (ValueError, IndexError):
            continue
    flat = np.asarray(values, dtype=np.float64)
    finite = flat[np.isfinite(flat)]
    return {
        "path": str(path),
        "readable": bool(len(finite)),
        "line_count": len(lines),
        "count": int(len(flat)),
        "finite_count": int(len(finite)),
        "minimum": float(np.min(finite)) if len(finite) else None,
        "maximum": float(np.max(finite)) if len(finite) else None,
        "median_delta": float(np.median(np.diff(finite))) if len(finite) >= 2 else None,
        "size_bytes": int(path.stat().st_size),
    }


def tactile_sample_count(path: Path) -> dict[str, Any]:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[1:] != (16, 32):
            raise ValueError(f"unexpected headered shape {array.shape}")
        return {
            "path": str(path),
            "format": "npy",
            "sample_count": int(array.shape[0]),
            "dtype": str(array.dtype),
            "size_bytes": int(path.stat().st_size),
        }
    except (OSError, ValueError):
        frame_bytes = 16 * 32 * np.dtype(np.float32).itemsize
        size = path.stat().st_size
        return {
            "path": str(path),
            "format": "headerless-float32"
            if size % frame_bytes == 0
            else "unsupported",
            "sample_count": int(size // frame_bytes)
            if size % frame_bytes == 0
            else None,
            "dtype": "float32" if size % frame_bytes == 0 else None,
            "size_bytes": int(size),
        }


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def episode_directory(processed_object: Path, episode_id: int) -> Path | None:
    for name in (
        f"episode_{episode_id}",
        f"episode_{episode_id:04d}",
        f"episode-{episode_id}",
    ):
        candidate = processed_object / name
        if candidate.is_dir():
            return candidate
    return None


def _frame_count(info: Mapping[str, Any]) -> int:
    shape = info.get("shape")
    if not isinstance(shape, list) or not shape:
        return 0
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return 0


def inspect_object(root: Path, object_id: str) -> dict[str, Any]:
    raw_object = root / "raw-repository" / "raw" / object_id
    processed_object = root / "processed-repository" / "processed" / object_id
    metadata_path = raw_object / "metadata.json"
    if not metadata_path.is_file():
        return {
            "object_id": object_id,
            "supported": False,
            "reason": "metadata-missing",
        }
    episodes = episode_records(load_json(metadata_path))
    tactile_groups: list[tuple[str, list[Path]]] = []
    for child in sorted(
        (path for path in raw_object.iterdir() if path.is_dir()), key=lambda p: p.name
    ):
        if not TACTILE_RE.search(child.name):
            continue
        files = sorted(
            (
                path
                for path in child.glob("*.npy")
                if not path.name.lower().startswith("median_")
                and path.stat().st_size > 0
            ),
            key=lambda path: path.name,
        )
        if len(files) == len(episodes):
            tactile_groups.append((child.name, files))
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        directory = episode_directory(processed_object, episode_id)
        if directory is None:
            continue
        robot_candidates = sorted(
            {
                path.resolve()
                for path in directory.rglob("*.npy")
                if "robot" in path.name.lower()
                or "pose" in path.name.lower()
                or "opening" in path.name.lower()
            },
            key=str,
        )
        robot_arrays = [
            {**npy_info(path), "path": relative(path, root)}
            for path in robot_candidates
        ]
        timing_candidates = sorted(
            {
                path.resolve()
                for path in directory.rglob("*.txt")
                if "timestamp" in path.name.lower() or "time" in path.name.lower()
            },
            key=str,
        )
        neighboring_paths: list[str] = []
        for path in sorted(directory.rglob("*"), key=str):
            if len(path.relative_to(directory).parts) > 3:
                continue
            if path.is_file() and any(
                token in path.name.lower()
                for token in (
                    "robot",
                    "pose",
                    "opening",
                    "align",
                    "timestamp",
                    "tactile",
                    "control",
                )
            ):
                neighboring_paths.append(relative(path, root))
        tactile = []
        for group_name, files in tactile_groups:
            path = files[episode_id]
            sidecar = path.with_suffix(".txt")
            tactile.append(
                {
                    "group": group_name,
                    "payload": tactile_sample_count(path),
                    "timestamps": text_timing_info(sidecar)
                    if sidecar.is_file()
                    else None,
                }
            )
        episode_rows.append(
            {
                **episode,
                "processed_directory": relative(directory, root),
                "robot_arrays": robot_arrays,
                "timing_files": [
                    {**text_timing_info(path), "path": relative(path, root)}
                    for path in timing_candidates[:8]
                ],
                "neighboring_paths": neighboring_paths[:80],
                "raw_tactile": tactile,
            }
        )
    usable = [
        row
        for row in episode_rows
        if any(
            item.get("readable") and _frame_count(item) >= 12
            for item in row["robot_arrays"]
        )
        and any(
            int(entry["payload"].get("sample_count") or 0) >= 12
            and entry.get("timestamps") is not None
            and entry["timestamps"].get("readable")
            for entry in row["raw_tactile"]
        )
    ]
    return {
        "object_id": object_id,
        "supported": bool(usable),
        "episode_count": len(episodes),
        "processed_episode_count": len(episode_rows),
        "robot_tactile_usable_episode_count": len(usable),
        "tactile_group_count": len(tactile_groups),
        "episodes": episode_rows,
    }


def inspect(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    expected = Path(str(protocol["dataset"]["root"]))
    resolved = root.resolve(strict=True)
    if resolved != expected:
        raise ValueError(f"dataset root changed: {resolved} != {expected}")
    reserved = set(map(str, protocol["forbidden_reserved_object_ids"]))
    development = list(map(str, protocol["development_object_ids"]))
    if reserved & set(development):
        raise ValueError("development and reserved rosters overlap")
    priority = [
        object_id
        for object_id in development
        if any(token in object_id for token in ("rope", "cable", "line"))
    ]
    remaining = [object_id for object_id in development if object_id not in priority]
    objects = [
        inspect_object(resolved, object_id) for object_id in priority + remaining
    ]
    supported = [row for row in objects if row["supported"]]
    supported.sort(
        key=lambda row: (
            -int(row["robot_tactile_usable_episode_count"]),
            row["object_id"],
        )
    )
    result = {
        "schema": SCHEMA,
        "schema_version": 2,
        "dataset_root": str(resolved),
        "information_boundary": {
            "development_robot_arrays_opened": True,
            "official_legacy_pickle_contract_loaded": True,
            "development_tactile_payload_headers_opened": True,
            "development_timing_metadata_opened": True,
            "camera_pixels_decoded": False,
            "geometry_or_point_cloud_opened": False,
            "target_scores_computed": False,
            "reserved_object_payloads_opened": False,
        },
        "summary": {
            "inspected_object_count": len(objects),
            "supported_object_count": len(supported),
            "supported_episode_count": sum(
                int(row["robot_tactile_usable_episode_count"]) for row in supported
            ),
            "recommended_object_ids": [row["object_id"] for row in supported[:8]],
        },
        "objects": objects,
    }
    return result


def make_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Deform360 robot/tactile alignment inspection v2",
        "",
        f"- Supported objects: **{summary['supported_object_count']}/{summary['inspected_object_count']}**",
        f"- Supported episodes: **{summary['supported_episode_count']}**",
        f"- Recommended panel: `{', '.join(summary['recommended_object_ids'])}`",
        "",
        "| Object | Usable robot+tactile episodes | Tactile groups |",
        "|---|---:|---:|",
    ]
    recommended = set(summary["recommended_object_ids"])
    for row in result["objects"]:
        if row["supported"] or row["object_id"] in recommended:
            lines.append(
                f"| `{row['object_id']}` | {row['robot_tactile_usable_episode_count']} | {row['tactile_group_count']} |"
            )
    lines.extend(
        [
            "",
            "This diagnostic opened development robot arrays and timing metadata only.",
            "No camera pixels, geometry, point clouds, scores, or reserved-object payloads were opened.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.data_root, args.protocol)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(make_report(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
