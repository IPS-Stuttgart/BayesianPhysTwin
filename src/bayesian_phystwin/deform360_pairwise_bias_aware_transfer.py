"""Outcome-blind transfer contract for the already-open Deform360 source panel."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_bias_aware_prospective_protocol import (
    EXPECTED_FRAME_COUNT,
    EXPECTED_UPDATE_FRAMES,
)
from .deform360_online_belief_evaluation import (
    EXPECTED_SOURCE_EPISODES,
    _validate_deform360_outcome_manifest,
)
from .deform360_pairwise_bias_aware_development import PROTOCOL_ID

ARTIFACT_KIND = "Deform360PairwiseBiasAwareSourceTransferManifest"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "transfer_manifest.json"
ROOT_NAMES = ("source", "measurement", "uncertainty", "selected_baseline")
FILE_ROLES = (
    "prediction_seal",
    "prediction_archive",
    "source_outcome",
    "source_target",
    "measurement",
    "uncertainty",
    "selected_baseline",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_case_records() -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            f"{object_id}-ep{episode_id:04d}",
            object_id,
            int(episode_id),
        )
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def _source_archive_path(case_dir: Path, seal: Mapping[str, Any]) -> Path:
    archive = seal.get("prediction_archive")
    _require(isinstance(archive, Mapping), "prediction seal lacks archive metadata")
    declared = Path(str(archive.get("path", "")))
    _require(bool(declared.name), "prediction seal archive path is empty")
    candidates = (case_dir / declared.name, declared)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        expected = archive.get("file_sha256")
        _require(
            isinstance(expected, str) and len(expected) == 64,
            "prediction seal lacks archive SHA-256",
        )
        _require(
            _sha256(candidate) == expected,
            f"prediction archive checksum changed: {candidate}",
        )
        return candidate.resolve()
    raise FileNotFoundError(declared)


def _validate_case_contract(
    *,
    case: str,
    object_id: str,
    episode_id: int,
    prediction_seal_path: Path,
    prediction_archive_path: Path,
    outcome_path: Path,
    target_path: Path,
    measurement_path: Path,
    uncertainty_path: Path,
    selected_baseline_path: Path,
) -> dict[str, int]:
    seal = _read_json_object(prediction_seal_path)
    outcome = _read_json_object(outcome_path)
    _require(seal.get("object_id") == object_id, f"{case}: object ID changed")
    _require(int(seal.get("episode_id", -1)) == episode_id, f"{case}: episode changed")
    _require(
        seal.get("episode_key") == f"{object_id}/{episode_id}",
        f"{case}: episode key changed",
    )
    archive = seal.get("prediction_archive")
    _require(isinstance(archive, Mapping), f"{case}: archive metadata missing")
    _require(
        Path(str(archive.get("path", ""))).name == prediction_archive_path.name,
        f"{case}: staged archive basename differs from the seal",
    )
    _require(
        archive.get("file_sha256") == _sha256(prediction_archive_path),
        f"{case}: staged archive differs from the seal",
    )
    _validate_deform360_outcome_manifest(
        prediction_seal_path,
        target_path,
        seal,
        outcome,
    )

    with np.load(prediction_archive_path, allow_pickle=False) as stored:
        required = {
            "driven_readout_m",
            "zero_action_readout_m",
            "action_support",
            "frame_zero_points_m",
        }
        _require(required <= set(stored.files), f"{case}: physical archive incomplete")
        driven = np.asarray(stored["driven_readout_m"])
        zero = np.asarray(stored["zero_action_readout_m"])
        support = np.asarray(stored["action_support"])
        frame_zero = np.asarray(stored["frame_zero_points_m"])
    _require(
        driven.ndim == 3
        and driven.shape[0] == EXPECTED_FRAME_COUNT
        and driven.shape[2] == 3,
        f"{case}: physical trajectory shape changed",
    )
    _require(zero.shape == driven.shape, f"{case}: zero-action shape changed")
    _require(
        frame_zero.shape == driven.shape[1:],
        f"{case}: frame-zero geometry shape changed",
    )
    _require(
        support.shape == (driven.shape[1],),
        f"{case}: action-support shape changed",
    )
    _require(
        np.all(np.isfinite(driven))
        and np.all(np.isfinite(zero))
        and np.all(np.isfinite(frame_zero))
        and np.all(np.isfinite(support)),
        f"{case}: physical archive is non-finite",
    )
    _require(
        np.all((support >= 0.0) & (support <= 1.0)),
        f"{case}: action support is outside [0, 1]",
    )
    _require(
        np.array_equal(driven[0], frame_zero)
        and np.array_equal(zero[0], frame_zero),
        f"{case}: physical material identity changed at frame zero",
    )

    with np.load(selected_baseline_path, allow_pickle=False) as stored:
        _require(
            "selected_raw_backbone" in stored.files,
            f"{case}: selected baseline missing",
        )
        baseline = np.asarray(stored["selected_raw_backbone"])
    _require(
        baseline.shape == driven.shape,
        f"{case}: selected baseline shape differs from physical trajectory",
    )
    _require(
        np.all(np.isfinite(baseline)),
        f"{case}: selected baseline is non-finite",
    )
    _require(
        np.array_equal(baseline[0], frame_zero),
        f"{case}: selected baseline changed frame-zero identity",
    )

    with np.load(measurement_path, allow_pickle=False) as stored:
        required = {
            "measurement_m",
            "measurement_visibility",
            "measurement_validity",
            "center_ids",
            "selected_cameras",
            "update_frames",
            "triangulation_inlier_view_count",
            "triangulation_median_reprojection_px",
        }
        _require(required <= set(stored.files), f"{case}: measurement incomplete")
        measurement = np.asarray(stored["measurement_m"])
        visibility = np.asarray(stored["measurement_visibility"], dtype=bool)
        validity = np.asarray(stored["measurement_validity"], dtype=bool)
        center_ids = np.asarray(stored["center_ids"], dtype=np.int64)
        selected_cameras = np.asarray(stored["selected_cameras"])
        update_frames = tuple(int(value) for value in stored["update_frames"])
        inlier_count = np.asarray(stored["triangulation_inlier_view_count"])
        reprojection = np.asarray(stored["triangulation_median_reprojection_px"])
    _require(
        measurement.shape == driven.shape,
        f"{case}: measurement trajectory shape changed",
    )
    _require(
        visibility.shape == driven.shape[:2]
        and validity.shape == driven.shape[:2],
        f"{case}: measurement support shape changed",
    )
    _require(
        center_ids.ndim == 1
        and len(center_ids) > 0
        and len(np.unique(center_ids)) == len(center_ids)
        and np.all((center_ids >= 0) & (center_ids < driven.shape[1])),
        f"{case}: invalid center identities",
    )
    _require(
        selected_cameras.ndim == 1 and len(selected_cameras) >= 2,
        f"{case}: fewer than two selected cameras",
    )
    _require(
        update_frames == EXPECTED_UPDATE_FRAMES,
        f"{case}: update frames changed",
    )
    expected_diagnostic = (len(EXPECTED_UPDATE_FRAMES), len(center_ids))
    _require(
        inlier_count.shape == expected_diagnostic
        and reprojection.shape == expected_diagnostic,
        f"{case}: triangulation diagnostics changed shape",
    )
    supported = visibility & validity
    _require(
        np.all(np.isfinite(measurement[supported])),
        f"{case}: supported measurement is non-finite",
    )

    with np.load(uncertainty_path, allow_pickle=False) as stored:
        required = {
            "measurement_covariance_m2",
            "measurement_covariance_valid",
        }
        _require(required <= set(stored.files), f"{case}: uncertainty incomplete")
        covariance = np.asarray(stored["measurement_covariance_m2"])
        covariance_valid = np.asarray(
            stored["measurement_covariance_valid"],
            dtype=bool,
        )
    _require(
        covariance.shape == (*driven.shape[:2], 3, 3),
        f"{case}: covariance shape changed",
    )
    _require(
        covariance_valid.shape == driven.shape[:2],
        f"{case}: covariance-valid shape changed",
    )
    _require(
        np.all(np.isfinite(covariance[covariance_valid])),
        f"{case}: valid covariance is non-finite",
    )
    diagonal = np.diagonal(covariance[covariance_valid], axis1=-2, axis2=-1)
    _require(
        np.all(diagonal >= 0.0),
        f"{case}: covariance has a negative diagonal",
    )
    return {
        "frame_count": int(driven.shape[0]),
        "node_count": int(driven.shape[1]),
        "center_count": int(len(center_ids)),
        "camera_count": int(len(selected_cameras)),
    }


def _file_record(root: str, relative_path: Path, source: Path) -> dict[str, Any]:
    _require(root in ROOT_NAMES, f"unknown logical root: {root}")
    _require(source.is_file(), f"missing transfer file: {source}")
    _require(
        not relative_path.is_absolute() and ".." not in relative_path.parts,
        "transfer path escapes its logical root",
    )
    return {
        "root": root,
        "relative_path": relative_path.as_posix(),
        "size_bytes": int(source.stat().st_size),
        "sha256": _sha256(source),
    }


def build_open27_transfer_manifest(
    source_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    selected_baseline_root: str | Path,
) -> dict[str, Any]:
    """Inventory and validate the exact open-27 byte streams without scoring."""

    roots = {
        "source": Path(source_root).resolve(),
        "measurement": Path(measurement_root).resolve(),
        "uncertainty": Path(uncertainty_root).resolve(),
        "selected_baseline": Path(selected_baseline_root).resolve(),
    }
    cases: list[dict[str, Any]] = []
    seen_files: set[tuple[str, str]] = set()
    for case, object_id, episode_id in _expected_case_records():
        source_case = roots["source"] / case
        seal_path = source_case / "prediction_seal.json"
        outcome_path = source_case / "outcome.json"
        target_path = source_case / "target_data.pkl"
        measurement_path = roots["measurement"] / case / "measurement.npz"
        uncertainty_path = (
            roots["uncertainty"] / case / "measurement_cycle_uncertainty.npz"
        )
        baseline_path = roots["selected_baseline"] / f"{case}.npz"
        for path in (
            seal_path,
            outcome_path,
            target_path,
            measurement_path,
            uncertainty_path,
            baseline_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        seal = _read_json_object(seal_path)
        prediction_path = _source_archive_path(source_case, seal)
        schema = _validate_case_contract(
            case=case,
            object_id=object_id,
            episode_id=episode_id,
            prediction_seal_path=seal_path,
            prediction_archive_path=prediction_path,
            outcome_path=outcome_path,
            target_path=target_path,
            measurement_path=measurement_path,
            uncertainty_path=uncertainty_path,
            selected_baseline_path=baseline_path,
        )
        sources = {
            "prediction_seal": (
                "source",
                Path(case) / "prediction_seal.json",
                seal_path,
            ),
            "prediction_archive": (
                "source",
                Path(case) / prediction_path.name,
                prediction_path,
            ),
            "source_outcome": (
                "source",
                Path(case) / "outcome.json",
                outcome_path,
            ),
            "source_target": (
                "source",
                Path(case) / "target_data.pkl",
                target_path,
            ),
            "measurement": (
                "measurement",
                Path(case) / "measurement.npz",
                measurement_path,
            ),
            "uncertainty": (
                "uncertainty",
                Path(case) / "measurement_cycle_uncertainty.npz",
                uncertainty_path,
            ),
            "selected_baseline": (
                "selected_baseline",
                Path(f"{case}.npz"),
                baseline_path,
            ),
        }
        files = {
            role: _file_record(root, relative, path)
            for role, (root, relative, path) in sources.items()
        }
        for record in files.values():
            identity = (record["root"], record["relative_path"])
            _require(identity not in seen_files, "transfer file is reused")
            seen_files.add(identity)
        cases.append(
            {
                "case": case,
                "object_id": object_id,
                "episode_id": episode_id,
                "schema": schema,
                "files": files,
            }
        )
    payload: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "logical_roots": list(ROOT_NAMES),
        "case_count": len(cases),
        "file_count": len(seen_files),
        "cases": cases,
        "claim_boundary": (
            "This manifest inventories already-open source-development bytes. "
            "Target payloads are copied and hashed as opaque files; no score or "
            "fresh-object outcome is read or produced."
        ),
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_manifest(payload: Mapping[str, Any]) -> None:
    _require(payload.get("artifact_kind") == ARTIFACT_KIND, "bad artifact kind")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "bad schema version")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        payload.get("logical_roots") == list(ROOT_NAMES),
        "logical roots changed",
    )
    expected = _expected_case_records()
    cases = payload.get("cases")
    _require(isinstance(cases, list), "manifest cases are missing")
    _require(payload.get("case_count") == len(expected), "case count changed")
    _require(len(cases) == len(expected), "manifest case panel is incomplete")
    seen_files: set[tuple[str, str]] = set()
    for row, (case, object_id, episode_id) in zip(cases, expected, strict=True):
        _require(isinstance(row, Mapping), "manifest case is not an object")
        _require(row.get("case") == case, "manifest case order changed")
        _require(row.get("object_id") == object_id, f"{case}: object ID changed")
        _require(row.get("episode_id") == episode_id, f"{case}: episode changed")
        files = row.get("files")
        _require(isinstance(files, Mapping), f"{case}: files missing")
        _require(set(files) == set(FILE_ROLES), f"{case}: file roles changed")
        for record in files.values():
            _require(isinstance(record, Mapping), f"{case}: bad file record")
            root = record.get("root")
            relative = Path(str(record.get("relative_path", "")))
            digest = record.get("sha256")
            size = record.get("size_bytes")
            _require(root in ROOT_NAMES, f"{case}: unknown logical root")
            _require(
                bool(relative.parts)
                and not relative.is_absolute()
                and ".." not in relative.parts,
                f"{case}: transfer path escapes its root",
            )
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest),
                f"{case}: invalid file SHA-256",
            )
            _require(
                isinstance(size, int) and size >= 0,
                f"{case}: invalid file size",
            )
            identity = (str(root), relative.as_posix())
            _require(identity not in seen_files, "transfer file is reused")
            seen_files.add(identity)
    _require(payload.get("file_count") == len(seen_files), "file count changed")
    _require(
        payload.get("manifest_sha256") == _canonical_sha256(payload),
        "transfer manifest digest changed",
    )


def _bundle_roots(bundle_root: str | Path) -> dict[str, Path]:
    bundle = Path(bundle_root).resolve()
    return {name: bundle / name for name in ROOT_NAMES}


def validate_open27_transfer_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Rehash and schema-check a staged bundle before any source evaluation."""

    bundle = Path(bundle_root).resolve()
    manifest_path = bundle / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = _read_json_object(manifest_path)
    _validate_manifest(payload)
    roots = _bundle_roots(bundle)
    total_bytes = 0
    for row in payload["cases"]:
        paths: dict[str, Path] = {}
        for role, record in row["files"].items():
            path = roots[record["root"]] / record["relative_path"]
            _require(path.is_file(), f"staged file is missing: {path}")
            _require(
                path.stat().st_size == record["size_bytes"],
                f"staged file size changed: {path}",
            )
            _require(
                _sha256(path) == record["sha256"],
                f"staged file digest changed: {path}",
            )
            paths[role] = path
            total_bytes += int(record["size_bytes"])
        schema = _validate_case_contract(
            case=row["case"],
            object_id=row["object_id"],
            episode_id=int(row["episode_id"]),
            prediction_seal_path=paths["prediction_seal"],
            prediction_archive_path=paths["prediction_archive"],
            outcome_path=paths["source_outcome"],
            target_path=paths["source_target"],
            measurement_path=paths["measurement"],
            uncertainty_path=paths["uncertainty"],
            selected_baseline_path=paths["selected_baseline"],
        )
        _require(schema == row["schema"], f"{row['case']}: schema changed")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256(manifest_path),
        "manifest_sha256": payload["manifest_sha256"],
        "case_count": int(payload["case_count"]),
        "file_count": int(payload["file_count"]),
        "total_bytes": total_bytes,
        "roots": {name: str(path) for name, path in roots.items()},
    }


def stage_open27_transfer_bundle(
    source_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    selected_baseline_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Copy only manifest-bound files and verify the complete staged bundle."""

    source_roots = {
        "source": Path(source_root).resolve(),
        "measurement": Path(measurement_root).resolve(),
        "uncertainty": Path(uncertainty_root).resolve(),
        "selected_baseline": Path(selected_baseline_root).resolve(),
    }
    payload = build_open27_transfer_manifest(**{
        "source_root": source_roots["source"],
        "measurement_root": source_roots["measurement"],
        "uncertainty_root": source_roots["uncertainty"],
        "selected_baseline_root": source_roots["selected_baseline"],
    })
    destination_path = Path(destination).resolve()
    _require(not destination_path.exists(), "transfer destination already exists")
    partial = destination_path.with_name(f"{destination_path.name}.partial")
    _require(not partial.exists(), "partial transfer destination already exists")
    try:
        partial.mkdir(parents=True)
        for name in ROOT_NAMES:
            (partial / name).mkdir()
        for row in payload["cases"]:
            for record in row["files"].values():
                source = source_roots[record["root"]] / record["relative_path"]
                if record["root"] == "source" and not source.is_file():
                    seal = _read_json_object(
                        source_roots["source"]
                        / row["case"]
                        / "prediction_seal.json"
                    )
                    source = _source_archive_path(
                        source_roots["source"] / row["case"],
                        seal,
                    )
                target = partial / record["root"] / record["relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        manifest_path = partial / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_open27_transfer_bundle(partial)
        partial.rename(destination_path)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return validate_open27_transfer_bundle(destination_path)


def write_open27_transfer_manifest(
    source_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    selected_baseline_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a portable inventory without copying any payload."""

    output = Path(output_path).resolve()
    _require(not output.exists(), "transfer manifest output already exists")
    payload = build_open27_transfer_manifest(
        source_root,
        measurement_root,
        uncertainty_root,
        selected_baseline_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_path": str(output),
        "manifest_file_sha256": _sha256(output),
        "manifest_sha256": payload["manifest_sha256"],
        "case_count": payload["case_count"],
        "file_count": payload["file_count"],
    }


__all__ = [
    "ARTIFACT_KIND",
    "FILE_ROLES",
    "MANIFEST_FILENAME",
    "ROOT_NAMES",
    "SCHEMA_VERSION",
    "build_open27_transfer_manifest",
    "stage_open27_transfer_bundle",
    "validate_open27_transfer_bundle",
    "write_open27_transfer_manifest",
]
