#!/usr/bin/env python3
"""Header-only census of authorized DLO4/DLO5 parent-run artifacts.

The program is run only after a content-addressed DLO2/DLO3 source model has
been sealed and a separate target authorization exists. It hashes downloaded
files, lists archive members, and reads NumPy headers (shape/dtype/Fortran flag)
without loading numerical target values. JSON score/result payloads are opaque
hash-only records; only manifest/receipt metadata keys may be parsed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

import numpy as np

TARGET_DLOS = ("DLO4", "DLO5")
SOURCE_MODEL_SCHEMA = "bayesian-phystwin.sealed-hierarchical-missing-physics-model"
SAFE_JSON_NAME_TOKENS = ("manifest", "receipt", "metadata", "request", "protocol", "seal")
OPAQUE_JSON_NAME_TOKENS = ("result", "score", "metric", "evaluation", "target")
RELEVANT_TOKENS = (
    "dlo4",
    "dlo5",
    "dlo45",
    "target",
    "prediction",
    "physical",
    "baseline",
    "residual",
    "trajectory",
    "action",
    "contact",
    "force",
    "state",
    "position",
    "coefficient",
    "seal",
)
ROLE_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "residual": ("residual", "discrep", "correction", "delta"),
    "observation": ("ground_truth", "groundtruth", "truth", "observ", "measur", "reference", "real", "target"),
    "physical": ("physical", "baseline", "simulat", "rollout", "prediction_phys", "pred_phys"),
    "state": ("state", "position", "node", "point", "coord", "configuration", "trajectory"),
    "action": ("action", "command", "control", "input", "actuat", "endpoint", "end_effector", "robot"),
    "contact": ("contact", "force", "wrench", "torque", "grip"),
    "trajectory_id": ("trajectory_id", "trajectory_name", "case_id", "case_name", "sequence_id", "episode_id"),
    "time": ("time", "timestamp", "step"),
    "coefficient": ("coefficient", "coeff", "weight", "parameter"),
}


class CensusError(RuntimeError):
    """Raised when authorization or artifact provenance is invalid."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(name: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return [
        role
        for role, patterns in ROLE_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    ]


def target_identities(name: str) -> list[str]:
    normalized = name.lower()
    result = []
    for dlo in TARGET_DLOS:
        number = dlo[-1]
        if any(token in normalized for token in (dlo.lower(), f"dlo_{number}", f"dlo-{number}")):
            result.append(dlo)
    return result


def read_npy_header(stream: BinaryIO) -> dict[str, Any]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise CensusError(f"unsupported NPY version {version}")
    return {
        "shape": [int(value) for value in shape],
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
        "elements": int(np.prod(shape, dtype=np.int64)) if shape else 1,
        "npy_version": list(version),
    }


def inspect_npy(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return {"kind": "npy", **read_npy_header(stream)}


def inspect_npz(path: Path) -> dict[str, Any]:
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in sorted(archive.infolist(), key=lambda value: value.filename):
            if member.is_dir() or not member.filename.endswith(".npy"):
                continue
            with archive.open(member, "r") as stream:
                header = read_npy_header(stream)
            key = Path(member.filename).name.removesuffix(".npy")
            arrays[key] = {
                **header,
                "member": member.filename,
                "compressed_bytes": int(member.compress_size),
                "uncompressed_bytes": int(member.file_size),
                "roles": classify(key),
                "target_identities": target_identities(key),
            }
    return {"kind": "npz", "arrays": arrays}


def inspect_nested_zip(path: Path) -> dict[str, Any]:
    members = []
    numpy_headers = []
    with zipfile.ZipFile(path, "r") as archive:
        for member in sorted(archive.infolist(), key=lambda value: value.filename):
            if member.is_dir():
                continue
            record = {
                "member": member.filename,
                "compressed_bytes": int(member.compress_size),
                "uncompressed_bytes": int(member.file_size),
                "suffix": Path(member.filename).suffix.lower(),
                "roles": classify(member.filename),
                "target_identities": target_identities(member.filename),
            }
            members.append(record)
            if member.filename.endswith(".npy"):
                try:
                    with archive.open(member, "r") as stream:
                        header = read_npy_header(stream)
                except Exception as error:
                    numpy_headers.append({**record, "header_error": repr(error)})
                else:
                    numpy_headers.append({**record, **header})
            elif member.filename.endswith(".npz") and member.file_size <= 500_000_000:
                # Nested NPZ needs a seekable in-memory view. Values remain unread.
                try:
                    with archive.open(member, "r") as stream:
                        payload = stream.read()
                    with zipfile.ZipFile(io.BytesIO(payload), "r") as nested:
                        arrays = {}
                        for nested_member in sorted(nested.infolist(), key=lambda value: value.filename):
                            if nested_member.is_dir() or not nested_member.filename.endswith(".npy"):
                                continue
                            with nested.open(nested_member, "r") as stream:
                                header = read_npy_header(stream)
                            key = Path(nested_member.filename).name.removesuffix(".npy")
                            arrays[key] = {
                                **header,
                                "member": nested_member.filename,
                                "roles": classify(key),
                                "target_identities": target_identities(key),
                            }
                    numpy_headers.append({**record, "kind": "nested-npz", "arrays": arrays})
                except Exception as error:
                    numpy_headers.append({**record, "nested_npz_error": repr(error)})
    return {
        "kind": "zip",
        "member_count": len(members),
        "members": members,
        "numpy_headers": numpy_headers,
    }


def safe_json_metadata(path: Path) -> dict[str, Any]:
    lower = path.name.lower()
    if any(token in lower for token in OPAQUE_JSON_NAME_TOKENS):
        return {
            "kind": "json",
            "payload_parsed": False,
            "reason": "score/result/target JSON remains opaque during census",
        }
    if not any(token in lower for token in SAFE_JSON_NAME_TOKENS):
        return {
            "kind": "json",
            "payload_parsed": False,
            "reason": "JSON name is outside the metadata allow-list",
        }
    if path.stat().st_size > 5_000_000:
        return {"kind": "json", "payload_parsed": False, "reason": "metadata size limit"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"kind": "json", "payload_parsed": False, "parse_error": repr(error)}
    if not isinstance(value, dict):
        return {"kind": "json", "payload_parsed": True, "type": type(value).__name__}
    allowed = {}
    for key in (
        "schema",
        "schema_version",
        "protocol_id",
        "request_id",
        "model_id",
        "artifact_id",
        "seal_id",
        "joint_seal_id",
        "files",
        "paths",
        "dlo",
        "dataset",
        "information_boundary",
    ):
        if key in value:
            encoded = json.dumps(value[key], sort_keys=True, default=str)
            allowed[key] = value[key] if len(encoded) <= 100_000 else {"omitted_bytes": len(encoded)}
    return {
        "kind": "json",
        "payload_parsed": True,
        "keys": sorted(value)[:500],
        "allowlisted_metadata": allowed,
    }


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return inspect_npy(path)
    if suffix == ".npz":
        return inspect_npz(path)
    if suffix == ".zip":
        try:
            return inspect_nested_zip(path)
        except zipfile.BadZipFile as error:
            return {"kind": "zip", "inspection_error": repr(error)}
    if suffix == ".json":
        return safe_json_metadata(path)
    return {"kind": "opaque", "payload_parsed": False}


def score(path: Path, schema: Mapping[str, Any]) -> tuple[int, list[str]]:
    text = (path.as_posix() + " " + json.dumps(schema, sort_keys=True)).lower()
    weights = {
        "dlo4": 8,
        "dlo5": 8,
        "target": 5,
        "prediction": 6,
        "physical": 7,
        "baseline": 5,
        "residual": 7,
        "state": 4,
        "position": 4,
        "trajectory": 4,
        "action": 4,
        "contact": 4,
        "force": 3,
        "seal": 3,
    }
    total = 0
    reasons = []
    for token, weight in weights.items():
        if token in text:
            total += weight
            reasons.append(token)
    if schema.get("kind") in {"npy", "npz", "zip"}:
        total += 3
        reasons.append("structured-container")
    return total, reasons


def verify_authorization(path: Path, source_model: Path, source_arrays: Path) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    claimed_id = authorization.pop("authorization_id")
    actual_id = canonical_hash(authorization)
    authorization["authorization_id"] = claimed_id
    if actual_id != claimed_id:
        raise CensusError("target authorization ID mismatch")
    if authorization.get("target_artifact_census_authorized") is not True:
        raise CensusError("target artifact census is not authorized")
    if authorization.get("selection_frozen_before_target") is not True:
        raise CensusError("source selection was not frozen")
    if authorization.get("coefficient_refit_authorized") is not False:
        raise CensusError("authorization permits coefficient refit")
    if authorization.get("standardizer_refit_authorized") is not False:
        raise CensusError("authorization permits standardizer refit")
    model = json.loads(source_model.read_text(encoding="utf-8"))
    if model.get("schema") != SOURCE_MODEL_SCHEMA:
        raise CensusError("unexpected source-model schema")
    if model["model_id"] != authorization["source_model_id"]:
        raise CensusError("authorization names another source model")
    if sha256(source_model) != authorization["source_model_manifest_sha256"]:
        raise CensusError("source-model manifest hash mismatch")
    if sha256(source_arrays) != authorization["source_model_arrays_sha256"]:
        raise CensusError("source-model arrays hash mismatch")
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    root = arguments.artifact_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise CensusError("artifact root must be a real directory")
    authorization = verify_authorization(
        arguments.authorization,
        arguments.source_model,
        arguments.source_arrays,
    )

    records = []
    symlinks_skipped = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            symlinks_skipped += 1
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lower = relative.lower()
        if not any(token in lower for token in RELEVANT_TOKENS) and path.suffix.lower() not in {".npz", ".npy", ".zip"}:
            continue
        schema = inspect_file(path)
        value_score, reasons = score(path, schema)
        records.append(
            {
                "path": path.as_posix(),
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "suffix": path.suffix.lower(),
                "target_identities": target_identities(relative),
                "roles": classify(relative),
                "score": value_score,
                "score_reasons": reasons,
                "schema": schema,
            }
        )
    records.sort(key=lambda value: (-value["score"], value["relative_path"]))

    result = {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-dlo45-artifact-census",
        "schema_version": 1,
        "authorization_id": authorization["authorization_id"],
        "source_model_id": authorization["source_model_id"],
        "parent_run_id": authorization["parent"]["run"]["id"],
        "artifact_root": root.as_posix(),
        "file_count": len(records),
        "symlinks_skipped": symlinks_skipped,
        "ranked_files": records,
        "target_identity_counts": {
            dlo: sum(dlo in record["target_identities"] for record in records)
            for dlo in TARGET_DLOS
        },
        "numeric_container_count": sum(
            record["schema"].get("kind") in {"npy", "npz", "zip"}
            for record in records
        ),
        "information_boundary": {
            "artifact_bytes_streamed_for_sha256": True,
            "archive_central_directories_read": True,
            "numpy_headers_read": True,
            "target_numeric_array_values_loaded": False,
            "target_score_result_json_parsed": False,
            "source_group_selection_changed": False,
            "source_coefficients_changed": False,
            "target_performance_metric_computed": False,
        },
    }
    result["census_id"] = canonical_hash(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "census_id": result["census_id"],
                "file_count": result["file_count"],
                "numeric_container_count": result["numeric_container_count"],
                "target_identity_counts": result["target_identity_counts"],
                "top_files": records[:30],
                "information_boundary": result["information_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
