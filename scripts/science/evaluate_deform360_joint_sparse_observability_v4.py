#!/usr/bin/env python3
"""Evaluate a frozen Deform360 v4 joint-sparse development manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from bayesian_phystwin._canonical_contracts import genuine_integer, plain_json
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from bayesian_phystwin.deform360_joint_sparse_observability_v4 import (
    DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
    DEFORM360_JOINT_SPARSE_INPUT_SCHEMA,
    DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
    DEFORM360_JOINT_SPARSE_SEMANTICS,
    DEFORM360_JOINT_SPARSE_VERSION,
    Deform360JointSparseDevelopmentReportV4,
    Deform360JointSparseFactorBatchV4,
    Deform360JointSparseObservabilityPolicyV4,
    default_deform360_joint_sparse_information_boundary_v4,
    evaluate_deform360_joint_sparse_observability_v4,
    technical_failure_deform360_joint_sparse_result_v4,
)

MANIFEST_SCHEMA = "bayesian-phystwin.deform360-joint-sparse-observability-manifest"
MANIFEST_VERSION = 4
FILE_FIELDS = frozenset({"path", "sha256", "byte_count"})
CASE_FIELDS = frozenset(
    {"object_id", "episode_id", "stratum", "input_id", "descriptor", "arrays"}
)
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "protocol_id",
        "manifest_id",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "implementation_revision",
        "policy_id",
        "cases",
        "information_boundary",
        "claim_boundary",
    }
)
ARRAY_NAMES = frozenset(
    {
        "observation_covariance_m2",
        "state_jacobian",
        "local_gauge_jacobian",
        "gauge_indices",
        "parent_indices",
        "transition_matrices",
        "innovation_scale_tril",
        "query_jacobian",
        "prior_reliability",
        "association_probability",
        "composite_weight",
        "shared_bias_jacobian",
        "view_bias_jacobian",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: Path, *, name: str) -> Path:
    _require(not path.is_symlink(), f"{name} must not be a symbolic link")
    resolved = path.resolve(strict=True)
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{name} is not an ordinary file",
    )
    return resolved


def _relative_path(value: object, *, name: str) -> str:
    text = nonempty_string(value, name=name)
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require(path.as_posix() == text, f"{name} is not canonical POSIX")
    _require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} is unsafe",
    )
    return text


def _record(value: object, *, name: str) -> dict[str, object]:
    _require(isinstance(value, Mapping), f"{name} must be a file record")
    record = cast(Mapping[str, Any], value)
    require_exact_fields(record, expected=FILE_FIELDS, name=name)
    return {
        "path": _relative_path(record["path"], name=f"{name}.path"),
        "sha256": sha256_digest(record["sha256"], name=f"{name}.sha256"),
        "byte_count": genuine_integer(
            record["byte_count"], name=f"{name}.byte_count", minimum=1
        ),
    }


def _verify_record(root: Path, value: object, *, name: str) -> Path:
    record = _record(value, name=name)
    candidate = root / PurePosixPath(cast(str, record["path"]))
    path = _ordinary_file(candidate, name=name)
    _require(root == path or root in path.parents, f"{name} escapes manifest root")
    _require(path.stat().st_size == record["byte_count"], f"{name} byte count changed")
    _require(_sha256_file(path) == record["sha256"], f"{name} SHA-256 changed")
    return path


def _load_policy(path: Path) -> Deform360JointSparseObservabilityPolicyV4:
    return Deform360JointSparseObservabilityPolicyV4.from_record(
        _load_json(_ordinary_file(path, name="v4 policy"), name="v4 policy")
    )


def _validate_manifest(
    value: dict[str, Any], policy: Deform360JointSparseObservabilityPolicyV4
) -> dict[str, Any]:
    require_exact_fields(value, expected=MANIFEST_FIELDS, name="v4 manifest")
    _require(value["schema"] == MANIFEST_SCHEMA, "unsupported v4 manifest schema")
    _require(
        value["schema_version"] == MANIFEST_VERSION,
        "unsupported v4 manifest version",
    )
    _require(
        value["semantics"] == DEFORM360_JOINT_SPARSE_SEMANTICS,
        "v4 manifest semantics changed",
    )
    _require(
        value["protocol_id"] == DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
        "v4 manifest protocol changed",
    )
    _require(value["policy_id"] == policy.policy_id, "v4 manifest policy differs")
    value["selection_artifact_sha256"] = sha256_digest(
        value["selection_artifact_sha256"], name="selection_artifact_sha256"
    )
    value["visual_provider_lock_id"] = sha256_digest(
        value["visual_provider_lock_id"], name="visual_provider_lock_id"
    )
    value["implementation_revision"] = exact_revision(
        value["implementation_revision"], name="implementation_revision"
    )
    _require(
        value["information_boundary"]
        == plain_json(default_deform360_joint_sparse_information_boundary_v4()),
        "v4 manifest information boundary changed",
    )
    _require(
        value["claim_boundary"] == DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        "v4 manifest claim boundary changed",
    )
    declared = sha256_digest(value["manifest_id"], name="manifest_id")
    identity = {key: item for key, item in value.items() if key != "manifest_id"}
    _require(content_id(identity) == declared, "v4 manifest ID changed")
    cases = value["cases"]
    _require(isinstance(cases, list) and bool(cases), "v4 manifest has no cases")
    ordering: list[tuple[str, int]] = []
    for index, raw in enumerate(cases):
        _require(isinstance(raw, Mapping), f"v4 case {index} is not an object")
        case = cast(Mapping[str, Any], raw)
        require_exact_fields(case, expected=CASE_FIELDS, name=f"v4 case {index}")
        object_id = nonempty_string(
            case["object_id"], name=f"v4 case {index} object_id"
        )
        episode_id = genuine_integer(
            case["episode_id"], name=f"v4 case {index} episode_id", minimum=0
        )
        _require(
            case["stratum"] in {"sheet", "volumetric"},
            f"v4 case {index} stratum changed",
        )
        sha256_digest(case["input_id"], name=f"v4 case {index} input_id")
        _record(case["descriptor"], name=f"v4 case {index} descriptor")
        _record(case["arrays"], name=f"v4 case {index} arrays")
        ordering.append((object_id, episode_id))
    _require(
        ordering == sorted(ordering) and len(ordering) == len(set(ordering)),
        "v4 cases are unsorted or repeated",
    )
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == ARRAY_NAMES, "v4 factor array roster changed"
            )
            return {name: np.asarray(archive[name]) for name in sorted(ARRAY_NAMES)}
    except (OSError, ValueError) as error:
        raise ValueError("cannot read v4 factor arrays") from error


def _load_batch(
    root: Path,
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Deform360JointSparseFactorBatchV4:
    descriptor_path = _verify_record(
        root, case["descriptor"], name="v4 factor descriptor"
    )
    arrays_path = _verify_record(root, case["arrays"], name="v4 factor arrays")
    descriptor = _load_json(descriptor_path, name="v4 factor descriptor")
    _require(
        descriptor.pop("schema", None) == DEFORM360_JOINT_SPARSE_INPUT_SCHEMA,
        "v4 input schema changed",
    )
    _require(
        descriptor.pop("schema_version", None) == DEFORM360_JOINT_SPARSE_VERSION,
        "v4 input version changed",
    )
    _require(
        descriptor.pop("semantics", None) == DEFORM360_JOINT_SPARSE_SEMANTICS,
        "v4 input semantics changed",
    )
    _require(
        descriptor.pop("claim_boundary", None) == DEFORM360_JOINT_SPARSE_CLAIM_BOUNDARY,
        "v4 input claim boundary changed",
    )
    _require("array_records" in descriptor, "v4 input array records are missing")
    declared_array_records = descriptor.pop("array_records")
    arrays = _load_arrays(arrays_path)
    batch = Deform360JointSparseFactorBatchV4(**descriptor, **arrays)
    _require(
        batch.identity_record()["array_records"] == declared_array_records,
        "v4 input array records changed",
    )
    _require(batch.input_id == case["input_id"], "v4 case input ID changed")
    _require(
        (batch.object_id, batch.episode_id, batch.stratum)
        == (case["object_id"], case["episode_id"], case["stratum"]),
        "v4 case identity changed",
    )
    _require(
        batch.selection_artifact_sha256 == manifest["selection_artifact_sha256"],
        "v4 selection lineage changed",
    )
    _require(
        batch.visual_provider_lock_id == manifest["visual_provider_lock_id"],
        "v4 provider lineage changed",
    )
    _require(
        batch.implementation_revision == manifest["implementation_revision"],
        "v4 implementation lineage changed",
    )
    return batch


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish(
    output: Path,
    *,
    policy_path: Path,
    manifest_path: Path,
    policy: Deform360JointSparseObservabilityPolicyV4,
    manifest: Mapping[str, Any],
    report: Deform360JointSparseDevelopmentReportV4,
) -> None:
    _require(not output.exists(), "v4 output directory already exists")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        shutil.copyfile(policy_path, temporary / "policy.json")
        shutil.copyfile(manifest_path, temporary / "manifest.json")
        for index, result in enumerate(report.results):
            _write_json(temporary / "cases" / f"{index:02d}.json", result.to_record())
        _write_json(temporary / "development-report.json", report.to_record(policy))
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in files
            ),
            encoding="ascii",
        )
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def evaluate_manifest(
    *, policy_path: Path, manifest_path: Path, output: Path
) -> Deform360JointSparseDevelopmentReportV4:
    policy_source = _ordinary_file(policy_path, name="v4 policy")
    manifest_source = _ordinary_file(manifest_path, name="v4 manifest")
    policy = _load_policy(policy_source)
    manifest = _validate_manifest(
        _load_json(manifest_source, name="v4 manifest"), policy
    )
    root = manifest_source.parent
    results = []
    for raw_case in cast(Sequence[Mapping[str, Any]], manifest["cases"]):
        batch = _load_batch(root, raw_case, manifest)
        try:
            result = evaluate_deform360_joint_sparse_observability_v4(
                batch,
                policy,
                implementation_revision=cast(str, manifest["implementation_revision"]),
            )
        except Exception as error:
            result = technical_failure_deform360_joint_sparse_result_v4(
                batch,
                policy,
                implementation_revision=cast(str, manifest["implementation_revision"]),
                reason="joint-sparse-observability-evaluation-failed",
                detail=f"{type(error).__name__}: {error}",
            )
        results.append(result)
    source_artifacts = {
        "manifest.json": _sha256_file(manifest_source),
        "policy.json": _sha256_file(policy_source),
    }
    report = Deform360JointSparseDevelopmentReportV4(
        selection_artifact_sha256=cast(str, manifest["selection_artifact_sha256"]),
        visual_provider_lock_id=cast(str, manifest["visual_provider_lock_id"]),
        policy_id=cast(str, policy.policy_id),
        implementation_revision=cast(str, manifest["implementation_revision"]),
        results=tuple(results),
        source_artifacts=source_artifacts,
        metadata={
            "manifest_id": manifest["manifest_id"],
            "cohort_role": "development",
        },
    )
    _publish(
        output,
        policy_path=policy_source,
        manifest_path=manifest_source,
        policy=policy,
        manifest=manifest,
        report=report,
    )
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate target-free object-level Deform360 v4 observability."
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    report = evaluate_manifest(
        policy_path=arguments.policy,
        manifest_path=arguments.manifest,
        output=arguments.output_dir.resolve(),
    )
    policy = _load_policy(arguments.policy.resolve())
    return 0 if report.support_gate(policy)["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
