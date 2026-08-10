#!/usr/bin/env python3
"""Build the real prefix-only v5 plan inputs used by the v6 source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_common_v4 import (
    METRIC_PLAN_SCHEMA,
    METRIC_PLAN_SEMANTICS,
    METRIC_PLAN_VERSION,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON,
    CONTACT_PREFIX_UNAVAILABLE,
    build_deform360_joint_sparse_source_prediction_plan_v5,
)

AMENDMENT_SCHEMA = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-source-prediction-execution"
)
AMENDMENT_ID = "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
RANKING_DOMAIN = b"v6-source-likelihood-panel-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _canonical_id(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ValueError(f"{name} must be a canonical string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _canonical_id(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _ordinary_root(value: Path, *, name: str) -> Path:
    root = value.resolve(strict=True)
    _require(
        root.is_dir() and not root.is_symlink(),
        f"{name} must be an ordinary directory",
    )
    return root


def _ordinary_file(
    root: Path,
    relative: str,
    *,
    digest: str,
    name: str,
) -> Path:
    requested = root / relative
    _require(
        requested.is_file() and not requested.is_symlink(),
        f"{name} is missing",
    )
    resolved = requested.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes its root") from error
    _require(_sha256_file(resolved) == digest, f"{name} SHA-256 changed")
    return resolved


def _file_record(value: object, *, name: str) -> tuple[str, str]:
    record = _mapping(value, name=name)
    required = {"path", "sha256"}
    _require(required <= set(record), f"{name} fields changed")
    relative = _canonical_id(record.get("path"), name=f"{name}.path")
    digest = _sha256(record.get("sha256"), name=f"{name}.sha256")
    return relative, digest


def _relative(path: Path, root: Path, *, name: str) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{name} is outside the results root") from error


def _load_amendment(path: Path) -> Mapping[str, Any]:
    amendment = load_strict_json_object(
        path,
        label="v6 source prediction amendment",
    )
    _require(
        amendment.get("schema") == AMENDMENT_SCHEMA,
        "amendment schema changed",
    )
    _require(amendment.get("schema_version") == 1, "amendment version changed")
    declared = amendment.get("amendment_id")
    identity = {key: value for key, value in amendment.items() if key != "amendment_id"}
    _require(
        declared == content_id(identity) == AMENDMENT_ID,
        "amendment identity changed",
    )
    boundary = _mapping(
        amendment.get("information_boundary"),
        name="information_boundary",
    )
    _require(
        boundary.get("development_suffix_opened") is False
        and boundary.get("v6_target_payloads_opened") is False
        and boundary.get("v6_target_outcomes_used") is False,
        "amendment crosses its information boundary",
    )
    return amendment


def _cohort(lock: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    cohort = _mapping(lock.get("cohort"), name="cohort")
    rows = _sequence(
        cohort.get("development_objects"),
        name="development_objects",
    )
    result: dict[str, tuple[int, str]] = {}
    for raw in rows:
        row = _mapping(raw, name="development object")
        object_id = _canonical_id(row.get("object_id"), name="object_id")
        episode_id = row.get("episode_id")
        stratum = row.get("stratum")
        _require(
            type(episode_id) is int and episode_id >= 0,
            "episode_id changed",
        )
        _require(stratum in {"sheet", "volumetric"}, "stratum changed")
        _require(object_id not in result, "development object repeats")
        result[object_id] = (cast(int, episode_id), cast(str, stratum))
    _require(len(result) == 10, "development cohort changed")
    return result


def _rank_camera(object_id: str, camera_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        object_id.encode("utf-8")
        + b"\0"
        + camera_id.encode("utf-8")
        + b"\0"
        + RANKING_DOMAIN
    ).digest()
    return digest, camera_id


def _prediction_archive(
    prediction_root: Path,
    stream: Mapping[str, Any],
) -> tuple[Path, str]:
    relative, digest = _file_record(
        stream.get("prediction_manifest"),
        name="prediction_manifest",
    )
    manifest_path = _ordinary_file(
        prediction_root,
        relative,
        digest=digest,
        name="prediction manifest",
    )
    manifest = load_strict_json_object(
        manifest_path,
        label="MotionCrafter manifest",
    )
    archive_name = manifest.get("disjoint_baseline")
    _require(
        type(archive_name) is str and bool(archive_name),
        "disjoint baseline is missing",
    )
    archive = (manifest_path.parent / archive_name).resolve(strict=True)
    _require(
        archive.is_file() and not archive.is_symlink(),
        "disjoint baseline is missing",
    )
    _require(
        archive.parent == manifest_path.parent,
        "disjoint baseline escaped its view",
    )
    return archive, _sha256_file(archive)


def _physical_record(
    results_root: Path,
    physical_work_root: Path,
    object_id: str,
    episode_id: int,
) -> dict[str, str]:
    case = f"{object_id}-ep{episode_id:04d}"
    directory = physical_work_root / case
    archive = directory / "prediction.npz"
    manifest_path = directory / "physical_prediction_manifest.json"
    _require(
        archive.is_file() and not archive.is_symlink(),
        f"physical archive missing: {case}",
    )
    manifest = load_strict_json_object(
        manifest_path,
        label=f"physical manifest {case}",
    )
    _require(
        manifest.get("object_id") == object_id,
        "physical object identity changed",
    )
    _require(
        manifest.get("episode_id") == episode_id,
        "physical episode identity changed",
    )
    mode = manifest.get("physical_mode")
    _require(
        mode in {"warp_twin", "persistence_fallback"},
        "physical mode changed",
    )
    archive_record = _mapping(
        manifest.get("physical_prediction_archive"),
        name="physical prediction archive",
    )
    archive_digest = _sha256_file(archive)
    _require(
        archive_record.get("file_sha256") == archive_digest,
        "physical archive changed",
    )
    return {
        "path": _relative(archive, results_root, name="physical archive"),
        "sha256": archive_digest,
        "physical_mode": cast(str, mode),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--metric-batch-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--physical-work-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    amendment = _load_amendment(args.execution_amendment)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(args.execution_lock)
    _require(
        amendment.get("v5_source_execution_lock_id") == lock.get("execution_lock_id"),
        "amendment uses another v5 source lock",
    )
    cohort = _cohort(lock)
    results_root = _ordinary_root(args.results_root, name="results root")
    metric_root = _ordinary_root(
        args.metric_batch_root,
        name="metric batch root",
    )
    prediction_root = _ordinary_root(
        args.prediction_root,
        name="prediction root",
    )
    physical_root = _ordinary_root(
        args.physical_work_root,
        name="physical work root",
    )
    for root, name in (
        (metric_root, "metric batch root"),
        (prediction_root, "prediction root"),
        (physical_root, "physical work root"),
    ):
        try:
            root.relative_to(results_root)
        except ValueError as error:
            raise ValueError(f"{name} is outside the results root") from error

    plan = load_strict_json_object(
        metric_root / "metric-prefix-plan.json",
        label="v4 metric-prefix plan",
    )
    _require(plan.get("schema") == METRIC_PLAN_SCHEMA, "metric plan schema changed")
    _require(
        plan.get("schema_version") == METRIC_PLAN_VERSION,
        "metric plan version changed",
    )
    _require(
        plan.get("semantics") == METRIC_PLAN_SEMANTICS,
        "metric plan semantics changed",
    )
    cases = _sequence(plan.get("cases"), name="metric plan cases")
    exclusions = _sequence(plan.get("excluded_streams"), name="excluded_streams")
    excluded_by_object: dict[str, set[str]] = {object_id: set() for object_id in cohort}
    for raw in exclusions:
        row = _mapping(raw, name="excluded stream")
        object_id = _canonical_id(
            row.get("object_id"),
            name="excluded object_id",
        )
        if object_id in excluded_by_object:
            excluded_by_object[object_id].add(
                _canonical_id(
                    row.get("camera_id"),
                    name="excluded camera_id",
                )
            )

    by_object: dict[str, Mapping[str, Any]] = {}
    for raw in cases:
        row = _mapping(raw, name="metric case")
        object_id = _canonical_id(row.get("object_id"), name="object_id")
        _require(object_id not in by_object, "metric plan repeats an object")
        by_object[object_id] = row
    _require(set(by_object) == set(cohort), "metric plan cohort changed")

    panel = _mapping(
        amendment.get("visual_likelihood_panel"),
        name="visual_likelihood_panel",
    )
    maximum = panel.get("maximum_cameras_per_object")
    minimum = panel.get("minimum_cameras_per_object")
    _require(
        type(maximum) is int and type(minimum) is int,
        "camera-panel counts changed",
    )
    _require(2 <= minimum <= maximum, "camera-panel counts are invalid")
    metric_files_root = _ordinary_root(
        metric_root / "metrics",
        name="metric files root",
    )
    objects: list[dict[str, Any]] = []
    for object_id, (episode_id, stratum) in sorted(cohort.items()):
        case = by_object[object_id]
        _require(case.get("episode_id") == episode_id, "metric episode changed")
        _require(case.get("stratum") == stratum, "metric stratum changed")
        prefix = list(
            _sequence(
                case.get("causal_frame_range_half_open"),
                name="causal range",
            )
        )
        _require(
            len(prefix) == 2
            and all(type(value) is int for value in prefix)
            and prefix[1] - prefix[0] == 58,
            "causal prefix changed",
        )
        streams = [
            _mapping(value, name="metric stream")
            for value in _sequence(case.get("streams"), name="metric streams")
        ]
        included_cameras = {
            _canonical_id(stream.get("camera_id"), name="camera_id")
            for stream in streams
        }
        all_cameras = tuple(sorted(included_cameras | excluded_by_object[object_id]))
        reserved = select_reserved_endpoint_views_v5(
            object_id,
            all_cameras,
            count=2,
        )
        eligible = [
            stream
            for stream in streams
            if _canonical_id(stream.get("camera_id"), name="camera_id") not in reserved
        ]
        eligible.sort(
            key=lambda stream: _rank_camera(
                object_id,
                _canonical_id(stream.get("camera_id"), name="camera_id"),
            )
        )
        selected = eligible[:maximum]
        _require(
            len(selected) >= minimum,
            f"too few visual cameras: {object_id}",
        )
        visual_windows: list[dict[str, Any]] = []
        for stream in selected:
            camera_id = _canonical_id(
                stream.get("camera_id"),
                name="camera_id",
            )
            decoded, decoded_digest = _prediction_archive(
                prediction_root,
                stream,
            )
            metric_relative, metric_digest = _file_record(
                stream.get("metric_prefix"),
                name="metric_prefix",
            )
            metric = _ordinary_file(
                metric_files_root,
                metric_relative,
                digest=metric_digest,
                name="metric prefix",
            )
            visual_windows.append(
                {
                    "camera_id": camera_id,
                    "decoded_uniform": {
                        "path": _relative(
                            decoded,
                            results_root,
                            name="decoded uniform",
                        ),
                        "sha256": decoded_digest,
                    },
                    "metric_prefix": {
                        "path": _relative(
                            metric,
                            results_root,
                            name="metric prefix",
                        ),
                        "sha256": metric_digest,
                    },
                }
            )
        objects.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "raw_prefix_range_half_open": prefix,
                "all_camera_ids": list(all_cameras),
                "reserved_endpoint_camera_ids": list(reserved),
                "physical": _physical_record(
                    results_root,
                    physical_root,
                    object_id,
                    episode_id,
                ),
                "visual_windows": visual_windows,
                "contact_prefix": {
                    "status": CONTACT_PREFIX_UNAVAILABLE,
                    "path": None,
                    "manifest_file_sha256": None,
                    "materialization_id": None,
                    "unavailable_reason": (CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON),
                },
            }
        )

    source_plan = build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=args.implementation_revision,
        objects=objects,
    )
    descriptor = {
        "schema": "bayesian-phystwin.deform360-v6-source-plan-inputs",
        "schema_version": 1,
        "execution_amendment_id": AMENDMENT_ID,
        "v4_metric_plan_id": plan.get("plan_id"),
        "source_prediction_plan": source_plan,
        "information_boundary": {
            "development_suffix_opened": False,
            "future_object_observations_used_for_prediction": False,
            "v5_confirmation_payloads_opened": False,
            "v6_target_payloads_opened": False,
            "target_outcomes_used": False,
        },
    }
    output = {**descriptor, "materialization_id": content_id(descriptor)}
    write_atomic_json(output, args.output, overwrite=False)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
