"""Post-seal development scoring for the camera-recovered v5.2 source batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from . import deform360_joint_sparse_source_scoring_v5 as _v5
from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from .deform360_joint_sparse_endpoint_v5 import (
    score_deform360_joint_sparse_endpoint_v5,
    select_reserved_endpoint_views_v5,
)
from .deform360_joint_sparse_prediction_v5 import RAW_METHOD_IDS
from .deform360_joint_sparse_source_evidence_v5 import (
    assemble_deform360_joint_sparse_source_evidence_v5,
    build_deform360_joint_sparse_source_outcomes_v5,
    publish_deform360_joint_sparse_source_evidence_v5,
    validate_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_seal_v5,
)
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_joint_sparse_source_runner_v5 import (
    _ordinary_root,
    _publish_or_validate_json,
    _sha256_file,
)
from .deform360_joint_sparse_source_runner_v5_2 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5_2,
    validate_deform360_joint_sparse_source_prediction_receipt_v5_2,
)

SOURCE_ENDPOINT_PLAN_SCHEMA: Final = _v5.SOURCE_ENDPOINT_PLAN_SCHEMA
SOURCE_ENDPOINT_PLAN_VERSION: Final = 2
SOURCE_ENDPOINT_PLAN_SEMANTICS: Final = (
    "post-camera-recovery-prediction-development-suffix-endpoint-plan-v1"
)
SOURCE_ENDPOINT_BOUNDARY: Final = dict(_v5.SOURCE_ENDPOINT_BOUNDARY)
SOURCE_SCORING_RECEIPT_SCHEMA: Final = _v5.SOURCE_SCORING_RECEIPT_SCHEMA
SOURCE_SCORING_RECEIPT_VERSION: Final = 2

_FILE_FIELDS = frozenset({"path", "sha256"})
_VIEW_FIELDS = frozenset({"camera_id", "endpoint_archive"})
_OBJECT_FIELDS = frozenset(
    {
        "all_camera_ids",
        "episode_id",
        "object_id",
        "raw_endpoint_range_half_open",
        "reserved_views",
        "stratum",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "endpoint_plan_id",
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "objects",
        "prediction_batch_id",
        "schema",
        "schema_version",
        "semantics",
        "source_prediction_plan_id",
        "source_prediction_receipt_id",
    }
)
_SCORING_RECEIPT_FIELDS = frozenset(
    {
        "authorization_id",
        "endpoint_plan_file_sha256",
        "endpoint_plan_id",
        "endpoint_report_count",
        "evidence_file_sha256",
        "evidence_id",
        "execution_lock_id",
        "information_boundary",
        "outcome_count",
        "prediction_batch_id",
        "receipt_id",
        "schema",
        "schema_version",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _identifier(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    _require(result == result.strip() and "\x00" not in result, f"invalid {name}")
    return result


def _file_record(value: object, *, name: str) -> dict[str, str]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(record.get("path"), name=f"{name}.path"),
        "sha256": sha256_digest(record.get("sha256"), name=f"{name}.sha256"),
    }


def _source_objects(source_plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], source_plan["objects"])
    }


def build_deform360_joint_sparse_source_endpoint_plan_v5_2(
    *,
    lock: Mapping[str, Any],
    source_prediction_plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    source_prediction_receipt: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind suffix files only after the replacement 100-forecast batch exists."""

    source_plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        source_prediction_plan, lock=lock
    )
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        source_prediction_receipt,
        lock=lock,
        plan=source_plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=cast(
            str, source_prediction_receipt["prediction_batch_file_sha256"]
        ),
    )
    expected = _source_objects(source_plan)
    normalized: list[dict[str, Any]] = []
    for raw in objects:
        row = _mapping(raw, name="endpoint object")
        require_exact_fields(row, expected=_OBJECT_FIELDS, name="endpoint object")
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(object_id in expected, "endpoint object is outside the source cohort")
        source = expected[object_id]
        _require(
            row.get("episode_id") == source["episode_id"]
            and row.get("stratum") == source["stratum"],
            "endpoint object identity changed",
        )
        cameras = tuple(
            _identifier(value, name="camera_id")
            for value in _sequence(row.get("all_camera_ids"), name="all_camera_ids")
        )
        _require(cameras == tuple(source["all_camera_ids"]), "camera roster changed")
        source_prefix = cast(Sequence[int], source["raw_prefix_range_half_open"])
        expected_range = (int(source_prefix[1]), int(source_prefix[1]) + 18)
        endpoint_range = _sequence(
            row.get("raw_endpoint_range_half_open"),
            name="raw_endpoint_range_half_open",
        )
        _require(tuple(endpoint_range) == expected_range, "endpoint range changed")
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        views: list[dict[str, Any]] = []
        for index, raw_view in enumerate(
            _sequence(row.get("reserved_views"), name="reserved_views")
        ):
            view = _mapping(raw_view, name=f"reserved_views[{index}]")
            require_exact_fields(
                view, expected=_VIEW_FIELDS, name=f"reserved_views[{index}]"
            )
            views.append(
                {
                    "camera_id": _identifier(view.get("camera_id"), name="camera_id"),
                    "endpoint_archive": _file_record(
                        view.get("endpoint_archive"),
                        name=f"reserved_views[{index}].endpoint_archive",
                    ),
                }
            )
        views.sort(key=lambda item: cast(str, item["camera_id"]))
        _require(
            tuple(cast(str, item["camera_id"]) for item in views)
            == tuple(sorted(reserved)),
            "endpoint reserved-view roster changed",
        )
        normalized.append(
            {
                "object_id": object_id,
                "episode_id": source["episode_id"],
                "stratum": source["stratum"],
                "all_camera_ids": list(cameras),
                "raw_endpoint_range_half_open": list(expected_range),
                "reserved_views": views,
            }
        )
    normalized.sort(key=lambda item: cast(str, item["object_id"]))
    _require(
        [item["object_id"] for item in normalized] == sorted(expected),
        "endpoint plan differs from the exact source cohort",
    )
    identity: dict[str, Any] = {
        "schema": SOURCE_ENDPOINT_PLAN_SCHEMA,
        "schema_version": SOURCE_ENDPOINT_PLAN_VERSION,
        "semantics": SOURCE_ENDPOINT_PLAN_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "source_prediction_plan_id": source_plan["plan_id"],
        "source_prediction_receipt_id": receipt["receipt_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "implementation_revision": exact_revision(
            batch["implementation_revision"], name="implementation_revision"
        ),
        "objects": normalized,
        "information_boundary": dict(SOURCE_ENDPOINT_BOUNDARY),
    }
    return {**identity, "endpoint_plan_id": content_id(identity)}


def validate_deform360_joint_sparse_source_endpoint_plan_v5_2(
    value: object,
    *,
    lock: Mapping[str, Any],
    source_prediction_plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    source_prediction_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _mapping(value, name="source endpoint plan")
    require_exact_fields(plan, expected=_PLAN_FIELDS, name="source endpoint plan")
    _require(
        plan.get("schema") == SOURCE_ENDPOINT_PLAN_SCHEMA
        and plan.get("schema_version") == SOURCE_ENDPOINT_PLAN_VERSION
        and plan.get("semantics") == SOURCE_ENDPOINT_PLAN_SEMANTICS
        and plan.get("information_boundary") == SOURCE_ENDPOINT_BOUNDARY,
        "v5.2 endpoint-plan contract changed",
    )
    rebuilt = build_deform360_joint_sparse_source_endpoint_plan_v5_2(
        lock=lock,
        source_prediction_plan=source_prediction_plan,
        prediction_batch=prediction_batch,
        source_prediction_receipt=source_prediction_receipt,
        objects=cast(
            Sequence[Mapping[str, Any]],
            _sequence(plan.get("objects"), name="objects"),
        ),
    )
    _require(plain_json(plan) == rebuilt, "v5.2 endpoint plan identity changed")
    return rebuilt


def publish_deform360_joint_sparse_source_scores_v5_2(
    *,
    execution_lock_path: str | Path,
    source_prediction_plan_path: str | Path,
    source_prediction_root: str | Path,
    endpoint_plan_path: str | Path,
    endpoint_input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Authorize, then score, the v5.2 public development suffix once."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    source_plan_path = Path(source_prediction_plan_path).resolve(strict=True)
    source_plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        load_strict_json_object(source_plan_path, label="v5.2 source prediction plan"),
        lock=lock,
    )
    prediction_root = _ordinary_root(source_prediction_root)
    batch_path = prediction_root / "source-prediction-batch.json"
    receipt_path = prediction_root / "source-prediction-receipt.json"
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_path, label="source prediction batch"), lock
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        load_strict_json_object(receipt_path, label="v5.2 source prediction receipt"),
        lock=lock,
        plan=source_plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256_file(batch_path),
    )
    ordered_ids = tuple(sorted(_source_objects(source_plan)))
    records = _v5._prediction_records(batch)  # noqa: SLF001
    seal_root = prediction_root / "source-seals"
    prediction_directories: dict[tuple[str, str], Path] = {}
    for outer_index, outer_id in enumerate(ordered_ids):
        for target_index, target_id in enumerate(ordered_ids):
            pair = (outer_id, target_id)
            source_seal_path = seal_root / f"{outer_index:02d}-{target_index:02d}.json"
            expected_digest = cast(
                Mapping[str, str], receipt["source_prediction_seal_file_sha256"]
            )[source_seal_path.name]
            _require(
                _sha256_file(source_seal_path) == expected_digest,
                "source prediction seal file digest changed",
            )
            source_seal = validate_deform360_joint_sparse_source_prediction_seal_v5(
                load_strict_json_object(
                    source_seal_path, label="source prediction seal"
                ),
                lock,
            )
            _require(source_seal == records[pair], "source seal order changed")
            directory = _v5._prediction_directory(  # noqa: SLF001
                prediction_root / "predictions",
                ordered_ids=ordered_ids,
                outer_id=outer_id,
                target_id=target_id,
            )
            _v5._validate_prediction_artifact(  # noqa: SLF001
                directory,
                record=records[pair],
                lock=lock,
                implementation_revision=cast(str, batch["implementation_revision"]),
            )
            prediction_directories[pair] = directory

    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    _require(
        output.is_dir()
        and not output.is_symlink()
        and not any(parent.is_symlink() for parent in output.parents),
        "source scoring output root is invalid",
    )
    authorization = _v5._opening_authorization(  # noqa: SLF001
        lock=lock,
        source_plan=source_plan,
        batch=batch,
        receipt=receipt,
        source_plan_path=source_plan_path,
        batch_path=batch_path,
        receipt_path=receipt_path,
    )
    require_exact_fields(
        authorization,
        expected=_v5._AUTHORIZATION_FIELDS,  # noqa: SLF001
        name="source suffix authorization",
    )
    authorization_path = output / "source-suffix-opening-authorization.json"
    _publish_or_validate_json(
        authorization, authorization_path, label="source suffix authorization"
    )

    # This is the first read of an artifact that names development-suffix files.
    endpoint_plan_source = Path(endpoint_plan_path).resolve(strict=True)
    endpoint_plan = validate_deform360_joint_sparse_source_endpoint_plan_v5_2(
        load_strict_json_object(endpoint_plan_source, label="v5.2 endpoint plan"),
        lock=lock,
        source_prediction_plan=source_plan,
        prediction_batch=batch,
        source_prediction_receipt=receipt,
    )
    endpoint_root = _ordinary_root(endpoint_input_root)
    views_by_object, endpoint_sources = _v5._endpoint_views_by_object(  # noqa: SLF001
        endpoint_plan=endpoint_plan, endpoint_root=endpoint_root
    )
    source_objects = _source_objects(source_plan)
    methods_by_seal: dict[str, Mapping[str, Any]] = {}
    artifacts_by_seal: dict[str, Mapping[str, str]] = {}
    report_root = output / "endpoint-reports"
    outcome_root = output / "source-outcomes"
    report_root.mkdir(parents=True, exist_ok=True)
    outcome_root.mkdir(parents=True, exist_ok=True)
    for outer_index, outer_id in enumerate(ordered_ids):
        for target_index, target_id in enumerate(ordered_ids):
            record = records[(outer_id, target_id)]
            _prediction_seal, result = _v5._validate_prediction_artifact(  # noqa: SLF001
                prediction_directories[(outer_id, target_id)],
                record=record,
                lock=lock,
                implementation_revision=cast(str, batch["implementation_revision"]),
            )
            source_object = source_objects[target_id]
            report = score_deform360_joint_sparse_endpoint_v5(
                object_id=target_id,
                episode_id=cast(int, source_object["episode_id"]),
                stratum=cast(str, source_object["stratum"]),
                prediction_seal_id=cast(str, record["seal_id"]),
                trajectories_m=result.trajectories_m,
                reserved_views=views_by_object[target_id],
                all_camera_ids=cast(Sequence[str], source_object["all_camera_ids"]),
                evaluation_role="development_source",
            )
            report_path = report_root / f"{outer_index:02d}-{target_index:02d}.json"
            _publish_or_validate_json(report, report_path, label="endpoint report")
            seal_id = cast(str, record["seal_id"])
            method_artifacts = cast(Mapping[str, Mapping[str, Any]], record["methods"])
            losses = cast(Mapping[str, float], report["method_loss_mm"])
            methods_by_seal[seal_id] = {
                method_id: {
                    "artifact_id": method_artifacts[method_id]["artifact_id"],
                    "loss_mm": losses[method_id],
                }
                for method_id in RAW_METHOD_IDS
            }
            artifacts_by_seal[seal_id] = {
                **dict(endpoint_sources[target_id]),
                f"endpoint-reports/{report_path.name}": _sha256_file(report_path),
                "source-suffix-opening-authorization.json": _sha256_file(
                    authorization_path
                ),
            }

    outcomes = build_deform360_joint_sparse_source_outcomes_v5(
        lock=lock,
        prediction_batch=batch,
        methods_by_prediction_seal_id=methods_by_seal,
        scoring_artifacts_by_prediction_seal_id=artifacts_by_seal,
    )
    for index, outcome in enumerate(outcomes):
        _publish_or_validate_json(
            outcome,
            outcome_root / f"{index:03d}.json",
            label="source outcome",
        )
    evidence = assemble_deform360_joint_sparse_source_evidence_v5(
        lock=lock, prediction_batch=batch, outcomes=outcomes
    )
    evidence_path = output / "source-evidence.json"
    if evidence_path.exists():
        _publish_or_validate_json(evidence, evidence_path, label="source evidence")
    else:
        publish_deform360_joint_sparse_source_evidence_v5(
            evidence, lock=lock, output_path=evidence_path
        )
    receipt_identity: dict[str, Any] = {
        "schema": SOURCE_SCORING_RECEIPT_SCHEMA,
        "schema_version": SOURCE_SCORING_RECEIPT_VERSION,
        "execution_lock_id": lock["execution_lock_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "authorization_id": authorization["authorization_id"],
        "endpoint_plan_id": endpoint_plan["endpoint_plan_id"],
        "endpoint_plan_file_sha256": _sha256_file(endpoint_plan_source),
        "endpoint_report_count": 100,
        "outcome_count": 100,
        "evidence_id": evidence["evidence_id"],
        "evidence_file_sha256": _sha256_file(evidence_path),
        "information_boundary": dict(SOURCE_ENDPOINT_BOUNDARY),
    }
    scoring_receipt = {
        **receipt_identity,
        "receipt_id": content_id(receipt_identity),
    }
    require_exact_fields(
        scoring_receipt,
        expected=_SCORING_RECEIPT_FIELDS,
        name="source scoring receipt",
    )
    _publish_or_validate_json(
        scoring_receipt,
        output / "source-scoring-receipt.json",
        label="v5.2 source scoring receipt",
    )
    return scoring_receipt


__all__ = [
    "SOURCE_ENDPOINT_BOUNDARY",
    "SOURCE_ENDPOINT_PLAN_SCHEMA",
    "SOURCE_ENDPOINT_PLAN_SEMANTICS",
    "SOURCE_ENDPOINT_PLAN_VERSION",
    "build_deform360_joint_sparse_source_endpoint_plan_v5_2",
    "publish_deform360_joint_sparse_source_scores_v5_2",
    "validate_deform360_joint_sparse_source_endpoint_plan_v5_2",
]
