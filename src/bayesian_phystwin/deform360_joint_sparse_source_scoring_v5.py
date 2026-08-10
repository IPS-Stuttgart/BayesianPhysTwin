"""Post-seal development scoring for the public Deform360 v5 source panel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_joint_sparse_endpoint_v5 import (
    Deform360ReservedViewGeometryV5,
    score_deform360_joint_sparse_endpoint_v5,
    select_reserved_endpoint_views_v5,
)
from .deform360_joint_sparse_prediction_artifacts_v5 import (
    load_deform360_joint_sparse_prediction_v5,
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
    _verified_file,
    validate_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_receipt_v5,
)

SOURCE_ENDPOINT_PLAN_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-endpoint-plan"
)
SOURCE_ENDPOINT_PLAN_VERSION: Final = 1
SOURCE_ENDPOINT_PLAN_SEMANTICS: Final = (
    "post-prediction-development-suffix-endpoint-plan-v1"
)
SOURCE_SUFFIX_AUTHORIZATION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-suffix-authorization"
)
SOURCE_SUFFIX_AUTHORIZATION_VERSION: Final = 1
SOURCE_SCORING_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-scoring-receipt"
)
SOURCE_SCORING_RECEIPT_VERSION: Final = 1

SOURCE_ENDPOINT_BOUNDARY: Final = {
    "confirmation_payloads_opened": False,
    "development_suffix_opened_after_prediction_batch": True,
    "future_geometry_used_for_prediction": False,
    "human_approval_required": False,
    "new_measurements_required": False,
    "public_released_measurements_used": True,
    "target_outcomes_used": False,
}

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
_AUTHORIZATION_FIELDS = frozenset(
    {
        "authorization_id",
        "confirmation_payloads_opened",
        "development_suffix_access_authorized",
        "execution_lock_id",
        "human_approval_required",
        "prediction_batch_file_sha256",
        "prediction_batch_id",
        "prediction_record_count",
        "schema",
        "schema_version",
        "source_prediction_plan_file_sha256",
        "source_prediction_plan_id",
        "source_prediction_receipt_file_sha256",
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
_ENDPOINT_ARRAY_NAMES = frozenset(
    {
        "camera_to_world",
        "depth_m",
        "frame_indices",
        "intrinsics",
        "object_mask",
        "raw_frame_indices",
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


def build_deform360_joint_sparse_source_endpoint_plan_v5(
    *,
    lock: Mapping[str, Any],
    source_prediction_plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    source_prediction_receipt: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the development endpoint plan only after all forecasts exist."""

    source_plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        source_prediction_plan,
        lock=lock,
    )
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch,
        lock,
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5(
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
        _require(
            cameras == tuple(source["all_camera_ids"]),
            "endpoint camera roster changed",
        )
        source_prefix = cast(Sequence[int], source["raw_prefix_range_half_open"])
        expected_raw_endpoint = (int(source_prefix[1]), int(source_prefix[1]) + 18)
        raw_endpoint = _sequence(
            row.get("raw_endpoint_range_half_open"),
            name="raw_endpoint_range_half_open",
        )
        _require(
            tuple(raw_endpoint) == expected_raw_endpoint,
            "raw endpoint range changed",
        )
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        views: list[dict[str, Any]] = []
        for index, raw_view in enumerate(
            _sequence(row.get("reserved_views"), name="reserved_views")
        ):
            view = _mapping(raw_view, name=f"reserved_views[{index}]")
            require_exact_fields(
                view,
                expected=_VIEW_FIELDS,
                name=f"reserved_views[{index}]",
            )
            camera_id = _identifier(view.get("camera_id"), name="camera_id")
            views.append(
                {
                    "camera_id": camera_id,
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
                "raw_endpoint_range_half_open": list(expected_raw_endpoint),
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


def validate_deform360_joint_sparse_source_endpoint_plan_v5(
    value: object,
    *,
    lock: Mapping[str, Any],
    source_prediction_plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    source_prediction_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _mapping(value, name="source endpoint plan")
    require_exact_fields(plan, expected=_PLAN_FIELDS, name="source endpoint plan")
    _require(plan.get("schema") == SOURCE_ENDPOINT_PLAN_SCHEMA, "endpoint plan schema changed")
    _require(
        plan.get("schema_version") == SOURCE_ENDPOINT_PLAN_VERSION,
        "endpoint plan version changed",
    )
    _require(
        plan.get("semantics") == SOURCE_ENDPOINT_PLAN_SEMANTICS,
        "endpoint plan semantics changed",
    )
    _require(
        plan.get("information_boundary") == SOURCE_ENDPOINT_BOUNDARY,
        "endpoint plan information boundary changed",
    )
    rebuilt = build_deform360_joint_sparse_source_endpoint_plan_v5(
        lock=lock,
        source_prediction_plan=source_prediction_plan,
        prediction_batch=prediction_batch,
        source_prediction_receipt=source_prediction_receipt,
        objects=cast(
            Sequence[Mapping[str, Any]],
            _sequence(plan.get("objects"), name="objects"),
        ),
    )
    _require(plain_json(plan) == rebuilt, "endpoint plan identity changed")
    return rebuilt


def _prediction_records(
    batch: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (
            cast(str, row["outer_held_out_object_id"]),
            cast(str, row["object_id"]),
        ): row
        for row in cast(Sequence[Mapping[str, Any]], batch["records"])
    }


def _prediction_directory(
    prediction_root: Path,
    *,
    ordered_ids: Sequence[str],
    outer_id: str,
    target_id: str,
) -> Path:
    outer_index = ordered_ids.index(outer_id)
    target_index = ordered_ids.index(target_id)
    return prediction_root / f"{outer_index:02d}-{outer_id}/{target_index:02d}-{target_id}"


def _validate_prediction_artifact(
    directory: Path,
    *,
    record: Mapping[str, Any],
    lock: Mapping[str, Any],
    implementation_revision: str,
) -> tuple[Mapping[str, Any], Any]:
    prediction_seal, result = load_deform360_joint_sparse_prediction_v5(directory)
    methods = cast(Mapping[str, Mapping[str, Any]], record["methods"])
    _require(
        prediction_seal["execution_lock_id"] == lock["execution_lock_id"]
        and prediction_seal["implementation_revision"] == implementation_revision
        and prediction_seal["prediction_fit_artifact_id"]
        == record["prediction_fit_artifact_id"]
        and prediction_seal["prediction_fit_object_ids"]
        == record["prediction_fit_object_ids"]
        and prediction_seal["factor_admitted"] == record["factor_admitted"]
        and prediction_seal["physical_mode"] == record["physical_mode"]
        and float(prediction_seal["risk_score"]) == float(record["risk_score"]),
        "published prediction differs from its source seal",
    )
    artifact_ids = cast(Mapping[str, str], prediction_seal["method_artifact_ids"])
    _require(
        all(
            artifact_ids[method_id] == methods[method_id]["artifact_id"]
            for method_id in RAW_METHOD_IDS
        ),
        "published method artifact differs from its source seal",
    )
    if record["technical_failure"]:
        baseline = result.trajectories_m[RAW_METHOD_IDS[0]]
        _require(
            not record["factor_admitted"]
            and all(
                np.array_equal(result.trajectories_m[method_id], baseline)
                for method_id in RAW_METHOD_IDS
            ),
            "technical prediction failure did not preserve exact fallback",
        )
    return prediction_seal, result


def _opening_authorization(
    *,
    lock: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_plan_path: Path,
    batch_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": SOURCE_SUFFIX_AUTHORIZATION_SCHEMA,
        "schema_version": SOURCE_SUFFIX_AUTHORIZATION_VERSION,
        "execution_lock_id": lock["execution_lock_id"],
        "source_prediction_plan_id": source_plan["plan_id"],
        "source_prediction_plan_file_sha256": _sha256_file(source_plan_path),
        "prediction_batch_id": batch["prediction_batch_id"],
        "prediction_batch_file_sha256": _sha256_file(batch_path),
        "source_prediction_receipt_id": receipt["receipt_id"],
        "source_prediction_receipt_file_sha256": _sha256_file(receipt_path),
        "prediction_record_count": 100,
        "development_suffix_access_authorized": True,
        "confirmation_payloads_opened": False,
        "human_approval_required": False,
    }
    return {**identity, "authorization_id": content_id(identity)}


def _load_endpoint_view(
    path: Path,
    *,
    object_id: str,
    episode_id: int,
    camera_id: str,
    raw_endpoint_range_half_open: tuple[int, int],
    source_artifacts: Mapping[str, str],
) -> Deform360ReservedViewGeometryV5:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == set(_ENDPOINT_ARRAY_NAMES),
                "endpoint archive member roster changed",
            )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load endpoint archive") from error
    raw_start, raw_stop = raw_endpoint_range_half_open
    _require(
        np.array_equal(
            arrays["raw_frame_indices"],
            np.arange(raw_start, raw_stop, dtype=np.int64),
        ),
        "endpoint raw-frame roster changed",
    )
    return Deform360ReservedViewGeometryV5(
        object_id=object_id,
        episode_id=episode_id,
        camera_id=camera_id,
        frame_indices=arrays["frame_indices"],
        depth_m=arrays["depth_m"],
        object_mask=arrays["object_mask"],
        intrinsics=arrays["intrinsics"],
        camera_to_world=arrays["camera_to_world"],
        source_artifact_ids=source_artifacts,
    )


def _technical_endpoint_views(
    *,
    object_id: str,
    episode_id: int,
    camera_ids: Sequence[str],
    source_artifacts: Mapping[str, str],
    failure: Exception,
) -> tuple[Deform360ReservedViewGeometryV5, ...]:
    failure_identity = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-endpoint-technical-failure-v5",
        "schema_version": 1,
        "object_id": object_id,
        "episode_id": episode_id,
        "exception_type": type(failure).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(failure).encode("utf-8")
        ).hexdigest(),
        "confirmation_payloads_opened": False,
    }
    failure_id = content_id(failure_identity)
    sources = {
        **dict(source_artifacts),
        f"technical-failures/{object_id}.json": failure_id,
    }
    frames: np.ndarray = np.arange(58, 76, dtype=np.int64)
    return tuple(
        Deform360ReservedViewGeometryV5(
            object_id=object_id,
            episode_id=episode_id,
            camera_id=camera_id,
            frame_indices=frames,
            depth_m=np.ones((18, 1, 1), dtype=np.float32),
            object_mask=np.zeros((18, 1, 1), dtype=np.bool_),
            intrinsics=np.eye(3, dtype=np.float64),
            camera_to_world=np.eye(4, dtype=np.float64),
            source_artifact_ids=sources,
        )
        for camera_id in camera_ids
    )


def _endpoint_views_by_object(
    *,
    endpoint_plan: Mapping[str, Any],
    endpoint_root: Path,
) -> tuple[
    dict[str, tuple[Deform360ReservedViewGeometryV5, ...]],
    dict[str, Mapping[str, str]],
]:
    result: dict[str, tuple[Deform360ReservedViewGeometryV5, ...]] = {}
    sources_by_object: dict[str, Mapping[str, str]] = {}
    for row in cast(Sequence[Mapping[str, Any]], endpoint_plan["objects"]):
        object_id = cast(str, row["object_id"])
        episode_id = cast(int, row["episode_id"])
        raw_endpoint = cast(Sequence[int], row["raw_endpoint_range_half_open"])
        raw_endpoint_range = (int(raw_endpoint[0]), int(raw_endpoint[1]))
        verified: list[tuple[str, Path]] = []
        sources: dict[str, str] = {}
        for view in cast(Sequence[Mapping[str, Any]], row["reserved_views"]):
            camera_id = cast(str, view["camera_id"])
            record = cast(Mapping[str, Any], view["endpoint_archive"])
            path = _verified_file(
                endpoint_root,
                record,
                name=f"endpoint archive for {object_id}/{camera_id}",
            )
            verified.append((camera_id, path))
            sources[f"endpoint/{object_id}/{camera_id}.npz"] = _sha256_file(path)
        try:
            views = tuple(
                _load_endpoint_view(
                    path,
                    object_id=object_id,
                    episode_id=episode_id,
                    camera_id=camera_id,
                    raw_endpoint_range_half_open=raw_endpoint_range,
                    source_artifacts={
                        f"endpoint/{object_id}/{camera_id}.npz": _sha256_file(path)
                    },
                )
                for camera_id, path in verified
            )
        except (OSError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            views = _technical_endpoint_views(
                object_id=object_id,
                episode_id=episode_id,
                camera_ids=tuple(camera_id for camera_id, _path in verified),
                source_artifacts=sources,
                failure=error,
            )
        result[object_id] = views
        sources_by_object[object_id] = source_artifact_mapping(
            sources,
            name=f"endpoint sources for {object_id}",
        )
    return result, sources_by_object


def publish_deform360_joint_sparse_source_scores_v5(
    *,
    execution_lock_path: str | Path,
    source_prediction_plan_path: str | Path,
    source_prediction_root: str | Path,
    endpoint_plan_path: str | Path,
    endpoint_input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Authorize, then score, the public development suffix exactly once."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    source_plan_path = Path(source_prediction_plan_path).resolve(strict=True)
    source_plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        load_strict_json_object(source_plan_path, label="source prediction plan"),
        lock=lock,
    )
    prediction_root = _ordinary_root(source_prediction_root)
    batch_path = prediction_root / "source-prediction-batch.json"
    receipt_path = prediction_root / "source-prediction-receipt.json"
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_path, label="source prediction batch"),
        lock,
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5(
        load_strict_json_object(receipt_path, label="source prediction receipt"),
        lock=lock,
        plan=source_plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256_file(batch_path),
    )
    ordered_ids = tuple(sorted(_source_objects(source_plan)))
    records = _prediction_records(batch)
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
                load_strict_json_object(source_seal_path, label="source prediction seal"),
                lock,
            )
            _require(source_seal == records[pair], "source prediction seal order changed")
            directory = _prediction_directory(
                prediction_root / "predictions",
                ordered_ids=ordered_ids,
                outer_id=outer_id,
                target_id=target_id,
            )
            _validate_prediction_artifact(
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
    authorization = _opening_authorization(
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
        expected=_AUTHORIZATION_FIELDS,
        name="source suffix authorization",
    )
    authorization_path = output / "source-suffix-opening-authorization.json"
    _publish_or_validate_json(
        authorization,
        authorization_path,
        label="source suffix authorization",
    )

    # This is the first read of an artifact that names development-suffix files.
    endpoint_plan_source = Path(endpoint_plan_path).resolve(strict=True)
    endpoint_plan = validate_deform360_joint_sparse_source_endpoint_plan_v5(
        load_strict_json_object(endpoint_plan_source, label="source endpoint plan"),
        lock=lock,
        source_prediction_plan=source_plan,
        prediction_batch=batch,
        source_prediction_receipt=receipt,
    )
    endpoint_root = _ordinary_root(endpoint_input_root)
    views_by_object, endpoint_sources = _endpoint_views_by_object(
        endpoint_plan=endpoint_plan,
        endpoint_root=endpoint_root,
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
            _prediction_seal, result = _validate_prediction_artifact(
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
        lock=lock,
        prediction_batch=batch,
        outcomes=outcomes,
    )
    evidence_path = output / "source-evidence.json"
    if evidence_path.exists():
        _publish_or_validate_json(evidence, evidence_path, label="source evidence")
    else:
        publish_deform360_joint_sparse_source_evidence_v5(
            evidence,
            lock=lock,
            output_path=evidence_path,
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
        label="source scoring receipt",
    )
    return scoring_receipt


__all__ = [
    "SOURCE_ENDPOINT_BOUNDARY",
    "SOURCE_ENDPOINT_PLAN_SCHEMA",
    "SOURCE_ENDPOINT_PLAN_SEMANTICS",
    "SOURCE_ENDPOINT_PLAN_VERSION",
    "build_deform360_joint_sparse_source_endpoint_plan_v5",
    "publish_deform360_joint_sparse_source_scores_v5",
    "validate_deform360_joint_sparse_source_endpoint_plan_v5",
]
