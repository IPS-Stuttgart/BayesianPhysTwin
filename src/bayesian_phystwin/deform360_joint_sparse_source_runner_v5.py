"""Outcome-blind execution of the nested public Deform360 v5 source panel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._gauge_aware_contracts import GaugeAwareObservationBatch
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_bias_aware_prospective_artifacts import PHYSICAL_ARRAY_NAMES
from .deform360_joint_sparse_endpoint_v5 import select_reserved_endpoint_views_v5
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparsePrefixFitV5,
    materialize_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_prediction_artifacts_v5 import (
    PREDICTION_SEAL_FILENAME,
    load_deform360_joint_sparse_prediction_v5,
    publish_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_prediction_v5 import (
    RAW_METHOD_IDS,
    Deform360JointSparsePredictionInputV5,
    run_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_public_inputs_v5 import (
    estimate_deform360_last_causal_residual_v5,
    prepare_deform360_joint_sparse_contact_rows_v5,
    prepare_deform360_joint_sparse_visual_window_v5,
)
from .deform360_joint_sparse_source_evidence_v5 import (
    build_deform360_joint_sparse_source_prediction_batch_v5,
    build_deform360_joint_sparse_source_prediction_seal_v5,
    publish_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_seal_v5,
)
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_public_contact_prefix import (
    validate_deform360_public_contact_prefix,
)

SOURCE_PLAN_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-prediction-plan"
)
SOURCE_PLAN_VERSION: Final = 5
SOURCE_PLAN_SEMANTICS: Final = (
    "public-prefix-only-nested-source-prediction-plan-v1"
)
SOURCE_PANEL_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-prediction-receipt"
)
SOURCE_PANEL_RECEIPT_VERSION: Final = 1
SOURCE_PLAN_BOUNDARY: Final = {
    "confirmation_payloads_opened": False,
    "development_suffix_opened": False,
    "future_object_observations_used": False,
    "human_approval_required": False,
    "new_measurements_required": False,
    "prob4d_used": True,
    "public_released_prefix_measurements_used": True,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}

_FILE_FIELDS = frozenset({"path", "sha256"})
_PHYSICAL_FIELDS = frozenset({"path", "physical_mode", "sha256"})
_CONTACT_FIELDS = frozenset(
    {"manifest_file_sha256", "materialization_id", "path"}
)
_VISUAL_FIELDS = frozenset({"camera_id", "decoded_uniform", "metric_prefix"})
_OBJECT_FIELDS = frozenset(
    {
        "all_camera_ids",
        "contact_prefix",
        "episode_id",
        "object_id",
        "physical",
        "raw_prefix_range_half_open",
        "reserved_endpoint_camera_ids",
        "stratum",
        "visual_windows",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "cohort_selection_sha256",
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "objects",
        "plan_id",
        "schema",
        "schema_version",
        "semantics",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "plan_id",
        "prediction_batch_file_sha256",
        "prediction_batch_id",
        "prediction_record_count",
        "receipt_id",
        "schema",
        "schema_version",
        "source_prediction_seal_file_sha256",
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


def _canonical_identifier(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    _require(result == result.strip() and "\x00" not in result, f"invalid {name}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cohort(lock: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    cohort = _mapping(lock.get("cohort"), name="cohort")
    rows = _sequence(cohort.get("development_objects"), name="development_objects")
    result: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"development_objects[{index}]")
        object_id = _canonical_identifier(row.get("object_id"), name="object_id")
        episode_id = row.get("episode_id")
        stratum = row.get("stratum")
        _require(
            type(episode_id) is int and episode_id >= 0,
            "invalid development episode_id",
        )
        _require(stratum in {"sheet", "volumetric"}, "invalid development stratum")
        _require(object_id not in result, "development object repeats")
        result[object_id] = (cast(int, episode_id), cast(str, stratum))
    _require(len(result) == 10, "source execution lock must bind ten objects")
    return result


def _normalized_file_record(value: object, *, name: str) -> dict[str, str]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(record.get("path"), name=f"{name}.path"),
        "sha256": sha256_digest(record.get("sha256"), name=f"{name}.sha256"),
    }


def _normalized_object(
    value: object,
    *,
    cohort: Mapping[str, tuple[int, str]],
) -> dict[str, Any]:
    row = _mapping(value, name="source plan object")
    require_exact_fields(row, expected=_OBJECT_FIELDS, name="source plan object")
    object_id = _canonical_identifier(row.get("object_id"), name="object_id")
    _require(object_id in cohort, "source plan object is outside the cohort")
    expected_episode, expected_stratum = cohort[object_id]
    _require(row.get("episode_id") == expected_episode, "episode_id changed")
    _require(row.get("stratum") == expected_stratum, "stratum changed")
    prefix = _sequence(
        row.get("raw_prefix_range_half_open"),
        name="raw_prefix_range_half_open",
    )
    _require(
        len(prefix) == 2
        and all(type(item) is int for item in prefix)
        and 0 <= prefix[0] < prefix[1]
        and prefix[1] - prefix[0] == 58,
        "source prefix must contain exactly 58 causal frames",
    )
    all_cameras = tuple(
        _canonical_identifier(item, name="camera_id")
        for item in _sequence(row.get("all_camera_ids"), name="all_camera_ids")
    )
    _require(
        len(all_cameras) == len(set(all_cameras)) and len(all_cameras) >= 4,
        "all_camera_ids must contain at least four unique cameras",
    )
    _require(all_cameras == tuple(sorted(all_cameras)), "camera IDs are not sorted")
    reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
    _require(
        tuple(row.get("reserved_endpoint_camera_ids", ())) == reserved,
        "reserved endpoint camera identities changed",
    )

    physical_raw = _mapping(row.get("physical"), name="physical")
    require_exact_fields(physical_raw, expected=_PHYSICAL_FIELDS, name="physical")
    physical_mode = physical_raw.get("physical_mode")
    _require(
        physical_mode in {"warp_twin", "persistence_fallback"},
        "physical mode changed",
    )
    physical = {
        **_normalized_file_record(
            {"path": physical_raw.get("path"), "sha256": physical_raw.get("sha256")},
            name="physical archive",
        ),
        "physical_mode": physical_mode,
    }

    contact_raw = _mapping(row.get("contact_prefix"), name="contact_prefix")
    require_exact_fields(contact_raw, expected=_CONTACT_FIELDS, name="contact_prefix")
    contact = {
        "path": canonical_relative_posix_path(
            contact_raw.get("path"), name="contact_prefix.path"
        ),
        "manifest_file_sha256": sha256_digest(
            contact_raw.get("manifest_file_sha256"),
            name="contact_prefix.manifest_file_sha256",
        ),
        "materialization_id": sha256_digest(
            contact_raw.get("materialization_id"),
            name="contact_prefix.materialization_id",
        ),
    }

    windows: list[dict[str, Any]] = []
    for index, raw_window in enumerate(
        _sequence(row.get("visual_windows"), name="visual_windows")
    ):
        window = _mapping(raw_window, name=f"visual_windows[{index}]")
        require_exact_fields(
            window,
            expected=_VISUAL_FIELDS,
            name=f"visual_windows[{index}]",
        )
        camera = _canonical_identifier(window.get("camera_id"), name="camera_id")
        _require(camera in all_cameras, "visual camera is outside the camera roster")
        _require(camera not in reserved, "reserved endpoint camera entered likelihood")
        windows.append(
            {
                "camera_id": camera,
                "decoded_uniform": _normalized_file_record(
                    window.get("decoded_uniform"),
                    name=f"visual_windows[{index}].decoded_uniform",
                ),
                "metric_prefix": _normalized_file_record(
                    window.get("metric_prefix"),
                    name=f"visual_windows[{index}].metric_prefix",
                ),
            }
        )
    windows.sort(key=lambda item: cast(str, item["camera_id"]))
    camera_roster = [cast(str, item["camera_id"]) for item in windows]
    _require(
        len(windows) >= 2 and len(camera_roster) == len(set(camera_roster)),
        "source plan needs at least two unique visual cameras",
    )
    return {
        "object_id": object_id,
        "episode_id": expected_episode,
        "stratum": expected_stratum,
        "raw_prefix_range_half_open": list(prefix),
        "all_camera_ids": list(all_cameras),
        "reserved_endpoint_camera_ids": list(reserved),
        "physical": physical,
        "visual_windows": windows,
        "contact_prefix": contact,
    }


def build_deform360_joint_sparse_source_prediction_plan_v5(
    *,
    lock: Mapping[str, Any],
    implementation_revision: str,
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the portable prefix-only plan before any suffix is opened."""

    cohort = _cohort(lock)
    normalized = [
        _normalized_object(value, cohort=cohort)
        for value in objects
    ]
    normalized.sort(key=lambda item: cast(str, item["object_id"]))
    _require(
        [item["object_id"] for item in normalized] == sorted(cohort),
        "source plan differs from the exact development cohort",
    )
    cohort_record = _mapping(lock.get("cohort"), name="cohort")
    identity: dict[str, Any] = {
        "schema": SOURCE_PLAN_SCHEMA,
        "schema_version": SOURCE_PLAN_VERSION,
        "semantics": SOURCE_PLAN_SEMANTICS,
        "execution_lock_id": sha256_digest(
            lock.get("execution_lock_id"), name="execution_lock_id"
        ),
        "cohort_selection_sha256": sha256_digest(
            cohort_record.get("selection_sha256"), name="selection_sha256"
        ),
        "implementation_revision": exact_revision(
            implementation_revision, name="implementation_revision"
        ),
        "objects": normalized,
        "information_boundary": dict(SOURCE_PLAN_BOUNDARY),
    }
    return {**identity, "plan_id": content_id(identity)}


def validate_deform360_joint_sparse_source_prediction_plan_v5(
    value: object,
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one prefix-only source plan."""

    plan = _mapping(value, name="source prediction plan")
    require_exact_fields(plan, expected=_PLAN_FIELDS, name="source prediction plan")
    _require(plan.get("schema") == SOURCE_PLAN_SCHEMA, "plan schema changed")
    _require(plan.get("schema_version") == SOURCE_PLAN_VERSION, "plan version changed")
    _require(plan.get("semantics") == SOURCE_PLAN_SEMANTICS, "plan semantics changed")
    _require(
        plan.get("information_boundary") == SOURCE_PLAN_BOUNDARY,
        "plan information boundary changed",
    )
    rebuilt = build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=cast(str, plan.get("implementation_revision")),
        objects=cast(
            Sequence[Mapping[str, Any]],
            _sequence(plan.get("objects"), name="objects"),
        ),
    )
    _require(plain_json(plan) == rebuilt, "source prediction plan identity changed")
    return rebuilt


def _ordinary_root(path: str | Path) -> Path:
    requested = Path(path).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "input root must be an ordinary non-symlink directory",
    )
    return requested.resolve(strict=True)


def _verified_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    relative = canonical_relative_posix_path(record.get("path"), name=f"{name}.path")
    expected = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    requested = root / relative
    _require(
        requested.is_file()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        f"{name} must be an ordinary file",
    )
    path = requested.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the input root") from error
    _require(_sha256_file(path) == expected, f"{name} SHA-256 changed")
    return path


def _verified_contact_directory(
    root: Path,
    record: Mapping[str, Any],
) -> Path:
    relative = canonical_relative_posix_path(
        record.get("path"), name="contact_prefix.path"
    )
    requested = root / relative
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "contact prefix must be an ordinary directory",
    )
    path = requested.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("contact prefix escapes the input root") from error
    _require(
        _sha256_file(path / "contact-prefix.json")
        == record.get("manifest_file_sha256"),
        "contact-prefix manifest SHA-256 changed",
    )
    manifest = validate_deform360_public_contact_prefix(path)
    _require(
        manifest["materialization_id"] == record.get("materialization_id"),
        "contact-prefix materialization identity changed",
    )
    return path


def _load_physical_archive(
    path: Path,
    *,
    physical_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == set(PHYSICAL_ARRAY_NAMES),
                "physical archive member roster changed",
            )
            prediction = np.asarray(archive["prediction_m"])
            persistence = np.asarray(archive["persistence_m"])
            frame_zero = np.asarray(archive["frame_zero_points_m"])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load physical source archive") from error
    _require(
        prediction.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and persistence.dtype == prediction.dtype
        and prediction.shape == persistence.shape
        and prediction.ndim == 3
        and prediction.shape[0] == 76
        and prediction.shape[1] >= 128
        and prediction.shape[2] == 3
        and frame_zero.shape == prediction.shape[1:]
        and np.all(np.isfinite(prediction))
        and np.all(np.isfinite(persistence)),
        "physical source trajectory changed",
    )
    _require(
        np.array_equal(prediction[0], frame_zero)
        and np.array_equal(
            persistence,
            np.repeat(frame_zero[None], 76, axis=0),
        ),
        "physical source frame-zero identity changed",
    )
    if physical_mode == "persistence_fallback":
        _require(
            np.array_equal(prediction, persistence),
            "persistence fallback is not exact",
        )
    return prediction, persistence


def _publish_or_validate_json(
    value: Mapping[str, Any],
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    normalized = cast(dict[str, Any], plain_json(value))
    if path.exists():
        existing = load_strict_json_object(path, label=label)
        _require(existing == normalized, f"existing {label} differs")
        return normalized
    write_atomic_json(normalized, path, overwrite=False)
    return normalized


def validate_deform360_joint_sparse_source_prediction_receipt_v5(
    value: object,
    *,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    prediction_batch_file_sha256: str,
) -> dict[str, Any]:
    """Validate the complete outcome-blind receipt before suffix access."""

    normalized_plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        plan,
        lock=lock,
    )
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch,
        lock,
    )
    receipt = _mapping(value, name="source prediction receipt")
    require_exact_fields(
        receipt,
        expected=_RECEIPT_FIELDS,
        name="source prediction receipt",
    )
    _require(
        receipt.get("schema") == SOURCE_PANEL_RECEIPT_SCHEMA,
        "source receipt schema changed",
    )
    _require(
        receipt.get("schema_version") == SOURCE_PANEL_RECEIPT_VERSION,
        "source receipt version changed",
    )
    _require(
        receipt.get("information_boundary") == SOURCE_PLAN_BOUNDARY,
        "source receipt information boundary changed",
    )
    _require(
        receipt.get("execution_lock_id") == lock.get("execution_lock_id")
        and receipt.get("plan_id") == normalized_plan["plan_id"]
        and receipt.get("prediction_batch_id") == batch["prediction_batch_id"]
        and receipt.get("implementation_revision")
        == batch["implementation_revision"]
        == normalized_plan["implementation_revision"],
        "source receipt lineage changed",
    )
    _require(
        receipt.get("prediction_batch_file_sha256")
        == sha256_digest(
            prediction_batch_file_sha256,
            name="prediction_batch_file_sha256",
        ),
        "source receipt prediction-batch digest changed",
    )
    _require(
        receipt.get("prediction_record_count") == 100,
        "source receipt must bind exactly 100 predictions",
    )
    seal_digests = source_artifact_mapping(
        _mapping(
            receipt.get("source_prediction_seal_file_sha256"),
            name="source prediction seal digests",
        ),
        name="source prediction seal digests",
    )
    expected_names = {
        f"{outer_index:02d}-{target_index:02d}.json"
        for outer_index in range(10)
        for target_index in range(10)
    }
    _require(
        set(seal_digests) == expected_names,
        "source receipt seal roster changed",
    )
    identity = {key: item for key, item in receipt.items() if key != "receipt_id"}
    _require(
        receipt.get("receipt_id") == content_id(identity),
        "source receipt content identity changed",
    )
    return cast(dict[str, Any], plain_json(receipt))


def _fit_object_ids(
    cohort: Mapping[str, tuple[int, str]],
    *,
    outer_object_id: str,
    target_object_id: str,
) -> tuple[str, ...]:
    excluded = {outer_object_id}
    if target_object_id != outer_object_id:
        excluded.add(target_object_id)
    return tuple(sorted(set(cohort) - excluded))


def _technical_fallback_problem(
    *,
    object_id: str,
    episode_id: int,
    stratum: str,
    physical_prediction_m: np.ndarray,
    persistence_m: np.ndarray,
    physical_mode: str,
    implementation_revision: str,
    source_artifact_ids: Mapping[str, str],
    failure_stage: str,
    failure: Exception,
) -> Deform360JointSparsePredictionInputV5:
    """Construct an auditable exact-B0 carrier after a prefix provider failure."""

    physical = np.asarray(physical_prediction_m)
    persistence = np.asarray(persistence_m)
    state_count = 1
    evaluation_start, evaluation_stop = 58, 76
    future = np.zeros((*physical.shape, state_count), dtype=np.float64)
    query = future[evaluation_start:evaluation_stop].reshape(-1, 3, state_count)
    failure_identity = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-technical-failure-v5",
        "schema_version": 1,
        "object_id": object_id,
        "episode_id": episode_id,
        "failure_stage": failure_stage,
        "exception_type": type(failure).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(failure).encode("utf-8")
        ).hexdigest(),
        "implementation_revision": implementation_revision,
        "development_suffix_opened": False,
        "confirmation_payloads_opened": False,
    }
    failure_id = content_id(failure_identity)
    response = physical[evaluation_start:evaluation_stop] - persistence[
        evaluation_start:evaluation_stop
    ]
    response_scale = max(
        float(np.sqrt(np.mean(np.sum(np.square(response), axis=2)))),
        1e-9,
    )
    batch = GaugeAwareObservationBatch(
        innovation_m=np.zeros((1, 3), dtype=np.float64),
        observation_covariance_m2=np.eye(3, dtype=np.float64)[None],
        state_jacobian=np.zeros((1, 3, state_count), dtype=np.float64),
        gauge_jacobian=np.zeros((1, 3, 3), dtype=np.float64),
        shared_bias_jacobian=np.zeros((1, 3, 0), dtype=np.float64),
        view_bias_jacobian=np.zeros((1, 3, 0), dtype=np.float64),
        query_state_jacobian=query,
        gauge_prior_covariance=1e-4 * np.eye(3, dtype=np.float64),
        correlation_group_ids=(failure_id,),
        prior_reliability=np.zeros(1, dtype=np.float64),
        prior_nominal_probability=np.full(1, 0.9, dtype=np.float64),
        composite_weight=np.ones(1, dtype=np.float64),
        association_probability=np.zeros(1, dtype=np.float64),
        physical_response_scale_m=response_scale,
        state_prior_covariance_m2=1e-4 * np.eye(state_count, dtype=np.float64),
        metadata={
            **failure_identity,
            "failure_id": failure_id,
            "factor_admitted": False,
            "technical_failure": True,
            "synthetic_observation_used": False,
            "exact_physical_fallback_required": True,
        },
    )
    return Deform360JointSparsePredictionInputV5(
        object_id=object_id,
        episode_id=episode_id,
        stratum=stratum,
        physical_prediction_m=physical,
        persistence_m=persistence,
        last_causal_residual_m=np.zeros(physical.shape[1:], dtype=np.float64),
        future_state_jacobian_m=future,
        observation_batch=batch,
        causal_frame_stop=evaluation_start,
        evaluation_frame_range_half_open=(evaluation_start, evaluation_stop),
        factor_admitted=False,
        physical_mode=physical_mode,
        source_artifact_ids={
            **dict(source_artifact_ids),
            f"technical-failures/{object_id}.json": failure_id,
        },
    )


def publish_deform360_joint_sparse_source_prediction_panel_v5(
    *,
    execution_lock_path: str | Path,
    source_plan_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish all 100 source predictions without accepting any suffix path."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    plan_path = Path(source_plan_path).absolute().resolve(strict=True)
    plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        load_strict_json_object(plan_path, label="source prediction plan"),
        lock=lock,
    )
    root = _ordinary_root(input_root)
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    _require(output.is_dir() and not output.is_symlink(), "output root is invalid")
    cohort = _cohort(lock)
    revision = cast(str, plan["implementation_revision"])
    execution_lock_id = cast(str, plan["execution_lock_id"])
    common_sources = source_artifact_mapping(
        {
            "locks/source-execution-v5.json": _sha256_file(
                Path(execution_lock_path).resolve(strict=True)
            ),
            "plans/source-prediction-plan-v5.json": _sha256_file(plan_path),
        },
        name="source panel common artifacts",
    )
    object_rows = {
        cast(str, row["object_id"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Mapping[str, Any]], plan["objects"])
    }
    prepared: dict[str, tuple[Any, ...]] = {}
    base_fit = Deform360JointSparsePrefixFitV5(
        fit_object_ids=tuple(sorted(cohort)),
        source_artifact_ids=common_sources,
    )
    for object_id in sorted(cohort):
        row = object_rows[object_id]
        physical_record = cast(Mapping[str, Any], row["physical"])
        physical_path = _verified_file(
            root,
            physical_record,
            name=f"physical archive for {object_id}",
        )
        physical_mode = cast(str, physical_record["physical_mode"])
        physical, persistence = _load_physical_archive(
            physical_path,
            physical_mode=physical_mode,
        )
        prefix = cast(Sequence[int], row["raw_prefix_range_half_open"])
        raw_prefix = (int(prefix[0]), int(prefix[1]))
        visual_inputs: list[tuple[str, Path, Path]] = []
        object_sources = {
            **dict(common_sources),
            f"physical/{object_id}.npz": _sha256_file(physical_path),
        }
        for visual in cast(Sequence[Mapping[str, Any]], row["visual_windows"]):
            decoded = _verified_file(
                root,
                cast(Mapping[str, Any], visual["decoded_uniform"]),
                name=f"decoded uniform for {object_id}/{visual['camera_id']}",
            )
            metric = _verified_file(
                root,
                cast(Mapping[str, Any], visual["metric_prefix"]),
                name=f"metric prefix for {object_id}/{visual['camera_id']}",
            )
            camera_id = cast(str, visual["camera_id"])
            visual_inputs.append((camera_id, decoded, metric))
            object_sources.update(
                {
                    f"visual/{object_id}/{camera_id}/decoded-uniform.npz": (
                        _sha256_file(decoded)
                    ),
                    f"visual/{object_id}/{camera_id}/metric-prefix.npz": (
                        _sha256_file(metric)
                    ),
                }
            )
        contact_path = _verified_contact_directory(
            root,
            cast(Mapping[str, Any], row["contact_prefix"]),
        )
        contact_record = cast(Mapping[str, Any], row["contact_prefix"])
        object_sources.update(
            {
                f"contact/{object_id}/contact-prefix.json": cast(
                    str, contact_record["manifest_file_sha256"]
                ),
                f"contact/{object_id}/materialization-id": cast(
                    str, contact_record["materialization_id"]
                ),
            }
        )
        episode_id, _stratum = cohort[object_id]
        technical_failure: tuple[str, Exception] | None = None
        visual_rows = []
        try:
            for camera_id, decoded, metric in visual_inputs:
                rows, _gauge = prepare_deform360_joint_sparse_visual_window_v5(
                    camera_id=camera_id,
                    decoded_uniform_path=decoded,
                    metric_prefix_path=metric,
                    raw_prefix_range_half_open=raw_prefix,
                    fit=base_fit,
                    source_artifact_ids=object_sources,
                )
                visual_rows.append(rows)
            contact = prepare_deform360_joint_sparse_contact_rows_v5(
                contact_prefix_directory=contact_path,
                object_id=object_id,
                episode_id=episode_id,
                raw_prefix_range_half_open=raw_prefix,
                physical_prediction_m=physical,
                source_artifact_ids=object_sources,
            )
            residual = estimate_deform360_last_causal_residual_v5(
                visual_windows=tuple(visual_rows),
                physical_prediction_m=physical,
                causal_frame_stop=58,
            )
        except (OSError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            technical_failure = ("prefix_provider", error)
            contact = None
            residual = np.zeros(physical.shape[1:], dtype=np.float64)
        prepared[object_id] = (
            physical,
            persistence,
            physical_mode,
            tuple(visual_rows),
            contact,
            residual,
            object_sources,
            technical_failure,
        )

    prediction_root = output / "predictions"
    source_seal_root = output / "source-seals"
    prediction_root.mkdir(parents=True, exist_ok=True)
    source_seal_root.mkdir(parents=True, exist_ok=True)
    source_seals: list[dict[str, Any]] = []
    seal_file_digests: dict[str, str] = {}
    ordered_ids = tuple(sorted(cohort))
    for outer_index, outer_id in enumerate(ordered_ids):
        for target_index, target_id in enumerate(ordered_ids):
            fit_ids = _fit_object_ids(
                cohort,
                outer_object_id=outer_id,
                target_object_id=target_id,
            )
            fit = Deform360JointSparsePrefixFitV5(
                fit_object_ids=fit_ids,
                source_artifact_ids=common_sources,
            )
            (
                physical,
                persistence,
                physical_mode,
                windows,
                contact,
                residual,
                sources,
                technical_failure,
            ) = prepared[target_id]
            episode_id, stratum = cohort[target_id]
            if technical_failure is None:
                materialized = materialize_deform360_joint_sparse_prediction_v5(
                    object_id=target_id,
                    episode_id=episode_id,
                    stratum=cast(Any, stratum),
                    physical_prediction_m=cast(np.ndarray, physical),
                    persistence_m=cast(np.ndarray, persistence),
                    last_causal_residual_m=cast(np.ndarray, residual),
                    physical_mode=cast(str, physical_mode),
                    causal_frame_stop=58,
                    evaluation_frame_range_half_open=(58, 76),
                    visual_windows=cast(Sequence[Any], windows),
                    contact_rows=cast(Any, contact),
                    fit=fit,
                    implementation_revision=revision,
                    source_artifact_ids=cast(Mapping[str, str], sources),
                )
                problem = materialized.problem
            else:
                failure_stage, failure = cast(tuple[str, Exception], technical_failure)
                problem = _technical_fallback_problem(
                    object_id=target_id,
                    episode_id=episode_id,
                    stratum=stratum,
                    physical_prediction_m=cast(np.ndarray, physical),
                    persistence_m=cast(np.ndarray, persistence),
                    physical_mode=cast(str, physical_mode),
                    implementation_revision=revision,
                    source_artifact_ids=cast(Mapping[str, str], sources),
                    failure_stage=failure_stage,
                    failure=failure,
                )
            result = run_deform360_joint_sparse_prediction_v5(problem)
            relative_directory = (
                f"{outer_index:02d}-{outer_id}/{target_index:02d}-{target_id}"
            )
            prediction_directory = prediction_root / relative_directory
            if prediction_directory.exists():
                prediction_seal, existing_result = (
                    load_deform360_joint_sparse_prediction_v5(prediction_directory)
                )
                _require(
                    prediction_seal["input_id"] == problem.input_id
                    and existing_result.result_id == result.result_id
                    and prediction_seal["prediction_fit_artifact_id"]
                    == fit.fit_artifact_id,
                    "existing source prediction differs",
                )
            else:
                prediction_seal = publish_deform360_joint_sparse_prediction_v5(
                    problem,
                    result,
                    prediction_directory,
                    execution_lock_id=execution_lock_id,
                    implementation_revision=revision,
                    prediction_fit_artifact_id=fit.fit_artifact_id,
                    prediction_fit_object_ids=fit_ids,
                )
            method_ids = cast(Mapping[str, str], prediction_seal["method_artifact_ids"])
            features = cast(
                Mapping[str, float], prediction_seal["predicted_loss_features_m"]
            )
            source_seal = build_deform360_joint_sparse_source_prediction_seal_v5(
                lock=lock,
                implementation_revision=revision,
                outer_held_out_object_id=outer_id,
                record_role="held_out" if outer_id == target_id else "training",
                object_id=target_id,
                factor_admitted=bool(prediction_seal["factor_admitted"]),
                technical_failure=technical_failure is not None,
                physical_mode=cast(str, prediction_seal["physical_mode"]),
                risk_score=float(prediction_seal["risk_score"]),
                prediction_fit_artifact_id=fit.fit_artifact_id,
                prediction_fit_object_ids=fit_ids,
                methods={
                    method_id: {
                        "artifact_id": method_ids[method_id],
                        "predicted_loss_mm": 1000.0 * float(features[method_id]),
                    }
                    for method_id in RAW_METHOD_IDS
                },
                source_artifacts={
                    **dict(cast(Mapping[str, str], prediction_seal["source_artifact_ids"])),
                    f"predictions/{relative_directory}/{PREDICTION_SEAL_FILENAME}": (
                        _sha256_file(prediction_directory / PREDICTION_SEAL_FILENAME)
                    ),
                },
            )
            validate_deform360_joint_sparse_source_prediction_seal_v5(
                source_seal,
                lock,
            )
            source_seal_path = source_seal_root / f"{outer_index:02d}-{target_index:02d}.json"
            _publish_or_validate_json(
                source_seal,
                source_seal_path,
                label="source prediction seal",
            )
            source_seals.append(source_seal)
            seal_file_digests[source_seal_path.name] = _sha256_file(source_seal_path)

    batch = build_deform360_joint_sparse_source_prediction_batch_v5(
        source_seals,
        lock,
    )
    batch_path = output / "source-prediction-batch.json"
    if batch_path.exists():
        _publish_or_validate_json(batch, batch_path, label="source prediction batch")
    else:
        publish_deform360_joint_sparse_source_prediction_batch_v5(
            batch,
            lock=lock,
            output_path=batch_path,
        )
    receipt_identity: dict[str, Any] = {
        "schema": SOURCE_PANEL_RECEIPT_SCHEMA,
        "schema_version": SOURCE_PANEL_RECEIPT_VERSION,
        "execution_lock_id": execution_lock_id,
        "implementation_revision": revision,
        "plan_id": plan["plan_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "prediction_batch_file_sha256": _sha256_file(batch_path),
        "prediction_record_count": 100,
        "source_prediction_seal_file_sha256": dict(sorted(seal_file_digests.items())),
        "information_boundary": dict(SOURCE_PLAN_BOUNDARY),
    }
    receipt = {**receipt_identity, "receipt_id": content_id(receipt_identity)}
    require_exact_fields(receipt, expected=_RECEIPT_FIELDS, name="source receipt")
    _publish_or_validate_json(
        receipt,
        output / "source-prediction-receipt.json",
        label="source prediction receipt",
    )
    return receipt


__all__ = [
    "SOURCE_PLAN_BOUNDARY",
    "SOURCE_PLAN_SCHEMA",
    "SOURCE_PLAN_SEMANTICS",
    "SOURCE_PLAN_VERSION",
    "build_deform360_joint_sparse_source_prediction_plan_v5",
    "publish_deform360_joint_sparse_source_prediction_panel_v5",
    "validate_deform360_joint_sparse_source_prediction_plan_v5",
    "validate_deform360_joint_sparse_source_prediction_receipt_v5",
]
