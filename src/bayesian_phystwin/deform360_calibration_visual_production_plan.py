"""Target-blind execution plans for official-Hub calibration visual production.

The visual-provider lock fixes model and estimator semantics, while the successful
calibration-source result fixes the exact ten objects, calibrated camera roster,
and action-only frame windows. This module closes the remaining execution
ambiguity without opening confirmation payloads or reading target outcomes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._canonical_contracts import plain_json
from ._deform360_calibration_run_common import load_json_object
from ._portable_contracts import content_id, exact_revision, write_atomic_json
from .deform360_calibration_observability_case_builder import _load_context
from .deform360_visual_provider_lock import load_deform360_visual_provider_lock

DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-visual-production-plan"
)
DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_VERSION: Final = 1
DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SEMANTICS: Final = (
    "target-blind-all-aligned-cameras-motioncrafter-prob4d-v1"
)
DEFORM360_CALIBRATION_OBJECT_SEED_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-object-seed.v1"
)
DEFORM360_CALIBRATION_VIEW_SEED_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-view-seed.v1"
)
DEFORM360_CALIBRATION_DEPENDENCE_GROUP_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-dependence-group.v1"
)
DEFORM360_CALIBRATION_OBJECT_COUNT: Final = 10
DEFORM360_CALIBRATION_PER_STRATUM_COUNT: Final = 5
DEFORM360_CALIBRATION_SELECTED_FRAME_COUNT: Final = 81
DEFORM360_CALIBRATION_PREDICTION_FRAME_COUNT: Final = 76
DEFORM360_CALIBRATION_PREFIX_FRAME_COUNT: Final = 58
DEFORM360_CALIBRATION_MINIMUM_CAMERA_COUNT: Final = 8
PROB4D_MOTIONCRAFTER_SEED_POLICY: Final = "derived-per-call"
CAMERA_ROSTER_POLICY: Final = (
    "all-source-prepared-calibrated-cameras-lexicographic-v1"
)
FRAME_POLICY: Final = "selected-81-predict-first-76-prefix-first-58-v1"
VIEW_SUBSTREAM_POLICY: Final = "per-camera-derived-substream-v1"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "plan_id",
        "implementation_revision",
        "protocol_id",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_result_sha256",
        "confirmation_object_set_sha256",
        "provider",
        "production_policy",
        "object_count",
        "camera_view_count",
        "objects",
        "information_boundary",
    }
)
_PROVIDER_FIELDS = frozenset(
    {
        "repository",
        "revision",
        "api_version",
        "stream_contract_version",
        "motioncrafter_repository",
        "motioncrafter_revision",
        "model_set_id",
        "root_seed",
        "seed_policy",
        "window_size",
        "overlap",
        "height",
        "width",
        "storage_dtype",
        "initial_metric_frame_prior_id",
        "additional_metric_anchor_policy",
        "max_gauge_rank",
        "minimum_retained_gauge_trace",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "camera_roster_policy",
        "frame_policy",
        "object_seed_schema",
        "view_seed_schema",
        "view_substream_policy",
        "prob4d_motioncrafter_seed_policy",
        "source_episode_directory",
        "source_video_filename",
        "source_timestamps_filename",
        "output_layout",
        "no_replacement",
    }
)
_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "episode_id",
        "stratum",
        "object_root_seed",
        "selected_source_frame_range_half_open",
        "prediction_source_frame_range_half_open",
        "prefix_source_frame_range_half_open",
        "camera_count",
        "cameras",
    }
)
_CAMERA_FIELDS = frozenset(
    {
        "camera_id",
        "view_root_seed",
        "call_namespace",
        "source_video_relative_path",
        "source_timestamps_relative_path",
        "output_relative_directory",
        "dependence_group_ids",
    }
)
_INFORMATION_BOUNDARY = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    _require(
        not missing and not extra,
        f"{name} fields changed: missing={missing}, extra={extra}",
    )


def _literal_string(value: object, *, name: str) -> str:
    _require(
        type(value) is str and bool(value),
        f"{name} must be a nonempty literal string",
    )
    result = cast(str, value)
    _require(
        result == result.strip(),
        f"{name} must not contain surrounding whitespace",
    )
    return result


def _literal_integer(value: object, *, name: str, minimum: int = 0) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    result = cast(int, value)
    _require(result >= minimum, f"{name} must be an integer >= {minimum}")
    return result


def _sha256(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    _require(len(result) == 64, f"{name} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    _require(result == result.lower(), f"{name} must be lowercase")
    return result


def _safe_relative_path(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    path = PurePosixPath(result)
    _require(
        not path.is_absolute()
        and "\\" not in result
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} must be a safe POSIX relative path",
    )
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _seed(payload: Mapping[str, Any]) -> int:
    return int.from_bytes(bytes.fromhex(_canonical_sha256(payload))[:4], "big")


def deform360_calibration_object_seed(
    *,
    root_seed: int,
    visual_provider_lock_id: str,
    object_id: str,
    episode_id: int,
) -> int:
    """Return the frozen 32-bit root seed for one calibration object."""

    return _seed(
        {
            "schema": DEFORM360_CALIBRATION_OBJECT_SEED_SCHEMA,
            "root_seed": _literal_integer(root_seed, name="root_seed"),
            "visual_provider_lock_id": _sha256(
                visual_provider_lock_id,
                name="visual_provider_lock_id",
            ),
            "object_id": _literal_string(object_id, name="object_id"),
            "episode_id": _literal_integer(episode_id, name="episode_id"),
        }
    )


def deform360_calibration_view_seed(*, object_seed: int, camera_id: str) -> int:
    """Return a deterministic camera substream seed for one object/view pair."""

    return _seed(
        {
            "schema": DEFORM360_CALIBRATION_VIEW_SEED_SCHEMA,
            "object_seed": _literal_integer(object_seed, name="object_seed"),
            "camera_id": _literal_string(camera_id, name="camera_id"),
        }
    )


def _dependence_group_id(*, kind: str, payload: Mapping[str, Any]) -> str:
    return content_id(
        {
            "schema": DEFORM360_CALIBRATION_DEPENDENCE_GROUP_SCHEMA,
            "kind": kind,
            **dict(payload),
        }
    )


def _frame_range(value: object, *, name: str, expected_count: int) -> list[int]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2,
        f"{name} must contain two integer bounds",
    )
    start = _literal_integer(value[0], name=f"{name}[0]")
    stop = _literal_integer(value[1], name=f"{name}[1]")
    _require(
        stop - start == expected_count,
        f"{name} must contain exactly {expected_count} frames",
    )
    return [start, stop]


def _camera_record(
    *,
    object_id: str,
    episode_id: int,
    camera_id: str,
    object_seed: int,
    model_set_id: str,
) -> dict[str, Any]:
    camera = _literal_string(camera_id, name="camera_id")
    _require(
        "/" not in camera and camera not in {".", ".."},
        "camera_id is not path-safe",
    )
    view_seed = deform360_calibration_view_seed(
        object_seed=object_seed,
        camera_id=camera,
    )
    source_root = f"{object_id}/episode_0000/{camera}"
    output_root = f"objects/{object_id}/episode_{episode_id:04d}/views/{camera}"
    call_namespace = (
        f"deform360-calibration:{object_id}:episode-{episode_id:04d}:{camera}"
    )
    return {
        "camera_id": camera,
        "view_root_seed": view_seed,
        "call_namespace": call_namespace,
        "source_video_relative_path": f"{source_root}/undistorted.mp4",
        "source_timestamps_relative_path": (
            f"{source_root}/aligned_timestamps.txt"
        ),
        "output_relative_directory": output_root,
        "dependence_group_ids": [
            _dependence_group_id(
                kind="shared-model-set",
                payload={"model_set_id": model_set_id},
            ),
            _dependence_group_id(
                kind="shared-object-scene",
                payload={"object_id": object_id, "episode_id": episode_id},
            ),
        ],
    }


def _validate_camera_record(
    value: object,
    *,
    object_id: str,
    episode_id: int,
    object_seed: int,
    model_set_id: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "camera plan must be a JSON object")
    record = dict(value)
    _exact_fields(record, _CAMERA_FIELDS, "camera plan")
    camera_id = _literal_string(record["camera_id"], name="camera_id")
    expected = _camera_record(
        object_id=object_id,
        episode_id=episode_id,
        camera_id=camera_id,
        object_seed=object_seed,
        model_set_id=model_set_id,
    )
    _require(
        record == expected,
        f"camera plan changed for {object_id}/{camera_id}",
    )
    for key in (
        "source_video_relative_path",
        "source_timestamps_relative_path",
        "output_relative_directory",
    ):
        _safe_relative_path(record[key], name=key)
    return record


def validate_deform360_calibration_visual_production_plan(
    value: object,
) -> dict[str, Any]:
    """Strictly validate and return an ordinary finite JSON plan."""

    _require(
        isinstance(value, Mapping),
        "visual production plan must be a JSON object",
    )
    try:
        plan = json.loads(
            json.dumps(plain_json(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "visual production plan must contain finite JSON"
        ) from error
    _require(
        isinstance(plan, dict),
        "visual production plan must be a JSON object",
    )
    _exact_fields(plan, _TOP_LEVEL_FIELDS, "visual production plan")
    _require(
        plan["schema"] == DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SCHEMA,
        "unsupported visual production plan schema",
    )
    _require(
        plan["schema_version"]
        == DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_VERSION,
        "unsupported visual production plan version",
    )
    _require(
        plan["semantics"]
        == DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SEMANTICS,
        "visual production plan semantics changed",
    )
    exact_revision(plan["implementation_revision"], name="implementation_revision")
    _literal_string(plan["protocol_id"], name="protocol_id")
    selection_id = _sha256(
        plan["selection_artifact_sha256"],
        name="selection_artifact_sha256",
    )
    provider_lock_id = _sha256(
        plan["visual_provider_lock_id"],
        name="visual_provider_lock_id",
    )
    _sha256(
        plan["calibration_source_run_record_sha256"],
        name="calibration_source_run_record_sha256",
    )
    _sha256(
        plan["calibration_source_result_sha256"],
        name="calibration_source_result_sha256",
    )
    _sha256(
        plan["confirmation_object_set_sha256"],
        name="confirmation_object_set_sha256",
    )

    provider = plan["provider"]
    _require(
        isinstance(provider, Mapping),
        "provider declaration must be a JSON object",
    )
    provider = dict(provider)
    _exact_fields(provider, _PROVIDER_FIELDS, "provider declaration")
    _require(
        provider["repository"] == "IPS-Stuttgart/Prob4D",
        "provider repository changed",
    )
    exact_revision(provider["revision"], name="provider revision")
    _require(provider["api_version"] == 2, "provider API version changed")
    _require(
        provider["stream_contract_version"] == 2,
        "stream contract version changed",
    )
    _require(
        provider["motioncrafter_repository"] == "TencentARC/MotionCrafter",
        "MotionCrafter repository changed",
    )
    exact_revision(
        provider["motioncrafter_revision"],
        name="MotionCrafter revision",
    )
    model_set_id = _sha256(provider["model_set_id"], name="model_set_id")
    root_seed = _literal_integer(provider["root_seed"], name="root_seed")
    _require(root_seed < 2**32, "root_seed must fit in 32 bits")
    _require(
        provider["seed_policy"] == "per-object-derived-seed-v1",
        "object seed policy changed",
    )
    _literal_integer(
        provider["window_size"],
        name="window_size",
        minimum=2,
    )
    overlap = _literal_integer(provider["overlap"], name="overlap")
    _require(
        overlap < provider["window_size"],
        "overlap must be smaller than window_size",
    )
    for key in ("height", "width"):
        _literal_integer(provider[key], name=key, minimum=1)
    _require(
        provider["storage_dtype"] in {"float32", "float64"},
        "storage dtype changed",
    )
    _sha256(
        provider["initial_metric_frame_prior_id"],
        name="initial_metric_frame_prior_id",
    )
    _require(
        provider["additional_metric_anchor_policy"]
        in {"none", "independent_sparse"},
        "additional metric-anchor policy changed",
    )
    max_gauge_rank = provider["max_gauge_rank"]
    if max_gauge_rank is not None:
        _literal_integer(max_gauge_rank, name="max_gauge_rank", minimum=1)
    retained_trace = provider["minimum_retained_gauge_trace"]
    _require(
        isinstance(retained_trace, (int, float))
        and not isinstance(retained_trace, bool)
        and 0.0 < float(retained_trace) <= 1.0,
        "minimum_retained_gauge_trace must be in (0, 1]",
    )

    policy = plan["production_policy"]
    _require(
        isinstance(policy, Mapping),
        "production policy must be a JSON object",
    )
    policy = dict(policy)
    _exact_fields(policy, _POLICY_FIELDS, "production policy")
    expected_policy = {
        "camera_roster_policy": CAMERA_ROSTER_POLICY,
        "frame_policy": FRAME_POLICY,
        "object_seed_schema": DEFORM360_CALIBRATION_OBJECT_SEED_SCHEMA,
        "view_seed_schema": DEFORM360_CALIBRATION_VIEW_SEED_SCHEMA,
        "view_substream_policy": VIEW_SUBSTREAM_POLICY,
        "prob4d_motioncrafter_seed_policy": PROB4D_MOTIONCRAFTER_SEED_POLICY,
        "source_episode_directory": "episode_0000",
        "source_video_filename": "undistorted.mp4",
        "source_timestamps_filename": "aligned_timestamps.txt",
        "output_layout": (
            "objects/{object_id}/episode_{episode_id:04d}/views/{camera_id}"
        ),
        "no_replacement": True,
    }
    _require(policy == expected_policy, "production policy changed")
    _require(
        plan["information_boundary"] == _INFORMATION_BOUNDARY,
        "information boundary changed",
    )

    objects = plan["objects"]
    _require(
        isinstance(objects, list)
        and len(objects) == DEFORM360_CALIBRATION_OBJECT_COUNT,
        "plan must contain exactly ten calibration objects",
    )
    object_ids: list[str] = []
    object_seeds: list[int] = []
    view_seeds: list[int] = []
    outputs: list[str] = []
    strata = {"sheet": 0, "volumetric": 0}
    camera_view_count = 0
    for item in objects:
        _require(isinstance(item, Mapping), "object plan must be a JSON object")
        object_plan = dict(item)
        _exact_fields(object_plan, _OBJECT_FIELDS, "object plan")
        object_id = _literal_string(object_plan["object_id"], name="object_id")
        episode_id = _literal_integer(
            object_plan["episode_id"],
            name="episode_id",
        )
        stratum = object_plan["stratum"]
        _require(stratum in strata, "object stratum changed")
        strata[cast(str, stratum)] += 1
        object_seed = _literal_integer(
            object_plan["object_root_seed"],
            name="object_root_seed",
        )
        _require(
            object_seed
            == deform360_calibration_object_seed(
                root_seed=root_seed,
                visual_provider_lock_id=provider_lock_id,
                object_id=object_id,
                episode_id=episode_id,
            ),
            f"object seed changed for {object_id}",
        )
        selected = _frame_range(
            object_plan["selected_source_frame_range_half_open"],
            name="selected source frame range",
            expected_count=DEFORM360_CALIBRATION_SELECTED_FRAME_COUNT,
        )
        prediction = _frame_range(
            object_plan["prediction_source_frame_range_half_open"],
            name="prediction source frame range",
            expected_count=DEFORM360_CALIBRATION_PREDICTION_FRAME_COUNT,
        )
        prefix = _frame_range(
            object_plan["prefix_source_frame_range_half_open"],
            name="prefix source frame range",
            expected_count=DEFORM360_CALIBRATION_PREFIX_FRAME_COUNT,
        )
        _require(
            selected[0] == prediction[0] == prefix[0],
            "object frame ranges must share the selected start",
        )
        _require(
            prediction[1] <= selected[1] and prefix[1] <= prediction[1],
            "object frame ranges are not nested",
        )
        cameras = object_plan["cameras"]
        camera_count = _literal_integer(
            object_plan["camera_count"],
            name="camera_count",
            minimum=DEFORM360_CALIBRATION_MINIMUM_CAMERA_COUNT,
        )
        _require(
            isinstance(cameras, list) and len(cameras) == camera_count,
            "camera count differs from camera plans",
        )
        validated_cameras = [
            _validate_camera_record(
                camera,
                object_id=object_id,
                episode_id=episode_id,
                object_seed=object_seed,
                model_set_id=model_set_id,
            )
            for camera in cameras
        ]
        camera_ids = [camera["camera_id"] for camera in validated_cameras]
        _require(
            camera_ids == sorted(camera_ids)
            and len(set(camera_ids)) == len(camera_ids),
            "camera roster must be sorted and unique",
        )
        object_ids.append(object_id)
        object_seeds.append(object_seed)
        view_seeds.extend(
            camera["view_root_seed"] for camera in validated_cameras
        )
        outputs.extend(
            camera["output_relative_directory"] for camera in validated_cameras
        )
        camera_view_count += camera_count
    _require(
        object_ids == sorted(object_ids)
        and len(set(object_ids)) == len(object_ids),
        "object plans must be sorted and unique",
    )
    _require(
        len(set(object_seeds)) == len(object_seeds),
        "object seed collision detected",
    )
    _require(
        len(set(view_seeds)) == len(view_seeds),
        "view seed collision detected",
    )
    _require(len(set(outputs)) == len(outputs), "output path collision detected")
    _require(
        strata == {"sheet": 5, "volumetric": 5},
        "plan must retain five objects per stratum",
    )
    _require(plan["object_count"] == len(objects), "object_count changed")
    _require(
        plan["camera_view_count"] == camera_view_count,
        "camera_view_count changed",
    )

    declared = _sha256(plan["plan_id"], name="plan_id")
    payload = {key: item for key, item in plan.items() if key != "plan_id"}
    _require(declared == content_id(payload), "plan_id does not match plan content")
    _require(
        selection_id == plan["selection_artifact_sha256"],
        "selection identity changed",
    )
    return plan


def build_deform360_calibration_visual_production_plan(
    *,
    source_protocol_path: str | Path,
    stage0_protocol_path: str | Path,
    selection_lock_path: str | Path,
    visual_provider_lock_path: str | Path,
    calibration_source_plan_path: str | Path,
    calibration_source_download_path: str | Path,
    calibration_source_run_record_path: str | Path,
    calibration_source_result_path: str | Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Build one target-blind plan from the successful ten-object source chain."""

    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    provider_lock = load_deform360_visual_provider_lock(
        visual_provider_lock_path
    )
    provider_record = provider_lock.to_record()
    _require(
        provider_record["seed_policy"] == "per-object-derived-seed-v1",
        "visual-provider seed policy is unsupported",
    )
    result_value, result_file_sha256 = load_json_object(
        Path(calibration_source_result_path)
    )
    rows = result_value.get("objects")
    _require(
        isinstance(rows, list)
        and len(rows) == DEFORM360_CALIBRATION_OBJECT_COUNT,
        "successful result must contain ten object rows",
    )
    row_ids = [row.get("object_id") for row in rows if isinstance(row, Mapping)]
    _require(
        len(row_ids) == DEFORM360_CALIBRATION_OBJECT_COUNT
        and all(type(value) is str for value in row_ids),
        "result object identities are malformed",
    )

    contexts = [
        _load_context(
            source_protocol_path=source_protocol_path,
            stage0_protocol_path=stage0_protocol_path,
            selection_lock_path=selection_lock_path,
            visual_provider_lock_path=visual_provider_lock_path,
            calibration_source_plan_path=calibration_source_plan_path,
            calibration_source_download_path=calibration_source_download_path,
            calibration_source_run_record_path=calibration_source_run_record_path,
            calibration_source_result_path=calibration_source_result_path,
            object_id=cast(str, object_id),
        )
        for object_id in sorted(cast(list[str], row_ids))
    ]
    _require(
        all(
            context.result_row.get("status") == "source_prepared"
            for context in contexts
        ),
        "visual production requires all ten prepared objects",
    )
    selection_id = contexts[0].selection_artifact_sha256
    visual_lock_id = contexts[0].visual_provider_lock_id
    run_record_id = contexts[0].run_record_sha256
    _require(
        all(
            context.selection_artifact_sha256 == selection_id
            for context in contexts
        ),
        "selection identity changed across objects",
    )
    _require(
        all(
            context.visual_provider_lock_id == visual_lock_id
            for context in contexts
        ),
        "provider identity changed across objects",
    )
    _require(
        all(
            context.run_record_sha256 == run_record_id for context in contexts
        ),
        "run-record identity changed across objects",
    )
    _require(
        visual_lock_id == provider_lock.artifact_id,
        "provider lock identity changed",
    )

    selection_value, _ = load_json_object(Path(selection_lock_path))
    selection = selection_value.get("selection")
    _require(isinstance(selection, Mapping), "selection lock lacks cohorts")
    confirmation = selection.get("confirmation")
    _require(
        isinstance(confirmation, list) and len(confirmation) == 12,
        "selection lock lacks twelve confirmation objects",
    )
    confirmation_ids = sorted(
        _literal_string(row.get("object_id"), name="confirmation object_id")
        for row in confirmation
        if isinstance(row, Mapping)
    )
    _require(
        len(confirmation_ids) == 12 and len(set(confirmation_ids)) == 12,
        "confirmation object set is malformed",
    )

    model_set_id = cast(str, provider_record["model_set_id"])
    root_seed = cast(int, provider_record["root_seed"])
    object_records: list[dict[str, Any]] = []
    for context in sorted(contexts, key=lambda item: item.object_id):
        row = context.result_row
        cameras_value = row.get("cameras")
        _require(
            isinstance(cameras_value, list),
            f"prepared camera roster is missing: {context.object_id}",
        )
        cameras = sorted(
            _literal_string(camera, name="camera_id")
            for camera in cameras_value
        )
        _require(
            len(cameras) >= DEFORM360_CALIBRATION_MINIMUM_CAMERA_COUNT
            and len(set(cameras)) == len(cameras),
            f"prepared camera roster is invalid: {context.object_id}",
        )
        _require(
            row.get("camera_count") == len(cameras),
            f"prepared camera count changed: {context.object_id}",
        )
        action_window = row.get("action_window")
        _require(
            isinstance(action_window, Mapping),
            f"action window is missing: {context.object_id}",
        )
        selected = _frame_range(
            action_window.get("selected_raw_frame_range_half_open"),
            name=f"{context.object_id} selected frame range",
            expected_count=DEFORM360_CALIBRATION_SELECTED_FRAME_COUNT,
        )
        prediction = _frame_range(
            action_window.get("prediction_raw_frame_range_half_open"),
            name=f"{context.object_id} prediction frame range",
            expected_count=DEFORM360_CALIBRATION_PREDICTION_FRAME_COUNT,
        )
        prefix = _frame_range(
            action_window.get("prefix_raw_frame_range_half_open"),
            name=f"{context.object_id} prefix frame range",
            expected_count=DEFORM360_CALIBRATION_PREFIX_FRAME_COUNT,
        )
        _require(
            selected[0] == prediction[0] == prefix[0],
            f"prepared frame starts differ: {context.object_id}",
        )
        object_seed = deform360_calibration_object_seed(
            root_seed=root_seed,
            visual_provider_lock_id=visual_lock_id,
            object_id=context.object_id,
            episode_id=context.episode_id,
        )
        object_records.append(
            {
                "object_id": context.object_id,
                "episode_id": context.episode_id,
                "stratum": context.stratum,
                "object_root_seed": object_seed,
                "selected_source_frame_range_half_open": selected,
                "prediction_source_frame_range_half_open": prediction,
                "prefix_source_frame_range_half_open": prefix,
                "camera_count": len(cameras),
                "cameras": [
                    _camera_record(
                        object_id=context.object_id,
                        episode_id=context.episode_id,
                        camera_id=camera,
                        object_seed=object_seed,
                        model_set_id=model_set_id,
                    )
                    for camera in cameras
                ],
            }
        )

    provider = {
        "repository": provider_record["provider_repository"],
        "revision": provider_record["provider_revision"],
        "api_version": provider_record["provider_api_version"],
        "stream_contract_version": provider_record["stream_contract_version"],
        "motioncrafter_repository": provider_record["motioncrafter_repository"],
        "motioncrafter_revision": provider_record["motioncrafter_revision"],
        "model_set_id": model_set_id,
        "root_seed": root_seed,
        "seed_policy": provider_record["seed_policy"],
        "window_size": provider_record["window_size"],
        "overlap": provider_record["overlap"],
        "height": provider_record["height"],
        "width": provider_record["width"],
        "storage_dtype": provider_record["storage_dtype"],
        "initial_metric_frame_prior_id": provider_record[
            "initial_metric_frame_prior_id"
        ],
        "additional_metric_anchor_policy": provider_record[
            "additional_metric_anchor_policy"
        ],
        "max_gauge_rank": provider_record["max_gauge_rank"],
        "minimum_retained_gauge_trace": provider_record[
            "minimum_retained_gauge_trace"
        ],
    }
    payload: dict[str, Any] = {
        "schema": DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SCHEMA,
        "schema_version": DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_VERSION,
        "semantics": DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SEMANTICS,
        "implementation_revision": implementation,
        "protocol_id": provider_record["protocol_id"],
        "selection_artifact_sha256": selection_id,
        "visual_provider_lock_id": visual_lock_id,
        "calibration_source_run_record_sha256": run_record_id,
        "calibration_source_result_sha256": cast(
            str,
            result_value["result_sha256"],
        ),
        "confirmation_object_set_sha256": _canonical_sha256(
            {"object_ids": confirmation_ids}
        ),
        "provider": provider,
        "production_policy": {
            "camera_roster_policy": CAMERA_ROSTER_POLICY,
            "frame_policy": FRAME_POLICY,
            "object_seed_schema": DEFORM360_CALIBRATION_OBJECT_SEED_SCHEMA,
            "view_seed_schema": DEFORM360_CALIBRATION_VIEW_SEED_SCHEMA,
            "view_substream_policy": VIEW_SUBSTREAM_POLICY,
            "prob4d_motioncrafter_seed_policy": (
                PROB4D_MOTIONCRAFTER_SEED_POLICY
            ),
            "source_episode_directory": "episode_0000",
            "source_video_filename": "undistorted.mp4",
            "source_timestamps_filename": "aligned_timestamps.txt",
            "output_layout": (
                "objects/{object_id}/episode_{episode_id:04d}/views/{camera_id}"
            ),
            "no_replacement": True,
        },
        "object_count": len(object_records),
        "camera_view_count": sum(
            cast(int, record["camera_count"]) for record in object_records
        ),
        "objects": object_records,
        "information_boundary": dict(_INFORMATION_BOUNDARY),
    }
    plan = {**payload, "plan_id": content_id(payload)}
    validated = validate_deform360_calibration_visual_production_plan(plan)
    _require(
        result_file_sha256
        == hashlib.sha256(Path(calibration_source_result_path).read_bytes()).hexdigest(),
        "calibration-source result changed during planning",
    )
    return validated


def save_deform360_calibration_visual_production_plan(
    path: str | Path,
    plan: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one validated production plan."""

    write_atomic_json(
        validate_deform360_calibration_visual_production_plan(plan),
        path,
        overwrite=overwrite,
    )


def load_deform360_calibration_visual_production_plan(
    path: str | Path,
) -> dict[str, Any]:
    """Load a duplicate-key-safe plan and revalidate every derived field."""

    value, _ = load_json_object(Path(path))
    return validate_deform360_calibration_visual_production_plan(value)


__all__ = [
    "CAMERA_ROSTER_POLICY",
    "DEFORM360_CALIBRATION_OBJECT_SEED_SCHEMA",
    "DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SCHEMA",
    "DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_SEMANTICS",
    "DEFORM360_CALIBRATION_VISUAL_PRODUCTION_PLAN_VERSION",
    "DEFORM360_CALIBRATION_VIEW_SEED_SCHEMA",
    "FRAME_POLICY",
    "PROB4D_MOTIONCRAFTER_SEED_POLICY",
    "VIEW_SUBSTREAM_POLICY",
    "build_deform360_calibration_visual_production_plan",
    "deform360_calibration_object_seed",
    "deform360_calibration_view_seed",
    "load_deform360_calibration_visual_production_plan",
    "save_deform360_calibration_visual_production_plan",
    "validate_deform360_calibration_visual_production_plan",
]
