"""Source-chain admission for Deform360 geometric v4 materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
)
from .deform360_joint_sparse_geometric_common_v4 import (
    METRIC_BATCH_SCHEMA,
    METRIC_BATCH_SEMANTICS,
    METRIC_BATCH_VERSION,
    METRIC_PLAN_SCHEMA,
    METRIC_PLAN_SEMANTICS,
    METRIC_PLAN_VERSION,
    PLAN_CASE_FIELDS,
    PLAN_EXCLUDED_FIELDS,
    PLAN_STREAM_FIELDS,
    _file_record,
    _integer,
    _literal,
    _ordinary_directory,
    _require,
    _selection_rows,
    _sha256_file,
    _verify_recursive_checksums,
    validate_materializer_policy,
)
from .deform360_joint_sparse_observability_v4 import (
    DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
    Deform360JointSparseObservabilityPolicyV4,
)


def _case_frame_range(value: object, *, name: str) -> tuple[int, int]:
    _require(
        not isinstance(value, (str, bytes))
        and isinstance(value, Sequence)
        and len(value) == 2,
        f"{name} changed",
    )
    values = cast(Sequence[object], value)
    start = _integer(values[0], name=f"{name} start")
    stop = _integer(values[1], name=f"{name} stop", minimum=1)
    _require(start < stop, f"{name} is empty")
    return start, stop


def _validate_plan_roster(
    *,
    plan: Mapping[str, Any],
    selected: Mapping[tuple[str, int], str],
) -> None:
    raw_cases = plan.get("cases")
    _require(
        isinstance(raw_cases, list) and len(raw_cases) == 10,
        "metric plan development cohort changed",
    )
    observed_selection: dict[tuple[str, int], str] = {}
    included_jobs: set[str] = set()
    included_streams: set[tuple[str, str]] = set()
    included_count = 0
    for case_index, raw_case in enumerate(cast(list[object], raw_cases)):
        _require(
            isinstance(raw_case, Mapping),
            f"metric plan case {case_index} changed",
        )
        case = cast(Mapping[str, Any], raw_case)
        require_exact_fields(
            case,
            expected=PLAN_CASE_FIELDS,
            name=f"metric plan case {case_index}",
        )
        object_id = _literal(case.get("object_id"), name="metric plan object_id")
        episode_id = _integer(
            case.get("episode_id"), name="metric plan episode_id"
        )
        identity_key = (object_id, episode_id)
        stratum = _literal(case.get("stratum"), name="metric plan stratum")
        _require(
            identity_key not in observed_selection,
            "metric plan repeats an object",
        )
        observed_selection[identity_key] = stratum
        declared_case_id = sha256_digest(case.get("case_id"), name="metric case_id")
        expected_case_id = content_id(
            {
                "schema": "bayesian-phystwin.deform360-prob4d-metric-case-id-v1",
                "object_id": object_id,
                "episode_id": episode_id,
            }
        )
        _require(declared_case_id == expected_case_id, "metric case ID changed")
        _case_frame_range(
            case.get("causal_frame_range_half_open"),
            name="metric plan causal frame range",
        )
        raw_streams = case.get("streams")
        _require(
            isinstance(raw_streams, list) and len(raw_streams) >= 2,
            "metric plan object lacks multiview support",
        )
        case_cameras: set[str] = set()
        for stream_index, raw_stream in enumerate(cast(list[object], raw_streams)):
            _require(
                isinstance(raw_stream, Mapping),
                f"metric plan stream {case_index}:{stream_index} changed",
            )
            stream = cast(Mapping[str, Any], raw_stream)
            require_exact_fields(
                stream,
                expected=PLAN_STREAM_FIELDS,
                name=f"metric plan stream {case_index}:{stream_index}",
            )
            job_id = sha256_digest(stream.get("job_id"), name="metric stream job_id")
            camera_id = _literal(
                stream.get("camera_id"), name="metric stream camera_id"
            )
            stream_key = (object_id, camera_id)
            _require(job_id not in included_jobs, "metric plan repeats a job")
            _require(
                stream_key not in included_streams,
                "metric plan repeats an object/camera stream",
            )
            _require(
                camera_id not in case_cameras,
                "metric plan repeats a camera within an object",
            )
            included_jobs.add(job_id)
            included_streams.add(stream_key)
            case_cameras.add(camera_id)
            _file_record(
                stream.get("prediction_manifest"),
                name="metric stream prediction_manifest",
            )
            _file_record(
                stream.get("metric_prefix"),
                name="metric stream metric_prefix",
            )
            _file_record(
                stream.get("metric_calibration"),
                name="metric stream metric_calibration",
            )
            included_count += 1
    _require(
        observed_selection == selected,
        "metric plan objects differ from the frozen development selection",
    )

    raw_excluded = plan.get("excluded_streams")
    _require(
        isinstance(raw_excluded, list) and len(raw_excluded) == 11,
        "metric plan excluded-stream accounting changed",
    )
    excluded_jobs: set[str] = set()
    excluded_streams: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(cast(list[object], raw_excluded)):
        _require(
            isinstance(raw_row, Mapping),
            f"metric excluded stream {index} changed",
        )
        row = cast(Mapping[str, Any], raw_row)
        require_exact_fields(
            row,
            expected=PLAN_EXCLUDED_FIELDS,
            name=f"metric excluded stream {index}",
        )
        job_id = sha256_digest(row.get("job_id"), name="excluded stream job_id")
        object_id = _literal(row.get("object_id"), name="excluded stream object_id")
        episode_id = _integer(
            row.get("episode_id"), name="excluded stream episode_id"
        )
        stratum = _literal(row.get("stratum"), name="excluded stream stratum")
        camera_id = _literal(row.get("camera_id"), name="excluded stream camera_id")
        reason = _literal(row.get("reason"), name="excluded stream reason")
        identity_key = (object_id, episode_id)
        stream_key = (object_id, camera_id)
        _require(
            selected.get(identity_key) == stratum,
            "excluded stream differs from the frozen development cohort",
        )
        _require(
            reason == "released-robot-geometry-outside-fixed-camera-prefix",
            "excluded stream reason changed",
        )
        _require(
            job_id not in included_jobs and job_id not in excluded_jobs,
            "metric plan repeats an excluded job",
        )
        _require(
            stream_key not in included_streams and stream_key not in excluded_streams,
            "metric plan repeats an excluded object/camera stream",
        )
        excluded_jobs.add(job_id)
        excluded_streams.add(stream_key)

    _require(included_count == 313, "metric plan supported-stream count changed")
    _require(
        len(excluded_jobs) == 11,
        "metric plan support-negative count changed",
    )
    _require(
        included_count + len(excluded_jobs) == 324,
        "metric plan admitted-stream accounting changed",
    )


def _validate_sources(
    *,
    metric_batch_root: Path,
    prediction_root: Path,
    production_result_path: Path,
    selection_path: Path,
    visual_provider_spec_path: Path,
    metric_policy_path: Path,
    camera_policy_path: Path,
    v4_policy_path: Path,
    materializer_policy_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Deform360JointSparseObservabilityPolicyV4,
    dict[str, Any],
]:
    metric_root = _ordinary_directory(metric_batch_root, name="metric batch root")
    predictions = _ordinary_directory(prediction_root, name="prediction root")
    _verify_recursive_checksums(metric_root)
    metric_result = cast(
        dict[str, Any],
        load_strict_json_object(
            metric_root / "metric-batch-result.json",
            label="metric batch result",
        ),
    )
    plan = cast(
        dict[str, Any],
        load_strict_json_object(
            metric_root / "metric-prefix-plan.json",
            label="metric prefix plan",
        ),
    )
    production = cast(
        dict[str, Any],
        load_strict_json_object(
            production_result_path,
            label="visual production result",
        ),
    )
    selection = cast(
        dict[str, Any],
        load_strict_json_object(selection_path, label="selection"),
    )
    provider = cast(
        dict[str, Any],
        load_strict_json_object(
            visual_provider_spec_path,
            label="visual provider specification",
        ),
    )
    metric_policy = cast(
        dict[str, Any],
        load_strict_json_object(metric_policy_path, label="metric policy"),
    )
    camera_policy = cast(
        dict[str, Any],
        load_strict_json_object(camera_policy_path, label="camera policy"),
    )
    v4_policy = Deform360JointSparseObservabilityPolicyV4.from_record(
        load_strict_json_object(v4_policy_path, label="v4 observability policy")
    )
    materializer_policy = validate_materializer_policy(
        load_strict_json_object(
            materializer_policy_path,
            label="v4 materializer policy",
        )
    )

    _require(
        metric_result.get("schema") == METRIC_BATCH_SCHEMA,
        "metric batch schema changed",
    )
    _require(
        metric_result.get("schema_version") == METRIC_BATCH_VERSION,
        "metric batch version changed",
    )
    _require(
        metric_result.get("semantics") == METRIC_BATCH_SEMANTICS,
        "metric batch semantics changed",
    )
    metric_identity = dict(metric_result)
    declared_metric_id = sha256_digest(
        metric_identity.pop("result_id"), name="metric batch result_id"
    )
    _require(
        content_id(metric_identity) == declared_metric_id,
        "metric batch result ID does not match its content",
    )
    _require(
        declared_metric_id == materializer_policy["metric_batch_result_id"],
        "metric batch result changed",
    )
    plan_record = _file_record(
        metric_result.get("plan_file"), name="metric batch plan_file"
    )
    _require(
        plan_record
        == {
            "path": "metric-prefix-plan.json",
            "sha256": _sha256_file(metric_batch_root / "metric-prefix-plan.json"),
            "byte_count": (metric_batch_root / "metric-prefix-plan.json")
            .stat()
            .st_size,
        },
        "metric batch plan file binding changed",
    )
    _require(
        metric_result.get("implementation_revision")
        == materializer_policy["metric_batch_implementation_revision"],
        "metric batch revision changed",
    )
    expected_metric_accounting = {
        "production_result_id": materializer_policy["production_result_id"],
        "object_count": 10,
        "admitted_stream_count": 324,
        "supported_stream_count": 313,
        "support_negative_stream_count": 11,
        "technical_failure_stream_count": 0,
        "supported_object_count": 10,
        "plan_emitted": True,
        "status": "target-free-visible-streams-supported",
    }
    _require(
        {key: metric_result.get(key) for key in expected_metric_accounting}
        == expected_metric_accounting,
        "metric batch accounting changed",
    )
    _require(
        plan.get("schema") == METRIC_PLAN_SCHEMA,
        "metric plan schema changed",
    )
    _require(
        plan.get("schema_version") == METRIC_PLAN_VERSION,
        "metric plan version changed",
    )
    _require(
        plan.get("semantics") == METRIC_PLAN_SEMANTICS,
        "metric plan semantics changed",
    )
    identity = dict(plan)
    declared_plan_id = sha256_digest(identity.pop("plan_id"), name="plan_id")
    _require(content_id(identity) == declared_plan_id, "metric plan ID changed")

    production_identity = dict(production)
    declared_production_id = sha256_digest(
        production_identity.pop("result_id"),
        name="visual production result_id",
    )
    _require(
        content_id(production_identity) == declared_production_id,
        "visual production result ID does not match its content",
    )
    _require(
        declared_production_id == materializer_policy["production_result_id"],
        "visual production result changed",
    )
    _require(
        production.get("visual_provider_lock_id")
        == materializer_policy["visual_provider_lock_id"],
        "visual provider lock changed",
    )
    _require(
        production.get("provider_revision") == materializer_policy["prob4d_revision"],
        "Prob4D production revision changed",
    )
    _require(
        production.get("motioncrafter_revision")
        == materializer_policy["motioncrafter_revision"],
        "MotionCrafter production revision changed",
    )
    _require(
        production.get("object_count") == 10
        and production.get("camera_view_count") == 324
        and production.get("completely_succeeded_object_count") == 10,
        "visual production accounting changed",
    )
    _require(
        production.get("succeeded_job_count") == 324
        and production.get("technical_failure_job_count") == 0,
        "visual production is incomplete",
    )
    _require(
        production.get("status") == "all-jobs-succeeded",
        "visual production status changed",
    )

    _require(
        selection.get("selection_artifact_sha256")
        == materializer_policy["selection_artifact_sha256"],
        "selection artifact changed",
    )
    selected = _selection_rows(selection)
    _require(len(selected) == 10, "development cohort size changed")
    _validate_plan_roster(plan=plan, selected=selected)
    _require(
        _sha256_file(selection_path) == plan.get("selection_file_sha256"),
        "selection bytes differ from plan",
    )
    _require(
        _sha256_file(visual_provider_spec_path)
        == plan.get("visual_provider_spec_file_sha256"),
        "provider specification bytes differ from plan",
    )
    _require(
        _sha256_file(metric_policy_path) == plan.get("metric_prior_policy_file_sha256"),
        "metric policy bytes differ from plan",
    )
    _require(
        _sha256_file(camera_policy_path)
        == plan.get("camera_eligibility_policy_file_sha256"),
        "camera policy bytes differ from plan",
    )
    _require(
        plan.get("prob4d_revision") == materializer_policy["prob4d_revision"],
        "plan Prob4D revision changed",
    )
    _require(
        plan.get("motioncrafter_revision")
        == materializer_policy["motioncrafter_revision"],
        "plan MotionCrafter revision changed",
    )
    _require(
        plan.get("visual_production_result_id")
        == materializer_policy["production_result_id"],
        "plan production result changed",
    )
    _require(
        provider.get("protocol_id") == selection.get("protocol_id"),
        "provider protocol differs",
    )
    _require(
        metric_policy.get("metric_source_kind")
        == materializer_policy["metric_source_kind"],
        "metric source kind changed",
    )
    _require(
        camera_policy.get("replacement_allowed") is False,
        "camera replacement became allowed",
    )
    _require(
        v4_policy.protocol_id == DEFORM360_JOINT_SPARSE_PROTOCOL_ID,
        "v4 policy protocol changed",
    )
    _require(
        float(materializer_policy["effective_samples_per_correlation_group"])
        == v4_policy.effective_samples_per_correlation_group,
        "materializer and v4 correlation-group caps differ",
    )
    _require(predictions == prediction_root.resolve(), "prediction root changed")
    return metric_result, plan, production, v4_policy, materializer_policy
