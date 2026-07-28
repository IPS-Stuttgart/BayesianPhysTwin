"""Versioned serialization facade for physics-guided query plans."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._phystwin_query_core import (
    PHYSICS_GUIDED_QUERY_PLAN_SCHEMA,
    PHYSICS_GUIDED_QUERY_PLAN_VERSION,
    PhysicsGuidedQueryConfigV2,
    PhysicsGuidedQueryStepV1,
    _validate_sha256,
    array_sha256,
    readonly,
    require,
)
from ._phystwin_query_plan import PhysicsGuidedQueryPlanV1


def save_physics_guided_query_plan_v1(
    path: str | Path,
    plan: PhysicsGuidedQueryPlanV1,
) -> None:
    """Write a non-pickled, content-addressed query-plan artifact."""

    if not isinstance(plan, PhysicsGuidedQueryPlanV1):
        raise TypeError("plan must be a PhysicsGuidedQueryPlanV1")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = plan.descriptor()
    descriptor["artifact_id"] = plan.artifact_id
    np.savez_compressed(
        target,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **plan.arrays(),
    )


def load_physics_guided_query_plan_v1(
    path: str | Path,
) -> PhysicsGuidedQueryPlanV1:
    """Load and fully revalidate a physics-guided query-plan artifact."""

    with np.load(path, allow_pickle=False) as archive:
        if "descriptor_json" not in archive:
            raise ValueError("query-plan artifact has no descriptor_json")
        descriptor = json.loads(str(archive["descriptor_json"]))
        if not isinstance(descriptor, dict):
            raise ValueError("query-plan descriptor must be a JSON object")
        if descriptor.get("schema_name") != PHYSICS_GUIDED_QUERY_PLAN_SCHEMA:
            raise ValueError("unsupported physics-guided query-plan schema")
        if int(descriptor.get("schema_version", -1)) != (
            PHYSICS_GUIDED_QUERY_PLAN_VERSION
        ):
            raise ValueError("unsupported physics-guided query-plan version")
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    required_arrays = {
        "node_ids",
        "seed_frames",
        "replaces_node_ids",
        "camera_mask",
        "seed_pixels_xy",
        "motion_score",
        "visibility_score",
        "mode_information_gain",
        "spatial_diversity_score",
        "contact_distance_score",
        "total_score",
    }
    missing = required_arrays - arrays.keys()
    extra = arrays.keys() - required_arrays
    if missing or extra:
        raise ValueError(
            "query-plan artifact arrays changed; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    input_sha256 = descriptor.get("input_sha256")
    expected_inputs = {
        "candidate_ids",
        "contact_position_m",
        "mode_basis",
        "nuisance_basis",
        "observation_precision",
        "physical_rollout_m",
        "predicted_support_probability",
        "projected_pixels_xy",
        "tracker_support_probability",
    }
    if not isinstance(input_sha256, dict) or set(input_sha256) != expected_inputs:
        raise ValueError("query-plan input digest inventory changed")
    config_values = descriptor.get("config")
    if not isinstance(config_values, dict):
        raise ValueError("query-plan artifact has no config")
    plan = PhysicsGuidedQueryPlanV1(
        **arrays,
        requested_active_queries=int(descriptor["requested_active_queries"]),
        minimum_camera_support=int(descriptor["minimum_camera_support"]),
        prefix_frame_count=int(descriptor["prefix_frame_count"]),
        config=PhysicsGuidedQueryConfigV2(**config_values),
        source_revision=str(descriptor["source_revision"]),
        support_model_id=str(descriptor["support_model_id"]),
        physical_rollout_sha256=str(input_sha256["physical_rollout_m"]),
        projected_pixels_sha256=str(input_sha256["projected_pixels_xy"]),
        predicted_support_sha256=str(
            input_sha256["predicted_support_probability"]
        ),
        mode_basis_sha256=str(input_sha256["mode_basis"]),
        nuisance_basis_sha256=str(input_sha256["nuisance_basis"]),
        observation_precision_sha256=str(input_sha256["observation_precision"]),
        candidate_ids_sha256=str(input_sha256["candidate_ids"]),
        contact_position_sha256=str(input_sha256["contact_position_m"]),
        tracker_support_sha256=str(input_sha256["tracker_support_probability"]),
    )
    expected = str(descriptor.get("artifact_id", ""))
    _validate_sha256(expected, name="artifact_id")
    if plan.artifact_id != expected:
        raise ValueError("query-plan artifact digest does not match its payload")
    return plan


__all__ = [
    "PHYSICS_GUIDED_QUERY_PLAN_SCHEMA",
    "PHYSICS_GUIDED_QUERY_PLAN_VERSION",
    "PhysicsGuidedQueryConfigV2",
    "PhysicsGuidedQueryPlanV1",
    "PhysicsGuidedQueryStepV1",
    "array_sha256",
    "load_physics_guided_query_plan_v1",
    "readonly",
    "require",
    "save_physics_guided_query_plan_v1",
]
