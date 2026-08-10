from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    score_deform360_joint_sparse_endpoint_v5,
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_prediction_v5 import RAW_METHOD_IDS
from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    build_deform360_joint_sparse_source_prediction_batch_v5,
    build_deform360_joint_sparse_source_prediction_seal_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    SOURCE_PANEL_RECEIPT_SCHEMA,
    SOURCE_PANEL_RECEIPT_VERSION,
    SOURCE_PLAN_BOUNDARY,
    build_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_receipt_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_scoring_v5 import (
    SOURCE_ENDPOINT_BOUNDARY,
    _technical_endpoint_views,
    build_deform360_joint_sparse_source_endpoint_plan_v5,
    publish_deform360_joint_sparse_source_scores_v5,
    validate_deform360_joint_sparse_source_endpoint_plan_v5,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
REVISION = "1" * 40


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_plan(lock: Mapping[str, Any]) -> dict[str, Any]:
    rows = cast(
        Sequence[Mapping[str, Any]],
        cast(Mapping[str, Any], lock["cohort"])["development_objects"],
    )
    objects = []
    for index, row in enumerate(rows):
        object_id = cast(str, row["object_id"])
        cameras = tuple(f"camera-{camera}" for camera in range(6))
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        likelihood = tuple(camera for camera in cameras if camera not in reserved)[:2]
        objects.append(
            {
                "object_id": object_id,
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "raw_prefix_range_half_open": [100 + index, 158 + index],
                "all_camera_ids": list(cameras),
                "reserved_endpoint_camera_ids": list(reserved),
                "physical": {
                    "path": f"objects/{index:02d}/physical.npz",
                    "sha256": _digest(f"physical-{index}"),
                    "physical_mode": "warp_twin",
                },
                "visual_windows": [
                    {
                        "camera_id": camera,
                        "decoded_uniform": {
                            "path": f"objects/{index:02d}/{camera}/decoded.npz",
                            "sha256": _digest(f"decoded-{index}-{camera}"),
                        },
                        "metric_prefix": {
                            "path": f"objects/{index:02d}/{camera}/metric.npz",
                            "sha256": _digest(f"metric-{index}-{camera}"),
                        },
                    }
                    for camera in likelihood
                ],
                "contact_prefix": {
                    "path": f"objects/{index:02d}/contact",
                    "manifest_file_sha256": _digest(f"contact-{index}"),
                    "materialization_id": _digest(f"materialization-{index}"),
                },
            }
        )
    return build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=REVISION,
        objects=objects,
    )


def _prediction_batch(lock: Mapping[str, Any]) -> dict[str, Any]:
    rows = cast(
        Sequence[Mapping[str, Any]],
        cast(Mapping[str, Any], lock["cohort"])["development_objects"],
    )
    object_ids = tuple(sorted(cast(str, row["object_id"]) for row in rows))
    seals = []
    for outer_id in object_ids:
        for object_id in object_ids:
            held_out = outer_id == object_id
            excluded = {outer_id} if held_out else {outer_id, object_id}
            methods = {
                method_id: {
                    "artifact_id": _digest(
                        f"{object_id}-{method_id}"
                        if method_id
                        in {"B0_physical_fallback", "B1_last_causal_residual"}
                        else f"{outer_id}-{object_id}-{method_id}"
                    ),
                    "predicted_loss_mm": 1.0,
                }
                for method_id in RAW_METHOD_IDS
            }
            seals.append(
                build_deform360_joint_sparse_source_prediction_seal_v5(
                    lock=lock,
                    implementation_revision=REVISION,
                    outer_held_out_object_id=outer_id,
                    record_role="held_out" if held_out else "training",
                    object_id=object_id,
                    factor_admitted=False,
                    technical_failure=False,
                    physical_mode="warp_twin",
                    risk_score=1.0,
                    prediction_fit_artifact_id=_digest(f"fit-{outer_id}-{object_id}"),
                    prediction_fit_object_ids=tuple(
                        sorted(set(object_ids) - excluded)
                    ),
                    methods=methods,
                    source_artifacts={
                        f"sources/{outer_id}/{object_id}.json": _digest(
                            f"source-{outer_id}-{object_id}"
                        )
                    },
                )
            )
    return build_deform360_joint_sparse_source_prediction_batch_v5(seals, lock)


def _receipt(
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": SOURCE_PANEL_RECEIPT_SCHEMA,
        "schema_version": SOURCE_PANEL_RECEIPT_VERSION,
        "execution_lock_id": lock["execution_lock_id"],
        "implementation_revision": REVISION,
        "plan_id": plan["plan_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "prediction_batch_file_sha256": "2" * 64,
        "prediction_record_count": 100,
        "source_prediction_seal_file_sha256": {
            f"{outer:02d}-{target:02d}.json": _digest(f"seal-{outer}-{target}")
            for outer in range(10)
            for target in range(10)
        },
        "information_boundary": dict(SOURCE_PLAN_BOUNDARY),
    }
    return {**identity, "receipt_id": content_id(identity)}


def _endpoint_objects(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in cast(Sequence[Mapping[str, Any]], plan["objects"]):
        result.append(
            {
                "object_id": row["object_id"],
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "all_camera_ids": row["all_camera_ids"],
                "raw_endpoint_range_half_open": [
                    row["raw_prefix_range_half_open"][1],
                    row["raw_prefix_range_half_open"][1] + 18,
                ],
                "reserved_views": [
                    {
                        "camera_id": camera_id,
                        "endpoint_archive": {
                            "path": f"endpoint/{row['object_id']}/{camera_id}.npz",
                            "sha256": _digest(
                                f"endpoint-{row['object_id']}-{camera_id}"
                            ),
                        },
                    }
                    for camera_id in row["reserved_endpoint_camera_ids"]
                ],
            }
        )
    return result


def _fixture() -> tuple[
    Mapping[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    plan = _source_plan(lock)
    batch = _prediction_batch(lock)
    receipt = _receipt(lock, plan, batch)
    return lock, plan, batch, receipt


def test_receipt_and_endpoint_plan_bind_the_complete_prediction_batch() -> None:
    lock, source_plan, batch, receipt = _fixture()
    assert (
        validate_deform360_joint_sparse_source_prediction_receipt_v5(
            receipt,
            lock=lock,
            plan=source_plan,
            prediction_batch=batch,
            prediction_batch_file_sha256="2" * 64,
        )
        == receipt
    )
    endpoint_plan = build_deform360_joint_sparse_source_endpoint_plan_v5(
        lock=lock,
        source_prediction_plan=source_plan,
        prediction_batch=batch,
        source_prediction_receipt=receipt,
        objects=_endpoint_objects(source_plan),
    )
    assert endpoint_plan["information_boundary"] == SOURCE_ENDPOINT_BOUNDARY
    assert endpoint_plan["prediction_batch_id"] == batch["prediction_batch_id"]
    assert (
        validate_deform360_joint_sparse_source_endpoint_plan_v5(
            endpoint_plan,
            lock=lock,
            source_prediction_plan=source_plan,
            prediction_batch=batch,
            source_prediction_receipt=receipt,
        )
        == endpoint_plan
    )


def test_endpoint_plan_rejects_a_likelihood_camera_as_endpoint() -> None:
    lock, source_plan, batch, receipt = _fixture()
    objects = _endpoint_objects(source_plan)
    first_source = cast(Sequence[Mapping[str, Any]], source_plan["objects"])[0]
    likelihood_camera = cast(Sequence[Mapping[str, Any]], first_source["visual_windows"])[
        0
    ]["camera_id"]
    objects[0]["reserved_views"][0]["camera_id"] = likelihood_camera
    with pytest.raises(ValueError, match="reserved-view roster"):
        build_deform360_joint_sparse_source_endpoint_plan_v5(
            lock=lock,
            source_prediction_plan=source_plan,
            prediction_batch=batch,
            source_prediction_receipt=receipt,
            objects=objects,
        )


def test_malformed_endpoint_becomes_retained_fixed_penalty() -> None:
    cameras = select_reserved_endpoint_views_v5(
        "026-sock-cloth", tuple(f"camera-{index}" for index in range(6)), count=2
    )
    views = _technical_endpoint_views(
        object_id="026-sock-cloth",
        episode_id=7,
        camera_ids=cameras,
        source_artifacts={"endpoint/source.npz": "3" * 64},
        failure=ValueError("private endpoint detail"),
    )
    trajectory = np.zeros((76, 128, 3), dtype=np.float32)
    report = score_deform360_joint_sparse_endpoint_v5(
        object_id="026-sock-cloth",
        episode_id=7,
        stratum="sheet",
        prediction_seal_id="4" * 64,
        trajectories_m={method_id: trajectory for method_id in RAW_METHOD_IDS},
        reserved_views=views,
        all_camera_ids=tuple(f"camera-{index}" for index in range(6)),
        evaluation_role="development_source",
    )
    assert report["technical_failure"] is True
    assert set(report["method_loss_mm"].values()) == {1000.0}
    assert "private endpoint detail" not in str(report)


def test_source_scorer_exposes_no_confirmation_authorization() -> None:
    parameters = inspect.signature(
        publish_deform360_joint_sparse_source_scores_v5
    ).parameters
    assert set(parameters) == {
        "endpoint_input_root",
        "endpoint_plan_path",
        "execution_lock_path",
        "output_root",
        "source_prediction_plan_path",
        "source_prediction_root",
    }
    assert not any("confirmation" in name or "target" in name for name in parameters)
