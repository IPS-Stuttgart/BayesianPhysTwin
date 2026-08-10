from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin import deform360_joint_sparse_camera_recovery_v5_2 as module
from bayesian_phystwin import deform360_joint_sparse_source_runner_v5 as source_module
from bayesian_phystwin import (
    deform360_joint_sparse_source_runner_v5_2 as source_module_v5_2,
)
from bayesian_phystwin import (
    deform360_joint_sparse_source_scoring_v5_2 as scoring_module_v5_2,
)
from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    evaluate_deform360_joint_sparse_source_gate_v5,
    load_deform360_joint_sparse_source_execution_lock_v5,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
AMENDMENT_PATH = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_camera_recovery_v5_2.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _support(
    camera_id: str,
    *,
    clusters: int,
    frames: int,
    points: int,
) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "artifact_id": camera_id.encode("utf-8").hex().ljust(64, "0")[:64],
        "manifest_file_sha256": "a" * 64,
        "metric_prefix_file_sha256": "b" * 64,
        "max_independent_cluster_count": clusters,
        "frames_with_minimum_clusters": frames,
        "total_projected_point_count": points,
        "eligible": clusters >= 8 and frames >= 1,
    }


def test_amendment_declares_public_data_and_no_human_approval() -> None:
    amendment = module.validate_deform360_joint_sparse_camera_recovery_amendment_v5_2(
        json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    )

    assert amendment["information_boundary"]["human_approval_used"] is False
    assert amendment["information_boundary"]["new_measurements_collected"] is False
    assert amendment["information_boundary"]["development_suffix_opened"] is False
    assert amendment["policy"]["minimum_passing_camera_count"] == 2
    assert amendment["policy"]["metric_gauge_threshold_relaxation_allowed"] is False
    assert amendment["policy"]["insufficient_support_action"] == (
        "exact-B0-physical-fallback"
    )


def test_metric_camera_ranking_is_deterministic_and_target_free() -> None:
    values = [
        _support("camera-d", clusters=7, frames=20, points=9000),
        _support("camera-c", clusters=12, frames=2, points=800),
        _support("camera-b", clusters=12, frames=2, points=800),
        _support("camera-a", clusters=10, frames=30, points=9000),
    ]

    ranked = module.rank_deform360_metric_camera_support_v5_2(values)

    assert ranked == ("camera-b", "camera-c", "camera-a")


def test_metric_camera_ranking_rejects_duplicate_camera() -> None:
    value = _support("camera-a", clusters=8, frames=1, points=100)

    with pytest.raises(ValueError, match="repeats a camera"):
        module.rank_deform360_metric_camera_support_v5_2([value, value])


def test_metric_support_summary_counts_independent_spatial_clusters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = np.zeros((3, 96, 160), dtype=np.bool_)
    for row in (1, 33, 65):
        for column in (1, 33, 65):
            valid[0, row, column] = True
    valid[1, :64, :64] = True
    np.savez(
        tmp_path / "metric-prefix.npz",
        frame_indices=np.arange(3, dtype=np.int64),
        points_world_m=np.zeros((3, 96, 160, 3), dtype=np.float64),
        valid_mask=valid,
    )
    (tmp_path / "metric-prefix.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "validate_deform360_robot_metric_prefix",
        lambda _path: {
            "camera_id": "camera-a",
            "artifact_id": "c" * 64,
            "projected_point_count": int(np.count_nonzero(valid)),
        },
    )

    support = module.summarize_deform360_metric_camera_support_v5_2(tmp_path)

    assert support["max_independent_cluster_count"] == 9
    assert support["frames_with_minimum_clusters"] == 1
    assert support["eligible"] is True


def test_camera_audit_preflight_and_recovery_plan_cover_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    input_root = tmp_path / "camera-inputs"
    input_root.mkdir()
    source_objects: list[dict[str, Any]] = []
    object_context: dict[str, dict[str, Any]] = {}
    for index, cohort_row in enumerate(lock["cohort"]["development_objects"]):
        object_id = cohort_row["object_id"]
        all_cameras = tuple(f"camera-{camera}" for camera in range(6))
        reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
        attempted = tuple(camera for camera in all_cameras if camera not in reserved)[
            :2
        ]
        windows = []
        for camera in attempted:
            decoded = input_root / object_id / camera / "decoded.npz"
            metric = input_root / object_id / camera / "metric.npz"
            decoded.parent.mkdir(parents=True, exist_ok=True)
            np.savez(decoded, value=np.asarray([index], dtype=np.int64))
            np.savez(metric, value=np.asarray([index + 1], dtype=np.int64))
            windows.append(
                {
                    "camera_id": camera,
                    "decoded_uniform": {
                        "path": decoded.relative_to(input_root).as_posix(),
                        "sha256": source_module._sha256_file(decoded),
                    },
                    "metric_prefix": {
                        "path": metric.relative_to(input_root).as_posix(),
                        "sha256": source_module._sha256_file(metric),
                    },
                }
            )
        source_objects.append(
            {
                "object_id": object_id,
                "raw_prefix_range_half_open": [0, 58],
                "visual_windows": windows,
            }
        )
        object_context[object_id] = {
            "episode_id": cohort_row["episode_id"],
            "stratum": cohort_row["stratum"],
            "all_cameras": all_cameras,
            "reserved": reserved,
            "attempted": attempted,
        }

    source_plan = {
        "plan_id": _sha("base-source-plan"),
        "implementation_revision": "1" * 40,
        "objects": source_objects,
    }
    monkeypatch.setattr(
        module,
        "validate_deform360_joint_sparse_source_prediction_plan_v5",
        lambda value, *, lock: value,
    )
    first_object = source_objects[0]["object_id"]
    first_camera = source_objects[0]["visual_windows"][0]["camera_id"]

    def prepare_window(**kwargs: Any) -> tuple[tuple[()], SimpleNamespace]:
        if (
            kwargs["camera_id"] == first_camera
            and str(kwargs["decoded_uniform_path"]).find(first_object) >= 0
        ):
            raise ValueError("metric gauge lacks eight independent causal clusters")
        gauge = SimpleNamespace(
            artifact_id=_sha(f"gauge-{kwargs['camera_id']}"),
            raw_frame_index=57,
            independent_cluster_count=12,
            inlier_independent_cluster_count=10,
            inlier_rmse_m=0.002,
        )
        return (), gauge

    monkeypatch.setattr(
        module, "prepare_deform360_joint_sparse_visual_window_v5", prepare_window
    )
    audit = module.audit_deform360_joint_sparse_source_cameras_v5_2(
        lock=lock,
        source_plan=source_plan,
        input_root=input_root,
    )
    assert audit["objects"][0]["passing_camera_ids"] == [
        source_objects[0]["visual_windows"][1]["camera_id"]
    ]
    assert all(len(row["passing_camera_ids"]) == 2 for row in audit["objects"][1:])

    model_set = {
        "schema": "prob4d.motioncrafter-model-set.v2",
        "model_type": "determ",
        "sources": {},
    }
    model_set_id = content_id(model_set)
    monkeypatch.setattr(module, "MOTIONCRAFTER_MODEL_SET_ID", model_set_id)
    base_objects: list[dict[str, Any]] = []
    inventory_objects: list[dict[str, Any]] = []
    for object_id, context in object_context.items():
        base_objects.append(
            {
                "object_id": object_id,
                "episode_id": context["episode_id"],
                "stratum": context["stratum"],
                "raw_prefix_range_half_open": [0, 58],
                "provider_range_half_open": [16, 58],
                "all_camera_ids": list(context["all_cameras"]),
                "reserved_endpoint_camera_ids": list(context["reserved"]),
            }
        )
        inventory_objects.append(
            {
                "object_id": object_id,
                "cameras": [
                    {
                        "camera": camera,
                        "video": {
                            "path": (
                                f"{object_id}/episode_0000/{camera}/undistorted.mp4"
                            ),
                            "sha256": _sha(f"video-{object_id}-{camera}"),
                            "byte_count": 100,
                        },
                    }
                    for camera in context["all_cameras"]
                ],
            }
        )
    inventory = {"inventory_id": _sha("inventory"), "objects": inventory_objects}
    base_provider_plan = {
        "manifest_sha256": _sha("base-provider-plan"),
        "prepared_source_inventory": {
            "inventory_id": inventory["inventory_id"],
            "file_sha256": _sha("inventory-file"),
        },
        "camera_roster_source": {
            "manifest_sha256": _sha("camera-roster"),
            "file_sha256": _sha("camera-roster-file"),
        },
        "provider_lock": {"provider_revision": module.PROB4D_REVISION},
        "motioncrafter": {
            "revision": module.MOTIONCRAFTER_REVISION,
            "model_set_id": model_set_id,
            "model_set_manifest": model_set,
        },
        "objects": base_objects,
    }
    monkeypatch.setattr(
        module,
        "validate_deform360_joint_sparse_motioncrafter_source_plan_v5",
        lambda value: value["manifest_sha256"],
    )
    metric_root = tmp_path / "metric-support"
    metric_root.mkdir()

    def summarize(path: Path) -> dict[str, object]:
        return _support(path.name, clusters=12, frames=4, points=1000)

    monkeypatch.setattr(
        module, "summarize_deform360_metric_camera_support_v5_2", summarize
    )
    preflight = module.build_deform360_joint_sparse_camera_recovery_preflight_v5_2(
        lock=lock,
        base_provider_plan=base_provider_plan,
        base_provider_plan_file_sha256=_sha("base-provider-file"),
        base_camera_audit=audit,
        base_camera_audit_file_sha256=_sha("base-audit-file"),
        metric_root=metric_root,
    )
    assert preflight["objects"][0]["recovery_required"] is True
    assert len(preflight["objects"][0]["selected_recovery_camera_ids"]) == 2
    assert all(not row["recovery_required"] for row in preflight["objects"][1:])

    monkeypatch.setattr(
        module, "validate_deform360_prepared_source_inventory", lambda value: value
    )
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    recovery_plan = (
        module.build_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
            lock=lock,
            execution_lock_file_sha256=_sha("lock-file"),
            inventory=inventory,
            base_provider_plan=base_provider_plan,
            base_provider_plan_file_sha256=_sha("base-provider-file"),
            recovery_preflight=preflight,
            recovery_preflight_file_sha256=_sha("preflight-file"),
            amendment=amendment,
            amendment_file_sha256=_sha("amendment-file"),
            implementation_revision="2" * 40,
            runner_source_sha256=_sha("runner"),
        )
    )
    assert recovery_plan["object_count"] == 1
    assert recovery_plan["job_count"] == 2
    assert {job["camera"] for job in recovery_plan["jobs"]} == set(
        preflight["objects"][0]["selected_recovery_camera_ids"]
    )


def _recovery_provider_plan(*, selected_count: int = 1) -> dict[str, object]:
    object_id = "001-public-object"
    all_cameras = tuple(f"camera-{index}" for index in range(6))
    reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
    available = tuple(camera for camera in all_cameras if camera not in reserved)
    attempted = available[:1]
    selected = available[1 : 1 + selected_count]
    object_row = {
        "object_id": object_id,
        "episode_id": 3,
        "stratum": "sheet",
        "raw_prefix_range_half_open": [10, 68],
        "provider_range_half_open": [26, 68],
        "all_camera_ids": list(all_cameras),
        "reserved_endpoint_camera_ids": list(reserved),
        "base_attempted_camera_ids": list(attempted),
        "base_passing_camera_ids": [attempted[0]],
        "selected_recovery_camera_ids": list(selected),
    }
    jobs = [
        module._recovery_job(
            object_id=object_id,
            episode_id=3,
            stratum="sheet",
            camera=camera,
            source_video={
                "path": f"{object_id}/episode_0000/{camera}/undistorted.mp4",
                "sha256": "d" * 64,
                "bytes": 100,
            },
            source_start=26,
            source_stop=68,
        )
        for camera in selected
    ]
    model_set = {
        "schema": "prob4d.motioncrafter-model-set.v2",
        "model_type": "determ",
        "sources": {},
    }
    original_model_id = module.MOTIONCRAFTER_MODEL_SET_ID
    module.MOTIONCRAFTER_MODEL_SET_ID = content_id(model_set)
    descriptor: dict[str, object] = {
        "schema": module.RECOVERY_PROVIDER_SCHEMA,
        "schema_version": module.RECOVERY_PROVIDER_VERSION,
        "semantics": module.RECOVERY_PROVIDER_SEMANTICS,
        "status": module.RECOVERY_PROVIDER_STATUS,
        "role": "development-source-prefix-camera-recovery",
        "implementation": {
            "revision": "1" * 40,
            "runner_source_sha256": "2" * 64,
        },
        "source_execution_lock": {
            "execution_lock_id": "3" * 64,
            "file_sha256": "4" * 64,
        },
        "prepared_source_inventory": {
            "inventory_id": "5" * 64,
            "file_sha256": "6" * 64,
        },
        "camera_roster_source": {
            "manifest_sha256": "7" * 64,
            "file_sha256": "8" * 64,
        },
        "base_provider_plan": {
            "manifest_sha256": "9" * 64,
            "file_sha256": "a" * 64,
        },
        "camera_recovery_preflight": {
            "preflight_id": "b" * 64,
            "file_sha256": "c" * 64,
        },
        "camera_recovery_amendment": {
            "amendment_id": "d" * 64,
            "file_sha256": "e" * 64,
        },
        "provider_lock": {"provider_revision": module.PROB4D_REVISION},
        "motioncrafter": {
            "revision": module.MOTIONCRAFTER_REVISION,
            "model_set_id": content_id(model_set),
            "model_set_manifest": model_set,
        },
        "run_configuration": dict(module.RUN_CONFIGURATION),
        "temporal_policy": dict(module.TEMPORAL_POLICY),
        "objects": [object_row],
        "object_count": 1,
        "jobs": jobs,
        "job_count": len(jobs),
        "smoke_job_id": jobs[0]["job_id"],
        "information_boundary": dict(module.RECOVERY_INFORMATION_BOUNDARY),
        "claim_boundary": module.RECOVERY_CLAIM_BOUNDARY,
    }
    plan = {"manifest_sha256": content_id(descriptor), **descriptor}
    module.MOTIONCRAFTER_MODEL_SET_ID = original_model_id
    return plan


def _provider_shard_report(
    plan: dict[str, object],
    *,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    jobs = [
        job
        for index, job in enumerate(plan["jobs"])
        if index % shard_count == shard_index
    ]
    completed = []
    for job in jobs:
        manifest = f"/provider/{job['job_id']}/predictions.json"
        completed.append(
            {
                "job_id": job["job_id"],
                "prediction_manifest": manifest,
                "prediction_manifest_sha256": "a" * 64,
                "verification": {
                    "hashes_verified": True,
                    "integrity_bound": True,
                    "manifest_path": manifest,
                    "member_count": len(job["windows"]),
                    "run_spec_sha256": "b" * 64,
                },
            }
        )
    descriptor: dict[str, object] = {
        "schema": module.PROVIDER_RUN_SCHEMA,
        "schema_version": module.PROVIDER_RUN_VERSION,
        "source_plan_sha256": plan["manifest_sha256"],
        "runtime_revision": plan["implementation"]["revision"],
        "mode": "complete" if shard_count == 1 else "shard",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "status": "complete",
        "requested_job_count": len(jobs),
        "completed_job_count": len(completed),
        "completed_jobs": completed,
        "information_boundary": dict(module.PROVIDER_RUN_INFORMATION_BOUNDARY),
        "claim_boundary": module.PROVIDER_RUN_CLAIM_BOUNDARY,
    }
    return {"run_sha256": content_id(descriptor), **descriptor}


def test_recovery_provider_rejects_one_camera_as_likelihood_policy_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _recovery_provider_plan()
    model_id = plan["motioncrafter"]["model_set_id"]
    monkeypatch.setattr(module, "MOTIONCRAFTER_MODEL_SET_ID", model_id)
    assert (
        module.validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(plan)
        == plan
    )

    changed = copy.deepcopy(plan)
    changed["jobs"][0]["likelihood_eligible"] = False
    job_descriptor = dict(changed["jobs"][0])
    job_descriptor.pop("job_id")
    changed["jobs"][0]["job_id"] = content_id(job_descriptor)
    changed["smoke_job_id"] = changed["jobs"][0]["job_id"]
    descriptor = dict(changed)
    descriptor.pop("manifest_sha256")
    changed["manifest_sha256"] = content_id(descriptor)

    with pytest.raises(ValueError, match="job identity changed"):
        module.validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(changed)


def test_recovery_provider_shards_merge_in_frozen_job_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _recovery_provider_plan(selected_count=2)
    monkeypatch.setattr(
        module, "MOTIONCRAFTER_MODEL_SET_ID", plan["motioncrafter"]["model_set_id"]
    )
    reports = [
        _provider_shard_report(plan, shard_index=index, shard_count=2)
        for index in range(2)
    ]

    merged = module.merge_deform360_joint_sparse_motioncrafter_recovery_runs_v5_2(
        plan=plan,
        shard_reports=reports,
    )

    assert merged["mode"] == "complete"
    assert merged["shard_count"] == 1
    assert [row["job_id"] for row in merged["completed_jobs"]] == [
        job["job_id"] for job in plan["jobs"]
    ]


def test_recovery_provider_merge_rejects_incomplete_shard_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _recovery_provider_plan(selected_count=2)
    monkeypatch.setattr(
        module, "MOTIONCRAFTER_MODEL_SET_ID", plan["motioncrafter"]["model_set_id"]
    )
    report = _provider_shard_report(plan, shard_index=0, shard_count=2)

    with pytest.raises(ValueError, match="shard roster is incomplete"):
        module.merge_deform360_joint_sparse_motioncrafter_recovery_runs_v5_2(
            plan=plan,
            shard_reports=[report],
        )


def test_combined_audit_plan_appends_only_bound_recovery_camera(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    objects, audit = _attempted_objects_and_audit()
    base_plan = source_module.build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision="1" * 40,
        objects=objects,
    )
    audit["base_source_plan_id"] = base_plan["plan_id"]
    audit_descriptor = {key: value for key, value in audit.items() if key != "audit_id"}
    audit["audit_id"] = content_id(audit_descriptor)

    base_object = base_plan["objects"][0]
    object_id = base_object["object_id"]
    attempted = tuple(window["camera_id"] for window in base_object["visual_windows"])
    selected = next(
        camera
        for camera in base_object["all_camera_ids"]
        if camera not in base_object["reserved_endpoint_camera_ids"]
        and camera not in attempted
    )
    recovery_object = {
        "object_id": object_id,
        "episode_id": base_object["episode_id"],
        "stratum": base_object["stratum"],
        "raw_prefix_range_half_open": base_object["raw_prefix_range_half_open"],
        "provider_range_half_open": [16, 58],
        "all_camera_ids": base_object["all_camera_ids"],
        "reserved_endpoint_camera_ids": base_object["reserved_endpoint_camera_ids"],
        "base_attempted_camera_ids": list(attempted),
        "base_passing_camera_ids": audit["objects"][0]["passing_camera_ids"],
        "selected_recovery_camera_ids": [selected],
    }
    job = module._recovery_job(
        object_id=object_id,
        episode_id=base_object["episode_id"],
        stratum=base_object["stratum"],
        camera=selected,
        source_video={
            "path": f"{object_id}/episode_0000/{selected}/undistorted.mp4",
            "sha256": "d" * 64,
            "bytes": 100,
        },
        source_start=16,
        source_stop=58,
    )
    model_set = {
        "schema": "prob4d.motioncrafter-model-set.v2",
        "model_type": "determ",
        "sources": {},
    }
    model_id = content_id(model_set)
    monkeypatch.setattr(module, "MOTIONCRAFTER_MODEL_SET_ID", model_id)
    provider_descriptor: dict[str, object] = {
        "schema": module.RECOVERY_PROVIDER_SCHEMA,
        "schema_version": module.RECOVERY_PROVIDER_VERSION,
        "semantics": module.RECOVERY_PROVIDER_SEMANTICS,
        "status": module.RECOVERY_PROVIDER_STATUS,
        "role": "development-source-prefix-camera-recovery",
        "implementation": {
            "revision": "2" * 40,
            "runner_source_sha256": "3" * 64,
        },
        "source_execution_lock": {
            "execution_lock_id": lock["execution_lock_id"],
            "file_sha256": "4" * 64,
        },
        "prepared_source_inventory": {
            "inventory_id": "5" * 64,
            "file_sha256": "6" * 64,
        },
        "camera_roster_source": {
            "manifest_sha256": "7" * 64,
            "file_sha256": "8" * 64,
        },
        "base_provider_plan": {
            "manifest_sha256": "9" * 64,
            "file_sha256": "a" * 64,
        },
        "camera_recovery_preflight": {
            "preflight_id": "b" * 64,
            "file_sha256": "c" * 64,
        },
        "camera_recovery_amendment": {
            "amendment_id": "d" * 64,
            "file_sha256": "e" * 64,
        },
        "provider_lock": {"provider_revision": module.PROB4D_REVISION},
        "motioncrafter": {
            "revision": module.MOTIONCRAFTER_REVISION,
            "model_set_id": model_id,
            "model_set_manifest": model_set,
        },
        "run_configuration": dict(module.RUN_CONFIGURATION),
        "temporal_policy": dict(module.TEMPORAL_POLICY),
        "objects": [recovery_object],
        "object_count": 1,
        "jobs": [job],
        "job_count": 1,
        "smoke_job_id": job["job_id"],
        "information_boundary": dict(module.RECOVERY_INFORMATION_BOUNDARY),
        "claim_boundary": module.RECOVERY_CLAIM_BOUNDARY,
    }
    provider_plan = {
        "manifest_sha256": content_id(provider_descriptor),
        **provider_descriptor,
    }
    provider_run = _provider_shard_report(provider_plan, shard_index=0, shard_count=1)

    input_root = tmp_path / "inputs"
    decoded_root = input_root / "recovery-decoded"
    metric_root = input_root / "recovery-metric"
    decoded_root.mkdir(parents=True)
    metric_root.mkdir(parents=True)
    decoded = decoded_root / object_id / f"{selected}.npz"
    decoded.parent.mkdir(parents=True)
    np.savez(decoded, point_map=np.zeros((42, 1, 1, 3), dtype=np.float16))
    decoded_report = {
        "schema_version": 1,
        "completed_at_utc": "2026-08-11T00:00:00+00:00",
        "method": "decoded uniform overlap fusion",
        "fixed_prob4d_vggt_blend": False,
        "manifest": provider_run["completed_jobs"][0]["prediction_manifest"],
        "manifest_sha256": provider_run["completed_jobs"][0][
            "prediction_manifest_sha256"
        ],
        "prob4d_root": "/provider/prob4d",
        "prob4d_revision": module.PROB4D_REVISION,
        "output_npz": str(decoded.resolve()),
        "output_npz_sha256": source_module._sha256_file(decoded),
        "frame_count": 42,
        "overlap_pixel_fraction": 0.5,
        "maximum_contributors": 2,
        "covariance_units": "m^2 after Prob4D gauge alignment",
    }
    decoded.with_suffix(".json").write_text(
        json.dumps(decoded_report, sort_keys=True) + "\n", encoding="utf-8"
    )
    metric_directory = metric_root / object_id / selected
    metric_directory.mkdir(parents=True)
    np.savez(metric_directory / "metric-prefix.npz", value=np.asarray([1]))
    monkeypatch.setattr(
        module,
        "validate_deform360_robot_metric_prefix",
        lambda _path: {"object_id": object_id, "camera_id": selected},
    )

    combined = module.build_deform360_joint_sparse_combined_camera_audit_plan_v5_2(
        lock=lock,
        base_source_plan=base_plan,
        base_camera_audit=audit,
        recovery_provider_plan=provider_plan,
        recovery_provider_run=provider_run,
        input_root=input_root,
        recovery_decoded_root=decoded_root,
        recovery_metric_root=metric_root,
        implementation_revision="f" * 40,
    )

    combined_object = next(
        row for row in combined["objects"] if row["object_id"] == object_id
    )
    assert [window["camera_id"] for window in combined_object["visual_windows"]] == (
        sorted([*attempted, selected])
    )
    assert combined_object["visual_windows"][-1]["decoded_uniform"]["sha256"] == (
        source_module._sha256_file(decoded)
    )


def test_base_lock_already_declares_no_human_or_new_measurement() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)

    assert lock["public_measurements"]["released_real_world_recordings"] is True
    assert lock["public_measurements"]["human_approval_required"] is False
    assert lock["public_measurements"]["new_measurements_required"] is False


def test_recovery_cli_has_no_manual_attempted_camera_payload() -> None:
    source = (
        ROOT / "scripts/science/materialize_deform360_joint_sparse_source_plan_v5_2.py"
    ).read_text(encoding="utf-8")
    recovery = (
        ROOT
        / "scripts/science/materialize_deform360_joint_sparse_camera_recovery_v5_2.py"
    ).read_text(encoding="utf-8")

    assert "--combined-camera-audit-plan" in source
    assert "--attempted-objects" not in source
    assert 'subparsers.add_parser("build-recovery-lineage")' in recovery
    assert 'subparsers.add_parser("merge-provider-runs")' in recovery


def _attempted_objects_and_audit() -> tuple[list[dict[str, object]], dict[str, object]]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    objects: list[dict[str, object]] = []
    audit_objects: list[dict[str, object]] = []
    for index, cohort_row in enumerate(lock["cohort"]["development_objects"]):
        object_id = cohort_row["object_id"]
        cameras = tuple(f"camera-{camera}" for camera in range(6))
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        attempted = tuple(camera for camera in cameras if camera not in reserved)[:2]
        passing = attempted[:1] if index == 0 else attempted
        results: list[dict[str, object]] = []
        for camera in attempted:
            passed = camera in passing
            results.append(
                {
                    "camera_id": camera,
                    "status": "passed" if passed else "rejected",
                    "failure_code": (
                        None
                        if passed
                        else "metric-gauge-lacks-eight-independent-causal-clusters"
                    ),
                    "decoded_uniform_sha256": "1" * 64,
                    "metric_prefix_sha256": "2" * 64,
                    "gauge_artifact_id": "3" * 64 if passed else None,
                    "raw_frame_index": 0 if passed else None,
                    "independent_cluster_count": 8 if passed else None,
                    "inlier_independent_cluster_count": 8 if passed else None,
                    "inlier_rmse_m": 0.001 if passed else None,
                }
            )
        failed = sorted(set(attempted) - set(passing))
        audit_objects.append(
            {
                "object_id": object_id,
                "attempted_camera_ids": list(attempted),
                "passing_camera_ids": list(passing),
                "failed_camera_ids": failed,
                "camera_results": results,
            }
        )
        objects.append(
            {
                "object_id": object_id,
                "episode_id": cohort_row["episode_id"],
                "stratum": cohort_row["stratum"],
                "raw_prefix_range_half_open": [0, 58],
                "all_camera_ids": list(cameras),
                "reserved_endpoint_camera_ids": list(reserved),
                "physical": {
                    "path": f"objects/{index}/physical.npz",
                    "sha256": f"{index + 10:064x}",
                    "physical_mode": "warp_twin",
                },
                "visual_windows": [
                    {
                        "camera_id": camera,
                        "decoded_uniform": {
                            "path": f"objects/{index}/{camera}/decoded.npz",
                            "sha256": "1" * 64,
                        },
                        "metric_prefix": {
                            "path": f"objects/{index}/{camera}/metric.npz",
                            "sha256": "2" * 64,
                        },
                    }
                    for camera in attempted
                ],
                "contact_prefix": {
                    "status": "unavailable",
                    "path": None,
                    "manifest_file_sha256": None,
                    "materialization_id": None,
                    "unavailable_reason": (
                        source_module.CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON
                    ),
                },
            }
        )
    audit_identity: dict[str, object] = {
        "schema": module.CAMERA_AUDIT_SCHEMA,
        "schema_version": module.CAMERA_AUDIT_VERSION,
        "semantics": module.CAMERA_AUDIT_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "base_source_plan_id": "4" * 64,
        "implementation_revision": "5" * 40,
        "objects": sorted(audit_objects, key=lambda row: str(row["object_id"])),
        "information_boundary": dict(module.AUDIT_INFORMATION_BOUNDARY),
    }
    return objects, {**audit_identity, "audit_id": content_id(audit_identity)}


def test_recovery_source_plan_uses_two_view_admission_and_exact_fallback() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    objects, audit = _attempted_objects_and_audit()
    artifact_names = source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
    lineage = {
        "artifact_ids": {name: "6" * 64 for name in artifact_names},
        "source_artifacts": {name: "7" * 64 for name in artifact_names},
        "policy": dict(module.RECOVERY_POLICY),
        "base_prediction_batch_preserved": True,
    }
    lineage["artifact_ids"]["final_camera_audit"] = audit["audit_id"]

    plan = source_module_v5_2.build_deform360_joint_sparse_source_prediction_plan_v5_2(
        lock=lock,
        implementation_revision="8" * 40,
        attempted_objects=objects,
        final_camera_audit=audit,
        camera_recovery=lineage,
    )

    assert plan["schema_version"] == 6
    assert plan["information_boundary"]["development_suffix_opened"] is False
    assert plan["information_boundary"]["human_approval_required"] is False
    assert plan["objects"][0]["camera_admission"]["admitted"] is False
    assert (
        plan["objects"][0]["camera_admission"]["exact_physical_fallback_required"]
        is True
    )
    assert plan["objects"][0]["visual_windows"] == []
    assert plan["objects"][1]["camera_admission"]["admitted"] is True
    assert len(plan["objects"][1]["visual_windows"]) == 2
    assert (
        source_module_v5_2.validate_deform360_joint_sparse_source_prediction_plan_v5_2(
            plan,
            lock=lock,
        )
        == plan
    )


def test_recovery_source_plan_rejects_audit_archive_digest_mismatch() -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    objects, audit = _attempted_objects_and_audit()
    lineage = {
        "artifact_ids": {
            name: "6" * 64 for name in source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
        },
        "source_artifacts": {
            name: "7" * 64 for name in source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
        },
        "policy": dict(module.RECOVERY_POLICY),
        "base_prediction_batch_preserved": True,
    }
    lineage["artifact_ids"]["final_camera_audit"] = audit["audit_id"]
    objects[0]["visual_windows"][0]["decoded_uniform"]["sha256"] = "9" * 64

    with pytest.raises(ValueError, match="does not bind the attempted provider"):
        source_module_v5_2.build_deform360_joint_sparse_source_prediction_plan_v5_2(
            lock=lock,
            implementation_revision="8" * 40,
            attempted_objects=objects,
            final_camera_audit=audit,
            camera_recovery=lineage,
        )


def _physical_archive(path: Path) -> None:
    frame_zero = np.zeros((128, 3), dtype=np.float32)
    trajectory = np.repeat(frame_zero[None], 76, axis=0)
    np.savez(
        path,
        prediction_m=trajectory,
        persistence_m=trajectory,
        driven_readout_m=trajectory,
        zero_action_readout_m=trajectory,
        action_support=np.zeros(128, dtype=np.float32),
        frame_zero_points_m=frame_zero,
    )


def _endpoint_archive(path: Path, *, raw_start: int) -> None:
    np.savez(
        path,
        frame_indices=np.arange(58, 76, dtype=np.int64),
        raw_frame_indices=np.arange(raw_start, raw_start + 18, dtype=np.int64),
        depth_m=np.ones((18, 8, 8), dtype=np.float32),
        object_mask=np.ones((18, 8, 8), dtype=np.bool_),
        intrinsics=np.asarray(
            [[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        camera_to_world=np.eye(4, dtype=np.float64),
    )


def test_recovery_publisher_seals_explicit_camera_fallbacks(
    tmp_path: Path,
) -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    objects, audit = _attempted_objects_and_audit()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    physical_path = input_root / "physical.npz"
    _physical_archive(physical_path)
    physical_sha256 = source_module._sha256_file(physical_path)
    for object_row, audit_row in zip(objects, audit["objects"], strict=True):
        object_row["physical"] = {
            "path": "physical.npz",
            "sha256": physical_sha256,
            "physical_mode": "warp_twin",
        }
        passing_camera = audit_row["attempted_camera_ids"][0]
        audit_row["passing_camera_ids"] = [passing_camera]
        audit_row["failed_camera_ids"] = [audit_row["attempted_camera_ids"][1]]
        for result in audit_row["camera_results"]:
            passed = result["camera_id"] == passing_camera
            result.update(
                {
                    "status": "passed" if passed else "rejected",
                    "failure_code": (
                        None
                        if passed
                        else "metric-gauge-lacks-eight-independent-causal-clusters"
                    ),
                    "gauge_artifact_id": "3" * 64 if passed else None,
                    "raw_frame_index": 0 if passed else None,
                    "independent_cluster_count": 8 if passed else None,
                    "inlier_independent_cluster_count": 8 if passed else None,
                    "inlier_rmse_m": 0.001 if passed else None,
                }
            )
    audit_identity = {key: value for key, value in audit.items() if key != "audit_id"}
    audit["audit_id"] = content_id(audit_identity)
    lineage = {
        "artifact_ids": {
            name: "6" * 64 for name in source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
        },
        "source_artifacts": {
            name: "7" * 64 for name in source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
        },
        "policy": dict(module.RECOVERY_POLICY),
        "base_prediction_batch_preserved": True,
    }
    lineage["artifact_ids"]["final_camera_audit"] = audit["audit_id"]
    plan = source_module_v5_2.build_deform360_joint_sparse_source_prediction_plan_v5_2(
        lock=lock,
        implementation_revision="8" * 40,
        attempted_objects=objects,
        final_camera_audit=audit,
        camera_recovery=lineage,
    )
    assert all(not row["visual_windows"] for row in plan["objects"])
    plan_path = tmp_path / "source-plan-v5-2.json"
    write_atomic_json(plan, plan_path, overwrite=False)

    output_root = tmp_path / "predictions-v5-2"
    receipt = (
        source_module_v5_2.publish_deform360_joint_sparse_source_prediction_panel_v5_2(
            execution_lock_path=LOCK_PATH,
            source_plan_path=plan_path,
            input_root=input_root,
            output_root=output_root,
        )
    )

    assert receipt["schema_version"] == 2
    assert receipt["prediction_record_count"] == 100
    assert receipt["information_boundary"] == source_module_v5_2.SOURCE_PLAN_BOUNDARY
    assert len(list((output_root / "source-seals").glob("*.json"))) == 100
    for seal_path in (output_root / "source-seals").glob("*.json"):
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal["technical_failure"] is True
        assert seal["factor_admitted"] is False

    batch = load_strict_json_object(
        output_root / "source-prediction-batch.json",
        label="source prediction batch",
    )
    endpoint_root = tmp_path / "endpoints"
    endpoint_root.mkdir()
    endpoint_objects = []
    for row in plan["objects"]:
        object_id = row["object_id"]
        raw_start = row["raw_prefix_range_half_open"][1]
        views = []
        for camera_id in row["reserved_endpoint_camera_ids"]:
            relative = Path(object_id) / f"{camera_id}.npz"
            path = endpoint_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _endpoint_archive(path, raw_start=raw_start)
            views.append(
                {
                    "camera_id": camera_id,
                    "endpoint_archive": {
                        "path": relative.as_posix(),
                        "sha256": source_module._sha256_file(path),
                    },
                }
            )
        endpoint_objects.append(
            {
                "object_id": object_id,
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "all_camera_ids": row["all_camera_ids"],
                "raw_endpoint_range_half_open": [raw_start, raw_start + 18],
                "reserved_views": views,
            }
        )
    endpoint_plan = (
        scoring_module_v5_2.build_deform360_joint_sparse_source_endpoint_plan_v5_2(
            lock=lock,
            source_prediction_plan=plan,
            prediction_batch=batch,
            source_prediction_receipt=receipt,
            objects=endpoint_objects,
        )
    )
    endpoint_plan_path = tmp_path / "endpoint-plan-v5-2.json"
    write_atomic_json(endpoint_plan, endpoint_plan_path, overwrite=False)
    scoring_root = tmp_path / "scores-v5-2"
    scoring_receipt = (
        scoring_module_v5_2.publish_deform360_joint_sparse_source_scores_v5_2(
            execution_lock_path=LOCK_PATH,
            source_prediction_plan_path=plan_path,
            source_prediction_root=output_root,
            endpoint_plan_path=endpoint_plan_path,
            endpoint_input_root=endpoint_root,
            output_root=scoring_root,
        )
    )
    assert scoring_receipt["schema_version"] == 2
    assert scoring_receipt["endpoint_report_count"] == 100
    evidence = load_strict_json_object(
        scoring_root / "source-evidence.json", label="source evidence"
    )
    gate = evaluate_deform360_joint_sparse_source_gate_v5(evidence, lock)
    assert gate["gate_passed"] is False
    assert gate["confirmation_access_authorized"] is False


def test_recovery_publisher_exercises_admitted_visual_and_contact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    objects, audit = _attempted_objects_and_audit()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    physical_path = input_root / "physical.npz"
    _physical_archive(physical_path)
    physical_sha256 = source_module._sha256_file(physical_path)
    decoded_path = input_root / "decoded.npz"
    metric_path = input_root / "metric.npz"
    np.savez(decoded_path, value=np.asarray([1], dtype=np.int64))
    np.savez(metric_path, value=np.asarray([2], dtype=np.int64))
    decoded_sha256 = source_module._sha256_file(decoded_path)
    metric_sha256 = source_module._sha256_file(metric_path)
    contact_object_id = objects[0]["object_id"]
    failed_provider_object_id = objects[1]["object_id"]
    contact_directory = input_root / "contacts" / contact_object_id
    contact_directory.mkdir(parents=True)
    contact_manifest = contact_directory / "contact-prefix.json"
    contact_manifest.write_text("{}\n", encoding="utf-8")
    materialization_id = _sha("contact-materialization")

    for object_row, audit_row in zip(objects, audit["objects"], strict=True):
        object_row["physical"] = {
            "path": "physical.npz",
            "sha256": physical_sha256,
            "physical_mode": "warp_twin",
        }
        for window in object_row["visual_windows"]:
            window["decoded_uniform"] = {
                "path": "decoded.npz",
                "sha256": decoded_sha256,
            }
            window["metric_prefix"] = {
                "path": "metric.npz",
                "sha256": metric_sha256,
            }
        audit_row["passing_camera_ids"] = list(audit_row["attempted_camera_ids"])
        audit_row["failed_camera_ids"] = []
        for result in audit_row["camera_results"]:
            result.update(
                {
                    "status": "passed",
                    "failure_code": None,
                    "decoded_uniform_sha256": decoded_sha256,
                    "metric_prefix_sha256": metric_sha256,
                    "gauge_artifact_id": _sha(
                        f"{object_row['object_id']}-{result['camera_id']}"
                    ),
                    "raw_frame_index": 57,
                    "independent_cluster_count": 12,
                    "inlier_independent_cluster_count": 10,
                    "inlier_rmse_m": 0.002,
                }
            )
    contact_record = {
        "status": source_module_v5_2.CONTACT_PREFIX_AVAILABLE,
        "path": contact_directory.relative_to(input_root).as_posix(),
        "manifest_file_sha256": source_module._sha256_file(contact_manifest),
        "materialization_id": materialization_id,
        "unavailable_reason": None,
    }
    monkeypatch.setattr(
        source_module_v5_2,
        "validate_deform360_public_contact_prefix",
        lambda _path: {"materialization_id": materialization_id},
    )
    assert (
        source_module_v5_2._verified_contact_directory(  # noqa: SLF001
            input_root, contact_record
        )
        == contact_directory.resolve()
    )
    audit_identity = {key: value for key, value in audit.items() if key != "audit_id"}
    audit["audit_id"] = content_id(audit_identity)
    lineage = {
        "artifact_ids": {
            name: _sha(f"artifact-{name}")
            for name in source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
        },
        "source_artifacts": {
            name: _sha(f"source-{name}")
            for name in source_module_v5_2.CAMERA_RECOVERY_ARTIFACT_NAMES
        },
        "policy": dict(module.RECOVERY_POLICY),
        "base_prediction_batch_preserved": True,
    }
    lineage["artifact_ids"]["final_camera_audit"] = audit["audit_id"]
    plan = source_module_v5_2.build_deform360_joint_sparse_source_prediction_plan_v5_2(
        lock=lock,
        implementation_revision="8" * 40,
        attempted_objects=objects,
        final_camera_audit=audit,
        camera_recovery=lineage,
    )
    assert all(row["camera_admission"]["admitted"] for row in plan["objects"])

    def prepare_visual(**kwargs: Any) -> object:
        sources = kwargs["source_artifact_ids"]
        if any(
            key.startswith(f"visual/{failed_provider_object_id}/") for key in sources
        ):
            raise ValueError("synthetic prefix provider failure")
        return SimpleNamespace(camera_id=kwargs["camera_id"]), SimpleNamespace()

    monkeypatch.setattr(
        source_module_v5_2,
        "prepare_deform360_joint_sparse_visual_window_v5",
        prepare_visual,
    )
    monkeypatch.setattr(
        source_module_v5_2,
        "estimate_deform360_last_causal_residual_v5",
        lambda **kwargs: np.zeros(
            kwargs["physical_prediction_m"].shape[1:], dtype=np.float64
        ),
    )

    def materialize(**kwargs: Any) -> SimpleNamespace:
        problem = source_module._technical_fallback_problem(
            object_id=kwargs["object_id"],
            episode_id=kwargs["episode_id"],
            stratum=kwargs["stratum"],
            physical_prediction_m=kwargs["physical_prediction_m"],
            persistence_m=kwargs["persistence_m"],
            physical_mode=kwargs["physical_mode"],
            implementation_revision=kwargs["implementation_revision"],
            source_artifact_ids=kwargs["source_artifact_ids"],
            failure_stage="test_materialization_proxy",
            failure=ValueError("deterministic test proxy"),
        )
        return SimpleNamespace(problem=problem)

    monkeypatch.setattr(
        source_module_v5_2,
        "materialize_deform360_joint_sparse_prediction_v5",
        materialize,
    )
    plan_path = tmp_path / "source-plan-v5-2.json"
    write_atomic_json(plan, plan_path, overwrite=False)
    output_root = tmp_path / "predictions-v5-2"
    first = (
        source_module_v5_2.publish_deform360_joint_sparse_source_prediction_panel_v5_2(
            execution_lock_path=LOCK_PATH,
            source_plan_path=plan_path,
            input_root=input_root,
            output_root=output_root,
        )
    )
    second = (
        source_module_v5_2.publish_deform360_joint_sparse_source_prediction_panel_v5_2(
            execution_lock_path=LOCK_PATH,
            source_plan_path=plan_path,
            input_root=input_root,
            output_root=output_root,
        )
    )
    assert first == second
    assert first["prediction_record_count"] == 100
