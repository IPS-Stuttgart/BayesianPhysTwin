from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin import deform360_joint_sparse_motioncrafter_source_v5 as module
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixtures(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], ...]:
    lock_id = _sha("lock")
    inventory_id = _sha("inventory")
    inventory_file_sha = _sha("inventory-file")
    model_set = {
        "schema": "prob4d.motioncrafter-model-set.v2",
        "model_type": "determ",
        "sources": {
            name: {
                "kind": "huggingface_revision",
                "repository": f"example/{name}",
                "revision": "1" * 40,
            }
            for name in ("unet", "vae", "image_vae", "base_pipeline")
        },
    }
    model_set_id = content_id(model_set)
    configuration = {
        **module.RUN_CONFIGURATION,
        "model_source_set_sha256": model_set_id,
    }
    monkeypatch.setattr(module, "V5_EXECUTION_LOCK_ID", lock_id)
    monkeypatch.setattr(module, "V5_PREPARED_INVENTORY_ID", inventory_id)
    monkeypatch.setattr(module, "V5_PREPARED_INVENTORY_FILE_SHA256", inventory_file_sha)
    monkeypatch.setattr(module, "MOTIONCRAFTER_MODEL_SET_ID", model_set_id)
    monkeypatch.setattr(module, "RUN_CONFIGURATION", configuration)

    cohort: list[dict[str, Any]] = []
    inventory_objects: list[dict[str, Any]] = []
    legacy_jobs: list[dict[str, Any]] = []
    for index in range(10):
        object_id = f"{index:03d}-public-object"
        episode_id = index
        stratum = "sheet" if index < 5 else "volumetric"
        cohort.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
            }
        )
        all_cameras = tuple(f"camera-{camera}" for camera in range(6))
        reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
        provider_cameras = tuple(
            camera for camera in all_cameras if camera not in reserved
        )[:3]
        prefix_start = 20 + index
        prefix_stop = prefix_start + module.PREFIX_FRAME_COUNT
        inventory_objects.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "action_window": {
                    "prefix_raw_frame_range_half_open": [
                        prefix_start,
                        prefix_stop,
                    ]
                },
                "cameras": [
                    {
                        "camera": camera,
                        "video": {
                            "path": (
                                f"{object_id}/episode_0000/{camera}/undistorted.mp4"
                            ),
                            "sha256": _sha(f"{object_id}-{camera}"),
                            "byte_count": 1000 + index,
                        },
                    }
                    for camera in all_cameras
                ],
            }
        )
        for camera in provider_cameras:
            descriptor = {
                "object_id": object_id,
                "episode": "episode_0000",
                "camera": camera,
            }
            legacy_jobs.append({"job_id": content_id(descriptor), **descriptor})

    lock = {
        "execution_lock_id": lock_id,
        "cohort": {"development_objects": cohort},
    }
    inventory = {
        "inventory_id": inventory_id,
        "information_boundary": {
            "calibration_target_metrics_computed": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
        "objects": inventory_objects,
    }
    legacy_descriptor = {
        "schema": "bayesian-phystwin.deform360-official-hub-motioncrafter-jobs",
        "schema_version": 1,
        "status": "locked-pre-provider-inference",
        "provider_lock": {"provider_revision": module.PROB4D_REVISION},
        "motioncrafter": {
            "revision": module.MOTIONCRAFTER_REVISION,
            "model_set_id": model_set_id,
            "model_set_manifest": model_set,
        },
        "run_configuration": configuration,
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used_for_prediction": False,
        },
        "jobs": legacy_jobs,
    }
    legacy = {
        "manifest_sha256": content_id(legacy_descriptor),
        **legacy_descriptor,
    }
    legacy_file_sha = _sha("legacy-file")
    monkeypatch.setattr(
        module, "LEGACY_CAMERA_ROSTER_MANIFEST_ID", legacy["manifest_sha256"]
    )
    monkeypatch.setattr(module, "LEGACY_CAMERA_ROSTER_FILE_SHA256", legacy_file_sha)
    return lock, inventory, legacy, inventory_file_sha, legacy_file_sha


def _plan(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    lock, inventory, legacy, inventory_sha, legacy_sha = _fixtures(monkeypatch)
    return module.build_deform360_joint_sparse_motioncrafter_source_plan_v5(
        lock=lock,
        execution_lock_file_sha256=_sha("lock-file"),
        inventory=inventory,
        inventory_file_sha256=inventory_sha,
        legacy_job_manifest=legacy,
        legacy_job_manifest_file_sha256=legacy_sha,
        implementation_revision="a" * 40,
        runner_source_sha256=_sha("runner"),
    )


def test_source_plan_uses_latest_42_frames_of_each_locked_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch)

    assert (
        module.validate_deform360_joint_sparse_motioncrafter_source_plan_v5(plan)
        == plan["manifest_sha256"]
    )
    assert len(plan["objects"]) == 10
    assert len(plan["jobs"]) == 30
    for item in plan["objects"]:
        prefix_start, prefix_stop = item["raw_prefix_range_half_open"]
        assert prefix_stop - prefix_start == 58
        assert item["provider_range_half_open"] == [prefix_stop - 42, prefix_stop]
        assert len(item["likelihood_camera_ids"]) >= 2
        assert not set(item["likelihood_camera_ids"]) & set(
            item["reserved_endpoint_camera_ids"]
        )
    for job in plan["jobs"]:
        start = job["source_frame_start"]
        stop = job["source_frame_stop_exclusive"]
        assert stop - start == 42
        assert job["windows"] == [
            {
                "window_id": "window_0000",
                "source_frame_start": start,
                "source_frame_stop_exclusive": start + 25,
            },
            {
                "window_id": "window_0001",
                "source_frame_start": start + 17,
                "source_frame_stop_exclusive": stop,
            },
        ]


def test_source_plan_rejects_noncausal_temporal_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch)
    changed = copy.deepcopy(plan)
    changed["objects"][0]["provider_range_half_open"][0] -= 1
    descriptor = dict(changed)
    descriptor.pop("manifest_sha256")
    changed["manifest_sha256"] = content_id(descriptor)

    with pytest.raises(ValueError, match="latest causal prefix"):
        module.validate_deform360_joint_sparse_motioncrafter_source_plan_v5(changed)


def test_source_plan_rejects_reserved_camera_in_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch)
    changed = copy.deepcopy(plan)
    first = changed["objects"][0]
    first["likelihood_camera_ids"][0] = first["reserved_endpoint_camera_ids"][0]
    descriptor = dict(changed)
    descriptor.pop("manifest_sha256")
    changed["manifest_sha256"] = content_id(descriptor)

    with pytest.raises(ValueError, match="camera policy changed"):
        module.validate_deform360_joint_sparse_motioncrafter_source_plan_v5(changed)


def test_source_plan_keeps_all_outcome_boundaries_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch)

    assert plan["camera_roster_source"]["legacy_frame_ranges_rejected"] is True
    assert plan["camera_roster_source"]["legacy_provider_outputs_rejected"] is True
    assert plan["information_boundary"] == {
        "public_source_prefix_payloads_authorized": True,
        "provider_outputs_opened": False,
        "development_suffix_opened": False,
        "future_object_observations_used": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "human_approval_required": False,
        "new_measurements_required": False,
    }


def test_source_plan_rejects_unbound_legacy_roster_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, inventory, legacy, inventory_sha, _ = _fixtures(monkeypatch)

    with pytest.raises(ValueError, match="camera-roster file changed"):
        module.build_deform360_joint_sparse_motioncrafter_source_plan_v5(
            lock=lock,
            execution_lock_file_sha256=_sha("lock-file"),
            inventory=inventory,
            inventory_file_sha256=inventory_sha,
            legacy_job_manifest=legacy,
            legacy_job_manifest_file_sha256=_sha("wrong-legacy-file"),
            implementation_revision="a" * 40,
            runner_source_sha256=_sha("runner"),
        )


def test_remote_runner_exposes_source_custody_arguments() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts/remote/run_deform360_joint_sparse_motioncrafter_source_v5.py"
    ).read_text(encoding="utf-8")

    assert "--source-execution-lock" in source
    assert "--prepared-source-inventory" in source
    assert "--camera-roster-manifest" in source
    assert "--base-provider-plan" in source
    assert "--camera-recovery-preflight" in source
    assert "--camera-recovery-amendment" in source
    assert "--shard-index" in source
    assert "development_suffix_opened" in source
