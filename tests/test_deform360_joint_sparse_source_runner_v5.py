from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import bayesian_phystwin.deform360_joint_sparse_source_runner_v5 as runner
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_prediction_v5 import (
    RAW_METHOD_IDS,
    run_deform360_joint_sparse_prediction_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
RUNNER_SCRIPT = (
    ROOT / "scripts/science/run_deform360_joint_sparse_source_predictions_v5.py"
)


def _load_script(path: Path, *, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _lock():
    return load_deform360_joint_sparse_source_execution_lock_v5(LOCK)


def _objects():
    lock = _lock()
    values = []
    for index, row in enumerate(lock["cohort"]["development_objects"]):
        object_id = row["object_id"]
        cameras = tuple(f"camera-{camera}" for camera in range(6))
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        likelihood = tuple(camera for camera in cameras if camera not in reserved)[:2]
        values.append(
            {
                "object_id": object_id,
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "raw_prefix_range_half_open": [100 + index, 158 + index],
                "all_camera_ids": list(cameras),
                "reserved_endpoint_camera_ids": list(reserved),
                "physical": {
                    "path": f"objects/{index:02d}/physical.npz",
                    "sha256": f"{index + 1:064x}",
                    "physical_mode": "warp_twin",
                },
                "visual_windows": [
                    {
                        "camera_id": camera,
                        "decoded_uniform": {
                            "path": f"objects/{index:02d}/{camera}/decoded.npz",
                            "sha256": f"{index + camera_index + 20:064x}",
                        },
                        "metric_prefix": {
                            "path": f"objects/{index:02d}/{camera}/metric.npz",
                            "sha256": f"{index + camera_index + 40:064x}",
                        },
                    }
                    for camera_index, camera in enumerate(likelihood)
                ],
                "contact_prefix": {
                    "status": "unavailable",
                    "path": None,
                    "manifest_file_sha256": None,
                    "materialization_id": None,
                    "unavailable_reason": (
                        runner.CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON
                    ),
                },
            }
        )
    return values


def test_source_plan_is_prefix_only_and_reserves_endpoint_by_identity() -> None:
    lock = _lock()
    plan = runner.build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision="1" * 40,
        objects=_objects(),
    )

    assert len(plan["objects"]) == 10
    assert plan["information_boundary"]["development_suffix_opened"] is False
    assert plan["information_boundary"]["confirmation_payloads_opened"] is False
    assert plan["information_boundary"]["human_approval_required"] is False
    assert plan["information_boundary"]["new_measurements_required"] is False
    assert plan["information_boundary"]["prob4d_used"] is True
    assert (
        plan["information_boundary"]["tactile_without_registered_axis_identity"]
        == "exact-no-contact-fallback"
    )
    for row in plan["objects"]:
        reserved = set(row["reserved_endpoint_camera_ids"])
        likelihood = {window["camera_id"] for window in row["visual_windows"]}
        assert likelihood.isdisjoint(reserved)
    assert (
        runner.validate_deform360_joint_sparse_source_prediction_plan_v5(
            plan,
            lock=lock,
        )
        == plan
    )


def test_source_plan_accepts_only_the_locked_unavailable_contact_reason() -> None:
    values = _objects()
    plan = runner.build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=_lock(),
        implementation_revision="1" * 40,
        objects=values,
    )
    assert plan["objects"][0]["contact_prefix"]["status"] == "unavailable"

    values[0]["contact_prefix"]["unavailable_reason"] = "operator-waived"
    with pytest.raises(ValueError, match="unavailability reason"):
        runner.build_deform360_joint_sparse_source_prediction_plan_v5(
            lock=_lock(),
            implementation_revision="1" * 40,
            objects=values,
        )


def test_source_plan_rejects_an_invented_tactile_axis_identity() -> None:
    values = _objects()
    values[0]["contact_prefix"] = {
        "status": "available",
        "path": "objects/00/contact-prefix",
        "manifest_file_sha256": "6" * 64,
        "materialization_id": "8" * 64,
        "unavailable_reason": None,
    }
    with pytest.raises(ValueError, match="unregistered tactile-to-robot axis"):
        runner.build_deform360_joint_sparse_source_prediction_plan_v5(
            lock=_lock(),
            implementation_revision="1" * 40,
            objects=values,
        )


def test_source_plan_rejects_reserved_endpoint_camera_in_likelihood() -> None:
    values = _objects()
    values[0]["visual_windows"][0]["camera_id"] = values[0][
        "reserved_endpoint_camera_ids"
    ][0]

    with pytest.raises(ValueError, match="reserved endpoint camera"):
        runner.build_deform360_joint_sparse_source_prediction_plan_v5(
            lock=_lock(),
            implementation_revision="1" * 40,
            objects=values,
        )


def test_prediction_runner_has_no_suffix_or_endpoint_argument() -> None:
    parameters = inspect.signature(
        runner.publish_deform360_joint_sparse_source_prediction_panel_v5
    ).parameters
    assert set(parameters) == {
        "execution_lock_path",
        "source_plan_path",
        "input_root",
        "output_root",
    }
    assert not any("suffix" in name or "endpoint" in name for name in parameters)

    cli = _load_script(RUNNER_SCRIPT, name="joint_sparse_source_runner_v5_cli")
    destinations = {action.dest for action in cli._parser()._actions}
    assert destinations == {
        "execution_lock",
        "help",
        "input_root",
        "output_root",
        "source_plan",
    }
    assert not any("suffix" in name or "endpoint" in name for name in destinations)


def test_physical_loader_preserves_exact_persistence_fallback(tmp_path: Path) -> None:
    points = np.zeros((128, 3), dtype=np.float32)
    persistence = np.repeat(points[None], 76, axis=0)
    path = tmp_path / "physical.npz"
    np.savez(
        path,
        prediction_m=persistence.copy(),
        persistence_m=persistence.copy(),
        driven_readout_m=persistence.copy(),
        zero_action_readout_m=persistence.copy(),
        action_support=np.zeros(128, dtype=np.float32),
        frame_zero_points_m=points,
    )

    prediction, loaded_persistence = runner._load_physical_archive(
        path,
        physical_mode="persistence_fallback",
    )
    np.testing.assert_array_equal(prediction, loaded_persistence)

    with np.load(path, allow_pickle=False) as archive:
        changed = {name: np.asarray(archive[name]) for name in archive.files}
    changed["prediction_m"] = changed["prediction_m"].copy()
    changed["prediction_m"][1, 0, 0] = 0.001
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="persistence fallback is not exact"):
        runner._load_physical_archive(path, physical_mode="persistence_fallback")


def test_technical_failure_carrier_is_exact_physical_fallback() -> None:
    frame_zero = np.zeros((128, 3), dtype=np.float32)
    persistence = np.repeat(frame_zero[None], 76, axis=0)
    physical = persistence.copy()
    physical[58:, :, 0] = np.linspace(0.0, 0.01, 18, dtype=np.float32)[:, None]
    problem = runner._technical_fallback_problem(
        object_id="026-sock",
        episode_id=7,
        stratum="sheet",
        physical_prediction_m=physical,
        persistence_m=persistence,
        physical_mode="warp_twin",
        implementation_revision="1" * 40,
        source_artifact_ids={"physical/source.npz": "2" * 64},
        failure_stage="prefix_provider",
        failure=ValueError("provider failed at a private path"),
    )
    result = run_deform360_joint_sparse_prediction_v5(problem)

    assert problem.factor_admitted is False
    assert problem.observation_batch.metadata["technical_failure"] is True
    assert problem.observation_batch.metadata["synthetic_observation_used"] is False
    assert "private path" not in str(problem.observation_batch.metadata)
    for method_id in RAW_METHOD_IDS:
        np.testing.assert_array_equal(result.trajectories_m[method_id], physical)
