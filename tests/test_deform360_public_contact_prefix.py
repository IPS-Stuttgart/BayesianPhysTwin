from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_public_contact_prefix as contact_api
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    _gripper_taxel_points,
)
from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY,
    DEFORM360_PREPARED_SOURCE_INVENTORY_SCHEMA,
    DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS,
    DEFORM360_PREPARED_SOURCE_INVENTORY_STATUS,
    DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION,
)
from bayesian_phystwin.deform360_public_contact_prefix import (
    TAXELS_PER_GRIPPER,
    build_deform360_tactile_axis_map,
    load_deform360_tactile_axis_map,
    materialize_deform360_public_contact_prefix,
    save_deform360_tactile_axis_map,
    validate_deform360_public_contact_prefix,
    validate_deform360_tactile_axis_map,
)

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/materialize_deform360_calibration_factors.py"
CLI_SPEC = importlib.util.spec_from_file_location("contact_prefix_cli", CLI_PATH)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI = importlib.util.module_from_spec(CLI_SPEC)
sys.modules[CLI_SPEC.name] = CLI
CLI_SPEC.loader.exec_module(CLI)

OBJECT_ID = "object-00"
INFORMATION_BOUNDARY = {
    "calibration_camera_payloads_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_robot_state_opened": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _dummy_record(name: str = "unused") -> dict[str, Any]:
    return {"path": name, "sha256": "a" * 64, "byte_count": 1}


def _camera() -> dict[str, Any]:
    record = _dummy_record()
    return {
        "camera": "camera-0",
        "video": record,
        "preview": record,
        "timestamps": record,
        "alignment": record,
        "metadata": record,
        "frame_count": 81,
        "width": 640,
        "height": 320,
        "fps": 30.0,
        "timeline_sha256": "b" * 64,
    }


def _robot(root: Path, *, bimanual: bool) -> tuple[Path, np.ndarray, np.ndarray]:
    path = root / OBJECT_ID / "episode_0000" / "robot" / "robot.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = 2 if bimanual else 1
    poses: np.ndarray
    openings: np.ndarray
    actions: np.ndarray
    if bimanual:
        poses = np.tile(np.eye(4), (81, axes, 1, 1))
        poses[:, 1, 0, 3] = 1.0
        openings = np.full((81, axes), 0.06)
        actions = np.zeros((81, axes, 5, 3), dtype=np.float64)
    else:
        poses = np.tile(np.eye(4), (81, 1, 1))
        openings = np.full(81, 0.06)
        actions = np.zeros((81, 5, 3), dtype=np.float64)
    np.savez(
        path,
        actions=actions,
        T_worlds=poses,
        openings=openings,
        bimanual=np.asarray(bimanual, dtype=np.bool_),
    )
    return path, poses, openings


def _tactile(
    root: Path,
    sensor: str,
    *,
    active_frames: tuple[int, ...],
    scale: float = 1.0,
) -> Path:
    path = root / OBJECT_ID / "episode_0000" / sensor / "synced_tactile.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    values: np.ndarray = np.zeros((81, 16, 32), dtype=np.float32)
    for frame in active_frames:
        values[frame, 0, 0] = scale
    np.save(path, values, allow_pickle=False)
    return path


def _inventory(
    tmp_path: Path,
    *,
    bimanual: bool = False,
    active_frames: tuple[int, ...] = (10, 11, 12),
    scale: float = 1.0,
) -> tuple[Path, Path, dict[str, Any], np.ndarray, np.ndarray]:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    robot_path, poses, openings = _robot(processed, bimanual=bimanual)
    groups = (
        ("brics-odroid_tactilel", "brics-odroid_tactiler")
        if bimanual
        else ("brics-odroid_tactilel",)
    )
    tactile_paths = []
    for group in groups:
        tactile_paths.extend(
            [
                _tactile(
                    processed,
                    f"{group}_left",
                    active_frames=active_frames,
                    scale=scale,
                ),
                _tactile(
                    processed,
                    f"{group}_right",
                    active_frames=active_frames,
                    scale=2.0 * scale,
                ),
            ]
        )
    objects = []
    for index in range(10):
        object_id = f"object-{index:02d}"
        if object_id == OBJECT_ID:
            episode_files = {"robot": _record(robot_path, root=processed)}
            tactile = [
                {
                    "sensor": path.parent.name,
                    **_record(path, root=processed),
                }
                for path in sorted(tactile_paths)
            ]
        else:
            episode_files = {"robot": _dummy_record(f"{object_id}/robot.npz")}
            tactile = []
        objects.append(
            {
                "object_id": object_id,
                "episode_id": 0,
                "stratum": "sheet" if index < 5 else "volumetric",
                "synthetic_episode_index": 0,
                "aligned_frame_count": 81,
                "action_window": {
                    "selected_raw_frame_range_half_open": [0, 81],
                    "prediction_raw_frame_range_half_open": [0, 76],
                    "prefix_raw_frame_range_half_open": [0, 58],
                },
                "episode_files": episode_files,
                "cameras": [_camera()],
                "tactile": tactile,
            }
        )
    identity = {
        "schema": DEFORM360_PREPARED_SOURCE_INVENTORY_SCHEMA,
        "schema_version": DEFORM360_PREPARED_SOURCE_INVENTORY_VERSION,
        "semantics": DEFORM360_PREPARED_SOURCE_INVENTORY_SEMANTICS,
        "status": DEFORM360_PREPARED_SOURCE_INVENTORY_STATUS,
        "implementation_revision": "1" * 40,
        "calibration_source_revision": "2" * 40,
        "processing_revision": "3" * 40,
        "selection_artifact_sha256": "4" * 64,
        "visual_provider_lock_id": "5" * 64,
        "calibration_source_run_record_sha256": "6" * 64,
        "object_count": 10,
        "objects": objects,
        "source_artifacts": {
            "sources/calibration-source/result.json": "7" * 64,
        },
        "information_boundary": INFORMATION_BOUNDARY,
        "claim_boundary": DEFORM360_PREPARED_SOURCE_INVENTORY_CLAIM_BOUNDARY,
    }
    inventory = {**identity, "inventory_id": content_id(identity)}
    inventory_path = tmp_path / "prepared-source-inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory_path, processed, inventory, poses, openings


def _axis_map(
    tmp_path: Path,
    inventory: dict[str, Any],
    mapping: dict[str, int],
) -> Path:
    path = tmp_path / "tactile-axis-map.json"
    value = build_deform360_tactile_axis_map(
        object_id=OBJECT_ID,
        episode_id=0,
        prepared_source_inventory_id=inventory["inventory_id"],
        group_to_robot_axis=dict(sorted(mapping.items())),
        selection_evidence_id="8" * 64,
    )
    save_deform360_tactile_axis_map(value, path)
    return path


def test_public_monomanual_prefix_uses_official_interleaving_and_geometry(
    tmp_path: Path,
) -> None:
    inventory_path, processed, inventory, poses, openings = _inventory(tmp_path)
    axis_map = _axis_map(
        tmp_path,
        inventory,
        {"brics-odroid_tactilel": 0},
    )
    output = tmp_path / "contact-prefix"

    result = materialize_deform360_public_contact_prefix(
        prepared_source_inventory_path=inventory_path,
        processed_root=processed,
        tactile_axis_map_path=axis_map,
        object_id=OBJECT_ID,
        output_directory=output,
    )

    assert result["status"] == "materialized"
    assert result["row_count"] == 3
    assert result["supported_robot_axes"] == [0]
    assert result["information_boundary"]["human_approval_required"] is False
    np.testing.assert_array_equal(
        np.load(output / "frame-ids.npy", allow_pickle=False),
        [10, 11, 12],
    )
    responses = np.load(output / "tactile-response.npy", allow_pickle=False)
    assert responses.shape == (3, TAXELS_PER_GRIPPER)
    np.testing.assert_array_equal(responses[:, :2], [[1.0, 2.0]] * 3)
    np.testing.assert_array_equal(responses[:, 2:], 0.0)
    positions = np.load(
        output / "taxel-world-positions-m.npy",
        allow_pickle=False,
    )
    expected = _gripper_taxel_points(float(openings[10]), poses[10])
    np.testing.assert_allclose(positions[0], expected, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(
        np.load(output / "source-reliability.npy", allow_pickle=False),
        np.ones(3),
    )
    assert str(tmp_path) not in (output / "contact-prefix.json").read_text()
    assert validate_deform360_public_contact_prefix(output) == result


def test_bimanual_map_controls_geometry_without_name_guessing(tmp_path: Path) -> None:
    inventory_path, processed, inventory, poses, openings = _inventory(
        tmp_path,
        bimanual=True,
        active_frames=(4,),
    )
    axis_map = _axis_map(
        tmp_path,
        inventory,
        {
            "brics-odroid_tactilel": 1,
            "brics-odroid_tactiler": 0,
        },
    )
    output = tmp_path / "contact-prefix"

    result = materialize_deform360_public_contact_prefix(
        prepared_source_inventory_path=inventory_path,
        processed_root=processed,
        tactile_axis_map_path=axis_map,
        object_id=OBJECT_ID,
        output_directory=output,
    )

    assert result["row_count"] == 2
    assert result["supported_robot_axes"] == [0, 1]
    assert [item["robot_axis"] for item in result["mapped_groups"]] == [0, 1]
    sensors = json.loads((output / "sensor-names.json").read_text())
    assert sensors == ["brics-odroid_tactiler", "brics-odroid_tactilel"]
    positions = np.load(
        output / "taxel-world-positions-m.npy",
        allow_pickle=False,
    )
    expected_axis_zero = _gripper_taxel_points(
        openings[4, 0],
        poses[4, 0],
    )
    expected_axis_one = _gripper_taxel_points(
        openings[4, 1],
        poses[4, 1],
    )
    np.testing.assert_allclose(positions[0], expected_axis_zero)
    np.testing.assert_allclose(positions[1], expected_axis_one)


def test_no_contact_is_retained_as_support_negative_and_cli_returns_three(
    tmp_path: Path,
) -> None:
    inventory_path, processed, inventory, _poses, _openings = _inventory(
        tmp_path,
        active_frames=(),
    )
    mapping_input = tmp_path / "mapping.json"
    mapping_input.write_text(
        json.dumps({"brics-odroid_tactilel": 0}) + "\n",
        encoding="utf-8",
    )
    axis_map = tmp_path / "axis-map.json"
    assert (
        CLI.main(
            [
                "tactile-axis-map",
                "--prepared-source-inventory",
                str(inventory_path),
                "--object-id",
                OBJECT_ID,
                "--group-to-robot-axis",
                str(mapping_input),
                "--selection-evidence-id",
                "8" * 64,
                "--output",
                str(axis_map),
            ]
        )
        == 0
    )
    output = tmp_path / "contact-prefix"
    arguments = [
        "public-contact-prefix",
        "--prepared-source-inventory",
        str(inventory_path),
        "--processed-root",
        str(processed),
        "--tactile-axis-map",
        str(axis_map),
        "--object-id",
        OBJECT_ID,
        "--output-dir",
        str(output),
    ]
    assert CLI.main(arguments) == 3
    result = validate_deform360_public_contact_prefix(output)
    assert result["status"] == "support-negative"
    assert result["row_count"] == 0
    assert result["missing_contact_robot_axes"] == [0]
    assert CLI.main(arguments) == 2


def test_axis_map_is_strict_content_addressed_and_one_to_one(tmp_path: Path) -> None:
    _inventory_path, _processed, inventory, _poses, _openings = _inventory(
        tmp_path,
        bimanual=True,
    )
    with pytest.raises(ValueError, match="one-to-one"):
        build_deform360_tactile_axis_map(
            object_id=OBJECT_ID,
            episode_id=0,
            prepared_source_inventory_id=inventory["inventory_id"],
            group_to_robot_axis={
                "brics-odroid_tactilel": 0,
                "brics-odroid_tactiler": 0,
            },
            selection_evidence_id="8" * 64,
        )

    valid = build_deform360_tactile_axis_map(
        object_id=OBJECT_ID,
        episode_id=0,
        prepared_source_inventory_id=inventory["inventory_id"],
        group_to_robot_axis={
            "brics-odroid_tactilel": 0,
            "brics-odroid_tactiler": 1,
        },
        selection_evidence_id="8" * 64,
    )
    changed = json.loads(json.dumps(valid))
    changed["group_to_robot_axis"]["brics-odroid_tactilel"] = 1
    with pytest.raises(ValueError, match="one-to-one|does not match"):
        validate_deform360_tactile_axis_map(changed)

    path = tmp_path / "axis-map.json"
    save_deform360_tactile_axis_map(valid, path)
    assert load_deform360_tactile_axis_map(path) == valid
    with pytest.raises(FileExistsError):
        save_deform360_tactile_axis_map(valid, path)


def test_axis_map_rejects_noncanonical_inputs_and_contract_tampering(
    tmp_path: Path,
) -> None:
    _inventory_path, _processed, inventory, _poses, _openings = _inventory(
        tmp_path,
        bimanual=True,
    )
    with pytest.raises(ValueError, match="keys must be sorted"):
        build_deform360_tactile_axis_map(
            object_id=OBJECT_ID,
            episode_id=0,
            prepared_source_inventory_id=inventory["inventory_id"],
            group_to_robot_axis={
                "brics-odroid_tactiler": 1,
                "brics-odroid_tactilel": 0,
            },
            selection_evidence_id="8" * 64,
        )
    with pytest.raises(ValueError, match="gripper groups"):
        build_deform360_tactile_axis_map(
            object_id=OBJECT_ID,
            episode_id=0,
            prepared_source_inventory_id=inventory["inventory_id"],
            group_to_robot_axis={"brics-odroid_tactilel_left": 0},
            selection_evidence_id="8" * 64,
        )

    valid = build_deform360_tactile_axis_map(
        object_id=OBJECT_ID,
        episode_id=0,
        prepared_source_inventory_id=inventory["inventory_id"],
        group_to_robot_axis={
            "brics-odroid_tactilel": 0,
            "brics-odroid_tactiler": 1,
        },
        selection_evidence_id="8" * 64,
    )
    mutations = {
        "schema": "changed",
        "schema_version": 2,
        "semantics": "changed",
        "information_boundary": {},
        "claim_boundary": "changed",
    }
    for field, value in mutations.items():
        changed = json.loads(json.dumps(valid))
        changed[field] = value
        with pytest.raises(ValueError):
            validate_deform360_tactile_axis_map(changed)


def test_scalar_contract_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="literal string"):
        contact_api._literal_string("", name="value")
    with pytest.raises(ValueError, match="integer"):
        contact_api._literal_integer(True, name="value")
    with pytest.raises(ValueError, match="JSON object"):
        contact_api._mapping([], name="value")
    with pytest.raises(ValueError, match="JSON array"):
        contact_api._sequence("value", name="value")


def test_tactile_magnitude_does_not_change_prior_reliability(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_inventory, first_root, first_value, _poses, _openings = _inventory(
        first,
        scale=1.0,
    )
    second_inventory, second_root, second_value, _poses, _openings = _inventory(
        second,
        scale=100.0,
    )
    first_map = _axis_map(
        first,
        first_value,
        {"brics-odroid_tactilel": 0},
    )
    second_map = _axis_map(
        second,
        second_value,
        {"brics-odroid_tactilel": 0},
    )
    first_output = first / "output"
    second_output = second / "output"
    materialize_deform360_public_contact_prefix(
        prepared_source_inventory_path=first_inventory,
        processed_root=first_root,
        tactile_axis_map_path=first_map,
        object_id=OBJECT_ID,
        output_directory=first_output,
    )
    materialize_deform360_public_contact_prefix(
        prepared_source_inventory_path=second_inventory,
        processed_root=second_root,
        tactile_axis_map_path=second_map,
        object_id=OBJECT_ID,
        output_directory=second_output,
    )
    np.testing.assert_array_equal(
        np.load(first_output / "source-reliability.npy", allow_pickle=False),
        np.load(second_output / "source-reliability.npy", allow_pickle=False),
    )


def test_source_tamper_and_published_artifact_tamper_are_rejected(
    tmp_path: Path,
) -> None:
    inventory_path, processed, inventory, _poses, _openings = _inventory(tmp_path)
    axis_map = _axis_map(
        tmp_path,
        inventory,
        {"brics-odroid_tactilel": 0},
    )
    tactile = next(processed.rglob("*_left/synced_tactile.npy"))
    with tactile.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="byte count changed|SHA-256 changed"):
        materialize_deform360_public_contact_prefix(
            prepared_source_inventory_path=inventory_path,
            processed_root=processed,
            tactile_axis_map_path=axis_map,
            object_id=OBJECT_ID,
            output_directory=tmp_path / "tampered-source-output",
        )

    clean = tmp_path / "clean"
    inventory_path, processed, inventory, _poses, _openings = _inventory(clean)
    axis_map = _axis_map(
        clean,
        inventory,
        {"brics-odroid_tactilel": 0},
    )
    output = clean / "output"
    materialize_deform360_public_contact_prefix(
        prepared_source_inventory_path=inventory_path,
        processed_root=processed,
        tactile_axis_map_path=axis_map,
        object_id=OBJECT_ID,
        output_directory=output,
    )
    with (output / "source-reliability.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="digest changed"):
        validate_deform360_public_contact_prefix(output)
