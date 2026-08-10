from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin import deform360_bias_aware_prospective_artifacts as artifacts
from bayesian_phystwin import (
    deform360_joint_sparse_physical_source_v5 as physical_source,
)
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    file_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID as HISTORICAL_PROTOCOL_ID,
)
from bayesian_phystwin.deform360_bias_aware_prospective_staging import (
    select_action_only_window,
)
from bayesian_phystwin.deform360_joint_sparse_physical_source_v5 import (
    PROTOCOL_ID,
    activate_joint_sparse_physical_runtime_v5,
    joint_sparse_physical_case_record_v5,
    joint_sparse_physical_case_records_v5,
    load_joint_sparse_physical_execution_protocol_v5,
    materialize_joint_sparse_physical_source_v5,
    patch_joint_sparse_physical_stage_v5,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)


def _write(path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "byte_count": len(data),
    }


def _relative_record(record: dict[str, object], root: Path) -> dict[str, object]:
    return {**record, "path": Path(str(record["path"])).relative_to(root).as_posix()}


def _synthetic_inventory(
    tmp_path: Path,
) -> tuple[Path, Path, str, dict[str, object]]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = lock["cohort"]["development_objects"][0]
    object_id = selected["object_id"]
    episode_id = selected["episode_id"]
    processed = tmp_path / "processed"
    source = processed / object_id / "episode_0000"
    source.mkdir(parents=True)

    actions = np.zeros((100, 5, 3), dtype=np.float64)
    actions[:, 0, 0] = np.linspace(0.0, 1.0, 100)
    openings = np.zeros(100, dtype=np.float64)
    robot = source / "robot/robot.npz"
    robot.parent.mkdir(parents=True)
    np.savez(
        robot,
        actions=actions,
        openings=openings,
        T_worlds=np.repeat(np.eye(4)[None], 100, axis=0),
        bimanual=np.asarray(False),
        format_version=np.asarray(1, dtype=np.uint16),
    )
    episode_records = {
        "alignment": _relative_record(
            _write(source / "alignment.json", b"{}\n"), processed
        ),
        "extrinsics": _relative_record(
            _write(source / "extrinsics.npy", b"extrinsics"), processed
        ),
        "robot": _relative_record(
            {
                "path": robot.as_posix(),
                "sha256": file_sha256(robot),
                "byte_count": robot.stat().st_size,
            },
            processed,
        ),
        "undistorted_intrinsics": _relative_record(
            _write(source / "undistorted_intrinsics.npy", b"intrinsics"), processed
        ),
    }
    camera_records = []
    for index in range(8):
        camera = f"camera-{index:02d}"
        camera_root = source / camera
        camera_records.append(
            {
                "camera": camera,
                "alignment": _relative_record(
                    _write(camera_root / "alignment.json", b"{}\n"), processed
                ),
                "metadata": _relative_record(
                    _write(camera_root / "metadata.json", b"{}\n"), processed
                ),
                "timestamps": _relative_record(
                    _write(camera_root / "aligned_timestamps.txt", b"0\n"), processed
                ),
                "video": _relative_record(
                    _write(camera_root / "undistorted.mp4", b"video"), processed
                ),
            }
        )
    selection = select_action_only_window(actions, openings)
    object_rows: list[dict[str, object]] = []
    for row in lock["cohort"]["development_objects"]:
        if row["object_id"] == object_id:
            object_rows.append(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "episode_files": episode_records,
                    "cameras": camera_records,
                    "action_window": selection,
                }
            )
        else:
            object_rows.append({"object_id": row["object_id"]})
    inventory = {
        "schema": "bayesian-phystwin.deform360-calibration-prepared-source-inventory",
        "schema_version": 1,
        "inventory_id": "a" * 64,
        "object_count": 10,
        "objects": object_rows,
        "information_boundary": {
            "calibration_target_metrics_computed": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    lock["physical_baseline"]["prepared_source_inventory"] = {
        "file_sha256": file_sha256(inventory_path),
        "inventory_id": inventory["inventory_id"],
    }
    identity = {key: value for key, value in lock.items() if key != "execution_lock_id"}
    lock["execution_lock_id"] = content_id(identity)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path, processed, object_id, inventory


def test_v5_physical_protocol_has_only_the_ten_public_source_cases() -> None:
    protocol = load_joint_sparse_physical_execution_protocol_v5(LOCK)
    assert protocol["config"]["protocol_id"] == PROTOCOL_ID
    assert len(joint_sparse_physical_case_records_v5(LOCK)) == 10
    assert joint_sparse_physical_case_record_v5(
        LOCK,
        object_id="026-sock-cloth",
        episode_id=7,
    ) == {
        "case": "026-sock-cloth-ep0007",
        "object_id": "026-sock-cloth",
        "episode_id": 7,
        "episode_key": "026-sock-cloth/7",
        "stratum": "sheet",
        "role": "calibration",
    }


def test_v5_physical_runtime_is_process_local() -> None:
    assert artifacts.PROTOCOL_ID == HISTORICAL_PROTOCOL_ID
    with activate_joint_sparse_physical_runtime_v5():
        assert artifacts.PROTOCOL_ID == PROTOCOL_ID
        assert (
            artifacts.prospective_case_record(
                LOCK,
                object_id="193-frog",
                episode_id=7,
            )["case"]
            == "193-frog-ep0007"
        )
    assert artifacts.PROTOCOL_ID == HISTORICAL_PROTOCOL_ID


@pytest.mark.parametrize(
    ("function", "value", "match"),
    [
        (physical_source._mapping, None, "JSON object"),
        (physical_source._sequence, "not-an-array", "JSON array"),
        (physical_source._nonempty_string, "", "non-empty string"),
        (physical_source._sha256, "short", "SHA-256 digest"),
    ],
)
def test_physical_source_json_guards_reject_malformed_values(
    function: object,
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        function(value, name="value")


def test_physical_source_execution_preflight_binds_exact_sources() -> None:
    lock = physical_source.validate_joint_sparse_physical_execution_v5(
        LOCK,
        repository=ROOT,
        require_clean_repository=False,
    )
    assert lock["physical_baseline"]["process_local_adapter_protocol_id"] == PROTOCOL_ID
    assert (
        Path(physical_source._git_output(ROOT, "rev-parse", "--show-toplevel"))
        .resolve()
        .samefile(ROOT)
    )


def test_nonphysical_stage_patch_does_not_install_subprocess_rewriter(
    tmp_path: Path,
) -> None:
    module = SimpleNamespace(PROTOCOL_ID=HISTORICAL_PROTOCOL_ID)
    physical_source._set(module, "missing", object(), [])
    patch_joint_sparse_physical_stage_v5(
        module,
        stage="frame-zero",
        repository=ROOT,
        execution_lock=tmp_path / "lock.json",
    )
    assert module.PROTOCOL_ID == PROTOCOL_ID
    assert not hasattr(module, "_run_logged")


def test_physical_prior_patch_rewrites_only_the_automatic_twin(tmp_path: Path) -> None:
    observed: list[list[str]] = []

    def run_logged(command: list[str], **_kwargs: object) -> tuple[int, float]:
        observed.append(command)
        return 0, 1.0

    module = SimpleNamespace(
        PROTOCOL_ID=HISTORICAL_PROTOCOL_ID,
        load_bias_aware_prospective_protocol=None,
        prospective_case_record=None,
        _run_logged=run_logged,
    )
    patch_joint_sparse_physical_stage_v5(
        module,
        stage="physical-prior",
        repository=ROOT,
        execution_lock=tmp_path / "lock.json",
    )
    module._run_logged(["python", "/old/build_deform360_bias_aware_automatic_twin.py"])
    module._run_logged(["python", "/old/run_deform360_official_phystwin_smoke.py"])
    assert "run_deform360_joint_sparse_physical_source_v5.py" in observed[0][1]
    assert observed[0][2:8] == [
        "--execution-repo",
        str(ROOT),
        "--execution-lock",
        str(tmp_path / "lock.json"),
        "--stage",
        "automatic-twin",
    ]
    assert observed[1][1] == "/old/run_deform360_official_phystwin_smoke.py"


def test_materializer_copies_only_attested_released_source(tmp_path: Path) -> None:
    lock, processed, object_id, inventory = _synthetic_inventory(tmp_path)
    output = tmp_path / "output"
    manifest = materialize_joint_sparse_physical_source_v5(
        execution_lock_path=lock,
        prepared_source_inventory_path=tmp_path / "inventory.json",
        processed_root=processed,
        object_id=object_id,
        output_root=output,
    )
    destination = output / object_id / f"episode_{manifest['episode_id']:04d}"
    assert manifest["protocol_id"] == PROTOCOL_ID
    assert manifest["prepared_source_inventory_id"] == inventory["inventory_id"]
    assert manifest["target_access_authorization"] is None
    assert manifest["information_boundary"] == {
        "released_calibration_recordings_copied": True,
        "development_suffix_scored": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "new_measurements_collected": False,
        "human_approval_used": False,
    }
    assert (destination / "robot/robot.npz").is_file()
    assert len(list(destination.glob("camera-*/undistorted.mp4"))) == 8
    assert (destination / "bias_aware_source_preparation_manifest.json").is_file()


def test_materializer_rejects_changed_inventory_bytes(tmp_path: Path) -> None:
    lock, processed, object_id, inventory = _synthetic_inventory(tmp_path)
    inventory["information_boundary"]["target_outcomes_used"] = True
    (tmp_path / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    try:
        materialize_joint_sparse_physical_source_v5(
            execution_lock_path=lock,
            prepared_source_inventory_path=tmp_path / "inventory.json",
            processed_root=processed,
            object_id=object_id,
            output_root=tmp_path / "output",
        )
    except ValueError as error:
        assert "inventory file changed" in str(error)
    else:
        raise AssertionError("changed prepared inventory was accepted")


def test_materializer_rejects_object_outside_source_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the v5 source lock"):
        materialize_joint_sparse_physical_source_v5(
            execution_lock_path=LOCK,
            prepared_source_inventory_path=tmp_path / "not-opened.json",
            processed_root=tmp_path / "not-opened",
            object_id="unregistered-object",
            output_root=tmp_path / "output",
        )
