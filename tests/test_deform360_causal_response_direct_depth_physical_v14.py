from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_physical import (
    GRAPH_BASIS_RANK,
    PHYSICAL_FRAME_COUNT,
    PRELOCK_PROTOCOL_CONTRACT,
    PRELOCK_PROTOCOL_ID,
    PRELOCK_PROTOCOL_KIND,
    build_v14_prediction_only_bundle,
    load_v14_physical_prelock_protocol,
    v14_physical_case_record,
    validate_v14_physical_artifacts,
    write_v14_physical_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
LOCKED_PROTOCOL = (
    ROOT / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_physical_prelock.json"
)
STAGING_QUEUE = (
    ROOT / "configs/sota/deform360_causal_response_direct_depth_v14_staging_queue.json"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-physical-prelock-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ledger_sha256(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-physical-geometry-v14\0"
        + json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _protocol(path: Path) -> dict[str, object]:
    cases = []
    for rank in range(3, 15):
        cases.append(
            {
                "queue_rank": rank,
                "object_hash": _digest(f"object-{rank}"),
                "case_hash": _digest(f"case-{rank}"),
                "metadata_sha256": _digest(f"metadata-{rank}"),
                "physical_node_count": 128 + rank,
                "successful_camera_count": 12,
                "runtime_contract_version": 1 if rank == 3 else 2,
                "geometry_manifest_artifact_sha256": _digest(
                    f"manifest-artifact-{rank}"
                ),
                "geometry_manifest_file_sha256": _digest(f"manifest-file-{rank}"),
                "geometry_result_artifact_sha256": _digest(f"result-artifact-{rank}"),
                "geometry_result_file_sha256": _digest(f"result-file-{rank}"),
                "runtime_application_artifact_sha256": _digest(
                    f"runtime-artifact-{rank}"
                ),
                "runtime_application_file_sha256": _digest(f"runtime-file-{rank}"),
            }
        )
    parents = {
        key: _digest(key)
        for key in (
            "method_protocol_config_sha256",
            "method_protocol_file_sha256",
            "staging_queue_sha256",
            "staging_queue_file_sha256",
            "geometry_protocol_config_sha256",
            "geometry_protocol_file_sha256",
            "runtime_v1_config_sha256",
            "runtime_v1_file_sha256",
            "validation_v1_config_sha256",
            "validation_v1_file_sha256",
            "runtime_v2_config_sha256",
            "runtime_v2_file_sha256",
            "validation_v2_config_sha256",
            "validation_v2_file_sha256",
        )
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": PRELOCK_PROTOCOL_KIND,
        "contract": PRELOCK_PROTOCOL_CONTRACT,
        "protocol_id": PRELOCK_PROTOCOL_ID,
        "method_protocol_id": ("deform360-causal-response-direct-depth-v14-source"),
        "status": "locked_after_geometry_before_physical_carrier_execution",
        "config_sha256": "0" * 64,
        "implementation": {
            "parent_commit": "1" * 40,
            "file_sha256": {
                "artifact_module": _digest("artifact-module"),
                "automatic_twin": _digest("automatic-twin"),
                "physical_runner": _digest("physical-runner"),
            },
        },
        "parent_artifacts": parents,
        "numerical_contract": {
            "canonical_node_count": 384,
            "graph_basis_rank": 8,
            "prediction_frame_count": 76,
            "automatic_twin_source": "frame_zero_geometry_only",
            "future_robot_action_known": True,
            "automatic_twin_inadmissible_fallback": "bit_exact_persistence",
        },
        "geometry_cases": cases,
        "geometry_ledger_sha256": _ledger_sha256(cases),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_robot_action_frames_used": list(range(76)),
            "future_object_observation_read": False,
            "prefix_tactile_read": False,
            "identity_or_metric_outcome_read": False,
            "source_lock_required_before_execution": False,
            "source_lock_construction_uses_output_hashes_only": True,
            "plaintext_identity_retained_in_sealed_output": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["config_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_ascii_ply(path: Path, count: int = 128) -> np.ndarray:
    points = np.column_stack(
        (
            np.linspace(0.0, 0.1, count),
            np.linspace(0.2, 0.3, count),
            np.linspace(0.4, 0.5, count),
        )
    ).astype(np.float32)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z} 10 20 30" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return points


def _write_robot(path: Path) -> None:
    poses = np.repeat(np.eye(4)[None], PHYSICAL_FRAME_COUNT, axis=0)
    np.savez(
        path,
        format_version=np.asarray(1, dtype=np.int64),
        actions=np.zeros((PHYSICAL_FRAME_COUNT, 1), dtype=np.float32),
        T_worlds=poses,
        openings=np.full(PHYSICAL_FRAME_COUNT, 0.08, dtype=np.float32),
        bimanual=np.asarray(False),
    )


def _physical_arrays(count: int = 4) -> dict[str, np.ndarray]:
    frame_zero = np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 100
    persistence = np.repeat(frame_zero[None], PHYSICAL_FRAME_COUNT, axis=0)
    vectors = np.arange(
        1,
        count * 3 * GRAPH_BASIS_RANK + 1,
        dtype=np.float64,
    ).reshape(count * 3, GRAPH_BASIS_RANK)
    vectors += np.eye(count * 3, GRAPH_BASIS_RANK)
    basis, _ = np.linalg.qr(vectors)
    return {
        "action_support": np.linspace(0.0, 1.0, count, dtype=np.float32),
        "driven_readout_m": persistence.copy(),
        "frame_zero_points_m": frame_zero,
        "graph_basis": basis[:, :GRAPH_BASIS_RANK]
        .reshape(count, 3, GRAPH_BASIS_RANK)
        .astype(np.float32),
        "persistence_prediction_m": persistence.copy(),
        "physical_prediction_m": persistence.copy(),
        "zero_action_readout_m": persistence.copy(),
    }


def test_prelock_protocol_and_hash_only_prediction_bundle(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol(protocol_path)
    assert load_v14_physical_prelock_protocol(protocol_path) == protocol

    ply = tmp_path / "start_obj_pcd.ply"
    expected = _write_ascii_ply(ply)
    robot = tmp_path / "robot.npz"
    _write_robot(robot)
    bundle = tmp_path / "prediction.pkl"
    record = {
        "queue_rank": 3,
        "object_hash": _digest("object"),
        "case_hash": _digest("case"),
    }
    summary = build_v14_prediction_only_bundle(
        ply,
        robot,
        bundle,
        case_record=record,
    )
    with bundle.open("rb") as stream:
        payload = pickle.load(stream)
    marker = payload["prediction_only_input"]
    assert marker["object_hash"] == record["object_hash"]
    assert marker["case_hash"] == record["case_hash"]
    assert marker["plaintext_object_or_episode_identity_present"] is False
    assert "object_id" not in marker
    assert "episode_id" not in marker
    assert np.array_equal(payload["object_points"][0], expected)
    assert np.array_equal(
        payload["object_points"],
        np.repeat(payload["object_points"][:1], PHYSICAL_FRAME_COUNT, axis=0),
    )
    assert summary["frame_count"] == PHYSICAL_FRAME_COUNT


def test_locked_prelock_protocol_binds_all_twelve_geometry_cases() -> None:
    protocol = load_v14_physical_prelock_protocol(LOCKED_PROTOCOL)
    records = [
        v14_physical_case_record(protocol, STAGING_QUEUE, queue_rank=rank)
        for rank in range(3, 15)
    ]
    assert [record["queue_rank"] for record in records] == list(range(3, 15))
    assert len({record["object_hash"] for record in records}) == 12
    assert len({record["case_hash"] for record in records}) == 12
    assert min(record["physical_node_count"] for record in records) == 353
    assert max(record["physical_node_count"] for record in records) == 2547


def test_physical_fallback_is_exact_and_validated(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _protocol(protocol_path)
    input_file = tmp_path / "input.txt"
    input_file.write_text("bound input", encoding="utf-8")
    arrays = _physical_arrays()
    record = {
        "queue_rank": 3,
        "object_hash": _digest("object"),
        "case_hash": _digest("case"),
        "category": "compact",
        "bimanual_value": "no",
        "metadata_sha256": _digest("metadata"),
        "physical_node_count": 128,
        "successful_camera_count": 12,
    }
    output = tmp_path / "physical"
    manifest = write_v14_physical_artifacts(
        output,
        arrays,
        prelock_protocol_path=protocol_path,
        case_record=record,
        physical_mode="automatic_twin_persistence_fallback",
        code_revision="1" * 40,
        input_files={"bound_input": input_file},
        runtime_provenance={"runtime": "test"},
        fallback_diagnostics={"reason": "test", "warp_attempted": False},
    )
    validated, stored = validate_v14_physical_artifacts(
        output,
        prelock_protocol_path=protocol_path,
    )
    assert validated == manifest
    assert validated["physical_admitted"] is False
    assert np.array_equal(
        stored["physical_prediction_m"],
        stored["persistence_prediction_m"],
    )
    assert (
        validated["information_boundary"][
            "plaintext_object_or_episode_identity_retained"
        ]
        is False
    )


def test_physical_archive_rejects_nonorthonormal_basis(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _protocol(protocol_path)
    input_file = tmp_path / "input.txt"
    input_file.write_text("bound input", encoding="utf-8")
    arrays = _physical_arrays()
    arrays["graph_basis"][:] = 0.0
    with pytest.raises(ValueError, match="not orthonormal"):
        write_v14_physical_artifacts(
            tmp_path / "physical",
            arrays,
            prelock_protocol_path=protocol_path,
            case_record={
                "queue_rank": 3,
                "object_hash": _digest("object"),
                "case_hash": _digest("case"),
                "category": "compact",
                "bimanual_value": "no",
                "metadata_sha256": _digest("metadata"),
                "physical_node_count": 128,
                "successful_camera_count": 12,
            },
            physical_mode="warp_twin",
            code_revision="1" * 40,
            input_files={"bound_input": input_file},
            runtime_provenance={"runtime": "test"},
        )


def test_v14_runners_do_not_require_an_existing_source_lock() -> None:
    for relative in (
        "scripts/remote/"
        "build_deform360_causal_response_direct_depth_v14_automatic_twin.py",
        "scripts/remote/run_deform360_causal_response_direct_depth_v14_physical.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "dynamic_provider_cohort" not in source
        assert "source-lock" not in source
        assert "--cohort-lock" not in source
