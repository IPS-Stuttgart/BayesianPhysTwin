from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_joint_sparse_geometric_materializer_v4 as materializer
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_geometric_candidates_v4 import (
    _Candidate,
    _collect_stream_candidates,
    _deterministic_select,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_common_v4 import (
    METRIC_BATCH_SCHEMA,
    METRIC_BATCH_SEMANTICS,
    METRIC_BATCH_VERSION,
    METRIC_PLAN_SCHEMA,
    METRIC_PLAN_SEMANTICS,
    METRIC_PLAN_VERSION,
    validate_materializer_policy,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_npz_v4 import (
    _load_npy_member,
    _read_exact,
    _read_npy_header,
    _zip_members,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_POLICY = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_geometric_materializer_v4.json"
)
V4_POLICY = (
    ROOT / "protocols/locks/deform360_official_hub_joint_sparse_observability_v4.json"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _npy_bytes(value: np.ndarray, *, version: tuple[int, int] | None = None) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(value),
        allow_pickle=False,
        version=version,
    )
    return buffer.getvalue()


def _write_prediction_bundle(root: Path, *, motioncrafter_revision: str) -> Path:
    window = root / "window.npz"
    with zipfile.ZipFile(window, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("window_id.npy", _npy_bytes(np.asarray("window-0")))
        archive.writestr(
            "frame_indices.npy",
            _npy_bytes(np.asarray([10, 11], dtype=np.int64)),
        )
        archive.writestr(
            "valid_mask.npy",
            _npy_bytes(np.ones((2, 2, 3), dtype=np.bool_)),
        )
        archive.writestr("point_map.npy", b"support-only fixture")
    run_spec = {"seed": 17, "fixture": "support-only"}
    run_spec_sha = hashlib.sha256(
        json.dumps(
            run_spec,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format_version": 1,
        "motioncrafter_commit": motioncrafter_revision,
        "overlap_windows": [
            {
                "window_id": "window-0",
                "path": window.name,
                "start_frame": 10,
                "stop_frame": 12,
            }
        ],
        "artifact_integrity": {
            "schema": "prob4d.motioncrafter-artifact-integrity.v1",
            "run_spec": run_spec,
            "run_spec_sha256": run_spec_sha,
            "members": [
                {
                    "path": window.name,
                    "sha256": _sha256(window),
                    "bytes": window.stat().st_size,
                    "kind": "independently_decoded_overlap_window",
                }
            ],
        },
    }
    path = root / "predictions.json"
    _write_json(path, manifest)
    return path


def _source_fixture(tmp_path: Path) -> dict[str, Any]:
    metric_batch = tmp_path / "metric-batch"
    metric_files = metric_batch / "metrics"
    prediction_root = tmp_path / "predictions"
    metric_files.mkdir(parents=True)
    prediction_root.mkdir()

    base_policy = json.loads(MATERIALIZER_POLICY.read_text(encoding="utf-8"))
    selection_artifact_id = "1" * 64
    visual_provider_lock_id = "2" * 64
    metric_revision = "a" * 40
    prob4d_revision = "b" * 40
    motioncrafter_revision = "c" * 40
    protocol_id = str(base_policy["protocol_id"])

    rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    for index in range(10):
        object_id = f"object-{index:02d}"
        episode_id = index
        stratum = "sheet" if index < 5 else "volumetric"
        rows.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
            }
        )
        streams: list[dict[str, object]] = []
        for camera_suffix in ("a", "b"):
            camera_id = f"camera-{camera_suffix}"
            job_id = hashlib.sha256(
                f"job:{object_id}:{episode_id}:{camera_id}".encode()
            ).hexdigest()
            prediction = prediction_root / f"{job_id}.json"
            metric_prefix = metric_files / f"{job_id}.npz"
            calibration = metric_files / f"{job_id}.json"
            _write_json(prediction, {"fixture": job_id})
            metric_prefix.write_bytes(b"metric-prefix-fixture\n")
            _write_json(calibration, {"fixture": job_id})
            streams.append(
                {
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "prediction_manifest": _record(prediction, root=prediction_root),
                    "metric_prefix": _record(metric_prefix, root=metric_files),
                    "metric_calibration": _record(calibration, root=metric_files),
                }
            )
        cases.append(
            {
                "case_id": hashlib.sha256(
                    f"case:{object_id}:{episode_id}".encode()
                ).hexdigest(),
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "causal_frame_range_half_open": [10, 12],
                "streams": streams,
            }
        )

    excluded_streams = []
    for index in range(11):
        row = rows[index % len(rows)]
        excluded_streams.append(
            {
                "job_id": hashlib.sha256(f"excluded:{index}".encode()).hexdigest(),
                "object_id": row["object_id"],
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "camera_id": f"excluded-camera-{index:02d}",
                "reason": "released-robot-geometry-outside-fixed-camera-prefix",
            }
        )

    selection = {
        "selection_artifact_sha256": selection_artifact_id,
        "protocol_id": protocol_id,
        "selection": {"calibration": rows},
    }
    provider = {"protocol_id": protocol_id}
    metric_policy = {"metric_source_kind": base_policy["metric_source_kind"]}
    camera_policy = {"replacement_allowed": False}
    selection_path = tmp_path / "selection.json"
    provider_path = tmp_path / "provider.json"
    metric_policy_path = tmp_path / "metric-policy.json"
    camera_policy_path = tmp_path / "camera-policy.json"
    _write_json(selection_path, selection)
    _write_json(provider_path, provider)
    _write_json(metric_policy_path, metric_policy)
    _write_json(camera_policy_path, camera_policy)

    production_identity = {
        "visual_provider_lock_id": visual_provider_lock_id,
        "provider_revision": prob4d_revision,
        "motioncrafter_revision": motioncrafter_revision,
        "object_count": 10,
        "camera_view_count": 324,
        "completely_succeeded_object_count": 10,
        "succeeded_job_count": 324,
        "technical_failure_job_count": 0,
        "status": "all-jobs-succeeded",
    }
    production = {
        **production_identity,
        "result_id": content_id(production_identity),
    }
    production_path = tmp_path / "production.json"
    _write_json(production_path, production)

    plan_identity = {
        "schema": METRIC_PLAN_SCHEMA,
        "schema_version": METRIC_PLAN_VERSION,
        "semantics": METRIC_PLAN_SEMANTICS,
        "selection_file_sha256": _sha256(selection_path),
        "visual_provider_spec_file_sha256": _sha256(provider_path),
        "metric_prior_policy_file_sha256": _sha256(metric_policy_path),
        "camera_eligibility_policy_file_sha256": _sha256(camera_policy_path),
        "prob4d_revision": prob4d_revision,
        "motioncrafter_revision": motioncrafter_revision,
        "visual_production_result_id": production["result_id"],
        "cases": cases,
        "excluded_streams": excluded_streams,
    }
    plan = {**plan_identity, "plan_id": content_id(plan_identity)}
    plan_path = metric_batch / "metric-prefix-plan.json"
    _write_json(plan_path, plan)

    metric_identity = {
        "schema": METRIC_BATCH_SCHEMA,
        "schema_version": METRIC_BATCH_VERSION,
        "semantics": METRIC_BATCH_SEMANTICS,
        "plan_file": _record(plan_path, root=metric_batch),
        "implementation_revision": metric_revision,
        "production_result_id": production["result_id"],
        "object_count": 10,
        "admitted_stream_count": 324,
        "supported_stream_count": 313,
        "support_negative_stream_count": 11,
        "technical_failure_stream_count": 0,
        "supported_object_count": 10,
        "plan_emitted": True,
        "status": "target-free-visible-streams-supported",
    }
    metric_result = {
        **metric_identity,
        "result_id": content_id(metric_identity),
    }
    metric_result_path = metric_batch / "metric-batch-result.json"
    _write_json(metric_result_path, metric_result)

    policy = dict(base_policy)
    policy.update(
        {
            "selection_artifact_sha256": selection_artifact_id,
            "visual_provider_lock_id": visual_provider_lock_id,
            "production_result_id": production["result_id"],
            "metric_batch_result_id": metric_result["result_id"],
            "metric_batch_implementation_revision": metric_revision,
            "prob4d_revision": prob4d_revision,
            "motioncrafter_revision": motioncrafter_revision,
        }
    )
    policy_identity = dict(policy)
    policy_identity.pop("artifact_id")
    policy["artifact_id"] = content_id(policy_identity)
    validate_materializer_policy(policy)
    materializer_policy_path = tmp_path / "materializer-policy.json"
    _write_json(materializer_policy_path, policy)

    v4_policy_path = tmp_path / "v4-policy.json"
    v4_policy_path.write_bytes(V4_POLICY.read_bytes())

    checksum_members = sorted(
        path
        for path in metric_batch.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (metric_batch / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(metric_batch).as_posix()}\n"
            for path in checksum_members
        ),
        encoding="ascii",
    )
    return {
        "metric_batch": metric_batch,
        "prediction_root": prediction_root,
        "production_path": production_path,
        "selection_path": selection_path,
        "provider_path": provider_path,
        "metric_policy_path": metric_policy_path,
        "camera_policy_path": camera_policy_path,
        "v4_policy_path": v4_policy_path,
        "materializer_policy_path": materializer_policy_path,
        "policy": policy,
        "metric_result": metric_result,
        "production": production,
    }


def _fake_collect_stream_candidates(**arguments: Any):
    object_id = str(arguments["object_id"])
    episode_id = int(arguments["episode_id"])
    camera_id = str(arguments["camera_id"])
    job_id = str(arguments["job_id"])
    camera_center = np.asarray(
        (-0.4 if camera_id.endswith("a") else 0.4, 0.0, -1.0),
        dtype=np.float64,
    )
    points = (
        (-0.20, -0.10, -0.05),
        (-0.10, 0.20, 0.10),
        (0.10, -0.20, -0.10),
        (0.20, 0.10, 0.05),
    )
    candidates: list[_Candidate] = []
    for index, point in enumerate(points):
        frame = 10 + index % 2
        cluster = hashlib.sha256(
            f"cluster:{object_id}:{frame}:{index}".encode()
        ).hexdigest()
        group = hashlib.sha256(
            f"group:{object_id}:{episode_id}:{frame}:{index}".encode()
        ).hexdigest()
        candidates.append(
            _Candidate(
                job_id=job_id,
                camera_id=camera_id,
                window_id=f"window-{camera_id}",
                frame=frame,
                row=index,
                column=index + 1,
                point_world_m=np.asarray(point, dtype=np.float64),
                camera_center_world_m=camera_center,
                spatial_cluster_id=cluster,
                correlation_group_id=group,
                support_digest=hashlib.sha256(f"support:{job_id}".encode()).hexdigest(),
            )
        )
    sources = {
        f"prediction/{job_id}.json": "3" * 64,
        f"metric/{job_id}.npz": "4" * 64,
        f"metric-calibration/{job_id}.json": "5" * 64,
        f"support/{job_id}.mask": "6" * 64,
    }
    details = {
        "camera_id": camera_id,
        "calibration_id": "7" * 64,
        "run_spec_sha256": "8" * 64,
        "window_factor_counts": {f"window-{camera_id}": len(candidates)},
        "dropped_by_camera_window_cap": 0,
    }
    return candidates, 0, sources, details


def _materialize_arguments(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "metric_batch_root": fixture["metric_batch"],
        "prediction_root": fixture["prediction_root"],
        "production_result_path": fixture["production_path"],
        "selection_path": fixture["selection_path"],
        "visual_provider_spec_path": fixture["provider_path"],
        "metric_policy_path": fixture["metric_policy_path"],
        "camera_policy_path": fixture["camera_policy_path"],
        "v4_policy_path": fixture["v4_policy_path"],
        "materializer_policy_path": fixture["materializer_policy_path"],
        "implementation_revision": "d" * 40,
        "output_directory": output,
    }


def test_source_chain_and_atomic_materializer_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _source_fixture(tmp_path)
    monkeypatch.setattr(
        materializer,
        "_collect_stream_candidates",
        _fake_collect_stream_candidates,
    )
    output = tmp_path / "materialized"
    result = materializer.materialize_manifest(
        **_materialize_arguments(fixture, output)
    )
    assert result["case_count"] == 10
    assert result["metric_batch_result_id"] == fixture["metric_result"]["result_id"]
    assert result["production_result_id"] == fixture["production"]["result_id"]
    assert (output / "manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert len(list((output / "cases").glob("*/descriptor.json"))) == 10
    assert len(list((output / "cases").glob("*/arrays.npz"))) == 10
    assert not list(tmp_path.glob(".materialized.tmp-*"))

    cli_output = tmp_path / "materialized-cli"
    exit_code = materializer.main(
        [
            "--metric-batch-root",
            str(fixture["metric_batch"]),
            "--prediction-root",
            str(fixture["prediction_root"]),
            "--production-result",
            str(fixture["production_path"]),
            "--selection",
            str(fixture["selection_path"]),
            "--visual-provider-spec",
            str(fixture["provider_path"]),
            "--metric-policy",
            str(fixture["metric_policy_path"]),
            "--camera-policy",
            str(fixture["camera_policy_path"]),
            "--v4-policy",
            str(fixture["v4_policy_path"]),
            "--materializer-policy",
            str(fixture["materializer_policy_path"]),
            "--implementation-revision",
            "e" * 40,
            "--output-dir",
            str(cli_output),
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["case_count"] == 10

    with pytest.raises(ValueError, match="output already exists"):
        materializer.materialize_manifest(**_materialize_arguments(fixture, output))

    failed_output = tmp_path / "materialized-failure"
    with monkeypatch.context() as context:
        context.setattr(
            materializer,
            "_build_object_batch",
            lambda **_: (_ for _ in ()).throw(RuntimeError("fixture failure")),
        )
        with pytest.raises(RuntimeError, match="fixture failure"):
            materializer.materialize_manifest(
                **_materialize_arguments(fixture, failed_output)
            )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".materialized-failure.tmp-*"))


def test_candidate_collection_uses_metric_geometry_and_support_only(
    tmp_path: Path,
) -> None:
    policy = json.loads(MATERIALIZER_POLICY.read_text(encoding="utf-8"))
    prediction_path = _write_prediction_bundle(
        tmp_path,
        motioncrafter_revision=str(policy["motioncrafter_revision"]),
    )
    frames = np.asarray([10, 11], dtype=np.int64)
    points = np.zeros((2, 2, 3, 3), dtype=np.float64)
    points[0] = np.asarray(
        [
            [[0.001, 0.001, 0.10], [0.002, 0.002, 0.10], [0.05, 0.0, 0.10]],
            [[0.10, 0.05, 0.10], [0.15, 0.0, 0.10], [0.20, 0.0, 0.10]],
        ]
    )
    points[1] = points[0] + np.asarray([0.0, 0.0, 0.01])
    valid = np.ones((2, 2, 3), dtype=np.bool_)
    metric_path = tmp_path / "metric.npz"
    np.savez_compressed(
        metric_path,
        frame_indices=frames,
        points_world_m=points,
        valid_mask=valid,
    )
    calibration_path = tmp_path / "calibration.json"
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [0.0, 0.0, -1.0]
    _write_json(
        calibration_path,
        {
            "schema": "bayesian-phystwin.deform360-robot-metric-calibration",
            "schema_version": 1,
            "object_id": "object-00",
            "episode_id": 0,
            "camera_id": "camera-a",
            "camera_to_world": camera_to_world.tolist(),
            "calibration_id": "9" * 64,
        },
    )
    job_id = hashlib.sha256(b"candidate-job").hexdigest()
    candidates, dropped, sources, metadata = _collect_stream_candidates(
        job_id=job_id,
        camera_id="camera-a",
        causal_range=(10, 12),
        prediction_manifest_path=prediction_path,
        metric_prefix_path=metric_path,
        metric_calibration_path=calibration_path,
        object_id="object-00",
        episode_id=0,
        policy=policy,
    )
    assert candidates
    assert dropped == 0
    assert len({item.spatial_cluster_id for item in candidates}) < int(np.sum(valid))
    assert metadata["calibration_id"] == "9" * 64
    assert metadata["dropped_by_camera_window_cap"] == 0
    assert f"prediction/{job_id}.json" in sources
    assert all(np.all(np.isfinite(item.point_world_m)) for item in candidates)

    values = [
        ((index, f"cluster-{index}", index, index), np.asarray([float(index)]))
        for index in range(6)
    ]
    assert _deterministic_select(values, maximum=10, seed="seed") == values
    selected = _deterministic_select(values, maximum=2, seed="seed")
    assert len(selected) == 2
    assert selected == _deterministic_select(
        list(reversed(values)), maximum=2, seed="seed"
    )


def test_bounded_npz_error_and_version_paths(tmp_path: Path) -> None:
    bad_archive = tmp_path / "bad.npz"
    bad_archive.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="cannot inspect NPZ"):
        _zip_members(bad_archive)

    invalid_member = tmp_path / "invalid-member.npz"
    with zipfile.ZipFile(invalid_member, "w") as archive:
        archive.writestr("value.npy", b"not an NPY member")
    with pytest.raises(ValueError, match="cannot load NPZ member"):
        _load_npy_member(
            invalid_member,
            "value.npy",
            maximum_uncompressed_bytes=1024,
        )

    with pytest.raises(ValueError, match="truncated NPY member"):
        _read_exact(io.BytesIO(b"x"), 2)

    version_two = io.BytesIO(
        _npy_bytes(np.asarray([1.0], dtype=np.float64), version=(2, 0))
    )
    assert np.lib.format.read_magic(version_two) == (2, 0)
    shape, fortran_order, dtype = _read_npy_header(version_two, (2, 0))
    assert shape == (1,)
    assert fortran_order is False
    assert np.dtype(dtype) == np.dtype(np.float64)

    with pytest.raises(ValueError, match="unsupported metric NPY format"):
        _read_npy_header(io.BytesIO(), (3, 0))
