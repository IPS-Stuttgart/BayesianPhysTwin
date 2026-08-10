from __future__ import annotations

import hashlib
import io
import json
import shutil
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
    _sha256_file,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_npz_v4 import (
    _load_camera_center,
    _read_exact,
    _read_npy_header,
    _zip_members,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_source_v4 import (
    _validate_sources,
)
from bayesian_phystwin.deform360_joint_sparse_observability_v4 import (
    Deform360JointSparseObservabilityPolicyV4,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_POLICY = (
    ROOT / "protocols/locks/"
    "deform360_official_hub_joint_sparse_geometric_materializer_v4.json"
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal(identity: dict[str, Any], field: str) -> dict[str, Any]:
    return {**identity, field: content_id(identity)}


def _record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _stream_record(index: int) -> dict[str, object]:
    placeholder = {
        "path": "shared.json",
        "sha256": "d" * 64,
        "byte_count": 1,
    }
    return {
        "job_id": _sha(f"job-{index}"),
        "camera_id": f"camera-{index:03d}",
        "prediction_manifest": dict(placeholder),
        "metric_prefix": dict(placeholder),
        "metric_calibration": dict(placeholder),
    }


def _plan_cases(*, full_roster: bool = True) -> list[object]:
    counts = [32, 32, 32, 31, 31, 31, 31, 31, 31, 31]
    if not full_roster:
        counts = [2] * 10
    cases: list[object] = []
    offset = 0
    for index, count in enumerate(counts):
        object_id = f"object-{index:02d}"
        cases.append(
            {
                "case_id": _sha(f"case-{index}"),
                "object_id": object_id,
                "episode_id": index,
                "stratum": "sheet" if index < 5 else "volumetric",
                "causal_frame_range_half_open": [10, 12],
                "streams": [
                    _stream_record(offset + stream_index)
                    for stream_index in range(count)
                ],
            }
        )
        offset += count
    return cases


def _excluded_rows() -> list[dict[str, object]]:
    return [
        {
            "job_id": _sha(f"excluded-{index}"),
            "object_id": f"object-{index % 10:02d}",
            "episode_id": index % 10,
            "stratum": "sheet" if index % 10 < 5 else "volumetric",
            "camera_id": f"excluded-camera-{index:02d}",
            "reason": "released-robot-geometry-outside-fixed-camera-prefix",
        }
        for index in range(11)
    ]


def _selection() -> dict[str, object]:
    calibration = [
        {
            "object_id": f"object-{index:02d}",
            "episode_id": index,
            "stratum": "sheet" if index < 5 else "volumetric",
        }
        for index in range(10)
    ]
    return {
        "protocol_id": "deform360-official-hub-visuotactile-v1",
        "selection_artifact_sha256": json.loads(
            MATERIALIZER_POLICY.read_text(encoding="utf-8")
        )["selection_artifact_sha256"],
        "selection": {"calibration": calibration},
    }


def _source_fixture(
    tmp_path: Path,
    *,
    malformed_first_case: bool = False,
) -> dict[str, Path]:
    metric_root = tmp_path / "metric-batch"
    prediction_root = tmp_path / "predictions"
    metric_root.mkdir()
    prediction_root.mkdir()
    (metric_root / "metrics").mkdir()

    base_policy = json.loads(MATERIALIZER_POLICY.read_text(encoding="utf-8"))
    selection_path = tmp_path / "selection.json"
    provider_path = tmp_path / "provider.json"
    metric_policy_path = tmp_path / "metric-policy.json"
    camera_policy_path = tmp_path / "camera-policy.json"
    v4_policy_path = tmp_path / "v4-policy.json"
    materializer_policy_path = tmp_path / "materializer-policy.json"
    production_path = tmp_path / "production.json"

    _write_json(selection_path, _selection())
    _write_json(
        provider_path,
        {"protocol_id": "deform360-official-hub-visuotactile-v1"},
    )
    _write_json(
        metric_policy_path,
        {"metric_source_kind": base_policy["metric_source_kind"]},
    )
    _write_json(camera_policy_path, {"replacement_allowed": False})
    shutil.copyfile(V4_POLICY, v4_policy_path)

    production_identity = {
        "visual_provider_lock_id": base_policy["visual_provider_lock_id"],
        "provider_revision": base_policy["prob4d_revision"],
        "motioncrafter_revision": base_policy["motioncrafter_revision"],
        "object_count": 10,
        "camera_view_count": 324,
        "completely_succeeded_object_count": 10,
        "succeeded_job_count": 324,
        "technical_failure_job_count": 0,
        "status": "all-jobs-succeeded",
    }
    production = _seal(production_identity, "result_id")
    _write_json(production_path, production)

    cases = _plan_cases(full_roster=True)
    if malformed_first_case:
        cases[0] = "not-a-case"
    plan_identity = {
        "schema": "bayesian-phystwin.deform360-prob4d-metric-prefix-plan",
        "schema_version": 2,
        "semantics": (
            "target-free-robot-visible-integrity-bound-streams-with-"
            "causal-public-metric-prefix-v2"
        ),
        "cases": cases,
        "excluded_streams": _excluded_rows(),
        "selection_file_sha256": _sha256_file(selection_path),
        "visual_provider_spec_file_sha256": _sha256_file(provider_path),
        "metric_prior_policy_file_sha256": _sha256_file(metric_policy_path),
        "camera_eligibility_policy_file_sha256": _sha256_file(camera_policy_path),
        "prob4d_revision": base_policy["prob4d_revision"],
        "motioncrafter_revision": base_policy["motioncrafter_revision"],
        "visual_production_result_id": production["result_id"],
    }
    plan = _seal(plan_identity, "plan_id")
    plan_path = metric_root / "metric-prefix-plan.json"
    _write_json(plan_path, plan)

    metric_identity = {
        "schema": "bayesian-phystwin.deform360-prob4d-metric-batch",
        "schema_version": 2,
        "semantics": (
            "target-free-robot-visible-calibration-streams-released-robot-gauge-v2"
        ),
        "plan_file": _record(plan_path, root=metric_root),
        "implementation_revision": base_policy["metric_batch_implementation_revision"],
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
    metric_result = _seal(metric_identity, "result_id")
    metric_result_path = metric_root / "metric-batch-result.json"
    _write_json(metric_result_path, metric_result)

    policy = dict(base_policy)
    policy["production_result_id"] = production["result_id"]
    policy["metric_batch_result_id"] = metric_result["result_id"]
    policy.pop("artifact_id", None)
    policy["artifact_id"] = content_id(policy)
    _write_json(materializer_policy_path, policy)

    files = sorted(
        path
        for path in metric_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (metric_root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(metric_root).as_posix()}\n"
            for path in files
        ),
        encoding="ascii",
    )
    return {
        "metric_batch_root": metric_root,
        "prediction_root": prediction_root,
        "production_result_path": production_path,
        "selection_path": selection_path,
        "visual_provider_spec_path": provider_path,
        "metric_policy_path": metric_policy_path,
        "camera_policy_path": camera_policy_path,
        "v4_policy_path": v4_policy_path,
        "materializer_policy_path": materializer_policy_path,
    }


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _prediction_fixture(root: Path) -> Path:
    window = root / "window.npz"
    with zipfile.ZipFile(window, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("window_id.npy", _npy_bytes(np.asarray("window-0")))
        archive.writestr("frame_indices.npy", _npy_bytes(np.asarray([10, 11])))
        archive.writestr(
            "valid_mask.npy",
            _npy_bytes(np.ones((2, 3, 4), dtype=np.bool_)),
        )
    run_spec = {"seed": 1, "support_only": True}
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
        "motioncrafter_commit": json.loads(
            MATERIALIZER_POLICY.read_text(encoding="utf-8")
        )["motioncrafter_revision"],
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
                    "sha256": _sha256_file(window),
                    "bytes": window.stat().st_size,
                    "kind": "independently_decoded_overlap_window",
                }
            ],
        },
    }
    path = root / "prediction.json"
    _write_json(path, manifest)
    return path


def _metric_fixture(root: Path) -> tuple[Path, Path]:
    frames = np.asarray([10, 11], dtype=np.int64)
    points = np.zeros((2, 3, 4, 3), dtype=np.float64)
    for frame in range(2):
        for row in range(3):
            for column in range(4):
                points[frame, row, column] = (
                    0.03 * column,
                    0.03 * row,
                    0.02 * frame,
                )
    metric = root / "metric.npz"
    np.savez_compressed(
        metric,
        frame_indices=frames,
        points_world_m=points,
        valid_mask=np.ones((2, 3, 4), dtype=np.bool_),
    )
    calibration = root / "calibration.json"
    value = {
        "schema": "bayesian-phystwin.deform360-robot-metric-calibration",
        "schema_version": 1,
        "object_id": "object",
        "episode_id": 0,
        "camera_id": "camera",
        "camera_to_world": np.eye(4).tolist(),
        "calibration_id": _sha("calibration"),
    }
    _write_json(calibration, value)
    return metric, calibration


def test_source_chain_validates_complete_313_stream_roster(tmp_path: Path) -> None:
    paths = _source_fixture(tmp_path)
    metric, plan, production, v4_policy, policy = _validate_sources(**paths)
    assert metric["supported_stream_count"] == 313
    assert sum(len(case["streams"]) for case in plan["cases"]) == 313
    assert len(plan["excluded_streams"]) == 11
    assert production["succeeded_job_count"] == 324
    assert v4_policy.policy_id == json.loads(V4_POLICY.read_text())["policy_id"]
    assert policy["metric_batch_result_id"] == metric["result_id"]


def test_source_chain_rejects_nonmapping_case(tmp_path: Path) -> None:
    paths = _source_fixture(tmp_path, malformed_first_case=True)
    with pytest.raises(ValueError, match="metric plan case 0 changed"):
        _validate_sources(**paths)


def test_candidate_collection_uses_support_only_and_caps_rows(
    tmp_path: Path,
) -> None:
    prediction = _prediction_fixture(tmp_path)
    metric, calibration = _metric_fixture(tmp_path)
    policy = json.loads(MATERIALIZER_POLICY.read_text(encoding="utf-8"))
    policy["maximum_factors_per_camera_window"] = 3
    policy["world_voxel_size_m"] = 0.005
    candidates, dropped, sources, metadata = _collect_stream_candidates(
        job_id=_sha("job"),
        camera_id="camera",
        causal_range=(10, 12),
        prediction_manifest_path=prediction,
        metric_prefix_path=metric,
        metric_calibration_path=calibration,
        object_id="object",
        episode_id=0,
        policy=policy,
    )
    assert len(candidates) == 3
    assert dropped == 21
    assert len({item.spatial_cluster_id for item in candidates}) == 3
    assert metadata["window_factor_counts"] == {"window-0": 3}
    assert metadata["dropped_by_camera_window_cap"] == 21
    assert set(sources) == {
        f"prediction/{_sha('job')}.json",
        f"metric/{_sha('job')}.npz",
        f"metric-calibration/{_sha('job')}.json",
        f"support/{_sha('job')}/{_sha('window-0')}.mask",
    }
    values = [((0, "a", 0, index), np.zeros(3)) for index in range(2)]
    assert _deterministic_select(values, maximum=3, seed="seed") == values


def test_npz_helpers_fail_closed_on_malformed_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported metric NPY"):
        _read_npy_header(io.BytesIO(), (3, 0))
    with pytest.raises(ValueError, match="truncated NPY"):
        _read_exact(io.BytesIO(b"x"), 2)
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not-a-zip")
    with pytest.raises(ValueError, match="cannot inspect NPZ"):
        _zip_members(bad)

    duplicate = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("same.npy", b"a")
        with pytest.warns(UserWarning):
            archive.writestr("same.npy", b"b")
    with pytest.raises(ValueError, match="repeats members"):
        _zip_members(duplicate)

    directory = tmp_path / "directory.npz"
    with zipfile.ZipFile(directory, "w") as archive:
        archive.writestr("folder/", b"")
    with pytest.raises(ValueError, match="contains a directory"):
        _zip_members(directory)

    _, calibration = _metric_fixture(tmp_path)
    center, calibration_id = _load_camera_center(
        calibration,
        object_id="object",
        episode_id=0,
        camera_id="camera",
    )
    assert np.array_equal(center, np.zeros(3))
    assert calibration_id == _sha("calibration")
    value = json.loads(calibration.read_text(encoding="utf-8"))
    value["camera_to_world"][0][0] = 2.0
    _write_json(calibration, value)
    with pytest.raises(ValueError, match="camera rotation changed"):
        _load_camera_center(
            calibration,
            object_id="object",
            episode_id=0,
            camera_id="camera",
        )


def _materializer_plan() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    job_index = 0
    for object_index in range(10):
        streams = []
        for camera_index in range(2):
            streams.append(_stream_record(job_index))
            streams[-1]["camera_id"] = f"camera-{camera_index}"
            job_index += 1
        cases.append(
            {
                "case_id": _sha(f"materializer-case-{object_index}"),
                "object_id": f"object-{object_index:02d}",
                "episode_id": object_index,
                "stratum": "sheet" if object_index < 5 else "volumetric",
                "causal_frame_range_half_open": [10, 12],
                "streams": streams,
            }
        )
    return {"cases": cases, "excluded_streams": _excluded_rows()}


def _fake_candidates(
    *,
    job_id: str,
    camera_id: str,
    object_id: str,
    episode_id: int,
    **_: object,
) -> tuple[list[_Candidate], int, dict[str, str], dict[str, object]]:
    camera_number = int(camera_id.rsplit("-", 1)[1])
    start = camera_number * 4
    candidates = []
    for local_index in range(4):
        cluster = start + local_index
        candidates.append(
            _Candidate(
                job_id=job_id,
                camera_id=camera_id,
                window_id=f"window-{camera_number}",
                frame=10 + camera_number,
                row=local_index,
                column=local_index,
                point_world_m=np.asarray(
                    (
                        0.04 * (cluster - 3.5),
                        0.03 * ((cluster % 3) - 1),
                        0.02 * (episode_id % 2),
                    ),
                    dtype=np.float64,
                ),
                camera_center_world_m=np.asarray(
                    (-0.5 + camera_number, 0.0, -1.0),
                    dtype=np.float64,
                ),
                spatial_cluster_id=_sha(f"cluster-{object_id}-{cluster}"),
                correlation_group_id=_sha(f"group-{object_id}-{cluster}"),
                support_digest=_sha(f"support-{job_id}"),
            )
        )
    return (
        candidates,
        0,
        {f"stream/{job_id}.json": _sha(f"source-{job_id}")},
        {
            "camera_id": camera_id,
            "window_factor_counts": {f"window-{camera_number}": 4},
        },
    )


def test_materializer_publishes_atomic_ten_object_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metric_root = tmp_path / "metric"
    prediction_root = tmp_path / "prediction"
    (metric_root / "metrics").mkdir(parents=True)
    prediction_root.mkdir()
    paths: dict[str, Path] = {}
    for name in (
        "production",
        "selection",
        "provider",
        "metric-policy",
        "camera-policy",
    ):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"name": name})
        paths[name] = path
    shutil.copyfile(V4_POLICY, tmp_path / "v4-policy.json")
    shutil.copyfile(MATERIALIZER_POLICY, tmp_path / "materializer-policy.json")
    _write_json(metric_root / "metric-batch-result.json", {"result_id": "a" * 64})
    plan = _materializer_plan()
    _write_json(metric_root / "metric-prefix-plan.json", plan)

    policy = json.loads(MATERIALIZER_POLICY.read_text(encoding="utf-8"))
    v4_policy = Deform360JointSparseObservabilityPolicyV4.from_record(
        json.loads(V4_POLICY.read_text(encoding="utf-8"))
    )

    def fake_validate(**_: object):
        return (
            {"result_id": "a" * 64},
            plan,
            {"result_id": "b" * 64},
            v4_policy,
            policy,
        )

    placeholder = tmp_path / "placeholder.bin"
    placeholder.write_bytes(b"x")

    def fake_record(*_: object, **__: object):
        return placeholder, {
            "path": "placeholder.bin",
            "sha256": _sha256_file(placeholder),
            "byte_count": 1,
        }

    monkeypatch.setattr(materializer, "_validate_sources", fake_validate)
    monkeypatch.setattr(materializer, "_verify_record", fake_record)
    monkeypatch.setattr(
        materializer,
        "_collect_stream_candidates",
        _fake_candidates,
    )

    arguments = {
        "metric_batch_root": metric_root,
        "prediction_root": prediction_root,
        "production_result_path": paths["production"],
        "selection_path": paths["selection"],
        "visual_provider_spec_path": paths["provider"],
        "metric_policy_path": paths["metric-policy"],
        "camera_policy_path": paths["camera-policy"],
        "v4_policy_path": tmp_path / "v4-policy.json",
        "materializer_policy_path": tmp_path / "materializer-policy.json",
        "implementation_revision": "e" * 40,
    }
    output = tmp_path / "output"
    result = materializer.materialize_manifest(
        **arguments,
        output_directory=output,
    )
    assert result["case_count"] == 10
    assert len(result["cases"]) == 10
    assert all(case["factor_count"] == 8 for case in result["cases"])
    assert (output / "manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert len(list((output / "cases").glob("*/arrays.npz"))) == 10

    with pytest.raises(ValueError, match="output already exists"):
        materializer.materialize_manifest(
            **arguments,
            output_directory=output,
        )

    cli_output = tmp_path / "cli-output"
    argv = [
        "--metric-batch-root",
        str(metric_root),
        "--prediction-root",
        str(prediction_root),
        "--production-result",
        str(paths["production"]),
        "--selection",
        str(paths["selection"]),
        "--visual-provider-spec",
        str(paths["provider"]),
        "--metric-policy",
        str(paths["metric-policy"]),
        "--camera-policy",
        str(paths["camera-policy"]),
        "--v4-policy",
        str(tmp_path / "v4-policy.json"),
        "--materializer-policy",
        str(tmp_path / "materializer-policy.json"),
        "--implementation-revision",
        "e" * 40,
        "--output-dir",
        str(cli_output),
    ]
    assert materializer.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["case_count"] == 10

    original_builder = materializer._build_object_batch

    def fail_builder(**_: object):
        raise RuntimeError("synthetic publication failure")

    monkeypatch.setattr(materializer, "_build_object_batch", fail_builder)
    failed_output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        materializer.materialize_manifest(
            **arguments,
            output_directory=failed_output,
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(f".{failed_output.name}.tmp-*"))
    monkeypatch.setattr(materializer, "_build_object_batch", original_builder)
