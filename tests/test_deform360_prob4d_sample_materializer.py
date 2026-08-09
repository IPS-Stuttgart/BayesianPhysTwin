from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_calibration_visual_production import (
    DEFORM360_CALIBRATION_VISUAL_PREDICTION_SEAL_SCHEMA,
    DEFORM360_CALIBRATION_VISUAL_PRODUCTION_RESULT_SCHEMA,
    PRODUCTION_INFORMATION_BOUNDARY,
)
from bayesian_phystwin.deform360_prob4d_camera_eligibility import (
    CAMERA_ELIGIBILITY_POLICY_SCHEMA,
    CAMERA_ELIGIBILITY_POLICY_SEMANTICS,
    CAMERA_ELIGIBILITY_POLICY_VERSION,
    SUPPORT_NEGATIVE_REASON,
    VISIBLE_STREAM_PLAN_SEMANTICS,
    VISIBLE_STREAM_PLAN_VERSION,
)
from bayesian_phystwin.deform360_prob4d_sample_materializer import (
    PLAN_SCHEMA,
    PLAN_SEMANTICS,
    PLAN_VERSION,
    Deform360Prob4DMaterializationConfig,
    materialize_deform360_prob4d_calibration_samples,
)

PROB4D_REVISION = "1" * 40
MOTIONCRAFTER_REVISION = "2" * 40
DATASET_REVISION = "3" * 40
PROCESSING_REVISION = "4" * 40
IMPLEMENTATION_REVISION = "5" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _json_record(root: Path, relative: str, value: object) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _record(root, path)


class _Sim3:
    scale = 1.0

    @classmethod
    def from_vector(cls, _value: object) -> _Sim3:
        return cls()

    def as_vector(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float64)

    def transform_points(self, points: object) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)

    def rotate_directions(self, directions: object) -> np.ndarray:
        return np.asarray(directions, dtype=np.float64)

    def inverse(self) -> _Sim3:
        return self

    def compose(self, _other: _Sim3) -> _Sim3:
        return self


class _Window:
    def __init__(
        self,
        window_id: str,
        frame_indices: np.ndarray,
        point_map: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        self.window_id = window_id
        self.frame_indices = frame_indices
        self.point_map = point_map
        self.valid_mask = valid_mask

    @classmethod
    def from_npz(cls, path: Path, *, window_id: str) -> _Window:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                window_id,
                np.asarray(archive["frame_indices"], dtype=np.int64),
                np.asarray(archive["point_map"], dtype=np.float64),
                np.asarray(archive["valid_mask"], dtype=np.bool_),
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.point_map.shape[:-1]


class _DepthModel:
    def predict(self, window: _Window, _evidence: object) -> SimpleNamespace:
        norm = np.linalg.norm(window.point_map, axis=-1, keepdims=True)
        rays = np.divide(
            window.point_map,
            norm,
            out=np.zeros_like(window.point_map),
            where=norm > 0,
        )
        return SimpleNamespace(
            ray_directions=rays,
            parallel_variance=np.full(window.shape, 1e-4),
            lateral_variance=np.full(window.shape, 2e-4),
        )


def _result(reference_id: str | None = None, moving_id: str | None = None):
    alignment_result = SimpleNamespace(
        transform=_Sim3(),
        covariance=np.eye(7, dtype=np.float64) * 1e-5,
        covariance_method="frame_spatial_cluster_robust_v1",
    )
    if reference_id is None:
        return alignment_result
    return SimpleNamespace(
        reference_id=reference_id,
        moving_id=moving_id,
        common_frames=np.asarray([17], dtype=np.int64),
        result=alignment_result,
    )


def _api() -> SimpleNamespace:
    @contextmanager
    def covariance_context(**_kwargs):
        yield SimpleNamespace()

    def align(reference: _Window, moving: _Window, **_kwargs):
        common = np.intersect1d(reference.frame_indices, moving.frame_indices)
        value = _result(reference.window_id, moving.window_id)
        value.common_frames = common
        return value

    return SimpleNamespace(
        PredictionWindow=_Window,
        Sim3=_Sim3,
        DepthDisagreementModel=_DepthModel,
        alignment_covariance_context=covariance_context,
        align_windows=align,
        accumulate_disagreement=lambda windows, _alignments: {
            key: None for key in windows
        },
        estimate_sim3_robust=lambda *_args, **_kwargs: _result(),
        verify_motioncrafter_prediction_manifest=lambda *_args, **_kwargs: {
            "integrity_bound": True,
            "hashes_verified": True,
            "member_count": 4,
        },
    )


def _seal(
    *,
    job_id: str,
    object_id: str,
    episode_id: int,
    stratum: str,
    camera_id: str,
    output_relative_directory: str,
    prediction_manifest: dict[str, object],
) -> dict[str, Any]:
    identity = {
        "schema": DEFORM360_CALIBRATION_VISUAL_PREDICTION_SEAL_SCHEMA,
        "schema_version": 1,
        "semantics": "integrity-bound-causal-prefix-motioncrafter-prediction-v1",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "admission_id": "6" * 64,
        "job_id": job_id,
        "object_id": object_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "camera_id": camera_id,
        "provider_revision": PROB4D_REVISION,
        "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        "visual_provider_lock_id": "7" * 64,
        "model_set_id": "8" * 64,
        "command_id": "9" * 64,
        "source_video": {"path": "video.mp4", "sha256": "a" * 64, "byte_count": 1},
        "source_timestamps": {
            "path": "timestamps.txt",
            "sha256": "b" * 64,
            "byte_count": 1,
        },
        "causal_prefix_frame_range_half_open": [0, 58],
        "reserved_evaluation_frame_range_half_open": [58, 76],
        "prediction_manifest": prediction_manifest,
        "run_spec_sha256": "c" * 64,
        "verified_member_count": 4,
        "output_relative_directory": output_relative_directory,
        "information_boundary": dict(PRODUCTION_INFORMATION_BOUNDARY),
    }
    return {**identity, "seal_id": content_id(identity)}


def _write_source_contracts(
    root: Path,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    calibration = []
    for index in range(8):
        calibration.append(
            {
                "object_id": f"object-{index:02d}",
                "episode_id": index,
                "stratum": "sheet" if index < 4 else "volumetric",
            }
        )
    selection = {
        "protocol_id": "unit-protocol",
        "dataset": {"resolved_revision": DATASET_REVISION},
        "selection": {
            "calibration": calibration,
            "confirmation": [{"object_id": "confirmation", "episode_id": 0}],
        },
    }
    provider = {
        "protocol_id": "unit-protocol",
        "provider": {
            "revision": PROB4D_REVISION,
            "api_version": 2,
            "export_mode": "exploratory",
        },
        "motioncrafter": {
            "revision": MOTIONCRAFTER_REVISION,
            "height": 3,
            "width": 3,
            "window_size": 25,
            "overlap": 8,
        },
    }
    metric_policy = {
        "protocol_id": "unit-protocol",
        "frame_selection": "first retained causal frame",
        "future_frames_used": False,
        "confirmation_payloads_opened": False,
    }
    paths = []
    for name, value in (
        ("selection.json", selection),
        ("provider.json", provider),
        ("metric-policy.json", metric_policy),
    ):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1], paths[2], calibration


def _metric_points() -> np.ndarray:
    rows, columns = np.indices((3, 3))
    base = np.stack((columns * 0.01, rows * 0.01, np.ones((3, 3))), axis=-1).astype(
        np.float64
    )
    return np.repeat(base[None], 58, axis=0)


def _write_fixture(tmp_path: Path) -> dict[str, Any]:
    selection, provider, metric_policy, calibration = _write_source_contracts(tmp_path)
    production_root = tmp_path / "production"
    prediction_root = tmp_path / "predictions"
    metric_root = tmp_path / "metric"
    production_root.mkdir()
    prediction_root.mkdir()
    metric_root.mkdir()
    points = _metric_points()
    valid = np.ones(points.shape[:-1], dtype=np.bool_)
    cases = []
    production_rows = []
    for case in calibration:
        case_id = f"{case['object_id']}-ep{case['episode_id']:04d}"
        streams = []
        for camera_index in range(2):
            camera_id = f"camera-{camera_index}"
            relative_directory = f"{case_id}/{camera_id}"
            prediction_directory = prediction_root / relative_directory
            windows_directory = prediction_directory / "windows"
            windows_directory.mkdir(parents=True)
            windows = []
            for window_index, (start, stop) in enumerate(((0, 25), (17, 42), (33, 58))):
                window_id = f"window-{window_index}"
                window_path = windows_directory / f"{window_id}.npz"
                np.savez_compressed(
                    window_path,
                    frame_indices=np.arange(start, stop),
                    point_map=points[start:stop],
                    valid_mask=valid[start:stop],
                )
                windows.append(
                    {
                        "window_id": window_id,
                        "path": f"windows/{window_id}.npz",
                        "start_frame": start,
                        "stop_frame": stop,
                    }
                )
            manifest_path = prediction_directory / "predictions.json"
            manifest_path.write_text(
                json.dumps({"format_version": 1, "overlap_windows": windows}),
                encoding="utf-8",
            )
            prediction_record = _record(prediction_root, manifest_path)
            sealed_prediction_record = {
                **prediction_record,
                "path": "predictions.json",
            }
            job_id = hashlib.sha256(f"{case_id}:{camera_id}".encode()).hexdigest()
            seal = _seal(
                job_id=job_id,
                object_id=case["object_id"],
                episode_id=case["episode_id"],
                stratum=case["stratum"],
                camera_id=camera_id,
                output_relative_directory=relative_directory,
                prediction_manifest=sealed_prediction_record,
            )
            receipt = _json_record(
                production_root,
                f"{relative_directory}/prediction-seal.json",
                seal,
            )
            production_rows.append(
                {
                    "job_id": job_id,
                    "object_id": case["object_id"],
                    "camera_id": camera_id,
                    "status": "succeeded",
                    "receipt": receipt,
                }
            )
            metric_directory = metric_root / relative_directory
            metric_directory.mkdir(parents=True)
            metric_path = metric_directory / "metric-prefix.npz"
            np.savez_compressed(
                metric_path,
                frame_indices=np.arange(58),
                points_world_m=points,
                valid_mask=valid,
            )
            calibration_path = metric_directory / "calibration.json"
            calibration_path.write_text(
                json.dumps({"camera_id": camera_id}), encoding="utf-8"
            )
            streams.append(
                {
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "prediction_manifest": prediction_record,
                    "metric_prefix": _record(metric_root, metric_path),
                    "metric_calibration": _record(metric_root, calibration_path),
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "object_id": case["object_id"],
                "episode_id": case["episode_id"],
                "stratum": case["stratum"],
                "causal_frame_range_half_open": [0, 58],
                "streams": streams,
            }
        )
    production_identity = {
        "schema": DEFORM360_CALIBRATION_VISUAL_PRODUCTION_RESULT_SCHEMA,
        "schema_version": 1,
        "semantics": "complete-admitted-calibration-view-accounting-v1",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "admission_id": "6" * 64,
        "visual_provider_lock_id": "7" * 64,
        "provider_revision": PROB4D_REVISION,
        "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        "model_set_id": "8" * 64,
        "object_count": 8,
        "camera_view_count": 16,
        "succeeded_job_count": 16,
        "technical_failure_job_count": 0,
        "completely_succeeded_object_count": 8,
        "status": "all-jobs-succeeded",
        "jobs": sorted(
            production_rows, key=lambda row: (row["object_id"], row["camera_id"])
        ),
        "information_boundary": dict(PRODUCTION_INFORMATION_BOUNDARY),
    }
    production = {
        **production_identity,
        "result_id": content_id(production_identity),
    }
    production_path = production_root / "visual-production-result.json"
    production_path.write_text(json.dumps(production), encoding="utf-8")
    plan_identity = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "semantics": PLAN_SEMANTICS,
        "protocol_id": "unit-protocol",
        "selection_file_sha256": _sha256(selection),
        "visual_provider_spec_file_sha256": _sha256(provider),
        "metric_prior_policy_file_sha256": _sha256(metric_policy),
        "dataset_revision": DATASET_REVISION,
        "processing_revision": PROCESSING_REVISION,
        "prob4d_revision": PROB4D_REVISION,
        "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        "visual_production_result_id": production["result_id"],
        "cases": cases,
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
        },
        "claim_boundary": "unit source-only materialization",
    }
    plan = {**plan_identity, "plan_id": content_id(plan_identity)}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return {
        "plan": plan_path,
        "production": production_path,
        "production_root": production_root,
        "prediction_root": prediction_root,
        "metric_root": metric_root,
        "selection": selection,
        "provider": provider,
        "metric_policy": metric_policy,
    }


def _write_visible_camera_fixture(tmp_path: Path) -> dict[str, Any]:
    fixture = _write_fixture(tmp_path)
    production_path = Path(fixture["production"])
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production_identity = {
        key: value for key, value in production.items() if key != "result_id"
    }

    object_id = "object-00"
    episode_id = 0
    stratum = "sheet"
    camera_id = "camera-2"
    case_id = f"{object_id}-ep{episode_id:04d}"
    relative_directory = f"{case_id}/{camera_id}"
    prediction_directory = Path(fixture["prediction_root"]) / relative_directory
    prediction_directory.mkdir(parents=True)
    prediction_manifest_path = prediction_directory / "predictions.json"
    prediction_manifest_path.write_text(
        json.dumps({"format_version": 1, "overlap_windows": []}),
        encoding="utf-8",
    )
    prediction_record = _record(
        Path(fixture["prediction_root"]), prediction_manifest_path
    )
    sealed_prediction_record = {**prediction_record, "path": "predictions.json"}
    job_id = hashlib.sha256(f"{case_id}:{camera_id}".encode()).hexdigest()
    seal = _seal(
        job_id=job_id,
        object_id=object_id,
        episode_id=episode_id,
        stratum=stratum,
        camera_id=camera_id,
        output_relative_directory=relative_directory,
        prediction_manifest=sealed_prediction_record,
    )
    receipt = _json_record(
        Path(fixture["production_root"]),
        f"{relative_directory}/prediction-seal.json",
        seal,
    )
    production_identity["jobs"].append(
        {
            "job_id": job_id,
            "object_id": object_id,
            "camera_id": camera_id,
            "status": "succeeded",
            "receipt": receipt,
        }
    )
    production_identity["jobs"].sort(
        key=lambda row: (row["object_id"], row["camera_id"])
    )
    production_identity["camera_view_count"] = 17
    production_identity["succeeded_job_count"] = 17
    production = {
        **production_identity,
        "result_id": content_id(production_identity),
    }
    production_path.write_text(json.dumps(production), encoding="utf-8")

    policy_identity = {
        "schema": CAMERA_ELIGIBILITY_POLICY_SCHEMA,
        "schema_version": CAMERA_ELIGIBILITY_POLICY_VERSION,
        "semantics": CAMERA_ELIGIBILITY_POLICY_SEMANTICS,
        "protocol_id": "unit-protocol",
        "eligibility_evidence": "released robot projection over the causal prefix",
        "eligible_status": "supported",
        "support_negative_action": "retain-and-exclude",
        "technical_failure_action": "terminal",
        "allowed_support_negative_reason": SUPPORT_NEGATIVE_REASON,
        "minimum_supported_streams_per_object": 2,
        "minimum_supported_object_count": 8,
        "minimum_supported_stream_fraction": 0.9,
        "camera_images_used_for_eligibility": False,
        "prediction_residuals_used_for_eligibility": False,
        "calibration_outcomes_used_for_eligibility": False,
        "replacement_allowed": False,
        "confirmation_payloads_opened": False,
        "future_frames_used": False,
        "target_outcomes_used": False,
        "human_approval_required": False,
        "new_measurements_required": False,
        "claim_boundary": "unit target-free eligibility policy",
    }
    policy = {**policy_identity, "artifact_id": content_id(policy_identity)}
    policy_path = tmp_path / "camera-eligibility-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    plan_path = Path(fixture["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_identity = {key: value for key, value in plan.items() if key != "plan_id"}
    plan_identity["schema_version"] = VISIBLE_STREAM_PLAN_VERSION
    plan_identity["semantics"] = VISIBLE_STREAM_PLAN_SEMANTICS
    plan_identity["visual_production_result_id"] = production["result_id"]
    plan_identity["camera_eligibility_policy_file_sha256"] = _sha256(policy_path)
    plan_identity["camera_eligibility_policy_id"] = policy["artifact_id"]
    plan_identity["excluded_streams"] = [
        {
            "job_id": job_id,
            "object_id": object_id,
            "episode_id": episode_id,
            "stratum": stratum,
            "camera_id": camera_id,
            "reason": SUPPORT_NEGATIVE_REASON,
        }
    ]
    plan = {**plan_identity, "plan_id": content_id(plan_identity)}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    fixture["camera_eligibility_policy"] = policy_path
    return fixture


def test_materializer_publishes_stream_anchors_and_clustered_causal_rows(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "output"

    result = materialize_deform360_prob4d_calibration_samples(
        plan_path=fixture["plan"],
        production_result_path=fixture["production"],
        production_root=fixture["production_root"],
        prediction_root=fixture["prediction_root"],
        metric_root=fixture["metric_root"],
        selection_path=fixture["selection"],
        visual_provider_spec_path=fixture["provider"],
        metric_prior_policy_path=fixture["metric_policy"],
        expected_processing_revision=PROCESSING_REVISION,
        api=_api(),
        output_directory=output,
        config=Deform360Prob4DMaterializationConfig(
            covariance_cluster_size_pixels=1,
            maximum_metric_fit_correspondences=100,
            maximum_point_rows_per_window=1_000,
            minimum_point_rows_per_window=8,
        ),
    )

    assert len(result["cases"]) == 8
    assert sum(len(case["metric_references"]) for case in result["cases"]) == 16
    assert result["information_boundary"]["confirmation_payloads_opened"] is False
    assert result["information_boundary"]["target_outcomes_used"] is False
    with np.load(output / "samples.npz", allow_pickle=False) as archive:
        assert archive["anchor_global_from_local"].shape == (16, 7)
        assert np.all(archive["point_frame_id"] < 58)
        assert np.all(archive["point_frame_id"] > 0)
        assert len(np.unique(archive["point_correlation_cluster_index"])) == (
            8 * 57 * 9
        )
        assert len(archive["point_correlation_cluster_index"]) > 8 * 57 * 9
    assert (output / "SHA256SUMS").is_file()


def test_materializer_rejects_metric_archive_that_crosses_causal_boundary(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    plan = json.loads(Path(fixture["plan"]).read_text(encoding="utf-8"))
    first = plan["cases"][0]["streams"][0]["metric_prefix"]
    metric_path = Path(fixture["metric_root"]) / first["path"]
    points = _metric_points()
    np.savez_compressed(
        metric_path,
        frame_indices=np.arange(1, 59),
        points_world_m=points,
        valid_mask=np.ones(points.shape[:-1], dtype=np.bool_),
    )
    first["sha256"] = _sha256(metric_path)
    first["byte_count"] = metric_path.stat().st_size
    plan["plan_id"] = content_id(
        {key: value for key, value in plan.items() if key != "plan_id"}
    )
    Path(fixture["plan"]).write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the causal frame range"):
        materialize_deform360_prob4d_calibration_samples(
            plan_path=fixture["plan"],
            production_result_path=fixture["production"],
            production_root=fixture["production_root"],
            prediction_root=fixture["prediction_root"],
            metric_root=fixture["metric_root"],
            selection_path=fixture["selection"],
            visual_provider_spec_path=fixture["provider"],
            metric_prior_policy_path=fixture["metric_policy"],
            expected_processing_revision=PROCESSING_REVISION,
            api=_api(),
            output_directory=tmp_path / "rejected",
            config=Deform360Prob4DMaterializationConfig(
                covariance_cluster_size_pixels=1,
                minimum_point_rows_per_window=8,
            ),
        )


def test_materializer_accepts_visible_camera_plan_with_retained_exclusion(
    tmp_path: Path,
) -> None:
    fixture = _write_visible_camera_fixture(tmp_path)
    output = tmp_path / "visible-output"

    result = materialize_deform360_prob4d_calibration_samples(
        plan_path=fixture["plan"],
        production_result_path=fixture["production"],
        production_root=fixture["production_root"],
        prediction_root=fixture["prediction_root"],
        metric_root=fixture["metric_root"],
        selection_path=fixture["selection"],
        visual_provider_spec_path=fixture["provider"],
        metric_prior_policy_path=fixture["metric_policy"],
        camera_eligibility_policy_path=fixture["camera_eligibility_policy"],
        expected_processing_revision=PROCESSING_REVISION,
        api=_api(),
        output_directory=output,
        config=Deform360Prob4DMaterializationConfig(
            covariance_cluster_size_pixels=1,
            maximum_metric_fit_correspondences=100,
            maximum_point_rows_per_window=1_000,
            minimum_point_rows_per_window=8,
        ),
    )

    assert len(result["cases"]) == 8
    assert sum(len(case["metric_references"]) for case in result["cases"]) == 16
    assert (
        "source-artifacts/camera-eligibility-policy.json" in result["source_artifacts"]
    )
    assert result["information_boundary"]["replacement_allowed"] is False


def test_materializer_rejects_non_visibility_exclusion(tmp_path: Path) -> None:
    fixture = _write_visible_camera_fixture(tmp_path)
    plan_path = Path(fixture["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["excluded_streams"][0]["reason"] = "outcome-dependent-exclusion"
    plan["plan_id"] = content_id(
        {key: value for key, value in plan.items() if key != "plan_id"}
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="target-free visibility negative"):
        materialize_deform360_prob4d_calibration_samples(
            plan_path=fixture["plan"],
            production_result_path=fixture["production"],
            production_root=fixture["production_root"],
            prediction_root=fixture["prediction_root"],
            metric_root=fixture["metric_root"],
            selection_path=fixture["selection"],
            visual_provider_spec_path=fixture["provider"],
            metric_prior_policy_path=fixture["metric_policy"],
            camera_eligibility_policy_path=fixture["camera_eligibility_policy"],
            expected_processing_revision=PROCESSING_REVISION,
            api=_api(),
            output_directory=tmp_path / "rejected-exclusion",
        )
