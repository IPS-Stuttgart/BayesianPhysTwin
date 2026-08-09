from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    POINT_CLUSTER_SEMANTICS,
    SAMPLE_SCHEMA,
    SAMPLE_SEMANTICS,
    SAMPLE_VERSION,
    collapse_point_correlation_clusters,
    fit_and_publish_deform360_prob4d_source_calibration,
    fit_object_balanced_gauge_inflation,
    load_deform360_prob4d_calibration_samples,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    ROOT / "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
PROVIDER = (
    ROOT
    / "protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json"
)
METRIC_POLICY = (
    ROOT
    / "protocols/locks/deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    tmp_path: Path,
    *,
    mutate_manifest=None,
    mutate_arrays=None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    locked = selection["selection"]["calibration"]
    chosen = [
        *[record for record in locked if record["stratum"] == "sheet"][:4],
        *[record for record in locked if record["stratum"] == "volumetric"][:4],
    ]
    provider = json.loads(PROVIDER.read_text(encoding="utf-8"))
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    metric_root = tmp_path / "metric-prefix"
    metric_root.mkdir()
    source_artifacts: dict[str, str] = {}
    cases = []
    for case_index, record in enumerate(chosen):
        case_id = f"{record['object_id']}-ep{record['episode_id']:04d}"
        metric_source_path = metric_root / f"{case_id}-source.json"
        metric_source_path.write_text(
            json.dumps({"case_id": case_id, "kind": "metric-source"}),
            encoding="utf-8",
        )
        metric_calibration_path = metric_root / f"{case_id}-calibration.json"
        metric_calibration_path.write_text(
            json.dumps({"case_id": case_id, "kind": "metric-calibration"}),
            encoding="utf-8",
        )
        source_artifacts[metric_source_path.relative_to(tmp_path).as_posix()] = _sha256(
            metric_source_path
        )
        source_artifacts[metric_calibration_path.relative_to(tmp_path).as_posix()] = (
            _sha256(metric_calibration_path)
        )
        prediction_records = []
        metric_references = []
        for camera_index in range(2):
            relative = f"{case_id}/camera{camera_index}/prediction-manifest.json"
            path = prediction_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"case_id": case_id, "camera": camera_index}),
                encoding="utf-8",
            )
            prediction_records.append(
                {
                    "job_id": hashlib.sha256(relative.encode()).hexdigest(),
                    "camera_id": f"camera{camera_index}",
                    "path": relative,
                    "sha256": _sha256(path),
                    "byte_count": path.stat().st_size,
                }
            )
            metric_references.append(
                {
                    "job_id": hashlib.sha256(relative.encode()).hexdigest(),
                    "camera_id": f"camera{camera_index}",
                    "window_id": f"window-{case_index}-{camera_index}",
                    "frame_id": 10,
                    "coordinate_frame": "deform360-world",
                    "source_kind": "official-deform360-metric-prefix",
                    "source_artifact_sha256": _sha256(metric_source_path),
                    "calibration_artifact_sha256": _sha256(metric_calibration_path),
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "object_id": record["object_id"],
                "episode_id": record["episode_id"],
                "stratum": record["stratum"],
                "causal_frame_range_half_open": [10, 16],
                "prediction_manifests": prediction_records,
                "metric_references": metric_references,
            }
        )
    case_count = len(cases)
    arrays: dict[str, np.ndarray] = {
        "point_errors_m": np.tile([[0.001, 0.002, 0.003]], (2 * case_count, 1)),
        "point_ray_directions": np.tile([[0.0, 0.0, 1.0]], (2 * case_count, 1)),
        "point_parallel_variance_m2": np.full(2 * case_count, 1e-6),
        "point_lateral_variance_m2": np.full(2 * case_count, 2e-6),
        "point_case_index": np.repeat(np.arange(case_count), 2),
        "point_frame_id": np.tile([10, 11], case_count),
        "point_correlation_cluster_index": np.repeat(np.arange(case_count), 2),
        "point_valid": np.ones(2 * case_count, dtype=np.bool_),
        "gauge_errors": np.tile(
            [[0.01, 0.001, 0.0, 0.0, 0.002, 0.0, 0.0]],
            (case_count, 1),
        ),
        "gauge_covariance": np.tile(np.eye(7)[None, :, :] * 1e-4, (case_count, 1, 1)),
        "gauge_case_index": np.arange(case_count),
        "gauge_frame_id": np.full(case_count, 11),
        "anchor_global_from_local": np.zeros((2 * case_count, 7)),
        "anchor_covariance": np.tile(
            np.eye(7)[None, :, :] * 1e-6,
            (2 * case_count, 1, 1),
        ),
        "anchor_prediction_index": np.arange(2 * case_count),
    }
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    arrays_path = tmp_path / "samples.npz"
    np.savez_compressed(arrays_path, **arrays)  # type: ignore[arg-type]
    manifest = {
        "schema": SAMPLE_SCHEMA,
        "schema_version": SAMPLE_VERSION,
        "semantics": SAMPLE_SEMANTICS,
        "protocol_id": selection["protocol_id"],
        "selection_file_sha256": _sha256(SELECTION),
        "visual_provider_spec_file_sha256": _sha256(PROVIDER),
        "metric_prior_policy_file_sha256": _sha256(METRIC_POLICY),
        "dataset_revision": selection["dataset"]["resolved_revision"],
        "prob4d_revision": provider["provider"]["revision"],
        "motioncrafter_revision": provider["motioncrafter"]["revision"],
        "visual_production_result_id": "a" * 64,
        "point_correlation_cluster_semantics": POINT_CLUSTER_SEMANTICS,
        "cases": cases,
        "arrays": {
            "path": arrays_path.name,
            "sha256": _sha256(arrays_path),
            "byte_count": arrays_path.stat().st_size,
        },
        "source_artifacts": source_artifacts,
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
        },
        "claim_boundary": "source-only calibration samples; no confirmation claim",
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest["bundle_id"] = content_id(manifest)
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, prediction_root


def _load(tmp_path: Path, **kwargs):
    manifest, prediction_root = _write_bundle(tmp_path, **kwargs)
    return load_deform360_prob4d_calibration_samples(
        manifest,
        selection_path=SELECTION,
        visual_provider_spec_path=PROVIDER,
        metric_prior_policy_path=METRIC_POLICY,
        prediction_root=prediction_root,
    )


def test_source_sample_bundle_loads_and_is_immutable(tmp_path: Path) -> None:
    samples = _load(tmp_path)

    assert len(samples.object_ids) == 8
    assert len(samples.prediction_manifest_paths) == 8
    assert samples.arrays["point_errors_m"].flags.writeable is False
    with pytest.raises(ValueError):
        samples.arrays["point_errors_m"].setflags(write=True)


def test_source_sample_bundle_rejects_confirmation_object(tmp_path: Path) -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    confirmation = selection["selection"]["confirmation"][0]

    def mutate(manifest: dict[str, object]) -> None:
        cases = cast(list[dict[str, object]], manifest["cases"])
        case = cases[0]
        case["object_id"] = confirmation["object_id"]
        case["episode_id"] = confirmation["episode_id"]
        case["stratum"] = confirmation["stratum"]

    with pytest.raises(ValueError, match="confirmation object"):
        _load(tmp_path, mutate_manifest=mutate)


def test_source_sample_bundle_rejects_future_frame(tmp_path: Path) -> None:
    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["point_frame_id"][0] = 16
        arrays["point_valid"][0] = False

    with pytest.raises(ValueError, match="outside the causal prefix"):
        _load(tmp_path, mutate_arrays=mutate)


def test_source_sample_bundle_rejects_cluster_spanning_objects(tmp_path: Path) -> None:
    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["point_correlation_cluster_index"][2:4] = 0

    with pytest.raises(ValueError, match="cluster spans physical objects"):
        _load(tmp_path, mutate_arrays=mutate)


def test_source_sample_bundle_rejects_camera_anchor_reordering(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        cases = cast(list[dict[str, object]], manifest["cases"])
        references = cast(list[dict[str, object]], cases[0]["metric_references"])
        references.reverse()

    with pytest.raises(ValueError, match="camera/job order"):
        _load(tmp_path, mutate_manifest=mutate)


def test_source_sample_bundle_requires_one_anchor_per_prediction(
    tmp_path: Path,
) -> None:
    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["anchor_global_from_local"] = arrays["anchor_global_from_local"][:-1]

    with pytest.raises(ValueError, match="one metric transform"):
        _load(tmp_path, mutate_arrays=mutate)


def test_duplicate_correlated_block_does_not_change_effective_point_rows() -> None:
    arguments = {
        "errors": np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]),
        "ray_directions": np.asarray([[0.0, 0.0, 1.0]] * 2),
        "parallel_variance": np.asarray([0.5, 0.6]),
        "lateral_variance": np.asarray([0.7, 0.8]),
        "case_index": np.asarray([0, 0]),
        "cluster_index": np.asarray([9, 9]),
        "valid": np.asarray([True, True]),
    }
    original = collapse_point_correlation_clusters(**arguments)
    duplicated = collapse_point_correlation_clusters(
        **{
            name: np.concatenate([value, value], axis=0)
            for name, value in arguments.items()
        }
    )

    for left, right in zip(original[:5], duplicated[:5], strict=True):
        np.testing.assert_allclose(left, right, atol=0.0, rtol=0.0)
    assert original[5]["effective_cluster_count"] == 1
    assert duplicated[5]["effective_cluster_count"] == 1


def test_gauge_fit_gives_equal_mass_to_each_physical_object() -> None:
    errors = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        ]
    )
    covariance = np.tile(np.eye(7)[None, :, :], (2, 1, 1))
    original, _ = fit_object_balanced_gauge_inflation(
        errors,
        covariance,
        np.asarray([0, 1]),
        object_ids=("object-a", "object-b"),
        trim_quantile=1.0,
    )
    duplicate_count = 50
    duplicated, report = fit_object_balanced_gauge_inflation(
        np.concatenate([np.repeat(errors[:1], duplicate_count, axis=0), errors[1:]]),
        np.tile(np.eye(7)[None, :, :], (duplicate_count + 1, 1, 1)),
        np.concatenate([np.zeros(duplicate_count, dtype=int), np.ones(1, dtype=int)]),
        object_ids=("object-a", "object-b"),
        trim_quantile=1.0,
    )

    assert duplicated == original
    assert report["group_count"] == 2
    assert report["groups"][0]["row_count"] == duplicate_count


class _FakeTarget:
    source_repository = "FlorianPfaff/Prob4D"
    motioncrafter_revision = "9cb4e9679f5f34e249945544052464ef46324bc2"
    model_identifier = "prob4d.motioncrafter-model.v2:" + "f" * 64
    image_resolution = (320, 640)
    window_size = 25
    window_overlap = 8
    covariance_cluster_size = 32
    gauge_covariance_method = "frame_spatial_cluster_robust_v1"
    point_covariance_method = "depth_disagreement_anisotropic_v1"

    def descriptor(self) -> dict[str, object]:
        return {
            "manifest_sha256": "0" * 64,
            "source_repository": self.source_repository,
            "motioncrafter_revision": self.motioncrafter_revision,
            "model_identifier": self.model_identifier,
            "image_resolution": list(self.image_resolution),
            "window_size": self.window_size,
            "window_overlap": self.window_overlap,
            "covariance_cluster_size": self.covariance_cluster_size,
            "gauge_covariance_method": self.gauge_covariance_method,
            "point_covariance_method": self.point_covariance_method,
        }


class _FakeArtifact:
    def __init__(self, **values: object) -> None:
        self.values = values
        self.metadata = values.get("metadata", {})
        self.artifact_id = hashlib.sha256(
            json.dumps(values, sort_keys=True, default=list).encode()
        ).hexdigest()


class _FakeAnchor(_FakeArtifact):
    pass


class _FakeReport:
    def to_dict(self) -> dict[str, object]:
        return {"group_count": 8, "aggregation": "fake-test"}


def _save_fake(artifact: _FakeArtifact, path: Path) -> None:
    path.write_text(json.dumps({"artifact_id": artifact.artifact_id}), encoding="utf-8")


def _load_fake(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_id=json.loads(path.read_text(encoding="utf-8"))["artifact_id"]
    )


def _fake_api() -> SimpleNamespace:
    class Structured:
        def __init__(self, **values: object) -> None:
            self.values = values

    class Sim3:
        @classmethod
        def from_vector(cls, value: np.ndarray) -> list[float]:
            return value.tolist()

    def fit_point(*args: object, **kwargs: object):
        del args
        return _FakeArtifact(**kwargs), _FakeReport()

    return SimpleNamespace(
        load_prediction_calibration_target=lambda path: _FakeTarget(),
        StructuredCovariance=Structured,
        DepthDisagreementModel=object,
        fit_group_balanced_point_uncertainty_calibration=fit_point,
        GaugeCovarianceCalibrationV1=_FakeArtifact,
        MetricGaugeAnchor=_FakeAnchor,
        Sim3=Sim3,
        save_gauge_covariance_calibration=_save_fake,
        save_point_uncertainty_calibration=_save_fake,
        save_metric_gauge_anchor=lambda path, anchor: _save_fake(anchor, path),
        load_gauge_covariance_calibration=_load_fake,
        load_point_uncertainty_calibration=_load_fake,
        load_metric_gauge_anchor=_load_fake,
    )


def test_fit_publishes_source_artifacts_without_authorizing_confirmation(
    tmp_path: Path,
) -> None:
    samples = _load(tmp_path / "inputs")
    output = tmp_path / "output"

    result = fit_and_publish_deform360_prob4d_source_calibration(
        samples,
        api=_fake_api(),
        output_directory=output,
    )

    assert result["physical_object_count"] == 8
    assert result["information_boundary"]["confirmation_access_authorized"] is False
    assert result["information_boundary"]["calibration_gate_evaluated"] is False
    assert len(result["artifacts"]["metric_anchors"]) == 16
    assert (output / "SHA256SUMS").is_file()
    assert (output / "source-calibration-result.json").is_file()


def test_bundle_id_rejects_post_lock_edit(tmp_path: Path) -> None:
    manifest, prediction_root = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["claim_boundary"] = "edited after lock"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle_id"):
        load_deform360_prob4d_calibration_samples(
            manifest,
            selection_path=SELECTION,
            visual_provider_spec_path=PROVIDER,
            metric_prior_policy_path=METRIC_POLICY,
            prediction_root=prediction_root,
        )


def test_confirmation_boundary_cannot_be_true(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        boundary = cast(dict[str, object], manifest["information_boundary"])
        boundary["confirmation_payloads_opened"] = True

    with pytest.raises(ValueError, match="information boundary"):
        _load(tmp_path, mutate_manifest=mutate)


def test_information_boundary_must_be_an_object(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        manifest["information_boundary"] = []

    with pytest.raises(ValueError, match="information_boundary must be a JSON object"):
        _load(tmp_path, mutate_manifest=mutate)


def test_prediction_manifest_hash_is_verified(tmp_path: Path) -> None:
    manifest, prediction_root = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    relative = payload["cases"][0]["prediction_manifests"][0]["path"]
    (prediction_root / relative).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="byte_count changed|SHA-256 changed"):
        load_deform360_prob4d_calibration_samples(
            manifest,
            selection_path=SELECTION,
            visual_provider_spec_path=PROVIDER,
            metric_prior_policy_path=METRIC_POLICY,
            prediction_root=prediction_root,
        )


def test_metric_source_artifact_hash_is_verified(tmp_path: Path) -> None:
    manifest, prediction_root = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    relative = next(iter(payload["source_artifacts"]))
    (manifest.parent / relative).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="source artifact changed"):
        load_deform360_prob4d_calibration_samples(
            manifest,
            selection_path=SELECTION,
            visual_provider_spec_path=PROVIDER,
            metric_prior_policy_path=METRIC_POLICY,
            prediction_root=prediction_root,
        )


def test_sample_manifest_mutator_does_not_modify_source_fixture(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def mutate(manifest: dict[str, object]) -> None:
        boundary = cast(dict[str, Any], manifest["information_boundary"])
        captured.update(copy.deepcopy(boundary))

    _load(tmp_path, mutate_manifest=mutate)
    assert captured["target_outcomes_used"] is False
