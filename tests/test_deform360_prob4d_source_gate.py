from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    RESULT_SCHEMA,
    RESULT_SEMANTICS,
    RESULT_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/evaluate_deform360_prob4d_source_gate.py"
LOCK = ROOT / ("protocols/locks/deform360_official_hub_prob4d_source_gate_v1.json")
SPEC = importlib.util.spec_from_file_location("deform360_prob4d_source_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
source_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = source_gate
SPEC.loader.exec_module(source_gate)


def _synthetic_samples(*, adversarial_objects: tuple[int, ...] = ()) -> Any:
    generator = np.random.default_rng(20260809)
    object_count = 10
    point_rows = 400
    gauge_rows = 160
    point_errors = []
    point_case = []
    point_clusters = []
    gauge_errors = []
    gauge_case = []
    cases = []
    cluster = 0
    for case_index in range(object_count):
        multiplier = 30.0 if case_index in adversarial_objects else 1.0
        point_errors.append(
            generator.normal(size=(point_rows, 3)) * np.sqrt(4.0) * multiplier
        )
        point_case.extend([case_index] * point_rows)
        point_clusters.extend(range(cluster, cluster + point_rows))
        cluster += point_rows
        gauge_errors.append(
            generator.normal(size=(gauge_rows, 7)) * np.sqrt(3.0) * multiplier
        )
        gauge_case.extend([case_index] * gauge_rows)
        cases.append(
            {
                "case_id": f"case-{case_index}",
                "object_id": f"object-{case_index}",
                "episode_id": case_index,
                "stratum": "sheet" if case_index < 5 else "volumetric",
                "metric_references": [{"camera_id": "a"}, {"camera_id": "b"}],
            }
        )
    point_error_array = np.concatenate(point_errors)
    point_count = len(point_error_array)
    gauge_error_array = np.concatenate(gauge_errors)
    gauge_count = len(gauge_error_array)
    rays = np.zeros((point_count, 3), dtype=np.float64)
    rays[:, 2] = 1.0
    arrays = {
        "point_errors_m": point_error_array,
        "point_ray_directions": rays,
        "point_parallel_variance_m2": np.ones(point_count),
        "point_lateral_variance_m2": np.ones(point_count),
        "point_case_index": np.asarray(point_case, dtype=np.int64),
        "point_frame_id": np.zeros(point_count, dtype=np.int64),
        "point_correlation_cluster_index": np.asarray(point_clusters, dtype=np.int64),
        "point_valid": np.ones(point_count, dtype=np.bool_),
        "gauge_errors": gauge_error_array,
        "gauge_covariance": np.tile(np.eye(7, dtype=np.float64), (gauge_count, 1, 1)),
        "gauge_case_index": np.asarray(gauge_case, dtype=np.int64),
        "gauge_frame_id": np.zeros(gauge_count, dtype=np.int64),
    }
    return SimpleNamespace(
        object_ids=tuple(f"object-{index}" for index in range(object_count)),
        cases=tuple(cases),
        arrays=arrays,
        protocol_id="deform360-official-hub-visuotactile-v1",
        bundle_id="a" * 64,
        manifest_file_sha256="b" * 64,
    )


def _source_calibration_result(root: Path, samples: Any) -> Path:
    root.mkdir()
    descriptor = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_VERSION,
        "semantics": RESULT_SEMANTICS,
        "protocol_id": samples.protocol_id,
        "calibration_sample_bundle_id": samples.bundle_id,
        "calibration_sample_manifest_sha256": samples.manifest_file_sha256,
        "visual_production_result_id": "c" * 64,
        "prob4d_revision": "d" * 40,
        "motioncrafter_revision": "e" * 40,
        "physical_object_count": len(samples.object_ids),
        "stratum_counts": {"sheet": 5, "volumetric": 5},
        "point_effective_cluster_count": 4000,
        "gauge_raw_row_count": 1600,
        "artifacts": {},
        "reports": {},
        "information_boundary": {
            "calibration_payloads_opened": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
            "confirmation_access_authorized": False,
            "calibration_gate_evaluated": False,
        },
        "claim_boundary": "source-only synthetic calibration fixture",
    }
    result = {"result_id": content_id(descriptor), **descriptor}
    path = root / "source-calibration-result.json"
    path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    (root / "SHA256SUMS").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="ascii",
    )
    return path


def test_gate_lock_is_content_addressed_and_requires_no_human() -> None:
    lock = source_gate.load_source_gate_lock(LOCK)

    assert lock["artifact_id"] == (
        "cc96d2cc03a5039479621acc7769690ef9bf8c15c00d418a0f14d3303ce3ebfb"
    )
    assert lock["cohort"]["exact_object_count"] == 10
    assert lock["thresholds"]["minimum_passing_folds"] == 8
    assert lock["information_boundary"]["new_measurements_required"] is False
    assert lock["information_boundary"]["human_approval_required"] is False


def test_well_calibrated_object_transfer_passes_frozen_gate() -> None:
    samples = _synthetic_samples()
    result = source_gate.evaluate_source_gate(
        samples, source_gate.load_source_gate_lock(LOCK)
    )

    assert result["gate_passed"] is True
    assert result["passed_check_count"] == result["total_check_count"]
    assert sum(fold["fold_passed"] for fold in result["folds"]) == 10
    assert 0.8 <= result["aggregate"]["point_after"]["coverage_90"] <= 0.98
    assert result["aggregate"]["gauge_after"]["coverage_90"] >= 0.75


def test_common_source_fit_does_not_hide_nontransferable_objects() -> None:
    samples = _synthetic_samples(adversarial_objects=(0, 1, 2))
    result = source_gate.evaluate_source_gate(
        samples, source_gate.load_source_gate_lock(LOCK)
    )

    assert result["gate_passed"] is False
    assert sum(fold["fold_passed"] for fold in result["folds"]) < 8
    assert result["passed_check_count"] < result["total_check_count"]


def test_published_gate_authorizes_only_after_all_checks_pass(tmp_path: Path) -> None:
    samples = _synthetic_samples()
    source_root = tmp_path / "source-calibration"
    source_result = _source_calibration_result(source_root, samples)
    output = tmp_path / "gate"

    result = source_gate.publish_source_gate_result(
        samples=samples,
        source_calibration_result_path=source_result,
        source_calibration_root=source_root,
        gate_lock_path=LOCK,
        implementation_revision="c" * 40,
        output_directory=output,
    )

    assert result["status"] == "source-gate-passed"
    assert result["gate_passed"] is True
    assert result["confirmation_access_authorized"] is True
    assert result["information_boundary"]["confirmation_payloads_opened"] is False
    assert result["information_boundary"]["human_approval_required"] is False
    source_gate.validate_source_gate_result(output)


def test_gate_rejects_older_source_calibration_contract(tmp_path: Path) -> None:
    samples = _synthetic_samples()
    source_root = tmp_path / "source-calibration"
    source_result = _source_calibration_result(source_root, samples)
    document = json.loads(source_result.read_text(encoding="utf-8"))
    identity = {key: value for key, value in document.items() if key != "result_id"}
    identity["schema_version"] = RESULT_VERSION - 1
    document = {"result_id": content_id(identity), **identity}
    source_result.write_text(
        json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
    )
    (source_root / "SHA256SUMS").write_text(
        f"{hashlib.sha256(source_result.read_bytes()).hexdigest()}  {source_result.name}\n",
        encoding="ascii",
    )

    with np.testing.assert_raises_regex(ValueError, "contract changed"):
        source_gate.publish_source_gate_result(
            samples=samples,
            source_calibration_result_path=source_result,
            source_calibration_root=source_root,
            gate_lock_path=LOCK,
            implementation_revision="c" * 40,
            output_directory=tmp_path / "gate",
        )
