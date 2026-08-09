from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import test_deform360_prob4d_metric_batch as batch_test

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/admit_deform360_prob4d_metric_support.py"
SPEC = importlib.util.spec_from_file_location(
    "deform360_prob4d_metric_support_admission", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
support_admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support_admission
SPEC.loader.exec_module(support_admission)

admit_deform360_prob4d_metric_support = (
    support_admission.admit_deform360_prob4d_metric_support
)
validate_deform360_prob4d_metric_support_admission = (
    support_admission.validate_deform360_prob4d_metric_support_admission
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_camera(arguments: dict[str, Any], *, object_id: str, camera_id: str) -> None:
    production_path = Path(arguments["production_result_path"])
    production_root = Path(arguments["production_root"])
    prediction_root = Path(arguments["prediction_root"])
    production = json.loads(production_path.read_text(encoding="utf-8"))
    object_index = int(object_id.rsplit("-", 1)[1])
    episode_id = object_index
    stratum = "sheet" if object_index == 0 else "volumetric"
    job_id = batch_test._job_id(object_id, camera_id)
    relative = Path("objects") / object_id / f"episode_{episode_id:04d}" / camera_id
    prediction_manifest = prediction_root / relative / "prediction.json"
    _write_json(prediction_manifest, {"job_id": job_id})
    seal = {
        "job_id": job_id,
        "object_id": object_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "camera_id": camera_id,
        "admission_id": batch_test.ADMISSION_ID,
        "implementation_revision": batch_test.IMPLEMENTATION_REVISION,
        "provider_revision": batch_test.PROB4D_REVISION,
        "motioncrafter_revision": batch_test.MOTIONCRAFTER_REVISION,
        "visual_provider_lock_id": batch_test.VISUAL_LOCK_ID,
        "model_set_id": batch_test.MODEL_SET_ID,
        "output_relative_directory": relative.as_posix(),
        "causal_prefix_frame_range_half_open": [0, 58],
        "prediction_manifest": {
            "path": "prediction.json",
            "sha256": _sha256(prediction_manifest),
            "byte_count": prediction_manifest.stat().st_size,
        },
    }
    seal_path = production_root / relative / "prediction-seal.json"
    _write_json(seal_path, seal)
    production["jobs"].append(
        {
            "job_id": job_id,
            "object_id": object_id,
            "camera_id": camera_id,
            "status": "succeeded",
            "receipt": batch_test._record(seal_path, root=production_root),
        }
    )
    production["jobs"].sort(
        key=lambda row: (row["object_id"], row["camera_id"], row["job_id"])
    )
    production["camera_view_count"] += 1
    production["succeeded_job_count"] += 1
    _write_json(production_path, production)


def _gate_lock(tmp_path: Path, *, minimum: int = 2) -> Path:
    identity = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-gate-v1",
        "schema_version": 1,
        "semantics": "test-object-balanced-source-gate",
        "protocol_id": batch_test.PROTOCOL_ID,
        "cohort": {
            "exact_object_count": 2,
            "exact_stratum_counts": {"sheet": 1, "volumetric": 1},
            "minimum_metric_streams_per_object": minimum,
            "minimum_point_clusters_per_object": 1,
            "minimum_gauge_rows_per_object": 1,
        },
        "thresholds": {},
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "future_frames_used": False,
            "human_approval_required": False,
            "new_measurements_required": False,
            "replacement_allowed": False,
            "target_outcomes_used": False,
        },
        "claim_boundary": "test-only gate lock",
    }
    path = tmp_path / "source-gate-lock.json"
    _write_json(path, {**identity, "artifact_id": content_id(identity)})
    return path


def _install_support_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_test._install_contract_stubs(monkeypatch)
    monkeypatch.setattr(
        support_admission,
        "_metric_batch_module",
        lambda: batch_test.metric_batch,
    )
    monkeypatch.setattr(
        support_admission,
        "validate_deform360_calibration_visual_production_result",
        lambda value: value,
    )
    monkeypatch.setattr(
        support_admission,
        "validate_deform360_robot_metric_prefix",
        lambda directory: json.loads(
            (Path(directory) / "metric-prefix.json").read_text(encoding="utf-8")
        ),
    )


def _admission_arguments(
    batch_arguments: dict[str, Any], *, gate_lock: Path
) -> dict[str, Any]:
    return {
        "metric_batch_root": batch_arguments["output_directory"],
        "production_result_path": batch_arguments["production_result_path"],
        "production_root": batch_arguments["production_root"],
        "prediction_root": batch_arguments["prediction_root"],
        "selection_path": batch_arguments["selection_path"],
        "visual_provider_spec_path": batch_arguments[
            "visual_provider_spec_path"
        ],
        "metric_prior_policy_path": batch_arguments["metric_prior_policy_path"],
        "source_gate_lock_path": gate_lock,
        "processing_revision": batch_test.PROCESSING_REVISION,
        "implementation_revision": "a" * 40,
        "output_directory": Path(batch_arguments["output_directory"]).parent
        / "support-admission",
    }


def test_retained_support_negative_is_admitted_when_object_minimum_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = batch_test._fixture(tmp_path)
    _add_camera(arguments, object_id="object-0", camera_id="camera-2")
    _install_support_stubs(monkeypatch)
    batch_test._install_metric_stub(
        monkeypatch,
        failure=lambda object_id, camera_id: (
            ValueError(batch_test.SUPPORT_NEGATIVE_DETAIL)
            if (object_id, camera_id) == ("object-0", "camera-2")
            else None
        ),
    )
    batch = batch_test.materialize_deform360_prob4d_metric_batch(**arguments)
    assert batch["status"] == "support-negatives-retained"
    assert batch["plan_emitted"] is False

    result = admit_deform360_prob4d_metric_support(
        **_admission_arguments(arguments, gate_lock=_gate_lock(tmp_path))
    )
    output = Path(arguments["output_directory"]).parent / "support-admission"
    plan = json.loads((output / "metric-prefix-plan.json").read_text())

    assert result["status"] == "admitted-with-retained-support-negatives"
    assert result["admitted_stream_count"] == 5
    assert result["supported_stream_count"] == 4
    assert result["support_negative_stream_count"] == 1
    assert result["technical_failure_stream_count"] == 0
    assert result["plan_emitted"] is True
    assert sum(len(case["streams"]) for case in plan["cases"]) == 4
    assert {row["supported_stream_count"] for row in result["support_by_object"]} == {
        2
    }
    assert result["information_boundary"]["replacement_allowed"] is False
    validate_deform360_prob4d_metric_support_admission(output)


def test_object_below_frozen_minimum_is_retained_without_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = batch_test._fixture(tmp_path)
    _install_support_stubs(monkeypatch)
    batch_test._install_metric_stub(
        monkeypatch,
        failure=lambda object_id, camera_id: (
            ValueError(batch_test.SUPPORT_NEGATIVE_DETAIL)
            if (object_id, camera_id) == ("object-0", "camera-1")
            else None
        ),
    )
    batch_test.materialize_deform360_prob4d_metric_batch(**arguments)

    result = admit_deform360_prob4d_metric_support(
        **_admission_arguments(arguments, gate_lock=_gate_lock(tmp_path))
    )
    output = Path(arguments["output_directory"]).parent / "support-admission"

    assert result["status"] == "insufficient-multiview-support"
    assert result["plan_emitted"] is False
    assert result["plan_id"] is None
    assert not (output / "metric-prefix-plan.json").exists()
    assert any(
        row["minimum_support_passed"] is False
        for row in result["support_by_object"]
    )
    validate_deform360_prob4d_metric_support_admission(output)


def test_technical_failure_cannot_be_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = batch_test._fixture(tmp_path)
    _install_support_stubs(monkeypatch)
    batch_test._install_metric_stub(
        monkeypatch,
        failure=lambda object_id, camera_id: (
            ValueError("broken metric source")
            if (object_id, camera_id) == ("object-0", "camera-1")
            else None
        ),
    )
    batch_test.materialize_deform360_prob4d_metric_batch(**arguments)

    result = admit_deform360_prob4d_metric_support(
        **_admission_arguments(arguments, gate_lock=_gate_lock(tmp_path))
    )

    assert result["status"] == "technical-failures-retained"
    assert result["technical_failure_stream_count"] == 1
    assert result["plan_emitted"] is False


def test_support_admission_is_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = batch_test._fixture(tmp_path)
    _install_support_stubs(monkeypatch)
    batch_test._install_metric_stub(monkeypatch)
    batch_test.materialize_deform360_prob4d_metric_batch(**arguments)
    admission = _admission_arguments(arguments, gate_lock=_gate_lock(tmp_path))

    admit_deform360_prob4d_metric_support(**admission)
    with pytest.raises(ValueError, match="already exists"):
        admit_deform360_prob4d_metric_support(**admission)


def test_validator_rejects_rehashed_support_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = batch_test._fixture(tmp_path)
    _install_support_stubs(monkeypatch)
    batch_test._install_metric_stub(monkeypatch)
    batch_test.materialize_deform360_prob4d_metric_batch(**arguments)
    admission = _admission_arguments(arguments, gate_lock=_gate_lock(tmp_path))
    result = admit_deform360_prob4d_metric_support(**admission)
    output = Path(admission["output_directory"])
    result_path = output / support_admission.SUPPORT_ADMISSION_RESULT_FILENAME
    value = json.loads(result_path.read_text())
    value["minimum_supported_streams_per_object"] = 3
    identity = dict(value)
    identity.pop("result_id")
    value["result_id"] = content_id(identity)
    _write_json(result_path, value)
    checksums = []
    for path in sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    ):
        checksums.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n")
    (output / "SHA256SUMS").write_text("".join(checksums), encoding="ascii")

    with pytest.raises(ValueError, match="object support minimum changed"):
        validate_deform360_prob4d_metric_support_admission(output)
    assert result["minimum_supported_streams_per_object"] == 2
