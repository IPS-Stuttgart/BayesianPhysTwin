from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin import deform360_v6_source_camera_reuse as reuse
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_camera_recovery_v5_2 import (
    AUDIT_INFORMATION_BOUNDARY,
    CAMERA_AUDIT_SCHEMA,
    CAMERA_AUDIT_SEMANTICS,
    CAMERA_AUDIT_VERSION,
    CAMERA_REUSE_POLICY,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5_2 import (
    CAMERA_REUSE_ARTIFACT_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = ROOT / "protocols/amendments/deform360_v6_source_camera_reuse_v1.json"


def _sha(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _content(value: dict[str, Any], field: str) -> dict[str, Any]:
    descriptor = dict(value)
    descriptor.pop(field, None)
    return {**descriptor, field: content_id(descriptor)}


def _base_plan() -> dict[str, Any]:
    objects = []
    for index in range(10):
        object_id = f"object-{index:02d}"
        objects.append(
            {
                "object_id": object_id,
                "reserved_endpoint_camera_ids": ["camera-r0", "camera-r1"],
                "visual_windows": [
                    {"camera_id": "camera-a"},
                    {"camera_id": "camera-b"},
                ],
            }
        )
    return {"plan_id": _sha("base-plan"), "objects": objects}


def _camera_audit(base: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for index, source in enumerate(base["objects"]):
        passing = ["camera-b"] if index == 0 else ["camera-a", "camera-b"]
        rows.append(
            {
                "object_id": source["object_id"],
                "attempted_camera_ids": ["camera-a", "camera-b"],
                "passing_camera_ids": passing,
                "failed_camera_ids": sorted({"camera-a", "camera-b"} - set(passing)),
                "camera_results": [],
            }
        )
    identity = {
        "schema": CAMERA_AUDIT_SCHEMA,
        "schema_version": CAMERA_AUDIT_VERSION,
        "semantics": CAMERA_AUDIT_SEMANTICS,
        "execution_lock_id": _sha("lock"),
        "base_source_plan_id": base["plan_id"],
        "implementation_revision": "1" * 40,
        "objects": rows,
        "information_boundary": dict(AUDIT_INFORMATION_BOUNDARY),
    }
    return {**identity, "audit_id": content_id(identity)}


def _metric_plan() -> dict[str, Any]:
    cases = []
    for index in range(10):
        object_id = f"object-{index:02d}"
        cases.append(
            {
                "object_id": object_id,
                "streams": [
                    {
                        "camera_id": camera,
                        "metric_prefix": {
                            "path": f"{object_id}/{camera}/metric-prefix.npz",
                            "sha256": _sha(f"metric-{object_id}-{camera}"),
                        },
                        "prediction_manifest": {
                            "path": f"{object_id}/{camera}/predictions.json",
                            "sha256": _sha(f"manifest-{object_id}-{camera}"),
                        },
                    }
                    for camera in (
                        "camera-a",
                        "camera-b",
                        "camera-c",
                        "camera-d",
                        "camera-r0",
                    )
                ],
            }
        )
    identity = {
        "schema": reuse.METRIC_PLAN_SCHEMA,
        "schema_version": reuse.METRIC_PLAN_VERSION,
        "semantics": reuse.METRIC_PLAN_SEMANTICS,
        "visual_production_result_id": _sha("visual-production"),
        "cases": cases,
    }
    return {**identity, "plan_id": content_id(identity)}


def test_locked_amendment_is_content_addressed_and_source_only() -> None:
    amendment = reuse.validate_deform360_v6_source_camera_reuse_amendment(
        json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    )

    assert amendment["policy"] == CAMERA_REUSE_POLICY
    assert amendment["information_boundary"]["new_provider_inference_run"] is False
    assert amendment["policy"]["prob4d_used"] is False
    assert amendment["information_boundary"]["development_suffix_opened"] is False
    assert amendment["information_boundary"]["target_outcomes_used"] is False
    assert amendment["information_boundary"]["human_approval_used"] is False
    assert amendment["information_boundary"]["new_measurements_collected"] is False


def test_amendment_binding_rejects_another_source_batch() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    values = {
        "execution_lock_id": amendment["base_execution_lock"]["execution_lock_id"],
        "execution_lock_file_sha256": amendment["base_execution_lock"]["file_sha256"],
        "base_source_plan_id": amendment["base_source_evidence"]["source_plan"][
            "plan_id"
        ],
        "base_source_plan_file_sha256": amendment["base_source_evidence"][
            "source_plan"
        ]["file_sha256"],
        "base_prediction_batch_id": amendment["base_source_evidence"][
            "prediction_batch"
        ]["prediction_batch_id"],
        "base_prediction_batch_file_sha256": amendment["base_source_evidence"][
            "prediction_batch"
        ]["file_sha256"],
        "base_prediction_receipt_id": amendment["base_source_evidence"][
            "prediction_receipt"
        ]["receipt_id"],
        "base_prediction_receipt_file_sha256": amendment["base_source_evidence"][
            "prediction_receipt"
        ]["file_sha256"],
        "metric_prefix_plan_id": amendment["all_camera_sources"]["metric_prefix_plan"][
            "plan_id"
        ],
        "metric_prefix_plan_file_sha256": amendment["all_camera_sources"][
            "metric_prefix_plan"
        ]["file_sha256"],
        "metric_batch_result_id": amendment["all_camera_sources"][
            "metric_batch_result"
        ]["result_id"],
        "metric_batch_result_file_sha256": amendment["all_camera_sources"][
            "metric_batch_result"
        ]["file_sha256"],
        "visual_production_result_id": amendment["all_camera_sources"][
            "visual_production"
        ]["result_id"],
        "visual_production_result_file_sha256": amendment["all_camera_sources"][
            "visual_production"
        ]["file_sha256"],
    }
    reuse.validate_deform360_v6_source_camera_reuse_amendment_bindings(
        amendment, **values
    )
    values["base_prediction_batch_id"] = "f" * 64
    with pytest.raises(ValueError, match="another source execution"):
        reuse.validate_deform360_v6_source_camera_reuse_amendment_bindings(
            amendment, **values
        )


def test_preflight_ranks_only_unattempted_nonreserved_metric_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_plan()
    audit = _camera_audit(base)
    metric_plan = _metric_plan()
    lock = {"execution_lock_id": _sha("lock")}
    monkeypatch.setattr(
        reuse,
        "validate_deform360_joint_sparse_source_prediction_plan_v5",
        lambda value, *, lock: value,
    )
    monkeypatch.setattr(
        reuse,
        "validate_deform360_joint_sparse_camera_audit_v5_2",
        lambda value, *, lock: value,
    )
    monkeypatch.setattr(reuse, "_ordinary_root", lambda value, *, name: tmp_path)
    monkeypatch.setattr(reuse, "_verified_file", lambda *args, **kwargs: tmp_path)

    def support(path: Path) -> dict[str, Any]:
        camera = reuse._ACTIVE_TEST_CAMERA  # type: ignore[attr-defined]
        clusters = 12 if camera == "camera-d" else 10
        return {
            "camera_id": camera,
            "artifact_id": _sha(f"artifact-{camera}"),
            "manifest_file_sha256": "a" * 64,
            "metric_prefix_file_sha256": _sha(f"metric-object-00-{camera}"),
            "max_independent_cluster_count": clusters,
            "frames_with_minimum_clusters": 3,
            "total_projected_point_count": 100,
            "eligible": True,
        }

    original_verified = reuse._verified_file

    def verified(*args: Any, **kwargs: Any) -> Path:
        name = kwargs["name"]
        camera = name.split("/")[1].split(" ")[0]
        reuse._ACTIVE_TEST_CAMERA = camera  # type: ignore[attr-defined]
        return original_verified(*args, **kwargs)

    monkeypatch.setattr(reuse, "_verified_file", verified)
    monkeypatch.setattr(
        reuse, "summarize_deform360_metric_camera_support_v5_2", support
    )
    metric_result_identity = {
        "plan_file": {"path": "metric-prefix-plan.json", "sha256": "c" * 64},
        "production_result_id": metric_plan["visual_production_result_id"],
        "plan_emitted": True,
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "future_frames_used": False,
            "target_outcomes_used": False,
        },
    }
    metric_result = _content(metric_result_identity, "result_id")
    monkeypatch.setattr(
        reuse,
        "_validate_metric_result",
        lambda value, *, metric_plan: value,
    )
    metric_result["plan_file"]["sha256"] = "c" * 64

    preflight = reuse.build_deform360_v6_source_camera_reuse_preflight(
        lock=lock,
        base_source_plan=base,
        base_source_plan_file_sha256="1" * 64,
        base_camera_audit=audit,
        base_camera_audit_file_sha256="2" * 64,
        metric_prefix_plan=metric_plan,
        metric_prefix_plan_file_sha256="c" * 64,
        metric_batch_result=metric_result,
        metric_batch_result_file_sha256="4" * 64,
        metric_files_root=tmp_path,
    )

    first = preflight["objects"][0]
    assert first["recovery_required"] is True
    assert first["ranked_eligible_camera_ids"] == ["camera-d", "camera-c"]
    assert first["selected_reuse_camera_ids"] == ["camera-d", "camera-c"]
    assert all(not row["recovery_required"] for row in preflight["objects"][1:])


def _write_prediction(
    root: Path,
    *,
    object_id: str,
    camera_id: str,
) -> tuple[dict[str, Any], Path]:
    view = root / object_id / camera_id
    view.mkdir(parents=True)
    archive = view / "baseline_disjoint.npz"
    archive.write_bytes(f"archive:{object_id}:{camera_id}".encode())
    digest = _sha(archive.read_bytes())
    manifest = {
        "format_version": 1,
        "disjoint_baseline": archive.name,
        "artifact_integrity": {
            "members": [
                {
                    "path": archive.name,
                    "sha256": digest,
                    "bytes": archive.stat().st_size,
                    "kind": "disjoint_baseline",
                }
            ]
        },
    }
    path = view / "predictions.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "camera_id": camera_id,
        "prediction_manifest": {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha(path.read_bytes()),
        },
        "metric_prefix": {
            "path": f"{object_id}/{camera_id}/metric-prefix.npz",
            "sha256": _sha(f"metric:{object_id}:{camera_id}"),
        },
    }, archive


def test_combined_plan_reuses_integrity_bound_archive_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    predictions = results / "predictions"
    metrics = results / "metrics"
    predictions.mkdir(parents=True)
    metric = metrics / "object-00" / "camera-c" / "metric-prefix.npz"
    metric.parent.mkdir(parents=True)
    metric.write_bytes(b"metric:object-00:camera-c")
    stream, archive = _write_prediction(
        predictions, object_id="object-00", camera_id="camera-c"
    )
    base = {
        "plan_id": _sha("base"),
        "objects": [
            {
                "object_id": "object-00",
                "visual_windows": [
                    {
                        "camera_id": "camera-a",
                        "decoded_uniform": {"path": "old-a", "sha256": "a" * 64},
                        "metric_prefix": {"path": "old-ma", "sha256": "b" * 64},
                    },
                    {
                        "camera_id": "camera-b",
                        "decoded_uniform": {"path": "old-b", "sha256": "c" * 64},
                        "metric_prefix": {"path": "old-mb", "sha256": "d" * 64},
                    },
                ],
            }
        ],
    }
    metric_identity = {
        "schema": reuse.METRIC_PLAN_SCHEMA,
        "schema_version": reuse.METRIC_PLAN_VERSION,
        "semantics": reuse.METRIC_PLAN_SEMANTICS,
        "visual_production_result_id": _sha("production"),
        "cases": [{"object_id": "object-00", "streams": [stream]}],
    }
    metric_plan = {**metric_identity, "plan_id": content_id(metric_identity)}
    preflight_identity = {
        "schema": reuse.PREFLIGHT_SCHEMA,
        "base_source_plan_id": base["plan_id"],
        "metric_prefix_plan_id": metric_plan["plan_id"],
        "policy": dict(CAMERA_REUSE_POLICY),
        "information_boundary": dict(reuse.INFORMATION_BOUNDARY),
        "objects": [
            {
                "object_id": "object-00",
                "selected_reuse_camera_ids": ["camera-c"],
            }
        ],
    }
    preflight = {
        **preflight_identity,
        "preflight_id": content_id(preflight_identity),
    }
    monkeypatch.setattr(
        reuse,
        "validate_deform360_joint_sparse_source_prediction_plan_v5",
        lambda value, *, lock: value,
    )
    monkeypatch.setattr(reuse, "_validate_metric_plan", lambda value: value)
    monkeypatch.setattr(
        reuse,
        "validate_deform360_v6_source_camera_reuse_preflight",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        reuse,
        "build_deform360_joint_sparse_source_prediction_plan_v5",
        lambda **kwargs: {"plan_id": _sha("combined"), "objects": kwargs["objects"]},
    )
    lock = {"execution_lock_id": reuse.EXECUTION_LOCK_ID}

    combined, receipt = reuse.build_deform360_v6_source_camera_reuse_plan(
        lock=lock,
        base_source_plan=base,
        base_camera_audit={},
        preflight=preflight,
        metric_prefix_plan=metric_plan,
        results_root=results,
        prediction_root=predictions,
        metric_files_root=metrics,
        implementation_revision="1" * 40,
    )

    windows = combined["objects"][0]["visual_windows"]
    assert [row["camera_id"] for row in windows] == [
        "camera-a",
        "camera-b",
        "camera-c",
    ]
    assert receipt["reused_camera_count"] == 1
    assert receipt["information_boundary"]["new_provider_inference_run"] is False

    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity changed"):
        reuse.build_deform360_v6_source_camera_reuse_plan(
            lock=lock,
            base_source_plan=base,
            base_camera_audit={},
            preflight=preflight,
            metric_prefix_plan=metric_plan,
            results_root=results,
            prediction_root=predictions,
            metric_files_root=metrics,
            implementation_revision="1" * 40,
        )


def test_reuse_artifact_roster_is_distinct_from_provider_rerun() -> None:
    assert CAMERA_REUSE_ARTIFACT_NAMES == {
        "amendment",
        "base_camera_audit",
        "base_prediction_batch",
        "base_prediction_receipt",
        "base_source_plan",
        "camera_reuse_preflight",
        "camera_reuse_receipt",
        "combined_camera_audit_plan",
        "final_camera_audit",
        "metric_batch_result",
        "metric_prefix_plan",
    }
    assert "recovery_provider_run" not in CAMERA_REUSE_ARTIFACT_NAMES


def test_execution_receipt_binds_exact_base_run_artifacts_and_seals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "8" * 40
    seals = {f"seal-{index:03d}.json": _sha(f"seal-{index}") for index in range(100)}
    plan = {
        "plan_id": _sha("plan"),
        "implementation_revision": revision,
        "camera_recovery": {
            "source_artifacts": {
                name: _sha(name) for name in reuse.CAMERA_REUSE_ARTIFACT_NAMES
            }
        },
    }
    batch = {
        "prediction_batch_id": _sha("batch"),
        "implementation_revision": revision,
    }
    panel_receipt = {
        "receipt_id": _sha("panel-receipt"),
        "implementation_revision": revision,
        "prediction_record_count": 100,
        "source_prediction_seal_file_sha256": seals,
    }
    artifacts = {name: _sha(name) for name in reuse.EXECUTION_ARTIFACT_NAMES}
    artifacts["base_execution_receipt"] = reuse.BASE_EXECUTION_RECEIPT_FILE_SHA256
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    artifacts["execution_lock"] = amendment["base_execution_lock"]["file_sha256"]
    artifacts["visual_production_result"] = amendment["all_camera_sources"][
        "visual_production"
    ]["file_sha256"]
    lock = {"execution_lock_id": amendment["base_execution_lock"]["execution_lock_id"]}

    monkeypatch.setattr(
        reuse,
        "validate_deform360_joint_sparse_source_prediction_plan_v5_2",
        lambda value, *, lock: value,
    )
    monkeypatch.setattr(
        reuse,
        "validate_deform360_joint_sparse_source_prediction_batch_v5",
        lambda value, lock: value,
    )
    monkeypatch.setattr(
        reuse,
        "validate_deform360_joint_sparse_source_prediction_receipt_v5_2",
        lambda value, **kwargs: value,
    )

    receipt = reuse.build_deform360_v6_source_camera_reuse_execution_receipt(
        amendment=amendment,
        lock=lock,
        source_revision=revision,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        base_source_execution=dict(reuse.BASE_SOURCE_EXECUTION),
        artifact_file_sha256=artifacts,
        source_plan=plan,
        prediction_batch=batch,
        source_prediction_receipt=panel_receipt,
        source_prediction_seal_file_sha256=seals,
    )

    assert receipt["base_source_execution"] == reuse.BASE_SOURCE_EXECUTION
    assert receipt["prediction_record_count"] == 100
    assert receipt["source_suffix_access_authorized"] is False
    assert receipt["independent_confirmation_authorized"] is False

    changed_base = dict(reuse.BASE_SOURCE_EXECUTION)
    changed_base["run_id"] += 1
    with pytest.raises(ValueError, match="frozen successful v6 run"):
        reuse.build_deform360_v6_source_camera_reuse_execution_receipt(
            amendment=amendment,
            lock=lock,
            source_revision=revision,
            runner_name="workstation2",
            workflow_run_id=123,
            workflow_run_attempt=1,
            base_source_execution=changed_base,
            artifact_file_sha256=artifacts,
            source_plan=plan,
            prediction_batch=batch,
            source_prediction_receipt=panel_receipt,
            source_prediction_seal_file_sha256=seals,
        )

    changed_seals = dict(seals)
    changed_seals["seal-000.json"] = "f" * 64
    with pytest.raises(ValueError, match="100 observed source seals"):
        reuse.build_deform360_v6_source_camera_reuse_execution_receipt(
            amendment=amendment,
            lock=lock,
            source_revision=revision,
            runner_name="workstation2",
            workflow_run_id=123,
            workflow_run_attempt=1,
            base_source_execution=dict(reuse.BASE_SOURCE_EXECUTION),
            artifact_file_sha256=artifacts,
            source_plan=plan,
            prediction_batch=batch,
            source_prediction_receipt=panel_receipt,
            source_prediction_seal_file_sha256=changed_seals,
        )

    missing_artifact = dict(artifacts)
    missing_artifact.pop("visual_production_result")
    with pytest.raises(ValueError, match="artifact roster changed"):
        reuse.build_deform360_v6_source_camera_reuse_execution_receipt(
            amendment=amendment,
            lock=lock,
            source_revision=revision,
            runner_name="workstation2",
            workflow_run_id=123,
            workflow_run_attempt=1,
            base_source_execution=dict(reuse.BASE_SOURCE_EXECUTION),
            artifact_file_sha256=missing_artifact,
            source_plan=plan,
            prediction_batch=batch,
            source_prediction_receipt=panel_receipt,
            source_prediction_seal_file_sha256=seals,
        )

    changed_lineage = dict(artifacts)
    changed_lineage["metric_prefix_plan"] = "f" * 64
    with pytest.raises(ValueError, match="source-plan lineage"):
        reuse.build_deform360_v6_source_camera_reuse_execution_receipt(
            amendment=amendment,
            lock=lock,
            source_revision=revision,
            runner_name="workstation2",
            workflow_run_id=123,
            workflow_run_attempt=1,
            base_source_execution=dict(reuse.BASE_SOURCE_EXECUTION),
            artifact_file_sha256=changed_lineage,
            source_plan=plan,
            prediction_batch=batch,
            source_prediction_receipt=panel_receipt,
            source_prediction_seal_file_sha256=seals,
        )


def test_technical_failure_receipt_is_one_shot_and_target_closed() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    retained = {"amendment.json": _sha("amendment")}
    receipt = reuse.build_deform360_v6_source_camera_reuse_technical_failure_receipt(
        amendment=amendment,
        lock={
            "execution_lock_id": amendment["base_execution_lock"]["execution_lock_id"]
        },
        source_revision="8" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        base_source_execution=dict(reuse.BASE_SOURCE_EXECUTION),
        terminal_stage="materialize-and-seal-source-panel",
        exit_code=9,
        retained_artifact_file_sha256=retained,
    )

    assert receipt["status"] == "source-camera-reuse-technical-failure-retained"
    assert receipt["exit_code"] == 9
    assert receipt["retained_artifacts"] == retained
    assert receipt["source_suffix_access_authorized"] is False
    assert receipt["independent_confirmation_authorized"] is False
    assert receipt["claim_authorized"] is False

    changed = dict(receipt)
    changed["independent_confirmation_authorized"] = True
    descriptor = dict(changed)
    descriptor.pop("receipt_id")
    changed["receipt_id"] = content_id(descriptor)
    with pytest.raises(ValueError, match="boundary changed"):
        reuse.validate_deform360_v6_source_camera_reuse_technical_failure_receipt(
            changed
        )

    with pytest.raises(ValueError, match="integer >= 1"):
        reuse.build_deform360_v6_source_camera_reuse_technical_failure_receipt(
            amendment=amendment,
            lock={
                "execution_lock_id": amendment["base_execution_lock"][
                    "execution_lock_id"
                ]
            },
            source_revision="8" * 40,
            runner_name="workstation2",
            workflow_run_id=123,
            workflow_run_attempt=1,
            base_source_execution=dict(reuse.BASE_SOURCE_EXECUTION),
            terminal_stage="materialize-and-seal-source-panel",
            exit_code=0,
            retained_artifact_file_sha256=retained,
        )
