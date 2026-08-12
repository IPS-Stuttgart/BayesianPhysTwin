from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.deform360_fresh_object_session_candidate_runner_v6_1 as runner
from bayesian_phystwin.deform360_fresh_object_session_candidate_runner_v6_1 import (
    build_deform360_v61_candidate_execution_receipt,
    build_deform360_v61_candidate_panel_receipt,
    build_deform360_v61_candidate_technical_failure_receipt,
    validate_deform360_v61_candidate_execution_receipt,
    validate_deform360_v61_candidate_panel_receipt,
    validate_deform360_v61_candidate_technical_failure_receipt,
)
from bayesian_phystwin.deform360_fresh_object_session_candidate_v6_1 import (
    CANDIDATE_AMENDMENT_FILE_SHA256,
    EXECUTION_LOCK_FILE_SHA256,
    UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
    UPSTREAM_EXECUTION_RECEIPT_ID,
    UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
    UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
    UPSTREAM_SOURCE_PLAN_FILE_SHA256,
)
from bayesian_phystwin.deform360_fresh_object_session_source_v6_1 import (
    UPSTREAM_PREDICTION_BATCH_ID,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _mapping(label: str) -> dict[str, str]:
    return {
        f"{outer:02d}-{target:02d}": _digest(f"{label}/{outer}/{target}")
        for outer in range(10)
        for target in range(10)
    }


def _batch() -> dict[str, object]:
    return {
        "prediction_batch_id": _digest("raw-batch"),
        "record_count": 100,
    }


def _receipt() -> dict[str, object]:
    return build_deform360_v61_candidate_panel_receipt(
        candidate_revision="2" * 40,
        upstream_prediction_receipt_id=_digest("upstream-receipt"),
        raw_prediction_batch=_batch(),
        raw_prediction_batch_file_sha256=_digest("raw-batch-file"),
        candidate_artifact_id_by_record=_mapping("artifact"),
        candidate_seal_file_sha256_by_record=_mapping("seal"),
        raw_record_file_sha256_by_record=_mapping("record"),
        technical_failure_record_count=3,
    )


def test_candidate_panel_receipt_is_complete_and_closed() -> None:
    receipt = validate_deform360_v61_candidate_panel_receipt(
        _receipt(),
        raw_prediction_batch=_batch(),
        raw_prediction_batch_file_sha256=_digest("raw-batch-file"),
    )

    assert receipt["upstream_prediction_batch_id"] == UPSTREAM_PREDICTION_BATCH_ID
    assert receipt["prediction_record_count"] == 100
    assert receipt["technical_failure_record_count"] == 3
    assert receipt["information_boundary"]["source_suffix_opened"] is False
    assert receipt["information_boundary"]["source_suffix_access_authorized"] is False
    assert (
        receipt["information_boundary"]["independent_confirmation_authorized"] is False
    )
    assert runner.SEALED_VISUAL_PRODUCT_FILENAME == "baseline_disjoint.npz"


def test_candidate_panel_receipt_rejects_incomplete_roster() -> None:
    artifacts = _mapping("artifact")
    artifacts.pop("09-09")

    with pytest.raises(ValueError, match="roster changed"):
        build_deform360_v61_candidate_panel_receipt(
            candidate_revision="2" * 40,
            upstream_prediction_receipt_id=_digest("upstream-receipt"),
            raw_prediction_batch=_batch(),
            raw_prediction_batch_file_sha256=_digest("raw-batch-file"),
            candidate_artifact_id_by_record=artifacts,
            candidate_seal_file_sha256_by_record=_mapping("seal"),
            raw_record_file_sha256_by_record=_mapping("record"),
            technical_failure_record_count=0,
        )


def test_candidate_panel_receipt_rejects_identity_and_boundary_drift() -> None:
    changed = copy.deepcopy(_receipt())
    changed["information_boundary"]["source_suffix_opened"] = True

    with pytest.raises(ValueError, match="contract changed"):
        validate_deform360_v61_candidate_panel_receipt(changed)

    changed = copy.deepcopy(_receipt())
    changed["receipt_id"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        validate_deform360_v61_candidate_panel_receipt(changed)


def test_candidate_panel_receipt_rejects_wrong_raw_batch() -> None:
    changed_batch = _batch()
    changed_batch["prediction_batch_id"] = _digest("other-batch")

    with pytest.raises(ValueError, match="another raw batch"):
        validate_deform360_v61_candidate_panel_receipt(
            _receipt(), raw_prediction_batch=changed_batch
        )


def _execution_artifacts(receipt: dict[str, object]) -> dict[str, str]:
    return {
        "candidate_amendment": CANDIDATE_AMENDMENT_FILE_SHA256,
        "candidate_panel_receipt": _digest("candidate-panel-receipt-file"),
        "candidate_raw_batch": str(receipt["raw_prediction_batch_file_sha256"]),
        "execution_lock": EXECUTION_LOCK_FILE_SHA256,
        "upstream_execution_receipt": UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
        "upstream_prediction_batch": UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
        "upstream_prediction_receipt": UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
        "upstream_source_plan": UPSTREAM_SOURCE_PLAN_FILE_SHA256,
    }


def test_candidate_execution_receipt_binds_one_closed_protected_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _receipt()
    monkeypatch.setattr(
        runner,
        "_validate_upstream_execution_receipt",
        lambda _value: {"receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID},
    )
    receipt = build_deform360_v61_candidate_execution_receipt(
        candidate_revision="2" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        upstream_execution_receipt={},
        candidate_panel_receipt=panel,
        artifact_file_sha256=_execution_artifacts(panel),
    )

    validated = validate_deform360_v61_candidate_execution_receipt(receipt)
    assert validated["prediction_record_count"] == 100
    assert validated["source_suffix_access_authorized"] is False
    assert validated["independent_confirmation_authorized"] is False
    boundary = validated["information_boundary"]
    assert boundary["prob4d_pipeline_artifacts_reused"] is True
    assert boundary["prob4d_decoded_uniform_fusion_used"] is False
    assert boundary["motioncrafter_disjoint_baseline_used"] is True

    changed = copy.deepcopy(validated)
    changed["source_suffix_access_authorized"] = True
    with pytest.raises(ValueError, match="identity changed"):
        validate_deform360_v61_candidate_execution_receipt(changed)


def test_candidate_technical_failure_is_terminal_and_closed() -> None:
    receipt = build_deform360_v61_candidate_technical_failure_receipt(
        candidate_revision="2" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        terminal_stage="publish-candidate-panel",
        exit_code=7,
        retained_artifact_file_sha256={"logs/runner.log": _digest("log")},
    )

    validated = validate_deform360_v61_candidate_technical_failure_receipt(receipt)
    assert validated["exit_code"] == 7
    assert validated["status"] == "candidate-prefix-technical-failure-retained"
    assert validated["source_suffix_access_authorized"] is False
    assert validated["independent_confirmation_authorized"] is False
    assert validated["claim_authorized"] is False


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_atomic_json(value, path, overwrite=False)
    return path


def test_sealed_prediction_validator_checks_lineage_and_exact_failure_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    methods = {
        method_id: {"artifact_id": _digest(f"method/{method_id}")}
        for method_id in runner.RAW_METHOD_IDS
    }
    record = {
        "prediction_fit_artifact_id": _digest("fit"),
        "prediction_fit_object_ids": ["object-a", "object-b"],
        "factor_admitted": False,
        "physical_mode": "warp_twin",
        "risk_score": 0.25,
        "technical_failure": False,
        "methods": methods,
    }
    seal = {
        "execution_lock_id": _digest("lock"),
        "implementation_revision": "1" * 40,
        "prediction_fit_artifact_id": record["prediction_fit_artifact_id"],
        "prediction_fit_object_ids": record["prediction_fit_object_ids"],
        "factor_admitted": False,
        "physical_mode": "warp_twin",
        "risk_score": 0.25,
        "method_artifact_ids": {
            method_id: row["artifact_id"] for method_id, row in methods.items()
        },
    }
    baseline = np.zeros((2, 3, 3), dtype=np.float64)
    result = SimpleNamespace(
        trajectories_m={
            method_id: baseline.copy() for method_id in runner.RAW_METHOD_IDS
        }
    )
    monkeypatch.setattr(
        runner,
        "load_deform360_joint_sparse_prediction_v5",
        lambda _directory: (seal, result),
    )

    returned_seal, returned_result = runner._validate_sealed_prediction_artifact(
        tmp_path,
        record=record,
        lock={"execution_lock_id": _digest("lock")},
        implementation_revision="1" * 40,
    )
    assert returned_seal is seal
    assert returned_result is result

    failed_record = {**record, "technical_failure": True}
    runner._validate_sealed_prediction_artifact(
        tmp_path,
        record=failed_record,
        lock={"execution_lock_id": _digest("lock")},
        implementation_revision="1" * 40,
    )

    changed = copy.deepcopy(record)
    changed["methods"][runner.RAW_METHOD_IDS[0]]["artifact_id"] = _digest("changed")
    with pytest.raises(ValueError, match="method artifact differs"):
        runner._validate_sealed_prediction_artifact(
            tmp_path,
            record=changed,
            lock={"execution_lock_id": _digest("lock")},
            implementation_revision="1" * 40,
        )


def test_upstream_execution_validator_rejects_lineage_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = {
        "receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID,
        "prediction_batch_id": UPSTREAM_PREDICTION_BATCH_ID,
        "source_prediction_receipt_id": runner.UPSTREAM_PREDICTION_RECEIPT_ID,
        "source_plan_id": runner.UPSTREAM_SOURCE_PLAN_ID,
        "source_revision": runner.UPSTREAM_REVISION,
    }
    monkeypatch.setattr(
        runner,
        "validate_deform360_v6_source_camera_reuse_execution_receipt",
        lambda value: dict(value),
    )
    assert runner._validate_upstream_execution_receipt(valid) == valid

    with pytest.raises(ValueError, match="another upstream execution"):
        runner._validate_upstream_execution_receipt(
            {**valid, "source_revision": "0" * 40}
        )


def test_prefix_only_candidate_panel_roundtrip_and_terminal_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "2" * 40
    object_ids = tuple(f"object-{index}" for index in range(10))
    cohort = {
        object_id: (index, "sheet" if index % 2 == 0 else "rope")
        for index, object_id in enumerate(object_ids)
    }
    lock = {"execution_lock_id": runner.EXECUTION_LOCK_ID}

    amendment_path = _write_json(tmp_path / "candidate-amendment.json", {})
    lock_path = _write_json(tmp_path / "execution-lock.json", lock)
    input_root = tmp_path / "prefix-inputs"
    input_root.mkdir()

    objects: list[dict[str, object]] = []
    for index, object_id in enumerate(object_ids):
        object_root = input_root / object_id
        object_root.mkdir()
        physical_path = object_root / "physical.npz"
        physical_path.write_bytes(f"physical/{object_id}".encode())
        visual_windows = []
        for camera_index in range(2):
            camera_id = (
                "camera-error"
                if index == 1 and camera_index == 0
                else f"camera-{camera_index}"
            )
            camera_root = object_root / camera_id
            camera_root.mkdir()
            visual_path = camera_root / runner.SEALED_VISUAL_PRODUCT_FILENAME
            metric_path = camera_root / "metric-prefix.npz"
            visual_path.write_bytes(f"visual/{object_id}/{camera_id}".encode())
            metric_path.write_bytes(f"metric/{object_id}/{camera_id}".encode())
            visual_windows.append(
                {
                    "camera_id": camera_id,
                    "decoded_uniform": {
                        "path": visual_path.relative_to(input_root).as_posix()
                    },
                    "metric_prefix": {
                        "path": metric_path.relative_to(input_root).as_posix()
                    },
                }
            )
        objects.append(
            {
                "object_id": object_id,
                "physical": {
                    "path": physical_path.relative_to(input_root).as_posix(),
                    "physical_mode": "warp_twin",
                },
                "camera_admission": {"exact_physical_fallback_required": index == 0},
                "raw_prefix_range_half_open": [10, 14],
                "visual_windows": visual_windows,
            }
        )

    plan = {
        "implementation_revision": runner.UPSTREAM_REVISION,
        "plan_id": runner.UPSTREAM_SOURCE_PLAN_ID,
        "objects": objects,
    }
    plan_path = _write_json(tmp_path / "upstream-plan.json", plan)
    upstream_records = [
        {
            "outer_held_out_object_id": outer_id,
            "object_id": target_id,
            "technical_failure": False,
        }
        for outer_id in object_ids
        for target_id in object_ids
    ]
    upstream_batch = {
        "prediction_batch_id": UPSTREAM_PREDICTION_BATCH_ID,
        "implementation_revision": runner.UPSTREAM_REVISION,
        "records": upstream_records,
    }
    batch_path = _write_json(tmp_path / "upstream-batch.json", upstream_batch)

    seal_root = tmp_path / "upstream-seals"
    seal_root.mkdir()
    receipt_seals: dict[str, str] = {}
    for outer_index, outer_id in enumerate(object_ids):
        for target_index, target_id in enumerate(object_ids):
            key = f"{outer_index:02d}-{target_index:02d}.json"
            row = next(
                record
                for record in upstream_records
                if record["outer_held_out_object_id"] == outer_id
                and record["object_id"] == target_id
            )
            seal_path = _write_json(seal_root / key, row)
            receipt_seals[key] = _file_digest(seal_path)
    upstream_receipt = {
        "receipt_id": runner.UPSTREAM_PREDICTION_RECEIPT_ID,
        "source_prediction_seal_file_sha256": receipt_seals,
    }
    upstream_receipt_path = _write_json(
        tmp_path / "upstream-prediction-receipt.json", upstream_receipt
    )
    upstream_execution_path = _write_json(
        tmp_path / "upstream-execution-receipt.json",
        {"receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID},
    )
    prediction_root = tmp_path / "upstream-predictions"
    prediction_root.mkdir()

    immutable_digests = {
        amendment_path.resolve(): CANDIDATE_AMENDMENT_FILE_SHA256,
        lock_path.resolve(): EXECUTION_LOCK_FILE_SHA256,
        plan_path.resolve(): UPSTREAM_SOURCE_PLAN_FILE_SHA256,
        batch_path.resolve(): UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
        upstream_receipt_path.resolve(): UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
        upstream_execution_path.resolve(): UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
    }

    def fake_sha256(path: str | Path) -> str:
        candidate = Path(path).absolute()
        if candidate in immutable_digests:
            return immutable_digests[candidate]
        if candidate.is_file():
            return _file_digest(candidate)
        return _digest(candidate.as_posix())

    def verified_file(
        root: str | Path, record: dict[str, object], **_kwargs: object
    ) -> Path:
        path = Path(root) / str(record["path"])
        assert path.is_file()
        return path.resolve()

    physical = np.zeros((3, 4, 3), dtype=np.float64)
    b0 = np.zeros_like(physical)
    b1 = np.full_like(physical, 0.001)

    def prepare_visual(**kwargs: object) -> tuple[dict[str, object], object]:
        camera_id = str(kwargs["camera_id"])
        if camera_id == "camera-error":
            raise ValueError("synthetic prefix provider failure")
        return {"camera_id": camera_id}, object()

    def sealed_prediction(
        directory: Path, **_kwargs: object
    ) -> tuple[dict[str, str], SimpleNamespace]:
        return (
            {"prediction_seal_id": _digest(directory.as_posix())},
            SimpleNamespace(
                trajectories_m={
                    runner.B0_PHYSICAL_FALLBACK: b0,
                    runner.B1_LAST_CAUSAL_RESIDUAL: b1,
                }
            ),
        )

    def candidate_arrays(**_kwargs: object) -> dict[str, np.ndarray]:
        return {"d1": np.full_like(b0, 0.002), "b0": b0}

    def fallback_arrays(**_kwargs: object) -> dict[str, np.ndarray]:
        return {"d1": b0, "b0": b0}

    def publish_candidate(
        arrays: dict[str, np.ndarray],
        directory: str | Path,
        **kwargs: object,
    ) -> dict[str, object]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        archive_path = root / runner.CANDIDATE_ARCHIVE_FILENAME
        if not archive_path.exists():
            np.savez(archive_path, d1=arrays["d1"], b0=arrays["b0"])
        identity = {
            "candidate_artifact_id": _digest(root.as_posix()),
            "candidate_revision": kwargs["candidate_revision"],
            "technical_failure": kwargs["technical_failure"],
            "technical_failure_id": kwargs["technical_failure_id"],
        }
        runner._publish_or_validate_json(
            identity,
            root / runner.CANDIDATE_SEAL_FILENAME,
            label="synthetic candidate seal",
        )
        return identity

    def load_candidate(
        directory: str | Path,
    ) -> tuple[dict[str, object], SimpleNamespace]:
        root = Path(directory)
        seal = runner.load_strict_json_object(
            root / runner.CANDIDATE_SEAL_FILENAME, label="synthetic candidate seal"
        )
        with np.load(
            root / runner.CANDIDATE_ARCHIVE_FILENAME, allow_pickle=False
        ) as archive:
            arrays = {
                "trajectory__d1_native_model_average": np.asarray(archive["d1"]),
                "trajectory__b0_physical_fallback": np.asarray(archive["b0"]),
            }
        return seal, SimpleNamespace(arrays=arrays)

    def raw_prediction(**kwargs: object) -> dict[str, object]:
        return {
            "outer_held_out_object_id": kwargs["outer_held_out_object_id"],
            "object_id": kwargs["object_id"],
            "candidate_revision": kwargs["candidate_revision"],
            "variants": kwargs["variants"],
            "source_artifacts": kwargs["source_artifacts"],
        }

    def raw_batch(records: object, **_kwargs: object) -> dict[str, object]:
        rows = list(records)  # type: ignore[arg-type]
        return {
            "prediction_batch_id": _digest("synthetic-v61-raw-batch"),
            "record_count": len(rows),
            "records": rows,
        }

    monkeypatch.setattr(runner, "_sha256_file", fake_sha256)
    monkeypatch.setattr(
        runner, "load_deform360_v61_candidate_amendment", lambda _path: {}
    )
    monkeypatch.setattr(
        runner,
        "load_deform360_joint_sparse_source_execution_lock_v5",
        lambda _path: lock,
    )
    monkeypatch.setattr(
        runner,
        "validate_deform360_joint_sparse_source_prediction_plan_v5_2",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "validate_deform360_joint_sparse_source_prediction_batch_v5",
        lambda value, _lock: value,
    )
    monkeypatch.setattr(
        runner,
        "validate_deform360_joint_sparse_source_prediction_receipt_v5_2",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "_validate_upstream_execution_receipt",
        lambda _value: {"receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID},
    )
    monkeypatch.setattr(runner, "_ordinary_root", lambda path: Path(path).resolve())
    monkeypatch.setattr(runner, "_cohort", lambda _lock: cohort)
    monkeypatch.setattr(runner, "_verified_file", verified_file)
    monkeypatch.setattr(
        runner, "_load_physical_archive", lambda *_args, **_kwargs: (physical, b0)
    )
    monkeypatch.setattr(
        runner, "prepare_deform360_disjoint_visual_window_v6_1", prepare_visual
    )
    monkeypatch.setattr(
        runner,
        "validate_deform360_joint_sparse_source_prediction_seal_v5",
        lambda value, _lock: value,
    )
    monkeypatch.setattr(
        runner, "_validate_sealed_prediction_artifact", sealed_prediction
    )
    monkeypatch.setattr(
        runner, "build_deform360_v61_candidate_arrays", candidate_arrays
    )
    monkeypatch.setattr(
        runner, "build_deform360_v61_technical_fallback_arrays", fallback_arrays
    )
    monkeypatch.setattr(
        runner, "publish_deform360_v61_candidate_artifact", publish_candidate
    )
    monkeypatch.setattr(
        runner,
        "raw_variants_from_deform360_v61_candidate_seal",
        lambda seal: {"d1": seal["candidate_artifact_id"]},
    )
    monkeypatch.setattr(
        runner, "build_deform360_v6_raw_nested_prediction", raw_prediction
    )
    monkeypatch.setattr(runner, "build_deform360_v6_raw_nested_batch", raw_batch)
    monkeypatch.setattr(
        runner,
        "publish_deform360_v6_raw_nested_batch",
        lambda value, path, **_kwargs: runner._publish_or_validate_json(
            value, Path(path), label="synthetic raw batch"
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_deform360_v6_raw_nested_batch",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "validate_deform360_v6_raw_nested_prediction",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(runner, "load_deform360_v61_candidate_artifact", load_candidate)

    output_root = tmp_path / "candidate-panel"
    publish_kwargs = {
        "candidate_amendment_path": amendment_path,
        "execution_lock_path": lock_path,
        "source_plan_path": plan_path,
        "upstream_prediction_batch_path": batch_path,
        "upstream_prediction_receipt_path": upstream_receipt_path,
        "upstream_execution_receipt_path": upstream_execution_path,
        "upstream_source_seal_root": seal_root,
        "upstream_prediction_root": prediction_root,
        "input_root": input_root,
        "output_root": output_root,
        "candidate_revision": revision,
    }
    receipt = runner.publish_deform360_v61_candidate_panel(**publish_kwargs)
    assert receipt["prediction_record_count"] == 100
    assert receipt["technical_failure_record_count"] == 20
    assert runner.publish_deform360_v61_candidate_panel(**publish_kwargs) == receipt
    assert (
        runner.validate_deform360_v61_candidate_panel(
            execution_lock_path=lock_path, output_root=output_root
        )
        == receipt
    )

    execution_path = tmp_path / "candidate-execution-receipt.json"
    execution = runner.seal_deform360_v61_candidate_execution(
        candidate_amendment_path=amendment_path,
        execution_lock_path=lock_path,
        upstream_source_plan_path=plan_path,
        upstream_prediction_batch_path=batch_path,
        upstream_prediction_receipt_path=upstream_receipt_path,
        upstream_execution_receipt_path=upstream_execution_path,
        candidate_output_root=output_root,
        candidate_revision=revision,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        output_path=execution_path,
    )
    assert execution["status"] == "candidate-prefix-panel-sealed"
    assert execution["technical_failure_record_count"] == 20

    failure_root = tmp_path / "failed-run"
    (failure_root / "logs").mkdir(parents=True)
    (failure_root / "logs" / "runner.log").write_text("prefix-only failure\n")
    failure_path = failure_root / "technical-failure-receipt.json"
    failure = runner.retain_deform360_v61_candidate_execution_failure(
        candidate_revision=revision,
        runner_name="workstation2",
        workflow_run_id=124,
        workflow_run_attempt=1,
        terminal_stage="publish-candidate-panel",
        exit_code=7,
        artifact_root=failure_root,
        output_path=failure_path,
    )
    assert failure["retained_artifacts"] == {
        "logs/runner.log": _file_digest(failure_root / "logs" / "runner.log")
    }
    assert (
        runner.retain_deform360_v61_candidate_execution_failure(
            candidate_revision=revision,
            runner_name="workstation2",
            workflow_run_id=124,
            workflow_run_attempt=1,
            terminal_stage="publish-candidate-panel",
            exit_code=7,
            artifact_root=failure_root,
            output_path=failure_path,
        )
        == failure
    )
