from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin import deform360_process_isolation_qualification as gate


def _worker_runtime() -> dict[str, Any]:
    evidence = {
        "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
        "extension_loaded_and_retained": True,
        "target_or_outcome_path_accessed": False,
    }
    evidence["artifact_sha256"] = gate.artifact_sha256(evidence)
    return {
        "adapter_source": {
            "path": "/code/runtime.py",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "evidence": evidence,
        "backend_retained_before_original_trainer_import": True,
    }


def _write_signed(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = gate.artifact_sha256(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return result


def _source_tree(tmp_path: Path, head: str) -> tuple[Path, dict[str, Any]]:
    code = tmp_path / "code"
    for relative in gate.RUNTIME_SOURCE_BINDINGS.values():
        source = code / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {relative.as_posix()}\n", encoding="utf-8")
    runtime = {
        "code": {
            "path": str(code),
            "head": head,
            "tree": "b" * 40,
            "clean": True,
            "ordinary_untracked_file_count": 0,
            "ignored_untracked_file_count": 0,
        },
        "parent_python": {},
        "python": {},
        "deform360": {},
    }
    for name, relative in gate.RUNTIME_SOURCE_BINDINGS.items():
        runtime[name] = gate._file_record(
            code / relative,
            role=name,
        )
    return code, runtime


def _child(case_index: int) -> dict[str, Any]:
    fits = [
        {
            "fit_index": fit_index,
            "output_created": True,
            "output_absent_after_cleanup": True,
            "generated_outputs_absent_after_cleanup": True,
            "resource_boundary_stage": "after_cleanup",
        }
        for fit_index in range(gate.EXPECTED_FITS_PER_CASE)
    ]
    value = {
        "schema_version": 1,
        "artifact_kind": gate.CHILD_KIND,
        "qualification_id": gate.QUALIFICATION_ID,
        "case_index": case_index,
        "process_id": 3000 + case_index,
        "passed": True,
        "parameters": {
            "fit_count": gate.EXPECTED_FITS_PER_CASE,
            "iterations_per_fit": gate.EXPECTED_ITERATIONS_PER_FIT,
            "seed": gate.EXPECTED_SEED,
            "trainer_instance_count": 1,
            "trainer_variant": "original-pinned-default",
        },
        "fits": fits,
        "evaluation": {
            "passed": True,
            "predicates": {"complete": True},
        },
        "worker_entry_gsplat_runtime": _worker_runtime(),
        "information_boundary": dict(gate.EXPECTED_INFORMATION_BOUNDARY),
    }
    value["artifact_sha256"] = gate.artifact_sha256(value)
    return value


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, str]:
    head = "a" * 40
    monkeypatch.setattr(gate, "QUALIFICATION_BASE", tmp_path)
    _code, runtime = _source_tree(tmp_path, head)
    root = tmp_path / f"{gate.QUALIFICATION_ROOT_PREFIX}{head}"
    root.mkdir()
    canonical_parameters = {
        "dataset": "/public/source-only-dataset",
        "case_count": gate.EXPECTED_CASE_COUNT,
        "fit_count": gate.EXPECTED_FITS_PER_CASE,
        "iterations": gate.EXPECTED_ITERATIONS_PER_FIT,
        "seed": gate.EXPECTED_SEED,
        "cuda_device": gate.EXPECTED_PHYSICAL_GPU_INDEX,
        "case_timeout_seconds": 28_800,
    }
    attempt = _write_signed(
        root / gate.ATTEMPT_NAME,
        {
            "schema_version": 1,
            "artifact_kind": gate.ATTEMPT_KIND,
            "qualification_id": gate.QUALIFICATION_ID,
            "state": "canonical-root-consumed-at-creation",
            "output_root": str(root),
            "code_revision": head,
            "physical_gpu_index": gate.EXPECTED_PHYSICAL_GPU_INDEX,
            "canonical_parameters": canonical_parameters,
            "root_consumption_policy": dict(gate.ROOT_CONSUMPTION_POLICY),
            "information_boundary": dict(gate.EXPECTED_INFORMATION_BOUNDARY),
        },
    )
    attempt_record = gate._file_record(
        root / gate.ATTEMPT_NAME,
        role="fixture attempt",
    )
    cases: list[dict[str, Any]] = []
    for case_index in range(gate.EXPECTED_CASE_COUNT):
        case_root = root / f"case-{case_index:03d}"
        child = _write_signed(
            case_root / "case-child-evidence.json",
            {
                key: value
                for key, value in _child(case_index).items()
                if key != "artifact_sha256"
            },
        )
        (case_root / "case-child.log").write_text("source-only\n", encoding="utf-8")
        cases.append(
            {
                "case_index": case_index,
                "invocation": {
                    "return_code": 0,
                    "timed_out": False,
                    "timeout_error": None,
                },
                "child_evidence": child,
                "child_contract_valid": True,
                "child_validation_error": None,
                "child_evidence_file": gate._file_record(
                    case_root / "case-child-evidence.json",
                    role="fixture child",
                ),
                "materialized_inputs_stable": True,
                "source_inputs_stable": True,
                "generated_dataset_outputs_absent": True,
            }
        )
    evidence = _write_signed(
        root / gate.EVIDENCE_NAME,
        {
            "schema_version": 1,
            "artifact_kind": gate.QUALIFICATION_KIND,
            "qualification_id": gate.QUALIFICATION_ID,
            "passed": True,
            "canonical_parameters": canonical_parameters,
            "attempt_marker": {
                **{
                    key: attempt_record[key]
                    for key in ("path", "sha256", "size_bytes", "mode_octal")
                },
            },
            "host": {
                "hostname": gate.EXPECTED_HOST,
                "physical_gpu_index": gate.EXPECTED_PHYSICAL_GPU_INDEX,
            },
            "runtime_bindings": runtime,
            "process_boundary": {
                "one_original_trainer_per_child": True,
                "one_official_case_lifecycle_per_child": True,
                "fits_per_case": gate.EXPECTED_FITS_PER_CASE,
                "trainer_configuration_overridden": False,
                "process_exit_reclaims_case_resources": True,
                "parent_process_imports_nerfstudio": False,
                "worker_entry_gsplat_preload_required": True,
            },
            "cases": cases,
            "evaluation": {
                "passed": True,
                "predicates": {"all": True},
                "limits": {
                    "child_start_spread": 4,
                    "parent_fd_growth": 2,
                    "parent_task_growth": 2,
                },
            },
            "information_boundary": dict(gate.EXPECTED_INFORMATION_BOUNDARY),
        },
    )
    assert attempt["artifact_sha256"]
    assert evidence["artifact_sha256"]
    sealer_source = tmp_path / "sealer.py"
    sealer_source.write_text("# sealer\n", encoding="utf-8")
    completion = Path(f"{root}-integrity-completion.json")
    return root, completion, sealer_source, head


def test_sealer_closes_and_revalidates_exact_source_only_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, completion, sealer_source, head = _fixture(tmp_path, monkeypatch)

    lineage = gate.seal_process_isolation_qualification(
        root,
        completion,
        sealer_source_path=sealer_source,
    )

    assert lineage["process_isolation_qualification_integrity"]["source_head"] == head
    assert lineage["process_isolation_qualification_integrity"]["entry_count"] > 8
    assert stat_mode(root) == 0o500
    assert all(
        stat_mode(path) == (0o500 if path.is_dir() else 0o400)
        for path in root.rglob("*")
    )
    assert stat_mode(completion) == 0o400
    assert (
        gate.validate_process_isolation_qualification_lineage(
            evidence_path=root / gate.EVIDENCE_NAME,
            completion_path=completion,
            expected_source_head=head,
            verify_content_inventory=True,
        )
        == lineage
    )


def test_sealer_rejects_tampered_evidence_and_consumed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, completion, sealer_source, _head = _fixture(tmp_path, monkeypatch)
    evidence = root / gate.EVIDENCE_NAME
    value = json.loads(evidence.read_text(encoding="utf-8"))
    value["passed"] = False
    evidence.chmod(0o600)
    evidence.write_text(json.dumps(value), encoding="utf-8")
    evidence.chmod(0o444)

    with pytest.raises(ValueError, match="signature changed"):
        gate.seal_process_isolation_qualification(
            root,
            completion,
            sealer_source_path=sealer_source,
        )

    root, completion, sealer_source, _head = _fixture(
        tmp_path / "second",
        monkeypatch,
    )
    gate.seal_process_isolation_qualification(
        root,
        completion,
        sealer_source_path=sealer_source,
    )
    with pytest.raises(ValueError, match="already exists"):
        gate.seal_process_isolation_qualification(
            root,
            completion,
            sealer_source_path=sealer_source,
        )


def test_validator_detects_postseal_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, completion, sealer_source, head = _fixture(tmp_path, monkeypatch)
    gate.seal_process_isolation_qualification(
        root,
        completion,
        sealer_source_path=sealer_source,
    )
    log = root / "case-000" / "case-child.log"
    log.chmod(0o600)
    log.write_text("tampered-ok\n", encoding="utf-8")
    log.chmod(0o400)

    with pytest.raises(ValueError, match="content inventory changed"):
        gate.validate_process_isolation_qualification_lineage(
            evidence_path=root / gate.EVIDENCE_NAME,
            completion_path=completion,
            expected_source_head=head,
            verify_content_inventory=True,
        )


def test_validator_rejects_wrong_qualified_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, completion, sealer_source, _head = _fixture(tmp_path, monkeypatch)
    gate.seal_process_isolation_qualification(
        root,
        completion,
        sealer_source_path=sealer_source,
    )
    with pytest.raises(ValueError, match="source head changed"):
        gate.validate_process_isolation_qualification_lineage(
            evidence_path=root / gate.EVIDENCE_NAME,
            completion_path=completion,
            expected_source_head="f" * 40,
        )


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777
