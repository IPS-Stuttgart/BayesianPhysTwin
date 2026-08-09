from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_prob4d_visible_source_audit import (
    EXPECTED_METRIC_BATCH_RESULT_ID,
    EXPECTED_PLAN_ID,
    EXPECTED_SAMPLE_STDERR_SHA256,
    audit_deform360_prob4d_visible_source_v2,
)

SOURCE = Path("results/sota/deform360_prob4d_visible_source_v2/source-artifact")
INDEPENDENT_AUDIT = Path(
    "results/sota/deform360_prob4d_visible_source_v2/independent-audit"
)
VALIDATOR_REVISION = "136f72b996e9c76b0bab3ab5db5d0fe7172e0307"
ARTIFACT_DIGEST = (
    "sha256:caa8d5ea887ec5273c306dd8de59d57056181ef139c98ac7acb76185032a3828"
)


def _audit(source: Path, output: Path) -> dict[str, object]:
    return dict(
        audit_deform360_prob4d_visible_source_v2(
            source_root=source,
            output_directory=output,
            validator_revision=VALIDATOR_REVISION,
            source_run_id=31301431579,
            source_run_attempt=1,
            source_artifact_id=9034737368,
            source_artifact_name="deform360-prob4d-source-gate-31301431579-1",
            source_artifact_digest=ARTIFACT_DIGEST,
        )
    )


def _rehash(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    records = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in paths
    ]
    (root / "SHA256SUMS").write_text("".join(records), encoding="ascii")


def test_exact_compact_source_terminal_is_independently_validated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit"

    result = _audit(SOURCE, output)

    assert result["audit_status"] == "validated-source-sample-materialization-failure"
    assert result["result_kind"] == "source-pipeline-technical-terminal"
    assert result["support_gate_passed"] is True
    assert result["source_gate_evaluated"] is False
    assert result["source_gate_passed"] is None
    assert result["confirmation_access_authorized"] is False
    assert result["failure_detail_available_in_compact_artifact"] is False
    assert result["sample_stderr_sha256"] == EXPECTED_SAMPLE_STDERR_SHA256
    assert result["metric_batch_result_id"] == EXPECTED_METRIC_BATCH_RESULT_ID
    assert result["metric_prefix_plan_id"] == EXPECTED_PLAN_ID
    assert result["supported_stream_count"] == 313
    assert result["support_negative_stream_count"] == 11
    assert result["information_boundary"] == {
        "public_released_measurements_used": True,
        "new_measurements_required": False,
        "human_approval_required": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "future_frames_used": False,
        "replacement_allowed": False,
    }
    receipt = json.loads(
        (output / "independent-audit-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt == result
    digest, relative = (output / "SHA256SUMS").read_text(encoding="ascii").split()
    assert relative == "independent-audit-receipt.json"
    assert digest == hashlib.sha256((output / relative).read_bytes()).hexdigest()


def test_hosted_independent_audit_receipt_is_preserved() -> None:
    receipt_path = INDEPENDENT_AUDIT / "independent-audit-receipt.json"
    manifest = (INDEPENDENT_AUDIT / "SHA256SUMS").read_text(encoding="ascii")
    digest, relative = manifest.split()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert relative == receipt_path.name
    assert digest == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert receipt["validator_revision"] == ("dbeab4f5a5c8279b82404b3dc911c39ddccab10d")
    assert receipt["audit_status"] == (
        "validated-source-sample-materialization-failure"
    )
    assert receipt["source_gate_evaluated"] is False
    assert receipt["confirmation_access_authorized"] is False
    assert receipt["information_boundary"]["human_approval_required"] is False
    audit_id = receipt.pop("audit_id")
    assert audit_id == content_id(receipt)
    assert audit_id == (
        "86d6998941fe76769a007494c1ff84ac820ada96f414504172cd2daacba26511"
    )


def test_audit_rejects_changed_compact_member_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(SOURCE, source)
    pipeline = source / "pipeline-receipt.json"
    pipeline.write_bytes(pipeline.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="compact artifact member changed"):
        _audit(source, tmp_path / "audit")


def test_audit_rejects_rehashed_confirmation_authorization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(SOURCE, source)
    pipeline_path = source / "pipeline-receipt.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["confirmation_access_authorized"] = True
    pipeline_path.write_text(
        json.dumps(pipeline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash(source)

    with pytest.raises(ValueError, match="source-gate decision"):
        _audit(source, tmp_path / "audit")


def test_audit_rejects_rehashed_sample_error_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(SOURCE, source)
    pipeline_path = source / "pipeline-receipt.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["stderr_sha256"]["samples"] = hashlib.sha256(b"changed").hexdigest()
    pipeline_path.write_text(
        json.dumps(pipeline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash(source)

    with pytest.raises(ValueError, match="stderr accounting changed"):
        _audit(source, tmp_path / "audit")


def test_audit_rejects_changed_source_artifact_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source run or compact artifact identity"):
        audit_deform360_prob4d_visible_source_v2(
            source_root=SOURCE,
            output_directory=tmp_path / "audit",
            validator_revision=VALIDATOR_REVISION,
            source_run_id=31301431579,
            source_run_attempt=1,
            source_artifact_id=9034737368,
            source_artifact_name="deform360-prob4d-source-gate-changed",
            source_artifact_digest=ARTIFACT_DIGEST,
        )


def test_audit_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    _audit(SOURCE, output)

    with pytest.raises(ValueError, match="audit output already exists"):
        _audit(SOURCE, output)
