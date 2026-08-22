from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.quality.route_deform360_v6_source_receipt import (
    CLOSED_AUTHORIZATION_FIELDS,
    CLOSED_INFORMATION_FIELDS,
    EXPECTED_AMENDMENT_ID,
    EXPECTED_RUNNER_NAME,
    EXPECTED_SCHEMA,
    EXPECTED_SCHEMA_VERSION,
    ReceiptRoutingError,
    receipt_content_id,
    verify_source_receipt_bundle,
)

SOURCE_SHA = "a" * 40
SOURCE_RUN_ID = "12345"
SOURCE_RUN_ATTEMPT = "1"


def _receipt(
    *,
    status: str = "source-prediction-evidence-sealed",
    manifest_count: int = 10,
    seal_count: int = 100,
    exit_code: int = 0,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": EXPECTED_SCHEMA,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "amendment_id": EXPECTED_AMENDMENT_ID,
        "source_revision": SOURCE_SHA,
        "workflow_run_id": SOURCE_RUN_ID,
        "workflow_run_attempt": SOURCE_RUN_ATTEMPT,
        "runner_name": EXPECTED_RUNNER_NAME,
        "status": status,
        "terminal_stage": "source-prediction-evidence-sealed",
        "exit_code": exit_code,
        "artifacts": {},
        "physical_manifest_count": manifest_count,
        "source_prediction_seal_count": seal_count,
        "information_boundary": {name: False for name in CLOSED_INFORMATION_FIELDS},
    }
    receipt.update({name: False for name in CLOSED_AUTHORIZATION_FIELDS})
    receipt["receipt_id"] = receipt_content_id(receipt)
    return receipt


def _write_bundle(
    root: Path,
    receipt: dict[str, Any],
    *,
    extra_files: dict[str, str] | None = None,
) -> None:
    root.mkdir()
    (root / "execution-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for relative, content in (extra_files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(root).as_posix()
        lines.append(f"{digest}  ./{relative}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify(root: Path, *, conclusion: str = "success") -> dict[str, Any]:
    return verify_source_receipt_bundle(
        root,
        source_run_id=SOURCE_RUN_ID,
        source_run_attempt=SOURCE_RUN_ATTEMPT,
        source_head_sha=SOURCE_SHA,
        source_workflow_conclusion=conclusion,
    )


def test_complete_sealed_receipt_passes_closed_verification(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root, _receipt(), extra_files={"logs/source.log": "bounded\n"})

    verified = _verify(root)

    assert verified["status"] == "source-prediction-evidence-sealed"
    assert verified["physical_manifest_count"] == 10
    assert verified["source_prediction_seal_count"] == 100


def test_sealed_status_requires_complete_10_100_panel(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root, _receipt(seal_count=99))

    with pytest.raises(ReceiptRoutingError, match="complete 10/100 panel"):
        _verify(root)


def test_nonsealed_receipt_requires_failed_nonzero_termination(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(
        root,
        _receipt(
            status="source-technical-failure-retained",
            manifest_count=0,
            seal_count=0,
            exit_code=1,
        ),
    )

    verified = _verify(root, conclusion="failure")
    assert verified["status"] == "source-technical-failure-retained"

    with pytest.raises(ReceiptRoutingError, match="inconsistent workflow termination"):
        _verify(root, conclusion="success")


def test_open_information_boundary_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    receipt = _receipt()
    receipt["information_boundary"]["v6_target_payloads_opened"] = True
    receipt["receipt_id"] = receipt_content_id(receipt)
    _write_bundle(root, receipt)

    with pytest.raises(ReceiptRoutingError, match="v6_target_payloads_opened"):
        _verify(root)


def test_manifest_must_cover_every_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root, _receipt())
    (root / "unlisted.txt").write_text("not in manifest\n", encoding="utf-8")

    with pytest.raises(ReceiptRoutingError, match="does not close"):
        _verify(root)


def test_symlinked_bundle_member_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root, _receipt())
    (root / "receipt-link.json").symlink_to(root / "execution-receipt.json")

    with pytest.raises(ReceiptRoutingError, match="contains a symlink"):
        _verify(root)
