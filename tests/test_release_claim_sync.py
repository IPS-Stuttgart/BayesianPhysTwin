from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/release/check_release_claim_sync.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_release_claim_sync",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _write_fixture(root: Path) -> Path:
    files = {
        "README.md": "# Project\npoint result\nuncertainty boundary\n",
        "SUPPORT.md": "# Support\nexact fallback\n",
        "docs/source.md": "# Source evidence\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    contract = {
        "schema": tool.CONTRACT_SCHEMA,
        "schema_version": tool.SCHEMA_VERSION,
        "contract_name": "fixture-release-claim-v1",
        "claim_boundary": "software release wording only",
        "documents": [
            {
                "path": "README.md",
                "required_literals": ["point result", "uncertainty boundary"],
            },
            {
                "path": "SUPPORT.md",
                "required_literals": ["exact fallback"],
            },
        ],
        "source_documents": ["docs/source.md"],
    }
    path = root / tool.DEFAULT_CONTRACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_builds_content_addressed_sync_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = tool.check_release_claim_sync(tmp_path)

    assert report["contract_name"] == "fixture-release-claim-v1"
    assert len(report["documents"]) == 2
    assert len(report["source_documents"]) == 1
    descriptor = dict(report)
    supplied_id = descriptor.pop("report_id")
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert supplied_id == hashlib.sha256(canonical).hexdigest()


def test_missing_required_literal_fails_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Project\npoint result\n",
        encoding="utf-8",
    )

    with pytest.raises(tool.ReleaseClaimSyncError, match="uncertainty boundary"):
        tool.check_release_claim_sync(tmp_path)


def test_markdown_line_wrapping_preserves_literal_match(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Project\npoint\n  result\nuncertainty\tboundary\n",
        encoding="utf-8",
    )

    report = tool.check_release_claim_sync(tmp_path)

    assert report["contract_name"] == "fixture-release-claim-v1"


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    contract_path = _write_fixture(tmp_path)
    contract_path.write_text(
        '{"schema":"a","schema":"b"}\n',
        encoding="utf-8",
    )

    with pytest.raises(tool.ReleaseClaimSyncError, match="duplicate JSON object key"):
        tool.check_release_claim_sync(tmp_path)


def test_noncanonical_document_path_is_rejected(tmp_path: Path) -> None:
    contract_path = _write_fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["documents"][0]["path"] = "../README.md"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(tool.ReleaseClaimSyncError, match="not canonical"):
        tool.check_release_claim_sync(tmp_path)


def test_report_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = tool.check_release_claim_sync(tmp_path)
    output = tmp_path / "report.json"

    tool.write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(tool.ReleaseClaimSyncError, match="refusing to overwrite"):
        tool.write_report(output, report)


def test_repository_release_claim_contract_is_synchronized() -> None:
    report = tool.check_release_claim_sync(ROOT)

    assert report["contract_name"] == "phystwin-release-claim-v1"
    assert [entry["path"] for entry in report["documents"]] == [
        "CHANGELOG.md",
        "README.md",
        "SUPPORT.md",
        "docs/phystwin_release_claim_v1.md",
        "evidence/public_claim_snapshot_v1.json",
    ]
    assert [entry["path"] for entry in report["source_documents"]] == [
        "docs/full22_covariance_only_hybrid_v1.md",
        "docs/phystwin_release_claim_v1.md",
        "docs/phystwin_sota_22_v1.md",
        "protocols/locks/deform360_covariance_only_independent_validation_v1.json",
        "protocols/locks/deform360_official_hub_fresh_object_session_v6.json",
        "results/diagnostics/deform360_v61_one_shot_retirement_v1/retirement.json",
    ]
