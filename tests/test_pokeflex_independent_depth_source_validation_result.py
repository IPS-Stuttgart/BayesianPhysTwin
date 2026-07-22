import hashlib
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
RESULT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "pokeflex_independent_depth_source_validation_v2"
)


def test_source_validation_result_records_failed_registered_gate() -> None:
    result = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))

    assert result["artifact_kind"] == "PokeFlexIndependentDepthSourceValidationResult"
    assert result["object_count"] == 5
    assert result["take_count"] == 20
    assert result["object_balanced_selector"]["relative_improvement"] == pytest.approx(
        0.039335829870781096
    )
    assert result["object_balanced_selector"]["object_wins"] == 3
    assert result["registered_gate"]["all_passed"] is False
    assert result["registered_gate"]["T2_access_permitted"] is False
    assert result["registered_gate"]["checks"]["object_balanced_improvement"] is False
    assert result["registered_gate"]["checks"]["object_wins"] is False


def test_source_validation_execution_has_no_failure_or_replacement() -> None:
    manifest = json.loads(
        (RESULT_ROOT / "execution_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["replacement_allowed"] is False
    assert len(manifest["records"]) == 20
    assert all(
        record["status"] in {"completed", "existing"}
        for record in manifest["records"]
    )


def test_source_validation_verification_locks_evidence_and_legacy_runner() -> None:
    verification = json.loads(
        (RESULT_ROOT / "verification.json").read_text(encoding="utf-8")
    )

    assert verification["T2_access_permitted"] is False
    for relative_path, expected in verification["files"].items():
        observed = hashlib.sha256(
            (REPOSITORY_ROOT / relative_path).read_bytes()
        ).hexdigest()
        assert observed == expected
