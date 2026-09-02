from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_transport4d_public_development_v1.py"
PROTOCOL = ROOT / "protocols" / "transport4d_public_development_v1.json"
RESULT = ROOT / "evidence" / "transport4d_public_development_v1" / "result.json"
REPORT = ROOT / "evidence" / "transport4d_public_development_v1" / "report.md"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "transport4d_public_development", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Transport4D development script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_development_matrix_has_strict_positive_negative_separation() -> None:
    module = load_script()
    result = module.run(PROTOCOL)

    assert result["decision"] == "public-development-tier-separation-established"
    assert result["strongest_supported_tier_by_shift"] == {
        "same-object-cross-backend": "exact_coefficients",
        "cross-object-operator": "procedure_only",
    }
    rows = {
        (row["shift"], row["tested_tier"]): row for row in result["development_matrix"]
    }
    assert (
        rows[("same-object-cross-backend", "exact_coefficients")]["supported"] is True
    )
    assert rows[("cross-object-operator", "exact_coefficients")]["supported"] is False
    assert result["information_boundary"]["fresh_confirmation_claim"] is False


def test_committed_public_development_evidence_is_reproducible() -> None:
    module = load_script()
    generated = module.run(PROTOCOL)
    committed = json.loads(RESULT.read_text(encoding="utf-8"))

    assert generated == committed
    assert module.report(generated) == REPORT.read_text(encoding="utf-8")


def test_protocol_tampering_fails_closed(tmp_path: Path) -> None:
    module = load_script()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["evidence"]["cross_object_operator"]["exact_coefficients"]["wins"] = 28
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol_id"):
        module.run(path)
