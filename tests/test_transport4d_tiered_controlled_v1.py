from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_transport4d_tiered_controlled_v1.py"
RESULT = ROOT / "evidence" / "transport4d_tiered_controlled_v1" / "result.json"
REPORT = ROOT / "evidence" / "transport4d_tiered_controlled_v1" / "report.md"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "transport4d_tiered_controlled", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Transport4D controlled script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_tier_separation_and_committed_evidence() -> None:
    module = load_script()
    result = module.run()

    assert result["decision"] == "controlled-tier-separation-passed"
    assert result["checks"]["all_expected_tiers_selected"] is True
    assert (
        result["checks"]["query_conditional_descent_rejects_uncertain_exact_tier"]
        is True
    )
    assert result["checks"]["unsupported_case_selects_no_transport_tier"] is True
    assert result == json.loads(RESULT.read_text(encoding="utf-8"))
    assert module.report(result) == REPORT.read_text(encoding="utf-8")
