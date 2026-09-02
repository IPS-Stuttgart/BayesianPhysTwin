from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.query_portfolio_evidence_v2 import assemble

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/sota/query_portfolio_replication_v5"
WRAPPING = EVIDENCE / "wrapping_component_evidence.json"
SLINGSHOT = EVIDENCE / "slingshot_component_evidence.json"
JOINT = EVIDENCE / "joint_result.json"


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_portfolio_reproduces_and_passes_simultaneous_gates() -> None:
    expected = _read(JOINT)
    reproduced = assemble(
        {
            "dlolab_wrapping_v9": _read(WRAPPING),
            "dlolab_slingshot_v4": _read(SLINGSHOT),
        }
    )

    assert reproduced == expected
    assert expected["artifact_id"] == content_id(
        {key: value for key, value in expected.items() if key != "artifact_id"}
    )
    assert expected["joint_portfolio_claim_passed"] is True
    assert expected["simultaneous_positive_value_passed"] is True
    assert expected["simultaneous_harm_control_passed"] is True
    assert expected["complete_denominator"] is True
    assert expected["partial_results_used"] is False
    assert expected["cross_task_reward_pooling"] is False

    queries = expected["queries"]
    assert set(queries) == {"dlolab_wrapping_v9", "dlolab_slingshot_v4"}
    assert all(query["worlds"] == 320 for query in queries.values())
    assert all(query["gain_lower_bound"] > 0.0 for query in queries.values())
    assert all(query["harm_upper_bound"] <= 0.05 for query in queries.values())
    assert sum(query["harmful_worlds"] for query in queries.values()) == 2


def test_portfolio_result_evidence_retains_registered_hashes() -> None:
    assert _sha256(WRAPPING) == (
        "dfcd61f2e55a74fb763826bbac4c3dfe9b7223905a035084ae93cc6d7651078b"
    )
    assert _sha256(SLINGSHOT) == (
        "c7ff4fa1adfb8f50b5076b00e0265c9c016c478c9d545e8cb8e65c1fed67ea69"
    )
    assert _sha256(JOINT) == (
        "c42a52825f74b27d51152049239099f879385f8b2a3b48e6c9e4c8739eb0f713"
    )
    assert _read(JOINT)["artifact_id"] == (
        "711ca7a97017a3661e16980bd64d5481e61b32e86a5db287dcae63ebf92d907f"
    )
