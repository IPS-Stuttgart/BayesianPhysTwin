from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "evidence" / "rct_real_decision_source_terminal_v1.json"


def test_rct_source_terminal_receipt_is_content_bound() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    declared = receipt.pop("receipt_id")
    assert declared == content_id(receipt)


def test_rct_source_terminal_receipt_closes_scientific_claims() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["attempt_consumed"] is True
    assert receipt["attempt_limit"] == 1
    assert receipt["terminal_stage"] == "registered-force-trace-validation"
    assert receipt["information_boundary"] == {
        "confirmation_force_fields_parsed": False,
        "confirmation_material_rows_admitted": False,
        "confirmation_outcomes_opened": False,
        "dlo4_dlo5_accessed": False,
        "held_v8_accessed": False,
        "source_force_payload_accessed": True,
    }
    assert receipt["method_seal_produced"] is False
    assert receipt["source_test_material_results_produced"] is False
    assert receipt["scientific_performance_score_available"] is False
    assert receipt["source_gate_evaluated"] is False
    assert receipt["qualification_result_produced"] is False
    assert receipt["retry_authorized"] is False
    assert receipt["further_replacement_allowed"] is False
    assert receipt["confirmation_authorized"] is False
    assert receipt["target_authorized"] is False


def test_rct_source_terminal_receipt_binds_private_record() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    private = receipt["private_evidence"]
    assert private["repository"] == "FlorianPfaff/BayesianPhysTwin-Paper"
    assert private["pull_request"] == 178
    assert private["terminal_record_id"] == (
        "63f275a961367539d46df4fa88fcf89233f7101fe3ea6d2cde483fb7f027be4d"
    )
    assert private["terminal_record_file_sha256"] == (
        "7f1d58fff35a957581c2723a51c233e01a277d0662cff7e04ca4b5c76749807d"
    )
