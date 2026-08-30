from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rgbench_matphys_protocol_v1 import (
    load_rgbench_matphys_preaccess_amendment_v1,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "protocols/execution_requests/rgbench_matphys_technical_smoke_v1.json"
PROTOCOL = ROOT / "protocols/locks/rgbench_matphys_selective_risk_v1.json"
AMENDMENT = (
    ROOT / "protocols/amendments/rgbench_matphys_selective_risk_v1_preaccess.json"
)
RUNNER = ROOT / "scripts/remote/run_rgbench_matphys_technical_smoke_v1.py"
RECEIPT = ROOT / "evidence/rgbench_matphys_technical_smoke_terminal_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict[str, object]:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_technical_smoke_plan_has_canonical_identity_and_exact_sources() -> None:
    plan = _plan()
    identity = dict(plan)
    declared = identity.pop("plan_id")

    assert declared == content_id(identity)
    assert plan["schema"] == (
        "bayesian-phystwin.rgbench-matphys-technical-smoke-execution"
    )
    assert plan["schema_version"] == 1

    implementation = plan["implementation"]
    assert isinstance(implementation, dict)
    assert implementation["revision"] == ("99af3e589d75182272b4ddf696bf528377ca0661")
    assert implementation["runner_sha256"] == _sha256(RUNNER)

    cohort = plan["cohort"]
    assert isinstance(cohort, dict)
    assert cohort["protocol_sha256"] == _sha256(PROTOCOL)
    assert cohort["amendment_sha256"] == _sha256(AMENDMENT)

    upstreams = plan["upstreams"]
    assert isinstance(upstreams, dict)
    assert upstreams["matphys_revision"] == ("c16b858dfb79bf21024ead24b45a710600de7b4f")
    assert upstreams["rgbench_revision"] == ("5cc3d07209362b3bfdbfbc067168dea9a791690a")
    assert upstreams["rgbench_experiment_library_relative_path"] == (
        "configs/experiment_library.yaml"
    )


def test_technical_smoke_plan_is_one_attempt_and_target_closed() -> None:
    plan = _plan()
    authorization = plan["authorization"]
    custody = plan["custody"]
    runtime = plan["runtime"]
    cohort = plan["cohort"]
    assert isinstance(authorization, dict)
    assert isinstance(custody, dict)
    assert isinstance(runtime, dict)
    assert isinstance(cohort, dict)

    assert authorization == {
        "technical_source_smoke_authorized": True,
        "scientific_source_gate_authorized": False,
        "source_future_scoring_authorized": False,
        "target_execution_authorized": False,
        "attempt_limit": 1,
        "replacement_allowed": False,
    }
    assert custody["decoded_source_frame_indices"] == [0]
    assert custody["source_future_payload_read"] is False
    assert custody["source_future_outcomes_scored"] is False
    assert custody["target_payload_read"] is False
    assert custody["target_outcomes_opened"] is False
    assert custody["held_v8_accessed"] is False
    assert custody["dlo4_dlo5_accessed"] is False
    assert custody["prob4d_used"] is False
    assert custody["attempt_ledger_path"] != custody["output_root"]
    assert runtime["device"] == "cuda:1"
    assert runtime["physical_gpu_index"] == 1

    amended = load_rgbench_matphys_preaccess_amendment_v1(PROTOCOL, AMENDMENT)
    source_cell = cohort["source_cell"]
    assert isinstance(source_cell, dict)
    identity = (
        source_cell["garment_id"],
        source_cell["action"],
        source_cell["sample_id"],
        source_cell["data_subfolder"],
    )
    assert identity in {cell.identity for cell in amended.source_cells}
    assert identity not in {cell.identity for cell in amended.target_cells}


def test_terminal_receipt_is_hash_bound_and_closes_advancement() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    receipt_id = receipt.pop("receipt_id")

    assert receipt_id == (
        "182b47e681e85d7bb4e6408240e95a15f3526cecb6fc04ef8d03b3b632ce2dd8"
    )
    assert content_id(receipt) == receipt_id
    assert receipt["attempt_consumed"] is True
    assert receipt["technical_smoke_passed"] is False
    assert receipt["scientific_source_gate_passed"] is False
    assert receipt["source_competence_claim_authorized"] is False
    assert receipt["target_authorized"] is False
    assert receipt["retry_authorized"] is False
    assert receipt["further_replacement_allowed"] is False
    assert receipt["source_frame_zero_decoded"] is True
    assert receipt["source_future_payload_decoded"] is False
    assert all(value is False for value in receipt["information_boundary"].values())
