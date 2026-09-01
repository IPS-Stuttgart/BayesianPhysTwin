from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "science"
    / "deform_dlo45_decision_gate_audit_v1"
    / "official-dlo45-gate-risk-audit-20260902-v2-support-contract"
)


def load_compact() -> dict[str, object]:
    return json.loads((RESULT / "compact_result.json").read_text(encoding="utf-8"))


def test_compact_evidence_is_complete_and_source_frozen() -> None:
    value = load_compact()

    assert value["contract"] == "deform-dlo45-support-contract-audit-summary-v1"
    assert value["schema_version"] == 1
    assert value["status"] == "complete"
    assert value["run_key"] == "official-dlo45-gate-risk-audit-20260902-v2-support-contract"
    assert (RESULT / "thresholds.json").is_file()
    assert (RESULT / "source_seal.json").is_file()
    assert (RESULT / "target_audit.json").is_file()
    assert (RESULT / "per_decision.jsonl").is_file()


def test_certificate_reproduces_parent_result_and_obeys_support_contract() -> None:
    value = load_compact()
    certificate = value["certificate"]
    assert isinstance(certificate, dict)

    assert certificate["decision_count"] == 532
    assert certificate["nonfallback_count"] == 82
    assert certificate["harmful_nonfallback_count"] == 3
    assert certificate["registered_support_violation_count_nonfallback"] == 0
    assert certificate["registered_support_violation_fraction_nonfallback"] == 0.0
    assert float(certificate["registered_support_nonfallback_maximum_worst_case_regret"]) <= 0.05
    assert math.isclose(
        float(certificate["rmse_reduction"]),
        0.04270043213487973,
        rel_tol=0.0,
        abs_tol=1e-7,
    )


def test_best_heuristic_improves_utility_but_violates_registered_support() -> None:
    value = load_compact()
    certificate = value["certificate"]
    best = value["best_source_calibrated_gate"]
    assert isinstance(certificate, dict)
    assert isinstance(best, dict)

    assert value["best_source_calibrated_gate_name"] == "expected_fallback_advantage"
    assert float(best["rmse_reduction"]) > float(certificate["rmse_reduction"])
    assert best["harmful_nonfallback_count"] == 1
    assert best["nonfallback_count"] == 63
    assert best["registered_support_violation_count_nonfallback"] == 51
    assert math.isclose(
        float(best["registered_support_violation_fraction_nonfallback"]),
        51 / 63,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert float(best["registered_support_nonfallback_p95_worst_case_regret"]) > 0.05


def test_exact_coverage_match_retains_the_utility_admissibility_tradeoff() -> None:
    value = load_compact()
    matched = value["target_covariate_matched_gates"]
    assert isinstance(matched, dict)
    best = matched["expected_fallback_advantage"]
    assert isinstance(best, dict)

    assert best["nonfallback_count"] == 82
    assert best["harmful_nonfallback_count"] == 1
    assert best["registered_support_violation_count_nonfallback"] == 66
    assert math.isclose(
        float(best["rmse_reduction"]),
        0.08703895840064657,
        rel_tol=0.0,
        abs_tol=1e-7,
    )


def test_claim_boundary_rejects_empirical_dominance_and_safety_overclaim() -> None:
    value = load_compact()
    conclusion = str(value["comparative_conclusion"])
    boundary = str(value["claim_boundary"])

    assert "only policy constrained" in conclusion
    assert "may favor a heuristic" in conclusion
    assert "unseen-object generalization" in boundary
    assert "arbitrary-action safety" in boundary
    assert "deployment authorization" in boundary
