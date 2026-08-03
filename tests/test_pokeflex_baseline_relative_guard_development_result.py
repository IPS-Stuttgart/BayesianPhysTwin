from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_baseline_relative_guard import (
    FEATURE_NAMES,
    certificate_from_payload,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results" / "sota" / "pokeflex_baseline_relative_guard_development_v2"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((RESULT_ROOT / name).read_bytes()).hexdigest()


def test_development_result_is_hash_bound_and_explicitly_post_open() -> None:
    result = _load("development_evaluation.json")

    assert _sha256("development_evaluation.json") == (
        "49007cc03f2ed10e59e3aa2588f2a5130b70d047e919d75200c2769143bf3c71"
    )
    assert result["claim_status"] == (
        "post-open source plus public-paired-v1 development; fresh outcomes required"
    )
    assert result["development_gate_passed"] is True
    assert all(result["development_gates"].values())
    assert result["feature_names"] == list(FEATURE_NAMES)
    assert result["evidence"] == {
        "alpha_scale_control_sha256": (
            "1ecf48c0af7c08e10edd9628619187d475c178ff55679b247ca4799fd821ac18"
        ),
        "public_paired_raw_rows_sha256": (
            "45297345e9e9b366b23031655251b77ace8e96ce3c9570bfe48575f9c8186494"
        ),
        "public_paired_result_sha256": (
            "d3a03ca0c5f834cb5ae8fff840027e4ae919d3f94451a11847ab0b319221bb3c"
        ),
        "source_result_sha256": (
            "0075c331fc23ffadb2e9ebdd4b58093c76d25ce39c2bcf33e84d80d50a338bda"
        ),
        "source_rows_sha256": (
            "49e21b7aa6bda47b62b6d4475a8afaa4ebf73d1a737109a44630c5bc956f6ddb"
        ),
    }


def test_cross_fitted_guard_has_no_object_regressions() -> None:
    result = _load("development_evaluation.json")
    cross_fit = result["leave_one_physical_object_out"]

    assert cross_fit["source"]["object_count"] == 9
    assert cross_fit["source"]["object_wins"] == 9
    assert cross_fit["source"]["object_losses"] == 0
    assert cross_fit["public_paired_v1"]["object_count"] == 15
    assert cross_fit["public_paired_v1"]["object_wins"] == 11
    assert cross_fit["public_paired_v1"]["object_ties"] == 4
    assert cross_fit["public_paired_v1"]["object_losses"] == 0
    assert cross_fit["audit"]["false_safe_rate"] == pytest.approx(
        0.06985294117647059
    )
    assert cross_fit["audit"]["upper_coverage"] == pytest.approx(
        0.897226173541963
    )


def test_deployment_certificate_and_fallback_inventory_are_frozen() -> None:
    result = _load("development_evaluation.json")
    deployment = result["deployment_fit"]
    public = deployment["public_paired_v1"]
    certificate = certificate_from_payload(deployment["certificate"])

    assert certificate.source_group_count == 16
    assert certificate.finite_sample_rank == 14
    assert certificate.finite_sample_coverage == pytest.approx(14 / 17)
    assert public["object_count"] == 15
    assert public["object_wins"] == 12
    assert public["object_ties"] == 3
    assert public["object_losses"] == 0
    assert public["supported_object_count"] == 12
    assert public["object_balanced_relative_improvement"] == pytest.approx(
        0.003884689045054804
    )
    volleyball = next(
        row for row in public["objects"] if row["object"] == "PlushVolleyball"
    )
    assert volleyball["accepted_frame_count"] == 0
    assert volleyball["guarded_CD_UL1_mm"] == volleyball["baseline_CD_UL1_mm"]


def test_global_scale_control_reproduces_v1_and_retains_a_loss() -> None:
    result = _load("development_evaluation.json")
    control = result["global_scale_control"]

    assert control["maximum_alpha_one_reproduction_error_mm"] == 0.0
    for alpha in ("0.25", "0.5", "0.75", "1.0"):
        assert control["summary"][alpha]["losses"] == 1
