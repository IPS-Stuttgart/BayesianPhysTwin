import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    canonical_payload_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "sota" / "pokeflex_fresh12_scale_headroom_v1" / "audit.json"


def test_scale_headroom_artifact_is_bound_and_scoped() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))

    assert hashlib.sha256(AUDIT.read_bytes()).hexdigest() == (
        "114fbf7c3437311f625c9c022742a2c707d6db1e61ca1548b0d8f2500f83d494"
    )
    assert payload["audit_sha256"] == canonical_payload_sha256(
        payload,
        digest_field="audit_sha256",
    )
    assert payload["status"] == "post-open diagnostic; not prospective evidence"
    assert len(payload["takes"]) == 12


def test_sealed_scale_is_safe_and_larger_uniform_scales_are_not() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    by_multiplier = {
        float(row["multiplier"]): row for row in payload["uniform_scale_results"]
    }

    sealed = by_multiplier[1.0]
    assert sealed["object_balanced_CD_UL1_mm"] == 4.888144286804533
    assert sealed["object_loss_count"] == 0
    assert by_multiplier[1.5]["object_loss_count"] == 2
    assert by_multiplier[2.0]["object_loss_count"] == 2
    assert payload["postopen_best_uniform"] == by_multiplier[2.0]


def test_postopen_oracles_are_reported_only_as_headroom() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    baseline = np.asarray(
        [row["mean_CD_UL1_mm_by_multiplier"]["0.0"] for row in payload["takes"]]
    )
    per_take_oracle = np.asarray(
        [row["postopen_best_CD_UL1_mm"] for row in payload["takes"]]
    )
    relative = float((np.mean(baseline) - np.mean(per_take_oracle)) / np.mean(baseline))

    assert relative == 0.02392103172944582
    assert payload["per_frame_scale_oracle_relative_improvement"] == (
        0.02956184512432954
    )
    assert "No multiplier selected here may be claimed" in payload["claim_boundary"]
