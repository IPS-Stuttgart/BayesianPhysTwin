from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_reusable_association import (
    CANONICAL_REUSABLE_ASSOCIATION_CONFIG_SHA256,
    load_reusable_association_config,
    load_reusable_association_source_evidence,
    validate_reusable_association_config,
)


CONFIG = (
    Path(__file__).parents[1]
    / "configs/causal4d_public/deform360_reusable_association_v2.json"
)
SOURCE_EVIDENCE = (
    Path(__file__).parents[1]
    / "milestones/deform360-reusable-association-v2-source/artifacts/source_evidence.json"
)


def test_canonical_reusable_association_config_is_locked() -> None:
    payload = load_reusable_association_config(CONFIG)

    result = validate_reusable_association_config(payload)

    assert result["config_sha256"] == CANONICAL_REUSABLE_ASSOCIATION_CONFIG_SHA256
    assert (
        payload["config"]["calibration_gate"]["future_prediction_metrics_allowed"]
        is False
    )


def test_reusable_association_rejects_future_or_residual_leakage() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    for section, key in (
        ("mask_candidates", "simulator_residual_allowed"),
        ("mask_candidates", "future_frame_allowed"),
        ("calibration_gate", "future_prediction_metrics_allowed"),
    ):
        changed = copy.deepcopy(payload)
        changed["config"][section][key] = True
        with pytest.raises(ValueError, match="checksum mismatch"):
            validate_reusable_association_config(changed)


def test_source_evidence_is_checksummed_and_claim_bounded() -> None:
    payload = load_reusable_association_source_evidence(SOURCE_EVIDENCE)

    assert len(payload["mask_cases"]) == 7
    assert payload["conclusion"]["source_mask_gate_passed"] is True
    assert payload["conclusion"]["state_of_the_art_claim"] is False
