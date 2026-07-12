from __future__ import annotations

from copy import deepcopy

import pytest

from causal4d.preacquisition_protocol import build_preacquisition_amendment
from causal4d.preacquisition_protocol_v3 import (
    build_preacquisition_v3,
    preacquisition_v3_sha256,
    validate_preacquisition_v3,
)
from causal4d.real_protocol import build_same_object_real_protocol


def test_v3_preserves_acquisition_and_crossfits_every_source_session() -> None:
    protocol = build_same_object_real_protocol()
    v2 = build_preacquisition_amendment(protocol)
    v3 = build_preacquisition_v3(protocol, v2)

    result = validate_preacquisition_v3(v3, protocol, v2)
    assert result["passed"] is True
    assert result["physical_execution_count_changed"] is False
    assert result["source_crossfit_fold_count"] == 3
    assert v3["calibration_resolution"][
        "selected_threshold_is_maximum_calibration_score"
    ]
    assert v3["source_panel_role"]["confirmatory_claim_allowed"] is False


def test_v3_rejects_in_sample_shrinkage_boundary() -> None:
    protocol = build_same_object_real_protocol()
    v2 = build_preacquisition_amendment(protocol)
    v3 = build_preacquisition_v3(protocol, v2)
    mutated = deepcopy(v3)
    first = mutated["source_panel_crossfit"]["folds"][0]
    first["mechanism_fit_execution_ids"].append(
        first["heldout_shrinkage_execution_ids"][0]
    )
    mutated["amendment_sha256"] = preacquisition_v3_sha256(mutated)

    with pytest.raises(ValueError, match="locked canonical design|cross-fit fold"):
        validate_preacquisition_v3(mutated, protocol, v2)
