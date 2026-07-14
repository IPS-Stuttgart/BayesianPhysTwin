from __future__ import annotations

import pytest

from causal4d_public.deform360_sam2_suffix import (
    sam2_suffix_mask_artifact_sha256,
    validate_sam2_suffix_mask_artifact,
)


def test_suffix_mask_artifact_requires_prior_prediction_seal() -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360TargetRopeSam2PostSealMaskAudit",
        "outputs": [],
        "information_boundary": {
            "deployable_predictions_previously_sealed": True,
            "target_tactile_oracle_read": False,
        },
    }
    payload["result_sha256"] = sam2_suffix_mask_artifact_sha256(payload)

    assert validate_sam2_suffix_mask_artifact(payload, verify_outputs=False)["passed"]
    payload["information_boundary"]["deployable_predictions_previously_sealed"] = False
    payload["result_sha256"] = sam2_suffix_mask_artifact_sha256(payload)
    with pytest.raises(ValueError, match="before prediction sealing"):
        validate_sam2_suffix_mask_artifact(payload, verify_outputs=False)
