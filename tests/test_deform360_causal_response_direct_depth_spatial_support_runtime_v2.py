from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_spatial_support_runtime_v2 import (
    RUNTIME_NAMESPACE,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_spatial_support_runtime_v2.json"
)


def test_sparse_spatial_runtime_is_crash_to_rejection_only() -> None:
    payload = json.loads(RUNTIME.read_text(encoding="utf-8"))
    registered = payload.pop("config_sha256")
    observed = hashlib.sha256(
        RUNTIME_NAMESPACE
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert registered == observed
    assert (
        payload["amendment"]["adequately_supported_cases_numerically_changed"] is False
    )
    assert payload["amendment"]["gate_threshold_changed"] is False
    assert payload["amendment"]["crash_becomes_registered_rejection"] is True


def test_sparse_spatial_runtime_preserves_prediction_boundary() -> None:
    payload = json.loads(RUNTIME.read_text(encoding="utf-8"))
    assert payload["trigger"]["prediction_artifact_created"] is False
    assert payload["trigger"]["source_outcome_read"] is False
    assert payload["information_boundary"]["permitted_prefix_read"] is True
    assert payload["information_boundary"]["future_object_observation_read"] is False
    assert payload["information_boundary"]["source_outcome_read"] is False
