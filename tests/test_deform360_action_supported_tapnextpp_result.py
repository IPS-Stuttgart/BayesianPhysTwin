from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_action_supported_tapnextpp import (
    validate_action_supported_provider_artifacts,
    validate_action_supported_query_artifacts,
)

ROOT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "sota"
    / "diagnostics"
    / "deform360_action_supported_tapnextpp_source_v11"
)
EXPECTED_RESULT_SHA256 = (
    "9321bcc81e8833c6c80904f1e984dd89b44026540571764d78fb12a4ae39b0a6"
)
EXPECTED_BARRIER_SHA256 = (
    "06767567f48fe9b10c21eeaa9fe52bd6ab4d15be402b18834128a59b39a48280"
)


def _canonical(payload: dict[str, object], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_frozen_v11_source_result_and_prediction_carriers() -> None:
    result = json.loads(
        (ROOT / "evaluation" / "source_competence_result.json").read_text()
    )
    barrier = json.loads(
        (
            ROOT / "evaluation" / "prediction_completeness_barrier.json"
        ).read_text()
    )

    assert result["result_sha256"] == EXPECTED_RESULT_SHA256
    assert result["result_sha256"] == _canonical(result, "result_sha256")
    assert barrier["barrier_sha256"] == EXPECTED_BARRIER_SHA256
    assert barrier["barrier_sha256"] == _canonical(
        barrier,
        "barrier_sha256",
    )
    assert result["source_gate_passed"] is False
    assert result["decision"] == "stop_action_supported_tapnextpp_route"
    assert result["aggregate"]["provider_prediction_count"] == 8
    assert result["aggregate"]["supported_identity_count"] == 5
    assert result["aggregate"]["scheduled_identity_count"] == 64
    assert result["aggregate"]["pooled_supported_fraction"] == pytest.approx(
        0.078125
    )
    assert result["aggregate"][
        "object_balanced_provider_rmse_m"
    ] == pytest.approx(0.0065674363021174)
    assert result["aggregate"][
        "object_balanced_persistence_rmse_m"
    ] == pytest.approx(0.0007619157446403812)
    assert result["aggregate"][
        "relative_gain_over_persistence"
    ] == pytest.approx(-7.619635895852477)
    assert sum(result["gates"].values()) == 1
    assert result["gates"]["provider_prediction_count"] is True
    assert result["information_boundary"]["state_update_constructed"] is False
    assert (
        result["information_boundary"][
            "held_v8_artifact_or_process_access"
        ]
        is False
    )

    prediction_dirs = sorted(
        path for path in (ROOT / "predictions").iterdir() if path.is_dir()
    )
    assert len(prediction_dirs) == 8
    for case_dir in prediction_dirs:
        query_report, query_arrays = (
            validate_action_supported_query_artifacts(case_dir / "query")
        )
        provider_report, provider_arrays = (
            validate_action_supported_provider_artifacts(
                case_dir / "provider"
            )
        )
        assert query_report["status"] == "admitted"
        assert provider_report["status"] == (
            "prediction_sealed_before_identity_scoring"
        )
        assert len(query_arrays["entity_ids"]) == 8
        assert len(provider_arrays["entity_ids"]) == 8
        assert provider_report["information_boundary"][
            "identity_target_read"
        ] is False
        assert provider_report["information_boundary"][
            "state_update_constructed"
        ] is False
