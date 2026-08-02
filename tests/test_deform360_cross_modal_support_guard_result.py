from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results/sota/diagnostics/deform360_cross_modal_support_guard_source_v2/result.json"
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    stripped = dict(payload)
    stripped.pop("artifact_sha256", None)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def test_registered_cross_modal_support_source_v2_result() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["artifact_sha256"] == _canonical_sha256(payload)
    assert payload["artifact_sha256"] == (
        "9568028f00a93ee74900145ad2da73ac51d01db0751a725dd024a28224166e11"
    )
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "1c8e8ccabc24f5dd12733de26d4efce09e4a8f35a318e96fe23df9c037da5078"
    )

    cross_fitted = payload["cross_fitted"]
    combined = cross_fitted["combined"]
    open27 = cross_fitted["open27"]
    stress12 = cross_fitted["stress12"]
    assert combined["accepted_beneficial_update_count"] == 14
    assert combined["accepted_regressive_update_count"] == 0
    assert combined["admitted_object_count"] == 5
    assert cross_fitted["joint_case_wins"] == 8
    assert open27["metrics"]["identity"]["guarded_relative_percent"] == pytest.approx(
        -6.971066937671322
    )
    assert open27["metrics"]["chamfer"]["guarded_relative_percent"] == pytest.approx(
        -5.467690103463219
    )
    assert stress12["accepted_update_count"] == 0
    assert payload["all_development_advancement_checks_passed"] is True
    assert all(payload["development_advancement_checks"].values())

    model = payload["full_source_model_for_future_lock"]
    route = model["stable_tactile_coherent_correction"]
    assert route["maximum_cumulative_energy_change"] == pytest.approx(
        -0.0009105394946505805
    )
    assert route["minimum_correction_coherence"] == pytest.approx(
        0.8014452753318334
    )
    assert route["source_regressive_admission_count"] == 0
    assert payload["predecessor"]["preserved_without_relabeling_noops"] is True
