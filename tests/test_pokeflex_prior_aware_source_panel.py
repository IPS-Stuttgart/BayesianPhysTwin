from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SCRIPTS = REPOSITORY_ROOT / "scripts" / "remote"
if str(REMOTE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REMOTE_SCRIPTS))
if importlib.util.find_spec("trimesh") is None:
    trimesh_stub = types.ModuleType("trimesh")
    trimesh_stub.Trimesh = object
    sys.modules["trimesh"] = trimesh_stub

from run_pokeflex_prior_aware_belief_source_panel import (  # noqa: E402
    _load_panel_protocol,
    _summarize_panel,
    _surface_posterior_diagnostic,
    _take_suffix,
)


def _protocol() -> dict[str, object]:
    path = (
        REPOSITORY_ROOT
        / "configs"
        / "sota"
        / "pokeflex_prior_aware_belief_source_panel_v1.json"
    )
    return _load_panel_protocol(path)


def _take_result(
    take_id: str,
    *,
    baseline: float = 5.0,
    candidate: float = 4.9,
    admitted_harmful: bool = False,
) -> dict[str, object]:
    selected_frame = baseline + 0.1 if admitted_harmful else candidate
    return {
        "take_id": take_id,
        "aggregate": {
            "released_checkpoint_CD_UL1_mm": baseline,
            "prior_aware_selected_CD_UL1_mm": candidate,
        },
        "frames": [
            {
                "inference_admissible": True,
                "released_checkpoint_CD_UL1_mm": baseline,
                "prior_aware_selected_CD_UL1_mm": selected_frame,
                "surface_proxy_coverage_90": 0.9,
                "surface_proxy_NEES": 3.0,
            }
        ],
    }


def test_panel_protocol_is_valid_json_and_binds_unchanged_smoke() -> None:
    protocol = _protocol()
    smoke_path = (
        REPOSITORY_ROOT
        / "configs"
        / "sota"
        / "pokeflex_prior_aware_belief_source_smoke_v1.json"
    )
    json.loads(smoke_path.read_text(encoding="utf-8"))
    assert protocol["method_lock"]["post_smoke_parameter_changes"] is False
    assert protocol["cohort"]["calibration_or_target_objects_allowed"] is False


def test_take_suffix_rejects_noncanonical_identity() -> None:
    assert _take_suffix("FoamDice_T3") == "T3"
    try:
        _take_suffix("FoamDice-3")
    except ValueError as error:
        assert "invalid PokeFlex take id" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("noncanonical take id was accepted")


def test_surface_posterior_diagnostic_uses_metric_covariance() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    covariance = np.zeros((2, 3, 3), dtype=np.float64)
    coverage, nees = _surface_posterior_diagnostic(points, covariance, points)
    assert coverage == 1.0
    assert nees == 0.0

    shifted = points + np.asarray([0.02, 0.0, 0.0])
    shifted_coverage, shifted_nees = _surface_posterior_diagnostic(
        shifted,
        covariance,
        points,
    )
    assert shifted_coverage < coverage
    assert shifted_nees > nees


def test_complete_positive_panel_passes_registered_gate() -> None:
    protocol = _protocol()
    results = [
        _take_result(f"{object_name}_{suffix}")
        for object_name in protocol["cohort"]["development_objects"]
        for suffix in protocol["cohort"]["take_suffixes"]
    ]
    summary = _summarize_panel(protocol, results)
    assert summary["cohort_complete"] is True
    assert summary["gate_passed"] is True
    assert summary["object_balanced"]["object_wins"] == 5


def test_harmful_admissions_fail_false_safe_gate() -> None:
    protocol = _protocol()
    results = [
        _take_result(
            f"{object_name}_{suffix}",
            admitted_harmful=True,
        )
        for object_name in protocol["cohort"]["development_objects"]
        for suffix in protocol["cohort"]["take_suffixes"]
    ]
    summary = _summarize_panel(protocol, results)
    assert (
        summary["gate_checks"]["maximum_false_safe_rate_among_admitted_frames"] is False
    )
    assert summary["gate_passed"] is False
