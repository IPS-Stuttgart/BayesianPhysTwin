from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "science" / "run_full22_endpoint_model_average_diagnostic.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("full22_model_average", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_last_valid_residual_uses_latest_supported_frame() -> None:
    module = _load_script()
    residual = np.array(
        [
            [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    valid = np.array(
        [
            [True, False],
            [False, True],
            [True, False],
        ]
    )

    result = module._last_valid_residual(residual, valid, end_frame=3)

    np.testing.assert_allclose(result, [[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])


def test_predictive_events_match_identity_covariance() -> None:
    module = _load_script()
    errors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    covariance = np.repeat(np.eye(3)[None], 2, axis=0)

    events = module._regularized_predictive_events(errors, covariance)

    np.testing.assert_allclose(events["nees"], [1.0, 4.0])
    np.testing.assert_allclose(events["predictive_std_m"], [1.0, 1.0])
    summary = module._summarize_event_arrays(events)
    assert summary["count"] == 2
    assert summary["mean_nees"] == pytest.approx(2.5)
    assert summary["coverage_90"] == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic_and_preserves_direction() -> None:
    module = _load_script()
    candidate = np.array([1.0, 2.0, 3.0])
    reference = np.array([2.0, 3.0, 4.0])

    first = module._paired_bootstrap(
        candidate,
        reference,
        samples=200,
        seed=7,
    )
    second = module._paired_bootstrap(
        candidate,
        reference,
        samples=200,
        seed=7,
    )

    assert first == second
    assert first["mean_delta_m"] == pytest.approx(-1.0)
    assert first["candidate_win_count"] == 3
    assert first["bootstrap_probability_mean_improvement"] == pytest.approx(1.0)


def test_horizon_groups_cover_each_frame_exactly_once() -> None:
    module = _load_script()

    groups = module._horizon_groups(8)

    combined = np.concatenate(list(groups.values()))
    np.testing.assert_array_equal(np.sort(combined), np.arange(8))
    assert len(np.unique(combined)) == 8


def test_case_csv_uses_repository_stable_lf_line_endings(
    tmp_path: Path,
) -> None:
    module = _load_script()
    point = {
        method: {
            "chamfer_distance_m": 1.0,
            "track_error_m": 2.0,
        }
        for method in module.METHODS
    }
    output = tmp_path / "per_case.csv"

    module._write_case_csv(
        output,
        {
            "case": {
                "cohort": "confirmation",
                "anchor_validation": {"accepted": True},
                "point": point,
            }
        },
    )

    raw = output.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2
