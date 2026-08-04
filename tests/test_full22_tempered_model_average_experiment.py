from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "science" / "run_full22_tempered_model_average_experiment.py"
)
PROTOCOL = ROOT / "protocols" / "full22_tempered_model_average_experiment_v1.json"
PROTOCOL_SHA256 = "a351cf37ba19130feca4dcfb87b1e7ab9a2e601d22edeed3a39a00c904ecbbe3"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("full22_tempered_experiment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_development(module: ModuleType) -> dict[str, object]:
    scores = {
        "single_lift_sloth": 0.001,
        "double_lift_sloth": 0.002,
        "double_stretch_sloth": 0.003,
    }
    losses = {
        "single_lift_sloth": 0.8,
        "double_lift_sloth": 0.9,
        "double_stretch_sloth": 1.2,
    }
    quantiles = {
        "single_lift_sloth": 10.0,
        "double_lift_sloth": 12.0,
        "double_stretch_sloth": 11.0,
    }
    result: dict[str, object] = {}
    for case in module.DEVELOPMENT_CASES:
        result[case] = {
            "temperature_candidates": {
                "2": {
                    "guard_score_m": scores[case],
                    "combined_point_loss_vs_last_residual": losses[case],
                    "group_quantiles": {
                        label: {"count": 10, "quantile": quantiles[case]}
                        for label in module.HORIZON_LABELS
                    },
                }
            }
        }
    return result


def test_locked_protocol_digest_and_claim_boundary() -> None:
    module = _load_script()
    payload, digest = module._load_protocol(PROTOCOL)

    assert digest == PROTOCOL_SHA256
    assert payload["status"] == "retrospective-non-claim-bearing"
    assert payload["temperature_selection"]["uses_confirmation_outcomes"] is False
    assert payload["regret_guard"]["uses_confirmation_outcomes"] is False
    assert payload["group_conformal"]["uses_confirmation_outcomes"] is False


def test_temperature_one_reproduces_untempered_moments() -> None:
    module = _load_script()
    residual = np.zeros((4, 2, 3), dtype=np.float64)
    residual[:, 0, 0] = [0.0, 0.001, 0.002, 0.003]
    residual[:, 1, 1] = [0.0, -0.001, -0.002, -0.003]
    valid = np.ones((4, 2), dtype=bool)
    posterior = module.infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=4,
    )

    tempered = module._tempered_posterior(posterior, temperature=1.0)

    np.testing.assert_allclose(tempered.component_weights, posterior.component_weights)
    np.testing.assert_allclose(tempered.mean_m, posterior.mean_m)
    np.testing.assert_allclose(tempered.covariance_m2, posterior.covariance_m2)
    assert not hasattr(tempered, "final_nominal_probability")
    baseline_prediction = module.predict_model_averaged_endpoint(
        posterior,
        horizon_steps=3,
    )
    mean, covariance = module._predict_tempered_endpoint(
        tempered,
        horizon_steps=3,
    )
    np.testing.assert_allclose(mean, baseline_prediction.mean_m)
    np.testing.assert_allclose(covariance, baseline_prediction.covariance_m2)


def test_higher_temperature_flattens_nonuniform_component_weights() -> None:
    module = _load_script()
    residual = np.zeros((6, 1, 3), dtype=np.float64)
    residual[:, 0, 0] = np.linspace(0.0, 0.01, 6)
    valid = np.ones((6, 1), dtype=bool)
    posterior = module.infer_model_averaged_endpoint(
        residual,
        valid,
        end_frame=6,
    )

    cold = module._tempered_posterior(posterior, temperature=1.0)
    warm = module._tempered_posterior(posterior, temperature=64.0)
    cold_entropy = -np.sum(
        cold.component_weights * np.log(np.maximum(cold.component_weights, 1e-300))
    )
    warm_entropy = -np.sum(
        warm.component_weights * np.log(np.maximum(warm.component_weights, 1e-300))
    )

    assert warm_entropy >= cold_entropy


def test_finite_sample_higher_quantile_uses_conformal_rank() -> None:
    module = _load_script()
    values = np.arange(1.0, 11.0)

    result = module._finite_sample_higher_quantile(values, coverage=0.9)

    assert result == 10.0


def test_guard_selects_only_zero_regret_prefix() -> None:
    module = _load_script()
    development = _synthetic_development(module)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    selection = module._select_guard(
        development,
        protocol,
        temperature=2.0,
    )

    assert selection["selected_threshold_m"] == pytest.approx(0.002)
    selected = selection["selected_record"]
    assert selected["accepted_case_count"] == 2
    assert selected["maximum_case_relative_regret"] <= 0.0


def test_group_conformal_uses_worst_development_case() -> None:
    module = _load_script()
    development = _synthetic_development(module)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    selection = module._fit_group_conformal(
        development,
        protocol,
        temperature=2.0,
    )

    reference = selection["chi_square_reference"]
    expected = max(1.0, 12.0 / reference)
    for label in module.HORIZON_LABELS:
        assert selection["scales_by_horizon"][label] == pytest.approx(expected)


def test_unavailable_guard_score_fails_closed() -> None:
    module = _load_script()

    assert module._guard_accepts(None, 0.1) is False
    assert module._guard_accepts(0.01, None) is False
    assert module._guard_accepts(0.01, 0.02) is True


def test_case_csv_uses_lf_line_endings(tmp_path: Path) -> None:
    module = _load_script()
    point = {
        method: {
            "chamfer_distance_m": 1.0,
            "track_error_m": 2.0,
        }
        for method in module.POINT_METHODS
    }
    output = tmp_path / "per_case.csv"
    module._write_case_csv(
        output,
        {
            "case": {
                "cohort": "confirmation",
                "temperature": 2.0,
                "guard": {
                    "score_m": 0.001,
                    "threshold_m": 0.002,
                    "accepted": True,
                },
                "point": point,
            }
        },
    )

    raw = output.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2
