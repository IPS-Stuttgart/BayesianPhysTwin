import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_alltrain_protocol,
)
from bayesian_phystwin.deform_dlo_official import (
    evaluate_deform_dlo2_official_uncertainty,
    load_deform_dlo2_official_protocol,
    summarize_deform_dlo2_official_records,
    validate_deform_dlo2_official_authorization,
)
from bayesian_phystwin.deform_dlo_source import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_official_eval_v1.json"
ALLTRAIN_PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_alltrain_refit_v1.json"
)
RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo2_official.py"


def _load_runner():
    scripts_root = str(RUNNER.parent)
    sys.path.insert(0, scripts_root)
    try:
        spec = importlib.util.spec_from_file_location("deform_dlo2_official_runner", RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_root)


def _authorization_artifacts():
    baseline = {
        "path": "/tmp/update_6400.pt",
        "sha256": "6" * 64,
        "size_bytes": 100,
        "update": 6400,
    }
    member = {
        "path": "/tmp/update_6040.pt",
        "sha256": "5" * 64,
        "size_bytes": 100,
        "update": 6040,
    }
    weights = {"6040": 0.4, "6400": 0.6}
    method_spec = {
        "contract": "deform-dlo2-alltrain-method-spec-v1",
        "official_eval_read": False,
        "operator": "predictive_mean",
        "checkpoint_weights": weights,
        "comparison_baseline_update": 6400,
        "validation_fitted_variance_scale": 2.0,
        "variance_floor_m2": 0.000025,
        "nominal_coordinate_coverage": 0.9,
    }
    final_method = {
        "contract": "deform-dlo2-alltrain-final-method-v1",
        "official_eval_read": False,
        "operator": "predictive_mean",
        "checkpoint_weights": weights,
        "comparison_baseline_checkpoint": baseline,
        "member_checkpoints": {"6040": member, "6400": baseline},
        "parameter_mean_checkpoint": None,
        "method_spec": {"path": "/tmp/method.json", "sha256": "b" * 64},
        "variance_calibration": {
            "scale": 2.0,
            "floor_m2": 0.000025,
            "nominal_coordinate_coverage": 0.9,
        },
    }
    alltrain_result = {
        "contract": "deform-dlo2-alltrain-result-v1",
        "official_eval_read": False,
        "official_eval_execution_authorized": True,
        "protocol": {"sha256": sha256_file(ALLTRAIN_PROTOCOL)},
        "method_spec": {"path": "/tmp/method.json", "sha256": "b" * 64},
        "final_method": {"path": "/tmp/final.json", "sha256": "c" * 64},
        "runtime": {"torch": "test", "cuda": "test"},
        "checkpoints": [member, baseline],
    }
    return alltrain_result, final_method, method_spec


def test_official_protocol_is_one_shot_and_target_blind() -> None:
    protocol = load_deform_dlo2_official_protocol(PROTOCOL)

    assert protocol["evaluation"]["expected_trajectory_count"] == 14
    assert protocol["evaluation"]["failure_policy"] == "seal-failure-no-retry-v1"
    assert protocol["methods"]["target_selection"] is False
    assert protocol["methods"]["target_calibration"] is False
    assert protocol["methods"]["target_retries"] is False
    assert protocol["methods"]["case_replacement"] is False


def test_official_protocol_rejects_target_selection(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["methods"]["target_selection"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="method policy"):
        load_deform_dlo2_official_protocol(changed)


def test_official_authorization_preserves_exact_alltrain_method() -> None:
    protocol = load_deform_dlo2_official_protocol(PROTOCOL)
    alltrain_protocol = load_deform_dlo2_alltrain_protocol(ALLTRAIN_PROTOCOL)
    alltrain_result, final_method, method_spec = _authorization_artifacts()

    selected = validate_deform_dlo2_official_authorization(
        protocol,
        alltrain_protocol,
        alltrain_result,
        final_method,
        method_spec,
        alltrain_protocol_sha256=sha256_file(ALLTRAIN_PROTOCOL),
        alltrain_result_sha256="a" * 64,
        final_method_sha256="c" * 64,
        method_spec_sha256="b" * 64,
    )

    assert selected["operator"] == "predictive_mean"
    assert selected["weights"] == {6040: 0.4, 6400: 0.6}
    assert selected["comparison_baseline_checkpoint"]["update"] == 6400
    assert selected["variance_scale"] == 2.0


def test_official_authorization_rejects_post_source_weight_change() -> None:
    protocol = load_deform_dlo2_official_protocol(PROTOCOL)
    alltrain_protocol = load_deform_dlo2_alltrain_protocol(ALLTRAIN_PROTOCOL)
    alltrain_result, final_method, method_spec = _authorization_artifacts()
    final_method["checkpoint_weights"] = {"6040": 0.5, "6400": 0.5}

    with pytest.raises(ValueError, match="frozen spec"):
        validate_deform_dlo2_official_authorization(
            protocol,
            alltrain_protocol,
            alltrain_result,
            final_method,
            method_spec,
            alltrain_protocol_sha256=sha256_file(ALLTRAIN_PROTOCOL),
            alltrain_result_sha256="a" * 64,
            final_method_sha256="c" * 64,
            method_spec_sha256="b" * 64,
        )


def test_official_authorization_accepts_frozen_parameter_mean() -> None:
    protocol = load_deform_dlo2_official_protocol(PROTOCOL)
    alltrain_protocol = load_deform_dlo2_alltrain_protocol(ALLTRAIN_PROTOCOL)
    alltrain_result, final_method, method_spec = _authorization_artifacts()
    parameter_mean = {
        "path": "/tmp/final_parameter_mean.pt",
        "sha256": "d" * 64,
        "size_bytes": 100,
    }
    final_method["operator"] = "parameter_mean"
    final_method["parameter_mean_checkpoint"] = parameter_mean
    method_spec["operator"] = "parameter_mean"

    selected = validate_deform_dlo2_official_authorization(
        protocol,
        alltrain_protocol,
        alltrain_result,
        final_method,
        method_spec,
        alltrain_protocol_sha256=sha256_file(ALLTRAIN_PROTOCOL),
        alltrain_result_sha256="a" * 64,
        final_method_sha256="c" * 64,
        method_spec_sha256="b" * 64,
    )

    assert selected["operator"] == "parameter_mean"
    assert selected["parameter_mean_checkpoint"] == parameter_mean


def _record(name: str, model: float, persistence: float = 0.02):
    return {
        "name": name,
        "model_l1_m": model,
        "persistence_l1_m": persistence,
        "early_l1_m": model * 0.5,
        "middle_l1_m": model,
        "late_l1_m": model * 1.5,
    }


def test_official_summary_requires_all_cases_and_passes_all_three_gates() -> None:
    names = [f"case-{index:02d}" for index in range(14)]
    candidate = [_record(name, 0.008) for name in names]
    baseline = [_record(name, 0.010) for name in names]

    summary = summarize_deform_dlo2_official_records(
        candidate,
        baseline,
        expected_case_count=14,
        published_reference_l1_m=0.0097,
        minimum_relative_improvement=0.01,
        minimum_case_wins=8,
    )

    assert summary["bayesian_case_wins"] == 14
    assert summary["candidate_horizon_l1_m"]["late"] == pytest.approx(0.012)
    assert summary["claim_gate"]["passed"] is True

    with pytest.raises(ValueError, match="frozen cohort"):
        summarize_deform_dlo2_official_records(
            candidate[:-1],
            baseline[:-1],
            expected_case_count=14,
            published_reference_l1_m=0.0097,
            minimum_relative_improvement=0.01,
            minimum_case_wins=8,
        )


def test_official_uncertainty_uses_fixed_scale_and_reports_horizons() -> None:
    predictions = np.zeros((2, 6, 3, 3), dtype=float)
    targets = np.full_like(predictions, 0.01)
    variance = np.full_like(predictions, 0.000025)

    metrics = evaluate_deform_dlo2_official_uncertainty(
        predictions,
        targets,
        variance,
        variance_floor_m2=0.000025,
        variance_scale=2.0,
        nominal_coverage=0.9,
    )

    assert metrics["overall"]["coordinate_nees"] == pytest.approx(2.0)
    assert set(metrics["horizon"]) == {"early", "middle", "late"}


def test_eval_manifest_is_sorted_and_rejects_partial_cohort(tmp_path: Path) -> None:
    runner = _load_runner()
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    for index in reversed(range(14)):
        (eval_root / f"case-{index:02d}.pkl").write_bytes(bytes([index]))

    manifest = runner._build_eval_manifest(
        eval_root,
        expected_count=14,
        protocol_path=PROTOCOL,
        alltrain_result_path=ALLTRAIN_PROTOCOL,
    )

    assert manifest["ordered_names"] == sorted(manifest["ordered_names"])
    (eval_root / "case-13.pkl").unlink()
    with pytest.raises(ValueError, match="expected 14"):
        runner._build_eval_manifest(
            eval_root,
            expected_count=14,
            protocol_path=PROTOCOL,
            alltrain_result_path=ALLTRAIN_PROTOCOL,
        )
