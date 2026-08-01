import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_alltrain import (
    load_deform_dlo2_deep_alltrain_protocol,
)
from bayesian_phystwin.deform_dlo_official import (
    DEFORM_DLO2_DEEP_OFFICIAL_CONTRACT,
    load_deform_dlo2_deep_official_protocol,
    validate_deform_dlo2_deep_official_authorization,
)
from bayesian_phystwin.deform_dlo_source import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform_dlo2_deep_official_eval_v1.json"
)
ALLTRAIN_PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform_dlo2_deep_alltrain_refit_v1.json"
)
RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "remote"
    / "run_deform_dlo2_deep_official.py"
)


def _load_runner():
    scripts_root = str(RUNNER.parent)
    sys.path.insert(0, scripts_root)
    try:
        spec = importlib.util.spec_from_file_location(
            "deform_dlo2_deep_official_runner", RUNNER
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_root)


def _authorization_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    members = {
        str(seed): {
            "path": f"/tmp/seed-{seed}.pt",
            "sha256": str(seed)[0] * 64,
            "size_bytes": 100,
            "update": 6400,
        }
        for seed in (42, 43)
    }
    seed_results = {
        str(seed): {
            "path": f"/tmp/seed-{seed}.json",
            "sha256": ("b" if seed == 42 else "c") * 64,
        }
        for seed in (42, 43)
    }
    final_method = {
        "contract": "deform-dlo2-deep-alltrain-final-method-v1",
        "official_eval_read": False,
        "operator": "predictive_mean",
        "seed_weights": {"42": 0.5, "43": 0.5},
        "member_updates": {"42": 6400, "43": 6400},
        "comparison_baseline_seed": 42,
        "comparison_baseline_checkpoint": members["42"],
        "member_checkpoints": members,
        "variance_calibration": {
            "scale": 2.0,
            "floor_m2": 0.000025,
            "nominal_coordinate_coverage": 0.9,
        },
        "seed_results": seed_results,
    }
    result = {
        "contract": "deform-dlo2-deep-alltrain-result-v1",
        "official_eval_read": False,
        "official_eval_execution_authorized": True,
        "protocol": {"sha256": sha256_file(ALLTRAIN_PROTOCOL)},
        "final_method": {"sha256": "d" * 64},
        "selected_method": {
            "operator": "predictive_mean",
            "seed_weights": {"42": 0.5, "43": 0.5},
            "member_updates": {"42": 6400, "43": 6400},
            "comparison_baseline_seed": 42,
            "variance_calibration": {
                "scale": 2.0,
                "floor_m2": 0.000025,
                "nominal_coordinate_coverage": 0.9,
            },
        },
        "seed_results": seed_results,
        "runtime": {"torch": "test", "cuda": "test"},
    }
    return result, final_method


def test_deep_official_protocol_is_one_shot_and_binds_alltrain() -> None:
    protocol = load_deform_dlo2_deep_official_protocol(PROTOCOL)

    assert protocol["contract"] == DEFORM_DLO2_DEEP_OFFICIAL_CONTRACT
    assert protocol["parent_alltrain_protocol"]["sha256"] == sha256_file(
        ALLTRAIN_PROTOCOL
    )
    assert protocol["parent_alltrain_protocol"]["sha256"] == sha256_file(
        ALLTRAIN_PROTOCOL
    )
    assert protocol["methods"]["target_selection"] is False
    assert protocol["methods"]["target_calibration"] is False
    assert protocol["methods"]["target_retries"] is False
    assert protocol["claim_gate"]["ensemble_minimum_case_wins"] == 8


def test_deep_official_authorization_preserves_two_seed_method() -> None:
    protocol = load_deform_dlo2_deep_official_protocol(PROTOCOL)
    alltrain_protocol = load_deform_dlo2_deep_alltrain_protocol(ALLTRAIN_PROTOCOL)
    result, final_method = _authorization_artifacts()

    selected = validate_deform_dlo2_deep_official_authorization(
        protocol,
        alltrain_protocol,
        result,
        final_method,
        alltrain_protocol_sha256=sha256_file(ALLTRAIN_PROTOCOL),
        alltrain_result_sha256="a" * 64,
        final_method_sha256="d" * 64,
    )

    assert selected["weights"] == {42: 0.5, 43: 0.5}
    assert selected["member_updates"] == {42: 6400, 43: 6400}
    assert selected["comparison_baseline_seed"] == 42
    assert selected["variance_scale"] == 2.0


def test_deep_official_authorization_rejects_changed_weight_or_baseline() -> None:
    protocol = load_deform_dlo2_deep_official_protocol(PROTOCOL)
    alltrain_protocol = load_deform_dlo2_deep_alltrain_protocol(ALLTRAIN_PROTOCOL)
    result, final_method = _authorization_artifacts()
    kwargs = {
        "alltrain_protocol_sha256": sha256_file(ALLTRAIN_PROTOCOL),
        "alltrain_result_sha256": "a" * 64,
        "final_method_sha256": "d" * 64,
    }

    final_method["seed_weights"] = {"42": 0.6, "43": 0.4}
    with pytest.raises(ValueError, match="member bank"):
        validate_deform_dlo2_deep_official_authorization(
            protocol,
            alltrain_protocol,
            result,
            final_method,
            **kwargs,
        )

    result, final_method = _authorization_artifacts()
    final_method["comparison_baseline_checkpoint"] = final_method[
        "member_checkpoints"
    ]["43"]
    with pytest.raises(ValueError, match="member bank"):
        validate_deform_dlo2_deep_official_authorization(
            protocol,
            alltrain_protocol,
            result,
            final_method,
            **kwargs,
        )


def test_deep_official_protocol_rejects_target_retry(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["methods"]["target_retries"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="method policy"):
        load_deform_dlo2_deep_official_protocol(changed)


def test_deep_official_failure_seal_forbids_retry() -> None:
    runner = _load_runner()
    failure = runner._failure_payload(
        stage="fixed-rollouts", error=RuntimeError("failed")
    )

    assert failure["official_eval_read"] is True
    assert failure["retry_authorized"] is False
    assert failure["stage"] == "fixed-rollouts"
