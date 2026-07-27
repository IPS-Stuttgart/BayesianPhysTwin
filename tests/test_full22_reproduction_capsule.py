from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1] / "reproductions" / "full22_anchor_v1" / "reproduce.py"
)
EXPECTED_PATH = SCRIPT_PATH.with_name("expected_metrics.json")


def _load_capsule() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "full22_reproduction_capsule", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(expected: dict[str, object]) -> dict[str, object]:
    methods: dict[str, object] = {}
    expected_methods = expected["methods"]
    assert isinstance(expected_methods, dict)
    for method, method_record in expected_methods.items():
        assert isinstance(method_record, dict)
        equal_case = method_record["equal_case"]
        frame_weighted = method_record["frame_weighted"]
        assert isinstance(equal_case, dict)
        assert isinstance(frame_weighted, dict)
        methods[method] = {
            "cohorts": {
                "all_22_table_compatible": {
                    metric: {
                        "equal_case_mean_m": equal_case[metric],
                        "frame_weighted_mean_m": frame_weighted[metric],
                    }
                    for metric in equal_case
                }
            }
        }
    return {"schema_version": 2, "methods": methods}


def test_full22_expected_metrics_are_verified() -> None:
    capsule = _load_capsule()
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))

    report = capsule.verify_comparison(_comparison(expected), expected)

    assert report["status"] == "verified"
    assert report["check_count"] == 8
    assert all(record["passed"] for record in report["checks"])


def test_full22_metric_drift_fails_closed() -> None:
    capsule = _load_capsule()
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    comparison = _comparison(expected)
    comparison["methods"]["bayesian_anchor"]["cohorts"]["all_22_table_compatible"][
        "track_error_m"
    ]["equal_case_mean_m"] += 1e-4

    with pytest.raises(ValueError, match="metric verification failed"):
        capsule.verify_comparison(comparison, expected)


def test_confirmation_summary_requires_protocol_and_complete_cohort() -> None:
    capsule = _load_capsule()
    summary = {
        "protocol_id": capsule.EXPECTED_PROTOCOL_ID,
        "case_results": {f"case-{index:02d}": {} for index in range(22)},
    }
    capsule.verify_confirmation_summary(summary)

    summary["protocol_id"] = "changed"
    with pytest.raises(ValueError, match="protocol ID changed"):
        capsule.verify_confirmation_summary(summary)


def test_data_manifest_validation_binds_digest_and_case_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _load_capsule()
    cases = [f"case-{index:02d}" for index in range(22)]
    manifest = tmp_path / "evaluation_subset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "selected_cases": cases,
                "available_cases": cases,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capsule, "EXPECTED_DATA_MANIFEST_SHA256", capsule._sha256(manifest)
    )

    assert capsule.validate_data_root(tmp_path) == manifest

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["available_cases"] = list(reversed(cases))
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        capsule, "EXPECTED_DATA_MANIFEST_SHA256", capsule._sha256(manifest)
    )
    with pytest.raises(ValueError, match="available_cases"):
        capsule.validate_data_root(tmp_path)


def test_manifest_command_binds_claim_protocol_and_outputs(tmp_path: Path) -> None:
    capsule = _load_capsule()
    command = capsule._manifest_command(tmp_path, tmp_path / "source", "run command")

    assert "bpt.full22_anchor_released_contract" in command
    assert capsule.EXPECTED_PROTOCOL_ID in command
    assert "full22_comparison=full22_comparison.json" in command
    assert "verification=verification.json" in command
