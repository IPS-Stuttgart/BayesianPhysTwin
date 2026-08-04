from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "science" / "run_full22_endpoint_model_average_diagnostic.py"
)
RESULT_ROOT = ROOT / "results" / "diagnostics" / "full22_endpoint_model_average_v1"
PROTOCOL = ROOT / "protocols" / "full22_endpoint_model_average_diagnostic_v1.json"
PROTOCOL_SHA256 = "8c4021f082b03ef761bc97300eeac11b6f3f92a2bdc52c1941020f6c1f340217"
HOSTED_CSV_SHA256 = "19184cfefe707ed49739a18ee667402cfea24b46297f0217d2edcd85d5fc3b31"
REPOSITORY_CSV_SHA256 = "ac37b5004987f94145dbca6ea8e08d60582f4fac323a5b5ae9cdcaa578eafa1d"


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


def test_committed_evidence_has_locked_byte_identity_and_claim_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_digest = hashlib.sha256(
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert protocol_digest == PROTOCOL_SHA256

    manifest = json.loads(
        (RESULT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol_sha256"] == PROTOCOL_SHA256
    assert manifest["repository_per_case_csv_line_endings"] == "LF"

    per_case = RESULT_ROOT / "per_case.csv"
    raw = per_case.read_bytes()
    assert b"\r\n" not in raw
    repository_digest = hashlib.sha256(raw).hexdigest()
    hosted_digest = hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()
    assert repository_digest == REPOSITORY_CSV_SHA256
    assert repository_digest == manifest["repository_per_case_csv_sha256"]
    assert hosted_digest == HOSTED_CSV_SHA256
    assert hosted_digest == manifest["hosted_per_case_csv_sha256"]
    assert hosted_digest == manifest["per_case_csv_sha256"]

    with per_case.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 22

    readout = json.loads((RESULT_ROOT / "readout.json").read_text(encoding="utf-8"))
    assert readout["protocol_sha256"] == PROTOCOL_SHA256
    assert readout["classification"] == "retrospective-non-claim-bearing-diagnostic"
    assert readout["confirmation_19"]["case_count"] == 19
