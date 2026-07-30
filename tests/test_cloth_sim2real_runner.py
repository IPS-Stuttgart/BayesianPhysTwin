from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


def _runner():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "held"
        / "run_cloth_sim2real_v1.py"
    )
    spec = importlib.util.spec_from_file_location("cloth_sim2real_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_authorization_is_explicit(tmp_path: Path) -> None:
    runner = _runner()
    source_gate = tmp_path / "source.json"
    source_gate.write_text(
        json.dumps(
            {
                "artifact_kind": "ClothSim2RealSourceGate",
                "calibration_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    calibration_gate = tmp_path / "calibration.json"
    calibration_gate.write_text(
        json.dumps(
            {
                "artifact_kind": "ClothSim2RealCalibrationGate",
                "target_authorized": True,
            }
        ),
        encoding="utf-8",
    )

    assert runner._authorization("source", None) == (None, None)
    assert runner._authorization("calibration", source_gate)[0] == str(
        source_gate.resolve()
    )
    assert runner._authorization("target", calibration_gate)[0] == str(
        calibration_gate.resolve()
    )
    with pytest.raises(ValueError, match="require an authorization"):
        runner._authorization("target", None)


def test_calibration_aggregate_freezes_accuracy_and_scale(tmp_path: Path) -> None:
    runner = _runner()
    source_gate = tmp_path / "source_gate.json"
    source_gate.write_text(
        json.dumps(
            {
                "artifact_kind": "ClothSim2RealSourceGate",
                "calibration_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    improvements = {
        "chequered_rag_1/dynamic": 0.08,
        "cotton_rag_1/dynamic": 0.06,
        "linen_rag_1/dynamic": 0.02,
        "chequered_rag_1/quasi_static": 0.01,
        "cotton_rag_1/quasi_static": -0.02,
        "linen_rag_1/quasi_static": 0.03,
    }
    for index, (case_id, improvement) in enumerate(improvements.items()):
        case_dir = results_root / case_id.replace("/", "_")
        case_dir.mkdir(parents=True)
        (case_dir / "result.json").write_text(
            json.dumps(
                {
                    "authorized_split": "calibration",
                    "case_id": case_id,
                    "metrics": {
                        "symmetric_relative_improvement": improvement,
                        "trial_coordinate_abs_standardized_q90": 2.0 + index,
                    },
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "calibration_gate.json"

    runner._aggregate_calibration(
        argparse.Namespace(
            source_gate=source_gate,
            results_root=results_root,
            output=output,
        )
    )
    gate = json.loads(output.read_text(encoding="utf-8"))

    assert gate["calibration_accuracy_gate_passed"] is True
    assert gate["target_authorized"] is True
    assert gate["formal_90_split_conformal_claim"] is False
    assert gate["uncertainty_std_multiplier"] == pytest.approx(
        7.0 / 1.6448536269514722
    )
