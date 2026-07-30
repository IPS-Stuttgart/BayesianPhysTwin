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


def test_target_aggregate_binds_results_and_keeps_tasks_separate(
    tmp_path: Path,
) -> None:
    runner = _runner()
    calibration_gate = tmp_path / "calibration_gate.json"
    calibration_gate.write_text(
        json.dumps(
            {
                "artifact_kind": "ClothSim2RealCalibrationGate",
                "target_authorized": True,
                "uncertainty_std_multiplier": 3.0,
            }
        ),
        encoding="utf-8",
    )
    calibration_sha256 = runner._sha256(calibration_gate)
    target_lock = tmp_path / "target_lock.json"
    target_lock.write_text(
        json.dumps(
            {
                "protocol_id": "cloth-sim2real-online-belief-v1",
                "method_id": runner.METHOD_ID,
                "status": "pre_target_prefix_lock",
                "calibration_evidence": {
                    "calibration_gate_sha256": calibration_sha256,
                    "target_authorized": True,
                },
                "target_scope": {"case_count": 6},
            }
        ),
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    tasks = ("dynamic", "quasi_static")
    for cloth_index in range(3):
        for task in tasks:
            case_id = f"cloth_{cloth_index}/{task}"
            improvement = 0.1 if task == "dynamic" else -0.05
            case_dir = results_root / case_id.replace("/", "_")
            case_dir.mkdir(parents=True)
            metrics = {
                "physical_symmetric_l1_chamfer_m": 0.1,
                "candidate_symmetric_l1_chamfer_m": 0.09,
                "symmetric_relative_improvement": improvement,
                "directed_relative_improvement": improvement,
                "released_window_directed_relative_improvement": improvement,
                "hausdorff_relative_improvement": improvement,
                "raw_90_coordinate_coverage": 0.5,
                "reported_90_coordinate_coverage": 0.9,
                "mean_90_interval_width_m": 0.2,
                "mean_energy_score_m": 0.03,
                "mean_readout_correction_m": 0.01,
            }
            horizons = [
                {
                    "name": name,
                    "physical_symmetric_l1_chamfer_m": 0.1,
                    "candidate_symmetric_l1_chamfer_m": 0.09,
                    "relative_improvement": improvement,
                    "raw_90_coordinate_coverage": 0.5,
                    "reported_90_coordinate_coverage": 0.9,
                }
                for name in ("early", "middle", "late")
            ]
            (case_dir / "result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "ClothSim2RealPredictionResult",
                        "method_id": runner.METHOD_ID,
                        "authorized_split": "target",
                        "case_id": case_id,
                        "accepted": True,
                        "calibration_artifact_sha256": calibration_sha256,
                        "prediction_seal_sha256": f"seal-{cloth_index}-{task}",
                        "future_outcomes_read_only_after_prediction_seal": True,
                        "metrics": metrics,
                        "horizons": horizons,
                    }
                ),
                encoding="utf-8",
            )
    output = tmp_path / "target_result.json"

    runner._aggregate_target(
        argparse.Namespace(
            calibration_gate=calibration_gate,
            target_lock=target_lock,
            results_root=results_root,
            output=output,
        )
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["artifact_kind"] == "ClothSim2RealTargetResult"
    assert result["calibration_gate_sha256"] == calibration_sha256
    assert result["dynamic_primary"][
        "object_balanced_symmetric_relative_improvement"
    ] == pytest.approx(0.1)
    assert result["dynamic_primary"]["symmetric_win_count"] == 3
    assert result["quasi_static_secondary"][
        "object_balanced_symmetric_relative_improvement"
    ] == pytest.approx(-0.05)
    assert result["formal_90_split_conformal_claim"] is False
