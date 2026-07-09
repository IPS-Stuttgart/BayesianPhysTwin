import csv
import json
from pathlib import Path

from bayesian_phystwin import replay_residual_csv


def _write_fixture(path: Path) -> None:
    path.write_text(
        "frame,track_id,observed_x,observed_y,predicted_x,predicted_y,variance,"
        "confidence,occluded,boundary_distance,flow_inconsistency,is_corrupted\n"
        "0,1,0.01,0.00,0.00,0.00,0.0001,0.98,false,0.08,0.001,false\n"
        "0,2,0.00,-0.01,0.00,0.00,0.0001,0.95,false,0.06,0.002,false\n"
        "1,1,0.40,-0.25,0.01,0.00,0.0001,0.10,true,0.002,0.30,true\n",
        encoding="utf-8",
    )


def test_replay_reports_robustness_calibration_and_frame_metrics(tmp_path: Path) -> None:
    input_csv = tmp_path / "residuals.csv"
    _write_fixture(input_csv)

    result = replay_residual_csv(input_csv, calibration_bins=3)

    assert result.summary["measurement_count"] == 3
    assert result.summary["measurement_dimension"] == 2
    assert result.summary["labels"] == {"inlier_count": 2, "outlier_count": 1}
    assert result.summary["calibration"]["prior_reliability"]["roc_auc"] == 1.0
    assert len(result.summary["per_frame"]) == 2
    assert len(result.scored_rows) == 3
    assert float(result.scored_rows[-1]["posterior_inlier_probability"]) < 1e-6


def test_replay_writes_machine_readable_artifacts(tmp_path: Path) -> None:
    input_csv = tmp_path / "residuals.csv"
    summary_json = tmp_path / "summary.json"
    scored_csv = tmp_path / "scored.csv"
    _write_fixture(input_csv)

    result = replay_residual_csv(input_csv)
    result.write_summary_json(summary_json)
    result.write_scored_csv(scored_csv)

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    with scored_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["schema_version"] == 1
    assert len(rows) == 3
    assert "prior_reliability" in rows[0]
    assert "robust_negative_log_likelihood" in rows[0]


def test_replay_accepts_compact_obs_pred_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "compact.csv"
    input_csv.write_text(
        "obs_x,obs_y,pred_x,pred_y\n1.0,2.0,1.1,2.1\n",
        encoding="utf-8",
    )

    result = replay_residual_csv(input_csv, default_variance=0.01)

    assert result.summary["measurement_dimension"] == 2
    assert result.summary["columns"]["variance_mode"] == "default"
