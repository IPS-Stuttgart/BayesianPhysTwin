from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.matphys_loo_sota_report as report_module
from bayesian_phystwin.matphys_loo_sota_report import build_matphys_loo_sota_report


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_case(root: Path, case: str) -> tuple[Path, Path]:
    case_root = root / case
    case_root.mkdir(parents=True)
    observed = np.zeros((6, 2, 3), dtype=np.float32)
    observed[3:, :, 0] = 0.02
    final_data = {
        "object_points": observed,
        "object_visibilities": np.ones((6, 2), dtype=bool),
        "surface_points": np.empty((0, 3), dtype=np.float32),
    }
    with (case_root / "final_data.pkl").open("wb") as handle:
        pickle.dump(final_data, handle)
    with (case_root / "gt_track_3d.pkl").open("wb") as handle:
        pickle.dump(observed[:, :1], handle)
    (case_root / "split.json").write_text(
        json.dumps({"test": [3, 6]}), encoding="utf-8"
    )
    identity = np.zeros_like(observed)
    selected = np.zeros_like(observed)
    selected[3:, :, 0] = 0.01
    identity_path = case_root / "identity.pkl"
    selected_path = case_root / "selected.pkl"
    for path, trajectory in ((identity_path, identity), (selected_path, selected)):
        with path.open("wb") as handle:
            pickle.dump(trajectory, handle)
    return identity_path, selected_path


def _write_summaries(root: Path, cases: tuple[str, ...]) -> tuple[Path, Path]:
    selected_cases = {}
    future_cases = {}
    for case in cases:
        identity_path, selected_path = _write_case(root / "data", case)
        selected_cases[case] = {
            "selected_family": "alpha_0250",
            "selected_within_family_method": "backbone",
            "train_end_frame_exclusive": 3,
            "frame_count": 6,
            "family_outputs": {
                "alpha_0000": _identity(identity_path),
                "alpha_0250": _identity(selected_path),
            },
            "output": _identity(selected_path),
        }
        future_cases[case] = {
            "selected_family": "alpha_0250",
            "selected_output": _identity(selected_path),
        }
    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "sealed",
                "future_metrics_opened": False,
                "contract": {"reference_family": "alpha_0000"},
                "case_results": selected_cases,
            }
        ),
        encoding="utf-8",
    )
    future = root / "future.json"
    future.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "future_metrics_opened": True,
                "contract": {
                    "selection_summary": _identity(selection),
                    "selection_protocol_id": "sealed",
                },
                "case_results": future_cases,
                "comparison": {
                    "selected_equal_case_mean": {
                        "chamfer_distance_m": 0.01,
                        "track_error_m": 0.01,
                    },
                    "family_equal_case_means": {
                        "alpha_0000": {
                            "chamfer_distance_m": 0.02,
                            "track_error_m": 0.02,
                        },
                        "alpha_0250": {
                            "chamfer_distance_m": 0.01,
                            "track_error_m": 0.01,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return selection, future


def test_report_computes_horizons_clusters_and_strict_sota_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = ("single_lift_cloth", "single_lift_rope")
    monkeypatch.setattr(report_module, "PHYSTWIN_TABLE1_CASES", cases)
    selection, future = _write_summaries(tmp_path, cases)

    result = build_matphys_loo_sota_report(
        tmp_path / "data",
        selection,
        future,
        tmp_path / "report.json",
        bootstrap_samples=20,
        bootstrap_block_length=1,
    )

    assert result["point_estimates"]["selected_equal_case_mean"] == pytest.approx(
        {"chamfer_distance_m": 0.01, "track_error_m": 0.01}
    )
    assert result["sota_point_estimate_gate"]["metric_passes"] == {
        "chamfer_distance_m": False,
        "track_error_m": True,
    }
    assert result["sota_point_estimate_gate"]["passed"] is False
    assert set(result["future_horizons"]) == {"early", "middle", "late"}
    assert result["paired_vs_identity"]["cluster_macro"]["cluster_count"] == 2
    assert result["worst_cases"]["track_error_m"]["improved_case_count"] == 2


def test_report_rejects_future_opened_from_different_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = ("single_lift_cloth", "single_lift_rope")
    monkeypatch.setattr(report_module, "PHYSTWIN_TABLE1_CASES", cases)
    selection, future = _write_summaries(tmp_path, cases)
    payload = json.loads(future.read_text(encoding="utf-8"))
    payload["contract"]["selection_summary"]["sha256"] = "0" * 64
    future.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different selection"):
        build_matphys_loo_sota_report(
            tmp_path / "data",
            selection,
            future,
            tmp_path / "report.json",
            bootstrap_samples=10,
        )
