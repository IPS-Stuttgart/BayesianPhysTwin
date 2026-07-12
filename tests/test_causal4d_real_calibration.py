import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.real_calibration import _load_case
from causal4d.real_calibration import (
    RealCalibrationCase,
    evaluate_real_calibration_case,
    fit_affine_variance_calibration,
    load_affine_variance_calibration,
    save_affine_variance_calibration,
)


def _case(identifier: str, seed: int) -> RealCalibrationCase:
    rng = np.random.default_rng(seed)
    mean = np.zeros((15, 6, 3), dtype=float)
    truth = rng.normal(0.0, 0.03, size=mean.shape)
    variance = np.full_like(mean, 0.01**2)
    return RealCalibrationCase(
        case_id=identifier,
        action_id=f"action-{identifier}",
        contact_region_id="left",
        mean_m=mean,
        variance_m2=variance,
        truth_m=truth,
        valid=np.ones(mean.shape[:2], dtype=bool),
        start_frame=3,
        node_group_labels=("contact", "contact", "middle", "middle", "far", "far"),
    )


def test_affine_calibration_uses_disjoint_equal_weighted_trials() -> None:
    fit = (_case("fit-a", 1), _case("fit-b", 2))
    heldout = (_case("cal-a", 3), _case("cal-b", 4))
    calibration, diagnostics = fit_affine_variance_calibration(
        fit,
        heldout,
        minimum_calibration_trials=10,
    )
    assert calibration.scale_a > 1.0 or calibration.floor_b_m2 > 0.0
    assert set(calibration.fit_case_ids).isdisjoint(calibration.calibration_case_ids)
    assert not calibration.claim_ready
    assert "requires at least 10" in diagnostics["blocking_reason"]

    target = _case("target", 5)
    result = evaluate_real_calibration_case(target, calibration)
    assert result["calibrated"]["all"]["coverage"] > result["raw"]["all"]["coverage"]
    assert set(result["calibrated"]) >= {
        "all",
        "horizon:early",
        "horizon:middle",
        "horizon:late",
        "graph_region:contact",
        "graph_region:far",
    }
    assert len(result["calibrated"]["all"]["calibration_curve"]) == 6


def test_affine_calibration_rejects_source_target_overlap() -> None:
    source = _case("same", 1)
    with pytest.raises(ValueError, match="disjoint"):
        fit_affine_variance_calibration((source,), (source,))


def test_affine_calibration_artifact_is_checksummed(tmp_path: Path) -> None:
    calibration, diagnostics = fit_affine_variance_calibration(
        (_case("fit", 1),),
        (_case("cal", 2),),
    )
    path = tmp_path / "calibration.json"
    save_affine_variance_calibration(path, calibration, diagnostics)
    assert load_affine_variance_calibration(path) == calibration

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["transform"]["scale_a"] *= 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_affine_variance_calibration(path)


def test_calibration_case_loads_label_free_graph_moments(tmp_path: Path) -> None:
    observed = np.zeros((6, 2, 3), dtype=np.float32)
    final_data = {
        "object_points": observed,
        "object_visibilities": np.ones((6, 2), dtype=bool),
        "object_motions_valid": np.ones((6, 2), dtype=bool),
    }
    final_path = tmp_path / "final.pkl"
    with final_path.open("wb") as handle:
        pickle.dump(final_data, handle)
    moments_path = tmp_path / "moments.npz"
    descriptor = {
        "schema_version": 1,
        "case_id": "source",
        "action_id": "lift",
        "endpoint_frame": 1,
        "start_frame": 2,
        "methods": ["graph_persistence"],
        "future_labels_stored": False,
    }
    np.savez(
        moments_path,
        descriptor_json=np.asarray(json.dumps(descriptor)),
        graph_persistence_mean_m=np.zeros((5, 2, 3)),
        graph_persistence_variance_m2=np.full((5, 2, 3), 1e-4),
    )
    case = _load_case(
        {
            "case_id": "source",
            "moments_npz": str(moments_path),
            "method": "graph_persistence",
            "final_data": str(final_path),
        }
    )
    assert case.case_id == "source"
    assert case.start_frame == 2
    assert case.mean_m.shape == (5, 2, 3)
