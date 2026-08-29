"""Synthetic-only data, leakage, numerical and evaluation contracts."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from experiments.tracking_cloth_deformation_v1.data import (
    Case, audit_dataset, input_view, layout, scoring_view,
)
from experiments.tracking_cloth_deformation_v1.model import (
    ARMS, Predictions, complete_beliefs, masks, parameter_bank, predict, score,
    source_weights,
)
from experiments.tracking_cloth_deformation_v1.run import aggregate, prepare, score_run

BASE = Path(__file__).resolve().parents[1] / "experiments/tracking_cloth_deformation_v1"


def config():
    protocol = json.loads((BASE / "protocol.json").read_text())
    protocol.update({"prefix_seconds": 0.2, "forecast_seconds": 0.2,
                     "stiffness_per_mass": [400.0], "damping_per_mass": [2.0],
                     "integration_substeps": 2, "bootstrap_repetitions": 100})
    return protocol


def initial_grid(size="A3"):
    rows, cols = (5, 4) if size == "A2" else (4, 3)
    spacing = 0.12
    return np.array([[spacing * c, 0.0, 1.0 - spacing * r]
                     for r in range(rows) for c in range(cols)])


def csv_text(size="A3", poison_future=False, missing=False):
    first = initial_grid(size)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Format Version", "1.0", "Length Units", "Meters"])
    writer.writerow(["", "", "Position"])
    writer.writerow(["", "", "Marker IDs"])
    writer.writerow(["", "", "Position"])
    writer.writerow(["Frame", "Time", *(["X", "Y", "Z"] * len(first))])
    for i in range(61):
        t = i / 120
        positions = first.copy()
        positions[:, 1] += 0.002 * np.sin(2 * np.pi * t)
        values = positions.reshape(-1).astype(object)
        if missing and i == 40:
            values[3 * 5:3 * 5 + 3] = ""
        if poison_future and t > 0.2 + 1e-8:
            # Only non-driven markers are poisoned, proving no numeric conversion.
            for marker in range(len(first)):
                if marker not in (0, 2 if size == "A3" else 3):
                    values[3 * marker:3 * marker + 3] = "UNOPENED_TARGET"
        writer.writerow([i, f"{t:.9f}", *values])
    return stream.getvalue()


def case_file(tmp_path, text=None, motion="twist"):
    path = tmp_path / f"cotton_A3_{motion}_fast_hands.csv"
    path.write_text(csv_text() if text is None else text)
    return Case(path, "cotton", "A3", motion, "fast", "hands")


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    payloads = {}
    for material, size, motion, speed, grasp in itertools.product(
            ("cotton", "denim", "polyester", "wool"), ("A2", "A3"),
            ("shake", "twist"), ("fast", "slow"), ("hands", "hanger")):
        name = f"Free-hanging/{material}_{size}_{motion}_{speed}_{grasp}.csv"
        payloads[name] = csv_text(size)
    for i in range(56):
        payloads[f"Reserved/unused_{i}.csv"] = "reserved; never numerically parsed\n"
    for folder in ("Free-hanging", "Tablecloth", "Hitting", "Self-collision"):
        payloads[f"{folder}/read_data.m"] = "% Synthetic fixture, no dataset content.\n"
    payloads["License.txt"] = "Synthetic fixture: CC BY-NC-SA 4.0\n"
    for name, content in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    archive = root / "dataset.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, content in payloads.items():
            zipped.writestr("dataset/" + name, content)
    protocol = config()
    protocol["archive_md5"] = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    return root, protocol


def test_complete_roster_inventory_without_numeric_target_reads(dataset):
    root, protocol = dataset
    cases, inventory = audit_dataset(root, protocol)
    assert len(cases) == 64
    assert inventory["source_count"] == inventory["target_count"] == 32
    assert inventory["unused_count"] == 56
    assert inventory["target_numeric_outcomes_read"] is False


def test_tampered_extraction_rejected(dataset):
    root, protocol = dataset
    next(root.rglob("*twist*.csv")).write_text("tampered")
    with pytest.raises(ValueError, match="Extracted bytes"):
        audit_dataset(root, protocol)


def test_duplicate_basename_rejected(dataset):
    root, protocol = dataset
    duplicate = root / "dup"
    duplicate.mkdir()
    original = next(root.rglob("*twist*.csv"))
    (duplicate / original.name).write_bytes(original.read_bytes())
    protocol["csv_count"] += 1
    with pytest.raises(ValueError, match="duplicate"):
        audit_dataset(root, protocol)


def test_wrong_archive_rejected(dataset):
    root, protocol = dataset
    protocol["archive_md5"] = "0" * 32
    with pytest.raises(ValueError, match="published MD5"):
        audit_dataset(root, protocol)


def test_canonical_grid_and_driven_corners():
    permutation = np.random.default_rng(8).permutation(20)
    order, corners = layout(initial_grid("A2")[permutation], "A2")
    np.testing.assert_array_equal(permutation[order], np.arange(20))
    np.testing.assert_array_equal(corners, [0, 3])


def test_future_free_marker_values_never_enter_prediction(tmp_path):
    case = case_file(tmp_path)
    protocol = config()
    clean = input_view(case, protocol, 1.0)
    before = predict(clean, protocol)
    case.path.write_text(csv_text(poison_future=True))
    closed = input_view(case, protocol, 1.0)
    after = predict(closed, protocol)
    np.testing.assert_array_equal(before.nominal, after.nominal)
    np.testing.assert_array_equal(before.bank, after.bank)
    with pytest.raises(ValueError, match="UNOPENED_TARGET"):
        scoring_view(case, closed)


def test_causal_missing_values_not_filled_from_future(tmp_path):
    case = case_file(tmp_path, csv_text(missing=True))
    inputs = input_view(case, config(), 1.0)
    truth = scoring_view(case, inputs)
    assert np.isnan(truth).any()
    assert np.isfinite(inputs.prefix).all()
    valid = masks(inputs, truth)
    assert not np.any(valid[:, inputs.corners])
    assert not np.any(valid[:inputs.cutoff + 1])


def test_bad_timestamps_rejected(tmp_path):
    case = case_file(tmp_path, csv_text().replace("1,0.008333333", "1,0.000000000"))
    with pytest.raises(ValueError, match="order"):
        input_view(case, config(), 1.0)


def test_gibbs_weights_prefer_lower_source_loss():
    weights = source_weights(np.array([[0.001, 0.1], [0.002, 0.2]]), 0.001)
    assert weights[0] > 0.999
    assert weights.sum() == pytest.approx(1.0)


def test_exact_fallback_includes_covariance(tmp_path):
    inputs = input_view(case_file(tmp_path), config(), 1.0)
    prediction = predict(inputs, config())
    fit = {"source_posterior_weights": [1.0], "guard_accepts": False,
           "source_residual_variance_m2": {arm: [1e-5, 2e-5, 3e-5] for arm in ARMS[:-1]}}
    beliefs = complete_beliefs(prediction, fit, config())
    assert beliefs["guarded_bayesian_physics"] is beliefs["nominal_physics"]
    np.testing.assert_array_equal(prediction.nominal[:, inputs.corners], inputs.boundary)
    np.testing.assert_array_equal(prediction.bank[0][:, inputs.corners], inputs.boundary)


def test_coordinate_score_ignores_driven_markers(tmp_path):
    inputs = input_view(case_file(tmp_path), config(), 1.0)
    truth = scoring_view(case_file(tmp_path), inputs)
    mean = truth.copy()
    mean[:, inputs.corners] += 10000
    values = score(mean, np.full_like(mean, 1e-4), truth, inputs)
    assert values["rmse_mm"] == 0
    assert values["coordinate_90_coverage"] == 1


def test_posterior_total_variance_includes_parameter_spread(tmp_path):
    protocol = config()
    protocol["stiffness_per_mass"] = [100.0, 400.0]
    inputs = input_view(case_file(tmp_path), protocol, 1.0)
    nominal = np.zeros((len(inputs.times), 12, 3))
    prediction = Predictions(inputs, nominal, np.stack([nominal, nominal + 2]))
    fit = {"source_posterior_weights": [0.5, 0.5], "guard_accepts": True,
           "source_residual_variance_m2": {arm: [1.0] * 3 for arm in ARMS[:-1]}}
    beliefs = complete_beliefs(prediction, fit, protocol)
    assert np.all(beliefs["bayesian_physics"][0] == 1)
    assert np.all(beliefs["bayesian_physics"][1] == 2)
    assert beliefs["guarded_bayesian_physics"] is beliefs["bayesian_physics"]
    assert parameter_bank(protocol) == [(100.0, 2.0), (400.0, 2.0)]


def test_source_only_run_does_not_touch_target_numeric_payload(dataset, tmp_path):
    root, protocol = dataset
    output = tmp_path / "source-output"
    prepare(root, output, protocol, "source")
    assert (output / "source_fit.json").is_file()
    assert not (output / "private_predictions").exists()
    assert not (output / "target_access.json").exists()
    assert not (output / "metrics.json").exists()
    assert json.loads((output / "source_fit.json").read_text())["target_outcomes_used"] is False


def test_complete_predict_seal_score_cycle(dataset, tmp_path):
    root, protocol = dataset
    output = tmp_path / "output"
    prepare(root, output, protocol, "predict")
    assert (output / "prediction_seal.json").is_file()
    assert not (output / "target_access.json").exists()
    assert not (output / "metrics.json").exists()
    score_run(root, output)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["target_recordings"] == 32
    assert metrics["paper_claim_authorized"] is False
    assert metrics["exact_fallback_violations"] == 0
    assert set(metrics["arms"]) == set(ARMS)
    with pytest.raises(ValueError, match="already started"):
        score_run(root, output)


def test_output_cannot_be_dataset_descendant(dataset):
    root, protocol = dataset
    with pytest.raises(ValueError, match="disjoint"):
        prepare(root, root / "outputs", protocol, "inventory")


def test_modified_source_fit_stops_target_opening(dataset, tmp_path):
    root, protocol = dataset
    output = tmp_path / "altered-output"
    prepare(root, output, protocol, "predict")
    (output / "source_fit.json").write_text("{}")
    with pytest.raises(ValueError, match="Source fit changed"):
        score_run(root, output)
    assert not (output / "target_access.json").exists()


def test_incomplete_result_never_pooled():
    with pytest.raises(ValueError, match="Incomplete roster"):
        aggregate([], config())


def test_checked_in_protocol_matches_requested_release():
    protocol = json.loads((BASE / "protocol.json").read_text())
    assert protocol["dataset_record"] == "14644526"
    assert protocol["archive_md5"] == "b4868b702f8a42b2ea1069d0f1a3b8f6"
    assert protocol["raw_data_upload"] is False
    assert protocol["paper_claim_authorized"] is False
    assert "NC" in protocol["license_policy"]
