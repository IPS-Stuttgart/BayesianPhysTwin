from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.tracking_cloth_self_collision_selective_twin_v1.data import (
    Case,
    prediction_input,
)
from experiments.tracking_cloth_self_collision_selective_twin_v1.model import (
    PhysicsFit,
    all_predictions,
    parameter_bank,
)
from experiments.tracking_cloth_self_collision_selective_twin_v1.selection import (
    apply_policy,
    confirmation_gate,
    fit_cross_material_policies,
    incremental_summary,
    source_gate,
    summarize_policy_rows,
)


def _grid() -> np.ndarray:
    points = []
    for row in range(5):
        for column in range(4):
            points.append([0.12 * column, 0.0, 0.12 * (4 - row)])
    return np.asarray(points, dtype=float)


def _write_case(path: Path, *, future_numeric: bool, forecast: float = 0.2) -> None:
    cloth = _grid()
    rod = np.asarray([[-0.7, 0.0, 0.24], [0.9, 0.0, 0.24]])
    dt = 1.0 / 120.0
    count = int(round((0.5 + forecast) / dt)) + 1
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Frame", "Time"] + [f"C{i}" for i in range(66)])
        for index in range(count):
            time = index * dt
            positions = np.concatenate([cloth, rod], axis=0)
            if time <= 0.5 or future_numeric:
                cells = [f"{value:.9f}" for value in positions.reshape(-1)]
            else:
                cells = ["SEALED" for _ in positions.reshape(-1)]
            writer.writerow([index, f"{time:.9f}", *cells])


def _small_protocol() -> dict:
    return {
        "prefix_seconds": 0.5,
        "forecast_seconds": 0.2,
        "sample_stride": 1,
        "initial_complete_frame_deadline_seconds": 0.05,
        "short_velocity_window_seconds": 0.1,
        "long_velocity_window_seconds": 0.4,
        "residual_velocity_decay_seconds": 0.2,
        "rod_velocity_window_seconds": 0.1,
        "stiffness_per_mass": [100.0],
        "damping_per_mass": [2.0],
        "self_collision_stiffness_per_mass": [0.0],
        "nominal_parameters": [100.0, 2.0, 0.0],
        "integration_substeps": 2,
        "gravity_m_s2": 0.0,
        "contact_radius_m": 0.008,
        "rod_friction_rate": 2.0,
        "self_collision_distance_m": 0.035,
        "measurement_floor_m": 0.001,
    }


def test_prediction_input_does_not_parse_future_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "cotton_a2_four_corners_normal_rep3.csv"
    _write_case(path, future_numeric=False)
    inputs = prediction_input(
        Case(path, "cotton", "four_corners_normal", 3), _small_protocol()
    )
    assert inputs.cutoff >= 5
    assert len(inputs.times) > inputs.cutoff + 1
    assert inputs.cloth_prefix.shape[1:] == (20, 3)
    assert inputs.rod_prefix.shape[1:] == (2, 3)


def test_contact_and_kinematic_predictions_are_finite(tmp_path: Path) -> None:
    path = tmp_path / "cotton_a2_four_corners_normal_rep1.csv"
    _write_case(path, future_numeric=True)
    protocol = _small_protocol()
    inputs = prediction_input(
        Case(path, "cotton", "four_corners_normal", 1), protocol
    )
    parameters = parameter_bank(protocol)
    fit = PhysicsFit(
        parameters=parameters,
        weights=np.ones(len(parameters)) / len(parameters),
        losses_m2=np.ones(len(parameters)),
        temperature_m2=1.0,
    )
    predictions = all_predictions(inputs, fit, protocol)
    required = {
        "persistence",
        "constant_velocity",
        "last_residual",
        "bayesian_contact_physics",
    }
    assert required <= set(predictions)
    assert all(np.isfinite(value).all() for value in predictions.values())


def _row(material: str, physics: float, residual: float) -> dict:
    losses = {
        "persistence": 10.0,
        "constant_velocity": 9.5,
        "last_residual": residual,
        "bayesian_contact_physics": physics,
    }
    return {
        "case_id": f"{material}:four_corners_normal:rep2",
        "recording": f"{material}.csv",
        "material": material,
        "interaction": "four_corners_normal",
        "repetition": 2,
        "query": "cloth_shape",
        "horizon_seconds": 2.0,
        "initial_diameter_mm": 700.0,
        "practical_harm_margin_mm": 17.5,
        "losses_mm": losses,
        "practical_harm": {name: False for name in losses},
    }


def _selection_protocol() -> dict:
    return {
        "materials": ["cotton", "denim", "polyester", "wool"],
        "interactions": ["four_corners_normal"],
        "queries": ["cloth_shape"],
        "horizons_seconds": [2.0],
        "primary_gate": {
            "minimum_relative_gain": 0.01,
            "maximum_training_practical_harm_fraction": 0.1,
            "minimum_source_incremental_relative_gain": 0.005,
        },
        "source_gate": {
            "minimum_physics_coverage": 0.05,
            "minimum_incremental_relative_gain": 0.005,
            "maximum_physics_practical_harm_fraction": 0.1,
        },
        "confirmation_gate": {
            "minimum_physics_coverage": 0.05,
            "minimum_incremental_relative_gain": 0.005,
            "maximum_physics_practical_harm_fraction": 0.1,
        },
        "bootstrap_repetitions": 1000,
        "bootstrap_seed": 4,
    }


def test_matched_selector_is_identical_except_for_physics_arm() -> None:
    protocol = _selection_protocol()
    rows = [
        _row(material, physics=7.0, residual=8.0)
        for material in protocol["materials"]
    ]
    policy = fit_cross_material_policies(rows, protocol)
    selected = apply_policy(rows, policy)
    summaries = summarize_policy_rows(selected, protocol)
    incremental = incremental_summary(selected, protocol)
    assert summaries["physics_enabled"]["physics_coverage"] == 1.0
    assert incremental["physics_minus_residual_mm"] == -1.0
    assert source_gate(summaries, incremental, protocol)["pass"]
    assert confirmation_gate(summaries, incremental, protocol)["pass"]
