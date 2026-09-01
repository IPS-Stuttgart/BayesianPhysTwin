from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.tracking_cloth_selective_twin_v1.run import (
    _fit_context_decisions,
    cross_material_policy_rows,
    query_value,
)


@dataclass(frozen=True)
class DummyInputs:
    order: np.ndarray
    corners: np.ndarray


def test_registered_queries_have_expected_shapes() -> None:
    inputs = DummyInputs(np.arange(12), np.array([0, 2]))
    positions = np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3)

    assert query_value(positions, inputs, "free_marker_shape").shape == (2, 10, 3)
    assert query_value(positions, inputs, "free_marker_centroid").shape == (2, 3)
    assert query_value(positions, inputs, "bottom_edge_centroid").shape == (2, 3)
    assert query_value(positions, inputs, "shape_radius").shape == (2,)


def _rows() -> list[dict[str, object]]:
    rows = []
    for material in ("cotton", "denim", "polyester", "wool"):
        for motion, regret in (("shake", -2.0), ("twist", 3.0)):
            rows.append(
                {
                    "material": material,
                    "motion": motion,
                    "query": "free_marker_centroid",
                    "horizon_seconds": 1.0,
                    "candidate_loss_mm": 8.0 + regret,
                    "fallback_loss_mm": 8.0,
                    "candidate_minus_fallback_mm": regret,
                    "practical_harm": regret > 1.0,
                    "practical_harm_margin_mm": 1.0,
                    "strict_regression": regret > 0,
                }
            )
    return rows


def _protocol() -> dict[str, object]:
    return {
        "materials": ["cotton", "denim", "polyester", "wool"],
        "primary_gate": {
            "minimum_relative_gain": 0.01,
            "maximum_training_practical_harm_fraction": 0.10,
        },
    }


def test_context_gate_excludes_heldout_material_and_separates_actions() -> None:
    rows = _rows()
    decisions = _fit_context_decisions(rows, "wool", "query_horizon_gate", _protocol())
    assert decisions[("shake", "free_marker_centroid", 1.0)] is True
    assert decisions[("twist", "free_marker_centroid", 1.0)] is False


def test_rejection_is_exact_fallback() -> None:
    selected = cross_material_policy_rows(_rows(), "query_horizon_gate", _protocol())
    twist = [row for row in selected if row["motion"] == "twist"]
    assert twist
    assert all(row["accepted"] is False for row in twist)
    assert all(row["selected_loss_mm"] == row["fallback_loss_mm"] for row in twist)
    assert all(row["exact_fallback"] is True for row in twist)
