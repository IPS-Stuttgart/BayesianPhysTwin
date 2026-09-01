from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.tracking_cloth_selective_router_v2 import run

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "experiments/tracking_cloth_selective_router_v2/protocol.json"
)
WORKFLOW = (
    ROOT / ".github/workflows/tracking-cloth-selective-router-v2.yml"
)


def _row(
    row_id: int,
    material: str,
    context: str,
    candidate_regret: float,
    residual_regret: float,
) -> dict[str, object]:
    motion, query, horizon = context.split("|")
    fallback = 20.0
    return {
        "row_id": row_id,
        "recording": f"{material}-{row_id}.csv",
        "specimen": f"{material}-A",
        "material": material,
        "size": "small" if row_id % 2 else "large",
        "motion": motion,
        "speed": "fast" if row_id % 2 else "slow",
        "grasp": "hands" if row_id % 3 else "corners",
        "query": query,
        "horizon_seconds": float(horizon),
        "candidate_loss_mm": fallback + candidate_regret,
        "fallback_loss_mm": fallback,
        "map_loss_mm": fallback + candidate_regret + 1.0,
        "last_residual_loss_mm": fallback + residual_regret,
        "nominal_loss_mm": fallback + 10.0,
        "candidate_minus_fallback_mm": candidate_regret,
        "candidate_fallback_disagreement_mm": 5.0,
        "ensemble_spread_mm": 1.0,
        "initial_diameter_mm": 400.0,
        "practical_harm_margin_mm": 10.0,
        "strict_regression": candidate_regret > 0.0,
        "practical_harm": candidate_regret > 10.0,
        "motion_query_horizon": context,
        "candidate_regret_mm": candidate_regret,
        "last_residual_regret_mm": residual_regret,
    }


def _rows() -> list[dict[str, object]]:
    rows = []
    row_id = 0
    for material in ("cotton", "denim", "polyester", "wool"):
        for _ in range(4):
            rows.append(
                _row(
                    row_id,
                    material,
                    "shake|free_marker_shape|5",
                    -4.0,
                    -2.0,
                )
            )
            row_id += 1
        for _ in range(4):
            rows.append(
                _row(
                    row_id,
                    material,
                    "twist|free_marker_shape|1",
                    8.0,
                    6.0,
                )
            )
            row_id += 1
    return rows


def _protocol() -> dict[str, object]:
    return {
        "materials": ["cotton", "denim", "polyester", "wool"],
        "ridge_alphas": [0.1, 1.0],
        "admission_thresholds_mm": [-2.5, 0.0],
        "bootstrap_repetitions": 100,
        "bootstrap_seed": 13,
        "inner_selection": {
            "minimum_selected_coverage": 0.2,
            "maximum_practical_harm_fraction": 0.1,
        },
    }


def test_numpy_ridge_predicts_context_regret_signs() -> None:
    rows = _rows()
    state = run.fit_ridge(rows, alpha=0.1)
    prediction = run.predict_ridge(state, rows)

    beneficial = np.asarray(
        [row["motion_query_horizon"].endswith("|5") for row in rows]
    )
    assert np.all(prediction[beneficial] < 0.0)
    assert np.all(prediction[~beneficial] > 0.0)


def test_nested_router_uses_exact_outer_material_exclusion() -> None:
    rows = _rows()
    routed, choices = run.nested_policy(
        rows,
        ("bayesian_physics", "last_residual"),
        _protocol(),
        policy="nested_triage",
    )

    assert len(choices) == 4
    assert all(choice.inner_feasible for choice in choices)
    assert all(
        row["outer_heldout_material"] == row["material"] for row in routed
    )
    assert all(
        row["selected_arm"] != "persistence"
        for row in routed
        if row["motion"] == "shake"
    )
    assert all(
        row["selected_arm"] == "persistence"
        for row in routed
        if row["motion"] == "twist"
    )
    assert all(row["exact_fallback"] for row in routed)


def test_drop_one_expert_reuses_primary_fold_choices() -> None:
    rows = _rows()
    primary, choices = run.nested_policy(
        rows,
        ("bayesian_physics", "last_residual"),
        _protocol(),
        policy="nested_triage",
    )
    dropped = run.apply_fold_choices(
        rows,
        choices,
        ("bayesian_physics",),
        _protocol(),
        policy="nested_triage_drop_last_residual",
    )

    primary_choice = {
        choice.heldout_material: (
            choice.alpha,
            choice.threshold_mm,
        )
        for choice in choices
    }
    assert len(primary) == len(dropped)
    for row in dropped:
        assert (
            row["ridge_alpha"],
            row["admission_threshold_mm"],
        ) == primary_choice[row["material"]]


def test_protocol_preserves_retrospective_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["schema"] == (
        "bayesian-phystwin.tracking-cloth-selective-router.v2"
    )
    assert protocol["outer_split"] == "leave-one-material-out"
    assert protocol["inner_split"].startswith("leave-one-material-out")
    assert protocol["fallback_arm"] == "persistence"
    assert protocol["candidate_arms"] == [
        "bayesian_physics",
        "last_residual",
    ]
    assert protocol["information_boundary"]["fresh_confirmation"] is False
    assert (
        protocol["information_boundary"]["retrospective_model_development"]
        is True
    )
    assert protocol["information_boundary"]["paper_claim_authorized"] is False


def test_workflow_is_hash_bound_hosted_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in text
    assert "tracking-cloth-selective-router-v2.json" in text
    assert "zenodo.org/records/14644526/files/tracking_dataset.zip" in text
    assert "b4868b702f8a42b2ea1069d0f1a3b8f6" in text
    assert (
        "14916efa89a26d991c024024cc9449397"
        "d3a6f654311e621bb91e9602e231e1a"
    ) in text
    assert "gpuserver4090" not in text
    assert "workflow_dispatch" not in text
