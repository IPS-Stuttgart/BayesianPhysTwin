"""Contract tests for the Tracking Cloth matched-coverage ablation."""

from __future__ import annotations

import copy

from experiments.tracking_cloth_matched_coverage_v1.run import (
    build_selections,
    canonical_sha256,
    case_id,
    load_protocol,
)


def _fixture():
    materials = ["a", "b", "c", "d"]
    rows = []
    policies = []
    for material_index, material in enumerate(materials):
        for index in range(10):
            spread = float(index + 1)
            regret = spread - 5.0 + 0.1 * material_index
            row = {
                "recording": f"{material}-{index // 2}",
                "specimen": f"{material}-s",
                "material": material,
                "size": "A2" if index % 2 else "A3",
                "motion": "shake" if index < 5 else "twist",
                "speed": "fast" if index % 2 else "slow",
                "grasp": "hands" if index % 3 else "hanger",
                "query": "q1" if index % 2 else "q2",
                "horizon_seconds": float(1 + index % 2),
                "candidate_loss_mm": 10.0 + regret,
                "fallback_loss_mm": 10.0,
                "map_loss_mm": 11.0 + regret,
                "last_residual_loss_mm": 9.0 + regret,
                "nominal_loss_mm": 12.0 + regret,
                "candidate_minus_fallback_mm": regret,
                "candidate_fallback_disagreement_mm": float(10 - index),
                "ensemble_spread_mm": spread,
                "initial_diameter_mm": 100.0,
                "practical_harm_margin_mm": 2.5,
                "strict_regression": regret > 0.0,
                "practical_harm": regret > 2.5,
            }
            rows.append(row)
            policies.append(
                {**row, "policy": "query_horizon_gate", "accepted": index == 0}
            )
    protocol = load_protocol()
    protocol = {**protocol, "materials": materials, "acceptance_count_per_material": 1}
    return rows, policies, protocol


def test_protocol_freezes_primary_contrast_and_coverage() -> None:
    protocol = load_protocol()
    assert protocol["acceptance_fraction"] == 0.1
    assert protocol["acceptance_count_per_material"] == 32
    assert protocol["primary_comparison"]["candidate_selector"] == (
        "context_plus_spread_ridge"
    )
    assert protocol["primary_comparison"]["comparator_selector"] == "context_ridge"
    assert protocol["information_boundary"]["retrospective"] is True
    assert protocol["information_boundary"][
        "heldout_material_outcomes_used_to_fit_or_rank_nonoracle_selectors"
    ] is False


def test_every_selector_has_exact_matched_coverage() -> None:
    rows, policies, protocol = _fixture()
    selections, _, _ = build_selections(rows, policies, protocol)
    for selected in selections.values():
        assert len(selected) == 4
        for material in protocol["materials"]:
            assert (
                sum(
                    case_id(row) in selected
                    for row in rows
                    if row["material"] == material
                )
                == 1
            )


def test_heldout_outcome_mutation_cannot_change_nonoracle_selection() -> None:
    rows, policies, protocol = _fixture()
    original, _, _ = build_selections(rows, policies, protocol)
    changed_rows = copy.deepcopy(rows)
    for row in changed_rows:
        if row["material"] == "d":
            row["candidate_minus_fallback_mm"] += 100000.0
            row["candidate_loss_mm"] += 100000.0
    changed, _, _ = build_selections(changed_rows, policies, protocol)
    for selector in protocol["selectors"]:
        if selector != "oracle_matched_coverage":
            assert original[selector] == changed[selector]


def test_canonical_hash_is_key_order_invariant() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
