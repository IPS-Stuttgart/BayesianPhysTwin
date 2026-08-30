from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping

import numpy as np
import pytest

from experiments.tracking_cloth_deformation_v1.active_probe_evaluation import (
    InputTemplate,
    SourceOutcome,
    fit_leave_one_material_out,
    replay_held_specimen,
)
from experiments.tracking_cloth_deformation_v1.data import Inputs
from experiments.tracking_cloth_deformation_v1.model import Predictions


def protocol() -> dict:
    return {
        "materials": ["cotton", "denim", "polyester", "wool"],
        "sizes": ["A2", "A3"],
        "probe_conditions": [
            "fast_hands",
            "fast_hanger",
            "slow_hands",
            "slow_hanger",
        ],
        "fixed_probe_order": [
            "fast_hands",
            "fast_hanger",
            "slow_hands",
            "slow_hanger",
        ],
        "probe_policies": [
            "fixed_order",
            "parameter_information",
            "task_directed",
        ],
        "probe_budgets": [0, 1, 2, 4],
        "primary_budget": 1,
        "measurement_floor_m": 0.001,
        "selection_templates": (
            "other-material shake and twist-input model disagreement only"
        ),
        "held_material_candidate_inputs_used_for_selection": False,
        "held_material_twist_inputs_used_for_selection": False,
        "paper_claim_authorized": False,
    }


def prediction(seed: int, mode: int = 0) -> Predictions:
    rng = np.random.default_rng(seed)
    times = np.arange(7, dtype=float)
    prefix = np.zeros((2, 4, 3), dtype=float)
    boundary = np.zeros((7, 2, 3), dtype=float)
    inputs = Inputs(
        times=times,
        prefix=prefix,
        boundary=boundary,
        order=np.arange(4),
        corners=np.array([0, 3]),
        cutoff=1,
        initial_time=0.0,
        scale=1.0,
    )
    nominal = np.zeros((7, 4, 3), dtype=float)
    bank = np.zeros((3, 7, 4, 3), dtype=float)
    time = np.maximum(times - 1.0, 0.0)[:, None]
    bank[1, :, 1:3, 0] = (0.003 + 0.001 * mode) * time
    bank[2, :, 1:3, 1] = (0.004 + 0.0005 * mode) * time
    bank[:, :, 1:3] += rng.normal(0.0, 1e-5, size=bank[:, :, 1:3].shape)
    bank[:, :2] = 0.0
    bank[:, :, [0, 3]] = 0.0
    return Predictions(inputs, nominal, bank)


def rosters() -> tuple[list[SourceOutcome], list[InputTemplate]]:
    p = protocol()
    held = "cotton"
    source: list[SourceOutcome] = []
    target: list[InputTemplate] = []
    index = 0
    for material in p["materials"]:
        if material == held:
            continue
        for size in p["sizes"]:
            for condition_index, condition in enumerate(p["probe_conditions"]):
                pred = prediction(index, condition_index)
                winner = (index + condition_index) % pred.bank.shape[0]
                truth = pred.bank[winner].copy()
                recording = f"{material}_{size}_shake_{condition}.csv"
                source.append(
                    SourceOutcome(
                        recording,
                        material,
                        size,
                        condition,
                        pred,
                        truth,
                    )
                )
                target.append(
                    InputTemplate(
                        f"{material}_{size}_twist_{condition}.csv",
                        material,
                        size,
                        condition,
                        prediction(1000 + index, 3 - condition_index),
                    )
                )
                index += 1
    return source, target


class AccessMap(Mapping[str, np.ndarray]):
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> np.ndarray:
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def fitted_fold() -> dict:
    source, target = rosters()
    return fit_leave_one_material_out(
        held_material="cotton",
        source_outcomes=source,
        target_templates=target,
        protocol=protocol(),
    )


def test_fold_is_complete_target_outcome_free_and_json_serializable() -> None:
    fold = fitted_fold()
    assert fold["source_record_count"] == 24
    assert fold["target_template_count"] == 24
    assert fold["training_materials"] == ["denim", "polyester", "wool"]
    assert fold["target_outcomes_used"] is False
    assert fold["held_material_twist_inputs_used_for_selection"] is False
    assert fold["residual_calibration"] == "leave-one-source-record-out"
    assert len(fold["source_crossfit_temperatures_m2"]) == 24
    assert len(fold["source_crossfit_weight_vectors"]) == 24
    assert len(fold["prior_weights"]) == 3
    assert sum(fold["prior_weights"]) == pytest.approx(1.0)
    assert set(fold["probe_distance_m2"]) == set(protocol()["probe_conditions"])
    assert np.asarray(fold["target_distance_m2"]).shape == (3, 3)
    json.dumps(fold, allow_nan=False)


def test_fold_identity_is_record_order_invariant() -> None:
    source, target = rosters()
    first = fit_leave_one_material_out(
        held_material="cotton",
        source_outcomes=source,
        target_templates=target,
        protocol=protocol(),
    )
    second = fit_leave_one_material_out(
        held_material="cotton",
        source_outcomes=list(reversed(source)),
        target_templates=list(reversed(target)),
        protocol=protocol(),
    )
    assert first["fold_id"] == second["fold_id"]
    assert first == second


def test_fold_rejects_any_held_material_or_incomplete_roster() -> None:
    source, target = rosters()
    contaminated = list(source)
    record = contaminated[0]
    contaminated[0] = SourceOutcome(
        record.recording,
        "cotton",
        record.size,
        record.condition,
        record.prediction,
        record.truth,
    )
    with pytest.raises(ValueError):
        fit_leave_one_material_out(
            held_material="cotton",
            source_outcomes=contaminated,
            target_templates=target,
            protocol=protocol(),
        )
    with pytest.raises(ValueError):
        fit_leave_one_material_out(
            held_material="cotton",
            source_outcomes=source[:-1],
            target_templates=target,
            protocol=protocol(),
        )


def test_target_template_has_no_outcome_field() -> None:
    with pytest.raises(TypeError):
        InputTemplate(
            recording="x",
            material="denim",
            size="A2",
            condition="fast_hands",
            prediction=prediction(0),
            truth=np.zeros((7, 4, 3)),
        )


def test_replay_audits_selected_access_and_canonicalizes_endpoints() -> None:
    p = protocol()
    values = {
        "fast_hands": np.array([0.0, 1e-4, 3e-4]),
        "fast_hanger": np.array([2e-4, 0.0, 1e-4]),
        "slow_hands": np.array([1e-4, 3e-4, 0.0]),
        "slow_hanger": np.array([0.0, 2e-4, 1e-4]),
    }
    access = AccessMap(values)
    result = replay_held_specimen(
        specimen="cotton_A2",
        fold=fitted_fold(),
        observed_losses=access,
        protocol=p,
    )
    for policy in p["probe_policies"]:
        states = result["policy_states"][policy]
        order = result["policy_outcome_access_order"][policy]
        assert order == states["4"]["selected_actions"]
        assert order[:1] == states["1"]["selected_actions"]
        assert order[:2] == states["2"]["selected_actions"]
        assert states["4"]["canonical_all_probe_endpoint"] is True
    zero = [
        result["policy_states"][policy]["0"]["weights"]
        for policy in p["probe_policies"]
    ]
    full = [
        result["policy_states"][policy]["4"]["weights"]
        for policy in p["probe_policies"]
    ]
    assert zero[0] == zero[1] == zero[2]
    assert full[0] == full[1] == full[2]
    assert result["selection_consumed_only_selected_outcomes"] is True
    # Four outcomes for each of three policies, then four single-probe controls.
    assert len(access.accessed) == 16
    json.dumps(result, allow_nan=False)


def test_replay_rejects_corrupted_fold_and_bad_losses() -> None:
    fold = fitted_fold()
    broken = copy.deepcopy(fold)
    broken["temperature_m2"] *= 2.0
    values = {condition: np.ones(3) for condition in protocol()["probe_conditions"]}
    with pytest.raises(ValueError, match="fold_id"):
        replay_held_specimen(
            specimen="cotton_A2",
            fold=broken,
            observed_losses=values,
            protocol=protocol(),
        )
    bad = dict(values)
    bad["fast_hands"] = np.ones(2)
    with pytest.raises(ValueError, match="wrong shape"):
        replay_held_specimen(
            specimen="cotton_A2",
            fold=fold,
            observed_losses=bad,
            protocol=protocol(),
        )


def test_replay_rejects_wrong_specimen_and_action_roster() -> None:
    values = {condition: np.ones(3) for condition in protocol()["probe_conditions"]}
    with pytest.raises(ValueError, match="specimen"):
        replay_held_specimen(
            specimen="denim_A2",
            fold=fitted_fold(),
            observed_losses=values,
            protocol=protocol(),
        )
    values.pop("slow_hanger")
    with pytest.raises(ValueError, match="exactly four"):
        replay_held_specimen(
            specimen="cotton_A2",
            fold=fitted_fold(),
            observed_losses=values,
            protocol=protocol(),
        )
