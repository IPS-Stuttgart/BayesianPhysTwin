from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tracking_cloth_self_collision_stable_bank_v1 import (
    entrypoint,
    model,
)


def _protocol() -> dict:
    return {
        "stiffness_per_mass": [100.0, 400.0, 1600.0],
        "damping_per_mass": [2.0],
        "self_collision_stiffness_per_mass": [0.0],
        "nominal_parameters": [400.0, 2.0, 0.0],
        "measurement_floor_m": 0.001,
        "stable_physics_bank": {
            "minimum_valid_fraction": 0.5,
            "nominal_must_survive": True,
            "prunable_error_messages": [
                "nonfinite contact rollout",
                "contact rollout escaped the registered domain",
            ],
            "selection_stage": "rep1-source-only",
            "target_side_pruning": False,
        },
    }


def _prediction(parameters: tuple[float, float, float]) -> np.ndarray:
    return np.full((1, 1, 1), parameters[0] / 1000.0, dtype=float)


def test_fit_prunes_only_explicit_parent_instability(monkeypatch) -> None:
    protocol = _protocol()

    def fake_parent_rollout(inputs, parameters, supplied_protocol):
        del inputs, supplied_protocol
        if parameters[0] == 100.0:
            raise ValueError("contact rollout escaped the registered domain")
        return _prediction(parameters)

    monkeypatch.setattr(model.parent_model, "contact_rollout", fake_parent_rollout)
    monkeypatch.setattr(
        model.parent_model,
        "trajectory_mse",
        lambda prediction, truth, inputs: float(prediction[0, 0, 0]),
    )
    fit = model.fit_physics(SimpleNamespace(), np.zeros(1), protocol)

    assert fit.parameters == ((400.0, 2.0, 0.0), (1600.0, 2.0, 0.0))
    assert fit.rejected_parameters == ((100.0, 2.0, 0.0),)
    assert fit.rejection_reasons == (
        "contact rollout escaped the registered domain",
    )
    assert fit.candidate_count == 3
    assert fit.valid_fraction == pytest.approx(2.0 / 3.0)
    assert np.sum(fit.weights) == pytest.approx(1.0)
    assert model.PhysicsFit.from_record(fit.record()).record() == fit.record()


def test_nominal_hypothesis_must_survive(monkeypatch) -> None:
    protocol = _protocol()

    def fake_parent_rollout(inputs, parameters, supplied_protocol):
        del inputs, supplied_protocol
        if parameters[0] == 400.0:
            raise ValueError("nonfinite contact rollout")
        return _prediction(parameters)

    monkeypatch.setattr(model.parent_model, "contact_rollout", fake_parent_rollout)
    monkeypatch.setattr(model.parent_model, "trajectory_mse", lambda *args: 1.0)
    with pytest.raises(ValueError, match="nominal contact hypothesis"):
        model.fit_physics(SimpleNamespace(), np.zeros(1), protocol)


def test_source_bank_fails_closed_below_registered_fraction(monkeypatch) -> None:
    protocol = _protocol()

    def fake_parent_rollout(inputs, parameters, supplied_protocol):
        del inputs, supplied_protocol
        if parameters[0] != 400.0:
            raise ValueError("nonfinite contact rollout")
        return _prediction(parameters)

    monkeypatch.setattr(model.parent_model, "contact_rollout", fake_parent_rollout)
    monkeypatch.setattr(model.parent_model, "trajectory_mse", lambda *args: 1.0)
    with pytest.raises(ValueError, match="below the registered minimum"):
        model.fit_physics(SimpleNamespace(), np.zeros(1), protocol)


def test_non_instability_parent_errors_remain_fatal(monkeypatch) -> None:
    protocol = _protocol()

    def invalid_geometry(*args):
        raise ValueError("degenerate cloth spring")

    monkeypatch.setattr(model.parent_model, "contact_rollout", invalid_geometry)
    with pytest.raises(ValueError, match="degenerate cloth spring"):
        model.fit_physics(SimpleNamespace(), np.zeros(1), protocol)


def test_prediction_uses_source_sealed_subset_without_target_pruning(monkeypatch) -> None:
    protocol = _protocol()
    fit = model.PhysicsFit(
        parameters=((400.0, 2.0, 0.0), (1600.0, 2.0, 0.0)),
        weights=np.asarray([0.75, 0.25]),
        losses_m2=np.asarray([0.4, 1.6]),
        temperature_m2=0.4,
        rejected_parameters=((100.0, 2.0, 0.0),),
        rejection_reasons=("nonfinite contact rollout",),
        candidate_count=3,
    )
    monkeypatch.setattr(
        model.parent_model,
        "kinematic_predictions",
        lambda inputs, supplied_protocol: {"persistence": np.zeros((1, 1, 1))},
    )
    seen: list[tuple[float, float, float]] = []

    def fake_parent_rollout(inputs, parameters, supplied_protocol):
        del inputs, supplied_protocol
        seen.append(parameters)
        return _prediction(parameters)

    monkeypatch.setattr(model.parent_model, "contact_rollout", fake_parent_rollout)
    predictions = model.all_predictions(SimpleNamespace(), fit, protocol)

    assert seen == list(fit.parameters)
    assert set(predictions) == {
        "persistence",
        "nominal_contact_physics",
        "map_contact_physics",
        "bayesian_contact_physics",
    }
    assert predictions["bayesian_contact_physics"][0, 0, 0] == pytest.approx(0.7)


def test_stable_contract_and_fit_records_fail_closed() -> None:
    protocol = _protocol()
    protocol["stable_physics_bank"]["minimum_valid_fraction"] = 0.0
    with pytest.raises(ValueError, match="minimum_valid_fraction"):
        model._stable_bank_contract(protocol)

    fit = model.PhysicsFit(
        parameters=((400.0, 2.0, 0.0),),
        weights=np.asarray([1.0]),
        losses_m2=np.asarray([1.0]),
        temperature_m2=1.0,
        rejected_parameters=((100.0, 2.0, 0.0),),
        rejection_reasons=("unregistered error",),
        candidate_count=2,
    )
    with pytest.raises(ValueError, match="unregistered rejection reason"):
        fit.record()


def test_entrypoint_patches_only_parent_model_interface(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(entrypoint.parent_run, "main", lambda: sentinel)
    assert entrypoint.main() is sentinel
    assert entrypoint.parent_run.PhysicsFit is model.PhysicsFit
    assert entrypoint.parent_run.fit_physics is model.fit_physics
    assert entrypoint.parent_run.all_predictions is model.all_predictions
