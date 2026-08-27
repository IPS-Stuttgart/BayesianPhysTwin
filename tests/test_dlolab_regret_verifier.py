"""Independent arithmetic verifier is qualified before simulator outcomes."""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.coupled_action_regret import (
    calibrate_simultaneous_regret,
)
from bayesian_phystwin_experiments.dlolab_regret_study import (
    MODES,
    infer_parts,
    make_decisions,
    protocol,
    realized_losses,
    score_decisions,
)


def _verifier():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/remote/verify_dlolab_regret_source.py"
    )
    spec = importlib.util.spec_from_file_location("independent_dlolab_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    rng = np.random.default_rng(91)
    bank = {
        "prefix": rng.normal(0, 0.001, (15, 25, 16, 3)),
        "future": rng.normal(0, 0.01, (15, 9, 40, 16, 3)),
    }
    partitions = []
    for n in (39, 64):
        observation = rng.normal(0, 0.004, (n, 3, 4, 3))
        goals = rng.normal(0, 0.01, (n, 3))
        parts = infer_parts(observation, goals, bank["prefix"], bank["future"])
        partitions.append({"observations": observation, "goals": goals, **parts})
    calibration, prediction = partitions
    calibration["future"] = rng.normal(0, 0.01, (39, 9, 40, 16, 3))
    calibration["losses"] = realized_losses(calibration["future"], calibration["goals"])
    calibrators = {
        name: calibrate_simultaneous_regret(
            calibration["raw_upper"][:, index], calibration["losses"]
        )
        for index, name in enumerate(MODES)
    }
    prediction["decisions"] = make_decisions(prediction, calibrators)
    outcome = {"future": rng.normal(0, 0.01, (64, 9, 40, 16, 3))}
    outcome["losses"] = realized_losses(outcome["future"], prediction["goals"])
    result = score_decisions(
        prediction["decisions"], outcome["losses"], prediction["raw_upper"], calibrators
    )
    return (
        bank,
        calibration,
        prediction,
        outcome,
        {k: dataclasses.asdict(v) for k, v in calibrators.items()},
        result,
    )


def test_dense_gaussian_cartesian_quantile_and_independent_score_match():
    assert _verifier().verify_arithmetic(protocol(), *_fixture()) > 8000


@pytest.mark.parametrize(
    "change", ["weight", "calibration", "decision", "truth", "metric", "gate"]
)
def test_verifier_rejects_tampered_scientific_payload(change):
    bank, calibration, prediction, outcome, calibrators, result = _fixture()
    if change == "weight":
        prediction["weights"][0, 0] += 0.01
    elif change == "calibration":
        calibrators["joint"]["offset"] += 0.01
    elif change == "decision":
        prediction["decisions"][0, 0] = 1
    elif change == "truth":
        outcome["losses"][0, 0] += 0.01
    elif change == "metric":
        result["arms"]["joint_regret_guard"]["harm_probability_upper_95"] += 0.1
    else:
        result["source_gate_passed"] = not result["source_gate_passed"]
    with pytest.raises(AssertionError):
        _verifier().verify_arithmetic(
            protocol(), bank, calibration, prediction, outcome, calibrators, result
        )
