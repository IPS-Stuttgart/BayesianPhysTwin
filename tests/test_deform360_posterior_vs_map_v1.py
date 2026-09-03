"""Focused contracts for the Deform360 posterior-predictive versus MAP study."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/science/run_deform360_posterior_vs_map_v1.py"
PROTOCOL_PATH = ROOT / "protocols/deform360_posterior_vs_map_v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "deform360_posterior_vs_map_v1_contract",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_contracts() -> None:
    load_module().self_test()


def test_identical_components_reduce_to_gaussian() -> None:
    module = load_module()
    truth = np.asarray([-0.2, 0.0, 0.3], dtype=np.float64)
    mean = np.asarray([0.1, -0.1, 0.2], dtype=np.float64)
    components = np.stack([mean, mean, mean])
    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    variance = 0.04
    assert np.allclose(
        module.mixture_nll(truth, components, weights, variance),
        module.gaussian_nll(truth, mean, variance),
        atol=1e-10,
    )
    assert np.allclose(
        module.mixture_crps(truth, components, weights, variance),
        module.gaussian_crps(truth, mean, variance),
        atol=2e-7,
    )


def test_protocol_freezes_nonactive_information_boundary() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen-before-retrospective-execution"
    assert protocol["evaluation"]["inferential_unit"] == "physical object"
    assert protocol["evaluation"]["primary_comparator"] == "map_gaussian"
    assert protocol["evaluation"]["arms"] == [
        "posterior_mixture",
        "posterior_mean_gaussian",
        "map_gaussian",
    ]
    boundary = protocol["information_boundary"]
    assert boundary["retrospective_target_reuse"] is True
    assert boundary["new_measurements_collected"] is False
    assert boundary["robot_actions_selected"] is False
    assert boundary["target_outcomes_may_tune_protocol"] is False
    assert boundary["target_outcomes_may_select_arms"] is False
    assert protocol["paper_claim_authorized"] is False


def test_mixture_event_probability_is_valid() -> None:
    module = load_module()
    components = np.asarray(
        [[-2.0, -1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    probability = module.mixture_event(
        components, weights, 0.01, 0.5, "upper"
    )
    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)
