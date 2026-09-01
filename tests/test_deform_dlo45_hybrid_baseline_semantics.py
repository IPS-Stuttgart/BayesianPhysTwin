"""Verify that the registered DEFORM comparator is the complete hybrid model."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "remote" / "run_deform_dlo_source.py"
PROTOCOL = ROOT / "experiments" / "deform_dlo45_frozen_v1" / "protocol.json"
README = ROOT / "experiments" / "deform_dlo45_frozen_v1" / "README.md"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


def test_registered_deform_base_trains_the_released_network_modules() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    optimizer = _function(tree, "_official_optimizer")
    trained_modules: set[str] = set()
    for node in ast.walk(optimizer):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "parameters":
            continue
        value = function.value
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "model"
        ):
            trained_modules.add(value.attr)
    assert {
        "vert_conv1",
        "vert_conv2",
        "delta_vert_conv1",
        "delta_vert_conv2",
        "fc",
    } <= trained_modules


def test_registered_deform_base_keeps_the_learned_residual_active_at_rollout() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    rollout = _function(tree, "_rollout_records")
    evaluation_calls: list[ast.Call] = []
    for node in ast.walk(rollout):
        if not isinstance(node, ast.Call):
            continue
        modes = [
            keyword.value.value
            for keyword in node.keywords
            if keyword.arg == "mode"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ]
        if "evaluation" in modes:
            evaluation_calls.append(node)
    assert evaluation_calls, "recursive rollout must call the complete DEFORM model"


def test_historical_key_is_retained_but_reader_facing_semantics_are_hybrid() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert (
        protocol["target_evaluation"]["primary_comparator"]
        == "matching-update-6400-physical-checkpoint"
    )
    readme = README.read_text(encoding="utf-8")
    assert "source-closed retraining of the released DEFORM hybrid" in readme
    assert "This is not a bare-physics baseline" in readme
    assert "no authors-released pretrained DLO4/DLO5 checkpoint" in readme
    assert "BayesianPhysTwin adds a second" in readme
