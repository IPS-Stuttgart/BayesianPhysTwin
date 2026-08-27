"""Source-independent contracts for the pinned branched-rod restart adapter."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deft_native_restart import (
    CHECKPOINT_SHA256,
    JUNCTIONS,
    PARENT_CLAMPS,
    SOURCE_SHA256,
    STATE_FIELDS,
    DeftState,
    _capture_state,
    clamp_only_inputs,
    constructor_ast,
    restart_method_ast,
    update_deft_state,
    valid_geometry_mask,
    verify_upstream,
)

ROOT = Path(__file__).resolve().parents[1]


def _geometry():
    result = np.zeros((2, 3, 13, 3))
    result[:, 0, :, 0] = np.arange(13) * 0.03
    for branch, count in ((1, 5), (2, 4)):
        result[:, branch, :count] = result[:, 0, JUNCTIONS[branch - 1], None]
        result[:, branch, :count, 1] = np.arange(count) * 0.02
    return result


def _state():
    fields = {name: np.ones((3, 13, 3)) for name in STATE_FIELDS}
    fields["b_DLOs_vertices"] = _geometry()[1]
    return DeftState(5, "a" * 64, fields)


def test_only_clamps_enter_prediction_tensor():
    actions = np.arange(12 * 4 * 3, dtype=float).reshape(12, 4, 3) / 1000
    current, previous, packed = clamp_only_inputs(_geometry(), actions)
    np.testing.assert_array_equal(current[0, 0], _geometry()[1])
    np.testing.assert_array_equal(previous[0, 0], _geometry()[0])
    np.testing.assert_array_equal(packed[0, :, 0][:, PARENT_CLAMPS], actions)
    mask = np.zeros((3, 13), dtype=bool)
    mask[0, PARENT_CLAMPS] = True
    np.testing.assert_array_equal(packed[0, :, ~mask], 0)
    assert packed.flags.c_contiguous


@pytest.mark.parametrize("shape", [(12, 3, 13, 3), (12, 5, 3), (0, 4, 3), (4, 3)])
def test_future_free_node_arrays_cannot_enter_api(shape):
    with pytest.raises(ValueError, match="four clamp"):
        clamp_only_inputs(_geometry(), np.zeros(shape))


@pytest.mark.parametrize(
    "change", ["initial_nan", "action_nan", "padding", "junction", "shape"]
)
def test_invalid_initial_or_action_contract(change):
    initial, actions = _geometry(), np.zeros((12, 4, 3))
    if change == "initial_nan":
        initial[0, 0, 0, 0] = np.nan
    elif change == "action_nan":
        actions[0, 0, 0] = np.nan
    elif change == "padding":
        initial[0, 1, -1, 0] = 0.1
    elif change == "junction":
        initial[0, 1, 0, 0] += 0.1
    else:
        initial = initial[:1]
    with pytest.raises(ValueError):
        clamp_only_inputs(initial, actions)


def test_zero_update_clones_every_native_memory_field_exactly():
    state = _state()
    updated = update_deft_state(state, np.zeros((3, 13, 3)), np.zeros((3, 13, 3)))
    assert updated is not state
    assert updated.digests() == state.digests()
    for name in STATE_FIELDS:
        assert not np.shares_memory(updated.fields[name], state.fields[name])


def test_nonzero_update_retains_frame_twist_and_junction_history():
    state = _state()
    dx = np.zeros((3, 13, 3))
    dx[0, 4:9] = 0.001
    dx[1, 0] = dx[0, 4]
    dx[2, 0] = dx[0, 8]
    before = state.digests()
    updated = update_deft_state(state, dx, dx * 0.5)
    assert state.digests() == before
    np.testing.assert_array_equal(
        updated.fields["b_DLOs_vertices"], state.fields["b_DLOs_vertices"] + dx
    )
    for name in set(STATE_FIELDS) - {"b_DLOs_vertices", "b_DLOs_velocity"}:
        np.testing.assert_array_equal(updated.fields[name], state.fields[name])


@pytest.mark.parametrize("kind", ["clamp", "padding", "junction", "nan", "shape"])
def test_invalid_update_rejected(kind):
    delta = np.zeros((3, 13, 3))
    if kind == "clamp":
        delta[0, 0] = 1
    elif kind == "padding":
        delta[1, -1] = 1
    elif kind == "junction":
        delta[0, 4] = 1
    elif kind == "nan":
        delta[0, 4] = np.nan
    else:
        delta = delta[:, :12]
    with pytest.raises(ValueError):
        update_deft_state(_state(), delta, np.zeros((3, 13, 3)))


def test_state_requires_complete_memory_and_model_binding():
    state = _state()
    for changes in (
        {"prediction_index": -1},
        {"model_id": "x" * 64},
        {"fields": {}},
        {"prediction_index": True},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(state, **changes)
    corrupted = dict(state.fields)
    corrupted["theta_full"] = np.array([np.nan])
    with pytest.raises(ValueError):
        dataclasses.replace(state, fields=corrupted)


def _native_fixture():
    fields = "\n".join(
        f"        {name} = torch.zeros(3, 13, 3)" for name in STATE_FIELDS
    )
    return f"""class DEFT_sim:
    def iterative_predict(self, time_horizon, initial, vis=False):
{fields}
        predicted_vertices_list = []
        for ith in range(time_horizon):
            if ith == 0:
                b_DLOs_vertices = initial.clone()
            else:
                theta_full = theta_full + 0.003
            b_DLOs_velocity = b_DLOs_velocity * 0.9 + 0.001 + theta_full * 0.1
            b_DLOs_vertices = b_DLOs_vertices + b_DLOs_velocity
            predicted_vertices_list.append(b_DLOs_vertices.clone())
            b_DLOs_vertices_old = b_DLOs_vertices.clone()
        return torch.stack(predicted_vertices_list)
"""


def test_ast_adapter_preserves_every_native_timestep_statement():
    source = _native_fixture()
    native = ast.parse(source).body[0].body[0]
    adapted = restart_method_ast(source).body[0]
    original_loop = next(node for node in native.body if isinstance(node, ast.For))
    adapted_loop = next(node for node in adapted.body if isinstance(node, ast.For))
    assert [ast.dump(node) for node in adapted_loop.body[:-1]] == [
        ast.dump(node) for node in original_loop.body
    ]
    assert isinstance(adapted_loop.body[-1].value, ast.Yield)
    assert ast.dump(native.body[-1]) == ast.dump(adapted.body[-1])


def test_resumed_synthetic_loop_is_bitwise_identical_and_copies_state():
    torch = pytest.importorskip("torch")
    source = _native_fixture()
    namespace = {"torch": torch, "_bpt_capture_state": _capture_state}
    exec(compile(source, "<native-fixture>", "exec"), namespace)
    exec(compile(restart_method_ast(source), "<resumable-fixture>", "exec"), namespace)
    model = namespace["DEFT_sim"]()
    model._bpt_model_id = "a" * 64
    initial = torch.tensor(_geometry()[1])
    native = model.iterative_predict(12, initial)
    adapted = namespace["_bpt_iterative_predict_resumable"]
    full = list(adapted(model, 12, initial))
    prefix = list(adapted(model, 6, initial))
    state = prefix[-1].clone()
    before = state.digests()
    suffix = list(adapted(model, 12, initial, _resume_state=state.clone()))
    assert torch.equal(native, torch.stack([s.fields["b_DLOs_vertices"] for s in full]))
    assert torch.equal(
        native, torch.stack([s.fields["b_DLOs_vertices"] for s in prefix + suffix])
    )
    assert state.digests() == before
    assert full[-1].digests() == suffix[-1].digests()


@pytest.mark.parametrize(
    "before,after",
    [
        ("range(time_horizon)", "range(1, time_horizon)"),
        (
            "b_DLOs_vertices_old = b_DLOs_vertices.clone()",
            "b_DLOs_vertices_old = b_DLOs_vertices",
        ),
        ("class DEFT_sim:", "class Other:"),
    ],
)
def test_changed_native_control_flow_is_rejected(before, after):
    with pytest.raises(ValueError):
        restart_method_ast(_native_fixture().replace(before, after))


def test_constructor_extracts_only_declared_object_and_no_data_loader():
    source = """def train():
    if BDLO_type == 1:
        geometry = [1, 2, 3]
    if BDLO_type == 2:
        forbidden()
    if inference_1_batch:
        eval_batch = 1
    else:
        eval_batch = 3
    DEFT_sim_train = construct(geometry)
    forbidden_dataset_loader()
"""
    namespace = {"inference_1_batch": False, "construct": tuple}
    exec(compile(constructor_ast(source), "<constructor-fixture>", "exec"), namespace)
    assert namespace["DEFT_sim_train"] == (1, 2, 3)
    assert "forbidden" not in ast.unparse(constructor_ast(source))


def test_source_hash_validator_never_needs_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_upstream(tmp_path)
    path = tmp_path / "deft/core/DEFT_sim.py"
    path.parent.mkdir(parents=True)
    path.write_text("wrong source")
    with pytest.raises(ValueError, match="pinned DEFT source changed"):
        verify_upstream(tmp_path)


def test_protocol_is_native_synthetic_only_and_matches_code():
    protocol = json.loads(
        (ROOT / "configs/sota/deft_native_source_v1.json").read_text()
    )
    assert protocol["upstream"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert all(value is False for value in protocol["boundaries"].values())
    assert protocol["runtime"]["device"] == "cpu"
    assert protocol["runtime"]["dt_s"] == 0.01
    assert protocol["synthetic_qualification"]["timesteps"] == 12
    assert len(SOURCE_SHA256) == 14
    assert valid_geometry_mask().sum() == 22
