"""Pinned, source-only DEFT restart adapter; no change to the DEFORM backend."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
)

UPSTREAM_REVISION = "5781c70c7737fb84b8bd43261e3ed00ef2fd0fbc"
CHECKPOINT_SHA256 = "8ce5d6c736edfd246506d0458efd95f852e0e6d2dd07fae01c798462209fbc9a"
SOURCE_SHA256 = {
    "deft/core/DEFT_sim.py": "a2524749a3d9136d95ea7270ef858c7f5a1df35d91f16e585d15b1bc63c82b1b",
    "deft/core/DEFT_func.py": "511833c8d8e75c5c7200511356a598cec833a7432fd2d9886d75efa8ce536c28",
    "deft/models/GNN_tree.py": "22c2db91b2a7282ab83d8e279ef1be35dc1d6d44223a0a700115fefb9242ff02",
    "deft/utils/util.py": "5dabd38ea273a2d59510dced0a882753325d839a2933fff76c9b455e8f011843",
    "deft/solvers/constraints_solver.py": "01ed67a1d6f37a4c9502556fbf58334c604dd5546e5b29a68b51ce76a5deadae",
    "deft/solvers/constraints_enforcement_numba.py": "61e04b9dacb511c82b9bf0ca81dd86329875103f1204293629444201c7687180",
    "deft/solvers/theta_solver.py": "b11faaa99d08119857ee8b9c36e18f11dabfc8ffec9c487284fa81dd04eb6d88",
    "deft/solvers/theta_solver_numpy.py": "5809fdf26bc6a6ad4caf6e7b1e5093f8898396eaeed879c3895b79c2f91c9a21",
    "scripts/DEFT_train.py": "686248213d655554985d2d0526b30f6af3ee2e9e1395f76bb3466def1fb97902",
    "deft/__init__.py": "ae348ba0c8b4d04b9c20889522ba85425b2271b018b4c797786500840bcd0fa1",
    "deft/core/__init__.py": "e03b1a7e724f7f4d2c44a589866093aff1f68823caf7ae1b808c8a398949adbf",
    "deft/models/__init__.py": "794b53d4a136f80c1a9e4df55d3340eace205189ca1b32ecf6c64763fa5f5440",
    "deft/solvers/__init__.py": "814ca63ab88897998d3f67f3e9be0e42d8def57e3cc5715090efa582ea3e8d3e",
    "deft/utils/__init__.py": "2e579248114927323af8d9595a11d3c5ad11ea9023f4dbdfa447860bcf5eb4c2",
}
BRANCH_LENGTHS = (13, 5, 4)
PARENT_CLAMPS = (0, 1, 11, 12)
JUNCTIONS = (4, 8)
DT_S = 0.01
STATE_FIELDS = (
    "b_DLOs_vertices",
    "b_DLOs_vertices_old",
    "b_DLOs_velocity",
    "m_u0",
    "theta_full",
    "optimization_mask",
    "parent_rod_orientation",
    "children_rod_orientation",
    "previous_parent_vertices_iteration_edge1",
    "previous_parent_vertices_iteration_edge2",
    "previous_children_vertices_iteration_edge",
)


def verify_upstream(root: Path) -> dict[str, str]:
    """Verify source bytes, without listing or opening any dataset directory."""
    root = root.resolve(strict=True)
    for relative, digest in SOURCE_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("upstream source path is not a regular contained file")
        if file_digest(path) != digest:
            raise ValueError(f"pinned DEFT source changed: {relative}")
    return dict(SOURCE_SHA256)


def valid_geometry_mask() -> np.ndarray:
    mask = np.zeros((3, 13), dtype=bool)
    for branch, length in enumerate(BRANCH_LENGTHS):
        mask[branch, :length] = True
    return mask


def clamp_only_inputs(
    initial_two: np.ndarray, clamps: np.ndarray
) -> tuple[np.ndarray, ...]:
    """The rollout API never accepts a full future object trajectory."""
    initial = np.asarray(initial_two, dtype=np.float64)
    actions = np.asarray(clamps, dtype=np.float64)
    if initial.shape != (2, 3, 13, 3) or not np.isfinite(initial).all():
        raise ValueError("initial geometry must be two finite padded BDLO1 states")
    if actions.ndim != 3 or actions.shape[1:] != (4, 3) or not len(actions):
        raise ValueError("only four clamp trajectories may enter the native rollout")
    if not np.isfinite(actions).all():
        raise ValueError("clamp trajectories must be finite")
    if np.any(initial[:, ~valid_geometry_mask()] != 0):
        raise ValueError("padded geometry must be zero")
    for child, junction in enumerate(JUNCTIONS, start=1):
        if not np.array_equal(initial[:, child, 0], initial[:, 0, junction]):
            raise ValueError("duplicate junction identities disagree")
    packed = np.zeros((1, len(actions), 3, 13, 3), dtype=np.float64)
    packed[0, :, 0, PARENT_CLAMPS] = actions.transpose(1, 0, 2)
    return (
        np.ascontiguousarray(initial[1][None, None]),
        np.ascontiguousarray(initial[0][None, None]),
        packed,
    )


def _clone_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    raise TypeError("native state fields must be tensors or numeric arrays")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


@dataclass(frozen=True)
class DeftState:
    prediction_index: int
    model_id: str
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.prediction_index) is not int or self.prediction_index < 0:
            raise ValueError("restart index must follow a completed native step")
        if len(self.model_id) != 64 or any(
            c not in "0123456789abcdef" for c in self.model_id
        ):
            raise ValueError("restart state must be bound to a model digest")
        if set(self.fields) != set(STATE_FIELDS):
            raise ValueError("restart state is missing native memory")
        for value in self.fields.values():
            array = _numpy(value)
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(
                    "restart state contains a nonfinite or nonnumeric field"
                )
        if _numpy(self.fields["b_DLOs_vertices"]).shape != (3, 13, 3):
            raise ValueError("only the declared batch-one BDLO1 topology is supported")

    def clone(self) -> DeftState:
        return DeftState(
            self.prediction_index,
            self.model_id,
            {name: _clone_value(value) for name, value in self.fields.items()},
        )

    def digests(self) -> dict[str, str]:
        return {
            name: array_digest(_numpy(value)) for name, value in self.fields.items()
        }


def _capture_state(index: int, model_id: str, values: Mapping[str, Any]) -> DeftState:
    return DeftState(
        index, model_id, {name: _clone_value(values[name]) for name in STATE_FIELDS}
    )


def update_deft_state(
    state: DeftState, delta_x: np.ndarray, delta_v: np.ndarray
) -> DeftState:
    """Correct pose/velocity, retaining all native material and junction memory."""
    dx, dv = (np.asarray(value, dtype=np.float64) for value in (delta_x, delta_v))
    for value in (dx, dv):
        if value.shape != (3, 13, 3) or not np.isfinite(value).all():
            raise ValueError("state increments must be finite padded BDLO1 arrays")
        if np.any(value[0, PARENT_CLAMPS] != 0) or np.any(
            value[~valid_geometry_mask()] != 0
        ):
            raise ValueError("state update may not alter clamps or padded identities")
        for child, junction in enumerate(JUNCTIONS, start=1):
            if not np.array_equal(value[child, 0], value[0, junction]):
                raise ValueError("duplicate junction increments must agree")
    result = state.clone()
    if not np.any(dx) and not np.any(dv):
        return result
    fields = dict(result.fields)
    for name, increment in (("b_DLOs_vertices", dx), ("b_DLOs_velocity", dv)):
        before = fields[name]
        addition = (
            increment
            if isinstance(before, np.ndarray)
            else before.new_tensor(increment)
        )
        fields[name] = before + addition
    return DeftState(result.prediction_index, result.model_id, fields)


def _one_function(
    tree: ast.Module, class_name: str, function_name: str
) -> ast.FunctionDef:
    matches = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError("native prediction method is not unique")
    return copy.deepcopy(matches[0])


def restart_method_ast(source: str) -> ast.Module:
    """Add a pause/restore boundary to the pinned loop, preserving every step."""
    function = _one_function(ast.parse(source), "DEFT_sim", "iterative_predict")
    loops = [
        node
        for node in function.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ith"
    ]
    if len(loops) != 1 or ast.unparse(loops[0].iter) != "range(time_horizon)":
        raise ValueError("native timestep loop changed")
    loop = loops[0]
    if ast.unparse(loop.body[-1]) != "b_DLOs_vertices_old = b_DLOs_vertices.clone()":
        raise ValueError("native end-of-step memory boundary changed")
    if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(function)):
        raise ValueError("native prediction method already yields")
    function.name = "_bpt_iterative_predict_resumable"
    function.args.args.append(ast.arg(arg="_resume_state"))
    function.args.defaults.append(ast.Constant(value=None))
    restore = ast.parse(
        "_start_index = 0\n"
        "if _resume_state is not None:\n"
        "    _start_index = _resume_state.prediction_index + 1\n"
        + "".join(
            f"    {name} = _resume_state.fields[{name!r}]\n" for name in STATE_FIELDS
        )
    ).body
    offset = function.body.index(loop)
    function.body[offset:offset] = restore
    loop.iter = ast.parse("range(_start_index, time_horizon)", mode="eval").body
    selected = ast.Dict(
        keys=[ast.Constant(value=name) for name in STATE_FIELDS],
        values=[ast.Name(id=name, ctx=ast.Load()) for name in STATE_FIELDS],
    )
    loop.body.append(
        ast.Expr(
            value=ast.Yield(
                value=ast.Call(
                    func=ast.Name(id="_bpt_capture_state", ctx=ast.Load()),
                    args=[
                        ast.Name(id="ith", ctx=ast.Load()),
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr="_bpt_model_id",
                            ctx=ast.Load(),
                        ),
                        selected,
                    ],
                    keywords=[],
                )
            )
        )
    )
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def constructor_ast(source: str) -> ast.Module:
    """Extract only the released BDLO1 constructor, never its data loaders."""
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    ]
    if len(functions) != 1:
        raise ValueError("native constructor entry point changed")
    body = functions[0].body
    bdlo = [
        node
        for node in body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "BDLO_type == 1"
    ]
    starts = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "inference_1_batch"
    ]
    ends = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "DEFT_sim_train"
    ]
    if len(bdlo) != 1 or len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("native source-only construction boundary changed")
    return ast.fix_missing_locations(
        ast.Module(
            body=copy.deepcopy(bdlo[0].body + body[starts[0] : ends[0] + 1]),
            type_ignores=[],
        )
    )


def _load_train_module(root: Path) -> Any:
    existing = sys.modules.get("deft")
    existing_path = getattr(existing, "__file__", None)
    if existing is not None and (
        not isinstance(existing_path, str)
        or not Path(existing_path).resolve().is_relative_to(root)
    ):
        raise ValueError("another DEFT checkout is already imported")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "_bpt_pinned_deft_train", root / "scripts/DEFT_train.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned DEFT construction code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeDeft:
    """Batch-one native DEFT with explicit, cloned restart state."""

    def __init__(self, upstream_root: Path, checkpoint_path: Path):
        self.root = upstream_root.resolve(strict=True)
        verify_upstream(self.root)
        if file_digest(checkpoint_path) != CHECKPOINT_SHA256:
            raise ValueError("DEFT checkpoint differs from the declared release")
        import torch

        self.torch = torch
        if torch.get_default_dtype() != torch.float64:
            raise ValueError(
                "native DEFT requires explicit float64 runtime initialization"
            )
        module = _load_train_module(self.root)
        # One upstream solver seeds at import; restore the declared construction seed.
        torch.manual_seed(1)
        np.random.seed(1)
        namespace = dict(module.__dict__)
        namespace.update(
            device="cpu",
            n_branch=3,
            train_batch=1,
            BDLO_type=1,
            clamp_type="ends",
            residual_learning=True,
            inference_1_batch=False,
            clamp_parent=True,
            clamp_child1=False,
            clamp_child2=False,
            use_orientation_constraints=True,
            use_attachment_constraints=True,
        )
        source = (self.root / "scripts/DEFT_train.py").read_text()
        exec(
            compile(
                constructor_ast(source),
                str(self.root / "scripts/DEFT_train.py"),
                "exec",
            ),
            namespace,
        )
        self.model = namespace["DEFT_sim_train"]
        self.reference_geometry = namespace["b_undeformed_vert"].detach().numpy().copy()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
        ):
            raise ValueError("checkpoint is not a tensor state dictionary")
        state = {
            name: value
            for name, value in state.items()
            if name != "GNN_tree.adjacency_batch"
        }
        incompatibility = self.model.load_state_dict(state, strict=False)
        if (
            incompatibility.missing_keys != ["GNN_tree.adjacency_batch"]
            or incompatibility.unexpected_keys
        ):
            raise ValueError(f"native checkpoint schema mismatch: {incompatibility}")
        self.model.eval()
        self.theta_clamps = tuple(
            namespace[name]
            for name in (
                "parent_theta_clamp",
                "child1_theta_clamp",
                "child2_theta_clamp",
            )
        )
        self.model_id = hashlib.sha256(
            (
                UPSTREAM_REVISION
                + CHECKPOINT_SHA256
                + "bdlo1-ends-float64-torch-native-constructor-v1"
            ).encode()
        ).hexdigest()
        self.model._bpt_model_id = self.model_id
        native_module = sys.modules[self.model.__class__.__module__]
        globals_ = dict(native_module.__dict__)
        globals_["_bpt_capture_state"] = _capture_state
        method_source = (self.root / "deft/core/DEFT_sim.py").read_text()
        exec(
            compile(
                restart_method_ast(method_source),
                str(self.root / "deft/core/DEFT_sim.py"),
                "exec",
            ),
            globals_,
        )
        self.resumable = globals_["_bpt_iterative_predict_resumable"]

    def _arguments(
        self, initial_two: np.ndarray, clamps: np.ndarray
    ) -> tuple[Any, ...]:
        current, previous, packed = clamp_only_inputs(initial_two, clamps)
        return (
            len(clamps),
            self.torch.tensor(current),
            self.torch.tensor(previous),
            self.torch.tensor(packed),
            DT_S,
            *self.theta_clamps,
            False,
            None,
            False,
        )

    def native_rollout(
        self, initial_two: np.ndarray, clamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad():
            positions, velocities = self.model.iterative_predict(
                *self._arguments(initial_two, clamps)
            )
        return positions[0].cpu().numpy().copy(), velocities[0].cpu().numpy().copy()

    def rollout(
        self,
        initial_two: np.ndarray,
        clamps: np.ndarray,
        state: DeftState | None = None,
    ) -> tuple[np.ndarray, np.ndarray, DeftState]:
        if state is not None and (
            state.model_id != self.model_id or state.prediction_index >= len(clamps) - 1
        ):
            raise ValueError("restart model or absolute horizon differs")
        states = []
        with self.torch.no_grad():
            for item in self.resumable(
                self.model,
                *self._arguments(initial_two, clamps),
                _resume_state=state.clone() if state is not None else None,
            ):
                states.append(item)
        if not states:
            raise ValueError("native rollout returned no completed state")
        return (
            np.stack([_numpy(item.fields["b_DLOs_vertices"]) for item in states]),
            np.stack([_numpy(item.fields["b_DLOs_velocity"]) for item in states]),
            states[-1],
        )
