"""Extract DLO-specific initialization from the locked external DEFORM source."""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class DeformDLOInitialization:
    """Official initial geometry and stiffness parsed from upstream code."""

    dlo_type: str
    rest_vertices_m: tuple[tuple[float, float, float], ...]
    bend_stiffness: float
    twist_stiffness: float
    source_sha256: str

    @property
    def node_count(self) -> int:
        return len(self.rest_vertices_m)

    def to_record(self) -> dict[str, object]:
        return {
            "contract": "official-deform-dlo-initialization-v1",
            "dlo_type": self.dlo_type,
            "node_count": self.node_count,
            "coordinate_transform": "raw-x-raw-z-negated-raw-y",
            "bend_stiffness": self.bend_stiffness,
            "twist_stiffness": self.twist_stiffness,
            "upstream_train_deform_sha256": self.source_sha256,
        }


def _branch_for_dlo(tree: ast.AST, dlo_type: str) -> ast.If:
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "DLO_type"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == dlo_type
        ):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"upstream DEFORM has {len(matches)} branches for {dlo_type}")
    return matches[0]


def _assigned_attribute(statement: ast.Assign) -> str | None:
    if len(statement.targets) != 1 or not isinstance(
        statement.targets[0], ast.Attribute
    ):
        return None
    return statement.targets[0].attr


def _tensor_literal(value: ast.AST) -> tuple[tuple[float, float, float], ...] | None:
    for node in ast.walk(value):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "tensor"
            and node.args
        ):
            literal = ast.literal_eval(node.args[0])
            if not isinstance(literal, (list, tuple)):
                return None
            rows: list[tuple[float, float, float]] = []
            for row in literal:
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    return None
                rows.append((float(row[0]), float(row[1]), float(row[2])))
            if rows:
                return tuple(rows)
    return None


def _ones_multiplier(value: ast.AST) -> float | None:
    for node in ast.walk(value):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        pairs = ((node.left, node.right), (node.right, node.left))
        for scalar, expression in pairs:
            if not isinstance(scalar, ast.Constant) or not isinstance(
                scalar.value, (int, float)
            ):
                continue
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and isinstance(expression.func.value, ast.Name)
                and expression.func.value.id == "torch"
                and expression.func.attr == "ones"
            ):
                return float(scalar.value)
    return None


@cache
def load_deform_dlo_initialization(
    train_deform_path: str | Path,
    dlo_type: str,
) -> DeformDLOInitialization:
    """Parse one official DLO branch without importing or vendoring upstream code."""

    if dlo_type not in ("DLO1", "DLO2"):
        raise ValueError(f"unsupported registered DEFORM DLO type: {dlo_type}")
    source = Path(train_deform_path).resolve()
    payload = source.read_bytes()
    tree = ast.parse(payload.decode("utf-8"), filename=str(source))
    branch = _branch_for_dlo(tree, dlo_type)

    raw_vertex_candidates = []
    bend_candidates = []
    twist_candidates = []
    for statement in branch.body:
        if not isinstance(statement, ast.Assign):
            continue
        if (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "rest_vert"
        ):
            candidate = _tensor_literal(statement.value)
            if candidate is not None:
                raw_vertex_candidates.append(candidate)
            continue
        attribute = _assigned_attribute(statement)
        if attribute == "bend_stiffness":
            bend_candidates.append(_ones_multiplier(statement.value))
        elif attribute == "twist_stiffness":
            twist_candidates.append(_ones_multiplier(statement.value))

    expected_nodes = 13 if dlo_type == "DLO1" else 12
    if (
        len(raw_vertex_candidates) != 1
        or len(bend_candidates) != 1
        or len(twist_candidates) != 1
    ):
        raise ValueError(f"upstream {dlo_type} initialization is ambiguous")
    raw_vertices = raw_vertex_candidates[0]
    bend_stiffness = bend_candidates[0]
    twist_stiffness = twist_candidates[0]
    if (
        len(raw_vertices) != expected_nodes
        or bend_stiffness is None
        or twist_stiffness is None
        or not math.isfinite(bend_stiffness)
        or bend_stiffness <= 0.0
        or not math.isfinite(twist_stiffness)
        or twist_stiffness <= 0.0
    ):
        raise ValueError(f"upstream {dlo_type} initialization is incomplete")
    transformed = tuple((x, z, -y) for x, y, z in raw_vertices)
    if any(not all(math.isfinite(value) for value in row) for row in transformed):
        raise ValueError(f"upstream {dlo_type} rest geometry is non-finite")
    return DeformDLOInitialization(
        dlo_type=dlo_type,
        rest_vertices_m=transformed,
        bend_stiffness=bend_stiffness,
        twist_stiffness=twist_stiffness,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )
