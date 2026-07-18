from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.matphys_teacher_residual import (
    TEACHER_PARAMETERIZATION,
    apply_matphys_teacher_residual,
    load_matphys_teacher_bundle,
    validate_matphys_teacher_manifest,
)


def _teacher(tmp_path: Path):
    torch = pytest.importorskip("torch")
    case = "case"
    checkpoint = tmp_path / "experiments" / case / "train" / "best_9.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "spring_Y": torch.tensor([100.0, 200.0, 300.0]),
            "collide_elas": torch.tensor([0.1]),
            "collide_fric": torch.tensor([0.2]),
            "collide_object_elas": torch.tensor([0.3]),
            "collide_object_fric": torch.tensor([0.4]),
        },
        checkpoint,
    )
    optimal = tmp_path / "optimization" / case / "optimal_params.pkl"
    optimal.parent.mkdir(parents=True)
    with optimal.open("wb") as handle:
        pickle.dump(
            {
                "collision_dist": 0.01,
                "dashpot_damping": 40.0,
                "drag_damping": 2.0,
            },
            handle,
        )
    return load_matphys_teacher_bundle(
        case, tmp_path / "experiments", tmp_path / "optimization"
    )


def test_zero_residual_is_exact_released_parameter_identity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    teacher = _teacher(tmp_path)
    output = apply_matphys_teacher_residual(
        {
            "log_k_raw": torch.tensor([20.0, -20.0]),
            "ctrl_log_k_raw": torch.tensor([7.0]),
            "dashpot_damping": torch.tensor([999.0]),
        },
        teacher,
        0.0,
    )
    assert torch.equal(
        output["log_k"], torch.tensor(np.log([100.0, 200.0]), dtype=torch.float32)
    )
    assert torch.equal(
        output["ctrl_log_k"], torch.tensor([math.log(300.0)], dtype=torch.float32)
    )
    assert output["dashpot_damping"].item() == 40.0
    assert output["teacher_parameterization"] == TEACHER_PARAMETERIZATION


def test_teacher_residual_is_bounded_and_preserves_gradients(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    teacher = _teacher(tmp_path)
    raw = torch.tensor([100.0, -100.0], requires_grad=True)
    scale = math.log(2.0)
    output = apply_matphys_teacher_residual(
        {"log_k_raw": raw, "ctrl_log_k_raw": torch.tensor([0.0])},
        teacher,
        scale,
    )
    delta = output["log_k"] - torch.tensor(
        np.log([100.0, 200.0]), dtype=torch.float32
    )
    assert torch.all(delta.abs() <= scale + 1e-6)
    output["log_k"].sum().backward()
    assert raw.grad is not None


def test_teacher_manifest_detects_changed_source(tmp_path: Path) -> None:
    teacher = _teacher(tmp_path)
    manifest = {"cases": [teacher.manifest()]}
    validate_matphys_teacher_manifest(manifest)
    teacher.optimal_params_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        validate_matphys_teacher_manifest(manifest)


def test_teacher_spring_count_must_match_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    teacher = _teacher(tmp_path)
    with pytest.raises(ValueError, match="spring counts disagree"):
        apply_matphys_teacher_residual(
            {"log_k_raw": torch.zeros(1)}, teacher, math.log(2.0)
        )
