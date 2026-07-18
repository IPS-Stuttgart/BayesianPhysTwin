"""Released-PhysTwin-centered parameterization for causal MatPhys fits."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


TEACHER_PARAMETERIZATION = "released-phystwin-bounded-logk-residual-v1"
_CHECKPOINT_GLOBALS = (
    "collide_elas",
    "collide_fric",
    "collide_object_elas",
    "collide_object_fric",
)
_OPTIMIZATION_GLOBALS = (
    "collision_dist",
    "dashpot_damping",
    "drag_damping",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: object, name: str) -> float:
    array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError(f"teacher parameter {name} must be one finite scalar")
    return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class MatPhysTeacherBundle:
    """Exact released parameter field and its byte-level provenance."""

    case_name: str
    spring_log_y: np.ndarray
    global_parameters: Mapping[str, float]
    checkpoint_path: Path
    checkpoint_sha256: str
    optimal_params_path: Path
    optimal_params_sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "spring_count": int(self.spring_log_y.size),
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
            },
            "optimal_params": {
                "path": str(self.optimal_params_path),
                "sha256": self.optimal_params_sha256,
            },
            "global_parameters": {
                name: float(value)
                for name, value in sorted(self.global_parameters.items())
            },
        }


def load_matphys_teacher_bundle(
    case_name: str,
    experiments_dir: str | Path,
    experiments_optimization_dir: str | Path,
) -> MatPhysTeacherBundle:
    """Load the same released PhysTwin artifacts used by the MatPhys dataset."""

    if not case_name:
        raise ValueError("case_name must be nonempty")
    checkpoint_candidates = sorted(
        (Path(experiments_dir).resolve() / case_name / "train").glob("best_*.pth")
    )
    if not checkpoint_candidates:
        raise FileNotFoundError(f"no released PhysTwin checkpoint for {case_name}")
    checkpoint_path = checkpoint_candidates[-1]
    optimal_path = (
        Path(experiments_optimization_dir).resolve()
        / case_name
        / "optimal_params.pkl"
    )
    if not optimal_path.is_file():
        raise FileNotFoundError(optimal_path)

    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "spring_Y" not in checkpoint:
        raise ValueError(f"{checkpoint_path}: checkpoint has no spring_Y")
    spring_y = np.asarray(
        checkpoint["spring_Y"].detach().cpu(), dtype=np.float64
    ).reshape(-1)
    if spring_y.size < 1 or not np.isfinite(spring_y).all() or np.any(spring_y <= 0):
        raise ValueError(f"{checkpoint_path}: spring_Y must be finite and positive")
    with optimal_path.open("rb") as handle:
        optimal = pickle.load(handle)
    if not isinstance(optimal, Mapping):
        raise ValueError(f"{optimal_path}: expected a parameter mapping")

    globals_: dict[str, float] = {}
    for name in _CHECKPOINT_GLOBALS:
        if name not in checkpoint:
            raise ValueError(f"{checkpoint_path}: missing {name}")
        globals_[name] = _scalar(checkpoint[name], name)
    for name in _OPTIMIZATION_GLOBALS:
        if name not in optimal:
            raise ValueError(f"{optimal_path}: missing {name}")
        globals_[name] = _scalar(optimal[name], name)

    return MatPhysTeacherBundle(
        case_name=case_name,
        spring_log_y=np.log(spring_y),
        global_parameters=globals_,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        optimal_params_path=optimal_path,
        optimal_params_sha256=_sha256_file(optimal_path),
    )


def validate_matphys_teacher_manifest(
    manifest: Mapping[str, object],
) -> None:
    """Fail closed if any released teacher artifact changed after fitting."""

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("teacher manifest must contain cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("teacher case entry must be an object")
        name = str(case.get("case_name", ""))
        if not name or name in seen:
            raise ValueError("teacher case names must be nonempty and unique")
        seen.add(name)
        for source_name in ("checkpoint", "optimal_params"):
            source = case.get(source_name)
            if not isinstance(source, Mapping):
                raise ValueError(f"{name}: missing teacher {source_name}")
            path = Path(str(source.get("path", ""))).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256_file(path) != str(source.get("sha256", "")):
                raise ValueError(f"{name}: teacher {source_name} bytes changed")


def load_matphys_teacher_manifest(
    manifest: Mapping[str, object],
) -> dict[str, MatPhysTeacherBundle]:
    """Reload hash-bound bundles recorded by a causal training audit."""

    validate_matphys_teacher_manifest(manifest)
    bundles: dict[str, MatPhysTeacherBundle] = {}
    for case in manifest["cases"]:
        assert isinstance(case, Mapping)
        name = str(case["case_name"])
        checkpoint = Path(str(case["checkpoint"]["path"])).resolve()
        optimal = Path(str(case["optimal_params"]["path"])).resolve()
        bundle = load_matphys_teacher_bundle(
            name,
            checkpoint.parents[2],
            optimal.parents[1],
        )
        if bundle.manifest() != dict(case):
            raise ValueError(f"{name}: reloaded teacher manifest differs from audit")
        bundles[name] = bundle
    return bundles


def apply_matphys_teacher_residual(
    model_out: Mapping[str, object],
    teacher: MatPhysTeacherBundle,
    residual_log_scale: float,
) -> dict[str, object]:
    """Center MatPhys on PhysTwin and bound every learned stiffness ratio.

    A scale of zero is the exact teacher identity arm. At positive scale, each
    spring can move by at most ``exp(scale)`` in either direction. Global
    contact and damping parameters stay at the released values.
    """

    if not np.isfinite(residual_log_scale) or residual_log_scale < 0.0:
        raise ValueError("residual_log_scale must be finite and nonnegative")
    import torch

    if "log_k_raw" not in model_out:
        raise ValueError("MatPhys output omits log_k_raw")
    object_raw = model_out["log_k_raw"]
    if not isinstance(object_raw, torch.Tensor):
        raise TypeError("log_k_raw must be a torch tensor")
    control_raw = model_out.get("ctrl_log_k_raw")
    control_count = (
        int(control_raw.numel()) if isinstance(control_raw, torch.Tensor) else 0
    )
    object_count = int(object_raw.numel())
    if object_count + control_count != int(teacher.spring_log_y.size):
        raise ValueError(
            f"{teacher.case_name}: teacher/model spring counts disagree "
            f"({teacher.spring_log_y.size} != {object_count}+{control_count})"
        )
    teacher_log = torch.as_tensor(
        teacher.spring_log_y,
        dtype=object_raw.dtype,
        device=object_raw.device,
    )
    result = dict(model_out)
    result["log_k"] = teacher_log[:object_count] + float(
        residual_log_scale
    ) * torch.tanh(object_raw.reshape(-1))
    if control_count:
        assert isinstance(control_raw, torch.Tensor)
        result["ctrl_log_k"] = teacher_log[object_count:] + float(
            residual_log_scale
        ) * torch.tanh(control_raw.reshape(-1))
    for name, value in teacher.global_parameters.items():
        reference = model_out.get(name, object_raw)
        if not isinstance(reference, torch.Tensor):
            reference = object_raw
        result[name] = torch.as_tensor(
            [value], dtype=reference.dtype, device=reference.device
        )
    result["teacher_parameterization"] = TEACHER_PARAMETERIZATION
    result["teacher_residual_log_scale"] = float(residual_log_scale)
    return result
