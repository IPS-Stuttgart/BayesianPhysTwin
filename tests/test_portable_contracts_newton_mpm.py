from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.newton_mpm_backend_v1 import (
    load_newton_particle_rollout,
    validate_newton_mpm_runtime_manifest,
)
from bayesian_phystwin.physical_rollout_v1 import validate_physical_rollout_arrays


def _focused_tests() -> ModuleType:
    path = Path(__file__).with_name("test_newton_mpm_backend_v1.py")
    spec = importlib.util.spec_from_file_location("_newton_mpm_focused_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load focused Newton MPM tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FOCUSED = _focused_tests()


@pytest.mark.parametrize(
    "name",
    [
        "test_materializes_generic_physical_rollout_and_preserves_query_identity",
        "test_materialization_is_byte_deterministic",
        "test_rejects_duplicate_or_out_of_range_material_queries",
        "test_rejects_frame_zero_drift_and_changed_units",
        "test_bundle_detects_mutated_particle_provenance",
    ],
)
def test_newton_mpm_focused_contracts_are_in_stable_suite(
    tmp_path: Path,
    name: str,
) -> None:
    getattr(_FOCUSED, name)(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda arrays: arrays.pop("action_support"), "cannot load raw"),
        (
            lambda arrays: arrays.__setitem__(
                "driven_particle_positions_m",
                arrays["driven_particle_positions_m"][0],
            ),
            "shape (T,P,3)",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "zero_action_particle_positions_m",
                arrays["zero_action_particle_positions_m"].astype(np.float64),
            ),
            "share a floating dtype",
        ),
        (
            lambda arrays: arrays["driven_particle_positions_m"].__setitem__(
                (1, 0, 0), np.nan
            ),
            "non-finite",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "material_query_indices", np.array([], dtype=np.int64)
            ),
            "must not be empty",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "action_support", np.array([0.0], dtype=np.float32)
            ),
            "action_support is invalid",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "action_support", arrays["action_support"].astype(np.float64)
            ),
            "dtype differs",
        ),
    ],
)
def test_raw_newton_rollout_validation_branches(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    arrays = _FOCUSED._raw_arrays()
    mutate(arrays)
    path = _FOCUSED._raw_archive(tmp_path / "raw.npz", arrays)
    with pytest.raises(ValueError, match=message):
        load_newton_particle_rollout(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda arrays: arrays.pop("action_support"), "array roster changed"),
        (
            lambda arrays: arrays.__setitem__(
                "prediction_m", arrays["prediction_m"][0]
            ),
            "shape (T,N,3)",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "action_support", arrays["action_support"].astype(np.float64)
            ),
            "dtypes differ",
        ),
        (
            lambda arrays: arrays["prediction_m"].__setitem__((1, 0, 0), np.nan),
            "non-finite",
        ),
        (
            lambda arrays: arrays["action_support"].__setitem__(0, -1.0),
            "outside",
        ),
        (
            lambda arrays: arrays["persistence_m"].__setitem__((1, 0, 0), 1.0),
            "persistence is not exact",
        ),
    ],
)
def test_physical_rollout_validation_branches(mutate: Any, message: str) -> None:
    raw = _FOCUSED._raw_arrays()
    indices = raw["material_query_indices"]
    prediction = raw["driven_particle_positions_m"][:, indices]
    arrays = {
        "prediction_m": prediction.copy(),
        "persistence_m": np.repeat(prediction[0][None], len(prediction), axis=0),
        "driven_readout_m": prediction.copy(),
        "zero_action_readout_m": raw["zero_action_particle_positions_m"][:, indices],
        "action_support": raw["action_support"].copy(),
        "frame_zero_points_m": prediction[0].copy(),
    }
    mutate(arrays)
    with pytest.raises(ValueError, match=message):
        validate_physical_rollout_arrays(arrays)


def _install_importable_optional_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    warp = types.ModuleType("warp")
    warp.__version__ = "1.16.0"
    warp.kernel = lambda function: function
    warp.get_device = lambda alias: SimpleNamespace(alias=alias, name="fake CUDA")

    class ScopedDevice:
        def __init__(self, device: Any) -> None:
            self.device = device

        def __enter__(self) -> ScopedDevice:
            return self

        def __exit__(self, *exc: Any) -> None:
            del exc

    warp.ScopedDevice = ScopedDevice

    newton = types.ModuleType("newton")
    newton.__version__ = "1.5.0"
    solvers = types.ModuleType("newton.solvers")
    solvers.SolverImplicitMPM = object
    newton.solvers = solvers

    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "newton.solvers", solvers)
    sys.modules.pop("bayesian_phystwin._newton_mpm_runtime", None)


def _runtime_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _install_importable_optional_runtime(monkeypatch)
    return importlib.import_module("bayesian_phystwin._newton_mpm_runtime")


def _trajectories(frame_count: int = 4) -> tuple[np.ndarray, np.ndarray]:
    frame_zero = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    zero = np.repeat(frame_zero[None], frame_count, axis=0)
    driven = zero.copy()
    driven[:, -1, 2] = np.linspace(0.0, 0.02, frame_count, dtype=np.float32)
    return driven, zero


def test_optional_runtime_wrapper_materializes_valid_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_module(monkeypatch)
    driven, zero = _trajectories()
    outputs = iter((driven, zero))
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: next(outputs))
    config = runtime.NewtonMpmSmokeConfig(frame_count=4, query_count=2)
    raw_path = tmp_path / "raw.npz"
    manifest_path = tmp_path / "runtime.json"
    result = runtime.run_newton_mpm_smoke(
        raw_rollout_path=raw_path,
        runtime_manifest_path=manifest_path,
        config=config,
    )

    assert result["maximum_action_response_m"] == pytest.approx(0.02)
    assert result["runtime"]["device_name"] == "fake CUDA"
    assert result["config"]["frame_count"] == 4
    validate_newton_mpm_runtime_manifest(
        result["runtime"],
        raw_rollout_path=raw_path,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_count", 1, "at least two"),
        ("query_count", 0, "positive"),
        ("fps", np.nan, "finite and positive"),
        ("substeps", 0, "positive"),
        ("voxel_size_m", 0.0, "finite and positive"),
        ("poisson_ratio", 0.5, "poisson_ratio"),
        ("damping", -1.0, "damping"),
        ("max_iterations", 0, "positive"),
    ],
)
def test_optional_runtime_config_validation(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    runtime = _runtime_module(monkeypatch)
    config = replace(runtime.NewtonMpmSmokeConfig(), **{field: value})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_optional_runtime_rejects_bad_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_module(monkeypatch)
    with pytest.raises(TypeError, match="NewtonMpmSmokeConfig"):
        runtime.run_newton_mpm_smoke(
            raw_rollout_path=tmp_path / "raw.npz",
            runtime_manifest_path=tmp_path / "runtime.json",
            config=object(),
        )

    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"exists")
    with pytest.raises(FileExistsError, match="already exists"):
        runtime.run_newton_mpm_smoke(
            raw_rollout_path=existing,
            runtime_manifest_path=tmp_path / "runtime.json",
        )

    config = runtime.NewtonMpmSmokeConfig(frame_count=4, query_count=2)
    driven, zero = _trajectories()
    zero[0, 0, 0] = 1.0
    outputs = iter((driven, zero))
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: next(outputs))
    with pytest.raises(RuntimeError, match="frame zero"):
        runtime.run_newton_mpm_smoke(
            raw_rollout_path=tmp_path / "mismatch.npz",
            runtime_manifest_path=tmp_path / "mismatch.json",
            config=config,
        )

    identical = _trajectories()[1]
    monkeypatch.setattr(runtime, "_simulate_one", lambda *args, **kwargs: identical)
    with pytest.raises(RuntimeError, match="no action-conditioned response"):
        runtime.run_newton_mpm_smoke(
            raw_rollout_path=tmp_path / "no-response.npz",
            runtime_manifest_path=tmp_path / "no-response.json",
            config=config,
        )


def test_newton_cli_dispatches_all_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bayesian_phystwin.cli.newton_mpm_backend as cli

    calls: list[tuple[str, Any]] = []

    def materialize(**kwargs: Any) -> dict[str, object]:
        calls.append(("materialize", kwargs))
        return {"kind": "materialized"}

    def validate(path: Path) -> dict[str, object]:
        calls.append(("validate", path))
        return {"kind": "validated"}

    monkeypatch.setattr(cli, "materialize_newton_mpm_backend", materialize)
    monkeypatch.setattr(cli, "validate_newton_mpm_backend", validate)
    assert cli.main(
        [
            "materialize",
            str(tmp_path / "raw.npz"),
            str(tmp_path / "runtime.json"),
            str(tmp_path / "bundle"),
        ]
    ) == 0
    assert cli.main(["validate", str(tmp_path / "bundle")]) == 0

    fake_runtime = types.ModuleType("bayesian_phystwin._newton_mpm_runtime")

    class Config:
        def __init__(self, **kwargs: Any) -> None:
            self.values = kwargs

    def run(**kwargs: Any) -> dict[str, object]:
        calls.append(("run", kwargs))
        return {}

    fake_runtime.NewtonMpmSmokeConfig = Config
    fake_runtime.run_newton_mpm_smoke = run
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin._newton_mpm_runtime",
        fake_runtime,
    )
    assert cli.main(["smoke", str(tmp_path / "smoke")]) == 0
    assert {name for name, _ in calls} == {"materialize", "validate", "run"}
    assert '"kind": "materialized"' in capsys.readouterr().out


def test_newton_cli_reports_missing_optional_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesian_phystwin.cli.newton_mpm_backend as cli

    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "bayesian_phystwin._newton_mpm_runtime":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("bayesian_phystwin._newton_mpm_runtime", None)
    args = cli.build_parser().parse_args(["smoke", str(tmp_path / "output")])
    with pytest.raises(RuntimeError, match="optional"):
        cli._smoke(args)
