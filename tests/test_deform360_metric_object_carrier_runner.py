from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "remote"
    / "run_deform360_metric_object_carrier_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("_metric_object_carrier_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_npz_writer_is_byte_deterministic(tmp_path: Path) -> None:
    arrays = {
        "z": np.arange(12, dtype=np.float64).reshape(4, 3),
        "a": np.asarray([True, False, True]),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    MODULE._deterministic_npz(first, arrays)
    MODULE._deterministic_npz(second, dict(reversed(tuple(arrays.items()))))
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as stored:
        assert np.array_equal(stored["z"], arrays["z"])
        assert np.array_equal(stored["a"], arrays["a"])


def test_selector_loader_does_not_execute_circular_package_init(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "causal4d_public"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('package init must not execute')\n", encoding="utf-8"
    )
    (package / "deform360_sam2.py").write_text("VALUE = 7\n", encoding="utf-8")
    source = package / "deform360_object_sam2.py"
    source.write_text(
        "from .deform360_sam2 import VALUE\n"
        "class DeformableObjectSam2VideoPredictor:\n"
        "    value = VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "causal4d_public", raising=False)
    monkeypatch.delitem(
        sys.modules, "causal4d_public.deform360_object_sam2", raising=False
    )
    selector = MODULE._load_selector(source)
    assert selector.value == 7
