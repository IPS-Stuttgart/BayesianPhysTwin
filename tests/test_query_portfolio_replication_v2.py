from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v2 import protocol

ROOT = Path(__file__).resolve().parents[1]


def _registered_runtime(canonical_library_path: str) -> dict[str, Any]:
    return {
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYOPENGL_PLATFORM": "osmesa",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LD_LIBRARY_PATH": canonical_library_path,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    }


def test_v2_is_outcome_blind_and_uses_fresh_slingshot_seeds() -> None:
    value = protocol()
    assert value["version"] == 2
    assert value["outcomes_opened"] is False
    assert value["recovery"]["ordinary_prefix_seals"] == 0
    assert value["recovery"]["worlds_reused"] is False
    assert value["world_seeds"]["dlolab_slingshot_v4"]["evaluation"] == 263_202


def test_v2_committed_lock_matches_protocol() -> None:
    path = ROOT / "configs/experiments/query_portfolio_replication_v2.json"
    assert json.loads(path.read_text(encoding="utf-8")) == protocol()


def test_v2_translates_worker_path_only(tmp_path: Path) -> None:
    path = ROOT / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v2.py"
    spec = importlib.util.spec_from_file_location("slingshot_v2_test", path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STAGED_LIBRARY_PATH = tmp_path
    (tmp_path / "libOSMesa.so.8").touch()
    wrapper = module._load_runner()
    registered = _registered_runtime(module.CANONICAL_PARENT_LIBRARY_PATH)
    worker = wrapper.runner.native_worker_environment(registered)
    assert worker["LD_LIBRARY_PATH"] == str(module.STAGED_LIBRARY_PATH)
    assert registered["environment"]["LD_LIBRARY_PATH"] != worker["LD_LIBRARY_PATH"]


def test_v2_rejects_changed_parent_worker_environment(
    tmp_path: Path,
) -> None:
    path = ROOT / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v2.py"
    spec = importlib.util.spec_from_file_location("slingshot_v2_changed_test", path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STAGED_LIBRARY_PATH = tmp_path
    (tmp_path / "libOSMesa.so.8").touch()
    wrapper = module._load_runner()
    registered = _registered_runtime("/changed")
    try:
        wrapper.runner.native_worker_environment(registered)
    except ValueError as error:
        assert str(error) == "canonical parent worker environment changed"
    else:
        raise AssertionError("changed canonical runtime was accepted")
