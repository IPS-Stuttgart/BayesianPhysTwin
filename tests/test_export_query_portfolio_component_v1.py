from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module() -> Any:
    path = ROOT / "scripts/remote/export_query_portfolio_component_v1.py"
    spec = importlib.util.spec_from_file_location("component_export_test", path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_export_record_is_content_bound() -> None:
    module = _module()
    record = module._component_record(
        query_id="dlolab_wrapping_v9",
        gain=np.zeros(320),
        deployed=np.zeros(320, dtype=np.bool_),
        component_result_id="1" * 64,
        component_result_sha256="2" * 64,
    )
    body = {key: value for key, value in record.items() if key != "artifact_id"}
    assert record["artifact_id"] == module._content_id(body)


def test_component_export_rejects_nonzero_fallback_gain() -> None:
    module = _module()
    gain = np.zeros(320)
    gain[0] = 0.1
    with pytest.raises(ValueError, match="invalid complete"):
        module._component_record(
            query_id="dlolab_slingshot_v4",
            gain=gain,
            deployed=np.zeros(320, dtype=np.bool_),
            component_result_id="1" * 64,
            component_result_sha256="2" * 64,
        )
