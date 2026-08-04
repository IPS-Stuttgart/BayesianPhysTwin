from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_full22_tempered_endpoint_diagnostic.py"
PROTOCOL = ROOT / "protocols" / "full22_tempered_endpoint_diagnostic_v1.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("full22_tempered_endpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_frozen_and_retrospective() -> None:
    module = _load_script()
    protocol, digest = module._load_protocol(PROTOCOL)

    assert len(digest) == 64
    assert protocol["status"] == "retrospective-non-claim-bearing"
    assert protocol["effective_evidence_count_caps"][-1] == 1_000_000.0
    assert "not fresh validation" in protocol["claim_boundary"]


def test_endpoint_rmse_uses_only_requested_valid_rows() -> None:
    module = _load_script()
    residual = np.array(
        [
            [[100.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
            [[200.0, 0.0, 0.0]],
        ]
    )
    valid = np.array([[True], [True], [False], [True]])

    rmse = module._endpoint_rmse(
        residual,
        valid,
        start_frame=1,
        end_frame=3,
        endpoint_mean_m=np.zeros((1, 3)),
    )

    assert rmse == pytest.approx(1.0)


def test_cap_selection_cannot_read_future_rows() -> None:
    module = _load_script()
    residual = np.zeros((8, 2, 3))
    residual[:4, :, 0] = np.arange(4)[:, None] * 0.001
    residual[4:6, :, 0] = 0.004
    valid = np.ones((8, 2), dtype=bool)
    kwargs = {
        "fit_end": 4,
        "train_end": 6,
        "caps": (1.0, 2.0, 4.0),
        "minimum_absolute_improvement_m": 0.0,
        "minimum_relative_improvement": 0.0,
    }

    first = module._select_tempered_cap(residual, valid, **kwargs)
    mutated = residual.copy()
    mutated[6:] = 10_000.0
    second = module._select_tempered_cap(mutated, valid, **kwargs)

    assert first == second


def test_guard_rejects_without_locked_margin() -> None:
    module = _load_script()
    residual = np.zeros((6, 1, 3))
    valid = np.ones((6, 1), dtype=bool)

    result = module._select_tempered_cap(
        residual,
        valid,
        fit_end=3,
        train_end=6,
        caps=(1.0, 2.0),
        minimum_absolute_improvement_m=0.0001,
        minimum_relative_improvement=0.005,
    )

    assert result["accepted"] is False
    assert result["fallback_validation_rmse_m"] == pytest.approx(0.0)


def test_protocol_rejects_reordered_caps(tmp_path: Path) -> None:
    module = _load_script()
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["effective_evidence_count_caps"] = [2.0, 1.0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique and sorted"):
        module._load_protocol(path)
