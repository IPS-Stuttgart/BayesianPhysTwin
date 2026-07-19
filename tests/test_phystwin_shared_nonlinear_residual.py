from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin.phystwin_shared_nonlinear_residual import (
    SHARED_NONLINEAR_RESIDUAL_CONTRACT,
    SharedNonlinearResidualConfig,
    _build_model,
    _feature_dimension,
    _knn_indices,
    _load_protocol,
    _write_json_with_digest,
    blend_with_persistence,
)


def test_knn_indices_are_deterministic_and_exclude_self() -> None:
    points = np.stack(
        (np.arange(6, dtype=float), np.zeros(6), np.zeros(6)), axis=1
    )
    first = _knn_indices(points, 2)
    second = _knn_indices(points, 2)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (6, 2)
    assert not np.any(first == np.arange(6)[:, None])


def test_blend_zero_is_exact_persistence_and_one_is_dynamic() -> None:
    endpoint = np.arange(12, dtype=float).reshape(4, 3)
    dynamic = np.stack((endpoint + 1.0, endpoint + 2.0))
    np.testing.assert_array_equal(
        blend_with_persistence(dynamic, endpoint, 0.0),
        np.broadcast_to(endpoint, dynamic.shape),
    )
    np.testing.assert_array_equal(
        blend_with_persistence(dynamic, endpoint, 1.0), dynamic
    )


def test_protocol_requires_disjoint_complete_fold_coverage(tmp_path) -> None:
    payload = {
        "contract": SHARED_NONLINEAR_RESIDUAL_CONTRACT,
        "source_cases": ["a", "b", "c"],
        "target_cases": ["target"],
        "source_folds": [
            {"held_out_cases": ["a", "b"]},
            {"held_out_cases": ["c"]},
        ],
        "model": {},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, config = _load_protocol(path)
    assert loaded["source_cases"] == ["a", "b", "c"]
    assert config == SharedNonlinearResidualConfig()

    payload["target_cases"] = ["a"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        _load_protocol(path)


def test_zero_initialized_model_predicts_no_residual_velocity() -> None:
    torch = pytest.importorskip("torch")
    config = SharedNonlinearResidualConfig(hidden_dim=8, hidden_layers=2)
    model = _build_model(torch, config)
    output = model(torch.randn(5, _feature_dimension()))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_json_digest_is_external_and_verifiable(tmp_path) -> None:
    path = tmp_path / "summary.json"
    digest = _write_json_with_digest(path, {"gate": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"gate": False}
    assert path.with_suffix(".json.sha256").read_text(encoding="ascii") == (
        f"{digest}  summary.json\n"
    )
