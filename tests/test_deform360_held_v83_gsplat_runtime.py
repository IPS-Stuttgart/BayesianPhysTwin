from __future__ import annotations

from bayesian_phystwin import deform360_held_gsplat_runtime as numerical
from bayesian_phystwin import deform360_held_v82_gsplat_runtime as previous
from bayesian_phystwin import deform360_held_v83_gsplat_runtime as runtime


def test_v83_adapter_has_a_separate_normalized_pycache_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        numerical,
        "_EXPECTED_NORMALIZED_ENVIRONMENT",
        {"PYTHONPYCACHEPREFIX": "/legacy"},
    )
    monkeypatch.setattr(
        numerical,
        "load_and_smoke_gsplat_runtime",
        lambda: dict(numerical._EXPECTED_NORMALIZED_ENVIRONMENT),
    )

    observed = runtime.load_and_smoke_gsplat_runtime()

    assert observed["PYTHONPYCACHEPREFIX"] == (
        "/nonexistent/bpt-held-v83-pycache"
    )
    assert runtime.V83_PYCACHE_PREFIX != previous.V82_PYCACHE_PREFIX
    assert previous.V82_PYCACHE_PREFIX == "/nonexistent/bpt-held-v82-pycache"
