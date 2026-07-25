"""Target-free held-v8.3 entry point for the frozen AOT gsplat runtime."""

from __future__ import annotations

from typing import Any, Mapping


V83_PYCACHE_PREFIX = "/nonexistent/bpt-held-v83-pycache"


def load_and_smoke_gsplat_runtime() -> Mapping[str, Any]:
    """Load and exercise the byte-pinned extension in this exact process."""

    from . import deform360_held_gsplat_runtime as numerical

    expected = dict(numerical._EXPECTED_NORMALIZED_ENVIRONMENT)
    expected["PYTHONPYCACHEPREFIX"] = V83_PYCACHE_PREFIX
    numerical._EXPECTED_NORMALIZED_ENVIRONMENT = expected
    return numerical.load_and_smoke_gsplat_runtime()


__all__ = ["V83_PYCACHE_PREFIX", "load_and_smoke_gsplat_runtime"]
