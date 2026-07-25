"""Target-free held-v8.2 entry point for the frozen AOT gsplat smoke."""

from __future__ import annotations

from typing import Any, Mapping


V82_PYCACHE_PREFIX = "/nonexistent/bpt-held-v82-pycache"


def load_and_smoke_gsplat_runtime() -> Mapping[str, Any]:
    """Run the byte-pinned smoke under the held-v8.2 environment."""

    from . import deform360_held_gsplat_runtime as numerical

    expected = dict(numerical._EXPECTED_NORMALIZED_ENVIRONMENT)
    expected["PYTHONPYCACHEPREFIX"] = V82_PYCACHE_PREFIX
    numerical._EXPECTED_NORMALIZED_ENVIRONMENT = expected
    return numerical.load_and_smoke_gsplat_runtime()


__all__ = ["V82_PYCACHE_PREFIX", "load_and_smoke_gsplat_runtime"]
