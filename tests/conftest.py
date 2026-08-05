"""Shared test-collection safeguards for coverage-focused CI runs."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure(config: object) -> None:
    """Keep horizon-contract tests in explicit coverage-suite invocations.

    The stable-core job intentionally names a bounded set of test files. When a
    new stable NumPy-only surface is added, its focused test must not disappear
    from changed-line coverage merely because that static list has not yet been
    extended. Normal pytest runs are unchanged; coverage-driven explicit-file
    runs additionally collect the horizon contract test exactly once.
    """

    if sys.gettrace() is None:
        return

    args = getattr(config, "args", None)
    if not isinstance(args, list):
        return

    horizon_test = (
        Path(__file__).with_name("test_horizon_conditioned_discrepancy.py").resolve()
    )
    resolved_args = [Path(argument).resolve() for argument in args]
    if horizon_test in resolved_args:
        return
    if any(
        path.is_dir() and horizon_test.is_relative_to(path) for path in resolved_args
    ):
        return

    args.append(str(horizon_test))
