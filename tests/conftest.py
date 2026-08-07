"""Shared test-collection safeguards for coverage-focused CI runs."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure(config: object) -> None:
    """Keep new stable contract tests in explicit coverage invocations.

    The stable-core job intentionally names a bounded set of test files. When a
    new stable NumPy-only surface is added, its focused tests must not disappear
    from changed-line coverage merely because that static list has not yet been
    extended. Normal pytest runs are unchanged; coverage-driven explicit-file
    runs additionally collect registered contract tests exactly once.
    """

    if sys.gettrace() is None:
        return

    args = getattr(config, "args", None)
    if not isinstance(args, list):
        return

    test_root = Path(__file__).resolve().parent
    resolved_args = [Path(argument).resolve() for argument in args]
    contract_tests = [
        *sorted(test_root.glob("test_bias_aware_belief_v2*.py")),
        *sorted(test_root.glob("test_horizon_conditioned_discrepancy*.py")),
        *sorted(test_root.glob("test_observed_information_covariance*.py")),
        *sorted(test_root.glob("test_prob4d_visual_bias_update*.py")),
        *sorted(test_root.glob("test_query_calibration*.py")),
    ]
    for contract_test in contract_tests:
        resolved_test = contract_test.resolve()
        if resolved_test in resolved_args:
            continue
        if any(
            path.is_dir() and resolved_test.is_relative_to(path)
            for path in resolved_args
        ):
            continue
        args.append(str(resolved_test))
