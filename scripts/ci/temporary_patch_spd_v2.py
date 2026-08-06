#!/usr/bin/env python3
"""Apply the reviewed prospective SPD-v2 finalization patch exactly once."""

from __future__ import annotations

import ast
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    count = content.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    path.write_text(content.replace(old, new), encoding="utf-8")


def test_functions(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def patch_spd_backend() -> None:
    path = Path("src/bayesian_phystwin/spd_system.py")
    replace_once(
        path,
        "from dataclasses import dataclass\nfrom typing import Final\n",
        "from dataclasses import dataclass\nfrom numbers import Real\nfrom typing import Final\n",
    )
    replace_once(path, "whitening, log\nDeterminants,", "whitening, log\ndeterminants,")
    replace_once(
        path,
        '''    if isinstance(value, (bool, np.bool_)):\n        raise TypeError(f"{name} must be a real scalar")\n    try:\n        result = float(value)\n    except (TypeError, ValueError) as error:\n        raise TypeError(f"{name} must be a real scalar") from error\n''',
        '''    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):\n        raise TypeError(f"{name} must be a real scalar")\n    result = float(value)\n''',
    )
    replace_once(
        path,
        '''        try:\n            candidate = np.asarray(value, dtype=np.float64)\n        except (TypeError, ValueError, OverflowError) as error:\n            raise SPDValidationError(\n                f"{system_name} must be a numeric float64 matrix"\n            ) from error\n''',
        '''        try:\n            untyped_candidate = np.asarray(value)\n        except (TypeError, ValueError, OverflowError) as error:\n            raise SPDValidationError(\n                f"{system_name} must be a numeric float64 matrix"\n            ) from error\n        if untyped_candidate.dtype.kind not in "fiu":\n            raise SPDValidationError(\n                f"{system_name} must be a numeric float64 matrix"\n            )\n        candidate = untyped_candidate.astype(np.float64, copy=False)\n''',
    )
    replace_once(
        path,
        '''        try:\n            right = np.asarray(value, dtype=np.float64)\n        except (TypeError, ValueError, OverflowError) as error:\n            raise SPDSolveError(f"{name} must be numeric") from error\n''',
        '''        try:\n            untyped_right = np.asarray(value)\n        except (TypeError, ValueError, OverflowError) as error:\n            raise SPDSolveError(f"{name} must be numeric") from error\n        if untyped_right.dtype.kind not in "fiu":\n            raise SPDSolveError(f"{name} must be numeric")\n        right = untyped_right.astype(np.float64, copy=False)\n''',
    )


def patch_bias_aware_v2() -> None:
    path = Path("src/bayesian_phystwin/bias_aware_belief_v2.py")
    replace_once(path, "    SPDSolveError,\n", "")
    replace_once(
        path,
        "            state_precision = state_prior_system.reconstruct_inverse()\n",
        '''            state_information_root = state_prior_system.whiten(\n                np.eye(state_count, dtype=np.float64)\n            )\n            state_precision = state_information_root.T @ state_information_root\n            state_precision = 0.5 * (state_precision + state_precision.T)\n''',
    )


def patch_stable_coverage_imports() -> None:
    path = Path("tests/test_calibration.py")
    modules = (
        "test_spd_system",
        "test_bias_aware_belief_v2",
        "test_spd_system_adversarial",
        "test_bias_aware_belief_v2_adversarial",
        "test_bias_aware_belief_v2_amendment",
    )
    imports: list[str] = []
    all_names: list[str] = []
    for module in modules:
        names = test_functions(Path("tests") / f"{module}.py")
        if not names:
            raise SystemExit(f"{module}: no test functions found")
        all_names.extend(names)
        rendered = "\n".join(f"    {name}," for name in names)
        imports.append(f"from {module} import (\n{rendered}\n)")
    if len(all_names) != len(set(all_names)):
        raise SystemExit("SPD-v2 stable test imports contain duplicate names")
    tuple_body = "\n".join(f"    {name}," for name in all_names)
    block = (
        "\n".join(imports)
        + "\n\n_SPD_V2_STABLE_TESTS = (\n"
        + tuple_body
        + "\n)\n\n"
    )
    marker = "from bayesian_phystwin import BinaryCalibrationMetrics, binary_calibration_metrics\n"
    replace_once(path, marker, block + marker)


def patch_changelog() -> None:
    path = Path("CHANGELOG.md")
    marker = "## [Unreleased]\n\n### Added\n\n"
    addition = (
        "- A versioned prospective SPD backend and bias-aware belief v2 path with "
        "strict numeric admission, one retained Cholesky factor per admitted "
        "system, residual-checked solves, explicit no-jitter failure semantics, "
        "and a protocol boundary that preserves historical v1 bytes.\n"
    )
    replace_once(path, marker, marker + addition)


def main() -> int:
    patch_spd_backend()
    patch_bias_aware_v2()
    patch_stable_coverage_imports()
    patch_changelog()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
