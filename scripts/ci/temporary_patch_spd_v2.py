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
    patch_bias_aware_v2()
    patch_stable_coverage_imports()
    patch_changelog()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
