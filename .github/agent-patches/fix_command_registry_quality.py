#!/usr/bin/env python3
"""Apply the narrow type-quality repair required by the command-registry PR."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_exact(
    relative_path: str,
    old: str,
    new: str,
    *,
    expected_count: int = 1,
) -> None:
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    actual_count = text.count(old)
    if actual_count != expected_count:
        raise SystemExit(
            f"{relative_path}: expected {expected_count} occurrence(s), "
            f"found {actual_count}"
        )
    path.write_text(
        text.replace(old, new, expected_count),
        encoding="utf-8",
    )


def main() -> None:
    replace_exact(
        "pyproject.toml",
        '[tool.mypy]\npython_version = "3.10"\ncheck_untyped_defs = true\n',
        '[tool.mypy]\npython_version = "3.10"\nno_site_packages = true\n'
        'check_untyped_defs = true\n',
    )
    replace_exact(
        ".github/workflows/tests.yml",
        '      - uses: actions/setup-python@v6\n'
        '        with:\n'
        '          python-version: "3.10"\n'
        '          cache: pip\n'
        '          cache-dependency-path: pyproject.toml\n'
        '      - name: Install development tools\n',
        '      - uses: actions/setup-python@v6\n'
        '        with:\n'
        '          python-version: "3.12"\n'
        '          cache: pip\n'
        '          cache-dependency-path: pyproject.toml\n'
        '      - name: Install development tools\n',
    )

    replace_exact(
        "src/bayesian_phystwin/_gauge_aware_contracts.py",
        "            query.ndim == 3 and query.shape[1:] == (3, state_count) and len(query),\n",
        "            query.ndim == 3\n"
        "            and query.shape[1:] == (3, state_count)\n"
        "            and len(query) > 0,\n",
    )
    replace_exact(
        "src/bayesian_phystwin/_gauge_aware_contracts.py",
        "        input_lineage=batch.metadata,\n",
        "        input_lineage=batch.metadata or {},\n",
    )
    replace_exact(
        "src/bayesian_phystwin/_gauge_aware_solver.py",
        "        input_lineage=batch.metadata,\n",
        "        input_lineage=batch.metadata or {},\n",
    )

    replace_exact(
        "src/bayesian_phystwin/prob4d_observation_contract.py",
        "from typing import Any\n",
        "from typing import Any, cast\n",
    )
    replace_exact(
        "src/bayesian_phystwin/prob4d_observation_contract.py",
        "    _require(\n"
        "        isinstance(parents, list) and len(parents) == window_count,\n"
        '        "gauge-posterior parent lineage changed length",\n'
        "    )\n"
        '    _require(parents[0] is None, "first Prob4D gauge must not have a parent")\n',
        "    _require(\n"
        "        isinstance(parents, list) and len(parents) == window_count,\n"
        '        "gauge-posterior parent lineage changed length",\n'
        "    )\n"
        "    parents = cast(list[Any], parents)\n"
        '    _require(parents[0] is None, "first Prob4D gauge must not have a parent")\n',
    )
    replace_exact(
        "src/bayesian_phystwin/prob4d_observation_contract.py",
        "    _require(\n"
        "        isinstance(selected, list)\n"
        "        and len(selected) == len(belief.window_names),\n"
        '        "causal lineage must identify every observation window",\n'
        "    )\n"
        "    for window_index, expected_window_id in enumerate(belief.window_names):\n",
        "    _require(\n"
        "        isinstance(selected, list)\n"
        "        and len(selected) == len(belief.window_names),\n"
        '        "causal lineage must identify every observation window",\n'
        "    )\n"
        "    selected = cast(list[Any], selected)\n"
        "    for window_index, expected_window_id in enumerate(belief.window_names):\n",
    )

    replace_exact(
        "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
        '    _require(views.ndim == 1 and len(views), "view_indices must be nonempty")\n',
        '    _require(\n'
        '        views.ndim == 1 and len(views) > 0,\n'
        '        "view_indices must be nonempty",\n'
        '    )\n',
    )
    replace_exact(
        "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
        "self.batch.metadata.get(",
        "(self.batch.metadata or {}).get(",
        expected_count=2,
    )
    replace_exact(
        "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
        "        and len(query),\n",
        "        and len(query) > 0,\n",
    )

    # The workflow and this script are scaffolding only; do not retain them.
    (ROOT / ".github/workflows/agent-fix-command-registry-quality.yml").unlink()
    Path(__file__).unlink()


if __name__ == "__main__":
    main()
