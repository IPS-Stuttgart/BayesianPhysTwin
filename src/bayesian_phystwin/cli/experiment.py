"""List, inspect, and run installed Bayesian-PhysTwin experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.experiment_registry import (
    list_experiments,
    resolve_experiment,
    run_experiment,
)


def _print_specs_json(specs: Sequence[object]) -> None:
    payload = [spec.as_dict() for spec in specs]
    print(json.dumps(payload, indent=2, sort_keys=True))


def list_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bpt experiment list",
        description="List installed compatibility experiment commands without importing them.",
    )
    parser.add_argument("--category")
    parser.add_argument("--json", action="store_true", dest="as_json")
    arguments = parser.parse_args(argv)
    try:
        specs = list_experiments(category=arguments.category)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if arguments.as_json:
        _print_specs_json(specs)
        return 0
    if not specs:
        print("No installed experiments matched the requested category.")
        return 0
    width = max(len(spec.experiment_id) for spec in specs)
    for spec in specs:
        print(f"{spec.experiment_id:<{width}}  {spec.category:<10}  {spec.console_script}")
    return 0


def describe_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bpt experiment describe",
        description="Describe one installed experiment without importing its implementation.",
    )
    parser.add_argument("experiment")
    parser.add_argument("--json", action="store_true", dest="as_json")
    arguments = parser.parse_args(argv)
    try:
        spec, _ = resolve_experiment(arguments.experiment)
    except (KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if arguments.as_json:
        print(json.dumps(spec.as_dict(), indent=2, sort_keys=True))
        return 0
    print(f"experiment_id: {spec.experiment_id}")
    print(f"category: {spec.category}")
    print(f"console_script: {spec.console_script}")
    print(f"target: {spec.target}")
    return 0


def run_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bpt experiment run",
        description=(
            "Run one installed experiment through its existing compatibility entry point. "
            "Use '--' before experiment-specific arguments when needed."
        ),
    )
    parser.add_argument("experiment")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    remaining = list(parsed.arguments)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    try:
        return run_experiment(parsed.experiment, remaining)
    except (KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))


__all__ = ["describe_main", "list_main", "run_main"]
