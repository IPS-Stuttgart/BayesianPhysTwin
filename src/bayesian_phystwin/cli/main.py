"""Grouped, lazily imported command surface for Bayesian-PhysTwin."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Final

Route = tuple[str, ...]
_ROUTES: Final[dict[Route, tuple[str, str, str]]] = {
    ("provider", "manifest"): (
        "bayesian_phystwin.cli.provider_manifest",
        "main",
        "print the Causal4D provider manifest",
    ),
    ("observation", "validate"): (
        "bayesian_phystwin.cli.observation_belief",
        "main",
        "validate or score an ObservationBeliefV1 artifact",
    ),
    ("residual", "replay"): (
        "bayesian_phystwin.cli.residual_replay",
        "main",
        "replay exported residuals through the robust likelihood",
    ),
    ("benchmark", "synthetic"): (
        "bayesian_phystwin.cli.synthetic_benchmark",
        "main",
        "run the controlled synthetic benchmark",
    ),
    ("run", "manifest"): (
        "bayesian_phystwin.cli.run_manifest",
        "main",
        "create or validate a content-addressed run manifest",
    ),
}


def _children(prefix: Route) -> list[str]:
    position = len(prefix)
    return sorted(
        {
            route[position]
            for route in _ROUTES
            if route[:position] == prefix and len(route) > position
        }
    )


def _render_help(prefix: Route = ()) -> str:
    command = "bpt" + (" " + " ".join(prefix) if prefix else "")
    children = _children(prefix)
    lines = [f"usage: {command} <command> [arguments]", ""]
    if not prefix:
        lines.append("Grouped access to stable Bayesian-PhysTwin commands.")
        lines.append("")
    lines.append("commands:")
    for child in children:
        candidate = (*prefix, child)
        route = _ROUTES.get(candidate)
        description = route[2] if route is not None else f"{child} commands"
        lines.append(f"  {child:<14} {description}")
    lines.extend(
        [
            "",
            "Legacy bpt-* entry points remain available for compatibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve(tokens: Sequence[str]) -> tuple[Route, list[str]] | None:
    prefix: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("-"):
            break
        prefix.append(token)
        candidate = tuple(prefix)
        if candidate in _ROUTES:
            return candidate, list(tokens[index + 1 :])
        if not _children(candidate):
            return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_render_help(), end="")
        return 0

    for length in range(1, len(arguments) + 1):
        prefix = tuple(arguments[:length])
        if arguments[length - 1] in {"-h", "--help"}:
            help_namespace = prefix[:-1]
            if help_namespace == () or _children(help_namespace):
                print(_render_help(help_namespace), end="")
                return 0
            break

    resolved = _resolve(arguments)
    if resolved is None:
        matched_namespace: list[str] = []
        for token in arguments:
            candidate = (*matched_namespace, token)
            if _children(candidate):
                matched_namespace.append(token)
            else:
                break
        if matched_namespace and len(matched_namespace) == len(arguments):
            print(_render_help(tuple(matched_namespace)), end="")
            return 0
        print(_render_help(tuple(matched_namespace)), file=sys.stderr, end="")
        return 2

    route, remaining = resolved
    module_name, function_name, _ = _ROUTES[route]
    function = getattr(importlib.import_module(module_name), function_name)
    return int(function(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
