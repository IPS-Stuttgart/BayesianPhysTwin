"""Discover and invoke legacy Bayesian-PhysTwin experiment entry points.

The registry is intentionally metadata-driven: listing and describing experiments
must not import their implementation modules or optional GPU/vision dependencies.
Frozen ``bpt-*`` console scripts remain installed and are the source of truth for
the callable target; the grouped CLI provides one discoverable access path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, Protocol

DISTRIBUTION_NAME = "bayesian-phystwin"
CONSOLE_SCRIPT_GROUP = "console_scripts"

# These commands already have explicit stable grouped routes.  They remain
# installed as compatibility aliases but are not research experiments.
_STABLE_CONSOLE_SCRIPTS = frozenset(
    {
        "bpt",
        "bpt-provider-manifest",
        "bpt-replay-residuals",
        "bpt-run-manifest",
        "bpt-synthetic-benchmark",
        "bpt-validate-observation-belief",
    }
)


class ConsoleEntryPoint(Protocol):
    """Minimal importlib.metadata entry-point surface used by the registry."""

    name: str
    value: str
    group: str

    def load(self) -> Callable[[Sequence[str] | None], int | None]: ...


@dataclass(frozen=True, order=True)
class ExperimentSpec:
    """One lazily discoverable compatibility experiment command."""

    experiment_id: str
    console_script: str
    category: str
    target: str

    def as_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "console_script": self.console_script,
            "category": self.category,
            "target": self.target,
        }


def _category(experiment_id: str) -> str:
    if "deform360" in experiment_id:
        return "deform360"
    if "matphys" in experiment_id:
        return "matphys"
    if "phystwin" in experiment_id or experiment_id.startswith("structural-"):
        return "phystwin"
    if experiment_id.startswith(("benchmark-", "audit-", "diagnose-")):
        return "diagnostic"
    return "estimation"


def _installed_entry_points() -> tuple[ConsoleEntryPoint, ...]:
    try:
        package = distribution(DISTRIBUTION_NAME)
    except PackageNotFoundError as error:
        raise RuntimeError(
            "the bayesian-phystwin distribution is not installed; "
            "install the package before using the experiment registry"
        ) from error
    return tuple(package.entry_points)


def _experiment_entries(
    entry_points: Iterable[ConsoleEntryPoint] | None = None,
) -> tuple[ConsoleEntryPoint, ...]:
    source = _installed_entry_points() if entry_points is None else tuple(entry_points)
    selected = tuple(
        entry
        for entry in source
        if entry.group == CONSOLE_SCRIPT_GROUP
        and entry.name.startswith("bpt-")
        and entry.name not in _STABLE_CONSOLE_SCRIPTS
    )
    names = [entry.name for entry in selected]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate Bayesian-PhysTwin console-script entry point")
    return tuple(sorted(selected, key=lambda entry: entry.name))


def list_experiments(
    *,
    category: str | None = None,
    entry_points: Iterable[ConsoleEntryPoint] | None = None,
) -> tuple[ExperimentSpec, ...]:
    """Return installed experiment metadata without importing experiment modules."""

    normalized_category = None if category is None else category.strip().lower()
    if normalized_category == "":
        raise ValueError("category must be nonempty when provided")
    result = []
    for entry in _experiment_entries(entry_points):
        experiment_id = entry.name.removeprefix("bpt-")
        spec = ExperimentSpec(
            experiment_id=experiment_id,
            console_script=entry.name,
            category=_category(experiment_id),
            target=entry.value,
        )
        if normalized_category is None or spec.category == normalized_category:
            result.append(spec)
    return tuple(result)


def resolve_experiment(
    name: str,
    *,
    entry_points: Iterable[ConsoleEntryPoint] | None = None,
) -> tuple[ExperimentSpec, ConsoleEntryPoint]:
    """Resolve an experiment ID or exact compatibility console-script name."""

    normalized = str(name).strip()
    if not normalized:
        raise ValueError("experiment name must be nonempty")
    requested_script = normalized if normalized.startswith("bpt-") else f"bpt-{normalized}"
    entries = _experiment_entries(entry_points)
    for entry in entries:
        if entry.name == requested_script:
            experiment_id = entry.name.removeprefix("bpt-")
            return (
                ExperimentSpec(
                    experiment_id=experiment_id,
                    console_script=entry.name,
                    category=_category(experiment_id),
                    target=entry.value,
                ),
                entry,
            )
    available = ", ".join(entry.name.removeprefix("bpt-") for entry in entries)
    raise KeyError(f"unknown experiment {normalized!r}; available: {available}")


def run_experiment(
    name: str,
    arguments: Sequence[str],
    *,
    entry_points: Iterable[ConsoleEntryPoint] | None = None,
) -> int:
    """Load one selected experiment lazily and invoke its existing CLI function."""

    _, entry = resolve_experiment(name, entry_points=entry_points)
    function = entry.load()
    result: Any = function(list(arguments))
    return 0 if result is None else int(result)


__all__ = [
    "CONSOLE_SCRIPT_GROUP",
    "DISTRIBUTION_NAME",
    "ConsoleEntryPoint",
    "ExperimentSpec",
    "list_experiments",
    "resolve_experiment",
    "run_experiment",
]
