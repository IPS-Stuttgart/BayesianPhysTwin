from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pytest

from bayesian_phystwin.experiment_registry import (
    ExperimentSpec,
    list_experiments,
    resolve_experiment,
    run_experiment,
)


@dataclass
class _FakeEntryPoint:
    name: str
    value: str
    group: str = "console_scripts"
    function: Callable[[list[str]], int | None] = lambda arguments: 0
    load_calls: list[bool] = field(default_factory=list)

    def load(self):
        self.load_calls.append(True)
        return self.function


def test_listing_is_lazy_sorted_and_excludes_stable_aliases() -> None:
    experiment = _FakeEntryPoint(
        "bpt-evaluate-phystwin-official",
        "bayesian_phystwin.cli.phystwin_official_evaluation:main",
    )
    deform = _FakeEntryPoint(
        "bpt-evaluate-deform360-online-belief",
        "bayesian_phystwin.cli.deform360_online_belief:main",
    )
    entries = (
        deform,
        _FakeEntryPoint("bpt", "bayesian_phystwin.cli.main:main"),
        _FakeEntryPoint(
            "bpt-provider-manifest",
            "bayesian_phystwin.cli.provider_manifest:main",
        ),
        experiment,
        _FakeEntryPoint("other-tool", "other:main"),
    )

    specs = list_experiments(entry_points=entries)
    assert specs == (
        ExperimentSpec(
            experiment_id="evaluate-deform360-online-belief",
            console_script="bpt-evaluate-deform360-online-belief",
            category="deform360",
            target="bayesian_phystwin.cli.deform360_online_belief:main",
        ),
        ExperimentSpec(
            experiment_id="evaluate-phystwin-official",
            console_script="bpt-evaluate-phystwin-official",
            category="phystwin",
            target="bayesian_phystwin.cli.phystwin_official_evaluation:main",
        ),
    )
    assert experiment.load_calls == []
    assert deform.load_calls == []


def test_category_filter_and_script_name_resolution() -> None:
    entries = (
        _FakeEntryPoint("bpt-report-matphys-loo-sota", "module_a:main"),
        _FakeEntryPoint("bpt-diagnose-phystwin-bias", "module_b:main"),
    )
    assert [
        spec.experiment_id
        for spec in list_experiments(category="MATPHYS", entry_points=entries)
    ] == ["report-matphys-loo-sota"]

    by_id, first_entry = resolve_experiment(
        "report-matphys-loo-sota", entry_points=entries
    )
    by_script, second_entry = resolve_experiment(
        "bpt-report-matphys-loo-sota", entry_points=entries
    )
    assert by_id == by_script
    assert first_entry is second_entry


def test_run_loads_only_the_selected_entry_point_and_forwards_arguments() -> None:
    received: list[list[str]] = []

    def selected(arguments: list[str]) -> int:
        received.append(arguments)
        return 7

    first = _FakeEntryPoint("bpt-first-experiment", "first:main")
    second = _FakeEntryPoint(
        "bpt-second-experiment",
        "second:main",
        function=selected,
    )
    result = run_experiment(
        "second-experiment",
        ["--alpha", "3"],
        entry_points=(first, second),
    )
    assert result == 7
    assert received == [["--alpha", "3"]]
    assert first.load_calls == []
    assert second.load_calls == [True]


def test_registry_rejects_duplicates_and_unknown_experiments() -> None:
    duplicate = _FakeEntryPoint("bpt-duplicate", "module:main")
    with pytest.raises(RuntimeError, match="duplicate"):
        list_experiments(entry_points=(duplicate, duplicate))
    with pytest.raises(KeyError, match="unknown experiment"):
        resolve_experiment("missing", entry_points=(duplicate,))


def test_installed_distribution_exposes_representative_experiments() -> None:
    by_id = {spec.experiment_id: spec for spec in list_experiments()}
    assert "build-phystwin-cues" in by_id
    assert "evaluate-deform360-online-belief" in by_id
    assert "report-matphys-loo-sota" in by_id
    assert "provider-manifest" not in by_id
