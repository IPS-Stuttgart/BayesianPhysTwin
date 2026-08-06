from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
)
from bayesian_phystwin.query_anchor_sufficiency import (
    QueryAnchorSufficiencyCurveV1,
    evaluate_query_anchor_sufficiency,
)


def _problem() -> tuple[
    NuisanceAwareInformationState,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    prior = NuisanceAwareInformationState.from_independent_priors(
        np.eye(2),
        np.asarray([[1e-4]], dtype=np.float64),
    )
    return (
        prior,
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        [
            np.asarray([[4.0, 0.0]], dtype=np.float64),
            np.asarray([[2.0, 0.0]], dtype=np.float64),
            np.asarray([[0.0, 8.0]], dtype=np.float64),
        ],
        [
            np.asarray([[4.0]], dtype=np.float64),
            np.asarray([[0.0]], dtype=np.float64),
            np.asarray([[0.0]], dtype=np.float64),
        ],
        [np.eye(1, dtype=np.float64) for _ in range(3)],
    )


def _curve() -> QueryAnchorSufficiencyCurveV1:
    prior, query, state, nuisance, covariance = _problem()
    return evaluate_query_anchor_sufficiency(
        prior,
        query,
        state,
        nuisance,
        covariance,
        precision_multipliers=[0.5, 1.0, 2.0, 4.0],
        costs=[1.0, 2.0, 1.0],
        maximum_count=3,
        target_remaining_variance_fraction=0.25,
    )


def test_curve_is_nested_and_tracks_target_support() -> None:
    curve = _curve()

    assert curve.support_counts.tolist() == [0, 1, 2, 3]
    assert curve.selected_indices[:, 0].tolist() == [1, 1, 1, 1]
    assert np.all(np.diff(curve.query_variance_traces, axis=1) <= 1e-12)
    assert np.all(np.diff(curve.cumulative_costs, axis=1) >= -1e-12)
    assert curve.first_sufficient_support.tolist() == [-1, 1, 1, 1]
    assert curve.sufficient_precision_mask.tolist() == [False, True, True, True]


def test_curve_reports_costs_prefixes_and_records() -> None:
    curve = _curve()

    assert curve.selected_prefix(1, 0).tolist() == []
    assert curve.selected_prefix(1, 1).tolist() == [1]
    assert curve.selected_prefix(1, 3).tolist() == [1, 0]
    assert curve.cumulative_costs[1].tolist() == pytest.approx(
        [0.0, 2.0, 3.0, 3.0]
    )

    records = curve.records()
    assert len(records) == 16
    assert records[0]["support_count"] == 0
    assert records[5]["selected_indices"] == [1]
    assert records[5]["target_met"] is True


def test_dependence_group_limit_is_filled_without_false_support() -> None:
    prior = NuisanceAwareInformationState.from_independent_priors(np.eye(1))
    curve = evaluate_query_anchor_sufficiency(
        prior,
        np.asarray([[1.0]]),
        [np.asarray([[2.0]]), np.asarray([[1.0]])],
        [None, None],
        [np.eye(1), np.eye(1)],
        precision_multipliers=[1.0],
        dependence_groups=["same-capture", "same-capture"],
        maximum_count=2,
    )

    assert curve.selected_counts.tolist() == [1]
    assert curve.selected_indices.tolist() == [[0, -1]]
    assert curve.query_variance_traces[0, 2] == pytest.approx(
        curve.query_variance_traces[0, 1]
    )
    assert curve.cumulative_costs[0, 2] == pytest.approx(
        curve.cumulative_costs[0, 1]
    )
    assert curve.selected_prefix(0, 2).tolist() == [0]


def test_curve_arrays_are_irreversibly_immutable() -> None:
    curve = _curve()

    for array in (
        curve.precision_multipliers,
        curve.support_counts,
        curve.selected_indices,
        curve.selected_counts,
        curve.query_variance_traces,
        curve.cumulative_costs,
        curve.first_sufficient_support,
        curve.remaining_variance_fractions,
        curve.sufficient_precision_mask,
        curve.selected_prefix(0, 2),
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_curve_contract_rejects_tampered_derived_values() -> None:
    curve = _curve()
    traces = np.asarray(curve.query_variance_traces).copy()
    traces[0, 2] = traces[0, 1] + 0.5
    with pytest.raises(ValueError, match="must not increase"):
        replace(curve, query_variance_traces=traces)

    first = np.asarray(curve.first_sufficient_support).copy()
    first[1] = -1
    with pytest.raises(ValueError, match="target crossing"):
        replace(curve, first_sufficient_support=first)

    indices = np.asarray(curve.selected_indices).copy()
    indices[0, 2] = 0
    with pytest.raises(ValueError, match="suffixes"):
        replace(curve, selected_indices=indices)


@pytest.mark.parametrize(
    ("reduction", "final_trace", "selected_cost", "total_cost", "message"),
    [
        (2.0, -1.0, 1.0, 1.0, "materially negative"),
        (0.25, 0.50, 1.0, 1.0, "do not reconstruct"),
        (0.25, 0.75, 1.0, 2.0, "selected costs"),
    ],
)
def test_planner_diagnostics_cannot_be_silently_repaired(
    monkeypatch: pytest.MonkeyPatch,
    reduction: float,
    final_trace: float,
    selected_cost: float,
    total_cost: float,
    message: str,
) -> None:
    prior, query, state, nuisance, covariance = _problem()
    selection = SimpleNamespace(
        selected_indices=(0,),
        initial_query_variance_trace=1.0,
        query_trace_reductions=np.asarray([reduction], dtype=np.float64),
        final_query_variance_trace=final_trace,
        selected_costs=np.asarray([selected_cost], dtype=np.float64),
        total_cost=total_cost,
    )
    monkeypatch.setattr(
        "bayesian_phystwin.query_anchor_sufficiency.greedy_query_aware_selection",
        lambda *_args, **_kwargs: selection,
    )

    with pytest.raises(ValueError, match=message):
        evaluate_query_anchor_sufficiency(
            prior,
            query,
            state,
            nuisance,
            covariance,
            precision_multipliers=[1.0],
            maximum_count=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision_multipliers": [1.0, 1.0]}, "strictly increasing"),
        ({"precision_multipliers": [0.0, 1.0]}, "positive"),
        ({"precision_multipliers": [True]}, "numeric vector"),
        ({"maximum_count": 4}, "candidate count"),
        ({"maximum_count": True}, "nonnegative integer"),
        ({"target_remaining_variance_fraction": 0.0}, "positive"),
        ({"target_remaining_variance_fraction": 1.1}, "at most"),
    ],
)
def test_invalid_curve_inputs_fail_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    prior, query, state, nuisance, covariance = _problem()
    with pytest.raises(ValueError, match=message):
        evaluate_query_anchor_sufficiency(
            prior,
            query,
            state,
            nuisance,
            covariance,
            **kwargs,  # type: ignore[arg-type]
        )


def test_wrong_prior_and_candidate_counts_fail_closed() -> None:
    prior, query, state, nuisance, covariance = _problem()
    with pytest.raises(TypeError, match="prior"):
        evaluate_query_anchor_sufficiency(
            object(),  # type: ignore[arg-type]
            query,
            state,
            nuisance,
            covariance,
        )
    with pytest.raises(ValueError, match="candidate input counts"):
        evaluate_query_anchor_sufficiency(
            prior,
            query,
            state,
            nuisance[:-1],
            covariance,
        )


def test_controlled_study_writes_replayable_outputs(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts/science/run_query_anchor_sufficiency_study.py"
    output_dir = tmp_path / "study"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")

    subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        check=True,
        cwd=repository_root,
        env=environment,
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["all_directional_checks_passed"] is True
    result_sha256 = summary.pop("result_sha256")
    canonical = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert result_sha256 == hashlib.sha256(canonical).hexdigest()
    assert summary["query_aware_selected_candidate_ids_at_unit_precision"][0] == (
        "independent-metric-x-efficient"
    )
    assert summary["full_state_information_selected_candidate_ids_at_unit_precision"][
        0
    ] == "query-irrelevant-y"
    assert (output_dir / "curve.csv").is_file()
    assert (output_dir / "report.md").is_file()

    repeated = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        check=False,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "refusing to replace" in repeated.stderr

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--force",
        ],
        check=True,
        cwd=repository_root,
        env=environment,
    )


def test_controlled_study_preflights_partial_outputs(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts/science/run_query_anchor_sufficiency_study.py"
    output_dir = tmp_path / "study"
    output_dir.mkdir()
    (output_dir / "curve.csv").write_text("existing\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")

    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        check=False,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "refusing to replace" in completed.stderr
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "report.md").exists()
    assert (output_dir / "curve.csv").read_text(encoding="utf-8") == "existing\n"
