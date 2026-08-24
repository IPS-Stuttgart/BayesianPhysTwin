from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.recursive_corruption_benchmark_v2 import (
    CONDITIONS,
    IMPERFECT_CUE_CONDITIONS,
    PRIMARY_ENDPOINTS,
    RecursiveCorruptionV2Config,
    draw_seed_domain,
    generate_corrupted_sequence_v2,
    run_methods_v2,
    run_recursive_corruption_benchmark_v2,
    write_deterministic_trace_npz,
    write_json,
    write_records_csv,
)


def _config() -> RecursiveCorruptionV2Config:
    return RecursiveCorruptionV2Config(
        step_count=200,
        recovery_window=30,
        bootstrap_replicates=1_000,
    )


def _load_analysis_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "science"
        / "analyze_recursive_corruption_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "recursive_corruption_v2_analysis",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_domains_vary_physics_and_schedule() -> None:
    config = _config()
    first = draw_seed_domain(1, config)
    second = draw_seed_domain(2, config)
    assert first != second
    assert first.stiffness != second.stiffness
    assert first.action_frequency_1 != second.action_frequency_1
    assert (
        first.corruption_start + first.corruption_length + config.recovery_window
        < config.step_count
    )
    assert (
        second.corruption_start + second.corruption_length + config.recovery_window
        < config.step_count
    )


def test_imperfect_cue_conditions_are_not_oracle_labels() -> None:
    config = _config()
    domain = draw_seed_domain(7, config)
    start, stop = domain.corruption_start, domain.corruption_stop

    false_negative = generate_corrupted_sequence_v2(
        "reliability_false_negative", domain=domain, config=config
    )
    assert np.all(false_negative.reliability[start:stop] > config.minimum_reliability)
    assert not np.allclose(
        false_negative.observation_m[start:stop],
        false_negative.true_position_m[start:stop],
    )

    false_positive = generate_corrupted_sequence_v2(
        "reliability_false_positive", domain=domain, config=config
    )
    clean = generate_corrupted_sequence_v2("clean", domain=domain, config=config)
    assert np.allclose(
        false_positive.observation_m,
        clean.observation_m,
        equal_nan=True,
    )
    assert np.any(false_positive.reliability[start:stop] < config.minimum_reliability)

    jitter = generate_corrupted_sequence_v2(
        "timestamp_jitter",
        domain=domain,
        config=config,
    )
    assert np.array_equal(jitter.actual_source_step[start:stop], np.arange(start, stop))
    assert np.any(
        jitter.reported_source_step[start:stop] != jitter.actual_source_step[start:stop]
    )

    partial = generate_corrupted_sequence_v2(
        "partial_stale",
        domain=domain,
        config=config,
    )
    stale = partial.actual_source_step[start:stop] != np.arange(start, stop)
    assert np.any(stale) and np.any(~stale)

    mixed = generate_corrupted_sequence_v2(
        "mixed_identity",
        domain=domain,
        config=config,
    )
    assert np.all(mixed.reliability[start:stop] >= 0.55)
    assert not np.allclose(
        mixed.observation_m[start:stop],
        mixed.true_position_m[start:stop],
    )
    assert tuple(IMPERFECT_CUE_CONDITIONS) == (
        "reliability_false_negative",
        "reliability_false_positive",
        "timestamp_jitter",
        "partial_stale",
        "mixed_identity",
    )


def test_matched_guarded_arms_receive_the_same_declared_cues() -> None:
    config = _config()
    domain = draw_seed_domain(11, config)
    sequence = generate_corrupted_sequence_v2(
        "outlier_burst",
        domain=domain,
        config=config,
    )
    traces = run_methods_v2(sequence, domain=domain, config=config)
    start, stop = domain.corruption_start, domain.corruption_stop
    guarded_last = traces["guarded_last_residual"]
    guarded_gaussian = traces["guarded_recursive"]
    assert np.all(guarded_last.exact_fallback[start:stop])
    assert np.all(guarded_gaussian.exact_fallback[start:stop])
    assert np.array_equal(
        guarded_last.fallback_reason_code[start:stop],
        guarded_gaussian.fallback_reason_code[start:stop],
    )
    assert np.all(guarded_last.exact_fallback_valid)
    assert np.all(guarded_gaussian.exact_fallback_valid)


def test_complete_result_has_two_primary_endpoints_and_null_unsupported_metrics() -> (
    None
):
    config = _config()
    result, traces = run_recursive_corruption_benchmark_v2(
        seeds=(1, 2, 3),
        conditions=CONDITIONS,
        config=config,
        retain_traces=True,
    )
    assert traces is not None
    assert tuple(result["primary_endpoints"]) == PRIMARY_ENDPOINTS
    assert len(result["records"]) == 3 * len(CONDITIONS) * 5
    physical = [
        record
        for record in result["records"]
        if record["method"] == "physical_baseline"
    ]
    assert physical
    assert all(
        record["materially_harmful_accepted_update_count"] is None
        for record in physical
    )
    deterministic = [
        record
        for record in result["records"]
        if record["method"] in {"last_residual", "guarded_last_residual"}
    ]
    assert all(record["gaussian_nll"] is None for record in deterministic)
    assert all(record["coverage_90"] is None for record in deterministic)
    assert all(
        record["mean_full_interval_width_90_m"] is None for record in deterministic
    )
    json.dumps(result, allow_nan=False)


def test_trace_archive_is_deterministic_and_analysis_is_exactly_reproducible(
    tmp_path: Path,
) -> None:
    config = _config()
    result, traces = run_recursive_corruption_benchmark_v2(
        seeds=(1, 2, 3, 4),
        conditions=CONDITIONS,
        config=config,
        retain_traces=True,
    )
    assert traces is not None
    result_path = tmp_path / "result.json"
    records_path = tmp_path / "records.csv"
    first_trace = tmp_path / "traces-a.npz"
    second_trace = tmp_path / "traces-b.npz"
    write_json(result, result_path)
    write_records_csv(result, records_path)
    write_deterministic_trace_npz(arrays=traces, result=result, path=first_trace)
    write_deterministic_trace_npz(arrays=traces, result=result, path=second_trace)
    assert (
        hashlib.sha256(first_trace.read_bytes()).digest()
        == hashlib.sha256(second_trace.read_bytes()).digest()
    )

    analysis_module = _load_analysis_module()
    analysis_dir = tmp_path / "analysis"
    analysis = analysis_module.analyze(
        result_path=result_path,
        trace_path=first_trace,
        output_dir=analysis_dir,
    )
    assert analysis["primary_endpoint_names"] == list(PRIMARY_ENDPOINTS)
    assert analysis["independent_unit"] == "fresh seed-domain"
    assert analysis["dynamics_vary_across_seeds"] is True
    assert analysis["metric_support"]["gaussian_nll"]["undefined_for_methods"] == [
        "physical_baseline",
        "last_residual",
        "guarded_last_residual",
    ]
    analysis_module.check_reproduction(
        result_path=result_path,
        trace_path=first_trace,
        output_dir=analysis_dir,
    )
    assert (analysis_dir / "time-summary.csv").stat().st_size > 0
