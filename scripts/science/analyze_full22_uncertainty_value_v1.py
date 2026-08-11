#!/usr/bin/env python3
"""Analyze where Bayesian structure adds value in the sealed full-22 evidence.

This is a retrospective source-only diagnostic. It compares every registered
Bayesian discrepancy candidate with the deterministic last-residual reference
using the already-sealed per-object, per-horizon future scores. Complete
physical object sessions are the resampling units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

REPORT_CONTRACT: Final = "bayesian-phystwin-full22-uncertainty-value-diagnostic-v1"
IMPLEMENTATION_ID: Final = "full22-uncertainty-value-analysis-v1"
REFERENCE_CANDIDATE: Final = "last_residual"
PHYSICAL_FALLBACK: Final = "physical_fallback"
COMPARISON_CANDIDATES: Final = (
    "independent_endpoint_v1",
    "dynamic_endpoint_v2",
    "structured_kernel_rank4_v1",
    "graph_dynamic_kernel_rank4_v1",
)
HORIZONS: Final = ("early", "middle", "late")
POINT_METRICS: Final = ("chamfer_distance_m", "track_error_m")
ENDPOINTS: Final = ("gaussian_nll", *POINT_METRICS)
STREAMS: Final = ("raw", "deployed")
EXPECTED_CASE_COUNT: Final = 22
EXPECTED_CANDIDATES: Final = (
    PHYSICAL_FALLBACK,
    REFERENCE_CANDIDATE,
    *COMPARISON_CANDIDATES,
)
LOWER_HEX: Final = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ScoredRow:
    case_id: str
    candidate_id: str
    horizon: str
    accepted: bool
    point: Mapping[str, float]
    fallback_point: Mapping[str, float]
    proper_score: float
    fallback_proper_score: float

    def value(self, endpoint: str, stream: str) -> float:
        if endpoint == "gaussian_nll":
            raw = self.proper_score
            fallback = self.fallback_proper_score
        else:
            raw = self.point[endpoint]
            fallback = self.fallback_point[endpoint]
        return raw if stream == "raw" or self.accepted else fallback


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _atomic_output_directory(target: Path, *, force: bool) -> Iterator[Path]:
    target = target.resolve()
    if target.exists():
        if not force:
            raise FileExistsError(f"output already exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        yield temporary
        if target.exists():
            raise FileExistsError(f"output appeared during publication: {target}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(character in value for character in "\r\n"):
        raise ValueError(f"{name} must be a single line")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _literal_sha(value: object, *, name: str, length: int) -> str:
    text = _text(value, name=name)
    if len(text) != length or set(text) - LOWER_HEX:
        raise ValueError(
            f"{name} must contain {length} lowercase hexadecimal characters"
        )
    return text


def _point_mapping(value: object, *, name: str) -> dict[str, float]:
    mapping = _mapping(value, name=name)
    if set(mapping) != set(POINT_METRICS):
        raise ValueError(f"{name} must contain exactly {list(POINT_METRICS)}")
    return {
        metric: _finite(mapping[metric], name=f"{name}.{metric}")
        for metric in POINT_METRICS
    }


def _parse_row(value: object, *, index: int) -> ScoredRow:
    name = f"rows[{index}]"
    row = _mapping(value, name=name)
    expected = {
        "case_id",
        "candidate_id",
        "horizon",
        "accepted",
        "point",
        "fallback_point",
        "proper_score",
        "fallback_proper_score",
    }
    if set(row) != expected:
        raise ValueError(
            f"{name} fields changed: missing={sorted(expected - set(row))}, "
            f"extra={sorted(set(row) - expected)}"
        )
    horizon = _text(row["horizon"], name=f"{name}.horizon")
    if horizon not in HORIZONS:
        raise ValueError(f"{name}.horizon must be one of {list(HORIZONS)}")
    candidate_id = _text(row["candidate_id"], name=f"{name}.candidate_id")
    if candidate_id not in EXPECTED_CANDIDATES:
        raise ValueError(
            f"{name}.candidate_id must be one of {list(EXPECTED_CANDIDATES)}"
        )
    return ScoredRow(
        case_id=_text(row["case_id"], name=f"{name}.case_id"),
        candidate_id=candidate_id,
        horizon=horizon,
        accepted=_boolean(row["accepted"], name=f"{name}.accepted"),
        point=_point_mapping(row["point"], name=f"{name}.point"),
        fallback_point=_point_mapping(
            row["fallback_point"], name=f"{name}.fallback_point"
        ),
        proper_score=_finite(row["proper_score"], name=f"{name}.proper_score"),
        fallback_proper_score=_finite(
            row["fallback_proper_score"],
            name=f"{name}.fallback_proper_score",
        ),
    )


def _validate_protocol(payload: object) -> tuple[Mapping[str, object], str]:
    protocol = _mapping(payload, name="protocol")
    if protocol.get("contract") != (
        "bayesian-phystwin-full22-discrepancy-candidate-tournament"
    ):
        raise ValueError("unexpected full-22 tournament protocol contract")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported full-22 tournament protocol version")
    if protocol.get("status") != "retrospective-source-only-non-claim-bearing":
        raise ValueError("protocol must remain retrospective and non-claim-bearing")
    candidates = _sequence(protocol.get("candidates"), name="protocol.candidates")
    candidate_ids = tuple(
        _text(
            _mapping(candidate, name=f"protocol.candidates[{index}]").get(
                "candidate_id"
            ),
            name=f"protocol.candidates[{index}].candidate_id",
        )
        for index, candidate in enumerate(candidates)
    )
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("protocol candidate roster or order changed")
    scoring = _mapping(protocol.get("scoring"), name="protocol.scoring")
    _text(scoring.get("proper_score_id"), name="protocol.scoring.proper_score_id")
    _finite(
        scoring.get("proper_score_observation_std_m"),
        name="protocol.scoring.proper_score_observation_std_m",
    )
    _finite(
        scoring.get("covariance_eigenvalue_floor_m2"),
        name="protocol.scoring.covariance_eigenvalue_floor_m2",
    )
    return protocol, _canonical_sha256(protocol)


def parse_inputs(
    scored_payload: object,
    protocol_payload: object,
) -> tuple[str, tuple[ScoredRow, ...], Mapping[str, object]]:
    """Validate the exact sealed score table and its frozen protocol."""

    root = _mapping(scored_payload, name="scored rows")
    expected_root = {"protocol_id", "claim_authorized", "rows"}
    if set(root) != expected_root:
        raise ValueError("raw scored-row root fields changed")
    if root["claim_authorized"] is not False:
        raise ValueError("source scored rows unexpectedly authorize a claim")
    protocol, protocol_id = _validate_protocol(protocol_payload)
    source_protocol_id = _literal_sha(
        root["protocol_id"], name="scored rows protocol_id", length=64
    )
    if source_protocol_id != protocol_id:
        raise ValueError("scored rows do not bind the supplied frozen protocol")
    rows = tuple(
        _parse_row(value, index=index)
        for index, value in enumerate(_sequence(root["rows"], name="rows"))
    )
    expected_row_count = (
        EXPECTED_CASE_COUNT * len(EXPECTED_CANDIDATES) * len(HORIZONS)
    )
    if len(rows) != expected_row_count:
        raise ValueError(
            f"expected {expected_row_count} sealed rows, received {len(rows)}"
        )
    keys = tuple((row.case_id, row.candidate_id, row.horizon) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("sealed scored rows contain duplicate units")
    case_ids = tuple(sorted({row.case_id for row in rows}))
    if len(case_ids) != EXPECTED_CASE_COUNT:
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} independent cases")
    expected_keys = {
        (case_id, candidate_id, horizon)
        for case_id in case_ids
        for candidate_id in EXPECTED_CANDIDATES
        for horizon in HORIZONS
    }
    if set(keys) != expected_keys:
        raise ValueError("sealed scored rows have an incomplete rectangular roster")
    by_unit = {(row.case_id, row.candidate_id, row.horizon): row for row in rows}
    for case_id in case_ids:
        for horizon in HORIZONS:
            fallback = by_unit[(case_id, PHYSICAL_FALLBACK, horizon)]
            for candidate_id in EXPECTED_CANDIDATES:
                row = by_unit[(case_id, candidate_id, horizon)]
                if row.fallback_point != fallback.point:
                    raise ValueError("candidate fallback point score is not exact")
                if row.fallback_proper_score != fallback.proper_score:
                    raise ValueError("candidate fallback proper score is not exact")
    return source_protocol_id, rows, protocol


def _difference_matrix(
    rows: Sequence[ScoredRow],
    *,
    candidate_id: str,
    endpoint: str,
    stream: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    by_key = {(row.case_id, row.candidate_id, row.horizon): row for row in rows}
    case_ids = tuple(sorted({row.case_id for row in rows}))
    matrix = np.empty((len(case_ids), len(HORIZONS)), dtype=np.float64)
    for case_index, case_id in enumerate(case_ids):
        for horizon_index, horizon in enumerate(HORIZONS):
            candidate = by_key[(case_id, candidate_id, horizon)]
            reference = by_key[(case_id, REFERENCE_CANDIDATE, horizon)]
            matrix[case_index, horizon_index] = candidate.value(
                endpoint, stream
            ) - reference.value(endpoint, stream)
    return case_ids, matrix


def _sign_test_pvalue(values: np.ndarray) -> float:
    nonzero = np.asarray(values, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    count = len(nonzero)
    if count == 0:
        return 1.0
    smaller = min(int(np.sum(nonzero < 0.0)), int(np.sum(nonzero > 0.0)))
    tail = sum(math.comb(count, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**count))


def _holm_adjust(pvalues: Sequence[float]) -> tuple[float, ...]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return tuple(adjusted)


def _contrast_vectors(matrix: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    if matrix.shape[1] != len(HORIZONS):
        raise AssertionError("horizon matrix changed shape")
    labels = ("overall", *HORIZONS)
    vectors = np.column_stack((np.mean(matrix, axis=1), matrix))
    return labels, vectors


def _bootstrap_family(
    matrices: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[tuple[str, str], dict[str, float]]:
    if replicates < 1000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    if not 0.5 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie strictly inside (0.5, 1)")
    columns: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    for candidate_id in COMPARISON_CANDIDATES:
        labels, vectors = _contrast_vectors(matrices[candidate_id])
        for index, label in enumerate(labels):
            keys.append((candidate_id, label))
            columns.append(vectors[:, index])
    observations = np.column_stack(columns)
    case_count, contrast_count = observations.shape
    estimates = np.mean(observations, axis=0)
    standard_deviation = np.std(observations, axis=0, ddof=1)
    standard_errors = standard_deviation / math.sqrt(case_count)
    bootstrap_means = np.empty((replicates, contrast_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk_size = min(5000, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, case_count, size=(stop - start, case_count))
        bootstrap_means[start:stop] = np.mean(observations[indices], axis=1)
    alpha = 1.0 - confidence
    lower = np.quantile(bootstrap_means, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap_means, 1.0 - alpha / 2.0, axis=0)
    denominator = np.where(standard_errors > 0.0, standard_errors, np.inf)
    max_t = np.max(np.abs((bootstrap_means - estimates) / denominator), axis=1)
    critical = float(np.quantile(max_t, confidence))
    simultaneous_lower = estimates - critical * standard_errors
    simultaneous_upper = estimates + critical * standard_errors
    result: dict[tuple[str, str], dict[str, float]] = {}
    for index, key in enumerate(keys):
        values = observations[:, index]
        result[key] = {
            "mean_difference": float(estimates[index]),
            "median_difference": float(np.median(values)),
            "standard_error": float(standard_errors[index]),
            "paired_standardized_effect": (
                0.0
                if standard_deviation[index] == 0.0
                else float(estimates[index] / standard_deviation[index])
            ),
            "individual_interval_lower": float(lower[index]),
            "individual_interval_upper": float(upper[index]),
            "simultaneous_interval_lower": float(simultaneous_lower[index]),
            "simultaneous_interval_upper": float(simultaneous_upper[index]),
            "bootstrap_probability_candidate_better": float(
                np.mean(bootstrap_means[:, index] < 0.0)
            ),
            "candidate_better_case_count": int(np.sum(values < 0.0)),
            "exact_tie_case_count": int(np.sum(values == 0.0)),
            "candidate_worse_case_count": int(np.sum(values > 0.0)),
            "worst_case_difference": float(np.max(values)),
            "best_case_difference": float(np.min(values)),
            "sign_test_pvalue": _sign_test_pvalue(values),
            "familywise_critical_value": critical,
        }
    overall_keys = [
        (candidate_id, "overall") for candidate_id in COMPARISON_CANDIDATES
    ]
    overall_pvalues = [result[key]["sign_test_pvalue"] for key in overall_keys]
    for key, adjusted in zip(overall_keys, _holm_adjust(overall_pvalues), strict=True):
        result[key]["holm_adjusted_overall_sign_pvalue"] = adjusted
    return result


def _leave_one_out_summary(matrix: np.ndarray) -> dict[str, object]:
    values = np.mean(matrix, axis=1)
    full = float(np.mean(values))
    leave_one_out = np.asarray(
        [float(np.mean(np.delete(values, index))) for index in range(len(values))]
    )
    if full < 0.0:
        sign_stable = bool(np.all(leave_one_out < 0.0))
    elif full > 0.0:
        sign_stable = bool(np.all(leave_one_out > 0.0))
    else:
        sign_stable = bool(np.all(leave_one_out == 0.0))
    return {
        "minimum_leave_one_case_out_mean": float(np.min(leave_one_out)),
        "maximum_leave_one_case_out_mean": float(np.max(leave_one_out)),
        "leave_one_case_out_sign_stable": sign_stable,
    }


def analyze_uncertainty_value(
    scored_payload: object,
    protocol_payload: object,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_confidence: float,
    source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Run the paired case-clustered retrospective scientific diagnostic."""

    protocol_id, rows, protocol = parse_inputs(scored_payload, protocol_payload)
    case_ids = tuple(sorted({row.case_id for row in rows}))
    families: dict[tuple[str, str], dict[tuple[str, str], dict[str, float]]] = {}
    matrices_by_family: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for stream in STREAMS:
        for endpoint in ENDPOINTS:
            matrices = {
                candidate_id: _difference_matrix(
                    rows,
                    candidate_id=candidate_id,
                    endpoint=endpoint,
                    stream=stream,
                )[1]
                for candidate_id in COMPARISON_CANDIDATES
            }
            matrices_by_family[(stream, endpoint)] = matrices
            families[(stream, endpoint)] = _bootstrap_family(
                matrices,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed
                + 1000 * STREAMS.index(stream)
                + 10 * ENDPOINTS.index(endpoint),
                confidence=bootstrap_confidence,
            )
    comparison_rows: list[dict[str, object]] = []
    for stream in STREAMS:
        for endpoint in ENDPOINTS:
            family = families[(stream, endpoint)]
            matrices = matrices_by_family[(stream, endpoint)]
            for candidate_id in COMPARISON_CANDIDATES:
                for aggregation in ("overall", *HORIZONS):
                    summary = dict(family[(candidate_id, aggregation)])
                    lower = float(summary["simultaneous_interval_lower"])
                    upper = float(summary["simultaneous_interval_upper"])
                    if upper < 0.0:
                        familywise_decision = "candidate_better"
                    elif lower > 0.0:
                        familywise_decision = "candidate_worse"
                    else:
                        familywise_decision = "inconclusive"
                    row: dict[str, object] = {
                        "candidate_id": candidate_id,
                        "reference_candidate": REFERENCE_CANDIDATE,
                        "stream": stream,
                        "endpoint": endpoint,
                        "aggregation": aggregation,
                        "difference_semantics": (
                            "candidate_minus_reference; lower_is_better"
                        ),
                        "independent_case_count": len(case_ids),
                        "familywise_decision": familywise_decision,
                        **summary,
                    }
                    if aggregation == "overall":
                        row.update(_leave_one_out_summary(matrices[candidate_id]))
                    comparison_rows.append(row)
    overall_raw_nll = {
        row["candidate_id"]: row
        for row in comparison_rows
        if row["stream"] == "raw"
        and row["endpoint"] == "gaussian_nll"
        and row["aggregation"] == "overall"
    }
    supported = [
        candidate_id
        for candidate_id in COMPARISON_CANDIDATES
        if overall_raw_nll[candidate_id]["familywise_decision"]
        == "candidate_better"
    ]
    regressed = [
        candidate_id
        for candidate_id in COMPARISON_CANDIDATES
        if overall_raw_nll[candidate_id]["familywise_decision"]
        == "candidate_worse"
    ]
    if supported:
        conclusion = "retrospective-uncertainty-score-signal"
    elif len(regressed) == len(COMPARISON_CANDIDATES):
        conclusion = "all-bayesian-candidates-worse-on-uncertainty-score"
    else:
        conclusion = "no-familywise-supported-uncertainty-score-gain"
    scoring = _mapping(protocol["scoring"], name="protocol.scoring")
    report: dict[str, object] = {
        "contract": REPORT_CONTRACT,
        "schema_version": 1,
        "implementation": IMPLEMENTATION_ID,
        "analysis_status": "retrospective-source-only-diagnostic",
        "protocol_id": protocol_id,
        "reference_candidate": REFERENCE_CANDIDATE,
        "comparison_candidates": list(COMPARISON_CANDIDATES),
        "statistical_unit": "physical-object-session",
        "independent_case_count": len(case_ids),
        "horizons": list(HORIZONS),
        "proper_score": {
            "proper_score_id": scoring["proper_score_id"],
            "observation_std_m": scoring["proper_score_observation_std_m"],
            "covariance_eigenvalue_floor_m2": scoring[
                "covariance_eigenvalue_floor_m2"
            ],
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence": bootstrap_confidence,
            "resampling_unit": "case_id",
            "case_weighting": "equal",
            "horizon_weighting_for_overall": "equal",
            "simultaneous_family": (
                "four Bayesian candidates times overall/early/middle/late, "
                "separately for each stream and endpoint"
            ),
            "simultaneous_interval_method": "case-clustered max-t bootstrap",
        },
        "comparisons": comparison_rows,
        "summary": {
            "primary_conclusion": conclusion,
            "familywise_supported_raw_nll_candidates": supported,
            "familywise_regressed_raw_nll_candidates": regressed,
            "best_observed_raw_nll_candidate": min(
                COMPARISON_CANDIDATES,
                key=lambda candidate_id: float(
                    overall_raw_nll[candidate_id]["mean_difference"]
                ),
            ),
            "selection_authorized": False,
            "claim_authorized": False,
        },
        "source": dict(source_metadata),
        "scientific_boundary": (
            "This analysis reuses an already-open source-only cohort and may "
            "localize uncertainty-score behavior. It cannot authorize model "
            "selection, fresh-object transfer, calibrated deployment, or a "
            "state-of-the-art claim."
        ),
        "claim_authorized": False,
        "promotion_authorized": False,
    }
    report["report_id"] = _canonical_sha256(report)
    return report


def _write_csv(path: Path, report: Mapping[str, object]) -> None:
    comparisons = _sequence(report["comparisons"], name="report.comparisons")
    fieldnames = (
        "candidate_id",
        "stream",
        "endpoint",
        "aggregation",
        "mean_difference",
        "median_difference",
        "individual_interval_lower",
        "individual_interval_upper",
        "simultaneous_interval_lower",
        "simultaneous_interval_upper",
        "bootstrap_probability_candidate_better",
        "candidate_better_case_count",
        "exact_tie_case_count",
        "candidate_worse_case_count",
        "worst_case_difference",
        "best_case_difference",
        "familywise_decision",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in comparisons:
            writer.writerow(_mapping(row, name="comparison row"))


def _format_effect(value: object) -> str:
    number = float(value)
    return f"{number:.8g}"


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    summary = _mapping(report["summary"], name="report.summary")
    comparisons = [
        _mapping(row, name="comparison row")
        for row in _sequence(report["comparisons"], name="report.comparisons")
        if _mapping(row, name="comparison row")["stream"] == "raw"
        and _mapping(row, name="comparison row")["aggregation"] == "overall"
    ]
    by_key = {
        (str(row["candidate_id"]), str(row["endpoint"])): row
        for row in comparisons
    }
    lines = [
        "# Full-22 Bayesian uncertainty-value diagnostic",
        "",
        f"**Primary conclusion:** `{summary['primary_conclusion']}`.",
        "",
        "All effects are candidate minus `last_residual`; lower is better.",
        "Intervals in the final two columns are simultaneous across all four",
        "Bayesian candidates and all four time aggregations for that endpoint.",
        "",
        (
            "| Candidate | Gaussian NLL effect | simultaneous 95% CI | "
            "Track effect (m) | Chamfer effect (m) |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate_id in COMPARISON_CANDIDATES:
        nll = by_key[(candidate_id, "gaussian_nll")]
        track = by_key[(candidate_id, "track_error_m")]
        chamfer = by_key[(candidate_id, "chamfer_distance_m")]
        interval = (
            f"[{_format_effect(nll['simultaneous_interval_lower'])}, "
            f"{_format_effect(nll['simultaneous_interval_upper'])}]"
        )
        lines.append(
            "| "
            f"`{candidate_id}` | {_format_effect(nll['mean_difference'])} | "
            f"{interval} | {_format_effect(track['mean_difference'])} | "
            f"{_format_effect(chamfer['mean_difference'])} |"
        )
    lines.extend(
        [
            "",
            "The result is retrospective and source-only. It can distinguish",
            "whether Bayesian structure changes the registered Gaussian proper",
            "score despite near-identical point predictions, but it does not",
            "authorize candidate selection or a confirmatory claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_scored_rows_json", type=Path)
    parser.add_argument("protocol_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-run-attempt", required=True)
    parser.add_argument("--source-head-sha", required=True)
    parser.add_argument("--source-artifact-id", required=True)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--analyzer-revision", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scored_payload = json.loads(args.raw_scored_rows_json.read_text(encoding="utf-8"))
    protocol_payload = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    source_metadata = {
        "run_id": _text(args.source_run_id, name="source_run_id"),
        "run_attempt": _text(args.source_run_attempt, name="source_run_attempt"),
        "head_sha": _literal_sha(
            args.source_head_sha, name="source_head_sha", length=40
        ),
        "artifact_id": _text(args.source_artifact_id, name="source_artifact_id"),
        "artifact_name": _text(
            args.source_artifact_name, name="source_artifact_name"
        ),
        "artifact_digest": (
            "sha256:"
            + _literal_sha(
                args.source_artifact_digest.removeprefix("sha256:"),
                name="source_artifact_digest",
                length=64,
            )
        ),
        "raw_scored_rows_sha256": _file_sha256(args.raw_scored_rows_json),
        "raw_scored_rows_bytes": args.raw_scored_rows_json.stat().st_size,
        "protocol_file_sha256": _file_sha256(args.protocol_json),
        "analyzer_revision": _literal_sha(
            args.analyzer_revision, name="analyzer_revision", length=40
        ),
    }
    report = analyze_uncertainty_value(
        scored_payload,
        protocol_payload,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
        source_metadata=source_metadata,
    )
    with _atomic_output_directory(args.output_dir, force=args.force) as temporary:
        _write_json(temporary / "full22_uncertainty_value_report.json", report)
        _write_csv(temporary / "full22_uncertainty_value_table.csv", report)
        _write_markdown(temporary / "full22_uncertainty_value_summary.md", report)
        shutil.copy2(
            args.raw_scored_rows_json,
            temporary / "raw_scored_rows.json",
        )
        shutil.copy2(
            args.protocol_json,
            temporary / "full22_discrepancy_candidate_tournament_v1.json",
        )
    print(
        json.dumps(
            {
                "status": "written",
                "output_dir": str(args.output_dir.resolve(strict=False)),
                "report_id": report["report_id"],
                "primary_conclusion": report["summary"]["primary_conclusion"],
                "claim_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
