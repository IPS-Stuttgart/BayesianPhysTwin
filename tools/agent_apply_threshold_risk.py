from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new)


root = Path(__file__).resolve().parents[1]

source_path = root / "src/bayesian_phystwin/decisive_evidence.py"
source = source_path.read_text(encoding="utf-8")
source = replace_once(
    source,
    '''DECISIVE_EVIDENCE_SUMMARY_CONTRACT = (
    "bayesian-phystwin-decisive-evidence-summary-v1"
)
DEFAULT_TARGET_COVERAGES = tuple(float(value) / 10.0 for value in range(11))
''',
    '''DECISIVE_EVIDENCE_SUMMARY_CONTRACT = (
    "bayesian-phystwin-decisive-evidence-summary-v1"
)
THRESHOLD_RISK_COVERAGE_CONTRACT = (
    "bayesian-phystwin-threshold-risk-coverage-v1"
)
MATCHED_COUNT_RISK_COVERAGE_CONTRACT = (
    "bayesian-phystwin-matched-count-risk-coverage-v1"
)
DEFAULT_TARGET_COVERAGES = tuple(float(value) / 10.0 for value in range(11))
''',
    label="evidence contract constants",
)
source = replace_once(
    source,
    "def _risk_coverage_curves(\n",
    "def _matched_count_risk_coverage_curves(\n",
    label="matched-count function name",
)
source = replace_once(
    source,
    '''    return {
        "risk_score_order": "lower_is_safer",
        "selection_rule": (
            "accept the exact same count per method at each target coverage; sort "
            "by risk_score and break boundary ties deterministically by unit_id; "
            "rejected units use the common exact fallback"
        ),
        "target_coverages": list(coverages),
        "methods": curves,
    }


def _conditioned_summary(
''',
    '''    return {
        "contract": MATCHED_COUNT_RISK_COVERAGE_CONTRACT,
        "role": "secondary_equal_count_diagnostic",
        "risk_score_order": "lower_is_safer",
        "selection_rule": (
            "accept the exact same count per method at each target coverage; sort "
            "by risk_score and break boundary ties deterministically by unit_id; "
            "rejected units use the common exact fallback"
        ),
        "target_coverages": list(coverages),
        "methods": curves,
    }


def _threshold_risk_coverage_curves(
    records_by_method: Mapping[str, Sequence[EvidenceRecord]],
    *,
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    # Evaluate every distinct risk threshold without splitting tied scores.
    methods = tuple(sorted(records_by_method))
    curves: dict[str, list[dict[str, object]]] = {}
    for method in methods:
        records = tuple(
            sorted(
                records_by_method[method],
                key=lambda record: (
                    record.risk_score,
                    record.loss,
                    record.fallback_loss,
                    record.deployed_loss,
                    record.group_id,
                    record.horizon,
                ),
            )
        )
        losses = np.asarray([record.loss for record in records], dtype=float)
        fallbacks = np.asarray(
            [record.fallback_loss for record in records], dtype=float
        )
        risk_scores = np.asarray(
            [record.risk_score for record in records], dtype=float
        )
        thresholds: tuple[float | None, ...] = (None,) + tuple(
            float(value) for value in np.unique(risk_scores)
        )
        points: list[dict[str, object]] = []
        for threshold in thresholds:
            accepted = (
                np.zeros(len(records), dtype=bool)
                if threshold is None
                else risk_scores <= threshold
            )
            accepted_count = int(np.sum(accepted))
            coverage = accepted_count / len(records)
            deployed_losses = np.where(accepted, losses, fallbacks)
            harmful = accepted & (losses > fallbacks)
            points.append(
                {
                    "threshold": threshold,
                    "coverage": coverage,
                    "attained_coverage": coverage,
                    "accepted_count": accepted_count,
                    "fallback_count": len(records) - accepted_count,
                    "fallback_frequency": 1.0 - coverage,
                    "exact_fallback_verified": True,
                    "maximum_accepted_risk_score": (
                        None
                        if not accepted_count
                        else float(np.max(risk_scores[accepted]))
                    ),
                    "boundary_tie_count": (
                        0
                        if threshold is None
                        else int(np.sum(risk_scores == threshold))
                    ),
                    "boundary_tie_split": False,
                    "selective_mean_loss": (
                        None
                        if not accepted_count
                        else float(np.mean(losses[accepted]))
                    ),
                    "harmful_accepted_count": int(np.sum(harmful)),
                    "harmful_update_frequency_accepted": (
                        None
                        if not accepted_count
                        else float(np.sum(harmful) / accepted_count)
                    ),
                    "deployed": _loss_summary(
                        deployed_losses, fallbacks, quantiles=quantiles
                    ),
                }
            )
        curves[method] = points

    return {
        "contract": THRESHOLD_RISK_COVERAGE_CONTRACT,
        "role": "primary_threshold_native_view",
        "risk_score_order": "lower_is_safer",
        "selection_rule": (
            "include the zero-acceptance exact-fallback endpoint, then accept every "
            "unit with risk_score <= each distinct threshold; tied scores enter "
            "together and rejected units use the common exact fallback"
        ),
        "confirmatory_threshold_freeze": (
            "select thresholds using source or calibration data and freeze them "
            "before target outcomes are opened"
        ),
        "methods": curves,
    }


def _conditioned_summary(
''',
    label="risk-coverage return and threshold-native function",
)
source = replace_once(
    source,
    '''        first_method = next(iter(records_by_method))
        metric_summaries[metric] = {
            "unit_count": len(records_by_method[first_method]),
            "group_count": len(
                {record.group_id for record in records_by_method[first_method]}
            ),
            "methods": method_summaries,
            "matched_risk_coverage": _risk_coverage_curves(
                records_by_method,
                coverages=coverages,
                quantiles=quantiles,
                reference_method=resolved_reference,
            ),
        }
''',
    '''        first_method = next(iter(records_by_method))
        threshold_risk_coverage = _threshold_risk_coverage_curves(
            records_by_method,
            quantiles=quantiles,
        )
        matched_count_risk_coverage = _matched_count_risk_coverage_curves(
            records_by_method,
            coverages=coverages,
            quantiles=quantiles,
            reference_method=resolved_reference,
        )
        metric_summaries[metric] = {
            "unit_count": len(records_by_method[first_method]),
            "group_count": len(
                {record.group_id for record in records_by_method[first_method]}
            ),
            "methods": method_summaries,
            "threshold_risk_coverage": threshold_risk_coverage,
            "matched_count_risk_coverage": matched_count_risk_coverage,
            "matched_risk_coverage": {
                **matched_count_risk_coverage,
                "deprecated_alias_for": "matched_count_risk_coverage",
            },
        }
''',
    label="metric risk-coverage outputs",
)
source = replace_once(
    source,
    '''            "risk_score_order": "lower_is_safer",
            "matched_fallback": True,
''',
    '''            "risk_score_order": "lower_is_safer",
            "matched_fallback": True,
            "primary_risk_coverage_contract": THRESHOLD_RISK_COVERAGE_CONTRACT,
            "secondary_risk_coverage_contract": (
                MATCHED_COUNT_RISK_COVERAGE_CONTRACT
            ),
            "confirmatory_thresholds_must_be_source_or_calibration_frozen": True,
''',
    label="analysis configuration",
)
source_path.write_text(source, encoding="utf-8")

tests_path = root / "tests/test_decisive_evidence.py"
tests = tests_path.read_text(encoding="utf-8")
tests = replace_once(
    tests,
    '''    curve = metric["matched_risk_coverage"]["methods"]["bayesian"]
    assert curve[0]["deployed"]["mean_loss"] == 10.0
''',
    '''    matched = metric["matched_count_risk_coverage"]
    assert matched["contract"] == (
        "bayesian-phystwin-matched-count-risk-coverage-v1"
    )
    assert metric["matched_risk_coverage"]["deprecated_alias_for"] == (
        "matched_count_risk_coverage"
    )
    curve = matched["methods"]["bayesian"]
    assert curve[0]["deployed"]["mean_loss"] == 10.0
''',
    label="matched-count test assertions",
)
tests = replace_once(
    tests,
    '''


def test_summary_reports_horizon_calibration_reliability_and_rank() -> None:
''',
    '''


def test_threshold_risk_coverage_preserves_ties_and_identifiers() -> None:
    payload = _payload()
    for record in payload["records"]:
        if record["method"] == "bayesian" and record["unit_id"] in {"u2", "u4"}:
            record["risk_score"] = 0.3

    summary = analyze_decisive_evidence(
        payload,
        target_coverages=(0.0, 0.5, 1.0),
        regression_quantiles=(0.5, 0.95),
    )
    threshold_view = summary["metrics"]["track_error_m"][
        "threshold_risk_coverage"
    ]
    assert threshold_view["contract"] == (
        "bayesian-phystwin-threshold-risk-coverage-v1"
    )
    assert threshold_view["role"] == "primary_threshold_native_view"
    curve = threshold_view["methods"]["bayesian"]
    assert [point["threshold"] for point in curve] == [None, 0.1, 0.2, 0.3]
    assert [point["accepted_count"] for point in curve] == [0, 1, 2, 4]
    assert [point["coverage"] for point in curve] == [0.0, 0.25, 0.5, 1.0]
    assert curve[0]["deployed"]["mean_loss"] == 10.0
    assert curve[-1]["boundary_tie_count"] == 2
    assert curve[-1]["boundary_tie_split"] is False
    assert all(point["exact_fallback_verified"] for point in curve)

    renamed = json.loads(json.dumps(payload))
    renamed_ids = {
        "u1": "case-z",
        "u2": "case-y",
        "u3": "case-x",
        "u4": "case-w",
    }
    for record in renamed["records"]:
        record["unit_id"] = renamed_ids[record["unit_id"]]
    renamed["records"].reverse()
    renamed_summary = analyze_decisive_evidence(
        renamed,
        target_coverages=(0.0, 0.5, 1.0),
        regression_quantiles=(0.5, 0.95),
    )
    renamed_curve = renamed_summary["metrics"]["track_error_m"][
        "threshold_risk_coverage"
    ]["methods"]["bayesian"]
    assert renamed_curve == curve


def test_summary_reports_horizon_calibration_reliability_and_rank() -> None:
''',
    label="threshold-native tests",
)
tests = replace_once(
    tests,
    '''    assert written["analysis_configuration"]["matched_fallback"] is True
    assert written["reference_method"] == "last_residual"
''',
    '''    assert written["analysis_configuration"]["matched_fallback"] is True
    assert written["analysis_configuration"]["primary_risk_coverage_contract"] == (
        "bayesian-phystwin-threshold-risk-coverage-v1"
    )
    assert (
        written["analysis_configuration"][
            "confirmatory_thresholds_must_be_source_or_calibration_frozen"
        ]
        is True
    )
    assert written["reference_method"] == "last_residual"
''',
    label="CLI contract assertions",
)
tests_path.write_text(tests, encoding="utf-8")

docs_path = root / "docs/decisive_evidence_protocol.md"
docs = docs_path.read_text(encoding="utf-8")
docs = replace_once(
    docs,
    "- the target-coverage grid;",
    (
        "- confirmatory risk thresholds selected only on source/calibration data, "
        "plus the secondary target-coverage grid;"
    ),
    label="protocol freeze list",
)
docs = replace_once(
    docs,
    (
        "Every nonfallback arm must be evaluated under the same fallback contract. "
        "Rejected units return the exact `B0_physical_fallback` outcome; they are "
        "not dropped from the denominator. Risk–coverage comparisons accept the "
        "same number of units for every method at each target coverage."
    ),
    (
        "Every nonfallback arm must be evaluated under the same fallback contract. "
        "Rejected units return the exact `B0_physical_fallback` outcome; they are "
        "not dropped from the denominator. The primary threshold-native "
        "risk–coverage view evaluates `risk_score <= threshold` at every distinct "
        "score and never splits a tied score block. Confirmatory thresholds must "
        "be selected on source or calibration data and frozen before target "
        "outcomes are opened. A separately named matched-count view accepts the "
        "same number of units per method and remains a secondary equal-coverage "
        "diagnostic."
    ),
    label="primary and secondary risk-coverage policy",
)
docs = replace_once(
    docs,
    (
        "5. matched risk–coverage curves for every candidate and comparator;\n"
        "6. paired deployed performance against the registered reference method "
        "at equal coverage;"
    ),
    (
        "5. threshold-native risk–coverage curves for every candidate, including "
        "zero- and full-acceptance endpoints;\n"
        "6. separately labeled matched-count curves and paired deployed "
        "performance against the registered reference method at equal coverage;"
    ),
    label="required endpoints",
)
docs = replace_once(
    docs,
    (
        "`risk_score` is ordered so that lower values mean safer predictions. At "
        "each requested target coverage, the analyzer accepts the exact same count "
        "for every method. Ties at the acceptance boundary are broken "
        "deterministically by `unit_id` and explicitly marked in the output."
    ),
    (
        "`risk_score` is ordered so that lower values mean safer predictions. The "
        "primary `bayesian-phystwin-threshold-risk-coverage-v1` output accepts "
        "every unit satisfying `risk_score <= threshold` at each distinct "
        "threshold, includes exact zero- and full-acceptance endpoints, and admits "
        "tied scores only as a complete block. Its points are invariant to row "
        "order and `unit_id` naming. The secondary "
        "`bayesian-phystwin-matched-count-risk-coverage-v1` output accepts the "
        "exact same count for every method at each requested target coverage; "
        "boundary ties may be broken deterministically by `unit_id` and are "
        "explicitly marked. Paper tables must identify which contract produced "
        "every reported risk–coverage point."
    ),
    label="risk score output contracts",
)
docs_path.write_text(docs, encoding="utf-8")

pyproject_path = root / "pyproject.toml"
pyproject = pyproject_path.read_text(encoding="utf-8")
pyproject = replace_once(
    pyproject,
    'python_version = "3.10"\n',
    'python_version = "3.12"\n',
    label="mypy dependency-stub grammar target",
)
pyproject_path.write_text(pyproject, encoding="utf-8")

quality_docs_path = root / "docs/development_quality.md"
quality_docs = quality_docs_path.read_text(encoding="utf-8")
quality_docs = replace_once(
    quality_docs,
    (
        "- the mature public-interface subset passes `mypy --strict`.\n\n"
        "The changed-only debt set is explicit in the quality helper"
    ),
    (
        "- the mature public-interface subset passes `mypy --strict`.\n\n"
        "Mypy targets Python 3.12 syntax so it can parse the current NumPy type "
        "stubs, which use PEP 695 declarations. Runtime compatibility remains "
        "Python 3.10 and is enforced independently by Ruff's `py310` target and "
        "the Python 3.10 test jobs.\n\n"
        "The changed-only debt set is explicit in the quality helper"
    ),
    label="quality documentation",
)
quality_docs_path.write_text(quality_docs, encoding="utf-8")
