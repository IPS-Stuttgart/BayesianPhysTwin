"""Four-arm decomposition of guarded Bayesian predictive value."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from ._value_decomposition_bootstrap import _bootstrap_decomposition
from ._value_decomposition_core import (
    _canonical_json_sha256,
    _decomposition,
    _equal_group_vector,
    _integer,
    _interval_widths_match,
    _loss_vector,
    _number,
    _records_by_metric_method,
    _require,
    _text,
)
from .decisive_evidence import (
    DECISIVE_EVIDENCE_INPUT_CONTRACT,
    EvidenceRecord,
    parse_decisive_evidence,
)
from .decisive_evidence_bootstrap import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    GROUP_CLUSTERED_BOOTSTRAP_CONTRACT,
)

BAYESIAN_VALUE_DECOMPOSITION_CONTRACT: Final = (
    "bayesian-phystwin-bayesian-value-decomposition-v1"
)
BAYESIAN_VALUE_DECOMPOSITION_VERSION: Final = 1
DEFAULT_RAW_EQUALITY_TOLERANCE: Final = 1e-12


def _validate_metric_invariants(
    *,
    metric: str,
    deterministic: Mapping[str, EvidenceRecord],
    guarded: Mapping[str, EvidenceRecord],
    mean: Mapping[str, EvidenceRecord],
    full: Mapping[str, EvidenceRecord],
    tolerance: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    units = tuple(sorted(deterministic))
    for method_name, records in (
        ("guarded_reference", guarded),
        ("bayesian_mean", mean),
        ("full_belief", full),
    ):
        _require(
            set(records) == set(units),
            f"metric {metric!r}/{method_name} changed the unit set",
        )
    groups = tuple(
        sorted({record.group_id for record in deterministic.values()})
    )
    for unit in units:
        deterministic_record = deterministic[unit]
        guarded_record = guarded[unit]
        mean_record = mean[unit]
        _require(
            deterministic_record.accepted,
            f"{metric}/{unit} deterministic reference must always deploy",
        )
        _require(
            abs(deterministic_record.loss - guarded_record.loss)
            <= tolerance
            * (
                1.0
                + max(
                    abs(deterministic_record.loss),
                    abs(guarded_record.loss),
                )
            ),
            f"{metric}/{unit} guarded reference changed the deterministic mean",
        )
        _require(
            guarded_record.accepted == mean_record.accepted,
            f"{metric}/{unit} guarded and Bayesian-mean acceptance differ",
        )
        _require(
            abs(guarded_record.risk_score - mean_record.risk_score)
            <= tolerance
            * (
                1.0
                + max(
                    abs(guarded_record.risk_score),
                    abs(mean_record.risk_score),
                )
            ),
            f"{metric}/{unit} guarded and Bayesian-mean risk scores differ",
        )
        _require(
            guarded_record.reliability == mean_record.reliability,
            f"{metric}/{unit} guarded and Bayesian-mean reliability differ",
        )
        _require(
            guarded_record.identifiable_rank
            == mean_record.identifiable_rank,
            f"{metric}/{unit} guarded and Bayesian-mean rank differ",
        )
        _require(
            _interval_widths_match(
                guarded_record,
                mean_record,
                tolerance,
            ),
            f"{metric}/{unit} guarded and Bayesian-mean interval widths differ",
        )
    return units, groups


def _metric_report(
    *,
    metric: str,
    methods: Mapping[str, Mapping[str, EvidenceRecord]],
    arms: Mapping[str, str],
    tolerance: float,
) -> dict[str, object]:
    deterministic = methods[arms["deterministic_reference"]]
    guarded = methods[arms["guarded_reference"]]
    mean = methods[arms["bayesian_mean"]]
    full = methods[arms["full_belief"]]
    units, groups = _validate_metric_invariants(
        metric=metric,
        deterministic=deterministic,
        guarded=guarded,
        mean=mean,
        full=full,
        tolerance=tolerance,
    )
    arm_records = {
        arms["deterministic_reference"]: deterministic,
        arms["guarded_reference"]: guarded,
        arms["bayesian_mean"]: mean,
        arms["full_belief"]: full,
    }

    def vectors(*, deployed: bool, equal_group: bool) -> dict[str, np.ndarray]:
        return {
            method: (
                _equal_group_vector(records, groups, deployed=deployed)
                if equal_group
                else _loss_vector(records, units, deployed=deployed)
            )
            for method, records in arm_records.items()
        }

    def decompose(*, deployed: bool, equal_group: bool) -> dict[str, object]:
        return _decomposition(
            vectors(deployed=deployed, equal_group=equal_group),
            deterministic_reference=arms["deterministic_reference"],
            guarded_reference=arms["guarded_reference"],
            bayesian_mean=arms["bayesian_mean"],
            full_belief=arms["full_belief"],
            tolerance=tolerance,
        )

    return {
        "unit_count": len(units),
        "group_count": len(groups),
        "invariants": {
            "deterministic_reference_always_deployed": True,
            "guarded_reference_raw_mean_unchanged": True,
            "guarded_and_bayesian_mean_share_guard": True,
            "common_exact_fallback_validated": True,
        },
        "unit_weighted": {
            "raw": decompose(deployed=False, equal_group=False),
            "deployed": decompose(deployed=True, equal_group=False),
        },
        "equal_group_weighted": {
            "raw": decompose(deployed=False, equal_group=True),
            "deployed": decompose(deployed=True, equal_group=True),
        },
    }


def analyze_bayesian_value_decomposition(
    payload: Mapping[str, object],
    *,
    deterministic_reference: str,
    guarded_reference: str,
    bayesian_mean: str,
    full_belief: str,
    raw_equality_tolerance: float = DEFAULT_RAW_EQUALITY_TOLERANCE,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
) -> dict[str, object]:
    """Decompose predictive value across a frozen four-arm comparison."""

    arms = {
        "deterministic_reference": _text(
            deterministic_reference,
            name="deterministic_reference",
        ),
        "guarded_reference": _text(
            guarded_reference,
            name="guarded_reference",
        ),
        "bayesian_mean": _text(bayesian_mean, name="bayesian_mean"),
        "full_belief": _text(full_belief, name="full_belief"),
    }
    _require(
        len(set(arms.values())) == 4,
        "the four decomposition arms must be distinct",
    )
    tolerance = _number(
        raw_equality_tolerance,
        name="raw_equality_tolerance",
        minimum=0.0,
    )
    _integer(bootstrap_replicates, name="bootstrap_replicates", minimum=1)
    _integer(bootstrap_seed, name="bootstrap_seed", minimum=0)
    confidence = _number(
        bootstrap_confidence,
        name="bootstrap_confidence",
        minimum=0.0,
        maximum=1.0,
    )
    _require(
        confidence not in {0.0, 1.0},
        "bootstrap_confidence must lie strictly inside (0, 1)",
    )

    bundle = parse_decisive_evidence(payload)
    indexed = _records_by_metric_method(bundle.records)
    metric_reports: dict[str, object] = {}
    for metric, methods in sorted(indexed.items()):
        missing = sorted(set(arms.values()) - set(methods))
        _require(not missing, f"metric {metric!r} is missing arms {missing}")
        metric_reports[metric] = _metric_report(
            metric=metric,
            methods=methods,
            arms=arms,
            tolerance=tolerance,
        )

    bootstrap = _bootstrap_decomposition(
        payload,
        tuple(metric_reports),
        deterministic_reference=arms["deterministic_reference"],
        guarded_reference=arms["guarded_reference"],
        bayesian_mean=arms["bayesian_mean"],
        full_belief=arms["full_belief"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    for metric, values in metric_reports.items():
        if not isinstance(values, dict):
            raise AssertionError("metric report changed type")
        values["group_clustered_bootstrap"] = bootstrap[metric]

    report: dict[str, object] = {
        "schema_version": BAYESIAN_VALUE_DECOMPOSITION_VERSION,
        "contract": BAYESIAN_VALUE_DECOMPOSITION_CONTRACT,
        "source_contract": DECISIVE_EVIDENCE_INPUT_CONTRACT,
        "protocol_id": bundle.protocol_id,
        "statistical_unit": bundle.statistical_unit,
        "claim_boundary": bundle.claim_boundary,
        "arms": arms,
        "analysis_configuration": {
            "raw_equality_tolerance": tolerance,
            "bootstrap_contract": GROUP_CLUSTERED_BOOTSTRAP_CONTRACT,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_confidence": confidence,
            "bootstrap_resampling_unit": "group_id",
            "bootstrap_group_weighting": "equal",
            "improvement_sign": "positive_is_better",
        },
        "source_evidence_id": _canonical_json_sha256(payload),
        "metrics": metric_reports,
    }
    report["report_id"] = _canonical_json_sha256(report)
    return report


__all__ = [
    "BAYESIAN_VALUE_DECOMPOSITION_CONTRACT",
    "BAYESIAN_VALUE_DECOMPOSITION_VERSION",
    "DEFAULT_RAW_EQUALITY_TOLERANCE",
    "analyze_bayesian_value_decomposition",
]
