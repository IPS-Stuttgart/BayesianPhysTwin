"""Proper-score conversion for matched guarded predictive evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ._proper_scoring_contracts import (
    _array_identity,
    _boolean,
    _canonical_json_sha256,
    _closed_fields,
    _horizon,
    _integer,
    _mapping,
    _number,
    _pair_identity,
    _parse_forecast,
    _parse_pairs,
    _query_signature,
    _require,
    _sequence,
    _text,
    _vector,
)
from ._proper_scoring_rules import (
    _score_forecast,
    empirical_energy_score,
    gaussian_log_score,
)
from .decisive_evidence import (
    DECISIVE_EVIDENCE_INPUT_CONTRACT,
    parse_decisive_evidence,
)

PROPER_SCORING_INPUT_CONTRACT: Final = (
    "bayesian-phystwin-proper-scoring-input-v1"
)
PROPER_SCORING_VERSION: Final = 1
DEFAULT_MAXIMUM_RECORDS: Final = 100_000
DEFAULT_MAXIMUM_SAMPLES_PER_FORECAST: Final = 2_048
DEFAULT_MAXIMUM_DIMENSION: Final = 16_384
DEFAULT_MAXIMUM_VARIOGRAM_PAIRS: Final = 100_000
DEFAULT_MAXIMUM_ENERGY_PAIR_EVALUATIONS: Final = 16_777_216
DEFAULT_MAXIMUM_VARIOGRAM_EVALUATIONS: Final = 16_777_216
DEFAULT_MAXIMUM_ARRAY_ELEMENTS: Final = 16_777_216

_INPUT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract",
        "protocol_id",
        "statistical_unit",
        "claim_boundary",
        "reference_method",
        "score_configuration",
        "records",
    }
)
_RECORD_FIELDS: Final = frozenset(
    {
        "unit_id",
        "group_id",
        "query_id",
        "method",
        "horizon",
        "risk_score",
        "accepted",
        "reliability",
        "identifiable_rank",
        "observation",
        "prediction",
        "fallback_prediction",
        "variogram_pairs",
    }
)
_CONFIGURATION_FIELDS: Final = frozenset(
    {"variogram_power", "gaussian_log_score_offset"}
)


def _score_metric_name(query_id: str, family: str, power: float) -> str:
    if family != "variogram_score":
        return f"{query_id}/{family}"
    rendered_power = f"{power:.12g}".replace("-", "m").replace(".", "p")
    return f"{query_id}/variogram_score_power_{rendered_power}"


def build_proper_score_evidence(
    payload: Mapping[str, object],
    *,
    maximum_records: int = DEFAULT_MAXIMUM_RECORDS,
    maximum_samples_per_forecast: int = (
        DEFAULT_MAXIMUM_SAMPLES_PER_FORECAST
    ),
    maximum_dimension: int = DEFAULT_MAXIMUM_DIMENSION,
    maximum_variogram_pairs: int = DEFAULT_MAXIMUM_VARIOGRAM_PAIRS,
    maximum_energy_pair_evaluations: int = (
        DEFAULT_MAXIMUM_ENERGY_PAIR_EVALUATIONS
    ),
    maximum_variogram_evaluations: int = (
        DEFAULT_MAXIMUM_VARIOGRAM_EVALUATIONS
    ),
    maximum_array_elements: int = DEFAULT_MAXIMUM_ARRAY_ELEMENTS,
) -> dict[str, object]:
    """Convert predictive distributions into decisive-evidence loss records."""

    for value, name in (
        (maximum_records, "maximum_records"),
        (maximum_samples_per_forecast, "maximum_samples_per_forecast"),
        (maximum_dimension, "maximum_dimension"),
        (maximum_variogram_pairs, "maximum_variogram_pairs"),
        (maximum_energy_pair_evaluations, "maximum_energy_pair_evaluations"),
        (maximum_variogram_evaluations, "maximum_variogram_evaluations"),
        (maximum_array_elements, "maximum_array_elements"),
    ):
        _integer(value, name=name, minimum=1)

    _closed_fields(payload, _INPUT_FIELDS, name="input")
    schema_version = payload.get("schema_version")
    _require(
        type(schema_version) is int and schema_version == 1,
        "schema_version must be the integer 1",
    )
    _require(
        payload.get("contract") == PROPER_SCORING_INPUT_CONTRACT,
        f"contract must be {PROPER_SCORING_INPUT_CONTRACT!r}",
    )
    configuration = _mapping(
        payload.get("score_configuration", {}),
        name="score_configuration",
    )
    _closed_fields(
        configuration,
        _CONFIGURATION_FIELDS,
        name="score_configuration",
    )
    variogram_power = _number(
        configuration.get("variogram_power", 1.0),
        name="score_configuration.variogram_power",
        minimum=0.0,
        maximum=2.0,
    )
    _require(
        variogram_power > 0.0,
        "score_configuration.variogram_power must be positive",
    )
    gaussian_offset = _number(
        configuration.get("gaussian_log_score_offset", 0.0),
        name="score_configuration.gaussian_log_score_offset",
        minimum=0.0,
    )
    raw_records = _sequence(payload.get("records"), name="records")
    _require(bool(raw_records), "records must not be empty")
    _require(
        len(raw_records) <= maximum_records,
        "records exceeds the record-count budget",
    )

    output_records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    unit_bindings: dict[tuple[str, str], Mapping[str, object]] = {}
    query_signatures: dict[str, Mapping[str, object]] = {}
    query_methods: dict[str, frozenset[str]] = {}
    methods_by_query_unit: dict[tuple[str, str], set[str]] = {}
    query_metric_names: dict[str, set[str]] = {}
    metric_owners: dict[str, str] = {}

    for index, raw_record in enumerate(raw_records):
        name = f"records[{index}]"
        record = _mapping(raw_record, name=name)
        _closed_fields(record, _RECORD_FIELDS, name=name)
        unit_id = _text(record.get("unit_id"), name=f"{name}.unit_id")
        query_id = _text(record.get("query_id"), name=f"{name}.query_id")
        method = _text(record.get("method"), name=f"{name}.method")
        key = (query_id, unit_id, method)
        _require(key not in seen, f"duplicate query/unit/method record: {key}")
        seen.add(key)
        group_id = _text(
            record.get("group_id", unit_id),
            name=f"{name}.group_id",
        )
        horizon = _horizon(record.get("horizon"), name=f"{name}.horizon")
        risk_score = _number(
            record.get("risk_score"), name=f"{name}.risk_score"
        )
        accepted = _boolean(record.get("accepted"), name=f"{name}.accepted")
        reliability_value = record.get("reliability")
        reliability = (
            None
            if reliability_value is None
            else _number(
                reliability_value,
                name=f"{name}.reliability",
                minimum=0.0,
                maximum=1.0,
            )
        )
        rank_value = record.get("identifiable_rank")
        identifiable_rank = (
            None
            if rank_value is None
            else _integer(
                rank_value,
                name=f"{name}.identifiable_rank",
                minimum=0,
            )
        )
        observation = _vector(
            record.get("observation"),
            name=f"{name}.observation",
            maximum_dimension=maximum_dimension,
        )
        prediction = _parse_forecast(
            record.get("prediction"),
            name=f"{name}.prediction",
            dimension=len(observation),
            maximum_samples_per_forecast=maximum_samples_per_forecast,
            maximum_dimension=maximum_dimension,
            maximum_array_elements=maximum_array_elements,
        )
        fallback = _parse_forecast(
            record.get("fallback_prediction"),
            name=f"{name}.fallback_prediction",
            dimension=len(observation),
            maximum_samples_per_forecast=maximum_samples_per_forecast,
            maximum_dimension=maximum_dimension,
            maximum_array_elements=maximum_array_elements,
        )
        _require(
            prediction.families == fallback.families,
            f"{name} prediction and fallback score families differ",
        )
        pairs = _parse_pairs(
            record.get("variogram_pairs"),
            name=f"{name}.variogram_pairs",
            dimension=len(observation),
            maximum_variogram_pairs=maximum_variogram_pairs,
        )
        _require(
            not pairs or prediction.samples is not None,
            f"{name}.variogram_pairs requires empirical samples",
        )

        signature = _query_signature(prediction, pairs)
        previous_signature = query_signatures.setdefault(query_id, signature)
        _require(
            previous_signature == signature,
            f"query {query_id!r} changed score family or query registration",
        )
        unit_binding: Mapping[str, object] = {
            "group_id": group_id,
            "horizon": horizon,
            "observation": _array_identity(observation),
            "fallback_prediction": fallback.identity,
            "variogram_pairs": _pair_identity(pairs),
        }
        unit_key = (query_id, unit_id)
        previous_binding = unit_bindings.setdefault(unit_key, unit_binding)
        _require(
            previous_binding == unit_binding,
            f"{query_id}/{unit_id} changed truth, fallback, or registration",
        )
        methods_by_query_unit.setdefault(unit_key, set()).add(method)

        scores = _score_forecast(
            observation,
            prediction,
            pairs,
            variogram_power=variogram_power,
            gaussian_log_score_offset=gaussian_offset,
            maximum_energy_pair_evaluations=maximum_energy_pair_evaluations,
            maximum_variogram_evaluations=maximum_variogram_evaluations,
        )
        fallback_scores = _score_forecast(
            observation,
            fallback,
            pairs,
            variogram_power=variogram_power,
            gaussian_log_score_offset=gaussian_offset,
            maximum_energy_pair_evaluations=maximum_energy_pair_evaluations,
            maximum_variogram_evaluations=maximum_variogram_evaluations,
        )
        _require(
            scores.keys() == fallback_scores.keys(),
            f"{name} prediction and fallback score sets differ",
        )
        for family, (score, intervals, score_metadata) in scores.items():
            fallback_score, fallback_intervals, fallback_metadata = (
                fallback_scores[family]
            )
            metric = _score_metric_name(query_id, family, variogram_power)
            owner = metric_owners.setdefault(metric, query_id)
            _require(
                owner == query_id,
                f"query identifiers collide on generated metric {metric!r}",
            )
            query_metric_names.setdefault(query_id, set()).add(metric)
            deployed_intervals = intervals if accepted else fallback_intervals
            output_records.append(
                {
                    "unit_id": unit_id,
                    "group_id": group_id,
                    "metric": metric,
                    "method": method,
                    "loss": score,
                    "fallback_loss": fallback_score,
                    "risk_score": risk_score,
                    "accepted": accepted,
                    "deployed_loss": score if accepted else fallback_score,
                    "horizon": horizon,
                    "reliability": reliability,
                    "identifiable_rank": identifiable_rank,
                    "intervals": list(deployed_intervals),
                    "proper_score": {
                        "family": family,
                        "candidate": score_metadata,
                        "fallback": fallback_metadata,
                    },
                }
            )

    for query_id in sorted(query_signatures):
        units = sorted(
            key for key in methods_by_query_unit if key[0] == query_id
        )
        expected_methods = frozenset(methods_by_query_unit[units[0]])
        for unit_key in units[1:]:
            _require(
                frozenset(methods_by_query_unit[unit_key]) == expected_methods,
                f"query {query_id!r} must contain every method on every unit",
            )
        query_methods[query_id] = expected_methods

    reference_value = payload.get("reference_method")
    reference_method = (
        None
        if reference_value is None
        else _text(reference_value, name="reference_method")
    )
    if reference_method is not None:
        for query_id, methods in query_methods.items():
            _require(
                reference_method in methods,
                f"reference method is absent for query {query_id!r}",
            )

    evidence: dict[str, object] = {
        "schema_version": 1,
        "contract": DECISIVE_EVIDENCE_INPUT_CONTRACT,
        "protocol_id": _text(payload.get("protocol_id"), name="protocol_id"),
        "statistical_unit": _text(
            payload.get("statistical_unit"), name="statistical_unit"
        ),
        "claim_boundary": _text(
            payload.get("claim_boundary"), name="claim_boundary"
        ),
        "reference_method": reference_method,
        "records": output_records,
        "proper_scoring": {
            "schema_version": PROPER_SCORING_VERSION,
            "source_contract": PROPER_SCORING_INPUT_CONTRACT,
            "score_configuration": {
                "variogram_power": variogram_power,
                "gaussian_log_score_offset": gaussian_offset,
            },
            "resource_limits": {
                "maximum_records": maximum_records,
                "maximum_samples_per_forecast": (
                    maximum_samples_per_forecast
                ),
                "maximum_dimension": maximum_dimension,
                "maximum_variogram_pairs": maximum_variogram_pairs,
                "maximum_energy_pair_evaluations": (
                    maximum_energy_pair_evaluations
                ),
                "maximum_variogram_evaluations": (
                    maximum_variogram_evaluations
                ),
                "maximum_array_elements": maximum_array_elements,
            },
            "query_metrics": {
                query_id: sorted(metrics)
                for query_id, metrics in sorted(query_metric_names.items())
            },
            "gaussian_log_score_semantics": (
                "negative log predictive density plus one predeclared common "
                "additive offset; raw scores remain attached to every record"
            ),
            "fallback_semantics": (
                "rejected methods deploy scores and interval observations from "
                "the common registered fallback prediction"
            ),
        },
    }
    proper_metadata = evidence["proper_scoring"]
    if not isinstance(proper_metadata, dict):
        raise AssertionError("proper-scoring metadata changed type")
    proper_metadata["evidence_id"] = _canonical_json_sha256(evidence)
    parse_decisive_evidence(evidence)
    return evidence


__all__ = [
    "DEFAULT_MAXIMUM_ARRAY_ELEMENTS",
    "DEFAULT_MAXIMUM_DIMENSION",
    "DEFAULT_MAXIMUM_ENERGY_PAIR_EVALUATIONS",
    "DEFAULT_MAXIMUM_RECORDS",
    "DEFAULT_MAXIMUM_SAMPLES_PER_FORECAST",
    "DEFAULT_MAXIMUM_VARIOGRAM_EVALUATIONS",
    "DEFAULT_MAXIMUM_VARIOGRAM_PAIRS",
    "PROPER_SCORING_INPUT_CONTRACT",
    "PROPER_SCORING_VERSION",
    "build_proper_score_evidence",
    "empirical_energy_score",
    "gaussian_log_score",
]
