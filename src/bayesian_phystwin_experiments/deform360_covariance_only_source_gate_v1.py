"""Frozen Deform360 covariance-only source authorization gate.

This module evaluates only the ten already-open source object sessions. It
validates a 100-record, prefix-only prediction barrier and the subsequent
source-score table. It never loads or authorizes confirmation outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final

SCHEMA_VERSION: Final = 1
BATCH_SCHEMA: Final = "bayesian-phystwin.deform360-covariance-only-source-batch"
SCORES_SCHEMA: Final = "bayesian-phystwin.deform360-covariance-only-source-scores"
DECISION_SCHEMA: Final = "bayesian-phystwin.deform360-covariance-only-source-decision"
SOFTWARE_PROTOCOL_ID: Final = (
    "0f13d7a1f1610588ca9e7119f94814c99940fb31050419de16fa9cae06f683cc"
)
PAPER_PROTOCOL_ID: Final = (
    "fa16c105e6d535d1e229ccf086fd69d05b2be74592b5c4e3f6c5289b8915fee3"
)
REFERENCE_PREDICTOR_ID: Final = "last_residual"
COVARIANCE_DONOR_ID: Final = "independent_endpoint_v1"
COVARIANCE_SCALES: Final = (8.0, 16.0, 16.0)
OBSERVATION_STD_M: Final = 0.005

SOURCE_ROSTER: Final = (
    ("167-glove-gray-cloth", 0, "sheet"),
    ("198-kneepad-cloth", 2, "sheet"),
    ("026-sock-cloth", 7, "sheet"),
    ("031-cotton-cloth", 0, "sheet"),
    ("036-napkin-cloth", 9, "sheet"),
    ("153-cake", 5, "volumetric"),
    ("152-slime", 8, "volumetric"),
    ("186-monster", 6, "volumetric"),
    ("058-roll-napkin", 1, "volumetric"),
    ("193-frog", 7, "volumetric"),
)


def _plain_json(value: Any) -> Any:
    """Return a finite, JSON-compatible value without coercing booleans."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return json.loads(encoded)


def _content_id(payload: Mapping[str, Any], identity_field: str) -> str:
    document = dict(_plain_json(payload))
    document.pop(identity_field, None)
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _literal_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _literal_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be Boolean")
    return value


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_key(object_id: str, episode: int) -> tuple[str, int]:
    return (object_id, episode)


def _validate_identity(document: Mapping[str, Any], field: str) -> str:
    declared = _sha256(document.get(field), name=field)
    observed = _content_id(document, field)
    if declared != observed:
        raise ValueError(f"{field} does not match document content")
    return declared


def seal_prediction_batch(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and content-address the target-closed 100-record batch."""
    document = dict(_plain_json(payload))
    document.pop("batch_id", None)
    document["batch_id"] = _content_id(document, "batch_id")
    validate_prediction_batch(document)
    return document


def validate_prediction_batch(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact prediction-first source barrier."""
    document = dict(_plain_json(_mapping(payload, name="prediction batch")))
    if (
        document.get("schema") != BATCH_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("prediction batch schema changed")
    if document.get("software_protocol_id") != SOFTWARE_PROTOCOL_ID:
        raise ValueError("software protocol identity changed")
    if document.get("paper_protocol_id") != PAPER_PROTOCOL_ID:
        raise ValueError("paper protocol identity changed")
    _validate_identity(document, "batch_id")

    candidate = _mapping(document.get("candidate"), name="candidate")
    if candidate.get("reference_predictor_id") != REFERENCE_PREDICTOR_ID:
        raise ValueError("reference predictor changed")
    if candidate.get("covariance_donor_id") != COVARIANCE_DONOR_ID:
        raise ValueError("covariance donor changed")
    scales = tuple(
        _finite(value, name="candidate.early_middle_late_covariance_scales")
        for value in _sequence(
            candidate.get("early_middle_late_covariance_scales"),
            name="candidate.early_middle_late_covariance_scales",
        )
    )
    if scales != COVARIANCE_SCALES:
        raise ValueError("covariance scales changed")
    observation_std_m = _finite(
        candidate.get("observation_std_m"),
        name="candidate.observation_std_m",
    )
    if observation_std_m != OBSERVATION_STD_M:
        raise ValueError("observation standard deviation changed")
    if _boolean(
        candidate.get("point_prediction_change_allowed"),
        name="candidate.point_prediction_change_allowed",
    ):
        raise ValueError("point prediction changes are forbidden")

    boundary = _mapping(
        document.get("information_boundary"),
        name="information_boundary",
    )
    required_boundary = {
        "sealed_before_source_suffix": True,
        "source_suffix_used": False,
        "confirmation_payloads_opened": False,
        "confirmation_predictions_run": False,
        "confirmation_outcomes_used": False,
        "replacement_used": False,
        "target_informed_selection_used": False,
    }
    for field, expected in required_boundary.items():
        observed = _boolean(
            boundary.get(field),
            name=f"information_boundary.{field}",
        )
        if observed is not expected:
            raise ValueError(f"information boundary changed: {field}")

    units = _sequence(document.get("source_units"), name="source_units")
    parsed_units: list[tuple[str, int, str]] = []
    for index, raw in enumerate(units):
        unit = _mapping(raw, name=f"source_units[{index}]")
        parsed_units.append(
            (
                _literal_string(unit.get("object_id"), name="object_id"),
                _integer(unit.get("episode"), name="episode"),
                _literal_string(unit.get("stratum"), name="stratum"),
            )
        )
    if tuple(parsed_units) != SOURCE_ROSTER:
        raise ValueError("source roster or ordering changed")

    records = _sequence(document.get("records"), name="records")
    if len(records) != 100:
        raise ValueError("exactly 100 prefix-only prediction records are required")
    prediction_ids: set[str] = set()
    count_by_unit: Counter[tuple[str, int]] = Counter()
    folds_by_unit: dict[tuple[str, int], set[int]] = defaultdict(set)
    record_by_id: dict[str, Mapping[str, Any]] = {}
    roster_by_key = {
        _unit_key(obj, ep): (index, stratum)
        for index, (obj, ep, stratum) in enumerate(SOURCE_ROSTER)
    }
    for index, raw in enumerate(records):
        record = _mapping(raw, name=f"records[{index}]")
        prediction_id = _sha256(record.get("prediction_id"), name="prediction_id")
        if prediction_id in prediction_ids:
            raise ValueError("duplicate prediction_id")
        prediction_ids.add(prediction_id)
        object_id = _literal_string(record.get("object_id"), name="object_id")
        episode = _integer(record.get("episode"), name="episode")
        key = _unit_key(object_id, episode)
        if key not in roster_by_key:
            raise ValueError("prediction record uses an unknown source unit")
        fold = _integer(record.get("outer_fold_index"), name="outer_fold_index")
        if fold < 0 or fold >= len(SOURCE_ROSTER):
            raise ValueError("outer_fold_index is outside the frozen range")
        count_by_unit[key] += 1
        folds_by_unit[key].add(fold)
        if record.get("stratum") != roster_by_key[key][1]:
            raise ValueError("prediction stratum changed")
        mean_id = _sha256(record.get("mean_sha256"), name="mean_sha256")
        reference_id = _sha256(
            record.get("reference_mean_sha256"), name="reference_mean_sha256"
        )
        if mean_id != reference_id:
            raise ValueError("covariance-only prediction changed the registered mean")
        disposition = record.get("disposition")
        if disposition not in {"candidate", "exact_fallback"}:
            raise ValueError("invalid prediction disposition")
        exact_fallback = _boolean(record.get("exact_fallback"), name="exact_fallback")
        if exact_fallback is not (disposition == "exact_fallback"):
            raise ValueError("fallback disposition is inconsistent")
        if _boolean(record.get("source_suffix_used"), name="source_suffix_used"):
            raise ValueError("prediction used the source suffix")
        if _boolean(
            record.get("confirmation_outcomes_used"),
            name="confirmation_outcomes_used",
        ):
            raise ValueError("prediction used confirmation outcomes")
        record_by_id[prediction_id] = record

    expected_folds = set(range(len(SOURCE_ROSTER)))
    for object_id, episode, _ in SOURCE_ROSTER:
        key = _unit_key(object_id, episode)
        if count_by_unit[key] != 10 or folds_by_unit[key] != expected_folds:
            raise ValueError(
                "prediction records do not form the complete 10x10 barrier"
            )

    selected = _mapping(
        document.get("scoring_prediction_by_source_unit"),
        name="scoring_prediction_by_source_unit",
    )
    expected_selected_keys = {f"{obj}#{episode}" for obj, episode, _ in SOURCE_ROSTER}
    if set(selected) != expected_selected_keys:
        raise ValueError("scoring prediction roster changed")
    for unit_index, (object_id, episode, _) in enumerate(SOURCE_ROSTER):
        key = f"{object_id}#{episode}"
        prediction_id = _sha256(
            selected[key],
            name=f"scoring_prediction_by_source_unit.{key}",
        )
        record = record_by_id.get(prediction_id)
        if record is None:
            raise ValueError("selected scoring prediction is absent from the batch")
        if record.get("object_id") != object_id or record.get("episode") != episode:
            raise ValueError(
                "selected scoring prediction belongs to another source unit"
            )
        if record.get("outer_fold_index") != unit_index:
            raise ValueError(
                "selected scoring prediction is not the frozen diagonal record"
            )
    return document


def seal_source_scores(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address the post-suffix source score table before evaluation."""
    document = dict(_plain_json(payload))
    document.pop("score_set_id", None)
    document["score_set_id"] = _content_id(document, "score_set_id")
    return document


def _validated_score_rows(
    payload: Mapping[str, Any], batch: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = dict(_plain_json(_mapping(payload, name="source scores")))
    if (
        document.get("schema") != SCORES_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("source-score schema changed")
    if document.get("batch_id") != batch["batch_id"]:
        raise ValueError("source scores bind another prediction batch")
    _validate_identity(document, "score_set_id")
    boundary = _mapping(
        document.get("information_boundary"),
        name="information_boundary",
    )
    required_boundary = {
        "source_suffix_opened": True,
        "confirmation_payloads_opened": False,
        "confirmation_predictions_run": False,
        "confirmation_outcomes_used": False,
        "candidate_retuned": False,
        "replacement_used": False,
    }
    for field, expected in required_boundary.items():
        observed = _boolean(
            boundary.get(field),
            name=f"information_boundary.{field}",
        )
        if observed is not expected:
            raise ValueError(f"source-score information boundary changed: {field}")

    selected = _mapping(
        batch["scoring_prediction_by_source_unit"],
        name="scoring_prediction_by_source_unit",
    )
    records = _sequence(batch["records"], name="records")
    record_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, name=f"records[{index}]")
        prediction_id = _sha256(record.get("prediction_id"), name="prediction_id")
        record_by_id[prediction_id] = record

    rows = _sequence(document.get("rows"), name="rows")
    if len(rows) != len(SOURCE_ROSTER):
        raise ValueError("source score table must contain all ten source units")
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(rows):
        row = dict(_plain_json(_mapping(raw, name=f"rows[{index}]")))
        object_id = _literal_string(row.get("object_id"), name="object_id")
        episode = _integer(row.get("episode"), name="episode")
        key = _unit_key(object_id, episode)
        if key in seen:
            raise ValueError("duplicate source score row")
        seen.add(key)
        try:
            frozen_index = next(
                idx
                for idx, (obj, ep, _) in enumerate(SOURCE_ROSTER)
                if obj == object_id and ep == episode
            )
        except StopIteration as error:
            raise ValueError("source score row uses an unknown unit") from error
        frozen_stratum = SOURCE_ROSTER[frozen_index][2]
        if row.get("stratum") != frozen_stratum:
            raise ValueError("source score stratum changed")
        selected_id = _sha256(
            selected[f"{object_id}#{episode}"],
            name=f"scoring_prediction_by_source_unit.{object_id}#{episode}",
        )
        prediction_id = _sha256(row.get("prediction_id"), name="prediction_id")
        if prediction_id != selected_id:
            raise ValueError("source score row binds another prediction")
        selected_record = record_by_id[selected_id]
        disposition = row.get("disposition")
        if disposition not in {"candidate", "exact_fallback", "technical_failure"}:
            raise ValueError("invalid score-row disposition")
        if (
            disposition != "technical_failure"
            and disposition != selected_record.get("disposition")
        ):
            raise ValueError(
                "score-row disposition does not match the selected prediction"
            )
        row["point_mean_identity"] = _boolean(
            row.get("point_mean_identity"), name="point_mean_identity"
        )
        row["point_metric_difference_m"] = _finite(
            row.get("point_metric_difference_m"), name="point_metric_difference_m"
        )
        row["supported_or_exact_fallback"] = _boolean(
            row.get("supported_or_exact_fallback"),
            name="supported_or_exact_fallback",
        )
        row["exact_fallback"] = _boolean(
            row.get("exact_fallback"),
            name="exact_fallback",
        )
        if disposition == "technical_failure":
            if row["supported_or_exact_fallback"] or row["exact_fallback"]:
                raise ValueError(
                    "technical failures cannot claim support or exact fallback"
                )
            if (
                row.get("candidate_nll") is not None
                or row.get("reference_nll") is not None
            ):
                raise ValueError("technical failures must not carry invented scores")
            _literal_string(
                row.get("technical_failure_reason"),
                name="technical_failure_reason",
            )
        else:
            if not row["supported_or_exact_fallback"]:
                raise ValueError(
                    "nontechnical source rows must be supported or exact fallback"
                )
            row["candidate_nll"] = _finite(
                row.get("candidate_nll"),
                name="candidate_nll",
            )
            row["reference_nll"] = _finite(
                row.get("reference_nll"),
                name="reference_nll",
            )
            if row["exact_fallback"] is not (disposition == "exact_fallback"):
                raise ValueError("score-row fallback disposition is inconsistent")
            if (
                disposition == "exact_fallback"
                and row["candidate_nll"] != row["reference_nll"]
            ):
                raise ValueError("exact fallback must reproduce the reference score")
        parsed.append(row)
    expected = {_unit_key(obj, ep) for obj, ep, _ in SOURCE_ROSTER}
    if seen != expected:
        raise ValueError("source score roster is incomplete")
    return document, parsed


def evaluate_source_gate(
    prediction_batch: Mapping[str, Any], source_scores: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen source rule without opening confirmation data."""
    batch = validate_prediction_batch(prediction_batch)
    scores, rows = _validated_score_rows(source_scores, batch)
    technical = [row for row in rows if row["disposition"] == "technical_failure"]
    reasons: list[str] = []
    if technical:
        reasons.append("retained-technical-failure")

    support_by_stratum: Counter[str] = Counter()
    support_total = 0
    all_point_identity = True
    all_point_metric_identity = True
    all_fallback_exact = True
    differences: list[float] = []
    stratum_differences: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        stratum = str(row["stratum"])
        if row["supported_or_exact_fallback"]:
            support_total += 1
            support_by_stratum[stratum] += 1
        all_point_identity = all_point_identity and bool(row["point_mean_identity"])
        all_point_metric_identity = all_point_metric_identity and (
            float(row["point_metric_difference_m"]) == 0.0
        )
        if row["disposition"] == "exact_fallback":
            all_fallback_exact = all_fallback_exact and bool(row["exact_fallback"])
        if row["disposition"] != "technical_failure":
            difference = float(row["candidate_nll"]) - float(row["reference_nll"])
            differences.append(difference)
            stratum_differences[stratum].append(difference)

    if support_total < 8:
        reasons.append("fewer-than-8-of-10-supported-or-fallback")
    for stratum in ("sheet", "volumetric"):
        if support_by_stratum[stratum] < 4:
            reasons.append(f"fewer-than-4-of-5-{stratum}-supported-or-fallback")
    if not all_point_identity:
        reasons.append("point-mean-identity-failed")
    if not all_point_metric_identity:
        reasons.append("point-metric-identity-failed")
    if not all_fallback_exact:
        reasons.append("exact-fallback-failed")

    mean_difference = None if technical else sum(differences) / len(differences)
    stratum_means: dict[str, float | None] = {}
    for stratum in ("sheet", "volumetric"):
        values = stratum_differences[stratum]
        stratum_means[stratum] = None if technical else sum(values) / len(values)
    if not technical:
        if mean_difference is None or mean_difference >= 0.0:
            reasons.append("overall-mean-nll-difference-not-negative")
        for stratum in ("sheet", "volumetric"):
            value = stratum_means[stratum]
            if value is None or value > 0.0:
                reasons.append(f"{stratum}-mean-nll-difference-positive")

    if technical:
        status = "source-technical-negative"
    elif reasons:
        status = "source-negative"
    else:
        status = "source-positive"
    authorized = status == "source-positive"
    decision: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "batch_id": batch["batch_id"],
        "score_set_id": scores["score_set_id"],
        "status": status,
        "reasons": sorted(reasons),
        "source_unit_count": len(SOURCE_ROSTER),
        "prediction_record_count": len(batch["records"]),
        "supported_or_exact_fallback_count": support_total,
        "supported_or_exact_fallback_by_stratum": {
            "sheet": support_by_stratum["sheet"],
            "volumetric": support_by_stratum["volumetric"],
        },
        "technical_failure_count": len(technical),
        "mean_candidate_minus_reference_nll": mean_difference,
        "stratum_mean_candidate_minus_reference_nll": stratum_means,
        "all_point_means_identical": all_point_identity,
        "all_point_metrics_identical": all_point_metric_identity,
        "all_fallbacks_exact": all_fallback_exact,
        "confirmation_prediction_authorized": authorized,
        "confirmation_payload_opening_authorized": False,
        "confirmation_outcome_opening_authorized": False,
        "claim_authorized": False,
        "target_side_retuning_allowed": False,
    }
    decision["decision_id"] = _content_id(decision, "decision_id")
    return decision


__all__ = [
    "BATCH_SCHEMA",
    "COVARIANCE_DONOR_ID",
    "COVARIANCE_SCALES",
    "DECISION_SCHEMA",
    "OBSERVATION_STD_M",
    "PAPER_PROTOCOL_ID",
    "REFERENCE_PREDICTOR_ID",
    "SCHEMA_VERSION",
    "SCORES_SCHEMA",
    "SOFTWARE_PROTOCOL_ID",
    "SOURCE_ROSTER",
    "evaluate_source_gate",
    "seal_prediction_batch",
    "seal_source_scores",
    "validate_prediction_batch",
]
