"""Corrected nested source contract for the Deform360 v6 study.

The original v6 adapter accepted guard decisions and calibrated interval summaries
inside outcome-free prediction seals.  That cannot implement the frozen policy:
guard thresholds, covariance scale, and grouped residual scores depend on source
outcomes and must be fitted inside each outer fold.  This additive v6.1 contract
therefore seals 100 raw nested predictions and only outcome sufficient statistics.
Every deployment decision is then reconstructed from the other nine source units.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import genuine_boolean, genuine_integer, plain_json
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_fresh_object_session_source_v6 import (
    AMENDMENT_ID,
    B0,
    B1,
    CHALLENGER_VARIANTS,
    D1_NATIVE,
    POLICY_ID,
    VARIANT_COMPLEXITY,
    VARIANT_COVARIANCE,
    VARIANT_IDS,
    VARIANT_POLICY_CANDIDATE,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
)

NESTED_REPAIR_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-nested-source-contract-repair"
)
NESTED_REPAIR_ID: Final = (
    "4792bcf1c69cf7885edd0c4e1a2ffbb1d7cd5af9032839841057ba329d5f1196"
)
RAW_PREDICTION_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-raw-nested-prediction"
)
RAW_BATCH_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-raw-nested-batch"
)
RAW_OUTCOME_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-raw-nested-outcome"
)
NESTED_EVIDENCE_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-nested-evidence"
)
NESTED_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-nested-result"
)
SCHEMA_VERSION: Final = 1
QUERY_DIMENSION: Final = 3
MINIMUM_SCALE: Final = 1e-6
MAXIMUM_SCALE: Final = 1e6
NOMINAL_COVERAGE: Final = 0.9
SOURCE_SELECTION_ARTIFACT_ID: Final = (
    "dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82"
)
UPSTREAM_PREDICTION_BATCH_ID: Final = (
    "5b1cdf3f047b52665650dcbf56d8ec205ced8788e2cdd0e528793a9ece9387f0"
)
UPSTREAM_REVISION: Final = "913909596b71ac6ad717835ce7a87ae01e42c5ab"

_PREDICTION_BOUNDARY = {
    "source_suffix_opened": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "future_object_observations_used_for_prediction": False,
    "human_selection_used": False,
    "replacement_allowed": False,
    "existing_source_provider_products_reused": True,
    "prob4d_used": True,
    "new_prob4d_inference_run": False,
    "new_motioncrafter_inference_run": False,
}
_OUTCOME_BOUNDARY = {
    **_PREDICTION_BOUNDARY,
    "source_suffix_opened": True,
}
_EVIDENCE_BOUNDARY = {
    "all_nested_predictions_sealed_before_source_suffix_scoring": True,
    "guard_or_calibration_fit_before_source_suffix_scoring": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "human_selection_used": False,
    "replacement_allowed": False,
}

_RAW_VARIANT_FIELDS = frozenset(
    {
        "available",
        "covariance_artifact_id",
        "fit_artifact_id",
        "fit_object_ids",
        "prediction_artifact_id",
        "risk_score",
        "unavailable_reason",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "candidate_revision",
        "covariance_amendment_id",
        "episode_id",
        "information_boundary",
        "nested_repair_id",
        "object_id",
        "outer_held_out_object_id",
        "policy_id",
        "prediction_record_id",
        "record_role",
        "schema",
        "schema_version",
        "source_artifacts",
        "source_selection_artifact_sha256",
        "stratum",
        "upstream_prediction_batch_id",
        "upstream_revision",
        "variants",
    }
)
_BATCH_FIELDS = frozenset(
    {
        "candidate_revision",
        "covariance_amendment_id",
        "information_boundary",
        "nested_repair_id",
        "policy_id",
        "prediction_batch_id",
        "record_count",
        "records",
        "schema",
        "schema_version",
        "source_selection_artifact_sha256",
        "upstream_prediction_batch_id",
        "upstream_revision",
    }
)
_OUTCOME_VARIANT_FIELDS = frozenset(
    {
        "available",
        "maximum_raw_mahalanobis_norm",
        "mean_log_determinant",
        "mean_raw_mahalanobis_squared",
        "mean_raw_radius",
        "point_loss",
        "prediction_artifact_id",
        "query_count",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "information_boundary",
        "nested_repair_id",
        "outcome_id",
        "prediction_batch_id",
        "prediction_record_id",
        "schema",
        "schema_version",
        "scoring_artifacts",
        "variants",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "information_boundary",
        "nested_repair_id",
        "prediction_batch_id",
        "record_count",
        "records",
        "schema",
        "schema_version",
    }
)
_EVIDENCE_RECORD_FIELDS = frozenset({"outcome", "prediction"})


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _identifier(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result != result.strip() or "\x00" in result:
        raise ValueError(f"{name} must be a canonical string")
    return result


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _content_identity(value: Mapping[str, Any], *, field: str, name: str) -> None:
    declared = sha256_digest(value.get(field), name=field)
    body = {key: item for key, item in value.items() if key != field}
    if declared != content_id(body):
        raise ValueError(f"{name} content identity changed")


def load_deform360_v6_nested_source_repair(path: str | Path) -> Mapping[str, Any]:
    """Load the source-independent correction frozen before v6 suffix scoring."""

    repair = load_strict_json_object(path, label="Deform360 v6 nested repair")
    if (
        repair.get("schema") != NESTED_REPAIR_SCHEMA
        or repair.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("v6 nested repair schema changed")
    _content_identity(repair, field="amendment_id", name="v6 nested repair")
    if repair.get("amendment_id") != NESTED_REPAIR_ID:
        raise ValueError("v6 nested repair identity changed")
    base = _mapping(repair.get("base_policy"), name="base_policy")
    covariance = _mapping(
        repair.get("base_covariance_amendment"), name="base_covariance_amendment"
    )
    if (
        base.get("policy_id") != POLICY_ID
        or covariance.get("amendment_id") != AMENDMENT_ID
    ):
        raise ValueError("v6 nested repair binds another policy")
    correction = _mapping(repair.get("correction"), name="correction")
    required_true = (
        "candidate_revision_separate_from_upstream_batch_revision",
        "covariance_method_selected_inside_each_outer_fold",
        "exact_source_selection_bound_in_each_raw_record",
        "guard_threshold_fitted_inside_each_outer_fold",
        "legacy_public_evaluator_retired",
        "legacy_precomputed_acceptance_not_admissible",
        "legacy_precomputed_interval_summary_not_admissible",
        "legacy_prediction_seals_are_nonprogressing_diagnostic_only",
        "legacy_split_conformal_label_superseded",
        "later_covariance_amendment_tie_break_applied",
        "raw_prediction_and_risk_score_sealed_before_source_suffix_scoring",
    )
    if any(correction.get(key) is not True for key in required_true):
        raise ValueError("v6 nested repair correction changed")
    calibration = _mapping(
        repair.get("covariance_calibration"), name="covariance_calibration"
    )
    required_calibration = {
        "calibration_method": (
            "nested-source-cross-fitted-grouped-residual-calibration"
        ),
        "finite_sample_split_conformal_coverage_claimed": False,
        "full_source_target_calibration_uses_ten_leave-one-object-out_residuals": (
            True
        ),
        "grouped_residual_quantile_rank_rule": ("ceil((n+1)*nominal-coverage)"),
        "outer_held_out_unit_excluded_from_scale_and_quantile_fit": True,
        "scale_and_quantile_share_outer_training_residuals": True,
        "supersedes_base_covariance_amendment_requirement": (
            "source_only_variance_scale_and_split_conformal_required"
        ),
        "supersedes_base_policy_interval_calibration_label": (
            "group-clustered-split-conformal"
        ),
        "variance_scale_and_grouped_residual_quantile_fitted_on_raw_candidate_outcomes_regardless_of_guard_acceptance": (
            True
        ),
    }
    if any(
        calibration.get(key) != value for key, value in required_calibration.items()
    ):
        raise ValueError("v6 nested repair covariance calibration changed")
    nested = _mapping(repair.get("nested_selection"), name="nested_selection")
    if nested.get("variant_tie_break") != [
        "lowest-training-deployed-gaussian-nll",
        "lowest-training-deployed-point-loss",
        "lowest-training-deployed-full-interval-width",
        "lower-registered-complexity",
    ]:
        raise ValueError("v6 nested repair selection order changed")
    boundary = _mapping(repair.get("information_boundary"), name="boundary")
    if any(
        boundary.get(key) is not False
        for key in (
            "source_outcomes_used_to_design_this_repair",
            "source_suffix_opened_for_v6_candidates",
            "v6_target_payloads_used",
            "v6_target_outcomes_used",
        )
    ):
        raise ValueError("v6 nested repair crossed its information boundary")
    feeder = _mapping(repair.get("observation_feeder"), name="observation_feeder")
    if feeder != {
        "existing_decoded_uniform_prob4d_artifacts_used": True,
        "existing_source_provider_products_reused": True,
        "new_motioncrafter_inference_run": False,
        "new_prob4d_inference_run": False,
        "upstream_execution_receipt_id": (
            "a408e44eaecf9e63311a2f1a6f511f130e586031e8a0e8e795d58fa5696e3026"
        ),
        "upstream_prediction_batch_id": UPSTREAM_PREDICTION_BATCH_ID,
        "upstream_prediction_receipt_id": (
            "04a5a8b71603b66850e35122405bc24c5de1e766c14cc2b58974f1ea97fb49ef"
        ),
        "upstream_receipt_declares_prob4d_used": True,
        "upstream_revision": UPSTREAM_REVISION,
    }:
        raise ValueError("v6 nested repair observation feeder changed")
    return repair


def _cohort(value: Mapping[str, tuple[int, str]]) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for raw_id, raw_identity in value.items():
        object_id = _identifier(raw_id, name="object_id")
        if not isinstance(raw_identity, tuple) or len(raw_identity) != 2:
            raise ValueError("cohort identity must be an (episode, stratum) tuple")
        episode = genuine_integer(raw_identity[0], name="episode_id", minimum=0)
        stratum = _identifier(raw_identity[1], name="stratum")
        if stratum not in {"sheet", "volumetric"}:
            raise ValueError("source stratum changed")
        result[object_id] = (episode, stratum)
    if len(result) != 10 or Counter(item[1] for item in result.values()) != {
        "sheet": 5,
        "volumetric": 5,
    }:
        raise ValueError("nested source cohort must contain five units per stratum")
    return dict(sorted(result.items()))


def _fit_roster(
    source_ids: set[str], *, outer_id: str, object_id: str, challenger: bool
) -> tuple[str, ...]:
    if not challenger:
        return ()
    return tuple(sorted(source_ids - {outer_id, object_id}))


def _raw_variant(
    value: object,
    *,
    variant_id: str,
    source_ids: set[str],
    outer_id: str,
    object_id: str,
) -> dict[str, Any]:
    row = _mapping(value, name=variant_id)
    require_exact_fields(row, expected=_RAW_VARIANT_FIELDS, name=variant_id)
    challenger = variant_id in CHALLENGER_VARIANTS
    expected_fit = _fit_roster(
        source_ids,
        outer_id=outer_id,
        object_id=object_id,
        challenger=challenger,
    )
    fit_ids = canonical_sorted_strings(
        cast(Sequence[str], row.get("fit_object_ids")),
        name="fit_object_ids",
        allow_empty=True,
    )
    if fit_ids != expected_fit:
        raise ValueError("v6.1 candidate fit roster changed")
    available = genuine_boolean(row.get("available"), name="available")
    if variant_id in {B0, B1, D1_NATIVE} and not available:
        raise ValueError(f"{variant_id} must be available")
    reason = row.get("unavailable_reason")
    artifact_fields = (
        "prediction_artifact_id",
        "fit_artifact_id",
        "covariance_artifact_id",
    )
    if available:
        artifacts = {
            key: sha256_digest(row.get(key), name=f"{variant_id}.{key}")
            for key in artifact_fields
        }
        risk = (
            _finite(row.get("risk_score"), name="risk_score", minimum=0.0)
            if challenger
            else None
        )
        if not challenger and row.get("risk_score") is not None:
            raise ValueError("a source baseline must not carry a risk score")
        if reason is not None:
            raise ValueError("an available source variant has an unavailable reason")
        unavailable_reason = None
    else:
        if any(row.get(key) is not None for key in artifact_fields):
            raise ValueError("an unavailable source variant has an artifact")
        if row.get("risk_score") is not None:
            raise ValueError("an unavailable source variant has a risk score")
        artifacts = {key: None for key in artifact_fields}
        risk = None
        unavailable_reason = _identifier(reason, name="unavailable_reason")
    return {
        "available": available,
        **artifacts,
        "fit_object_ids": list(fit_ids),
        "risk_score": risk,
        "unavailable_reason": unavailable_reason,
    }


def _raw_variants(
    value: object, *, source_ids: set[str], outer_id: str, object_id: str
) -> dict[str, dict[str, Any]]:
    rows = _mapping(value, name="variants")
    require_exact_fields(rows, expected=frozenset(VARIANT_IDS), name="variants")
    result = {
        variant_id: _raw_variant(
            rows[variant_id],
            variant_id=variant_id,
            source_ids=source_ids,
            outer_id=outer_id,
            object_id=object_id,
        )
        for variant_id in VARIANT_IDS
    }
    available_vt1 = [
        result[item]
        for item in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH)
        if result[item]["available"]
    ]
    if available_vt1:
        reference = available_vt1[0]
        for row in available_vt1[1:]:
            for field in (
                "prediction_artifact_id",
                "fit_artifact_id",
                "fit_object_ids",
                "risk_score",
            ):
                if row[field] != reference[field]:
                    raise ValueError(
                        "VT1 covariance variants must share one raw mean and fit"
                    )
    return result


def build_deform360_v6_raw_nested_prediction(
    *,
    cohort: Mapping[str, tuple[int, str]],
    upstream_prediction_batch_id: str,
    upstream_revision: str,
    candidate_revision: str,
    outer_held_out_object_id: str,
    object_id: str,
    variants: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build one of the 100 outcome-free nested raw prediction records."""

    registered = _cohort(cohort)
    outer_id = _identifier(outer_held_out_object_id, name="outer_held_out_object_id")
    unit_id = _identifier(object_id, name="object_id")
    if outer_id not in registered or unit_id not in registered:
        raise ValueError("raw nested prediction uses an unregistered source unit")
    upstream = exact_revision(upstream_revision, name="upstream_revision")
    candidate = exact_revision(candidate_revision, name="candidate_revision")
    if upstream == candidate:
        raise ValueError(
            "candidate revision must differ from the upstream batch revision"
        )
    if upstream != UPSTREAM_REVISION:
        raise ValueError("raw nested prediction binds another upstream revision")
    if upstream_prediction_batch_id != UPSTREAM_PREDICTION_BATCH_ID:
        raise ValueError("raw nested prediction binds another upstream batch")
    episode, stratum = registered[unit_id]
    identity: dict[str, Any] = {
        "schema": RAW_PREDICTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "covariance_amendment_id": AMENDMENT_ID,
        "nested_repair_id": NESTED_REPAIR_ID,
        "upstream_prediction_batch_id": sha256_digest(
            upstream_prediction_batch_id, name="upstream_prediction_batch_id"
        ),
        "upstream_revision": upstream,
        "candidate_revision": candidate,
        "outer_held_out_object_id": outer_id,
        "object_id": unit_id,
        "episode_id": episode,
        "stratum": stratum,
        "record_role": "held_out" if outer_id == unit_id else "training",
        "variants": _raw_variants(
            variants,
            source_ids=set(registered),
            outer_id=outer_id,
            object_id=unit_id,
        ),
        "source_artifacts": plain_json(
            source_artifact_mapping(source_artifacts, name="source_artifacts")
        ),
        "source_selection_artifact_sha256": SOURCE_SELECTION_ARTIFACT_ID,
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }
    return {**identity, "prediction_record_id": content_id(identity)}


def validate_deform360_v6_raw_nested_prediction(
    value: object, *, cohort: Mapping[str, tuple[int, str]]
) -> dict[str, Any]:
    payload = _mapping(value, name="raw nested prediction")
    require_exact_fields(payload, expected=_PREDICTION_FIELDS, name="raw prediction")
    if payload.get("schema") != RAW_PREDICTION_SCHEMA:
        raise ValueError("raw nested prediction schema changed")
    if payload.get("information_boundary") != _PREDICTION_BOUNDARY:
        raise ValueError("raw nested prediction crossed its boundary")
    rebuilt = build_deform360_v6_raw_nested_prediction(
        cohort=cohort,
        upstream_prediction_batch_id=cast(
            str, payload.get("upstream_prediction_batch_id")
        ),
        upstream_revision=cast(str, payload.get("upstream_revision")),
        candidate_revision=cast(str, payload.get("candidate_revision")),
        outer_held_out_object_id=cast(str, payload.get("outer_held_out_object_id")),
        object_id=cast(str, payload.get("object_id")),
        variants=cast(Mapping[str, Any], payload.get("variants")),
        source_artifacts=cast(Mapping[str, str], payload.get("source_artifacts")),
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("raw nested prediction content changed")
    return rebuilt


def build_deform360_v6_raw_nested_batch(
    records: Sequence[Mapping[str, Any]], *, cohort: Mapping[str, tuple[int, str]]
) -> dict[str, Any]:
    """Seal exactly 10 x 10 nested raw predictions before suffix access."""

    registered = _cohort(cohort)
    if len(records) != 100:
        raise ValueError("v6.1 raw prediction batch must contain 100 records")
    validated = [
        validate_deform360_v6_raw_nested_prediction(row, cohort=registered)
        for row in records
    ]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    upstream_ids: set[str] = set()
    upstream_revisions: set[str] = set()
    candidate_revisions: set[str] = set()
    for row in validated:
        key = (row["outer_held_out_object_id"], row["object_id"])
        if key in by_key:
            raise ValueError("v6.1 raw prediction batch repeats a nested record")
        by_key[key] = row
        upstream_ids.add(row["upstream_prediction_batch_id"])
        upstream_revisions.add(row["upstream_revision"])
        candidate_revisions.add(row["candidate_revision"])
    expected = {(outer, target) for outer in registered for target in registered}
    if set(by_key) != expected:
        raise ValueError("v6.1 raw prediction batch has an incomplete nested roster")
    if not (
        len(upstream_ids) == len(upstream_revisions) == len(candidate_revisions) == 1
    ):
        raise ValueError(
            "v6.1 raw prediction batch mixes revisions or upstream batches"
        )
    ordered = [by_key[key] for key in sorted(by_key)]
    identity: dict[str, Any] = {
        "schema": RAW_BATCH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "covariance_amendment_id": AMENDMENT_ID,
        "nested_repair_id": NESTED_REPAIR_ID,
        "source_selection_artifact_sha256": SOURCE_SELECTION_ARTIFACT_ID,
        "upstream_prediction_batch_id": next(iter(upstream_ids)),
        "upstream_revision": next(iter(upstream_revisions)),
        "candidate_revision": next(iter(candidate_revisions)),
        "record_count": 100,
        "records": ordered,
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }
    return {**identity, "prediction_batch_id": content_id(identity)}


def validate_deform360_v6_raw_nested_batch(
    value: object, *, cohort: Mapping[str, tuple[int, str]]
) -> dict[str, Any]:
    payload = _mapping(value, name="raw nested batch")
    require_exact_fields(payload, expected=_BATCH_FIELDS, name="raw nested batch")
    if payload.get("schema") != RAW_BATCH_SCHEMA:
        raise ValueError("raw nested batch schema changed")
    rebuilt = build_deform360_v6_raw_nested_batch(
        cast(Sequence[Mapping[str, Any]], payload.get("records")), cohort=cohort
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("raw nested batch content changed")
    return rebuilt


def _outcome_variant(
    value: object, *, variant_id: str, prediction: Mapping[str, Any]
) -> dict[str, Any]:
    row = _mapping(value, name=variant_id)
    require_exact_fields(row, expected=_OUTCOME_VARIANT_FIELDS, name=variant_id)
    available = genuine_boolean(row.get("available"), name="available")
    if available != prediction.get("available"):
        raise ValueError("raw outcome availability differs from its prediction")
    if row.get("prediction_artifact_id") != prediction.get("prediction_artifact_id"):
        raise ValueError("raw outcome prediction identity differs from its prediction")
    return {
        "available": available,
        "prediction_artifact_id": prediction.get("prediction_artifact_id"),
        "query_count": genuine_integer(
            row.get("query_count"), name="query_count", minimum=1
        ),
        "point_loss": _finite(row.get("point_loss"), name="point_loss", minimum=0.0),
        "mean_raw_mahalanobis_squared": _finite(
            row.get("mean_raw_mahalanobis_squared"),
            name="mean_raw_mahalanobis_squared",
            minimum=0.0,
        ),
        "mean_log_determinant": _finite(
            row.get("mean_log_determinant"), name="mean_log_determinant"
        ),
        "maximum_raw_mahalanobis_norm": _finite(
            row.get("maximum_raw_mahalanobis_norm"),
            name="maximum_raw_mahalanobis_norm",
            minimum=0.0,
        ),
        "mean_raw_radius": _finite(
            row.get("mean_raw_radius"), name="mean_raw_radius", minimum=0.0
        ),
    }


def _same_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    ignored = {"available", "prediction_artifact_id"}
    return all(left[key] == right[key] for key in left if key not in ignored)


def build_deform360_v6_raw_nested_outcome(
    *,
    prediction_batch: Mapping[str, Any],
    prediction_record_id: str,
    variants: Mapping[str, Any],
    scoring_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Attach raw scoring sufficient statistics after the batch barrier."""

    record_id = sha256_digest(prediction_record_id, name="prediction_record_id")
    matches = [
        _mapping(row, name="prediction")
        for row in _sequence(prediction_batch.get("records"), name="records")
        if _mapping(row, name="prediction").get("prediction_record_id") == record_id
    ]
    if len(matches) != 1:
        raise ValueError("raw nested outcome does not bind one prediction record")
    prediction = matches[0]
    predictions = _mapping(prediction.get("variants"), name="prediction variants")
    rows = _mapping(variants, name="outcome variants")
    require_exact_fields(rows, expected=frozenset(VARIANT_IDS), name="variants")
    normalized = {
        variant_id: _outcome_variant(
            rows[variant_id],
            variant_id=variant_id,
            prediction=_mapping(predictions[variant_id], name=variant_id),
        )
        for variant_id in VARIANT_IDS
    }
    available_vt1 = [
        normalized[variant_id]
        for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH)
        if normalized[variant_id]["available"]
    ]
    if available_vt1:
        reference = available_vt1[0]
        if any(
            row["query_count"] != reference["query_count"]
            or row["point_loss"] != reference["point_loss"]
            for row in available_vt1[1:]
        ):
            raise ValueError(
                "VT1 covariance variants must share one scored point prediction"
            )
    fallback = normalized[B0]
    for variant_id in CHALLENGER_VARIANTS:
        if not normalized[variant_id]["available"] and not _same_score(
            normalized[variant_id], fallback
        ):
            raise ValueError("an unavailable variant must carry exact fallback scores")
    identity: dict[str, Any] = {
        "schema": RAW_OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "nested_repair_id": NESTED_REPAIR_ID,
        "prediction_batch_id": prediction_batch["prediction_batch_id"],
        "prediction_record_id": record_id,
        "variants": normalized,
        "scoring_artifacts": plain_json(
            source_artifact_mapping(scoring_artifacts, name="scoring_artifacts")
        ),
        "information_boundary": dict(_OUTCOME_BOUNDARY),
    }
    return {**identity, "outcome_id": content_id(identity)}


def validate_deform360_v6_raw_nested_outcome(
    value: object, *, prediction_batch: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _mapping(value, name="raw nested outcome")
    require_exact_fields(payload, expected=_OUTCOME_FIELDS, name="raw nested outcome")
    if payload.get("schema") != RAW_OUTCOME_SCHEMA:
        raise ValueError("raw nested outcome schema changed")
    if payload.get("information_boundary") != _OUTCOME_BOUNDARY:
        raise ValueError("raw nested outcome crossed its boundary")
    rebuilt = build_deform360_v6_raw_nested_outcome(
        prediction_batch=prediction_batch,
        prediction_record_id=cast(str, payload.get("prediction_record_id")),
        variants=cast(Mapping[str, Any], payload.get("variants")),
        scoring_artifacts=cast(Mapping[str, str], payload.get("scoring_artifacts")),
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("raw nested outcome content changed")
    return rebuilt


def assemble_deform360_v6_nested_evidence(
    *,
    prediction_batch: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, tuple[int, str]],
) -> dict[str, Any]:
    """Join all 100 outcomes to the immutable raw prediction batch."""

    validated_batch = validate_deform360_v6_raw_nested_batch(
        prediction_batch,
        cohort=cohort,
    )
    if len(outcomes) != 100:
        raise ValueError("v6.1 nested evidence requires 100 outcomes")
    validated = [
        validate_deform360_v6_raw_nested_outcome(
            row,
            prediction_batch=validated_batch,
        )
        for row in outcomes
    ]
    by_record: dict[str, dict[str, Any]] = {}
    for outcome in validated:
        record_id = outcome["prediction_record_id"]
        if record_id in by_record:
            raise ValueError("v6.1 nested evidence repeats an outcome")
        by_record[record_id] = outcome
    records: list[dict[str, Any]] = []
    for raw in _sequence(validated_batch.get("records"), name="records"):
        prediction = _mapping(raw, name="prediction")
        record_id = cast(str, prediction["prediction_record_id"])
        if record_id not in by_record:
            raise ValueError("v6.1 nested evidence omits an outcome")
        records.append({"prediction": prediction, "outcome": by_record[record_id]})
    identity: dict[str, Any] = {
        "schema": NESTED_EVIDENCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "nested_repair_id": NESTED_REPAIR_ID,
        "prediction_batch_id": validated_batch["prediction_batch_id"],
        "record_count": 100,
        "records": records,
        "information_boundary": dict(_EVIDENCE_BOUNDARY),
    }
    return {**identity, "evidence_id": content_id(identity)}


def validate_deform360_v6_nested_evidence(
    value: object,
    *,
    cohort: Mapping[str, tuple[int, str]],
) -> dict[str, Any]:
    payload = _mapping(value, name="nested evidence")
    require_exact_fields(payload, expected=_EVIDENCE_FIELDS, name="nested evidence")
    if payload.get("schema") != NESTED_EVIDENCE_SCHEMA:
        raise ValueError("nested evidence schema changed")
    if payload.get("information_boundary") != _EVIDENCE_BOUNDARY:
        raise ValueError("nested evidence crossed its boundary")
    records = _sequence(payload.get("records"), name="records")
    if (
        genuine_integer(payload.get("record_count"), name="record_count") != 100
        or len(records) != 100
    ):
        raise ValueError("nested evidence must contain 100 records")
    registered = _cohort(cohort)
    predictions: list[dict[str, Any]] = []
    outcomes: list[Mapping[str, Any]] = []
    for index, raw in enumerate(records):
        row = _mapping(raw, name=f"records[{index}]")
        require_exact_fields(
            row, expected=_EVIDENCE_RECORD_FIELDS, name=f"records[{index}]"
        )
        predictions.append(
            validate_deform360_v6_raw_nested_prediction(
                row.get("prediction"),
                cohort=registered,
            )
        )
        outcomes.append(_mapping(row.get("outcome"), name="outcome"))
    batch = build_deform360_v6_raw_nested_batch(predictions, cohort=registered)
    if batch["prediction_batch_id"] != payload.get("prediction_batch_id"):
        raise ValueError("nested evidence prediction batch identity changed")
    rebuilt = assemble_deform360_v6_nested_evidence(
        prediction_batch=batch,
        outcomes=outcomes,
        cohort=registered,
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("nested evidence content changed")
    return rebuilt


def _stats(record: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
    return _mapping(
        _mapping(record.get("outcome"), name="outcome").get("variants"),
        name="outcome variants",
    )[variant_id]


def _prediction(record: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
    return _mapping(
        _mapping(record.get("prediction"), name="prediction").get("variants"),
        name="prediction variants",
    )[variant_id]


def _fit_calibration(
    records: Sequence[Mapping[str, Any]], variant_id: str
) -> dict[str, Any] | None:
    if not records or any(
        not bool(_stats(row, variant_id)["available"]) for row in records
    ):
        return None
    mean_mahal = float(
        np.mean(
            [
                float(_stats(row, variant_id)["mean_raw_mahalanobis_squared"])
                for row in records
            ]
        )
    )
    scale = float(np.clip(mean_mahal / QUERY_DIMENSION, MINIMUM_SCALE, MAXIMUM_SCALE))
    scores = sorted(
        float(_stats(row, variant_id)["maximum_raw_mahalanobis_norm"])
        / math.sqrt(scale)
        for row in records
    )
    rank = math.ceil((len(records) + 1) * NOMINAL_COVERAGE)
    if rank > len(scores):
        return None
    quantile = scores[rank - 1]
    descriptor = {
        "variant_id": variant_id,
        "training_object_ids": sorted(
            cast(str, _mapping(row["prediction"], name="prediction")["object_id"])
            for row in records
        ),
        "query_dimension": QUERY_DIMENSION,
        "covariance_scale": scale,
        "grouped_residual_rank": rank,
        "grouped_residual_quantile": quantile,
        "nominal_coverage_target": NOMINAL_COVERAGE,
    }
    return {**descriptor, "calibration_artifact_id": content_id(descriptor)}


def _calibrated_metrics(
    stats: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    scale = float(calibration["covariance_scale"])
    quantile = float(calibration["grouped_residual_quantile"])
    nll = 0.5 * (
        QUERY_DIMENSION * math.log(2.0 * math.pi)
        + float(stats["mean_log_determinant"])
        + QUERY_DIMENSION * math.log(scale)
        + float(stats["mean_raw_mahalanobis_squared"]) / scale
    )
    normalized_max = float(stats["maximum_raw_mahalanobis_norm"]) / math.sqrt(scale)
    width = 2.0 * quantile * math.sqrt(scale) * float(stats["mean_raw_radius"])
    return {
        "proper_score": nll,
        "interval_covered": normalized_max <= quantile + 1e-12,
        "interval_width": width,
    }


def _fit_guard(
    records: Sequence[Mapping[str, Any]], variant_id: str
) -> dict[str, Any] | None:
    candidates = sorted(
        {
            float(_prediction(row, variant_id)["risk_score"])
            for row in records
            if bool(_prediction(row, variant_id)["available"])
        }
    )
    minimum = math.ceil(0.8 * len(records))
    fitted: list[tuple[tuple[float, int, int, float], dict[str, Any]]] = []
    for threshold in candidates:
        accepted = [
            row
            for row in records
            if bool(_prediction(row, variant_id)["available"])
            and float(_prediction(row, variant_id)["risk_score"]) <= threshold
        ]
        if len(accepted) < minimum:
            continue
        harmful = sum(
            float(_stats(row, variant_id)["point_loss"])
            > 1.02 * float(_stats(row, B0)["point_loss"])
            for row in accepted
        )
        if harmful > 1:
            continue
        accepted_ids = {
            cast(str, _mapping(row["prediction"], name="prediction")["object_id"])
            for row in accepted
        }
        deployed = [
            float(
                _stats(
                    row,
                    variant_id
                    if cast(
                        str, _mapping(row["prediction"], name="prediction")["object_id"]
                    )
                    in accepted_ids
                    else B0,
                )["point_loss"]
            )
            for row in records
        ]
        mean_deployed_point_loss = float(np.mean(deployed))
        descriptor = {
            "variant_id": variant_id,
            "training_object_ids": sorted(
                cast(str, _mapping(row["prediction"], name="prediction")["object_id"])
                for row in records
            ),
            "threshold": threshold,
            "accepted_count": len(accepted),
            "harmful_accepted_count": harmful,
            "mean_deployed_point_loss": mean_deployed_point_loss,
        }
        objective: tuple[float, int, int, float] = (
            mean_deployed_point_loss,
            harmful,
            -len(accepted),
            threshold,
        )
        fitted.append((objective, descriptor))
    if not fitted:
        return None
    descriptor = min(fitted, key=lambda item: item[0])[1]
    return {**descriptor, "guard_artifact_id": content_id(descriptor)}


def _accepted(
    record: Mapping[str, Any], variant_id: str, guard: Mapping[str, Any] | None
) -> bool:
    return bool(
        guard is not None
        and _prediction(record, variant_id)["available"]
        and float(_prediction(record, variant_id)["risk_score"])
        <= float(guard["threshold"])
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fit_selection(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calibrations = {
        variant_id: _fit_calibration(records, variant_id)
        for variant_id in (B0, B1, *CHALLENGER_VARIANTS)
    }
    if calibrations[B0] is None or calibrations[B1] is None:
        raise ValueError("source baselines require complete covariance statistics")
    reference_nll = _mean(
        [
            float(
                _calibrated_metrics(
                    _stats(row, B1), cast(Mapping[str, Any], calibrations[B1])
                )["proper_score"]
            )
            for row in records
        ]
    )
    reference_point = _mean([float(_stats(row, B1)["point_loss"]) for row in records])
    summaries: dict[str, dict[str, Any]] = {}
    b1_width = _mean(
        [
            float(
                _calibrated_metrics(
                    _stats(row, B1), cast(Mapping[str, Any], calibrations[B1])
                )["interval_width"]
            )
            for row in records
        ]
    )
    summaries[B1] = {
        "variant_id": B1,
        "eligible": True,
        "guard": None,
        "calibration": calibrations[B1],
        "mean_deployed_point_loss": reference_point,
        "mean_deployed_proper_score": reference_nll,
        "mean_deployed_interval_width": b1_width,
        "accepted_count": len(records),
        "harmful_accepted_count": 0,
    }
    for variant_id in CHALLENGER_VARIANTS:
        guard = _fit_guard(records, variant_id)
        calibration = calibrations[variant_id]
        available = guard is not None and calibration is not None
        point_values: list[float] = []
        nll_values: list[float] = []
        width_values: list[float] = []
        accepted_count = 0
        harmful = 0
        if available:
            for row in records:
                accepted = _accepted(row, variant_id, guard)
                deployed_variant = variant_id if accepted else B0
                selected_calibration = cast(
                    Mapping[str, Any], calibrations[deployed_variant]
                )
                metrics = _calibrated_metrics(
                    _stats(row, deployed_variant), selected_calibration
                )
                point_values.append(float(_stats(row, deployed_variant)["point_loss"]))
                nll_values.append(float(metrics["proper_score"]))
                width_values.append(float(metrics["interval_width"]))
                accepted_count += int(accepted)
                harmful += int(
                    accepted
                    and float(_stats(row, variant_id)["point_loss"])
                    > 1.02 * float(_stats(row, B0)["point_loss"])
                )
        mean_point = _mean(point_values) if point_values else None
        mean_nll = _mean(nll_values) if nll_values else None
        mean_width = _mean(width_values) if width_values else None
        candidate_eligible = bool(
            available
            and mean_point is not None
            and mean_nll is not None
            and reference_point > 0.0
            and 1.0 - mean_point / reference_point >= 0.02 - 1e-12
            and mean_nll <= reference_nll + 1e-12
            and harmful <= 1
            and accepted_count >= math.ceil(0.8 * len(records))
        )
        summaries[variant_id] = {
            "variant_id": variant_id,
            "eligible": candidate_eligible,
            "guard": guard,
            "calibration": calibration,
            "mean_deployed_point_loss": mean_point,
            "mean_deployed_proper_score": mean_nll,
            "mean_deployed_interval_width": mean_width,
            "accepted_count": accepted_count,
            "harmful_accepted_count": harmful,
        }
    eligible_summaries = [
        summaries[B1],
        *(
            summaries[item]
            for item in CHALLENGER_VARIANTS
            if summaries[item]["eligible"]
        ),
    ]

    def ranking(
        row: Mapping[str, Any],
    ) -> tuple[float, float, float, tuple[int, int, float, int]]:
        return (
            float(row["mean_deployed_proper_score"]),
            float(row["mean_deployed_point_loss"]),
            float(row["mean_deployed_interval_width"]),
            VARIANT_COMPLEXITY[cast(str, row["variant_id"])],
        )

    selected = min(eligible_summaries, key=ranking)
    return {
        "selected_variant": selected["variant_id"],
        "summaries": summaries,
        "fallback_calibration": calibrations[B0],
        "reference_calibration": calibrations[B1],
    }


def _deploy(
    record: Mapping[str, Any], variant_id: str, fit: Mapping[str, Any]
) -> dict[str, Any]:
    summaries = cast(Mapping[str, Mapping[str, Any]], fit["summaries"])
    summary = summaries[variant_id]
    if variant_id == B1:
        accepted = True
        selected = B1
    else:
        accepted = _accepted(
            record, variant_id, cast(Mapping[str, Any] | None, summary["guard"])
        )
        selected = variant_id if accepted else B0
    calibration = (
        cast(Mapping[str, Any], summary["calibration"])
        if selected == variant_id
        else cast(Mapping[str, Any], fit["fallback_calibration"])
    )
    metrics = _calibrated_metrics(_stats(record, selected), calibration)
    return {
        "accepted": accepted,
        "deployed_variant": selected,
        "point_loss": float(_stats(record, selected)["point_loss"]),
        **metrics,
        "prediction_artifact_id": _prediction(record, selected)[
            "prediction_artifact_id"
        ],
        "calibration_artifact_id": calibration["calibration_artifact_id"],
        "guard_artifact_id": None
        if summary["guard"] is None
        else summary["guard"]["guard_artifact_id"],
        "guard_threshold": None
        if summary["guard"] is None
        else summary["guard"]["threshold"],
    }


def evaluate_deform360_v6_nested_source_gate(
    evidence: Mapping[str, Any], *, cohort: Mapping[str, tuple[int, str]]
) -> dict[str, Any]:
    """Run the corrected ten-fold candidate/covariance/guard/calibration gate."""

    registered = _cohort(cohort)
    validated = validate_deform360_v6_nested_evidence(
        evidence,
        cohort=registered,
    )
    records = cast(list[Mapping[str, Any]], validated["records"])
    by_key = {
        (
            cast(
                str,
                _mapping(row["prediction"], name="prediction")[
                    "outer_held_out_object_id"
                ],
            ),
            cast(str, _mapping(row["prediction"], name="prediction")["object_id"]),
        ): row
        for row in records
    }
    folds: list[dict[str, Any]] = []
    for outer_id in sorted(registered):
        training = [
            by_key[(outer_id, item)] for item in sorted(registered) if item != outer_id
        ]
        held = by_key[(outer_id, outer_id)]
        fit = _fit_selection(training)
        selected = cast(str, fit["selected_variant"])
        deployed = _deploy(held, selected, fit)
        reference = _deploy(held, B1, fit)
        # Exact physical fallback metrics use the separately fitted B0 calibration.
        fallback_metrics = _calibrated_metrics(
            _stats(held, B0), cast(Mapping[str, Any], fit["fallback_calibration"])
        )
        fallback = {
            "point_loss": float(_stats(held, B0)["point_loss"]),
            **fallback_metrics,
            "prediction_artifact_id": _prediction(held, B0)["prediction_artifact_id"],
            "calibration_artifact_id": cast(
                Mapping[str, Any], fit["fallback_calibration"]
            )["calibration_artifact_id"],
        }
        exact_fallback = bool(
            deployed["accepted"]
            or (
                deployed["deployed_variant"] == B0
                and deployed["point_loss"] == fallback["point_loss"]
                and deployed["proper_score"] == fallback["proper_score"]
                and deployed["interval_width"] == fallback["interval_width"]
                and deployed["interval_covered"] == fallback["interval_covered"]
                and deployed["prediction_artifact_id"]
                == fallback["prediction_artifact_id"]
                and deployed["calibration_artifact_id"]
                == fallback["calibration_artifact_id"]
            )
        )
        stratum = registered[outer_id][1]
        folds.append(
            {
                "outer_held_out_object_id": outer_id,
                "stratum": stratum,
                "selected_variant": selected,
                "accepted": deployed["accepted"] if selected != B1 else False,
                "deployed": deployed,
                "reference": reference,
                "fallback": fallback,
                "nonregressing_vs_reference": deployed["point_loss"]
                <= reference["point_loss"] + 1e-12,
                "harmful_accepted": bool(
                    selected != B1
                    and deployed["accepted"]
                    and deployed["point_loss"] > 1.02 * fallback["point_loss"]
                ),
                "exact_fallback": exact_fallback,
            }
        )
    full_records = [by_key[(item, item)] for item in sorted(registered)]
    full_fit = _fit_selection(full_records)
    selected_variant = cast(str, full_fit["selected_variant"])
    selected_fold_count = sum(
        row["selected_variant"] == selected_variant for row in folds
    )
    accepted = sum(bool(row["accepted"]) for row in folds)
    nonregressing = sum(bool(row["nonregressing_vs_reference"]) for row in folds)
    accepted_by_stratum = {
        stratum: sum(
            bool(row["accepted"]) for row in folds if row["stratum"] == stratum
        )
        for stratum in ("sheet", "volumetric")
    }
    nonregressing_by_stratum = {
        stratum: sum(
            bool(row["nonregressing_vs_reference"])
            for row in folds
            if row["stratum"] == stratum
        )
        for stratum in ("sheet", "volumetric")
    }
    coverage = _mean(
        [float(bool(row["deployed"]["interval_covered"])) for row in folds]
    )
    mean_width = _mean([float(row["deployed"]["interval_width"]) for row in folds])
    reference_width = _mean(
        [float(row["reference"]["interval_width"]) for row in folds]
    )
    mean_point = _mean([float(row["deployed"]["point_loss"]) for row in folds])
    reference_point = _mean([float(row["reference"]["point_loss"]) for row in folds])
    mean_nll = _mean([float(row["deployed"]["proper_score"]) for row in folds])
    reference_nll = _mean([float(row["reference"]["proper_score"]) for row in folds])
    point_improvement = (
        0.0 if reference_point <= 0.0 else 1.0 - mean_point / reference_point
    )
    checks = {
        "challenger_selected": selected_variant in CHALLENGER_VARIANTS,
        "stable_variant_selection": selected_fold_count >= 8,
        "minimum_relative_point_improvement_over_reference": point_improvement
        >= 0.02 - 1e-12,
        "proper_score_nonregression_over_reference": mean_nll <= reference_nll + 1e-12,
        "held_out_nonregression": nonregressing >= 8,
        "held_out_nonregression_per_stratum": all(
            value >= 4 for value in nonregressing_by_stratum.values()
        ),
        "minimum_source_acceptance": accepted >= 8,
        "minimum_source_acceptance_per_stratum": all(
            value >= 4 for value in accepted_by_stratum.values()
        ),
        "maximum_harmful_accepted_units": sum(
            bool(row["harmful_accepted"]) for row in folds
        )
        <= 1,
        "coverage_in_registered_range": 0.8 - 1e-12 <= coverage <= 0.98 + 1e-12,
        "interval_width_nonregression": mean_width <= 1.25 * reference_width + 1e-12,
        "all_rejections_are_exact_fallback": all(
            bool(row["exact_fallback"]) for row in folds
        ),
    }
    passed = all(checks.values())
    full_summary = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], full_fit["summaries"])[selected_variant],
    )
    descriptor: dict[str, Any] = {
        "schema": NESTED_RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "nested_repair_id": NESTED_REPAIR_ID,
        "evidence_id": validated["evidence_id"],
        "provisional_selected_variant": selected_variant,
        "selected_variant": selected_variant if passed else B1,
        "selected_candidate_family": VARIANT_POLICY_CANDIDATE[
            selected_variant if passed else B1
        ],
        "selected_covariance": VARIANT_COVARIANCE[selected_variant if passed else B1],
        "outer_fold_selected_count": selected_fold_count,
        "folds": folds,
        "full_source_fit": {
            "guard": full_summary["guard"] if passed else None,
            "calibration": full_summary["calibration"]
            if passed
            else full_fit["reference_calibration"],
        },
        "aggregate": {
            "accepted_count": accepted,
            "accepted_count_by_stratum": accepted_by_stratum,
            "nonregressing_count": nonregressing,
            "nonregressing_count_by_stratum": nonregressing_by_stratum,
            "coverage": coverage,
            "mean_interval_width": mean_width,
            "reference_mean_interval_width": reference_width,
            "mean_point_loss": mean_point,
            "reference_mean_point_loss": reference_point,
            "relative_point_improvement": point_improvement,
            "mean_proper_score": mean_nll,
            "reference_mean_proper_score": reference_nll,
        },
        "checks": checks,
        "source_gate_passed": passed,
        "source_continuation_authorized": passed,
        "fresh_target_selection_authorized": False,
        "fresh_target_payload_access_authorized": False,
        "claim_authorized": False,
        "status": "source-challenger-advanced"
        if passed
        else "source-reference-retained",
        "next_stage": "fit-full-source-artifacts"
        if passed
        else "terminate-v6-before-fresh-cohort-selection",
        "information_boundary": {
            "v5_confirmation_payloads_used": False,
            "v5_confirmation_outcomes_used": False,
            "v6_target_payloads_used": False,
            "v6_target_outcomes_used": False,
            "human_selection_used": False,
            "replacement_allowed": False,
        },
    }
    return {**descriptor, "result_id": content_id(descriptor)}


def publish_deform360_v6_raw_nested_batch(
    value: Mapping[str, Any],
    path: str | Path,
    *,
    cohort: Mapping[str, tuple[int, str]],
) -> None:
    validated = validate_deform360_v6_raw_nested_batch(value, cohort=cohort)
    write_atomic_json(validated, path, overwrite=False)


def publish_deform360_v6_nested_evidence(
    value: Mapping[str, Any],
    path: str | Path,
    *,
    cohort: Mapping[str, tuple[int, str]],
) -> None:
    validate_deform360_v6_nested_evidence(value, cohort=cohort)
    write_atomic_json(value, path, overwrite=False)


def publish_deform360_v6_nested_result(
    value: Mapping[str, Any],
    path: str | Path,
    *,
    evidence: Mapping[str, Any],
    cohort: Mapping[str, tuple[int, str]],
) -> None:
    expected = evaluate_deform360_v6_nested_source_gate(
        evidence,
        cohort=cohort,
    )
    if plain_json(value) != expected:
        raise ValueError("nested result differs from replayed source evidence")
    write_atomic_json(expected, path, overwrite=False)


__all__ = [
    "NESTED_REPAIR_ID",
    "NESTED_REPAIR_SCHEMA",
    "NESTED_RESULT_SCHEMA",
    "RAW_BATCH_SCHEMA",
    "RAW_OUTCOME_SCHEMA",
    "RAW_PREDICTION_SCHEMA",
    "assemble_deform360_v6_nested_evidence",
    "build_deform360_v6_raw_nested_batch",
    "build_deform360_v6_raw_nested_outcome",
    "build_deform360_v6_raw_nested_prediction",
    "evaluate_deform360_v6_nested_source_gate",
    "load_deform360_v6_nested_source_repair",
    "publish_deform360_v6_nested_evidence",
    "publish_deform360_v6_nested_result",
    "publish_deform360_v6_raw_nested_batch",
    "validate_deform360_v6_nested_evidence",
    "validate_deform360_v6_raw_nested_batch",
    "validate_deform360_v6_raw_nested_outcome",
    "validate_deform360_v6_raw_nested_prediction",
]
