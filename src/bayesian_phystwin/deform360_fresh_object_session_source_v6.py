"""Prediction-first source gate for Deform360 fresh object-session v6.

Ten outer-fold predictions are sealed before any source suffix is scored. The
module converts their exact, guarded candidate records into the repository's
candidate-agnostic source tournament and applies the additional v6 stability,
stratum, coverage, and interval-width rules. Fresh target selection remains a
separate, closed stage.
"""

from __future__ import annotations

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
from .discrepancy_candidate_tournament import (
    DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
    analyze_discrepancy_candidate_tournament,
)

POLICY_ID: Final = "480bcc287a6d8ee1523c2e0d09e31b9cc12557ea3788d62642b84a4f1897671f"
AMENDMENT_ID: Final = "6113d481321f176929ccab0a38a4efacbeeb6620f53d7954d102b0b4fb1879c7"
POLICY_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-prospective-policy"
)
AMENDMENT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-source-covariance-amendment"
)
PREDICTION_SEAL_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-source-prediction-seal"
)
PREDICTION_BATCH_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-source-prediction-batch"
)
SOURCE_OUTCOME_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-source-outcome"
)
SOURCE_EVIDENCE_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-source-evidence"
)
SOURCE_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-source-result"
)
SCHEMA_VERSION: Final = 1

B0: Final = "b0_physical_fallback"
B1: Final = "b1_last_causal_residual"
D1_NATIVE: Final = "d1_native_model_average"
VT1_WORKING: Final = "vt1_working_irls"
VT1_OBSERVED: Final = "vt1_observed_information"
VT1_SANDWICH: Final = "vt1_group_sandwich"
VARIANT_IDS: Final = (
    B0,
    B1,
    D1_NATIVE,
    VT1_WORKING,
    VT1_OBSERVED,
    VT1_SANDWICH,
)
CHALLENGER_VARIANTS: Final = (
    D1_NATIVE,
    VT1_WORKING,
    VT1_OBSERVED,
    VT1_SANDWICH,
)
VARIANT_FAMILY: Final = {
    B0: B0,
    B1: B1,
    D1_NATIVE: "d1_dynamic_endpoint_model_average_v2",
    VT1_WORKING: "vt1_joint_sparse_visuotactile_guarded_v5",
    VT1_OBSERVED: "vt1_joint_sparse_visuotactile_guarded_v5",
    VT1_SANDWICH: "vt1_joint_sparse_visuotactile_guarded_v5",
}
VARIANT_POLICY_CANDIDATE: Final = {
    B0: "B0_physical_fallback",
    B1: "B1_last_causal_residual",
    D1_NATIVE: "D1_dynamic_endpoint_model_average_v2",
    VT1_WORKING: "VT1_joint_sparse_visuotactile_guarded_v5",
    VT1_OBSERVED: "VT1_joint_sparse_visuotactile_guarded_v5",
    VT1_SANDWICH: "VT1_joint_sparse_visuotactile_guarded_v5",
}
VARIANT_COVARIANCE: Final = {
    B0: "reference",
    B1: "reference",
    D1_NATIVE: "native-model-average",
    VT1_WORKING: "working-irls",
    VT1_OBSERVED: "observed-information",
    VT1_SANDWICH: "group-sandwich",
}
# These are ordinal tie-break encodings, not measured performance quantities.
VARIANT_COMPLEXITY: Final = {
    B0: (0, 0, 0.0, 0),
    B1: (0, 0, 0.1, 0),
    D1_NATIVE: (1, 1, 1.0, 1),
    VT1_WORKING: (2, 1, 2.0, 1),
    VT1_OBSERVED: (2, 1, 2.0, 2),
    VT1_SANDWICH: (2, 1, 2.0, 3),
}

_PREDICTION_BOUNDARY = {
    "source_suffix_opened": False,
    "v5_terminal_outcome_used": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "future_object_observations_used_for_prediction": False,
    "human_selection_used": False,
    "replacement_allowed": False,
}
_OUTCOME_BOUNDARY = {
    "source_suffix_opened_after_prediction_batch": True,
    "v5_terminal_outcome_used": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "future_object_observations_used_for_prediction": False,
    "human_selection_used": False,
    "replacement_allowed": False,
}
_EVIDENCE_BOUNDARY = {
    "all_source_predictions_sealed_before_source_suffix_scoring": True,
    "v5_terminal_outcome_used": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "future_object_observations_used_for_prediction": False,
    "human_selection_used": False,
    "replacement_allowed": False,
}

_VARIANT_PREDICTION_FIELDS = frozenset(
    {
        "accepted",
        "available",
        "covariance_artifact_id",
        "fit_artifact_id",
        "fit_object_ids",
        "guard_artifact_id",
        "guard_threshold",
        "interval_artifact_id",
        "prediction_artifact_id",
        "risk_score",
        "unavailable_reason",
    }
)
_VARIANT_OUTCOME_FIELDS = frozenset(
    {
        "available",
        "interval_covered",
        "interval_width",
        "point_loss",
        "prediction_artifact_id",
        "proper_score",
    }
)
_SEAL_FIELDS = frozenset(
    {
        "amendment_id",
        "episode_id",
        "implementation_revision",
        "information_boundary",
        "object_id",
        "policy_id",
        "schema",
        "schema_version",
        "seal_id",
        "selection_artifact_sha256",
        "source_artifacts",
        "stratum",
        "variants",
    }
)
_BATCH_FIELDS = frozenset(
    {
        "amendment_id",
        "implementation_revision",
        "information_boundary",
        "policy_id",
        "prediction_batch_id",
        "record_count",
        "records",
        "schema",
        "schema_version",
        "selection_artifact_sha256",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "amendment_id",
        "implementation_revision",
        "information_boundary",
        "object_id",
        "outcome_id",
        "policy_id",
        "prediction_batch_id",
        "prediction_seal_id",
        "schema",
        "schema_version",
        "scoring_artifacts",
        "selection_artifact_sha256",
        "variants",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "amendment_id",
        "evidence_id",
        "implementation_revision",
        "information_boundary",
        "policy_id",
        "prediction_batch_id",
        "record_count",
        "records",
        "schema",
        "schema_version",
        "selection_artifact_sha256",
        "tournament_input",
    }
)
_EVIDENCE_RECORD_FIELDS = frozenset(
    {
        "episode_id",
        "object_id",
        "outcome",
        "prediction",
        "prediction_seal_id",
        "stratum",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _canonical_id(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result.strip() != result or "\x00" in result:
        raise ValueError(f"{name} must be a canonical string")
    return result


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _content_identity(value: Mapping[str, Any], id_field: str, name: str) -> None:
    declared = sha256_digest(value.get(id_field), name=id_field)
    identity = {key: item for key, item in value.items() if key != id_field}
    if declared != content_id(identity):
        raise ValueError(f"{name} content identity changed")


def load_deform360_fresh_object_session_v6_policy(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load and validate the immutable v6 design policy."""

    policy = load_strict_json_object(path, label="Deform360 v6 policy")
    if policy.get("schema") != POLICY_SCHEMA or policy.get("schema_version") != 6:
        raise ValueError("v6 policy schema changed")
    _content_identity(policy, "policy_id", "v6 policy")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError("v6 policy identity changed")
    tournament = _mapping(policy.get("source_tournament"), name="source_tournament")
    if tournament.get("source_unit_count") != 10:
        raise ValueError("v6 source unit count changed")
    return policy


def load_deform360_fresh_object_session_v6_covariance_amendment(
    path: str | Path,
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load the pre-score candidate-specific covariance amendment."""

    amendment = load_strict_json_object(path, label="Deform360 v6 amendment")
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("v6 covariance amendment schema changed")
    _content_identity(amendment, "amendment_id", "v6 covariance amendment")
    base = _mapping(amendment.get("base_policy"), name="base_policy")
    if base.get("policy_id") != policy.get("policy_id"):
        raise ValueError("v6 covariance amendment binds another policy")
    boundary = _mapping(amendment.get("information_boundary"), name="boundary")
    if boundary.get("source_outcomes_used") is not False:
        raise ValueError("v6 covariance amendment used source outcomes")
    candidate = _mapping(
        amendment.get("candidate_covariance_rosters"), name="candidate rosters"
    )
    expected = {
        "D1_dynamic_endpoint_model_average_v2": ("native-model-average",),
        "VT1_joint_sparse_visuotactile_guarded_v5": (
            "working-irls",
            "observed-information",
            "group-sandwich",
        ),
    }
    for candidate_id, roster in expected.items():
        row = _mapping(candidate.get(candidate_id), name=candidate_id)
        observed = tuple(
            _sequence(row.get("required_methods"), name="required_methods")
        )
        if observed != roster:
            raise ValueError("v6 candidate covariance roster changed")
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("v6 covariance amendment identity changed")
    return amendment


def load_deform360_v6_source_selection(
    path: str | Path,
    policy: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, tuple[int, str]]]:
    """Load the exact ten opened source object-session identities."""

    selection = load_strict_json_object(path, label="Deform360 source selection")
    return validate_deform360_v6_source_selection(selection, policy)


def validate_deform360_v6_source_selection(
    selection: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, tuple[int, str]]]:
    """Validate an already loaded v5-development source selection."""

    predecessor = _mapping(policy.get("predecessor_boundary"), name="predecessor")
    if selection.get("selection_artifact_sha256") != predecessor.get(
        "v5_selection_artifact_sha256"
    ):
        raise ValueError("v6 source selection identity changed")
    rows = _sequence(
        _mapping(selection.get("selection"), name="selection").get("calibration"),
        name="calibration",
    )
    cohort: dict[str, tuple[int, str]] = {}
    for raw in rows:
        row = _mapping(raw, name="calibration row")
        object_id = _canonical_id(row.get("object_id"), name="object_id")
        if object_id in cohort:
            raise ValueError("v6 source selection repeats an object")
        episode = genuine_integer(row.get("episode_id"), name="episode_id", minimum=0)
        stratum = _canonical_id(row.get("stratum"), name="stratum")
        if stratum not in {"sheet", "volumetric"}:
            raise ValueError("v6 source stratum changed")
        cohort[object_id] = (episode, stratum)
    if len(cohort) != 10:
        raise ValueError("v6 source selection must contain ten units")
    if Counter(stratum for _, stratum in cohort.values()) != {
        "sheet": 5,
        "volumetric": 5,
    }:
        raise ValueError("v6 source stratum counts changed")
    return selection, cohort


def _variant_prediction(
    value: object,
    *,
    variant_id: str,
    source_ids: set[str],
    object_id: str,
) -> dict[str, Any]:
    row = _mapping(value, name=variant_id)
    require_exact_fields(
        row,
        expected=_VARIANT_PREDICTION_FIELDS,
        name=variant_id,
    )
    available = genuine_boolean(row.get("available"), name="available")
    accepted = genuine_boolean(row.get("accepted"), name="accepted")
    reason = row.get("unavailable_reason")
    if variant_id in {B0, B1}:
        expected_fit: tuple[str, ...] = ()
    else:
        expected_fit = tuple(sorted(source_ids - {object_id}))
    supplied_fit = canonical_sorted_strings(
        cast(Sequence[str], row.get("fit_object_ids")),
        name="fit_object_ids",
        allow_empty=variant_id in {B0, B1},
    )
    if supplied_fit != expected_fit:
        raise ValueError("v6 source variant fit roster changed")
    if variant_id == B0 and accepted:
        raise ValueError("physical fallback must be recorded as rejected")
    if variant_id == B1 and not accepted:
        raise ValueError("last residual must remain the registered reference")
    if not available and accepted:
        raise ValueError("an unavailable v6 source variant cannot be accepted")
    artifact_fields = (
        "prediction_artifact_id",
        "fit_artifact_id",
        "guard_artifact_id",
        "covariance_artifact_id",
        "interval_artifact_id",
    )
    if available:
        artifacts = {
            key: sha256_digest(row.get(key), name=f"{variant_id}.{key}")
            for key in artifact_fields
        }
        if variant_id in {B0, B1}:
            risk = None
            threshold = None
            if (
                row.get("risk_score") is not None
                or row.get("guard_threshold") is not None
            ):
                raise ValueError("a source baseline must not carry a guard score")
        else:
            risk = _finite(row.get("risk_score"), name="risk_score", minimum=0.0)
            threshold_value = row.get("guard_threshold")
            threshold = (
                None
                if threshold_value is None
                else _finite(threshold_value, name="guard_threshold", minimum=0.0)
            )
            expected_acceptance = threshold is not None and risk <= threshold
            if accepted != expected_acceptance:
                raise ValueError("v6 source guard decision differs from its threshold")
        if reason is not None:
            raise ValueError("an available source variant has a failure reason")
        unavailable_reason = None
    else:
        if any(row.get(key) is not None for key in artifact_fields):
            raise ValueError("an unavailable source variant has an artifact")
        if row.get("risk_score") is not None or row.get("guard_threshold") is not None:
            raise ValueError("an unavailable source variant has a guard score")
        artifacts = {key: None for key in artifact_fields}
        risk = None
        threshold = None
        unavailable_reason = _canonical_id(reason, name="unavailable_reason")
    return {
        "available": available,
        "accepted": accepted,
        **artifacts,
        "fit_object_ids": list(supplied_fit),
        "risk_score": risk,
        "guard_threshold": threshold,
        "unavailable_reason": unavailable_reason,
    }


def _prediction_variants(
    value: object,
    *,
    source_ids: set[str],
    object_id: str,
) -> dict[str, dict[str, Any]]:
    variants = _mapping(value, name="variants")
    require_exact_fields(variants, expected=frozenset(VARIANT_IDS), name="variants")
    return {
        variant_id: _variant_prediction(
            variants[variant_id],
            variant_id=variant_id,
            source_ids=source_ids,
            object_id=object_id,
        )
        for variant_id in VARIANT_IDS
    }


def build_deform360_v6_source_prediction_seal(
    *,
    policy: Mapping[str, Any],
    amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
    implementation_revision: str,
    object_id: str,
    variants: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build one held-out source prediction seal before suffix scoring."""

    _, cohort = validate_deform360_v6_source_selection(selection, policy)
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("v6 source seal uses another covariance amendment")
    unit_id = _canonical_id(object_id, name="object_id")
    if unit_id not in cohort:
        raise ValueError("v6 source seal uses an unregistered object")
    episode, stratum = cohort[unit_id]
    identity: dict[str, Any] = {
        "schema": PREDICTION_SEAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "selection_artifact_sha256": selection["selection_artifact_sha256"],
        "implementation_revision": exact_revision(
            implementation_revision, name="implementation_revision"
        ),
        "object_id": unit_id,
        "episode_id": episode,
        "stratum": stratum,
        "variants": _prediction_variants(
            variants,
            source_ids=set(cohort),
            object_id=unit_id,
        ),
        "source_artifacts": plain_json(
            source_artifact_mapping(source_artifacts, name="source_artifacts")
        ),
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }
    return {**identity, "seal_id": content_id(identity)}


def validate_deform360_v6_source_prediction_seal(
    value: object,
    *,
    policy: Mapping[str, Any],
    amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one held-out prediction seal."""

    payload = _mapping(value, name="prediction seal")
    require_exact_fields(payload, expected=_SEAL_FIELDS, name="prediction seal")
    if payload.get("schema") != PREDICTION_SEAL_SCHEMA:
        raise ValueError("v6 source prediction seal schema changed")
    if payload.get("information_boundary") != _PREDICTION_BOUNDARY:
        raise ValueError("v6 source prediction seal crossed its boundary")
    rebuilt = build_deform360_v6_source_prediction_seal(
        policy=policy,
        amendment=amendment,
        selection=selection,
        implementation_revision=cast(str, payload.get("implementation_revision")),
        object_id=cast(str, payload.get("object_id")),
        variants=cast(Mapping[str, Any], payload.get("variants")),
        source_artifacts=cast(Mapping[str, str], payload.get("source_artifacts")),
    )
    if payload.get("seal_id") != rebuilt["seal_id"] or plain_json(payload) != rebuilt:
        raise ValueError("v6 source prediction seal content changed")
    return rebuilt


def build_deform360_v6_source_prediction_batch(
    seals: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the complete ten-unit outcome-free source batch."""

    _, cohort = validate_deform360_v6_source_selection(selection, policy)
    if len(seals) != 10:
        raise ValueError("v6 source prediction batch must contain ten seals")
    validated = [
        validate_deform360_v6_source_prediction_seal(
            seal,
            policy=policy,
            amendment=amendment,
            selection=selection,
        )
        for seal in seals
    ]
    by_id: dict[str, dict[str, Any]] = {}
    revisions: set[str] = set()
    for seal in validated:
        object_id = seal["object_id"]
        if object_id in by_id:
            raise ValueError("v6 source prediction batch repeats a unit")
        by_id[object_id] = seal
        revisions.add(seal["implementation_revision"])
    if set(by_id) != set(cohort):
        raise ValueError("v6 source prediction batch has an incomplete unit roster")
    if len(revisions) != 1:
        raise ValueError("v6 source prediction batch mixes implementation revisions")
    ordered = [by_id[object_id] for object_id in sorted(by_id)]
    identity: dict[str, Any] = {
        "schema": PREDICTION_BATCH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "selection_artifact_sha256": selection["selection_artifact_sha256"],
        "implementation_revision": next(iter(revisions)),
        "record_count": 10,
        "records": ordered,
        "information_boundary": dict(_PREDICTION_BOUNDARY),
    }
    return {**identity, "prediction_batch_id": content_id(identity)}


def validate_deform360_v6_source_prediction_batch(
    value: object,
    *,
    policy: Mapping[str, Any],
    amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete outcome-free source prediction batch."""

    payload = _mapping(value, name="prediction batch")
    require_exact_fields(payload, expected=_BATCH_FIELDS, name="prediction batch")
    if payload.get("schema") != PREDICTION_BATCH_SCHEMA:
        raise ValueError("v6 source prediction batch schema changed")
    rebuilt = build_deform360_v6_source_prediction_batch(
        cast(Sequence[Mapping[str, Any]], payload.get("records")),
        policy=policy,
        amendment=amendment,
        selection=selection,
    )
    if (
        payload.get("prediction_batch_id") != rebuilt["prediction_batch_id"]
        or plain_json(payload) != rebuilt
    ):
        raise ValueError("v6 source prediction batch content changed")
    return rebuilt


def _variant_outcome(
    value: object,
    *,
    variant_id: str,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(value, name=variant_id)
    require_exact_fields(row, expected=_VARIANT_OUTCOME_FIELDS, name=variant_id)
    available = genuine_boolean(row.get("available"), name="available")
    if available != prediction.get("available"):
        raise ValueError("v6 source outcome availability differs from its seal")
    if row.get("prediction_artifact_id") != prediction.get("prediction_artifact_id"):
        raise ValueError("v6 source outcome artifact differs from its seal")
    return {
        "available": available,
        "prediction_artifact_id": prediction.get("prediction_artifact_id"),
        "point_loss": _finite(row.get("point_loss"), name="point_loss", minimum=0.0),
        "proper_score": _finite(row.get("proper_score"), name="proper_score"),
        "interval_covered": genuine_boolean(
            row.get("interval_covered"), name="interval_covered"
        ),
        "interval_width": _finite(
            row.get("interval_width"), name="interval_width", minimum=0.0
        ),
    }


def _outcome_variants(
    value: object,
    *,
    predictions: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    variants = _mapping(value, name="variants")
    require_exact_fields(variants, expected=frozenset(VARIANT_IDS), name="variants")
    result = {
        variant_id: _variant_outcome(
            variants[variant_id],
            variant_id=variant_id,
            prediction=_mapping(predictions[variant_id], name=variant_id),
        )
        for variant_id in VARIANT_IDS
    }
    fallback = result[B0]
    for variant_id in CHALLENGER_VARIANTS:
        prediction = _mapping(predictions[variant_id], name=variant_id)
        if prediction.get("available") is False and result[variant_id] != {
            **fallback,
            "available": False,
            "prediction_artifact_id": None,
        }:
            raise ValueError(
                "an unavailable v6 source variant must score exact fallback"
            )
    return result


def build_deform360_v6_source_outcome(
    *,
    prediction_batch: Mapping[str, Any],
    prediction_seal_id: str,
    variants: Mapping[str, Any],
    scoring_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Attach source suffix scores to one exact pre-existing prediction seal."""

    seal_id = sha256_digest(prediction_seal_id, name="prediction_seal_id")
    matches = [
        _mapping(row, name="seal")
        for row in _sequence(prediction_batch.get("records"), name="records")
        if _mapping(row, name="seal").get("seal_id") == seal_id
    ]
    if len(matches) != 1:
        raise ValueError("v6 source outcome does not bind one prediction seal")
    seal = matches[0]
    identity: dict[str, Any] = {
        "schema": SOURCE_OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "selection_artifact_sha256": prediction_batch["selection_artifact_sha256"],
        "implementation_revision": prediction_batch["implementation_revision"],
        "prediction_batch_id": prediction_batch["prediction_batch_id"],
        "prediction_seal_id": seal_id,
        "object_id": seal["object_id"],
        "variants": _outcome_variants(
            variants,
            predictions=_mapping(seal.get("variants"), name="predictions"),
        ),
        "scoring_artifacts": plain_json(
            source_artifact_mapping(scoring_artifacts, name="scoring_artifacts")
        ),
        "information_boundary": dict(_OUTCOME_BOUNDARY),
    }
    return {**identity, "outcome_id": content_id(identity)}


def validate_deform360_v6_source_outcome(
    value: object,
    *,
    prediction_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one source outcome."""

    payload = _mapping(value, name="source outcome")
    require_exact_fields(payload, expected=_OUTCOME_FIELDS, name="source outcome")
    if payload.get("schema") != SOURCE_OUTCOME_SCHEMA:
        raise ValueError("v6 source outcome schema changed")
    if payload.get("information_boundary") != _OUTCOME_BOUNDARY:
        raise ValueError("v6 source outcome crossed its information boundary")
    rebuilt = build_deform360_v6_source_outcome(
        prediction_batch=prediction_batch,
        prediction_seal_id=cast(str, payload.get("prediction_seal_id")),
        variants=cast(Mapping[str, Any], payload.get("variants")),
        scoring_artifacts=cast(Mapping[str, str], payload.get("scoring_artifacts")),
    )
    if (
        payload.get("outcome_id") != rebuilt["outcome_id"]
        or plain_json(payload) != rebuilt
    ):
        raise ValueError("v6 source outcome content changed")
    return rebuilt


def _variant_digest(records: Sequence[Mapping[str, Any]], variant_id: str) -> str:
    return content_id(
        {
            "variant_id": variant_id,
            "records": [
                {
                    "object_id": record["object_id"],
                    "prediction": record["prediction"][variant_id],
                }
                for record in records
            ],
        }
    )


def _tournament_input(
    records: Sequence[Mapping[str, Any]],
    *,
    prediction_batch_id: str,
    implementation_revision: str,
) -> dict[str, Any]:
    roster_id = content_id(
        {"source_units": sorted(cast(str, row["object_id"]) for row in records)}
    )
    candidates = []
    for variant_id in VARIANT_IDS:
        state, parameters, runtime, covariance_bytes = VARIANT_COMPLEXITY[variant_id]
        candidates.append(
            {
                "candidate_id": variant_id,
                "family": VARIANT_FAMILY[variant_id],
                "state_dimension": state,
                "parameter_count": parameters,
                "runtime_milliseconds": runtime,
                "covariance_bytes": covariance_bytes,
                "source_revision": implementation_revision,
                "configuration_sha256": content_id(
                    {
                        "policy_id": POLICY_ID,
                        "amendment_id": AMENDMENT_ID,
                        "variant_id": variant_id,
                        "covariance": VARIANT_COVARIANCE[variant_id],
                    }
                ),
                "prediction_artifact_sha256": _variant_digest(records, variant_id),
            }
        )
    tournament_records: list[dict[str, Any]] = []
    for record in records:
        fallback = record["outcome"][B0]
        for variant_id in VARIANT_IDS:
            prediction = record["prediction"][variant_id]
            outcome = record["outcome"][variant_id]
            accepted = bool(prediction["accepted"])
            deployed = outcome if accepted else fallback
            tournament_records.append(
                {
                    "candidate_id": variant_id,
                    "unit_id": record["object_id"],
                    "group_id": record["object_id"],
                    "horizon": "frames-58-76",
                    "accepted": accepted,
                    "point_loss": outcome["point_loss"],
                    "fallback_point_loss": fallback["point_loss"],
                    "deployed_point_loss": deployed["point_loss"],
                    "proper_score": outcome["proper_score"],
                    "fallback_proper_score": fallback["proper_score"],
                    "deployed_proper_score": deployed["proper_score"],
                    "interval_covered": deployed["interval_covered"],
                    "interval_width": deployed["interval_width"],
                }
            )
    return {
        "contract": DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "deform360-fresh-object-session-v6-source",
        "statistical_unit": "physical-object-session",
        "split": "source-only",
        "reference_candidate": B1,
        "physical_fallback_candidate": B0,
        "information_boundary": {
            "candidate_predictions_sealed_before_scoring": True,
            "candidate_generation_used_scored_targets": False,
            "future_observations_used": False,
            "confirmation_payloads_opened": False,
            "replacement_allowed": False,
        },
        "evaluation": {
            "evaluator_revision": implementation_revision,
            "scoring_policy_sha256": content_id(
                {"policy_id": POLICY_ID, "amendment_id": AMENDMENT_ID}
            ),
            "scored_unit_roster_sha256": roster_id,
            "physical_fallback_artifact_sha256": _variant_digest(records, B0),
            "prediction_barrier_sha256": prediction_batch_id,
            "point_loss_id": "future-held-out-view-geometry-error-v6",
            "proper_score_id": "gaussian-nll-v6",
            "interval_semantics_id": "source-only-grouped-split-conformal-90-v6",
        },
        "selection": {
            "minimum_group_count": 10,
            "minimum_relative_point_improvement": 0.02,
            "maximum_worst_group_relative_regression": 10.0,
            "maximum_harmful_accepted_count": 1,
            "maximum_mean_proper_score_regression": 0.0,
            "require_paired_point_upper_bound_nonpositive": False,
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260811,
            "require_crossfit_stability": False,
            "nominal_interval_coverage": 0.9,
            "maximum_interval_coverage_shortfall": 0.1,
            "numerical_tolerance": 1e-12,
        },
        "candidates": candidates,
        "records": tournament_records,
    }


def assemble_deform360_v6_source_evidence(
    *,
    prediction_batch: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the source tournament only after all ten predictions are sealed."""

    if len(outcomes) != 10:
        raise ValueError("v6 source evidence must contain ten outcomes")
    validated = [
        validate_deform360_v6_source_outcome(outcome, prediction_batch=prediction_batch)
        for outcome in outcomes
    ]
    by_seal: dict[str, dict[str, Any]] = {}
    for outcome in validated:
        seal_id = outcome["prediction_seal_id"]
        if seal_id in by_seal:
            raise ValueError("v6 source evidence repeats an outcome")
        by_seal[seal_id] = outcome
    records: list[dict[str, Any]] = []
    for raw in _sequence(prediction_batch.get("records"), name="records"):
        seal = _mapping(raw, name="prediction seal")
        matched_outcome = by_seal.get(cast(str, seal.get("seal_id")))
        if matched_outcome is None:
            raise ValueError("v6 source evidence omits a sealed prediction")
        records.append(
            {
                "object_id": seal["object_id"],
                "episode_id": seal["episode_id"],
                "stratum": seal["stratum"],
                "prediction_seal_id": seal["seal_id"],
                "prediction": seal["variants"],
                "outcome": matched_outcome["variants"],
            }
        )
    ordered = sorted(records, key=lambda row: row["object_id"])
    tournament = _tournament_input(
        ordered,
        prediction_batch_id=cast(str, prediction_batch["prediction_batch_id"]),
        implementation_revision=cast(str, prediction_batch["implementation_revision"]),
    )
    identity: dict[str, Any] = {
        "schema": SOURCE_EVIDENCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "selection_artifact_sha256": prediction_batch["selection_artifact_sha256"],
        "implementation_revision": prediction_batch["implementation_revision"],
        "prediction_batch_id": prediction_batch["prediction_batch_id"],
        "record_count": 10,
        "records": ordered,
        "tournament_input": tournament,
        "information_boundary": dict(_EVIDENCE_BOUNDARY),
    }
    return {**identity, "evidence_id": content_id(identity)}


def validate_deform360_v6_source_evidence(value: object) -> dict[str, Any]:
    """Validate one canonical, prediction-first source evidence artifact."""

    payload = _mapping(value, name="source evidence")
    require_exact_fields(payload, expected=_EVIDENCE_FIELDS, name="source evidence")
    if payload.get("schema") != SOURCE_EVIDENCE_SCHEMA:
        raise ValueError("v6 source evidence schema changed")
    if (
        payload.get("policy_id") != POLICY_ID
        or payload.get("amendment_id") != AMENDMENT_ID
    ):
        raise ValueError("v6 source evidence uses another policy or amendment")
    if payload.get("information_boundary") != _EVIDENCE_BOUNDARY:
        raise ValueError("v6 source evidence crossed its boundary")
    if genuine_integer(payload.get("record_count"), name="record_count") != 10:
        raise ValueError("v6 source evidence record count changed")
    records = _sequence(payload.get("records"), name="records")
    if len(records) != 10:
        raise ValueError("v6 source evidence must contain ten records")
    ids: list[str] = []
    strata: list[str] = []
    canonical_records: list[Mapping[str, Any]] = []
    for index, raw in enumerate(records):
        row = _mapping(raw, name=f"records[{index}]")
        require_exact_fields(
            row, expected=_EVIDENCE_RECORD_FIELDS, name=f"records[{index}]"
        )
        object_id = _canonical_id(row.get("object_id"), name="object_id")
        ids.append(object_id)
        stratum = _canonical_id(row.get("stratum"), name="stratum")
        if stratum not in {"sheet", "volumetric"}:
            raise ValueError("v6 source evidence stratum changed")
        strata.append(stratum)
        genuine_integer(row.get("episode_id"), name="episode_id", minimum=0)
        sha256_digest(row.get("prediction_seal_id"), name="prediction_seal_id")
        prediction = _mapping(row.get("prediction"), name="prediction")
        outcome = _mapping(row.get("outcome"), name="outcome")
        require_exact_fields(
            prediction, expected=frozenset(VARIANT_IDS), name="prediction"
        )
        require_exact_fields(outcome, expected=frozenset(VARIANT_IDS), name="outcome")
        canonical_records.append(row)
    if len(set(ids)) != 10:
        raise ValueError("v6 source evidence repeats a source unit")
    if Counter(strata) != {"sheet": 5, "volumetric": 5}:
        raise ValueError("v6 source evidence stratum counts changed")
    expected_tournament = _tournament_input(
        canonical_records,
        prediction_batch_id=cast(str, payload.get("prediction_batch_id")),
        implementation_revision=cast(str, payload.get("implementation_revision")),
    )
    if payload.get("tournament_input") != expected_tournament:
        raise ValueError("v6 source tournament input differs from sealed evidence")
    _content_identity(payload, "evidence_id", "v6 source evidence")
    # The public tournament parser independently validates its complete
    # candidate and deployed-fallback contract when the evaluator runs below.
    return plain_json(payload)


def _records_for_variant(
    tournament: Mapping[str, Any], variant_id: str
) -> list[Mapping[str, Any]]:
    return [
        _mapping(row, name="tournament record")
        for row in _sequence(tournament.get("records"), name="records")
        if _mapping(row, name="tournament record").get("candidate_id") == variant_id
    ]


def evaluate_deform360_v6_source_gate(
    evidence: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply v6 stability and stratum checks without target authorization."""

    validated = validate_deform360_v6_source_evidence(evidence)
    tournament_input = _mapping(
        validated.get("tournament_input"), name="tournament_input"
    )
    tournament_report = analyze_discrepancy_candidate_tournament(tournament_input)
    provisional = cast(str, tournament_report["provisional_selected_candidate"])
    folds = cast(list[Mapping[str, Any]], tournament_report["cross_fitted"]["folds"])
    selected_fold_count = sum(
        fold["selected_candidate"] == provisional for fold in folds
    )
    records = cast(list[Mapping[str, Any]], validated["records"])
    stratum_by_id = {
        cast(str, record["object_id"]): cast(str, record["stratum"])
        for record in records
    }
    variant_records = _records_for_variant(tournament_input, provisional)
    reference_records = {
        cast(str, row["group_id"]): row
        for row in _records_for_variant(tournament_input, B1)
    }
    nonregressing_by_stratum = {"sheet": 0, "volumetric": 0}
    nonregressing = 0
    accepted_by_stratum = {"sheet": 0, "volumetric": 0}
    accepted = 0
    for row in variant_records:
        group_id = cast(str, row["group_id"])
        stratum = stratum_by_id[group_id]
        if bool(row["accepted"]):
            accepted += 1
            accepted_by_stratum[stratum] += 1
        if float(row["deployed_point_loss"]) <= float(
            reference_records[group_id]["deployed_point_loss"]
        ):
            nonregressing += 1
            nonregressing_by_stratum[stratum] += 1
    summaries = {
        cast(str, row["candidate_id"]): row
        for row in cast(
            list[Mapping[str, Any]], tournament_report["candidate_summaries"]
        )
    }
    selected_summary = summaries[provisional]
    reference_summary = summaries[B1]
    policy_tournament = _mapping(
        policy.get("source_tournament"), name="source_tournament"
    )
    guard = _mapping(policy.get("guard_calibration"), name="guard_calibration")
    covariance = _mapping(
        policy.get("covariance_calibration"), name="covariance_calibration"
    )
    coverage = selected_summary["interval_coverage"]
    width = selected_summary["mean_interval_width"]
    reference_width = reference_summary["mean_interval_width"]
    checks = {
        "challenger_selected": provisional in CHALLENGER_VARIANTS,
        "stable_variant_selection": selected_fold_count
        >= int(policy_tournament["final_winner_must_match_at_least_outer_folds"]),
        "held_out_nonregression": nonregressing
        >= int(policy_tournament["minimum_nonregressing_held_out_units"]),
        "held_out_nonregression_per_stratum": all(
            nonregressing_by_stratum[stratum]
            >= int(
                policy_tournament["minimum_nonregressing_held_out_units_per_stratum"]
            )
            for stratum in ("sheet", "volumetric")
        ),
        "minimum_source_acceptance": accepted
        >= int(guard["minimum_accepted_source_units"]),
        "minimum_source_acceptance_per_stratum": all(
            accepted_by_stratum[stratum]
            >= int(guard["minimum_accepted_source_units_per_stratum"])
            for stratum in ("sheet", "volumetric")
        ),
        "candidate_eligible": bool(selected_summary["eligible"]),
        "coverage_in_registered_range": coverage is not None
        and float(covariance["minimum_source_object_balanced_coverage"])
        <= float(coverage)
        <= float(covariance["maximum_source_object_balanced_coverage"]),
        "interval_width_nonregression": width is not None
        and reference_width is not None
        and float(width)
        <= float(covariance["maximum_mean_full_interval_width_ratio_vs_reference"])
        * float(reference_width),
    }
    source_gate_passed = all(checks.values())
    selected_variant = provisional if source_gate_passed else B1
    descriptor: dict[str, Any] = {
        "schema": SOURCE_RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "evidence_id": validated["evidence_id"],
        "tournament_report": tournament_report,
        "provisional_selected_variant": provisional,
        "selected_variant": selected_variant,
        "selected_candidate_family": VARIANT_POLICY_CANDIDATE[selected_variant],
        "selected_covariance": VARIANT_COVARIANCE[selected_variant],
        "outer_fold_selected_count": selected_fold_count,
        "accepted_count": accepted,
        "accepted_count_by_stratum": accepted_by_stratum,
        "nonregressing_count": nonregressing,
        "nonregressing_count_by_stratum": nonregressing_by_stratum,
        "checks": checks,
        "source_gate_passed": source_gate_passed,
        "source_continuation_authorized": source_gate_passed,
        "fresh_target_selection_authorized": False,
        "fresh_target_payload_access_authorized": False,
        "claim_authorized": False,
        "status": "source-challenger-advanced"
        if source_gate_passed
        else "source-reference-retained",
        "next_stage": (
            "fit-full-source-artifacts-and-await-v5-terminal-record"
            if source_gate_passed
            else "terminate-v6-before-fresh-cohort-selection"
        ),
        "information_boundary": {
            "v5_terminal_outcome_used": False,
            "v5_confirmation_payloads_used": False,
            "v5_confirmation_outcomes_used": False,
            "v6_target_payloads_used": False,
            "v6_target_outcomes_used": False,
            "human_selection_used": False,
            "replacement_allowed": False,
        },
        "claim_boundary": (
            "This result selects or retains a source candidate only. It does not "
            "select a fresh cohort, open a v6 target payload, reinterpret v5, "
            "authorize a performance claim, or establish deployment safety."
        ),
    }
    return {"result_id": content_id(descriptor), **plain_json(descriptor)}


def validate_deform360_v6_source_result(value: object) -> dict[str, Any]:
    """Validate one source-only terminal v6 decision."""

    payload = _mapping(value, name="source result")
    if payload.get("schema") != SOURCE_RESULT_SCHEMA:
        raise ValueError("v6 source result schema changed")
    if payload.get("fresh_target_selection_authorized") is not False:
        raise ValueError("v6 source result cannot authorize target selection")
    if payload.get("fresh_target_payload_access_authorized") is not False:
        raise ValueError("v6 source result cannot authorize target access")
    if payload.get("claim_authorized") is not False:
        raise ValueError("v6 source result cannot authorize a claim")
    _content_identity(payload, "result_id", "v6 source result")
    return plain_json(payload)


def publish_deform360_v6_source_prediction_batch(
    batch: Mapping[str, Any], output_path: str | Path
) -> None:
    """Publish the outcome-free prediction batch without replacement."""

    write_atomic_json(batch, output_path, overwrite=False)


def publish_deform360_v6_source_evidence(
    evidence: Mapping[str, Any], output_path: str | Path
) -> None:
    """Publish assembled source evidence without replacement."""

    validate_deform360_v6_source_evidence(evidence)
    write_atomic_json(evidence, output_path, overwrite=False)


def publish_deform360_v6_source_result(
    result: Mapping[str, Any], output_path: str | Path
) -> None:
    """Publish the source-only terminal decision without replacement."""

    validate_deform360_v6_source_result(result)
    write_atomic_json(result, output_path, overwrite=False)


__all__ = [
    "AMENDMENT_ID",
    "B0",
    "B1",
    "CHALLENGER_VARIANTS",
    "D1_NATIVE",
    "POLICY_ID",
    "VARIANT_COVARIANCE",
    "VARIANT_FAMILY",
    "VARIANT_POLICY_CANDIDATE",
    "VARIANT_IDS",
    "VT1_OBSERVED",
    "VT1_SANDWICH",
    "VT1_WORKING",
    "assemble_deform360_v6_source_evidence",
    "build_deform360_v6_source_outcome",
    "build_deform360_v6_source_prediction_batch",
    "build_deform360_v6_source_prediction_seal",
    "evaluate_deform360_v6_source_gate",
    "load_deform360_fresh_object_session_v6_covariance_amendment",
    "load_deform360_fresh_object_session_v6_policy",
    "load_deform360_v6_source_selection",
    "publish_deform360_v6_source_evidence",
    "publish_deform360_v6_source_prediction_batch",
    "publish_deform360_v6_source_result",
    "validate_deform360_v6_source_evidence",
    "validate_deform360_v6_source_outcome",
    "validate_deform360_v6_source_prediction_batch",
    "validate_deform360_v6_source_prediction_seal",
    "validate_deform360_v6_source_result",
    "validate_deform360_v6_source_selection",
]
