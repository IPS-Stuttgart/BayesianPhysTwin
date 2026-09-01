"""Retrospective public-real-data audit of exact competence domains.

The audit composes already-published public-data results without changing any
predictor.  Decisions are derived from object/profile, action, horizon, query,
and source-competence fields before the corresponding recorded losses are read.
It is mechanism evidence, not a prospective safety certificate.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import plain_json
from ._portable_contracts import content_id, load_strict_json_object, sha256_digest
from .guard_harm_risk import one_sided_binomial_upper_bound

PUBLIC_REAL_QUERY_COMPETENCE_SCHEMA: Final = (
    "bayesian-phystwin.public-real-query-competence-retrospective"
)
PUBLIC_REAL_QUERY_COMPETENCE_VERSION: Final = 1
EXACT_CONTEXT_SUPPORT_RULE: Final = (
    "exact-object-profile-action-family-horizon-query-runtime-support-v1"
)
CLAIM_BOUNDARY: Final = (
    "Retrospective mechanism evidence on already-published public real-world "
    "measurements. It tests whether exact context support plus exact fallback "
    "separates useful from unsupported simulator use. It is not a fresh "
    "prospective certificate, a universal safety guarantee, an official "
    "benchmark claim, or state of the art."
)

_DEFORM360_SCHEMA = "bayesian-phystwin/deform360-untouched-confirmation-result-v5"
_POKEFLEX_KIND = "PokeFlexIndependentDepthRegretGuardProspectiveEvaluation"
_TRACKING_SOURCE_QUERY = "free-marker-euclidean-trajectory-rmse-mm"
_DEFORM360_SOURCE_QUERY = "active-field-rmse"
_DEFORM360_QUERY_MISMATCH = "all-field-mae"


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without normalizing its bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(path: str | Path, expected_sha256: object, *, name: str) -> Path:
    resolved = Path(path)
    expected = sha256_digest(expected_sha256, name=f"{name}_sha256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 changed: expected {expected}, got {actual}")
    return resolved


def action_family_v1(action: object) -> str:
    """Return the frozen Deform360 action-family label."""

    token = "" if action is None else str(action).strip().lower()
    if any(word in token for word in ("lift", "raise")):
        return "lift"
    if any(word in token for word in ("move", "drag", "push", "pull")):
        return "translate"
    if any(word in token for word in ("fold", "curl", "curve", "twist")):
        return "shape"
    if any(word in token for word in ("squeeze", "press", "compress")):
        return "compress"
    if any(word in token for word in ("wave", "shake")):
        return "dynamic"
    return "other"


def _bootstrap_mean_difference(
    differences: NDArray[np.float64],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    values: NDArray[np.float64] = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap differences must be a nonempty finite vector")
    if replicates < 1 or seed < 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    means: NDArray[np.float64] = np.empty(replicates, dtype=np.float64)
    chunk_size = 4096
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = np.mean(values[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    interval = cast(
        NDArray[np.float64],
        np.quantile(means, [alpha, 1.0 - alpha], method="linear"),
    )
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "observed_mean_difference": float(np.mean(values)),
        "interval": [float(interval[0]), float(interval[1])],
        "probability_difference_below_zero": float(np.mean(means < 0.0)),
    }


def _policy_audit(
    *,
    group_ids: Sequence[str],
    candidate_losses: NDArray[np.float64],
    fallback_losses: NDArray[np.float64],
    accepted_mask: NDArray[np.bool_],
    harm_margin: float,
    confidence: float,
    target_harm_probability: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    candidate: NDArray[np.float64] = np.asarray(candidate_losses, dtype=np.float64)
    fallback: NDArray[np.float64] = np.asarray(fallback_losses, dtype=np.float64)
    accepted: NDArray[np.bool_] = np.asarray(accepted_mask, dtype=np.bool_)
    if (
        candidate.shape != fallback.shape
        or candidate.shape != accepted.shape
        or candidate.ndim != 1
        or len(candidate) != len(group_ids)
        or not len(candidate)
    ):
        raise ValueError("policy arrays must be aligned nonempty vectors")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(fallback)):
        raise ValueError("policy losses must be finite")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("policy group IDs must be unique")
    harmful = candidate > fallback + harm_margin
    accepted_count = int(np.sum(accepted))
    harmful_count = int(np.sum(accepted & harmful))
    upper = one_sided_binomial_upper_bound(
        harmful_count,
        accepted_count,
        confidence,
    )
    deployed = np.where(accepted, candidate, fallback)
    differences = deployed - fallback
    fallback_mean = float(np.mean(fallback))
    candidate_mean = float(np.mean(candidate))
    deployed_mean = float(np.mean(deployed))
    return {
        "group_count": len(group_ids),
        "accepted_count": accepted_count,
        "acceptance_fraction": accepted_count / len(group_ids),
        "candidate_wins_ties_losses": [
            int(np.sum(candidate < fallback)),
            int(np.sum(candidate == fallback)),
            int(np.sum(candidate > fallback)),
        ],
        "accepted_wins_ties_losses": [
            int(np.sum(accepted & (candidate < fallback))),
            int(np.sum(accepted & (candidate == fallback))),
            int(np.sum(accepted & (candidate > fallback))),
        ],
        "harm_margin": harm_margin,
        "harmful_accepted_count": harmful_count,
        "observed_harm_fraction": (
            None if accepted_count == 0 else harmful_count / accepted_count
        ),
        "one_sided_clopper_pearson_upper": upper,
        "target_harm_probability": target_harm_probability,
        "retrospective_harm_gate_passed": (
            accepted_count > 0 and upper <= target_harm_probability
        ),
        "fallback_mean_loss": fallback_mean,
        "always_candidate_mean_loss": candidate_mean,
        "deployed_mean_loss": deployed_mean,
        "always_candidate_relative_change": candidate_mean / fallback_mean - 1.0,
        "deployed_relative_change": deployed_mean / fallback_mean - 1.0,
        "exact_fallback_count": int(np.sum(~accepted)),
        "exact_fallback_fraction": float(np.mean(~accepted)),
        "paired_group_bootstrap": _bootstrap_mean_difference(
            differences,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence=confidence,
        ),
    }


def evaluate_deform360_action_support_v1(
    result: Mapping[str, object],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    target_harm_probability: float,
) -> dict[str, object]:
    """Audit exact action/query support on the published 92-object result."""

    if result.get("schema") != _DEFORM360_SCHEMA or result.get("schema_version") != 5:
        raise ValueError("unexpected Deform360 result schema")
    if result.get("status") != "complete":
        raise ValueError("Deform360 result is not complete")
    raw_objects = _require_sequence(result.get("objects"), name="objects")
    rows = sorted(
        (_require_mapping(value, name="object") for value in raw_objects),
        key=lambda value: _canonical_text(value.get("object_id"), name="object_id"),
    )
    if len(rows) != 92:
        raise ValueError("registered Deform360 cohort must contain 92 objects")
    group_ids: list[str] = []
    candidate_active: list[float] = []
    fallback_active: list[float] = []
    candidate_mae: list[float] = []
    fallback_mae: list[float] = []
    exact_support: list[bool] = []
    legacy_guard: list[bool] = []
    source_only_improvement: list[bool] = []
    support_rows: list[dict[str, object]] = []
    for row in rows:
        object_id = _canonical_text(row.get("object_id"), name="object_id")
        if object_id in group_ids:
            raise ValueError("duplicate Deform360 object")
        group_ids.append(object_id)
        target_family = _canonical_text(
            row.get("target_action_family"),
            name="target_action_family",
        )
        source_actions = _require_sequence(
            row.get("source_actions"),
            name="source_actions",
        )
        source_families = sorted(
            {action_family_v1(action) for action in source_actions}
        )
        family_supported = target_family in source_families
        source_cv = _require_mapping(
            row.get("source_cv_active_rmse"),
            name="source_cv_active_rmse",
        )
        source_candidate = _finite(
            source_cv.get("bayesian_action_ensemble"),
            name="source candidate RMSE",
            minimum=0.0,
        )
        source_fallback = _finite(
            source_cv.get("persistence"),
            name="source fallback RMSE",
            minimum=0.0,
        )
        source_improves = source_candidate <= source_fallback
        metrics = _require_mapping(row.get("metrics"), name="metrics")
        candidate_metrics = _require_mapping(
            metrics.get("bayesian_action_ensemble"),
            name="candidate metrics",
        )
        fallback_metrics = _require_mapping(
            metrics.get("persistence"),
            name="fallback metrics",
        )
        candidate_active.append(
            _finite(
                candidate_metrics.get("active_field_rmse"),
                name="candidate active RMSE",
                minimum=0.0,
            )
        )
        fallback_active.append(
            _finite(
                fallback_metrics.get("active_field_rmse"),
                name="fallback active RMSE",
                minimum=0.0,
            )
        )
        candidate_mae.append(
            _finite(
                candidate_metrics.get("field_mae"),
                name="candidate field MAE",
                minimum=0.0,
            )
        )
        fallback_mae.append(
            _finite(
                fallback_metrics.get("field_mae"),
                name="fallback field MAE",
                minimum=0.0,
            )
        )
        support = family_supported and source_improves
        exact_support.append(support)
        source_only_improvement.append(source_improves)
        if type(row.get("guard_accepts")) is not bool:
            raise ValueError("guard_accepts must be boolean")
        legacy_guard.append(bool(row["guard_accepts"]))
        support_rows.append(
            {
                "object_id": object_id,
                "target_action_family": target_family,
                "source_action_families": source_families,
                "action_family_supported": family_supported,
                "source_candidate_nonregressing": source_improves,
                "exact_context_supported": support,
            }
        )

    candidate_active_array = np.asarray(candidate_active, dtype=np.float64)
    fallback_active_array = np.asarray(fallback_active, dtype=np.float64)
    exact_support_array = np.asarray(exact_support, dtype=np.bool_)
    always: NDArray[np.bool_] = np.ones(len(rows), dtype=np.bool_)
    legacy = np.asarray(legacy_guard, dtype=np.bool_)
    source_only = np.asarray(source_only_improvement, dtype=np.bool_)
    exact_audit = _policy_audit(
        group_ids=group_ids,
        candidate_losses=candidate_active_array,
        fallback_losses=fallback_active_array,
        accepted_mask=exact_support_array,
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    always_audit = _policy_audit(
        group_ids=group_ids,
        candidate_losses=candidate_active_array,
        fallback_losses=fallback_active_array,
        accepted_mask=always,
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    source_only_audit = _policy_audit(
        group_ids=group_ids,
        candidate_losses=candidate_active_array,
        fallback_losses=fallback_active_array,
        accepted_mask=source_only,
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    legacy_audit = _policy_audit(
        group_ids=group_ids,
        candidate_losses=candidate_active_array,
        fallback_losses=fallback_active_array,
        accepted_mask=legacy,
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    candidate_gain = _finite(
        always_audit.get("fallback_mean_loss"), name="always fallback mean"
    ) - _finite(
        always_audit.get("always_candidate_mean_loss"),
        name="always candidate mean",
    )
    deployed_gain = _finite(
        exact_audit.get("fallback_mean_loss"), name="exact fallback mean"
    ) - _finite(exact_audit.get("deployed_mean_loss"), name="exact deployed mean")
    if candidate_gain <= 0.0:
        raise ValueError(
            "published Deform360 candidate must improve the registered query"
        )
    query_mismatch = _policy_audit(
        group_ids=group_ids,
        candidate_losses=np.asarray(candidate_mae, dtype=np.float64),
        fallback_losses=np.asarray(fallback_mae, dtype=np.float64),
        accepted_mask=np.zeros(len(rows), dtype=np.bool_),
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    active_candidate_mean = float(np.mean(candidate_active_array))
    active_fallback_mean = float(np.mean(fallback_active_array))
    mae_candidate_mean = float(np.mean(np.asarray(candidate_mae, dtype=np.float64)))
    mae_fallback_mean = float(np.mean(np.asarray(fallback_mae, dtype=np.float64)))
    query_rank_reversal = (
        active_candidate_mean < active_fallback_mean
        and mae_candidate_mean > mae_fallback_mean
    )
    return {
        "dataset": "Deform360",
        "public_real_world": True,
        "statistical_unit": "physical object",
        "group_count": len(rows),
        "candidate": "bayesian_action_ensemble",
        "fallback": "persistence",
        "registered_query": _DEFORM360_SOURCE_QUERY,
        "exact_context_rule": EXACT_CONTEXT_SUPPORT_RULE,
        "exact_context_policy": exact_audit,
        "ablations": {
            "always_candidate": always_audit,
            "source_nonregression_without_action_support": source_only_audit,
            "legacy_source_guard_with_exact_persistence_fallback": legacy_audit,
        },
        "gain_retained_fraction": deployed_gain / candidate_gain,
        "query_mismatch_control": {
            "query": _DEFORM360_QUERY_MISMATCH,
            "reason": "query-functional-out-of-scope",
            "policy": query_mismatch,
        },
        "query_rank_reversal": {
            "same_prediction_pair": True,
            "registered_query": _DEFORM360_SOURCE_QUERY,
            "registered_query_candidate_mean": active_candidate_mean,
            "registered_query_fallback_mean": active_fallback_mean,
            "registered_query_candidate_better": (
                active_candidate_mean < active_fallback_mean
            ),
            "alternate_query": _DEFORM360_QUERY_MISMATCH,
            "alternate_query_candidate_mean": mae_candidate_mean,
            "alternate_query_fallback_mean": mae_fallback_mean,
            "alternate_query_candidate_worse": mae_candidate_mean > mae_fallback_mean,
            "rank_reversal_observed": query_rank_reversal,
            "query_independent_routing_is_sufficient": not query_rank_reversal,
        },
        "support_rows": support_rows,
    }


def _read_specimen_scores(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "specimen",
            "arm",
            "rmse_mm",
            "mean_marker_error_mm",
            "coordinate_nll",
            "coordinate_90_coverage",
            "mean_full_90_width_mm",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("Tracking Cloth specimen score columns changed")
        for raw in reader:
            specimen = _canonical_text(raw["specimen"], name="specimen")
            arm = _canonical_text(raw["arm"], name="arm")
            try:
                parsed_rmse = float(raw["rmse_mm"])
            except (TypeError, ValueError) as error:
                raise ValueError("rmse_mm must be numeric") from error
            value = _finite(parsed_rmse, name="rmse_mm", minimum=0.0)
            if arm in rows.setdefault(specimen, {}):
                raise ValueError("duplicate specimen/arm score")
            rows[specimen][arm] = value
    return rows


def evaluate_tracking_cloth_action_support_v1(
    protocol: Mapping[str, object],
    metrics: Mapping[str, object],
    specimen_scores_path: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    target_harm_probability: float,
) -> dict[str, object]:
    """Apply the exact action-domain rule to the public shake-to-twist pilot."""

    source_motion = _canonical_text(protocol.get("source_motion"), name="source_motion")
    target_motion = _canonical_text(protocol.get("target_motion"), name="target_motion")
    rows = _read_specimen_scores(specimen_scores_path)
    if len(rows) != 8:
        raise ValueError("Tracking Cloth must contain eight material-size specimens")
    group_ids = sorted(rows)
    candidate = np.asarray(
        [rows[group]["bayesian_physics"] for group in group_ids],
        dtype=np.float64,
    )
    fallback = np.asarray(
        [rows[group]["persistence"] for group in group_ids],
        dtype=np.float64,
    )
    action_supported = source_motion == target_motion
    exact_support: NDArray[np.bool_] = np.full(
        len(group_ids), action_supported, dtype=np.bool_
    )
    exact_audit = _policy_audit(
        group_ids=group_ids,
        candidate_losses=candidate,
        fallback_losses=fallback,
        accepted_mask=exact_support,
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    always_audit = _policy_audit(
        group_ids=group_ids,
        candidate_losses=candidate,
        fallback_losses=fallback,
        accepted_mask=np.ones(len(group_ids), dtype=np.bool_),
        harm_margin=0.0,
        confidence=confidence,
        target_harm_probability=target_harm_probability,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    arms = _require_mapping(metrics.get("arms"), name="tracking metrics arms")
    guarded = _require_mapping(
        arms.get("guarded_bayesian_physics"),
        name="guarded_bayesian_physics metrics",
    )
    guarded_rmse = _finite(
        guarded.get("rmse_mm"),
        name="guarded_bayesian_physics rmse_mm",
        minimum=0.0,
    )
    fallback_mean = _finite(
        exact_audit.get("fallback_mean_loss"), name="tracking fallback mean"
    )
    return {
        "dataset": "Tracking Cloth Deformation",
        "public_real_world": True,
        "statistical_unit": "material-size specimen",
        "group_count": len(group_ids),
        "candidate": "bayesian_physics",
        "fallback": "persistence",
        "registered_query": _TRACKING_SOURCE_QUERY,
        "source_action_domain": source_motion,
        "target_action_domain": target_motion,
        "exact_context_rule": EXACT_CONTEXT_SUPPORT_RULE,
        "exact_context_policy": exact_audit,
        "always_candidate": always_audit,
        "legacy_guarded_candidate_mean_rmse_mm": guarded_rmse,
        "legacy_guarded_relative_change_to_persistence": (
            guarded_rmse / fallback_mean - 1.0
        ),
        "decision": "exact-fallback-action-domain-out-of-scope",
    }


def evaluate_pokeflex_profile_support_v1(
    same_profile: Mapping[str, object],
    independent_object: Mapping[str, object],
) -> dict[str, object]:
    """Summarize the published same-profile win and profile-shift failure."""

    for name, value in (
        ("same_profile", same_profile),
        ("independent_object", independent_object),
    ):
        if value.get("artifact_kind") != _POKEFLEX_KIND:
            raise ValueError(f"{name} PokeFlex artifact kind changed")
        if value.get("schema_version") != 1:
            raise ValueError(f"{name} PokeFlex schema version changed")
    same_objects = {
        _canonical_text(row.get("object"), name="same-profile object")
        for row in (
            _require_mapping(value, name="same-profile object")
            for value in _require_sequence(same_profile.get("objects"), name="objects")
        )
    }
    shifted_objects = {
        _canonical_text(row.get("object"), name="independent object")
        for row in (
            _require_mapping(value, name="independent object")
            for value in _require_sequence(
                independent_object.get("objects"),
                name="objects",
            )
        )
    }
    if same_objects & shifted_objects:
        raise ValueError("PokeFlex profile-shift objects overlap source profiles")
    if same_profile.get("gate_passed") is not True:
        raise ValueError("same-profile PokeFlex gate must have passed")
    if independent_object.get("gate_passed") is not False:
        raise ValueError("independent-object PokeFlex gate must have failed")
    same_baseline = _finite(
        same_profile.get("baseline_object_mean_CD_UL1_mm"),
        name="same-profile baseline",
        minimum=0.0,
    )
    same_selected = _finite(
        same_profile.get("selected_object_mean_CD_UL1_mm"),
        name="same-profile selected",
        minimum=0.0,
    )
    shifted_baseline = _finite(
        independent_object.get("baseline_object_mean_CD_UL1_mm"),
        name="independent-object baseline",
        minimum=0.0,
    )
    shifted_selected = _finite(
        independent_object.get("selected_object_mean_CD_UL1_mm"),
        name="independent-object selected",
        minimum=0.0,
    )
    return {
        "dataset": "PokeFlex",
        "public_real_world": True,
        "registered_query": "CD_UL1_mm",
        "same_profile_replication": {
            "object_count": _integer(
                same_profile.get("object_count"),
                name="same-profile object_count",
                minimum=1,
            ),
            "take_count": _integer(
                same_profile.get("take_count"),
                name="same-profile take_count",
                minimum=1,
            ),
            "baseline_mean_CD_UL1_mm": same_baseline,
            "selected_mean_CD_UL1_mm": same_selected,
            "relative_change": same_selected / same_baseline - 1.0,
            "object_wins": _integer(
                same_profile.get("object_wins"),
                name="same-profile object_wins",
            ),
            "object_losses": _integer(
                same_profile.get("object_losses"),
                name="same-profile object_losses",
            ),
            "decision": "registered-profile-policy-retained",
        },
        "independent_object_stress": {
            "object_count": _integer(
                independent_object.get("object_count"),
                name="independent-object object_count",
                minimum=1,
            ),
            "original_guard_baseline_mean_CD_UL1_mm": shifted_baseline,
            "original_guard_selected_mean_CD_UL1_mm": shifted_selected,
            "original_guard_relative_change": (
                shifted_selected / shifted_baseline - 1.0
            ),
            "original_guard_false_safe_rate": _finite(
                independent_object.get("false_safe_rate"),
                name="false_safe_rate",
                minimum=0.0,
            ),
            "exact_profile_policy_selected_mean_CD_UL1_mm": shifted_baseline,
            "exact_profile_policy_relative_change": 0.0,
            "avoided_mean_regression_mm": shifted_selected - shifted_baseline,
            "decision": "exact-fallback-object-profile-out-of-scope",
        },
    }


def build_public_real_query_competence_evidence_v1(
    *,
    protocol: Mapping[str, object],
    deform360_result_path: str | Path,
    tracking_protocol_path: str | Path,
    tracking_metrics_path: str | Path,
    tracking_specimen_scores_path: str | Path,
    pokeflex_same_profile_path: str | Path,
    pokeflex_independent_object_path: str | Path,
) -> dict[str, object]:
    """Build the hash-bound compact public-real-data evidence record."""

    if (
        protocol.get("schema")
        != "bayesian-phystwin.public-real-query-competence-protocol"
    ):
        raise ValueError("unexpected public-real query protocol schema")
    if protocol.get("schema_version") != PUBLIC_REAL_QUERY_COMPETENCE_VERSION:
        raise ValueError("unexpected public-real query protocol version")
    inputs = _require_mapping(protocol.get("inputs"), name="protocol inputs")
    statistics = _require_mapping(protocol.get("statistics"), name="statistics")
    bootstrap_replicates = _integer(
        statistics.get("bootstrap_replicates"),
        name="bootstrap_replicates",
        minimum=1,
    )
    bootstrap_seed = _integer(
        statistics.get("bootstrap_seed"),
        name="bootstrap_seed",
    )
    confidence = _finite(statistics.get("confidence"), name="confidence")
    target_harm_probability = _finite(
        statistics.get("target_harm_probability"),
        name="target_harm_probability",
    )
    if not 0.0 < confidence < 1.0 or not 0.0 < target_harm_probability < 1.0:
        raise ValueError("confidence and target harm probability must lie in (0, 1)")

    paths = {
        "deform360_result": Path(deform360_result_path),
        "tracking_protocol": Path(tracking_protocol_path),
        "tracking_metrics": Path(tracking_metrics_path),
        "tracking_specimen_scores": Path(tracking_specimen_scores_path),
        "pokeflex_same_profile": Path(pokeflex_same_profile_path),
        "pokeflex_independent_object": Path(pokeflex_independent_object_path),
    }
    verified_hashes: dict[str, str] = {}
    for name, path in paths.items():
        input_spec = _require_mapping(inputs.get(name), name=f"inputs.{name}")
        _verified_file(path, input_spec.get("sha256"), name=name)
        verified_hashes[name] = sha256_file(path)

    deform360 = load_strict_json_object(
        paths["deform360_result"], label="Deform360 result"
    )
    tracking_protocol = load_strict_json_object(
        paths["tracking_protocol"],
        label="Tracking Cloth protocol",
    )
    tracking_metrics = load_strict_json_object(
        paths["tracking_metrics"],
        label="Tracking Cloth metrics",
    )
    pokeflex_same = load_strict_json_object(
        paths["pokeflex_same_profile"],
        label="PokeFlex same-profile result",
    )
    pokeflex_shift = load_strict_json_object(
        paths["pokeflex_independent_object"],
        label="PokeFlex independent-object result",
    )
    evidence: dict[str, object] = {
        "schema": PUBLIC_REAL_QUERY_COMPETENCE_SCHEMA,
        "schema_version": PUBLIC_REAL_QUERY_COMPETENCE_VERSION,
        "status": "complete",
        "evidence_class": "retrospective-public-real-data-mechanism-evidence",
        "claim_boundary": CLAIM_BOUNDARY,
        "exact_context_support_rule": EXACT_CONTEXT_SUPPORT_RULE,
        "protocol_id": content_id(plain_json(protocol)),
        "input_sha256": verified_hashes,
        "deform360": evaluate_deform360_action_support_v1(
            deform360,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
            target_harm_probability=target_harm_probability,
        ),
        "tracking_cloth": evaluate_tracking_cloth_action_support_v1(
            tracking_protocol,
            tracking_metrics,
            paths["tracking_specimen_scores"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
            target_harm_probability=target_harm_probability,
        ),
        "pokeflex": evaluate_pokeflex_profile_support_v1(
            pokeflex_same,
            pokeflex_shift,
        ),
        "information_boundary": {
            "new_measurements_collected": False,
            "raw_media_opened_by_this_audit": False,
            "held_v8_opened": False,
            "dlo4_dlo5_opened": False,
            "protected_target_opened": False,
            "published_aggregate_outcomes_reused": True,
            "backend_predictions_changed": False,
        },
        "interpretation": {
            "exact_context_support_has_real_data_decision_value": True,
            "global_simulator_ranking_is_sufficient": False,
            "prospective_harm_certificate_supported": False,
            "universal_safety_supported": False,
            "state_of_the_art_supported": False,
        },
    }
    evidence["artifact_id"] = content_id(evidence)
    return evidence


__all__ = [
    "CLAIM_BOUNDARY",
    "EXACT_CONTEXT_SUPPORT_RULE",
    "PUBLIC_REAL_QUERY_COMPETENCE_SCHEMA",
    "PUBLIC_REAL_QUERY_COMPETENCE_VERSION",
    "action_family_v1",
    "build_public_real_query_competence_evidence_v1",
    "evaluate_deform360_action_support_v1",
    "evaluate_pokeflex_profile_support_v1",
    "evaluate_tracking_cloth_action_support_v1",
    "sha256_file",
]
