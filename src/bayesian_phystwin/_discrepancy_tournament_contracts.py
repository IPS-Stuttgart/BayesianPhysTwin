"""Contracts and validation for source-only discrepancy tournaments."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from ._canonical_contracts import literal_lower_hex

DISCREPANCY_TOURNAMENT_INPUT_CONTRACT: Final = (
    "bayesian-phystwin-discrepancy-candidate-tournament-v1"
)
DISCREPANCY_TOURNAMENT_REPORT_CONTRACT: Final = (
    "bayesian-phystwin-discrepancy-candidate-tournament-report-v1"
)
DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY: Final = (
    "Source-only method selection. A passing result may freeze one candidate for "
    "a separately registered prospective experiment, but it does not establish "
    "fresh-object transfer, calibrated deployment uncertainty, Causal4D benefit, "
    "deployment safety, or state of the art."
)
DEFAULT_MAXIMUM_INPUT_BYTES: Final = 64 * 1024 * 1024
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\Z")


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _identifier(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase identifier")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _optional_coverage(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    result = _number(value, name=name, minimum=0.0, maximum=1.0)
    if result in {0.0, 1.0}:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return result


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Frozen identity and implementation-complexity metadata for one candidate."""

    candidate_id: str
    family: str
    state_dimension: int
    parameter_count: int
    runtime_milliseconds: float
    covariance_bytes: int
    source_revision: str
    configuration_sha256: str
    prediction_artifact_sha256: str

    @property
    def complexity_key(self) -> tuple[int, int, float, int, str]:
        return (
            self.state_dimension,
            self.parameter_count,
            self.runtime_milliseconds,
            self.covariance_bytes,
            self.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class TournamentEvaluationSpec:
    """Common evaluator, roster, fallback, and prediction-barrier identities."""

    evaluator_revision: str
    scoring_policy_sha256: str
    scored_unit_roster_sha256: str
    physical_fallback_artifact_sha256: str
    prediction_barrier_sha256: str
    point_loss_id: str
    proper_score_id: str
    interval_semantics_id: str


@dataclass(frozen=True, slots=True)
class TournamentSelectionConfig:
    """Frozen whole-group gates and deterministic candidate preference."""

    minimum_group_count: int
    minimum_relative_point_improvement: float
    maximum_worst_group_relative_regression: float
    maximum_harmful_accepted_count: int
    maximum_mean_proper_score_regression: float
    require_paired_point_upper_bound_nonpositive: bool
    bootstrap_samples: int
    bootstrap_seed: int
    require_crossfit_stability: bool
    nominal_interval_coverage: float | None
    maximum_interval_coverage_shortfall: float
    numerical_tolerance: float


@dataclass(frozen=True, slots=True)
class TournamentRecord:
    """One matched candidate result for one scored unit."""

    candidate_id: str
    unit_id: str
    group_id: str
    horizon: str
    accepted: bool
    point_loss: float
    fallback_point_loss: float
    deployed_point_loss: float
    proper_score: float
    fallback_proper_score: float
    deployed_proper_score: float
    interval_covered: bool | None
    interval_width: float | None


@dataclass(frozen=True, slots=True)
class TournamentEvidence:
    """Validated source-only tournament evidence and frozen selection policy."""

    protocol_id: str
    statistical_unit: str
    reference_candidate: str
    physical_fallback_candidate: str
    evaluation: TournamentEvaluationSpec
    candidates: tuple[CandidateSpec, ...]
    config: TournamentSelectionConfig
    records: tuple[TournamentRecord, ...]


def _parse_candidate(value: object, *, index: int) -> CandidateSpec:
    name = f"candidates[{index}]"
    payload = _mapping(value, name=name)
    expected = {
        "candidate_id",
        "family",
        "state_dimension",
        "parameter_count",
        "runtime_milliseconds",
        "covariance_bytes",
        "source_revision",
        "configuration_sha256",
        "prediction_artifact_sha256",
    }
    if set(payload) != expected:
        raise ValueError(f"{name} fields changed")
    return CandidateSpec(
        candidate_id=_identifier(payload["candidate_id"], name=f"{name}.candidate_id"),
        family=_identifier(payload["family"], name=f"{name}.family"),
        state_dimension=_integer(
            payload["state_dimension"], name=f"{name}.state_dimension"
        ),
        parameter_count=_integer(
            payload["parameter_count"], name=f"{name}.parameter_count"
        ),
        runtime_milliseconds=_number(
            payload["runtime_milliseconds"],
            name=f"{name}.runtime_milliseconds",
            minimum=0.0,
        ),
        covariance_bytes=_integer(
            payload["covariance_bytes"], name=f"{name}.covariance_bytes"
        ),
        source_revision=literal_lower_hex(
            payload["source_revision"],
            name=f"{name}.source_revision",
            lengths={40},
        ),
        configuration_sha256=literal_lower_hex(
            payload["configuration_sha256"],
            name=f"{name}.configuration_sha256",
            lengths={64},
        ),
        prediction_artifact_sha256=literal_lower_hex(
            payload["prediction_artifact_sha256"],
            name=f"{name}.prediction_artifact_sha256",
            lengths={64},
        ),
    )


def _parse_evaluation(value: object) -> TournamentEvaluationSpec:
    payload = _mapping(value, name="evaluation")
    expected = {
        "evaluator_revision",
        "scoring_policy_sha256",
        "scored_unit_roster_sha256",
        "physical_fallback_artifact_sha256",
        "prediction_barrier_sha256",
        "point_loss_id",
        "proper_score_id",
        "interval_semantics_id",
    }
    if set(payload) != expected:
        raise ValueError("evaluation fields changed")
    return TournamentEvaluationSpec(
        evaluator_revision=literal_lower_hex(
            payload["evaluator_revision"],
            name="evaluation.evaluator_revision",
            lengths={40},
        ),
        scoring_policy_sha256=literal_lower_hex(
            payload["scoring_policy_sha256"],
            name="evaluation.scoring_policy_sha256",
            lengths={64},
        ),
        scored_unit_roster_sha256=literal_lower_hex(
            payload["scored_unit_roster_sha256"],
            name="evaluation.scored_unit_roster_sha256",
            lengths={64},
        ),
        physical_fallback_artifact_sha256=literal_lower_hex(
            payload["physical_fallback_artifact_sha256"],
            name="evaluation.physical_fallback_artifact_sha256",
            lengths={64},
        ),
        prediction_barrier_sha256=literal_lower_hex(
            payload["prediction_barrier_sha256"],
            name="evaluation.prediction_barrier_sha256",
            lengths={64},
        ),
        point_loss_id=_identifier(
            payload["point_loss_id"], name="evaluation.point_loss_id"
        ),
        proper_score_id=_identifier(
            payload["proper_score_id"], name="evaluation.proper_score_id"
        ),
        interval_semantics_id=_identifier(
            payload["interval_semantics_id"],
            name="evaluation.interval_semantics_id",
        ),
    )


def _parse_selection(value: object) -> TournamentSelectionConfig:
    payload = _mapping(value, name="selection")
    expected = {
        "minimum_group_count",
        "minimum_relative_point_improvement",
        "maximum_worst_group_relative_regression",
        "maximum_harmful_accepted_count",
        "maximum_mean_proper_score_regression",
        "require_paired_point_upper_bound_nonpositive",
        "bootstrap_samples",
        "bootstrap_seed",
        "require_crossfit_stability",
        "nominal_interval_coverage",
        "maximum_interval_coverage_shortfall",
        "numerical_tolerance",
    }
    if set(payload) != expected:
        raise ValueError("selection fields changed")
    config = TournamentSelectionConfig(
        minimum_group_count=_integer(
            payload["minimum_group_count"],
            name="selection.minimum_group_count",
            minimum=3,
        ),
        minimum_relative_point_improvement=_number(
            payload["minimum_relative_point_improvement"],
            name="selection.minimum_relative_point_improvement",
            minimum=0.0,
        ),
        maximum_worst_group_relative_regression=_number(
            payload["maximum_worst_group_relative_regression"],
            name="selection.maximum_worst_group_relative_regression",
            minimum=0.0,
        ),
        maximum_harmful_accepted_count=_integer(
            payload["maximum_harmful_accepted_count"],
            name="selection.maximum_harmful_accepted_count",
        ),
        maximum_mean_proper_score_regression=_number(
            payload["maximum_mean_proper_score_regression"],
            name="selection.maximum_mean_proper_score_regression",
            minimum=0.0,
        ),
        require_paired_point_upper_bound_nonpositive=_boolean(
            payload["require_paired_point_upper_bound_nonpositive"],
            name="selection.require_paired_point_upper_bound_nonpositive",
        ),
        bootstrap_samples=_integer(
            payload["bootstrap_samples"],
            name="selection.bootstrap_samples",
            minimum=100,
        ),
        bootstrap_seed=_integer(
            payload["bootstrap_seed"], name="selection.bootstrap_seed"
        ),
        require_crossfit_stability=_boolean(
            payload["require_crossfit_stability"],
            name="selection.require_crossfit_stability",
        ),
        nominal_interval_coverage=_optional_coverage(
            payload["nominal_interval_coverage"],
            name="selection.nominal_interval_coverage",
        ),
        maximum_interval_coverage_shortfall=_number(
            payload["maximum_interval_coverage_shortfall"],
            name="selection.maximum_interval_coverage_shortfall",
            minimum=0.0,
            maximum=1.0,
        ),
        numerical_tolerance=_number(
            payload["numerical_tolerance"],
            name="selection.numerical_tolerance",
            minimum=0.0,
        ),
    )
    if config.require_crossfit_stability:
        _require(
            config.minimum_group_count >= 3,
            "cross-fitted selection requires at least three training groups",
        )
    return config


def _parse_record(value: object, *, index: int) -> TournamentRecord:
    name = f"records[{index}]"
    payload = _mapping(value, name=name)
    expected = {
        "candidate_id",
        "unit_id",
        "group_id",
        "horizon",
        "accepted",
        "point_loss",
        "fallback_point_loss",
        "deployed_point_loss",
        "proper_score",
        "fallback_proper_score",
        "deployed_proper_score",
        "interval_covered",
        "interval_width",
    }
    if set(payload) != expected:
        raise ValueError(f"{name} fields changed")
    covered_value = payload["interval_covered"]
    width_value = payload["interval_width"]
    if covered_value is None and width_value is None:
        covered = None
        width = None
    elif covered_value is not None and width_value is not None:
        covered = _boolean(covered_value, name=f"{name}.interval_covered")
        width = _number(width_value, name=f"{name}.interval_width", minimum=0.0)
    else:
        raise ValueError(f"{name} must provide both interval fields or neither")
    accepted = _boolean(payload["accepted"], name=f"{name}.accepted")
    point_loss = _number(payload["point_loss"], name=f"{name}.point_loss", minimum=0.0)
    fallback_point_loss = _number(
        payload["fallback_point_loss"],
        name=f"{name}.fallback_point_loss",
        minimum=0.0,
    )
    deployed_point_loss = _number(
        payload["deployed_point_loss"],
        name=f"{name}.deployed_point_loss",
        minimum=0.0,
    )
    proper_score = _number(payload["proper_score"], name=f"{name}.proper_score")
    fallback_proper_score = _number(
        payload["fallback_proper_score"], name=f"{name}.fallback_proper_score"
    )
    deployed_proper_score = _number(
        payload["deployed_proper_score"], name=f"{name}.deployed_proper_score"
    )
    expected_point = point_loss if accepted else fallback_point_loss
    expected_proper = proper_score if accepted else fallback_proper_score
    if deployed_point_loss != expected_point:
        raise ValueError(f"{name}.deployed_point_loss violates exact fallback")
    if deployed_proper_score != expected_proper:
        raise ValueError(f"{name}.deployed_proper_score violates exact fallback")
    return TournamentRecord(
        candidate_id=_identifier(payload["candidate_id"], name=f"{name}.candidate_id"),
        unit_id=_text(payload["unit_id"], name=f"{name}.unit_id"),
        group_id=_text(payload["group_id"], name=f"{name}.group_id"),
        horizon=_text(payload["horizon"], name=f"{name}.horizon"),
        accepted=accepted,
        point_loss=point_loss,
        fallback_point_loss=fallback_point_loss,
        deployed_point_loss=deployed_point_loss,
        proper_score=proper_score,
        fallback_proper_score=fallback_proper_score,
        deployed_proper_score=deployed_proper_score,
        interval_covered=covered,
        interval_width=width,
    )


def parse_discrepancy_candidate_tournament(
    payload: Mapping[str, object],
) -> TournamentEvidence:
    """Validate one matched source-only discrepancy-candidate tournament."""

    expected = {
        "contract",
        "schema_version",
        "protocol_id",
        "statistical_unit",
        "split",
        "reference_candidate",
        "physical_fallback_candidate",
        "information_boundary",
        "evaluation",
        "selection",
        "candidates",
        "records",
    }
    if set(payload) != expected:
        raise ValueError("tournament input fields changed")
    if payload["contract"] != DISCREPANCY_TOURNAMENT_INPUT_CONTRACT:
        raise ValueError("unsupported tournament input contract")
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1")
    if payload["split"] != "source-only":
        raise ValueError("tournament split must be source-only")

    boundary = _mapping(payload["information_boundary"], name="information_boundary")
    expected_boundary = {
        "candidate_predictions_sealed_before_scoring",
        "candidate_generation_used_scored_targets",
        "future_observations_used",
        "confirmation_payloads_opened",
        "replacement_allowed",
    }
    if set(boundary) != expected_boundary:
        raise ValueError("information_boundary fields changed")
    _require(
        _boolean(
            boundary["candidate_predictions_sealed_before_scoring"],
            name="information_boundary.candidate_predictions_sealed_before_scoring",
        ),
        "candidate predictions were not sealed before scoring",
    )
    _require(
        not _boolean(
            boundary["candidate_generation_used_scored_targets"],
            name="information_boundary.candidate_generation_used_scored_targets",
        ),
        "candidate generation used scored targets",
    )
    _require(
        not _boolean(
            boundary["future_observations_used"],
            name="information_boundary.future_observations_used",
        ),
        "future observations were used",
    )
    _require(
        not _boolean(
            boundary["confirmation_payloads_opened"],
            name="information_boundary.confirmation_payloads_opened",
        ),
        "confirmation payloads were opened",
    )
    _require(
        not _boolean(
            boundary["replacement_allowed"],
            name="information_boundary.replacement_allowed",
        ),
        "candidate or group replacement is allowed",
    )

    candidates = tuple(
        _parse_candidate(value, index=index)
        for index, value in enumerate(
            _sequence(payload["candidates"], name="candidates")
        )
    )
    _require(
        len(candidates) >= 3,
        "tournament needs fallback, reference, and candidate",
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    _require(
        len(set(candidate_ids)) == len(candidate_ids), "candidate ids must be unique"
    )
    reference = _identifier(payload["reference_candidate"], name="reference_candidate")
    fallback = _identifier(
        payload["physical_fallback_candidate"],
        name="physical_fallback_candidate",
    )
    _require(reference != fallback, "reference and physical fallback must differ")
    _require(reference in candidate_ids, "reference candidate is absent")
    _require(fallback in candidate_ids, "physical fallback candidate is absent")

    records = tuple(
        _parse_record(value, index=index)
        for index, value in enumerate(_sequence(payload["records"], name="records"))
    )
    _require(bool(records), "records must not be empty")
    known = set(candidate_ids)
    seen: set[tuple[str, str]] = set()
    by_unit: dict[str, list[TournamentRecord]] = {}
    for record in records:
        _require(record.candidate_id in known, "record names an unknown candidate")
        key = (record.candidate_id, record.unit_id)
        _require(key not in seen, "candidate/unit record is duplicated")
        seen.add(key)
        by_unit.setdefault(record.unit_id, []).append(record)

    expected_candidate_set = set(candidate_ids)
    for unit_id, unit_records in by_unit.items():
        observed = {record.candidate_id for record in unit_records}
        _require(
            observed == expected_candidate_set,
            f"{unit_id} does not contain every registered candidate",
        )
        first = unit_records[0]
        for record in unit_records[1:]:
            _require(record.group_id == first.group_id, f"{unit_id} group_id changed")
            _require(record.horizon == first.horizon, f"{unit_id} horizon changed")
            _require(
                record.fallback_point_loss == first.fallback_point_loss,
                f"{unit_id} has candidate-dependent physical fallback loss",
            )
            _require(
                record.fallback_proper_score == first.fallback_proper_score,
                f"{unit_id} has candidate-dependent fallback proper score",
            )
        fallback_record = next(
            record for record in unit_records if record.candidate_id == fallback
        )
        _require(not fallback_record.accepted, f"{unit_id} fallback must be rejected")
        _require(
            fallback_record.point_loss == fallback_record.fallback_point_loss,
            f"{unit_id} fallback raw point loss changed",
        )
        _require(
            fallback_record.proper_score == fallback_record.fallback_proper_score,
            f"{unit_id} fallback raw proper score changed",
        )

    evaluation = _parse_evaluation(payload["evaluation"])
    config = _parse_selection(payload["selection"])
    if config.nominal_interval_coverage is None:
        _require(
            evaluation.interval_semantics_id == "none",
            "interval semantics must be none when intervals are disabled",
        )
    else:
        _require(
            evaluation.interval_semantics_id != "none",
            "interval semantics must identify the registered interval",
        )
    groups = {record.group_id for record in records}
    required_groups = config.minimum_group_count + int(
        config.require_crossfit_stability
    )
    _require(
        len(groups) >= required_groups,
        "tournament has too few independent groups for the frozen selection rule",
    )

    interval_presence: dict[str, set[bool]] = {
        candidate_id: set() for candidate_id in known
    }
    for record in records:
        interval_presence[record.candidate_id].add(record.interval_covered is not None)
    for candidate_id, presence in interval_presence.items():
        _require(
            len(presence) == 1,
            f"{candidate_id} mixes interval-bearing and interval-free records",
        )
    if config.nominal_interval_coverage is not None:
        missing = sorted(
            candidate_id
            for candidate_id, presence in interval_presence.items()
            if presence != {True}
        )
        _require(not missing, f"interval coverage is missing for candidates {missing}")

    return TournamentEvidence(
        protocol_id=_identifier(payload["protocol_id"], name="protocol_id"),
        statistical_unit=_text(payload["statistical_unit"], name="statistical_unit"),
        reference_candidate=reference,
        physical_fallback_candidate=fallback,
        evaluation=evaluation,
        candidates=tuple(
            sorted(candidates, key=lambda candidate: candidate.candidate_id)
        ),
        config=config,
        records=tuple(
            sorted(
                records,
                key=lambda record: (record.unit_id, record.candidate_id),
            )
        ),
    )


__all__ = [
    "DISCREPANCY_TOURNAMENT_CLAIM_BOUNDARY",
    "DISCREPANCY_TOURNAMENT_INPUT_CONTRACT",
    "DISCREPANCY_TOURNAMENT_REPORT_CONTRACT",
    "CandidateSpec",
    "TournamentEvaluationSpec",
    "TournamentEvidence",
    "TournamentRecord",
    "TournamentSelectionConfig",
    "parse_discrepancy_candidate_tournament",
]
