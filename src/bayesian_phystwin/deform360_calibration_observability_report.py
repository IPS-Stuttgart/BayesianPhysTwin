"""Object-balanced Deform360 calibration-only observability reports.

The low-level observability diagnostic compares two already constructed
nuisance-marginalized Gaussian information states. This module adds the missing
study boundary around those comparisons: exactly one case for each locked
Deform360 calibration object, technical failures retained without replacement,
an object-balanced summary, and a portable content identity for the Stage-1
evidence ledger.

No confirmation payload or target outcome is admitted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_calibration_execution import (
    Deform360Stage0SelectionV1,
    file_sha256,
    load_deform360_stage0_selection,
)
from .deform360_calibration_source_run_record import (
    load_deform360_calibration_source_run_record,
    validate_deform360_calibration_source_run_record,
)
from .deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
    load_deform360_visual_provider_lock,
)
from .observability_diagnostics import (
    MarginalObservabilityComparison,
    compare_marginal_observability,
)

DEFORM360_CALIBRATION_OBSERVABILITY_CASE_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-observability-case"
)
DEFORM360_CALIBRATION_OBSERVABILITY_REPORT_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-observability-report"
)
DEFORM360_CALIBRATION_OBSERVABILITY_VERSION = 1
DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS = (
    "visual-reference-vs-contact-candidate-after-nuisance-marginalization-v1"
)
DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID = (
    "deform360-official-hub-visuotactile-v1"
)
DEFORM360_CALIBRATION_OBSERVABILITY_CLAIM_BOUNDARY = (
    "Calibration-only physical-query observability evidence. A valid report does "
    "not establish Deform360 accuracy, tactile benefit, provider competence, "
    "calibrated deployment uncertainty, Causal4D benefit, safety, or state of "
    "the art."
)
DEFORM360_CALIBRATION_MINIMUM_SUPPORTED_OBJECTS = 8
DEFORM360_CALIBRATION_MINIMUM_SUPPORTED_PER_STRATUM = 4
DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM = 5

Deform360ObservabilityCaseStatus = Literal[
    "evaluated",
    "technical_failure_without_replacement",
]
Deform360Stratum = Literal["sheet", "volumetric"]

_CASE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "case_id",
        "protocol_id",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "implementation_revision",
        "object_id",
        "episode_id",
        "stratum",
        "physical_query_id",
        "status",
        "reference_state_artifact_id",
        "candidate_state_artifact_id",
        "contact_anchor_artifact_id",
        "reference_marginal_precision",
        "candidate_marginal_precision",
        "query_jacobian",
        "comparison",
        "source_artifacts",
        "information_boundary",
        "failure_reason",
        "claim_boundary",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "report_id",
        "protocol_id",
        "selection_artifact_sha256",
        "visual_provider_lock_id",
        "calibration_source_run_record_sha256",
        "calibration_source_revision",
        "implementation_revision",
        "physical_query_id",
        "numerical_positive_tolerance",
        "cases",
        "support_gate",
        "overall",
        "by_stratum",
        "status",
        "source_artifacts",
        "information_boundary",
        "metadata",
        "claim_boundary",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "calibration_payloads_opened",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "replacement_allowed",
    }
)
_REPORT_BOUNDARY = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _literal_string(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result != result.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(raw.item())
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _stratum(value: object) -> Deform360Stratum:
    if type(value) is not str or value not in {"sheet", "volumetric"}:
        raise ValueError("stratum must be sheet or volumetric")
    return cast(Deform360Stratum, value)


def _status(value: object) -> Deform360ObservabilityCaseStatus:
    allowed = {"evaluated", "technical_failure_without_replacement"}
    if type(value) is not str or value not in allowed:
        raise ValueError("calibration observability case status is unsupported")
    return cast(Deform360ObservabilityCaseStatus, value)


def _frozen_matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = np.array(raw, dtype=np.dtype("<f8"), copy=True, order="C")
    if array.ndim != 2 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.dtype("<f8")).reshape(array.shape)


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return sha256_digest(value, name=name)


def _information_boundary(
    value: Mapping[str, Any],
    *,
    evaluated: bool,
) -> Mapping[str, Any]:
    require_exact_fields(
        value,
        expected=_BOUNDARY_FIELDS,
        name="Deform360 calibration observability information boundary",
    )
    boundary = {
        key: genuine_boolean(value[key], name=f"information_boundary.{key}")
        for key in sorted(_BOUNDARY_FIELDS)
    }
    if evaluated and not boundary["calibration_payloads_opened"]:
        raise ValueError(
            "an evaluated case must acknowledge calibration payload access"
        )
    if boundary["confirmation_payloads_opened"]:
        raise ValueError("confirmation payload access is forbidden")
    if boundary["target_outcomes_used"]:
        raise ValueError("target outcomes are forbidden")
    if boundary["replacement_allowed"]:
        raise ValueError("replacement after calibration access is forbidden")
    return frozen_finite_json_mapping(boundary, name="information_boundary")


def _records_close(first: object, second: object) -> bool:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if set(first) != set(second):
            return False
        return all(_records_close(first[key], second[key]) for key in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _records_close(left, right)
            for left, right in zip(first, second, strict=True)
        )
    if (
        isinstance(first, (int, float))
        and not isinstance(first, bool)
        and isinstance(second, (int, float))
        and not isinstance(second, bool)
    ):
        return bool(np.isclose(first, second, rtol=1e-10, atol=1e-12))
    return first == second


@dataclass(frozen=True)
class _PrecisionState:
    precision: np.ndarray

    def marginal_state_precision(self) -> np.ndarray:
        return self.precision


@dataclass(frozen=True)
class Deform360CalibrationObservabilityCaseV1:
    """One locked calibration object in the contact-versus-visual comparison."""

    selection_artifact_sha256: str
    visual_provider_lock_id: str
    calibration_source_run_record_sha256: str
    implementation_revision: str
    object_id: str
    episode_id: int
    stratum: Deform360Stratum
    physical_query_id: str
    status: Deform360ObservabilityCaseStatus
    source_artifacts: Mapping[str, str]
    information_boundary: Mapping[str, Any]
    reference_state_artifact_id: str | None = None
    candidate_state_artifact_id: str | None = None
    contact_anchor_artifact_id: str | None = None
    reference_marginal_precision: np.ndarray | None = None
    candidate_marginal_precision: np.ndarray | None = None
    query_jacobian: np.ndarray | None = None
    failure_reason: str | None = None
    protocol_id: str = DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID
    case_id: str | None = None

    def __post_init__(self) -> None:
        protocol_id = _literal_string(self.protocol_id, name="protocol_id")
        if protocol_id != DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID:
            raise ValueError("calibration observability protocol_id changed")
        selection_id = sha256_digest(
            self.selection_artifact_sha256,
            name="selection_artifact_sha256",
        )
        visual_lock_id = sha256_digest(
            self.visual_provider_lock_id,
            name="visual_provider_lock_id",
        )
        run_record_id = sha256_digest(
            self.calibration_source_run_record_sha256,
            name="calibration_source_run_record_sha256",
        )
        implementation_revision = exact_revision(
            self.implementation_revision,
            name="implementation_revision",
        )
        object_id = _literal_string(self.object_id, name="object_id")
        episode_id = genuine_integer(self.episode_id, name="episode_id", minimum=0)
        stratum = _stratum(self.stratum)
        query_id = sha256_digest(self.physical_query_id, name="physical_query_id")
        status = _status(self.status)
        source_artifacts = source_artifact_mapping(
            self.source_artifacts,
            name="source_artifacts",
        )
        boundary = _information_boundary(
            self.information_boundary,
            evaluated=status == "evaluated",
        )

        reference_id = _optional_sha256(
            self.reference_state_artifact_id,
            name="reference_state_artifact_id",
        )
        candidate_id = _optional_sha256(
            self.candidate_state_artifact_id,
            name="candidate_state_artifact_id",
        )
        anchor_id = _optional_sha256(
            self.contact_anchor_artifact_id,
            name="contact_anchor_artifact_id",
        )
        reference_precision: np.ndarray | None = None
        candidate_precision: np.ndarray | None = None
        query: np.ndarray | None = None
        failure_reason = self.failure_reason

        if status == "evaluated":
            if reference_id is None or candidate_id is None or anchor_id is None:
                raise ValueError("evaluated cases require all source artifact IDs")
            if len({reference_id, candidate_id, anchor_id}) != 3:
                raise ValueError("evaluated source artifact IDs must be distinct")
            if self.reference_marginal_precision is None:
                raise ValueError("evaluated case lacks reference precision")
            if self.candidate_marginal_precision is None:
                raise ValueError("evaluated case lacks candidate precision")
            if self.query_jacobian is None:
                raise ValueError("evaluated case lacks physical query Jacobian")
            reference_precision = _frozen_matrix(
                self.reference_marginal_precision,
                name="reference_marginal_precision",
            )
            candidate_precision = _frozen_matrix(
                self.candidate_marginal_precision,
                name="candidate_marginal_precision",
            )
            query = _frozen_matrix(self.query_jacobian, name="query_jacobian")
            if failure_reason is not None:
                raise ValueError("evaluated case must not declare failure_reason")
            compare_marginal_observability(
                _PrecisionState(reference_precision),
                _PrecisionState(candidate_precision),
                query_jacobian=query,
            )
        else:
            forbidden = (
                reference_id,
                candidate_id,
                anchor_id,
                self.reference_marginal_precision,
                self.candidate_marginal_precision,
                self.query_jacobian,
            )
            if any(item is not None for item in forbidden):
                raise ValueError("technical failures must not carry numerical results")
            failure_reason = _literal_string(
                failure_reason,
                name="failure_reason",
            )

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "selection_artifact_sha256", selection_id)
        object.__setattr__(self, "visual_provider_lock_id", visual_lock_id)
        object.__setattr__(
            self,
            "calibration_source_run_record_sha256",
            run_record_id,
        )
        object.__setattr__(self, "implementation_revision", implementation_revision)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "physical_query_id", query_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(self, "information_boundary", boundary)
        object.__setattr__(self, "reference_state_artifact_id", reference_id)
        object.__setattr__(self, "candidate_state_artifact_id", candidate_id)
        object.__setattr__(self, "contact_anchor_artifact_id", anchor_id)
        object.__setattr__(self, "reference_marginal_precision", reference_precision)
        object.__setattr__(self, "candidate_marginal_precision", candidate_precision)
        object.__setattr__(self, "query_jacobian", query)
        object.__setattr__(self, "failure_reason", failure_reason)

        expected_id = content_id(self.identity_record())
        if self.case_id is not None:
            supplied_id = sha256_digest(self.case_id, name="case_id")
            if supplied_id != expected_id:
                raise ValueError("calibration observability case_id changed")
        object.__setattr__(self, "case_id", expected_id)

    @property
    def comparison(self) -> MarginalObservabilityComparison | None:
        if self.status != "evaluated":
            return None
        assert self.reference_marginal_precision is not None
        assert self.candidate_marginal_precision is not None
        assert self.query_jacobian is not None
        return compare_marginal_observability(
            _PrecisionState(self.reference_marginal_precision),
            _PrecisionState(self.candidate_marginal_precision),
            query_jacobian=self.query_jacobian,
        )

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_OBSERVABILITY_CASE_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_OBSERVABILITY_VERSION,
            "semantics": DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "calibration_source_run_record_sha256": (
                self.calibration_source_run_record_sha256
            ),
            "implementation_revision": self.implementation_revision,
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "physical_query_id": self.physical_query_id,
            "status": self.status,
            "reference_state_artifact_id": self.reference_state_artifact_id,
            "candidate_state_artifact_id": self.candidate_state_artifact_id,
            "contact_anchor_artifact_id": self.contact_anchor_artifact_id,
            "reference_marginal_precision": (
                None
                if self.reference_marginal_precision is None
                else self.reference_marginal_precision.tolist()
            ),
            "candidate_marginal_precision": (
                None
                if self.candidate_marginal_precision is None
                else self.candidate_marginal_precision.tolist()
            ),
            "query_jacobian": (
                None
                if self.query_jacobian is None
                else self.query_jacobian.tolist()
            ),
            "source_artifacts": plain_json(self.source_artifacts),
            "information_boundary": plain_json(self.information_boundary),
            "failure_reason": self.failure_reason,
            "claim_boundary": DEFORM360_CALIBRATION_OBSERVABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        comparison = self.comparison
        return {
            **self.identity_record(),
            "case_id": self.case_id,
            "comparison": None if comparison is None else comparison.to_record(),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> Deform360CalibrationObservabilityCaseV1:
        require_exact_fields(
            value,
            expected=_CASE_FIELDS,
            name="Deform360 calibration observability case",
        )
        if value["schema"] != DEFORM360_CALIBRATION_OBSERVABILITY_CASE_SCHEMA:
            raise ValueError("calibration observability case schema changed")
        version = genuine_integer(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if version != DEFORM360_CALIBRATION_OBSERVABILITY_VERSION:
            raise ValueError("calibration observability case version changed")
        if value["semantics"] != DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS:
            raise ValueError("calibration observability case semantics changed")
        if value["claim_boundary"] != (
            DEFORM360_CALIBRATION_OBSERVABILITY_CLAIM_BOUNDARY
        ):
            raise ValueError("calibration observability claim boundary changed")
        source_artifacts = value["source_artifacts"]
        boundary = value["information_boundary"]
        if not isinstance(source_artifacts, Mapping):
            raise ValueError("source_artifacts must be a JSON object")
        if not isinstance(boundary, Mapping):
            raise ValueError("information_boundary must be a JSON object")
        result = cls(
            protocol_id=value["protocol_id"],
            selection_artifact_sha256=value["selection_artifact_sha256"],
            visual_provider_lock_id=value["visual_provider_lock_id"],
            calibration_source_run_record_sha256=value[
                "calibration_source_run_record_sha256"
            ],
            implementation_revision=value["implementation_revision"],
            object_id=value["object_id"],
            episode_id=value["episode_id"],
            stratum=value["stratum"],
            physical_query_id=value["physical_query_id"],
            status=value["status"],
            reference_state_artifact_id=value["reference_state_artifact_id"],
            candidate_state_artifact_id=value["candidate_state_artifact_id"],
            contact_anchor_artifact_id=value["contact_anchor_artifact_id"],
            reference_marginal_precision=value["reference_marginal_precision"],
            candidate_marginal_precision=value["candidate_marginal_precision"],
            query_jacobian=value["query_jacobian"],
            source_artifacts=cast(Mapping[str, str], source_artifacts),
            information_boundary=cast(Mapping[str, Any], boundary),
            failure_reason=value["failure_reason"],
            case_id=value["case_id"],
        )
        expected_comparison = result.comparison
        expected_record = (
            None
            if expected_comparison is None
            else expected_comparison.to_record()
        )
        if not _records_close(value["comparison"], expected_record):
            raise ValueError("calibration observability comparison changed")
        return result


def _aggregate_cases(
    cases: Sequence[Deform360CalibrationObservabilityCaseV1],
    *,
    positive_tolerance: float,
) -> dict[str, object]:
    evaluated = [case for case in cases if case.status == "evaluated"]
    failures = len(cases) - len(evaluated)
    base: dict[str, object] = {
        "object_count": len(cases),
        "evaluated_object_count": len(evaluated),
        "technical_failure_count": failures,
        "evaluated_fraction": len(evaluated) / len(cases),
        "positive_mutual_information_object_count": 0,
        "positive_weakest_direction_object_count": 0,
        "positive_mean_variance_reduction_object_count": 0,
        "mean_mutual_information_gain_nats": None,
        "minimum_mutual_information_gain_nats": None,
        "mean_weakest_direction_precision_ratio": None,
        "minimum_weakest_direction_precision_ratio": None,
        "mean_variance_reduction_fraction": None,
        "minimum_variance_reduction_fraction": None,
        "mean_effective_rank_gain": None,
        "mean_numerical_rank_gain": None,
    }
    if not evaluated:
        return base
    comparisons = [
        cast(MarginalObservabilityComparison, case.comparison)
        for case in evaluated
    ]
    mutual_information = np.asarray(
        [value.mutual_information_gain_nats for value in comparisons],
        dtype=np.float64,
    )
    weakest_ratio = np.asarray(
        [value.weakest_direction_precision_ratio for value in comparisons],
        dtype=np.float64,
    )
    variance_reduction = np.asarray(
        [value.mean_variance_reduction_fraction for value in comparisons],
        dtype=np.float64,
    )
    effective_rank = np.asarray(
        [value.effective_rank_gain for value in comparisons],
        dtype=np.float64,
    )
    numerical_rank = np.asarray(
        [value.numerical_rank_gain for value in comparisons],
        dtype=np.float64,
    )
    base.update(
        {
            "positive_mutual_information_object_count": int(
                np.sum(mutual_information > positive_tolerance)
            ),
            "positive_weakest_direction_object_count": int(
                np.sum(weakest_ratio > 1.0 + positive_tolerance)
            ),
            "positive_mean_variance_reduction_object_count": int(
                np.sum(variance_reduction > positive_tolerance)
            ),
            "mean_mutual_information_gain_nats": float(
                np.mean(mutual_information)
            ),
            "minimum_mutual_information_gain_nats": float(
                np.min(mutual_information)
            ),
            "mean_weakest_direction_precision_ratio": float(
                np.mean(weakest_ratio)
            ),
            "minimum_weakest_direction_precision_ratio": float(
                np.min(weakest_ratio)
            ),
            "mean_variance_reduction_fraction": float(
                np.mean(variance_reduction)
            ),
            "minimum_variance_reduction_fraction": float(
                np.min(variance_reduction)
            ),
            "mean_effective_rank_gain": float(np.mean(effective_rank)),
            "mean_numerical_rank_gain": float(np.mean(numerical_rank)),
        }
    )
    return base


@dataclass(frozen=True)
class Deform360CalibrationObservabilityReportV1:
    """Exact ten-object calibration report before confirmation access."""

    selection_artifact_sha256: str
    visual_provider_lock_id: str
    calibration_source_run_record_sha256: str
    calibration_source_revision: str
    implementation_revision: str
    physical_query_id: str
    cases: Sequence[Deform360CalibrationObservabilityCaseV1]
    source_artifacts: Mapping[str, str]
    numerical_positive_tolerance: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protocol_id: str = DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID
    report_id: str | None = None

    def __post_init__(self) -> None:
        protocol_id = _literal_string(self.protocol_id, name="protocol_id")
        if protocol_id != DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID:
            raise ValueError("calibration observability report protocol changed")
        selection_id = sha256_digest(
            self.selection_artifact_sha256,
            name="selection_artifact_sha256",
        )
        visual_lock_id = sha256_digest(
            self.visual_provider_lock_id,
            name="visual_provider_lock_id",
        )
        run_record_id = sha256_digest(
            self.calibration_source_run_record_sha256,
            name="calibration_source_run_record_sha256",
        )
        source_revision = exact_revision(
            self.calibration_source_revision,
            name="calibration_source_revision",
        )
        implementation_revision = exact_revision(
            self.implementation_revision,
            name="implementation_revision",
        )
        query_id = sha256_digest(self.physical_query_id, name="physical_query_id")
        tolerance = _finite_nonnegative(
            self.numerical_positive_tolerance,
            name="numerical_positive_tolerance",
        )
        if isinstance(self.cases, (str, bytes)):
            raise ValueError("cases must be a sequence")
        cases = tuple(
            sorted(
                self.cases,
                key=lambda case: (case.stratum, case.object_id, case.episode_id),
            )
        )
        if len(cases) != 2 * DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM:
            raise ValueError("report must contain exactly ten calibration objects")
        if any(
            not isinstance(case, Deform360CalibrationObservabilityCaseV1)
            for case in cases
        ):
            raise ValueError("cases contain an unsupported value")
        unit_keys = [(case.object_id, case.episode_id) for case in cases]
        if len(set(unit_keys)) != len(unit_keys):
            raise ValueError("report repeats a calibration unit")
        object_ids = [case.object_id for case in cases]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("report repeats a physical object")
        for stratum in ("sheet", "volumetric"):
            count = sum(case.stratum == stratum for case in cases)
            if count != DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM:
                raise ValueError("report must contain five objects per stratum")
        for case in cases:
            if case.selection_artifact_sha256 != selection_id:
                raise ValueError("case selection identity differs from report")
            if case.visual_provider_lock_id != visual_lock_id:
                raise ValueError("case visual provider identity differs from report")
            if case.calibration_source_run_record_sha256 != run_record_id:
                raise ValueError("case calibration source record differs from report")
            if case.implementation_revision != implementation_revision:
                raise ValueError("case implementation revision differs from report")
            if case.physical_query_id != query_id:
                raise ValueError("cases use different physical queries")
        query_matrices = [
            case.query_jacobian for case in cases if case.query_jacobian is not None
        ]
        if query_matrices and any(
            not np.array_equal(query_matrices[0], matrix)
            for matrix in query_matrices[1:]
        ):
            raise ValueError("evaluated cases use different query Jacobians")
        source_artifacts = source_artifact_mapping(
            self.source_artifacts,
            name="source_artifacts",
        )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="Deform360 observability report metadata",
        )

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "selection_artifact_sha256", selection_id)
        object.__setattr__(self, "visual_provider_lock_id", visual_lock_id)
        object.__setattr__(
            self,
            "calibration_source_run_record_sha256",
            run_record_id,
        )
        object.__setattr__(self, "calibration_source_revision", source_revision)
        object.__setattr__(self, "implementation_revision", implementation_revision)
        object.__setattr__(self, "physical_query_id", query_id)
        object.__setattr__(self, "numerical_positive_tolerance", tolerance)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "source_artifacts", source_artifacts)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.identity_record())
        if self.report_id is not None:
            supplied_id = sha256_digest(self.report_id, name="report_id")
            if supplied_id != expected_id:
                raise ValueError("calibration observability report_id changed")
        object.__setattr__(self, "report_id", expected_id)

    @property
    def support_gate(self) -> dict[str, object]:
        evaluated_by_stratum = {
            stratum: sum(
                case.stratum == stratum and case.status == "evaluated"
                for case in self.cases
            )
            for stratum in ("sheet", "volumetric")
        }
        evaluated = sum(evaluated_by_stratum.values())
        passed = (
            evaluated >= DEFORM360_CALIBRATION_MINIMUM_SUPPORTED_OBJECTS
            and all(
                value >= DEFORM360_CALIBRATION_MINIMUM_SUPPORTED_PER_STRATUM
                for value in evaluated_by_stratum.values()
            )
        )
        return {
            "minimum_supported_objects": (
                DEFORM360_CALIBRATION_MINIMUM_SUPPORTED_OBJECTS
            ),
            "minimum_supported_objects_per_stratum": (
                DEFORM360_CALIBRATION_MINIMUM_SUPPORTED_PER_STRATUM
            ),
            "evaluated_object_count": evaluated,
            "technical_failure_count": len(self.cases) - evaluated,
            "evaluated_by_stratum": evaluated_by_stratum,
            "technical_failures_retained_without_replacement": True,
            "support_passed": passed,
        }

    @property
    def status(self) -> str:
        if self.support_gate["support_passed"] is True:
            return "completed-supported-calibration-observability"
        return "completed-insufficient-calibration-support"

    @property
    def overall(self) -> dict[str, object]:
        return _aggregate_cases(
            self.cases,
            positive_tolerance=self.numerical_positive_tolerance,
        )

    @property
    def by_stratum(self) -> dict[str, object]:
        return {
            stratum: _aggregate_cases(
                [case for case in self.cases if case.stratum == stratum],
                positive_tolerance=self.numerical_positive_tolerance,
            )
            for stratum in ("sheet", "volumetric")
        }

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_OBSERVABILITY_REPORT_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_OBSERVABILITY_VERSION,
            "semantics": DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "calibration_source_run_record_sha256": (
                self.calibration_source_run_record_sha256
            ),
            "calibration_source_revision": self.calibration_source_revision,
            "implementation_revision": self.implementation_revision,
            "physical_query_id": self.physical_query_id,
            "numerical_positive_tolerance": self.numerical_positive_tolerance,
            "case_ids": [cast(str, case.case_id) for case in self.cases],
            "source_artifacts": plain_json(self.source_artifacts),
            "information_boundary": dict(_REPORT_BOUNDARY),
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_CALIBRATION_OBSERVABILITY_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "schema": DEFORM360_CALIBRATION_OBSERVABILITY_REPORT_SCHEMA,
            "schema_version": DEFORM360_CALIBRATION_OBSERVABILITY_VERSION,
            "semantics": DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS,
            "report_id": self.report_id,
            "protocol_id": self.protocol_id,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "visual_provider_lock_id": self.visual_provider_lock_id,
            "calibration_source_run_record_sha256": (
                self.calibration_source_run_record_sha256
            ),
            "calibration_source_revision": self.calibration_source_revision,
            "implementation_revision": self.implementation_revision,
            "physical_query_id": self.physical_query_id,
            "numerical_positive_tolerance": self.numerical_positive_tolerance,
            "cases": [case.to_record() for case in self.cases],
            "support_gate": self.support_gate,
            "overall": self.overall,
            "by_stratum": self.by_stratum,
            "status": self.status,
            "source_artifacts": plain_json(self.source_artifacts),
            "information_boundary": dict(_REPORT_BOUNDARY),
            "metadata": plain_json(self.metadata),
            "claim_boundary": DEFORM360_CALIBRATION_OBSERVABILITY_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> Deform360CalibrationObservabilityReportV1:
        require_exact_fields(
            value,
            expected=_REPORT_FIELDS,
            name="Deform360 calibration observability report",
        )
        if value["schema"] != DEFORM360_CALIBRATION_OBSERVABILITY_REPORT_SCHEMA:
            raise ValueError("calibration observability report schema changed")
        version = genuine_integer(
            value["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if version != DEFORM360_CALIBRATION_OBSERVABILITY_VERSION:
            raise ValueError("calibration observability report version changed")
        if value["semantics"] != DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS:
            raise ValueError("calibration observability report semantics changed")
        if value["information_boundary"] != _REPORT_BOUNDARY:
            raise ValueError("calibration observability information boundary changed")
        if value["claim_boundary"] != (
            DEFORM360_CALIBRATION_OBSERVABILITY_CLAIM_BOUNDARY
        ):
            raise ValueError("calibration observability claim boundary changed")
        raw_cases = value["cases"]
        if isinstance(raw_cases, (str, bytes)) or not isinstance(raw_cases, Sequence):
            raise ValueError("cases must be a sequence")
        cases = tuple(
            Deform360CalibrationObservabilityCaseV1.from_mapping(case)
            for case in raw_cases
        )
        source_artifacts = value["source_artifacts"]
        metadata = value["metadata"]
        if not isinstance(source_artifacts, Mapping):
            raise ValueError("source_artifacts must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        result = cls(
            protocol_id=value["protocol_id"],
            selection_artifact_sha256=value["selection_artifact_sha256"],
            visual_provider_lock_id=value["visual_provider_lock_id"],
            calibration_source_run_record_sha256=value[
                "calibration_source_run_record_sha256"
            ],
            calibration_source_revision=value["calibration_source_revision"],
            implementation_revision=value["implementation_revision"],
            physical_query_id=value["physical_query_id"],
            numerical_positive_tolerance=value["numerical_positive_tolerance"],
            cases=cases,
            source_artifacts=cast(Mapping[str, str], source_artifacts),
            metadata=cast(Mapping[str, Any], metadata),
            report_id=value["report_id"],
        )
        if value["support_gate"] != result.support_gate:
            raise ValueError("calibration observability support gate changed")
        if not _records_close(value["overall"], result.overall):
            raise ValueError("calibration observability overall summary changed")
        if not _records_close(value["by_stratum"], result.by_stratum):
            raise ValueError("calibration observability stratum summary changed")
        if value["status"] != result.status:
            raise ValueError("calibration observability status changed")
        return result


def _validate_source_run_record(
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    validated = validate_deform360_calibration_source_run_record(value)
    record_id = sha256_digest(
        validated.get("record_sha256"),
        name="record_sha256",
    )
    source_revision = exact_revision(
        validated.get("source_revision"),
        name="calibration_source_revision",
    )
    if validated.get("status") != "succeeded" or validated.get("exit_code") != 0:
        raise ValueError("calibration source run did not succeed")
    if validated.get("confirmation_boundary_verified") is not True:
        raise ValueError("calibration source confirmation boundary is unverified")
    if validated.get("confirmation_payloads_opened") is not False:
        raise ValueError("calibration source reports confirmation payload access")
    gate = validated.get("support_gate")
    if not isinstance(gate, Mapping) or gate.get("support_passed") is not True:
        raise ValueError("calibration source support gate did not pass")
    return validated, record_id, source_revision


def build_deform360_calibration_observability_report(
    selection: Deform360Stage0SelectionV1,
    visual_provider_lock: Deform360VisualProviderLockV1,
    calibration_source_run_record: Mapping[str, Any],
    cases: Sequence[Deform360CalibrationObservabilityCaseV1],
    *,
    implementation_revision: str,
    physical_query_id: str,
    source_artifacts: Mapping[str, str],
    numerical_positive_tolerance: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360CalibrationObservabilityReportV1:
    """Bind the ten exact calibration units to one object-balanced report."""

    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    validated_run, record_id, source_revision = _validate_source_run_record(
        calibration_source_run_record
    )
    if validated_run.get("selection_artifact_sha256") != (
        selection.selection_artifact_sha256
    ):
        raise ValueError("calibration source selection differs from Stage 0")
    if validated_run.get("visual_provider_lock_id") != visual_provider_lock.artifact_id:
        raise ValueError("calibration source visual provider lock differs")
    expected_units = {
        (unit.object_id, unit.episode_id, unit.stratum)
        for unit in selection.calibration_units
    }
    observed_units = {
        (case.object_id, case.episode_id, case.stratum) for case in cases
    }
    if observed_units != expected_units or len(cases) != len(expected_units):
        raise ValueError("observability cases differ from the Stage-0 calibration set")
    query_id = sha256_digest(physical_query_id, name="physical_query_id")
    for case in cases:
        if case.selection_artifact_sha256 != selection.selection_artifact_sha256:
            raise ValueError("case selection identity differs from Stage 0")
        if case.visual_provider_lock_id != visual_provider_lock.artifact_id:
            raise ValueError("case visual provider lock differs")
        if case.calibration_source_run_record_sha256 != record_id:
            raise ValueError("case calibration source record differs")
        if case.implementation_revision != implementation:
            raise ValueError("case implementation revision differs")
        if case.physical_query_id != query_id:
            raise ValueError("case physical query identity differs")
    return Deform360CalibrationObservabilityReportV1(
        selection_artifact_sha256=selection.selection_artifact_sha256,
        visual_provider_lock_id=visual_provider_lock.artifact_id,
        calibration_source_run_record_sha256=record_id,
        calibration_source_revision=source_revision,
        implementation_revision=implementation,
        physical_query_id=query_id,
        cases=cases,
        source_artifacts=source_artifacts,
        numerical_positive_tolerance=numerical_positive_tolerance,
        metadata={} if metadata is None else metadata,
    )


def load_deform360_calibration_observability_case(
    path: str | Path,
) -> Deform360CalibrationObservabilityCaseV1:
    value = load_strict_json_object(
        path,
        label="Deform360 calibration observability case",
    )
    return Deform360CalibrationObservabilityCaseV1.from_mapping(value)


def save_deform360_calibration_observability_case(
    value: Deform360CalibrationObservabilityCaseV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    write_atomic_json(value.to_record(), path, overwrite=overwrite)


def load_deform360_calibration_observability_report(
    path: str | Path,
) -> Deform360CalibrationObservabilityReportV1:
    value = load_strict_json_object(
        path,
        label="Deform360 calibration observability report",
    )
    return Deform360CalibrationObservabilityReportV1.from_mapping(value)


def save_deform360_calibration_observability_report(
    value: Deform360CalibrationObservabilityReportV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    write_atomic_json(value.to_record(), path, overwrite=overwrite)


def build_report_from_paths(
    *,
    selection_lock_path: str | Path,
    stage0_protocol_path: str | Path,
    visual_provider_lock_path: str | Path,
    calibration_source_run_record_path: str | Path,
    case_paths: Sequence[str | Path],
    implementation_revision: str,
    physical_query_id: str,
    numerical_positive_tolerance: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> Deform360CalibrationObservabilityReportV1:
    """Load exact source files and build a self-contained calibration report."""

    selection_path = Path(selection_lock_path)
    protocol_path = Path(stage0_protocol_path)
    visual_lock_path = Path(visual_provider_lock_path)
    run_record_path = Path(calibration_source_run_record_path)
    paths = tuple(Path(path) for path in case_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("case_paths must be nonempty and unique")
    cases = tuple(
        load_deform360_calibration_observability_case(path) for path in paths
    )
    logical_sources = {
        "sources/stage0/protocol.json": file_sha256(protocol_path),
        "sources/stage0/selection.json": file_sha256(selection_path),
        "sources/locks/visual-provider-lock.json": file_sha256(visual_lock_path),
        "sources/calibration-source/execution-manifest.json": file_sha256(
            run_record_path
        ),
    }
    for path, case in zip(paths, cases, strict=True):
        logical_sources[f"sources/observability/cases/{case.case_id}.json"] = (
            file_sha256(path)
        )
    selection = load_deform360_stage0_selection(
        selection_path,
        protocol_path=protocol_path,
    )
    visual_lock = load_deform360_visual_provider_lock(visual_lock_path)
    run_record = load_deform360_calibration_source_run_record(run_record_path)
    return build_deform360_calibration_observability_report(
        selection,
        visual_lock,
        run_record,
        cases,
        implementation_revision=implementation_revision,
        physical_query_id=physical_query_id,
        source_artifacts=logical_sources,
        numerical_positive_tolerance=numerical_positive_tolerance,
        metadata=metadata,
    )


__all__ = [
    "DEFORM360_CALIBRATION_OBSERVABILITY_CASE_SCHEMA",
    "DEFORM360_CALIBRATION_OBSERVABILITY_REPORT_SCHEMA",
    "DEFORM360_CALIBRATION_OBSERVABILITY_SEMANTICS",
    "DEFORM360_CALIBRATION_OBSERVABILITY_VERSION",
    "Deform360CalibrationObservabilityCaseV1",
    "Deform360CalibrationObservabilityReportV1",
    "build_deform360_calibration_observability_report",
    "build_report_from_paths",
    "load_deform360_calibration_observability_case",
    "load_deform360_calibration_observability_report",
    "save_deform360_calibration_observability_case",
    "save_deform360_calibration_observability_report",
]
