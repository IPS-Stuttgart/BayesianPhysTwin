"""Target-blind physical-query contract for guarded provider promotion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .probabilistic_scoring import SCORE_ORDER
from .repository_provenance import RepositoryRole, RepositoryState

PHYSICAL_QUERY_SCHEMA: Final = "bayesian_phystwin.physical_query"
PHYSICAL_QUERY_VERSION: Final = 1
PHYSICAL_QUERY_CLAIM_BOUNDARY: Final = (
    "Target-blind query and decision-policy infrastructure only. A valid query "
    "does not establish provider competence, physical-query benefit, calibrated "
    "deployment uncertainty, Causal4D intervention benefit, deployment safety, "
    "or state of the art."
)

MARGINAL_GAUGE_COVARIANCE: Final = "marginal-gauge"
COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE: Final = "complete-explicit-joint-gauge"
_REQUIRED_COVARIANCE_TREATMENTS: Final = frozenset(
    {
        MARGINAL_GAUGE_COVARIANCE,
        COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    }
)
_ALLOWED_BOOTSTRAP_METHODS: Final = frozenset(
    {
        "paired-group-bootstrap",
        "paired-stratified-group-bootstrap",
        "exact-group-permutation",
    }
)
_REQUIRED_PACKAGE_BINDINGS: Final = frozenset({"bayesian-phystwin", "prob4d"})
_PRIMARY_REPOSITORY: Final = "IPS-Stuttgart/BayesianPhysTwin"
_OBSERVATION_REPOSITORY: Final = "IPS-Stuttgart/Prob4D"
_PACKAGE_NAME = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")

_INFORMATION_BOUNDARY: Final[Mapping[str, object]] = {
    "query_owner": "BayesianPhysTwin",
    "observation_provider": "Prob4D",
    "update_admission_owner": "BayesianPhysTwin",
    "downstream_consumer": "Causal4D",
    "downstream_consumes_only_accepted_belief": True,
    "target_outcomes_opened": False,
    "target_method_selection_allowed": False,
    "target_retuning_allowed": False,
}

_MARGIN_FIELDS: Final = frozenset(
    {
        "practical_equivalence_score",
        "maximum_harmful_score_increase",
        "minimum_accepted_coverage",
        "maximum_mean_width",
        "maximum_worst_group_score_regret",
        "minimum_shared_covariance_relevance",
        "width_unit",
    }
)
_BOOTSTRAP_FIELDS: Final = frozenset(
    {
        "independent_group_definition",
        "method",
        "resamples",
        "seed",
        "confidence_level",
        "stratification_keys",
    }
)
_REPOSITORY_FIELDS: Final = frozenset({"repository", "revision", "dirty", "role"})
_QUERY_FIELDS: Final = frozenset(
    {
        "query_id",
        "schema",
        "schema_version",
        "query_name",
        "dimension",
        "component_order",
        "physical_unit",
        "coordinate_frame",
        "horizon_values",
        "horizon_unit",
        "jacobian_provider_id",
        "baseline_physical_belief_id",
        "exact_fallback_id",
        "covariance_treatments",
        "principal_covariance_treatment",
        "primary_proper_score",
        "decision_margins",
        "shared_covariance_diagnostic",
        "computational_selection_rule",
        "bootstrap",
        "package_artifact_ids",
        "provider_manifest_id",
        "evidence_decision_ids",
        "repositories",
        "information_boundary",
        "claim_boundary",
        "metadata",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = ">" if minimum_exclusive else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        invalid = result >= maximum if maximum_exclusive else result > maximum
        if invalid:
            relation = "<" if maximum_exclusive else "<="
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _unique_text_tuple(
    value: object,
    *,
    name: str,
    allow_empty: bool,
    sort: bool,
) -> tuple[str, ...]:
    sequence = _sequence(value, name=name)
    result = tuple(_text(item, name=f"{name} entry") for item in sequence)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result)) if sort else result


def _horizon_tuple(value: object) -> tuple[float, ...]:
    sequence = _sequence(value, name="horizon_values")
    horizons = tuple(
        _finite_number(item, name="horizon value", minimum=0.0) for item in sequence
    )
    if not horizons:
        raise ValueError("horizon_values must not be empty")
    if any(right <= left for left, right in zip(horizons, horizons[1:], strict=False)):
        raise ValueError("horizon_values must be strictly increasing")
    return horizons


def _digest_mapping(
    value: object,
    *,
    name: str,
    required_keys: frozenset[str] | None = None,
) -> Mapping[str, str]:
    source = _mapping(value, name=name)
    if not source:
        raise ValueError(f"{name} must not be empty")
    normalized: dict[str, str] = {}
    for raw_key, raw_digest in source.items():
        key = _text(raw_key, name=f"{name} key")
        if _PACKAGE_NAME.fullmatch(key) is None:
            raise ValueError(f"{name} keys must be canonical lowercase identifiers")
        normalized[key] = sha256_digest(
            raw_digest,
            name=f"{name} entry {key}",
        )
    required = frozenset() if required_keys is None else required_keys
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"{name} is missing required bindings: {missing}")
    return cast(
        Mapping[str, str],
        frozen_finite_json_mapping(
            {key: normalized[key] for key in sorted(normalized)},
            name=name,
        ),
    )


def _validate_information_boundary(value: object) -> None:
    source = _mapping(value, name="information_boundary")
    require_exact_fields(
        source,
        expected=frozenset(_INFORMATION_BOUNDARY),
        name="information_boundary",
    )
    for key, expected in _INFORMATION_BOUNDARY.items():
        actual = source[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("physical-query information boundary changed")


def _repository_from_mapping(value: object) -> RepositoryState:
    source = _mapping(value, name="physical-query repository")
    require_exact_fields(
        source,
        expected=_REPOSITORY_FIELDS,
        name="physical-query repository",
    )
    return RepositoryState(
        repository=_text(source["repository"], name="repository"),
        revision=_text(source["revision"], name="repository revision"),
        dirty=_boolean(source["dirty"], name="repository dirty"),
        role=cast(
            RepositoryRole,
            _text(source["role"], name="repository role"),
        ),
    )


def _normalized_repositories(value: object) -> tuple[RepositoryState, ...]:
    sequence = _sequence(value, name="repositories")
    repositories = tuple(
        item if isinstance(item, RepositoryState) else _repository_from_mapping(item)
        for item in sequence
    )
    if not repositories:
        raise ValueError("repositories must not be empty")
    if any(not isinstance(item, RepositoryState) for item in repositories):
        raise ValueError("repositories must contain RepositoryState values")
    names = [item.repository for item in repositories]
    if len(names) != len(set(names)):
        raise ValueError("repositories must not repeat repository identities")
    if any(item.dirty for item in repositories):
        raise ValueError("a target-blind physical query cannot bind dirty repositories")

    primary = [item for item in repositories if item.role == "primary"]
    if len(primary) != 1 or primary[0].repository != _PRIMARY_REPOSITORY:
        raise ValueError(
            "repositories must contain exactly one BayesianPhysTwin primary"
        )
    observation = [
        item
        for item in repositories
        if item.role == "observation" and item.repository == _OBSERVATION_REPOSITORY
    ]
    if len(observation) != 1:
        raise ValueError(
            "repositories must bind Prob4D exactly once with observation role"
        )
    return (
        primary[0],
        *sorted(
            (item for item in repositories if item is not primary[0]),
            key=lambda item: (item.role, item.repository),
        ),
    )


@dataclass(frozen=True, slots=True)
class PhysicalQueryDecisionMarginsV1:
    """Frozen pass/fail margins for one physical-query promotion gate."""

    practical_equivalence_score: float
    maximum_harmful_score_increase: float
    minimum_accepted_coverage: float
    maximum_mean_width: float
    maximum_worst_group_score_regret: float
    minimum_shared_covariance_relevance: float
    width_unit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "practical_equivalence_score",
            _finite_number(
                self.practical_equivalence_score,
                name="practical_equivalence_score",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_harmful_score_increase",
            _finite_number(
                self.maximum_harmful_score_increase,
                name="maximum_harmful_score_increase",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_accepted_coverage",
            _finite_number(
                self.minimum_accepted_coverage,
                name="minimum_accepted_coverage",
                minimum=0.0,
                maximum=1.0,
                minimum_exclusive=True,
                maximum_exclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_mean_width",
            _finite_number(
                self.maximum_mean_width,
                name="maximum_mean_width",
                minimum=0.0,
                minimum_exclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_worst_group_score_regret",
            _finite_number(
                self.maximum_worst_group_score_regret,
                name="maximum_worst_group_score_regret",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "minimum_shared_covariance_relevance",
            _finite_number(
                self.minimum_shared_covariance_relevance,
                name="minimum_shared_covariance_relevance",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "width_unit",
            _text(self.width_unit, name="width_unit"),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "practical_equivalence_score": self.practical_equivalence_score,
            "maximum_harmful_score_increase": (self.maximum_harmful_score_increase),
            "minimum_accepted_coverage": self.minimum_accepted_coverage,
            "maximum_mean_width": self.maximum_mean_width,
            "maximum_worst_group_score_regret": (self.maximum_worst_group_score_regret),
            "minimum_shared_covariance_relevance": (
                self.minimum_shared_covariance_relevance
            ),
            "width_unit": self.width_unit,
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> PhysicalQueryDecisionMarginsV1:
        source = _mapping(value, name="decision_margins")
        require_exact_fields(
            source,
            expected=_MARGIN_FIELDS,
            name="decision_margins",
        )
        return cls(
            practical_equivalence_score=source["practical_equivalence_score"],
            maximum_harmful_score_increase=source["maximum_harmful_score_increase"],
            minimum_accepted_coverage=source["minimum_accepted_coverage"],
            maximum_mean_width=source["maximum_mean_width"],
            maximum_worst_group_score_regret=source["maximum_worst_group_score_regret"],
            minimum_shared_covariance_relevance=source[
                "minimum_shared_covariance_relevance"
            ],
            width_unit=source["width_unit"],
        )


@dataclass(frozen=True, slots=True)
class PhysicalQueryBootstrapV1:
    """Independent-group resampling policy fixed before target access."""

    independent_group_definition: str
    method: str
    resamples: int
    seed: int
    confidence_level: float
    stratification_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        definition = _text(
            self.independent_group_definition,
            name="independent_group_definition",
        )
        method = _text(self.method, name="bootstrap method")
        if method not in _ALLOWED_BOOTSTRAP_METHODS:
            raise ValueError("bootstrap method is not registered")
        object.__setattr__(self, "independent_group_definition", definition)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "resamples",
            genuine_integer(self.resamples, name="resamples", minimum=1),
        )
        object.__setattr__(
            self,
            "seed",
            genuine_integer(self.seed, name="seed", minimum=0),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _finite_number(
                self.confidence_level,
                name="confidence_level",
                minimum=0.0,
                maximum=1.0,
                minimum_exclusive=True,
                maximum_exclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "stratification_keys",
            _unique_text_tuple(
                self.stratification_keys,
                name="stratification_keys",
                allow_empty=True,
                sort=True,
            ),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "independent_group_definition": self.independent_group_definition,
            "method": self.method,
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "stratification_keys": list(self.stratification_keys),
        }

    @classmethod
    def from_mapping(cls, value: object) -> PhysicalQueryBootstrapV1:
        source = _mapping(value, name="bootstrap")
        require_exact_fields(
            source,
            expected=_BOOTSTRAP_FIELDS,
            name="bootstrap",
        )
        return cls(
            independent_group_definition=source["independent_group_definition"],
            method=source["method"],
            resamples=source["resamples"],
            seed=source["seed"],
            confidence_level=source["confidence_level"],
            stratification_keys=_unique_text_tuple(
                source["stratification_keys"],
                name="stratification_keys",
                allow_empty=True,
                sort=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalQueryV1:
    """Content-addressed target-blind query and admission-policy definition."""

    query_name: str
    dimension: int
    component_order: tuple[str, ...]
    physical_unit: str
    coordinate_frame: str
    horizon_values: tuple[float, ...]
    horizon_unit: str
    jacobian_provider_id: str
    baseline_physical_belief_id: str
    exact_fallback_id: str
    covariance_treatments: tuple[str, ...]
    principal_covariance_treatment: str
    primary_proper_score: str
    decision_margins: PhysicalQueryDecisionMarginsV1
    shared_covariance_diagnostic: str
    computational_selection_rule: str
    bootstrap: PhysicalQueryBootstrapV1
    package_artifact_ids: Mapping[str, str]
    provider_manifest_id: str
    evidence_decision_ids: Mapping[str, str]
    repositories: tuple[RepositoryState, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    query_id: str | None = None

    def __post_init__(self) -> None:
        query_name = _text(self.query_name, name="query_name")
        dimension = genuine_integer(
            self.dimension,
            name="dimension",
            minimum=1,
        )
        components = _unique_text_tuple(
            self.component_order,
            name="component_order",
            allow_empty=False,
            sort=False,
        )
        if len(components) != dimension:
            raise ValueError("component_order length must equal dimension")
        physical_unit = _text(self.physical_unit, name="physical_unit")
        coordinate_frame = _text(self.coordinate_frame, name="coordinate_frame")
        horizons = _horizon_tuple(self.horizon_values)
        horizon_unit = _text(self.horizon_unit, name="horizon_unit")
        treatments = _unique_text_tuple(
            self.covariance_treatments,
            name="covariance_treatments",
            allow_empty=False,
            sort=True,
        )
        missing_treatments = sorted(_REQUIRED_COVARIANCE_TREATMENTS - set(treatments))
        if missing_treatments:
            raise ValueError(
                "covariance_treatments is missing required variants: "
                f"{missing_treatments}"
            )
        principal = _text(
            self.principal_covariance_treatment,
            name="principal_covariance_treatment",
        )
        if principal not in treatments:
            raise ValueError(
                "principal_covariance_treatment must be a registered treatment"
            )
        proper_score = _text(
            self.primary_proper_score,
            name="primary_proper_score",
        )
        if proper_score not in SCORE_ORDER:
            raise ValueError("primary_proper_score must be a registered proper score")
        margins = self.decision_margins
        if not isinstance(margins, PhysicalQueryDecisionMarginsV1):
            raise ValueError(
                "decision_margins must be a PhysicalQueryDecisionMarginsV1"
            )
        if margins.width_unit != physical_unit:
            raise ValueError("decision width unit must equal the physical query unit")
        bootstrap = self.bootstrap
        if not isinstance(bootstrap, PhysicalQueryBootstrapV1):
            raise ValueError("bootstrap must be a PhysicalQueryBootstrapV1")

        object.__setattr__(self, "query_name", query_name)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "component_order", components)
        object.__setattr__(self, "physical_unit", physical_unit)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "horizon_values", horizons)
        object.__setattr__(self, "horizon_unit", horizon_unit)
        object.__setattr__(
            self,
            "jacobian_provider_id",
            sha256_digest(
                self.jacobian_provider_id,
                name="jacobian_provider_id",
            ),
        )
        object.__setattr__(
            self,
            "baseline_physical_belief_id",
            sha256_digest(
                self.baseline_physical_belief_id,
                name="baseline_physical_belief_id",
            ),
        )
        object.__setattr__(
            self,
            "exact_fallback_id",
            sha256_digest(self.exact_fallback_id, name="exact_fallback_id"),
        )
        object.__setattr__(self, "covariance_treatments", treatments)
        object.__setattr__(
            self,
            "principal_covariance_treatment",
            principal,
        )
        object.__setattr__(self, "primary_proper_score", proper_score)
        object.__setattr__(
            self,
            "shared_covariance_diagnostic",
            _text(
                self.shared_covariance_diagnostic,
                name="shared_covariance_diagnostic",
            ),
        )
        object.__setattr__(
            self,
            "computational_selection_rule",
            _text(
                self.computational_selection_rule,
                name="computational_selection_rule",
            ),
        )
        object.__setattr__(
            self,
            "package_artifact_ids",
            _digest_mapping(
                self.package_artifact_ids,
                name="package_artifact_ids",
                required_keys=_REQUIRED_PACKAGE_BINDINGS,
            ),
        )
        object.__setattr__(
            self,
            "provider_manifest_id",
            sha256_digest(
                self.provider_manifest_id,
                name="provider_manifest_id",
            ),
        )
        object.__setattr__(
            self,
            "evidence_decision_ids",
            _digest_mapping(
                self.evidence_decision_ids,
                name="evidence_decision_ids",
            ),
        )
        object.__setattr__(
            self,
            "repositories",
            _normalized_repositories(self.repositories),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="physical query metadata",
            ),
        )

        expected_id = content_id(self.descriptor())
        supplied_id = self.query_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="query_id")
            if supplied_id != expected_id:
                raise ValueError("query_id does not match physical-query content")
        object.__setattr__(self, "query_id", expected_id)

    def descriptor(self) -> dict[str, object]:
        """Return the canonical payload covered by :attr:`query_id`."""

        return {
            "schema": PHYSICAL_QUERY_SCHEMA,
            "schema_version": PHYSICAL_QUERY_VERSION,
            "query_name": self.query_name,
            "dimension": self.dimension,
            "component_order": list(self.component_order),
            "physical_unit": self.physical_unit,
            "coordinate_frame": self.coordinate_frame,
            "horizon_values": list(self.horizon_values),
            "horizon_unit": self.horizon_unit,
            "jacobian_provider_id": self.jacobian_provider_id,
            "baseline_physical_belief_id": self.baseline_physical_belief_id,
            "exact_fallback_id": self.exact_fallback_id,
            "covariance_treatments": list(self.covariance_treatments),
            "principal_covariance_treatment": (self.principal_covariance_treatment),
            "primary_proper_score": self.primary_proper_score,
            "decision_margins": self.decision_margins.descriptor(),
            "shared_covariance_diagnostic": self.shared_covariance_diagnostic,
            "computational_selection_rule": self.computational_selection_rule,
            "bootstrap": self.bootstrap.descriptor(),
            "package_artifact_ids": plain_json(self.package_artifact_ids),
            "provider_manifest_id": self.provider_manifest_id,
            "evidence_decision_ids": plain_json(self.evidence_decision_ids),
            "repositories": [item.as_dict() for item in self.repositories],
            "information_boundary": dict(_INFORMATION_BOUNDARY),
            "claim_boundary": PHYSICAL_QUERY_CLAIM_BOUNDARY,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {"query_id": self.query_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: object) -> PhysicalQueryV1:
        source = _mapping(value, name="physical query")
        require_exact_fields(
            source,
            expected=_QUERY_FIELDS,
            name="physical query",
        )
        if _text(source["schema"], name="schema") != PHYSICAL_QUERY_SCHEMA:
            raise ValueError("physical query schema changed")
        schema_version = genuine_integer(
            source["schema_version"],
            name="schema_version",
            minimum=1,
        )
        if schema_version != PHYSICAL_QUERY_VERSION:
            raise ValueError("physical query schema version changed")
        _validate_information_boundary(source["information_boundary"])
        if (
            _text(source["claim_boundary"], name="claim_boundary")
            != PHYSICAL_QUERY_CLAIM_BOUNDARY
        ):
            raise ValueError("physical-query claim boundary changed")
        return cls(
            query_name=source["query_name"],
            dimension=source["dimension"],
            component_order=_unique_text_tuple(
                source["component_order"],
                name="component_order",
                allow_empty=False,
                sort=False,
            ),
            physical_unit=source["physical_unit"],
            coordinate_frame=source["coordinate_frame"],
            horizon_values=_horizon_tuple(source["horizon_values"]),
            horizon_unit=source["horizon_unit"],
            jacobian_provider_id=source["jacobian_provider_id"],
            baseline_physical_belief_id=source["baseline_physical_belief_id"],
            exact_fallback_id=source["exact_fallback_id"],
            covariance_treatments=_unique_text_tuple(
                source["covariance_treatments"],
                name="covariance_treatments",
                allow_empty=False,
                sort=True,
            ),
            principal_covariance_treatment=source["principal_covariance_treatment"],
            primary_proper_score=source["primary_proper_score"],
            decision_margins=PhysicalQueryDecisionMarginsV1.from_mapping(
                source["decision_margins"]
            ),
            shared_covariance_diagnostic=source["shared_covariance_diagnostic"],
            computational_selection_rule=source["computational_selection_rule"],
            bootstrap=PhysicalQueryBootstrapV1.from_mapping(source["bootstrap"]),
            package_artifact_ids=cast(
                Mapping[str, str],
                _mapping(
                    source["package_artifact_ids"],
                    name="package_artifact_ids",
                ),
            ),
            provider_manifest_id=source["provider_manifest_id"],
            evidence_decision_ids=cast(
                Mapping[str, str],
                _mapping(
                    source["evidence_decision_ids"],
                    name="evidence_decision_ids",
                ),
            ),
            repositories=tuple(
                _repository_from_mapping(item)
                for item in _sequence(source["repositories"], name="repositories")
            ),
            metadata=_mapping(source["metadata"], name="metadata"),
            query_id=source["query_id"],
        )


def load_physical_query(path: str | Path) -> PhysicalQueryV1:
    """Strictly load and revalidate one physical-query JSON artifact."""

    return PhysicalQueryV1.from_mapping(
        load_strict_json_object(path, label="physical query")
    )


def write_physical_query(
    query: PhysicalQueryV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one physical query, refusing overwrite by default."""

    if not isinstance(query, PhysicalQueryV1):
        raise TypeError("query must be a PhysicalQueryV1")
    write_atomic_json(query.to_record(), path, overwrite=overwrite)


__all__ = [
    "PHYSICAL_QUERY_CLAIM_BOUNDARY",
    "PHYSICAL_QUERY_SCHEMA",
    "PHYSICAL_QUERY_VERSION",
    "PhysicalQueryBootstrapV1",
    "PhysicalQueryDecisionMarginsV1",
    "PhysicalQueryV1",
    "load_physical_query",
    "write_physical_query",
]
