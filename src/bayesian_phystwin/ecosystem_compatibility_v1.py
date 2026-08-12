"""Versioned compatibility table for BayesianPhysTwin, Prob4D, and Causal4D.

The installed table records supported development package ranges and exact
provider/schema boundaries. It is not an experiment revision lock:
claim-bearing runs still require exact repository revisions and artifact
digests.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    plain_json,
)
from ._portable_contracts import content_id, repository_name, require_exact_fields

ECOSYSTEM_COMPATIBILITY_SCHEMA = "bayesian_phystwin.ecosystem_compatibility_table"
ECOSYSTEM_COMPATIBILITY_SCHEMA_VERSION = 1
ECOSYSTEM_COMPATIBILITY_RESOURCE = "contract_data/ecosystem_compatibility_v1/table.json"

_COMPONENT_ORDER = ("bayesian_phystwin", "prob4d", "causal4d")
_EXPECTED_DISTRIBUTIONS = {
    "bayesian_phystwin": "bayesian-phystwin",
    "prob4d": "prob4d",
    "causal4d": "causal4d",
}
_ALLOWED_TABLE_STATUSES = frozenset({"development_interoperability"})
_ALLOWED_INTERFACE_LIFECYCLES = frozenset(
    {"frozen_compatibility", "production", "production_additive"}
)
_ALLOWED_MODULE_LIFECYCLES = frozenset(
    {
        "additive_development",
        "diagnostic",
        "frozen_compatibility",
        "production",
        "production_additive",
    }
)
_PACKAGE_NAME = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")
_IMPORT_MODULE = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+$")
_SPECIFIER = re.compile(
    r"^(?:>=|<=|==|!=|~=|>|<)[0-9]+(?:\.[0-9]+){0,2}"
    r"(?:,(?:>=|<=|==|!=|~=|>|<)[0-9]+(?:\.[0-9]+){0,2})*$"
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "table_name",
        "status",
        "components",
        "interfaces",
        "evidence_boundary",
    }
)
_COMPONENT_FIELDS = frozenset(
    {
        "component_id",
        "distribution_name",
        "repository",
        "supported_versions",
        "requires_python",
        "required_dependencies",
        "role",
    }
)
_INTERFACE_FIELDS = frozenset(
    {
        "interface_id",
        "participants",
        "distribution_ranges",
        "provider_modules",
        "required_artifact_schema_versions",
        "lifecycle",
        "supports_claim_bearing_admission",
        "exact_revisions_required_for_evidence",
        "notes",
    }
)
_PROVIDER_MODULE_FIELDS = frozenset({"module", "api_version", "role", "lifecycle"})
_EVIDENCE_BOUNDARY_FIELDS = frozenset(
    {
        "development_ranges_are_evidence_locks",
        "exact_revisions_required_for_claim_bearing_runs",
        "artifact_digests_required_for_claim_bearing_runs",
        "interoperability_implies_accuracy_or_calibration",
        "interoperability_implies_physical_benefit",
    }
)
_EXPECTED_EVIDENCE_BOUNDARY = {
    "artifact_digests_required_for_claim_bearing_runs": True,
    "development_ranges_are_evidence_locks": False,
    "exact_revisions_required_for_claim_bearing_runs": True,
    "interoperability_implies_accuracy_or_calibration": False,
    "interoperability_implies_physical_benefit": False,
}


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _literal_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a literal boolean")
    return cast(bool, value)


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _version_specifier(value: object, *, name: str) -> str:
    specifier = _canonical_text(value, name=name)
    if _SPECIFIER.fullmatch(specifier) is None:
        raise ValueError(f"{name} must be a canonical version specifier")
    return specifier


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _strict_json_object(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "ecosystem compatibility resource is malformed JSON"
        ) from error
    return _mapping(value, name="ecosystem compatibility table")


def _canonical_strings(
    value: object,
    *,
    name: str,
    expected_length: int | None = None,
) -> list[str]:
    sequence = _sequence(value, name=name)
    result = [
        _canonical_text(item, name=f"{name}[{index}]")
        for index, item in enumerate(sequence)
    ]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")
    return result


def _positive_versions(value: object, *, name: str) -> list[int]:
    sequence = _sequence(value, name=name)
    versions = [
        _positive_integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(sequence)
    ]
    if not versions:
        raise ValueError(f"{name} must not be empty")
    if versions != sorted(set(versions)):
        raise ValueError(f"{name} must contain sorted unique versions")
    return versions


def _validated_dependencies(
    value: object,
    *,
    name: str,
) -> dict[str, str]:
    dependencies = _mapping(value, name=name)
    if not dependencies:
        raise ValueError(f"{name} must not be empty")
    normalized: dict[str, str] = {}
    for package_name, specifier in dependencies.items():
        if _PACKAGE_NAME.fullmatch(package_name) is None:
            raise ValueError(f"{name} has an invalid package name")
        normalized[package_name] = _version_specifier(
            specifier,
            name=f"{name}[{package_name!r}]",
        )
    return dict(sorted(normalized.items()))


def _validated_component(value: object) -> dict[str, object]:
    payload = _mapping(value, name="component")
    require_exact_fields(payload, expected=_COMPONENT_FIELDS, name="component")
    component_id = _canonical_text(payload["component_id"], name="component_id")
    if component_id not in _COMPONENT_ORDER:
        raise ValueError(f"unsupported ecosystem component: {component_id}")
    distribution = _canonical_text(
        payload["distribution_name"],
        name=f"{component_id}.distribution_name",
    )
    if (
        _PACKAGE_NAME.fullmatch(distribution) is None
        or distribution != _EXPECTED_DISTRIBUTIONS[component_id]
    ):
        raise ValueError(
            f"{component_id}.distribution_name does not match the registry"
        )
    return {
        "component_id": component_id,
        "distribution_name": distribution,
        "repository": repository_name(
            payload["repository"],
            name=f"{component_id}.repository",
        ),
        "supported_versions": _version_specifier(
            payload["supported_versions"],
            name=f"{component_id}.supported_versions",
        ),
        "requires_python": _version_specifier(
            payload["requires_python"],
            name=f"{component_id}.requires_python",
        ),
        "required_dependencies": _validated_dependencies(
            payload["required_dependencies"],
            name=f"{component_id}.required_dependencies",
        ),
        "role": _canonical_text(
            payload["role"],
            name=f"{component_id}.role",
        ),
    }


def _validated_provider_module(value: object) -> dict[str, object]:
    payload = _mapping(value, name="provider module")
    require_exact_fields(
        payload,
        expected=_PROVIDER_MODULE_FIELDS,
        name="provider module",
    )
    module = _canonical_text(payload["module"], name="provider module")
    if _IMPORT_MODULE.fullmatch(module) is None:
        raise ValueError("provider module must be a canonical import path")
    lifecycle = _canonical_text(
        payload["lifecycle"],
        name=f"{module}.lifecycle",
    )
    if lifecycle not in _ALLOWED_MODULE_LIFECYCLES:
        raise ValueError(f"{module}.lifecycle is unsupported")
    return {
        "module": module,
        "api_version": _positive_integer(
            payload["api_version"],
            name=f"{module}.api_version",
        ),
        "role": _canonical_text(
            payload["role"],
            name=f"{module}.role",
        ),
        "lifecycle": lifecycle,
    }


def _validated_schema_versions(
    value: object,
    *,
    interface_id: str,
) -> dict[str, list[int]]:
    schemas = _mapping(
        value,
        name=f"{interface_id}.required_artifact_schema_versions",
    )
    if not schemas:
        raise ValueError(
            f"{interface_id}.required_artifact_schema_versions must not be empty"
        )
    normalized: dict[str, list[int]] = {}
    for schema_name, versions in schemas.items():
        canonical_name = _canonical_text(
            schema_name,
            name=f"{interface_id}.required_artifact_schema_versions key",
        )
        normalized[canonical_name] = _positive_versions(
            versions,
            name=(
                f"{interface_id}.required_artifact_schema_versions[{canonical_name!r}]"
            ),
        )
    return dict(sorted(normalized.items()))


def _validated_interface(value: object) -> dict[str, object]:
    payload = _mapping(value, name="interface")
    require_exact_fields(payload, expected=_INTERFACE_FIELDS, name="interface")
    interface_id = _canonical_text(payload["interface_id"], name="interface_id")
    participants = _canonical_strings(
        payload["participants"],
        name=f"{interface_id}.participants",
        expected_length=2,
    )
    if any(participant not in _COMPONENT_ORDER for participant in participants):
        raise ValueError(f"{interface_id}.participants contains an unknown component")

    raw_ranges = _mapping(
        payload["distribution_ranges"],
        name=f"{interface_id}.distribution_ranges",
    )
    if set(raw_ranges) != set(participants):
        raise ValueError(f"{interface_id}.distribution_ranges must match participants")
    ranges = {
        participant: _version_specifier(
            raw_ranges[participant],
            name=f"{interface_id}.distribution_ranges[{participant!r}]",
        )
        for participant in participants
    }

    modules = [
        _validated_provider_module(item)
        for item in _sequence(
            payload["provider_modules"],
            name=f"{interface_id}.provider_modules",
        )
    ]
    if not modules:
        raise ValueError(f"{interface_id}.provider_modules must not be empty")
    module_names = [cast(str, item["module"]) for item in modules]
    if module_names != sorted(set(module_names)):
        raise ValueError(f"{interface_id}.provider_modules must be sorted and unique")

    lifecycle = _canonical_text(
        payload["lifecycle"],
        name=f"{interface_id}.lifecycle",
    )
    if lifecycle not in _ALLOWED_INTERFACE_LIFECYCLES:
        raise ValueError(f"{interface_id}.lifecycle is unsupported")
    return {
        "interface_id": interface_id,
        "participants": participants,
        "distribution_ranges": ranges,
        "provider_modules": modules,
        "required_artifact_schema_versions": _validated_schema_versions(
            payload["required_artifact_schema_versions"],
            interface_id=interface_id,
        ),
        "lifecycle": lifecycle,
        "supports_claim_bearing_admission": _literal_boolean(
            payload["supports_claim_bearing_admission"],
            name=f"{interface_id}.supports_claim_bearing_admission",
        ),
        "exact_revisions_required_for_evidence": _literal_boolean(
            payload["exact_revisions_required_for_evidence"],
            name=f"{interface_id}.exact_revisions_required_for_evidence",
        ),
        "notes": _canonical_strings(
            payload["notes"],
            name=f"{interface_id}.notes",
        ),
    }


def _validate_cross_field_semantics(
    components: Sequence[Mapping[str, object]],
    interfaces: Sequence[Mapping[str, object]],
) -> None:
    component_ranges = {
        cast(str, component["component_id"]): cast(
            str, component["supported_versions"]
        )
        for component in components
    }
    for interface in interfaces:
        interface_id = cast(str, interface["interface_id"])
        participants = cast(Sequence[str], interface["participants"])
        ranges = cast(Mapping[str, str], interface["distribution_ranges"])
        for participant in participants:
            if ranges[participant] != component_ranges[participant]:
                raise ValueError(
                    f"{interface_id}.distribution_ranges[{participant!r}] must "
                    f"match {participant}.supported_versions"
                )

        participant_roots = frozenset(participants)
        modules = cast(
            Sequence[Mapping[str, object]], interface["provider_modules"]
        )
        for module in modules:
            module_name = cast(str, module["module"])
            if module_name.partition(".")[0] not in participant_roots:
                raise ValueError(
                    f"{interface_id}.provider_modules contains a module outside "
                    "the declared participants"
                )

        if cast(bool, interface["supports_claim_bearing_admission"]) and not cast(
            bool, interface["exact_revisions_required_for_evidence"]
        ):
            raise ValueError(
                f"{interface_id} permits claim-bearing admission without exact "
                "revision evidence"
            )


def _validated_evidence_boundary(value: object) -> dict[str, bool]:
    payload = _mapping(value, name="evidence_boundary")
    require_exact_fields(
        payload,
        expected=_EVIDENCE_BOUNDARY_FIELDS,
        name="evidence_boundary",
    )
    normalized = {
        key: _literal_boolean(payload[key], name=f"evidence_boundary.{key}")
        for key in sorted(payload)
    }
    if normalized != _EXPECTED_EVIDENCE_BOUNDARY:
        raise ValueError("evidence_boundary weakens the registered policy")
    return normalized


def _validated_status(value: object) -> str:
    status = _canonical_text(value, name="status")
    if status not in _ALLOWED_TABLE_STATUSES:
        raise ValueError("unsupported ecosystem compatibility table status")
    return status


def _validated_table(value: Mapping[str, Any]) -> dict[str, object]:
    require_exact_fields(
        value,
        expected=_TOP_LEVEL_FIELDS,
        name="ecosystem compatibility table",
    )
    if value["schema_name"] != ECOSYSTEM_COMPATIBILITY_SCHEMA:
        raise ValueError("unsupported ecosystem compatibility schema")
    if (
        _positive_integer(value["schema_version"], name="schema_version")
        != ECOSYSTEM_COMPATIBILITY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported ecosystem compatibility schema version")

    components = [
        _validated_component(item)
        for item in _sequence(value["components"], name="components")
    ]
    component_ids = [cast(str, item["component_id"]) for item in components]
    if tuple(component_ids) != _COMPONENT_ORDER:
        raise ValueError("components must use canonical ecosystem order")

    interfaces = [
        _validated_interface(item)
        for item in _sequence(value["interfaces"], name="interfaces")
    ]
    interface_ids = [cast(str, item["interface_id"]) for item in interfaces]
    if not interfaces or interface_ids != sorted(set(interface_ids)):
        raise ValueError("interfaces must be sorted and unique")
    _validate_cross_field_semantics(components, interfaces)
    return {
        "schema_name": ECOSYSTEM_COMPATIBILITY_SCHEMA,
        "schema_version": ECOSYSTEM_COMPATIBILITY_SCHEMA_VERSION,
        "table_name": _canonical_text(value["table_name"], name="table_name"),
        "status": _validated_status(value["status"]),
        "components": components,
        "interfaces": interfaces,
        "evidence_boundary": _validated_evidence_boundary(value["evidence_boundary"]),
    }


@dataclass(frozen=True, slots=True)
class EcosystemCompatibilityTableV1:
    """Validated, recursively immutable compatibility table version 1."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        source = _mapping(self.payload, name="ecosystem compatibility table")
        normalized = _validated_table(source)
        object.__setattr__(
            self,
            "payload",
            frozen_finite_json_mapping(
                normalized,
                name="ecosystem compatibility table",
            ),
        )

    @property
    def table_name(self) -> str:
        return cast(str, self.payload["table_name"])

    @property
    def status(self) -> str:
        return cast(str, self.payload["status"])

    @property
    def table_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, Any]:
        return cast(dict[str, Any], plain_json(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return {"table_id": self.table_id, **self.descriptor()}

    def component(self, component_id: str) -> Mapping[str, Any]:
        requested = _canonical_text(component_id, name="component_id")
        components = cast(Sequence[Mapping[str, Any]], self.payload["components"])
        try:
            return next(
                component
                for component in components
                if component["component_id"] == requested
            )
        except StopIteration as error:
            raise KeyError(requested) from error

    def interface(self, interface_id: str) -> Mapping[str, Any]:
        requested = _canonical_text(interface_id, name="interface_id")
        interfaces = cast(Sequence[Mapping[str, Any]], self.payload["interfaces"])
        try:
            return next(
                interface
                for interface in interfaces
                if interface["interface_id"] == requested
            )
        except StopIteration as error:
            raise KeyError(requested) from error


def validate_ecosystem_compatibility_table_v1(
    value: Mapping[str, Any],
) -> EcosystemCompatibilityTableV1:
    """Validate one serialized compatibility table."""

    return EcosystemCompatibilityTableV1(payload=value)


def load_ecosystem_compatibility_table_v1() -> EcosystemCompatibilityTableV1:
    """Load and strictly validate the installed compatibility resource."""

    resource = resources.files("bayesian_phystwin").joinpath(
        *ECOSYSTEM_COMPATIBILITY_RESOURCE.split("/")
    )
    try:
        text = resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(
            "cannot read installed ecosystem compatibility table"
        ) from error
    return validate_ecosystem_compatibility_table_v1(_strict_json_object(text))


__all__ = [
    "ECOSYSTEM_COMPATIBILITY_RESOURCE",
    "ECOSYSTEM_COMPATIBILITY_SCHEMA",
    "ECOSYSTEM_COMPATIBILITY_SCHEMA_VERSION",
    "EcosystemCompatibilityTableV1",
    "load_ecosystem_compatibility_table_v1",
    "validate_ecosystem_compatibility_table_v1",
]
