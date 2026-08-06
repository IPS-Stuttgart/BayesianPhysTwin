"""Machine-readable compatibility lock for the three-repository ecosystem."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import cast

ECOSYSTEM_COMPATIBILITY_SCHEMA = "bayesian-phystwin.ecosystem-compatibility-lock"
ECOSYSTEM_COMPATIBILITY_VERSION = 1
DEFAULT_ECOSYSTEM_COMPATIBILITY_RESOURCE = "data/ecosystem_compatibility_v1.json"
_COMPONENT_ORDER = ("bayesian_phystwin", "prob4d", "causal4d")
_COMPONENT_ALIASES = {
    "bayesian-phystwin": "bayesian_phystwin",
    "bayesian_phystwin": "bayesian_phystwin",
    "bpt": "bayesian_phystwin",
    "causal4d": "causal4d",
    "prob4d": "prob4d",
}
_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_VERSION_PREFIX_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.$")
_PACKAGE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _literal_string(value: object, *, name: str) -> str:
    _require(type(value) is str, f"{name} must be a literal string")
    result = cast(str, value)
    _require(result != "" and result == result.strip(), f"{name} is not canonical")
    return result


def _literal_integer(value: object, *, name: str, minimum: int = 0) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    result = cast(int, value)
    _require(result >= minimum, f"{name} must be at least {minimum}")
    return result


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("ecosystem compatibility lock is not strict JSON") from error
    _require(type(payload) is dict, "ecosystem compatibility lock must be an object")
    return payload


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    _require(
        not missing and not unexpected,
        f"{name} keys changed: missing={missing}, unexpected={unexpected}",
    )


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def normalize_ecosystem_component_id(value: str) -> str:
    """Normalize a supported human-facing component selector."""

    selector = _literal_string(value, name="component selector").lower()
    try:
        return _COMPONENT_ALIASES[selector]
    except KeyError as error:
        supported = ", ".join(sorted(_COMPONENT_ALIASES))
        raise ValueError(
            f"unknown ecosystem component {value!r}; expected one of {supported}"
        ) from error


@dataclass(frozen=True, slots=True)
class EcosystemComponentV1:
    """One package and repository combination in the compatibility lock."""

    component_id: str
    package_name: str
    repository: str
    revision: str
    locked_version: str
    compatible_version_prefix: str
    role: str

    def __post_init__(self) -> None:
        _require(
            self.component_id in _COMPONENT_ORDER,
            f"unsupported component id: {self.component_id}",
        )
        package_name = _literal_string(self.package_name, name="package_name")
        _require(
            _PACKAGE_PATTERN.fullmatch(package_name) is not None,
            "package_name is not canonical",
        )
        repository = _literal_string(self.repository, name="repository")
        _require(
            repository.count("/") == 1 and all(repository.split("/")),
            "repository must use owner/name form",
        )
        revision = _literal_string(self.revision, name="revision")
        _require(
            _SHA1_PATTERN.fullmatch(revision) is not None,
            "revision must be a lowercase 40-character Git commit",
        )
        locked_version = _literal_string(self.locked_version, name="locked_version")
        prefix = _literal_string(
            self.compatible_version_prefix,
            name="compatible_version_prefix",
        )
        _require(
            _VERSION_PREFIX_PATTERN.fullmatch(prefix) is not None,
            "compatible_version_prefix must have major.minor. form",
        )
        _require(
            locked_version.startswith(prefix),
            "locked_version must lie in the compatible version line",
        )
        _literal_string(self.role, name="role")

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "package_name": self.package_name,
            "repository": self.repository,
            "revision": self.revision,
            "locked_version": self.locked_version,
            "compatible_version_prefix": self.compatible_version_prefix,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class EcosystemCompatibilityLockV1:
    """Validated immutable compatibility lock and its content identity."""

    lock_name: str
    components: tuple[EcosystemComponentV1, ...]
    workflow_name: str
    workflow_run_id: int
    validated_date: str
    python_version: str
    tests_passed: int
    bayesian_phystwin_tested_revision: str
    lock_sha256: str

    def __post_init__(self) -> None:
        _literal_string(self.lock_name, name="lock_name")
        _require(
            tuple(component.component_id for component in self.components)
            == _COMPONENT_ORDER,
            "components must use the canonical BayesianPhysTwin, Prob4D, "
            "Causal4D order",
        )
        _literal_string(self.workflow_name, name="workflow_name")
        _literal_integer(self.workflow_run_id, name="workflow_run_id", minimum=1)
        _literal_string(self.validated_date, name="validated_date")
        _literal_string(self.python_version, name="python_version")
        _literal_integer(self.tests_passed, name="tests_passed", minimum=1)
        _require(
            _SHA1_PATTERN.fullmatch(self.bayesian_phystwin_tested_revision)
            is not None,
            "bayesian_phystwin_tested_revision must be a lowercase Git commit",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", self.lock_sha256) is not None,
            "lock_sha256 must be a lowercase SHA-256 digest",
        )

    def component(self, selector: str) -> EcosystemComponentV1:
        component_id = normalize_ecosystem_component_id(selector)
        return next(
            component
            for component in self.components
            if component.component_id == component_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lock_name": self.lock_name,
            "lock_sha256": self.lock_sha256,
            "validation": {
                "workflow_name": self.workflow_name,
                "workflow_run_id": self.workflow_run_id,
                "validated_date": self.validated_date,
                "python_version": self.python_version,
                "tests_passed": self.tests_passed,
                "bayesian_phystwin_tested_revision": (
                    self.bayesian_phystwin_tested_revision
                ),
            },
            "components": {
                component.component_id: component.to_dict()
                for component in self.components
            },
        }


@dataclass(frozen=True, slots=True)
class EcosystemComponentStatusV1:
    """Installed-package and optional source-revision compatibility status."""

    component_id: str
    package_name: str
    installed: bool
    installed_version: str | None
    locked_version: str
    compatible_version_prefix: str
    version_compatible: bool | None
    exact_version_match: bool | None
    locked_revision: str
    supplied_revision: str | None
    revision_compatible: bool | None
    required: bool
    compatible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "package_name": self.package_name,
            "installed": self.installed,
            "installed_version": self.installed_version,
            "locked_version": self.locked_version,
            "compatible_version_prefix": self.compatible_version_prefix,
            "version_compatible": self.version_compatible,
            "exact_version_match": self.exact_version_match,
            "locked_revision": self.locked_revision,
            "supplied_revision": self.supplied_revision,
            "revision_compatible": self.revision_compatible,
            "required": self.required,
            "compatible": self.compatible,
        }


@dataclass(frozen=True, slots=True)
class EcosystemCompatibilityReportV1:
    """Complete result of validating an installed ecosystem."""

    lock_name: str
    lock_sha256: str
    require_all: bool
    exact_versions: bool
    compatible: bool
    components: tuple[EcosystemComponentStatusV1, ...]

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(
            status.component_id for status in self.components if not status.installed
        )

    @property
    def incompatible_components(self) -> tuple[str, ...]:
        return tuple(
            status.component_id for status in self.components if not status.compatible
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "bayesian-phystwin.ecosystem-compatibility-report",
            "schema_version": 1,
            "lock_name": self.lock_name,
            "lock_sha256": self.lock_sha256,
            "require_all": self.require_all,
            "exact_versions": self.exact_versions,
            "compatible": self.compatible,
            "missing_components": list(self.missing_components),
            "incompatible_components": list(self.incompatible_components),
            "components": {
                status.component_id: status.to_dict() for status in self.components
            },
        }


def _component_from_payload(
    component_id: str,
    payload: object,
) -> EcosystemComponentV1:
    _require(type(payload) is dict, f"component {component_id} must be an object")
    component_payload = cast(dict[str, object], payload)
    expected = {
        "package_name",
        "repository",
        "revision",
        "locked_version",
        "compatible_version_prefix",
        "role",
    }
    _exact_keys(component_payload, expected, name=f"component {component_id}")
    return EcosystemComponentV1(
        component_id=component_id,
        package_name=_literal_string(
            component_payload["package_name"],
            name=f"{component_id}.package_name",
        ),
        repository=_literal_string(
            component_payload["repository"],
            name=f"{component_id}.repository",
        ),
        revision=_literal_string(
            component_payload["revision"],
            name=f"{component_id}.revision",
        ),
        locked_version=_literal_string(
            component_payload["locked_version"],
            name=f"{component_id}.locked_version",
        ),
        compatible_version_prefix=_literal_string(
            component_payload["compatible_version_prefix"],
            name=f"{component_id}.compatible_version_prefix",
        ),
        role=_literal_string(
            component_payload["role"],
            name=f"{component_id}.role",
        ),
    )


def _lock_from_payload(payload: dict[str, object]) -> EcosystemCompatibilityLockV1:
    _exact_keys(
        payload,
        {"schema", "schema_version", "lock_name", "validation", "components"},
        name="ecosystem compatibility lock",
    )
    _require(
        payload["schema"] == ECOSYSTEM_COMPATIBILITY_SCHEMA,
        "ecosystem compatibility schema changed",
    )
    _require(
        type(payload["schema_version"]) is int
        and payload["schema_version"] == ECOSYSTEM_COMPATIBILITY_VERSION,
        "unsupported ecosystem compatibility schema version",
    )
    lock_name = _literal_string(payload["lock_name"], name="lock_name")

    raw_validation = payload["validation"]
    _require(type(raw_validation) is dict, "validation must be an object")
    validation = cast(dict[str, object], raw_validation)
    _exact_keys(
        validation,
        {
            "workflow_name",
            "workflow_run_id",
            "validated_date",
            "python_version",
            "tests_passed",
            "bayesian_phystwin_tested_revision",
        },
        name="validation",
    )

    raw_components = payload["components"]
    _require(type(raw_components) is dict, "components must be an object")
    components_payload = cast(dict[str, object], raw_components)
    _exact_keys(components_payload, set(_COMPONENT_ORDER), name="components")
    components = tuple(
        _component_from_payload(component_id, components_payload[component_id])
        for component_id in _COMPONENT_ORDER
    )
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return EcosystemCompatibilityLockV1(
        lock_name=lock_name,
        components=components,
        workflow_name=_literal_string(
            validation["workflow_name"],
            name="validation.workflow_name",
        ),
        workflow_run_id=_literal_integer(
            validation["workflow_run_id"],
            name="validation.workflow_run_id",
            minimum=1,
        ),
        validated_date=_literal_string(
            validation["validated_date"],
            name="validation.validated_date",
        ),
        python_version=_literal_string(
            validation["python_version"],
            name="validation.python_version",
        ),
        tests_passed=_literal_integer(
            validation["tests_passed"],
            name="validation.tests_passed",
            minimum=1,
        ),
        bayesian_phystwin_tested_revision=_literal_string(
            validation["bayesian_phystwin_tested_revision"],
            name="validation.bayesian_phystwin_tested_revision",
        ),
        lock_sha256=digest,
    )


def load_ecosystem_compatibility_lock(
    path: str | Path | None = None,
) -> EcosystemCompatibilityLockV1:
    """Load and strictly validate the bundled or an explicitly supplied lock."""

    if path is None:
        text = (
            resources.files("bayesian_phystwin")
            .joinpath(DEFAULT_ECOSYSTEM_COMPATIBILITY_RESOURCE)
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    return _lock_from_payload(_strict_json(text))


def _metadata_versions(
    lock: EcosystemCompatibilityLockV1,
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for component in lock.components:
        try:
            versions[component.component_id] = metadata.version(component.package_name)
        except metadata.PackageNotFoundError:
            versions[component.component_id] = None
    return versions


def _normalize_versions(
    lock: EcosystemCompatibilityLockV1,
    installed_versions: Mapping[str, str | None] | None,
) -> dict[str, str | None]:
    if installed_versions is None:
        return _metadata_versions(lock)
    normalized: dict[str, str | None] = {
        component_id: None for component_id in _COMPONENT_ORDER
    }
    seen: set[str] = set()
    for selector, value in installed_versions.items():
        component_id = normalize_ecosystem_component_id(selector)
        _require(
            component_id not in seen,
            f"duplicate installed version for {component_id}",
        )
        seen.add(component_id)
        normalized[component_id] = (
            None
            if value is None
            else _literal_string(value, name=f"installed version for {component_id}")
        )
    return normalized


def _normalize_revisions(
    revisions: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for selector, value in (revisions or {}).items():
        component_id = normalize_ecosystem_component_id(selector)
        _require(
            component_id not in normalized,
            f"duplicate revision for {component_id}",
        )
        revision = _literal_string(value, name=f"revision for {component_id}")
        _require(
            _SHA1_PATTERN.fullmatch(revision) is not None,
            f"revision for {component_id} must be a lowercase 40-character commit",
        )
        normalized[component_id] = revision
    return normalized


def validate_installed_ecosystem(
    lock: EcosystemCompatibilityLockV1 | None = None,
    *,
    require_all: bool = False,
    exact_versions: bool = False,
    revisions: Mapping[str, str] | None = None,
    installed_versions: Mapping[str, str | None] | None = None,
) -> EcosystemCompatibilityReportV1:
    """Validate installed package lines and supplied exact repository revisions."""

    resolved_lock = lock or load_ecosystem_compatibility_lock()
    versions = _normalize_versions(resolved_lock, installed_versions)
    normalized_revisions = _normalize_revisions(revisions)
    statuses: list[EcosystemComponentStatusV1] = []
    for component in resolved_lock.components:
        installed_version = versions[component.component_id]
        installed = installed_version is not None
        required = require_all or component.component_id == "bayesian_phystwin"
        version_compatible = (
            None
            if installed_version is None
            else installed_version.startswith(component.compatible_version_prefix)
        )
        exact_version_match = (
            None
            if installed_version is None
            else installed_version == component.locked_version
        )
        supplied_revision = normalized_revisions.get(component.component_id)
        revision_compatible = (
            None
            if supplied_revision is None
            else supplied_revision == component.revision
        )
        compatible = (
            (not required and not installed and supplied_revision is None)
            or (
                installed
                and bool(version_compatible)
                and (not exact_versions or bool(exact_version_match))
                and (revision_compatible is not False)
            )
        )
        statuses.append(
            EcosystemComponentStatusV1(
                component_id=component.component_id,
                package_name=component.package_name,
                installed=installed,
                installed_version=installed_version,
                locked_version=component.locked_version,
                compatible_version_prefix=component.compatible_version_prefix,
                version_compatible=version_compatible,
                exact_version_match=exact_version_match,
                locked_revision=component.revision,
                supplied_revision=supplied_revision,
                revision_compatible=revision_compatible,
                required=required,
                compatible=compatible,
            )
        )
    result = tuple(statuses)
    return EcosystemCompatibilityReportV1(
        lock_name=resolved_lock.lock_name,
        lock_sha256=resolved_lock.lock_sha256,
        require_all=require_all,
        exact_versions=exact_versions,
        compatible=all(status.compatible for status in result),
        components=result,
    )
