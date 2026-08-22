"""Installed support and evidence boundary for the five maintained backends.

Integration support and empirical promotion are deliberately independent. A
backend can be fully executable, validated, documented, and protected by exact
fallback while a frozen scientific gate still rejects it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import require_exact_fields
from .material_backend_v1 import MATERIAL_BACKEND_SPECS

FIVE_BACKEND_SUPPORT_SCHEMA: Final = "bayesian-phystwin.five-backend-support"
FIVE_BACKEND_SUPPORT_VERSION: Final = 1
FIVE_BACKEND_SUPPORT_RESOURCE: Final = (
    "contract_data/backend_support_v1/five_backend_support_v1.json"
)
FIVE_BACKEND_SUPPORT_RESOURCE_SHA256: Final = (
    "4f865c8d01b5645bbc08c7eacd148b3370ef65923931d751c79c3220fe72d79b"
)
FIVE_BACKEND_IDS: Final = (
    "deform-dlo-v7",
    "matphys-warp-v1",
    "jax-fem-hyperelastic-v2",
    "mujoco-flex-v1",
    "sofa-fem-v3",
)

_REQUIRED_CAPABILITIES: Final = (
    "discoverable",
    "documented_claim_boundary",
    "exact_fallback",
    "executable_interface",
    "registered_tests",
    "release_compatible",
    "retained_evidence",
    "typed_artifact_or_protocol",
)
_TOP_LEVEL_FIELDS: Final = frozenset(
    {"schema", "schema_version", "support_definition", "backends"}
)
_SUPPORT_DEFINITION_FIELDS: Final = frozenset(
    {"status", "meaning", "required_capabilities"}
)
_BACKEND_FIELDS: Final = frozenset(
    {
        "backend_id",
        "display_name",
        "category",
        "distribution_scope",
        "canonical_material_profile",
        "portable_transport",
        "support_status",
        "support_capabilities",
        "implementation_paths",
        "test_paths",
        "documentation_paths",
        "fallback_semantics",
        "evidence",
    }
)
_EVIDENCE_FIELDS: Final = frozenset(
    {
        "stage",
        "latest_gate",
        "latest_gate_decision",
        "recommendation_authorized",
        "source_outcomes_opened",
        "public_benchmark_outcomes_opened",
        "protected_target_outcomes_opened",
        "held_v8_accessed",
        "dlo4_or_dlo5_accessed",
        "claim_scope",
        "artifacts",
    }
)
_ARTIFACT_FIELDS: Final = frozenset({"role", "path", "sha256"})
_CATEGORIES: Final = frozenset(
    {
        "learned-parameter-backend",
        "native-continuum-backend",
        "specialized-predictive-backend",
    }
)
_DISTRIBUTION_SCOPES: Final = frozenset(
    {"repository-source", "stable-package-and-repository-runner"}
)
_EVIDENCE_STAGES: Final = frozenset(
    {
        "benchmark-value-qualified",
        "source-covariance-value-rejected",
        "source-physics-rejected",
        "source-value-physical-rejected",
    }
)


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _literal_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a literal boolean")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _canonical_text(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _repository_path(value: object, *, name: str) -> str:
    path = _canonical_text(value, name=name)
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or path != parsed.as_posix()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return path


def _canonical_string_list(
    value: object,
    *,
    name: str,
    paths: bool = False,
) -> list[str]:
    values = _sequence(value, name=name)
    result = [
        (
            _repository_path(item, name=f"{name}[{index}]")
            if paths
            else _canonical_text(item, name=f"{name}[{index}]")
        )
        for index, item in enumerate(values)
    ]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values")
    if result != sorted(result):
        raise ValueError(f"{name} must use canonical lexical order")
    return result


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _strict_json_object(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("five-backend support resource is malformed") from error
    return _mapping(value, name="five-backend support resource")


def _validate_artifacts(value: object, *, backend_id: str) -> list[dict[str, str]]:
    records = _sequence(value, name=f"{backend_id}.evidence.artifacts")
    result: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for index, item in enumerate(records):
        artifact = _mapping(item, name=f"{backend_id}.evidence.artifacts[{index}]")
        require_exact_fields(
            artifact,
            expected=_ARTIFACT_FIELDS,
            name=f"{backend_id}.evidence.artifacts[{index}]",
        )
        role = _canonical_text(artifact["role"], name="artifact.role")
        path = _repository_path(artifact["path"], name="artifact.path")
        digest = _sha256(artifact["sha256"], name="artifact.sha256")
        identity = (role, path)
        if identity in identities:
            raise ValueError(f"{backend_id} has duplicate evidence artifacts")
        identities.add(identity)
        result.append({"role": role, "path": path, "sha256": digest})
    if not result:
        raise ValueError(f"{backend_id} must retain at least one evidence artifact")
    return result


def _validate_backend(value: object, *, expected_id: str) -> dict[str, object]:
    backend = _mapping(value, name=f"backend {expected_id}")
    require_exact_fields(backend, expected=_BACKEND_FIELDS, name=expected_id)
    backend_id = _canonical_text(backend["backend_id"], name="backend_id")
    if backend_id != expected_id:
        raise ValueError("five-backend roster or order changed")
    category = _canonical_text(backend["category"], name=f"{backend_id}.category")
    if category not in _CATEGORIES:
        raise ValueError(f"{backend_id} has an unsupported category")
    scope = _canonical_text(
        backend["distribution_scope"], name=f"{backend_id}.distribution_scope"
    )
    if scope not in _DISTRIBUTION_SCOPES:
        raise ValueError(f"{backend_id} has an unsupported distribution scope")
    profile_value = backend["canonical_material_profile"]
    profile: str | None
    if profile_value is None:
        profile = None
    else:
        profile = _canonical_text(
            profile_value, name=f"{backend_id}.canonical_material_profile"
        )
        if profile not in MATERIAL_BACKEND_SPECS:
            raise ValueError(f"{backend_id} refers to an unknown material profile")
    support_status = _canonical_text(
        backend["support_status"], name=f"{backend_id}.support_status"
    )
    if support_status != "fully-supported":
        raise ValueError(f"{backend_id} is not fully supported")
    capabilities = _mapping(
        backend["support_capabilities"],
        name=f"{backend_id}.support_capabilities",
    )
    require_exact_fields(
        capabilities,
        expected=frozenset(_REQUIRED_CAPABILITIES),
        name=f"{backend_id}.support_capabilities",
    )
    normalized_capabilities = {
        name: _literal_bool(capabilities[name], name=f"{backend_id}.{name}")
        for name in _REQUIRED_CAPABILITIES
    }
    if not all(normalized_capabilities.values()):
        raise ValueError(f"{backend_id} does not satisfy the full-support contract")

    evidence = _mapping(backend["evidence"], name=f"{backend_id}.evidence")
    require_exact_fields(evidence, expected=_EVIDENCE_FIELDS, name="evidence")
    stage = _canonical_text(evidence["stage"], name=f"{backend_id}.evidence.stage")
    if stage not in _EVIDENCE_STAGES:
        raise ValueError(f"{backend_id} has an unsupported evidence stage")
    decision = _canonical_text(
        evidence["latest_gate_decision"],
        name=f"{backend_id}.evidence.latest_gate_decision",
    )
    if decision not in {"qualified", "rejected"}:
        raise ValueError(f"{backend_id} has an invalid gate decision")
    recommendation = _literal_bool(
        evidence["recommendation_authorized"],
        name=f"{backend_id}.evidence.recommendation_authorized",
    )
    if recommendation != (
        stage == "benchmark-value-qualified" and decision == "qualified"
    ):
        raise ValueError("recommendation must follow the retained evidence decision")
    boolean_names = (
        "source_outcomes_opened",
        "public_benchmark_outcomes_opened",
        "protected_target_outcomes_opened",
        "held_v8_accessed",
        "dlo4_or_dlo5_accessed",
    )
    boundaries = {
        name: _literal_bool(evidence[name], name=f"{backend_id}.evidence.{name}")
        for name in boolean_names
    }
    if any(
        boundaries[name]
        for name in (
            "protected_target_outcomes_opened",
            "held_v8_accessed",
            "dlo4_or_dlo5_accessed",
        )
    ):
        raise ValueError("protected evidence cannot enter the support descriptor")
    if boundaries["public_benchmark_outcomes_opened"] != (
        backend_id == "deform-dlo-v7"
    ):
        raise ValueError("public benchmark access is recorded inconsistently")

    return {
        "backend_id": backend_id,
        "display_name": _canonical_text(
            backend["display_name"], name=f"{backend_id}.display_name"
        ),
        "category": category,
        "distribution_scope": scope,
        "canonical_material_profile": profile,
        "portable_transport": _canonical_text(
            backend["portable_transport"], name=f"{backend_id}.portable_transport"
        ),
        "support_status": support_status,
        "support_capabilities": normalized_capabilities,
        "implementation_paths": _canonical_string_list(
            backend["implementation_paths"],
            name=f"{backend_id}.implementation_paths",
            paths=True,
        ),
        "test_paths": _canonical_string_list(
            backend["test_paths"], name=f"{backend_id}.test_paths", paths=True
        ),
        "documentation_paths": _canonical_string_list(
            backend["documentation_paths"],
            name=f"{backend_id}.documentation_paths",
            paths=True,
        ),
        "fallback_semantics": _canonical_text(
            backend["fallback_semantics"], name=f"{backend_id}.fallback_semantics"
        ),
        "evidence": {
            "stage": stage,
            "latest_gate": _canonical_text(
                evidence["latest_gate"], name=f"{backend_id}.evidence.latest_gate"
            ),
            "latest_gate_decision": decision,
            "recommendation_authorized": recommendation,
            **boundaries,
            "claim_scope": _canonical_text(
                evidence["claim_scope"], name=f"{backend_id}.evidence.claim_scope"
            ),
            "artifacts": _validate_artifacts(
                evidence["artifacts"], backend_id=backend_id
            ),
        },
    }


def validate_five_backend_support(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and freeze one five-backend support descriptor."""

    require_exact_fields(value, expected=_TOP_LEVEL_FIELDS, name="support descriptor")
    if value["schema"] != FIVE_BACKEND_SUPPORT_SCHEMA:
        raise ValueError("five-backend support schema changed")
    if value["schema_version"] != FIVE_BACKEND_SUPPORT_VERSION:
        raise ValueError("five-backend support schema version changed")
    definition = _mapping(value["support_definition"], name="support_definition")
    require_exact_fields(
        definition,
        expected=_SUPPORT_DEFINITION_FIELDS,
        name="support_definition",
    )
    if definition["status"] != "fully-supported":
        raise ValueError("support definition status changed")
    required = _canonical_string_list(
        definition["required_capabilities"], name="required_capabilities"
    )
    if tuple(required) != _REQUIRED_CAPABILITIES:
        raise ValueError("full-support capability roster changed")
    backends = _sequence(value["backends"], name="backends")
    if len(backends) != len(FIVE_BACKEND_IDS):
        raise ValueError("five-backend support requires exactly five backends")
    normalized = [
        _validate_backend(item, expected_id=backend_id)
        for backend_id, item in zip(FIVE_BACKEND_IDS, backends, strict=True)
    ]
    result = {
        "schema": FIVE_BACKEND_SUPPORT_SCHEMA,
        "schema_version": FIVE_BACKEND_SUPPORT_VERSION,
        "support_definition": {
            "status": "fully-supported",
            "meaning": _canonical_text(
                definition["meaning"], name="support_definition.meaning"
            ),
            "required_capabilities": list(_REQUIRED_CAPABILITIES),
        },
        "backends": normalized,
    }
    return cast(
        Mapping[str, Any],
        frozen_finite_json_mapping(result, name="five-backend support"),
    )


def _resource_bytes() -> bytes:
    member = resources.files(__package__)
    for component in FIVE_BACKEND_SUPPORT_RESOURCE.split("/"):
        member = member.joinpath(component)
    raw = member.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIVE_BACKEND_SUPPORT_RESOURCE_SHA256:
        raise RuntimeError("installed five-backend support resource changed")
    return raw


def load_five_backend_support() -> Mapping[str, Any]:
    """Load the installed, hash-bound five-backend support descriptor."""

    return validate_five_backend_support(_strict_json_object(_resource_bytes()))


def describe_five_backend_support() -> dict[str, Any]:
    """Return a mutable JSON-ready copy of the installed descriptor."""

    return cast(dict[str, Any], plain_json(load_five_backend_support()))


def _ordinary_repository_file(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or any(parent.is_symlink() for parent in candidate.parents if parent != root)
    ):
        raise RuntimeError(f"declared backend support file is unavailable: {relative}")
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(
            f"declared backend support file escaped the repository: {relative}"
        )
    return resolved


def verify_five_backend_source_tree(repository_root: str | Path) -> dict[str, object]:
    """Verify declared implementation, test, documentation, and evidence files."""

    root = Path(repository_root).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root must be an ordinary directory")
    root = root.resolve(strict=True)
    descriptor = load_five_backend_support()
    implementation_count = 0
    test_count = 0
    documentation_count = 0
    artifact_count = 0
    promoted: list[str] = []
    for backend_value in cast(Sequence[Mapping[str, Any]], descriptor["backends"]):
        backend_id = cast(str, backend_value["backend_id"])
        for key, counter_name in (
            ("implementation_paths", "implementation"),
            ("test_paths", "test"),
            ("documentation_paths", "documentation"),
        ):
            for relative in cast(Sequence[str], backend_value[key]):
                _ordinary_repository_file(root, relative)
                if counter_name == "implementation":
                    implementation_count += 1
                elif counter_name == "test":
                    test_count += 1
                else:
                    documentation_count += 1
        evidence = cast(Mapping[str, Any], backend_value["evidence"])
        if evidence["recommendation_authorized"] is True:
            promoted.append(backend_id)
        for artifact in cast(Sequence[Mapping[str, str]], evidence["artifacts"]):
            path = _ordinary_repository_file(root, artifact["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact["sha256"]:
                raise RuntimeError(
                    f"retained backend evidence changed: {backend_id}/{artifact['role']}"
                )
            artifact_count += 1
    return {
        "schema": FIVE_BACKEND_SUPPORT_SCHEMA,
        "schema_version": FIVE_BACKEND_SUPPORT_VERSION,
        "backend_count": len(FIVE_BACKEND_IDS),
        "fully_supported_backend_count": len(FIVE_BACKEND_IDS),
        "implementation_file_count": implementation_count,
        "test_file_count": test_count,
        "documentation_file_count": documentation_count,
        "evidence_artifact_count": artifact_count,
        "recommendation_authorized_backend_ids": promoted,
        "status": "verified",
    }


__all__ = [
    "FIVE_BACKEND_IDS",
    "FIVE_BACKEND_SUPPORT_RESOURCE",
    "FIVE_BACKEND_SUPPORT_RESOURCE_SHA256",
    "FIVE_BACKEND_SUPPORT_SCHEMA",
    "FIVE_BACKEND_SUPPORT_VERSION",
    "describe_five_backend_support",
    "load_five_backend_support",
    "validate_five_backend_support",
    "verify_five_backend_source_tree",
]
