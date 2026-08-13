#!/usr/bin/env python3
"""Validate the public-module lifecycle registry and its API bindings."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "api/public-module-lifecycle-v1.json"
DEFAULT_ROOT_MIGRATION = REPOSITORY_ROOT / "api/root-export-migration-v1.json"

SCHEMA = "bayesian-phystwin.public-module-lifecycle"
SCHEMA_VERSION = 1
PACKAGE = "bayesian_phystwin"
COMPATIBILITY_LINE = "0.4"
TARGET_COMPATIBILITY_LINE = "0.5"
POLICY = "explicit-stable-compatibility-experimental"
ROOT_EXPORT_MIGRATION = "api/root-export-migration-v1.json"
STABLE_API_MANIFESTS = (
    "api/versioned-public-api-v1.json",
    "api/inference-public-api-v1.json",
)
UNREGISTERED_MODULE_POLICY = "internal-or-experimental-no-compatibility-promise"
_FIELDS = {
    "schema",
    "schema_version",
    "package",
    "compatibility_line",
    "target_compatibility_line",
    "policy",
    "root_export_migration",
    "stable_api_manifests",
    "unregistered_module_policy",
    "stable_modules",
    "compatibility_modules",
    "experimental_modules",
}
_ROOT_MIGRATION_FIELDS = {
    "schema",
    "schema_version",
    "source_package",
    "source_compatibility_line",
    "target_compatibility_line",
    "policy",
    "root_api_snapshot",
    "owners",
}
_ROOT_OWNER_FIELDS = {"module", "symbols"}
_REQUIRED_STABLE_MODULES = frozenset(
    {
        "bayesian_phystwin.causal4d_provider_v1",
        "bayesian_phystwin.causal4d_provider_v2",
        "bayesian_phystwin.claim_bundle_v1",
        "bayesian_phystwin.complete_belief_selection",
        "bayesian_phystwin.evidence_decision_v1",
        "bayesian_phystwin.inference.v1",
        "bayesian_phystwin.observation_belief",
        "bayesian_phystwin.physical_linearization",
        "bayesian_phystwin.physical_query_v1",
        "bayesian_phystwin.posterior_covariance_semantics",
        "bayesian_phystwin.prior_aware_gauge_belief",
        "bayesian_phystwin.prob4d_causal_lineage",
        "bayesian_phystwin.prospective_prob4d_update",
        "bayesian_phystwin.repository_provenance",
        "bayesian_phystwin.run_manifest",
        "bayesian_phystwin.run_manifest_v2",
        "bayesian_phystwin.v1",
    }
)


class PublicModuleLifecycleError(ValueError):
    """Raised when the public-module lifecycle registry is inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicModuleLifecycleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink():
        raise PublicModuleLifecycleError(f"{name} must not be a symlink")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OSError as error:
        raise PublicModuleLifecycleError(
            f"cannot read {name} {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise PublicModuleLifecycleError(f"invalid {name} JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise PublicModuleLifecycleError(f"{name} root must be a JSON object")
    return payload


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise PublicModuleLifecycleError(f"{name} must be nonempty canonical text")
    if any(character in value for character in "\x00\r\n"):
        raise PublicModuleLifecycleError(f"{name} must be single-line text")
    return value


def _repository_path(value: object, *, name: str) -> str:
    text = _canonical_text(value, name=name)
    path = Path(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise PublicModuleLifecycleError(
            f"{name} must be a canonical repository-relative path"
        )
    full_path = REPOSITORY_ROOT / path
    if not full_path.is_file() or full_path.is_symlink():
        raise PublicModuleLifecycleError(
            f"{name} does not identify a regular repository file"
        )
    return text


def _module_name(value: object, *, name: str) -> str:
    module = _canonical_text(value, name=name)
    prefix = f"{PACKAGE}."
    if not module.startswith(prefix):
        raise PublicModuleLifecycleError(f"{name} must be below {PACKAGE}")
    parts = module.split(".")
    if any(not part.isidentifier() for part in parts):
        raise PublicModuleLifecycleError(f"{name} must be a canonical Python module")
    if any(part.startswith("_") for part in parts[1:]):
        raise PublicModuleLifecycleError(f"{name} must identify a public module")
    return module


def _module_list(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicModuleLifecycleError(f"{name} must be a JSON array")
    modules = tuple(
        _module_name(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    if not modules:
        raise PublicModuleLifecycleError(f"{name} must not be empty")
    if len(set(modules)) != len(modules):
        raise PublicModuleLifecycleError(f"{name} contains duplicate modules")
    if modules != tuple(sorted(modules)):
        raise PublicModuleLifecycleError(f"{name} must use canonical lexical order")
    return modules


def load_lifecycle_manifest(
    path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    """Load and structurally validate the lifecycle registry."""

    payload = _load_json(path, name="public-module lifecycle manifest")
    if set(payload) != _FIELDS:
        missing = sorted(_FIELDS - set(payload))
        unknown = sorted(set(payload) - _FIELDS)
        raise PublicModuleLifecycleError(
            f"lifecycle manifest fields changed; missing={missing}, unknown={unknown}"
        )
    expected_scalars = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "package": PACKAGE,
        "compatibility_line": COMPATIBILITY_LINE,
        "target_compatibility_line": TARGET_COMPATIBILITY_LINE,
        "policy": POLICY,
        "root_export_migration": ROOT_EXPORT_MIGRATION,
        "unregistered_module_policy": UNREGISTERED_MODULE_POLICY,
    }
    for field, expected in expected_scalars.items():
        if payload[field] != expected:
            raise PublicModuleLifecycleError(f"public-module lifecycle {field} changed")

    raw_stable_manifests = payload["stable_api_manifests"]
    if isinstance(raw_stable_manifests, (str, bytes)) or not isinstance(
        raw_stable_manifests, Sequence
    ):
        raise PublicModuleLifecycleError("stable_api_manifests must be a JSON array")
    stable_manifests = tuple(
        _repository_path(
            item,
            name=f"stable_api_manifests[{index}]",
        )
        for index, item in enumerate(raw_stable_manifests)
    )
    if stable_manifests != STABLE_API_MANIFESTS:
        raise PublicModuleLifecycleError("stable API manifest bindings changed")
    _repository_path(
        payload["root_export_migration"],
        name="root_export_migration",
    )

    return {
        **expected_scalars,
        "stable_api_manifests": stable_manifests,
        "stable_modules": _module_list(
            payload["stable_modules"],
            name="stable_modules",
        ),
        "compatibility_modules": _module_list(
            payload["compatibility_modules"],
            name="compatibility_modules",
        ),
        "experimental_modules": _module_list(
            payload["experimental_modules"],
            name="experimental_modules",
        ),
    }


def _root_owner_modules(path: Path) -> frozenset[str]:
    payload = _load_json(path, name="root-export migration manifest")
    if set(payload) != _ROOT_MIGRATION_FIELDS:
        missing = sorted(_ROOT_MIGRATION_FIELDS - set(payload))
        unknown = sorted(set(payload) - _ROOT_MIGRATION_FIELDS)
        raise PublicModuleLifecycleError(
            "root-export migration fields changed; "
            f"missing={missing}, unknown={unknown}"
        )
    expected = {
        "schema": "bayesian-phystwin.root-export-migration",
        "schema_version": 1,
        "source_package": PACKAGE,
        "source_compatibility_line": COMPATIBILITY_LINE,
        "target_compatibility_line": TARGET_COMPATIBILITY_LINE,
        "policy": "lazy-legacy-root-to-owning-module",
        "root_api_snapshot": "api/root-public-api-v0.4.json",
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise PublicModuleLifecycleError(f"root-export migration {field} changed")

    raw_owners = payload["owners"]
    if isinstance(raw_owners, (str, bytes)) or not isinstance(raw_owners, Sequence):
        raise PublicModuleLifecycleError(
            "root-export migration owners must be a JSON array"
        )
    owners: set[str] = set()
    for index, raw_owner in enumerate(raw_owners):
        if not isinstance(raw_owner, Mapping) or set(raw_owner) != _ROOT_OWNER_FIELDS:
            raise PublicModuleLifecycleError(
                f"root-export owner {index} fields changed"
            )
        module = _module_name(
            raw_owner["module"],
            name=f"root owners[{index}].module",
        )
        if module in owners:
            raise PublicModuleLifecycleError(
                f"duplicate root-export owner module: {module}"
            )
        symbols = raw_owner["symbols"]
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
            raise PublicModuleLifecycleError(
                f"root owners[{index}].symbols must be a JSON array"
            )
        if not symbols:
            raise PublicModuleLifecycleError(
                f"root owners[{index}].symbols must not be empty"
            )
        owners.add(module)
    if not owners:
        raise PublicModuleLifecycleError(
            "root-export migration owners must not be empty"
        )
    return frozenset(owners)


def _module_source(module: str) -> Path:
    relative = module.removeprefix(f"{PACKAGE}.").replace(".", "/")
    candidates = (
        REPOSITORY_ROOT / "src" / PACKAGE / f"{relative}.py",
        REPOSITORY_ROOT / "src" / PACKAGE / relative / "__init__.py",
    )
    matches = tuple(
        path for path in candidates if path.is_file() and not path.is_symlink()
    )
    if len(matches) != 1:
        raise PublicModuleLifecycleError(
            f"classified module has no unique regular source file: {module}"
        )
    return matches[0]


def validate_public_module_lifecycle(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    root_migration_path: Path = DEFAULT_ROOT_MIGRATION,
) -> dict[str, object]:
    """Validate lifecycle categories, root ownership, and source identity."""

    manifest = load_lifecycle_manifest(manifest_path)
    stable = frozenset(manifest["stable_modules"])
    compatibility = frozenset(manifest["compatibility_modules"])
    experimental = frozenset(manifest["experimental_modules"])

    overlaps = {
        "stable/compatibility": sorted(stable & compatibility),
        "stable/experimental": sorted(stable & experimental),
        "compatibility/experimental": sorted(compatibility & experimental),
    }
    active_overlaps = {name: modules for name, modules in overlaps.items() if modules}
    if active_overlaps:
        raise PublicModuleLifecycleError(
            f"lifecycle categories overlap: {active_overlaps}"
        )

    missing_stable = sorted(_REQUIRED_STABLE_MODULES - stable)
    if missing_stable:
        raise PublicModuleLifecycleError(
            f"required stable modules are missing: {missing_stable}"
        )

    root_owners = _root_owner_modules(root_migration_path)
    non_root_lifecycle = sorted((compatibility | experimental) - root_owners)
    if non_root_lifecycle:
        raise PublicModuleLifecycleError(
            "compatibility or experimental entries are not root owners: "
            f"{non_root_lifecycle}"
        )
    classified = stable | compatibility | experimental
    missing_root_owners = sorted(root_owners - classified)
    if missing_root_owners:
        raise PublicModuleLifecycleError(
            f"root owners are not fully classified: {missing_root_owners}"
        )

    dataset_bound = {
        module
        for module in root_owners
        if module.startswith(f"{PACKAGE}.deform360_")
        or module == f"{PACKAGE}.synthetic_benchmark"
    }
    misclassified_dataset = sorted(dataset_bound - experimental)
    if misclassified_dataset:
        raise PublicModuleLifecycleError(
            "dataset or benchmark modules must be experimental: "
            f"{misclassified_dataset}"
        )

    for module in sorted(classified):
        _module_source(module)

    return {
        "package": PACKAGE,
        "compatibility_line": COMPATIBILITY_LINE,
        "target_compatibility_line": TARGET_COMPATIBILITY_LINE,
        "policy": POLICY,
        "stable_module_count": len(stable),
        "compatibility_module_count": len(compatibility),
        "experimental_module_count": len(experimental),
        "root_owner_count": len(root_owners),
        "status": "matched",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Public-module lifecycle manifest.",
    )
    parser.add_argument(
        "--root-migration",
        type=Path,
        default=DEFAULT_ROOT_MIGRATION,
        help="Bound root-export migration manifest.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = validate_public_module_lifecycle(
            arguments.manifest,
            root_migration_path=arguments.root_migration,
        )
    except PublicModuleLifecycleError as error:
        print(f"public-module lifecycle error: {error}", file=sys.stderr)
        return 2

    if arguments.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
