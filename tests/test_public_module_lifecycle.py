from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/quality/check_public_module_lifecycle.py"
MANIFEST_PATH = ROOT / "api/public-module-lifecycle-v1.json"
ROOT_MIGRATION_PATH = ROOT / "api/root-export-migration-v1.json"


def _tool() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "check_public_module_lifecycle",
        TOOL_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


tool = _tool()


def _payload() -> dict[str, object]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "public-module-lifecycle.json"
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_repository_public_module_lifecycle_matches_policy() -> None:
    report = tool.validate_public_module_lifecycle()

    assert report == {
        "package": "bayesian_phystwin",
        "compatibility_line": "0.4",
        "target_compatibility_line": "0.5",
        "policy": "explicit-stable-compatibility-experimental",
        "stable_module_count": 17,
        "compatibility_module_count": 19,
        "experimental_module_count": 5,
        "root_owner_count": 30,
        "status": "matched",
    }


def test_root_export_owners_are_classified_exactly_once() -> None:
    manifest = tool.load_lifecycle_manifest()
    root_owners = tool._root_owner_modules(ROOT_MIGRATION_PATH)
    categories = (
        set(manifest["stable_modules"]),
        set(manifest["compatibility_modules"]),
        set(manifest["experimental_modules"]),
    )

    assert root_owners <= set().union(*categories)
    for module in root_owners:
        assert sum(module in category for category in categories) == 1


def test_duplicate_and_overlapping_modules_are_rejected(
    tmp_path: Path,
) -> None:
    payload = _payload()
    stable = list(payload["stable_modules"])
    stable.append(stable[0])
    stable.sort()
    payload["stable_modules"] = stable
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="contains duplicate modules",
    ):
        tool.validate_public_module_lifecycle(path)

    payload = _payload()
    stable = list(payload["stable_modules"])
    compatibility = list(payload["compatibility_modules"])
    stable.append(compatibility[0])
    stable.sort()
    payload["stable_modules"] = stable
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="lifecycle categories overlap",
    ):
        tool.validate_public_module_lifecycle(path)


def test_lifecycle_lists_must_use_canonical_order(tmp_path: Path) -> None:
    payload = _payload()
    stable = list(payload["stable_modules"])
    stable[0], stable[1] = stable[1], stable[0]
    payload["stable_modules"] = stable
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="canonical lexical order",
    ):
        tool.validate_public_module_lifecycle(path)


def test_every_root_owner_must_remain_classified(tmp_path: Path) -> None:
    payload = _payload()
    compatibility = list(payload["compatibility_modules"])
    compatibility.pop(0)
    payload["compatibility_modules"] = compatibility
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="root owners are not fully classified",
    ):
        tool.validate_public_module_lifecycle(path)


def test_dataset_bound_modules_must_remain_experimental(
    tmp_path: Path,
) -> None:
    payload = _payload()
    compatibility = list(payload["compatibility_modules"])
    experimental = list(payload["experimental_modules"])
    compatibility.append(experimental.pop(0))
    compatibility.sort()
    payload["compatibility_modules"] = compatibility
    payload["experimental_modules"] = experimental
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="dataset or benchmark modules must be experimental",
    ):
        tool.validate_public_module_lifecycle(path)


def test_private_and_missing_module_identities_are_rejected(
    tmp_path: Path,
) -> None:
    payload = _payload()
    stable = list(payload["stable_modules"])
    stable.append("bayesian_phystwin._private")
    stable.sort()
    payload["stable_modules"] = stable
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="must identify a public module",
    ):
        tool.validate_public_module_lifecycle(path)

    payload = _payload()
    stable = list(payload["stable_modules"])
    stable.append("bayesian_phystwin.missing_public_module")
    stable.sort()
    payload["stable_modules"] = stable
    path = _write_payload(tmp_path, payload)

    with pytest.raises(
        tool.PublicModuleLifecycleError,
        match="has no unique regular source file",
    ):
        tool.validate_public_module_lifecycle(path)


def test_lifecycle_policy_is_shipped_and_unconditionally_checked() -> None:
    source_manifest = (ROOT / "MANIFEST.in").read_text(
        encoding="utf-8"
    ).splitlines()
    quality = (
        ROOT / "tools/quality/changed_python_quality.py"
    ).read_text(encoding="utf-8")

    assert "include api/public-module-lifecycle-v1.json" in source_manifest
    assert "include docs/public_module_lifecycle_v1.md" in source_manifest
    assert (
        "include tools/quality/check_public_module_lifecycle.py"
        in source_manifest
    )
    assert "tools/quality/check_public_module_lifecycle.py" in quality
