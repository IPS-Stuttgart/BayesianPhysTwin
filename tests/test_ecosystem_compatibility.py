from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.cli import ecosystem_validate as cli_module
from bayesian_phystwin.cli.ecosystem_validate import main as ecosystem_main
from bayesian_phystwin.ecosystem_compatibility import (
    DEFAULT_ECOSYSTEM_COMPATIBILITY_RESOURCE,
    EcosystemCompatibilityLockV1,
    EcosystemCompatibilityReportV1,
    EcosystemComponentStatusV1,
    load_ecosystem_compatibility_lock,
    normalize_ecosystem_component_id,
    validate_installed_ecosystem,
)


def _complete_versions() -> dict[str, str]:
    return {
        "bayesian_phystwin": "0.4.0",
        "prob4d": "0.3.1",
        "causal4d": "0.5.0",
    }


def _lock_payload() -> dict[str, Any]:
    text = (
        resources.files("bayesian_phystwin")
        .joinpath(DEFAULT_ECOSYSTEM_COMPATIBILITY_RESOURCE)
        .read_text(encoding="utf-8")
    )
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def _write_payload(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _bundled_lock() -> EcosystemCompatibilityLockV1:
    lock = load_ecosystem_compatibility_lock()
    assert lock.lock_name == "three-repository-installed-wheel-v1"
    assert len(lock.lock_sha256) == 64
    assert tuple(component.component_id for component in lock.components) == (
        "bayesian_phystwin",
        "prob4d",
        "causal4d",
    )
    assert lock.component("bpt").revision == (
        "3f37fbc87975f0581a0e58434e53b44c4d61b402"
    )
    assert lock.component("prob4d").revision == (
        "9ad07f89f9a85b68cf1375a4087ffa447b6af846"
    )
    assert lock.component("causal4d").revision == (
        "b0bf0c2de176b29534ef59484ad167b8f27d9dae"
    )
    assert lock.bayesian_phystwin_tested_revision == lock.component("bpt").revision
    assert lock.workflow_run_id == 31019529164
    serialized = lock.to_dict()
    assert serialized["lock_sha256"] == lock.lock_sha256
    components = serialized["components"]
    assert isinstance(components, dict)
    prob4d = components["prob4d"]
    assert isinstance(prob4d, dict)
    assert prob4d["repository"] == "IPS-Stuttgart/Prob4D"
    assert lock.component("bpt").to_dict()["locked_version"] == "0.4.0"
    return lock


def _invalid_payload_cases() -> list[tuple[str, dict[str, Any], str]]:
    cases: list[tuple[str, dict[str, Any], str]] = []

    payload = _lock_payload()
    payload.pop("schema")
    cases.append(("missing-top-level", payload, "keys changed"))

    payload = _lock_payload()
    payload["unexpected"] = True
    cases.append(("unexpected-top-level", payload, "keys changed"))

    payload = _lock_payload()
    payload["schema"] = "wrong"
    cases.append(("wrong-schema", payload, "schema changed"))

    payload = _lock_payload()
    payload["schema_version"] = 2
    cases.append(("wrong-version", payload, "unsupported"))

    payload = _lock_payload()
    payload["schema_version"] = True
    cases.append(("boolean-version", payload, "unsupported"))

    payload = _lock_payload()
    payload["lock_name"] = " "
    cases.append(("blank-lock-name", payload, "not canonical"))

    payload = _lock_payload()
    payload["validation"] = []
    cases.append(("validation-not-object", payload, "validation must be an object"))

    payload = _lock_payload()
    validation = payload["validation"]
    assert isinstance(validation, dict)
    validation.pop("workflow_name")
    cases.append(("validation-missing", payload, "validation keys changed"))

    payload = _lock_payload()
    validation = payload["validation"]
    assert isinstance(validation, dict)
    validation["unexpected"] = 1
    cases.append(("validation-extra", payload, "validation keys changed"))

    payload = _lock_payload()
    payload["components"] = []
    cases.append(("components-not-object", payload, "components must be an object"))

    payload = _lock_payload()
    components = payload["components"]
    assert isinstance(components, dict)
    components.pop("prob4d")
    cases.append(("components-missing", payload, "components keys changed"))

    payload = _lock_payload()
    components = payload["components"]
    assert isinstance(components, dict)
    components["other"] = {}
    cases.append(("components-extra", payload, "components keys changed"))

    payload = _lock_payload()
    components = payload["components"]
    assert isinstance(components, dict)
    components["prob4d"] = []
    cases.append(("component-not-object", payload, "component prob4d must be an object"))

    payload = _lock_payload()
    components = payload["components"]
    assert isinstance(components, dict)
    prob4d = components["prob4d"]
    assert isinstance(prob4d, dict)
    prob4d.pop("role")
    cases.append(("component-missing", payload, "component prob4d keys changed"))

    payload = _lock_payload()
    components = payload["components"]
    assert isinstance(components, dict)
    prob4d = components["prob4d"]
    assert isinstance(prob4d, dict)
    prob4d["unexpected"] = 1
    cases.append(("component-extra", payload, "component prob4d keys changed"))

    invalid_component_fields = (
        ("package-name", "package_name", "Bad Name", "package_name is not canonical"),
        ("repository", "repository", "invalid", "owner/name"),
        ("revision", "revision", "A" * 40, "lowercase 40-character"),
        ("version-prefix", "compatible_version_prefix", "0.3", "major.minor"),
        ("version-line", "locked_version", "1.0.0", "compatible version line"),
        ("role", "role", " ", "not canonical"),
    )
    for name, field, value, message in invalid_component_fields:
        payload = _lock_payload()
        components = payload["components"]
        assert isinstance(components, dict)
        prob4d = components["prob4d"]
        assert isinstance(prob4d, dict)
        prob4d[field] = value
        cases.append((f"component-{name}", payload, message))

    invalid_validation_fields = (
        ("workflow-name", "workflow_name", " ", "not canonical"),
        ("workflow-id-zero", "workflow_run_id", 0, "at least 1"),
        ("workflow-id-bool", "workflow_run_id", True, "must be an integer"),
        ("date", "validated_date", " ", "not canonical"),
        ("python", "python_version", " ", "not canonical"),
        ("tests-zero", "tests_passed", 0, "at least 1"),
        ("tests-bool", "tests_passed", False, "must be an integer"),
        (
            "tested-revision-format",
            "bayesian_phystwin_tested_revision",
            "x",
            "lowercase Git commit",
        ),
        (
            "tested-revision-mismatch",
            "bayesian_phystwin_tested_revision",
            "0" * 40,
            "must match the locked",
        ),
    )
    for name, field, value, message in invalid_validation_fields:
        payload = _lock_payload()
        validation = payload["validation"]
        assert isinstance(validation, dict)
        validation[field] = value
        cases.append((f"validation-{name}", payload, message))

    return cases


def _exercise_dataclass_validation(lock: EcosystemCompatibilityLockV1) -> None:
    component = lock.component("bpt")
    component_cases = (
        ({"component_id": "other"}, "unsupported component"),
        ({"package_name": 1}, "literal string"),
        ({"package_name": "Bad Name"}, "not canonical"),
        ({"repository": "/name"}, "owner/name"),
        ({"revision": "0"}, "lowercase 40-character"),
        ({"compatible_version_prefix": "0.4"}, "major.minor"),
        ({"locked_version": "1.0.0"}, "compatible version line"),
        ({"role": 1}, "literal string"),
    )
    for changes, message in component_cases:
        with pytest.raises(ValueError, match=message):
            replace(component, **changes)

    lock_cases = (
        ({"lock_name": ""}, "not canonical"),
        ({"components": tuple(reversed(lock.components))}, "canonical"),
        ({"workflow_name": " "}, "not canonical"),
        ({"workflow_run_id": 0}, "at least 1"),
        ({"workflow_run_id": True}, "must be an integer"),
        ({"validated_date": " "}, "not canonical"),
        ({"python_version": " "}, "not canonical"),
        ({"tests_passed": 0}, "at least 1"),
        ({"tests_passed": False}, "must be an integer"),
        ({"bayesian_phystwin_tested_revision": "A" * 40}, "lowercase Git commit"),
        ({"bayesian_phystwin_tested_revision": "0" * 40}, "must match the locked"),
        ({"lock_sha256": "x"}, "SHA-256"),
    )
    for changes, message in lock_cases:
        with pytest.raises(ValueError, match=message):
            replace(lock, **changes)


def _exercise_runtime_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = _bundled_lock()
    installed = {"bpt": "0.4.0"}
    optional = validate_installed_ecosystem(lock, installed_versions=installed)
    assert optional.compatible
    assert optional.missing_components == ("prob4d", "causal4d")
    assert optional.incompatible_components == ()
    assert optional.to_dict()["compatible"] is True

    required = validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=installed,
    )
    assert not required.compatible
    assert required.incompatible_components == ("prob4d", "causal4d")

    exact = validate_installed_ecosystem(
        lock,
        require_all=True,
        exact_versions=True,
        installed_versions=_complete_versions(),
        revisions={
            "bpt": lock.component("bpt").revision,
            "prob4d": lock.component("prob4d").revision,
            "causal4d": lock.component("causal4d").revision,
        },
    )
    assert exact.compatible
    assert all(status.compatible for status in exact.components)
    assert all(status.to_dict()["installed"] for status in exact.components)

    compatible_line = deepcopy(_complete_versions())
    compatible_line["prob4d"] = "0.3.9"
    assert validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=compatible_line,
    ).compatible
    assert not validate_installed_ecosystem(
        lock,
        require_all=True,
        exact_versions=True,
        installed_versions=compatible_line,
    ).compatible

    wrong_line = deepcopy(_complete_versions())
    wrong_line["prob4d"] = "0.4.0"
    assert not validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=wrong_line,
    ).compatible
    assert not validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=_complete_versions(),
        revisions={"causal4d": "0" * 40},
    ).compatible
    assert not validate_installed_ecosystem(
        lock,
        installed_versions={"bpt": "0.4.0", "prob4d": None},
        revisions={"prob4d": lock.component("prob4d").revision},
    ).compatible
    assert not validate_installed_ecosystem(lock, installed_versions={}).compatible

    with pytest.raises(ValueError, match="duplicate installed version"):
        validate_installed_ecosystem(
            lock,
            installed_versions={"bpt": "0.4.0", "bayesian-phystwin": "0.4.0"},
        )
    with pytest.raises(ValueError, match="literal string"):
        validate_installed_ecosystem(
            lock,
            installed_versions={"bpt": 1},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="duplicate revision"):
        validate_installed_ecosystem(
            lock,
            installed_versions=installed,
            revisions={"bpt": "0" * 40, "bayesian-phystwin": "0" * 40},
        )
    with pytest.raises(ValueError, match="literal string"):
        validate_installed_ecosystem(
            lock,
            installed_versions=installed,
            revisions={"bpt": 1},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="lowercase 40-character"):
        validate_installed_ecosystem(
            lock,
            installed_versions=installed,
            revisions={"causal4d": "ABC"},
        )

    with monkeypatch.context() as scoped:
        versions = {
            component.package_name: component.locked_version
            for component in lock.components
        }
        scoped.setattr(metadata, "version", lambda name: versions[name])
        discovered = validate_installed_ecosystem(
            lock,
            require_all=True,
            exact_versions=True,
        )
        assert discovered.compatible

    with monkeypatch.context() as scoped:
        def only_bpt(name: str) -> str:
            if name == "bayesian-phystwin":
                return "0.4.0"
            raise metadata.PackageNotFoundError(name)

        scoped.setattr(metadata, "version", only_bpt)
        discovered = validate_installed_ecosystem()
        assert discovered.compatible
        assert discovered.missing_components == ("prob4d", "causal4d")


def _incompatible_report() -> EcosystemCompatibilityReportV1:
    lock = _bundled_lock()
    statuses = (
        EcosystemComponentStatusV1(
            component_id="bayesian_phystwin",
            package_name="bayesian-phystwin",
            installed=True,
            installed_version="0.4.0",
            locked_version="0.4.0",
            compatible_version_prefix="0.4.",
            version_compatible=True,
            exact_version_match=True,
            locked_revision=lock.component("bpt").revision,
            supplied_revision=None,
            revision_compatible=None,
            required=True,
            compatible=True,
        ),
        EcosystemComponentStatusV1(
            component_id="prob4d",
            package_name="prob4d",
            installed=False,
            installed_version=None,
            locked_version="0.3.1",
            compatible_version_prefix="0.3.",
            version_compatible=None,
            exact_version_match=None,
            locked_revision=lock.component("prob4d").revision,
            supplied_revision=None,
            revision_compatible=None,
            required=False,
            compatible=True,
        ),
        EcosystemComponentStatusV1(
            component_id="causal4d",
            package_name="causal4d",
            installed=True,
            installed_version="0.6.0",
            locked_version="0.5.0",
            compatible_version_prefix="0.5.",
            version_compatible=False,
            exact_version_match=False,
            locked_revision=lock.component("causal4d").revision,
            supplied_revision="0" * 40,
            revision_compatible=False,
            required=True,
            compatible=False,
        ),
    )
    return EcosystemCompatibilityReportV1(
        lock_name=lock.lock_name,
        lock_sha256=lock.lock_sha256,
        require_all=True,
        exact_versions=False,
        compatible=False,
        components=statuses,
    )


def _exercise_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "reports" / "ecosystem.json"
    assert ecosystem_main(["--json", "--output-json", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == persisted
    assert stdout["schema"] == "bayesian-phystwin.ecosystem-compatibility-report"
    assert stdout["components"]["bayesian_phystwin"]["installed"] is True

    lock_path = _write_payload(tmp_path, "explicit-lock", _lock_payload())
    assert ecosystem_main(["--lock", str(lock_path)]) == 0
    assert "ecosystem lock:" in capsys.readouterr().out

    incompatible = _incompatible_report()
    cli_module._print_human(incompatible)
    human = capsys.readouterr().out
    assert "compatible 0.4.0" in human
    assert "not installed (optional)" in human
    assert "incompatible 0.6.0, revision mismatch" in human
    assert "compatible: no" in human

    required_missing = replace(
        incompatible.components[1],
        required=True,
        compatible=False,
    )
    cli_module._print_human(
        replace(
            incompatible,
            components=(
                incompatible.components[0],
                required_missing,
                incompatible.components[2],
            ),
        )
    )
    assert "missing (required)" in capsys.readouterr().out

    captured: dict[str, object] = {}
    with monkeypatch.context() as scoped:
        def fake_validate(
            resolved_lock,
            *,
            require_all: bool,
            exact_versions: bool,
            revisions,
        ) -> EcosystemCompatibilityReportV1:
            captured.update(
                require_all=require_all,
                exact_versions=exact_versions,
                revisions=revisions,
                lock_name=resolved_lock.lock_name,
            )
            return incompatible

        scoped.setattr(cli_module, "validate_installed_ecosystem", fake_validate)
        lock = _bundled_lock()
        result = ecosystem_main(
            [
                "--require-all",
                "--exact-versions",
                "--revision",
                f"bpt={lock.component('bpt').revision}",
            ]
        )
        assert result == 1
        assert captured["require_all"] is True
        assert captured["exact_versions"] is True
        assert captured["revisions"] == {
            "bayesian_phystwin": lock.component("bpt").revision
        }
        assert "compatible: no" in capsys.readouterr().out

    lock = _bundled_lock()
    assert cli_module._revision_map(
        [
            f"bpt={lock.component('bpt').revision}",
            f"prob4d={lock.component('prob4d').revision}",
        ]
    ) == {
        "bayesian_phystwin": lock.component("bpt").revision,
        "prob4d": lock.component("prob4d").revision,
    }
    with pytest.raises(ValueError, match="COMPONENT=COMMIT"):
        cli_module._revision_map(["invalid"])
    with pytest.raises(ValueError, match="duplicate"):
        cli_module._revision_map(
            [
                f"bpt={lock.component('bpt').revision}",
                f"bayesian-phystwin={lock.component('bpt').revision}",
            ]
        )

    error_cases = (
        ["--revision", "invalid"],
        ["--revision", "unknown=" + "0" * 40],
        ["--revision", "causal4d=ABC"],
        ["--lock", str(tmp_path / "missing.json")],
    )
    for arguments in error_cases:
        with pytest.raises(SystemExit) as error:
            ecosystem_main(arguments)
        assert error.value.code == 2
        assert "usage: bpt ecosystem validate" in capsys.readouterr().err


def exercise_ecosystem_contract_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise public, serialization, discovery, and fail-closed boundaries."""

    lock = _bundled_lock()
    for selector in (
        "bpt",
        "bayesian-phystwin",
        "bayesian_phystwin",
        "PROB4D",
        "causal4d",
    ):
        assert normalize_ecosystem_component_id(selector) in {
            "bayesian_phystwin",
            "prob4d",
            "causal4d",
        }
    with pytest.raises(ValueError, match="unknown ecosystem component"):
        normalize_ecosystem_component_id("unknown")
    with pytest.raises(ValueError, match="not canonical"):
        normalize_ecosystem_component_id(" ")
    with pytest.raises(ValueError, match="literal string"):
        normalize_ecosystem_component_id(1)  # type: ignore[arg-type]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"first","schema":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict JSON"):
        load_ecosystem_compatibility_lock(duplicate)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        load_ecosystem_compatibility_lock(malformed)

    scalar = _write_payload(tmp_path, "scalar", [1, 2, 3])
    with pytest.raises(ValueError, match="must be an object"):
        load_ecosystem_compatibility_lock(scalar)

    for name, payload, message in _invalid_payload_cases():
        with pytest.raises(ValueError, match=message):
            load_ecosystem_compatibility_lock(
                _write_payload(tmp_path, name, payload)
            )

    _exercise_dataclass_validation(lock)
    _exercise_runtime_validation(monkeypatch)
    _exercise_cli(tmp_path, monkeypatch, capsys)


def test_ecosystem_contract_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exercise_ecosystem_contract_coverage(tmp_path, monkeypatch, capsys)


def test_causal4d_workflow_has_locked_and_canary_lanes() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "causal4d-provider-compatibility.yml"
    ).read_text(encoding="utf-8")
    assert "data/ecosystem_compatibility_v1.json" in workflow
    assert "needs.resolve-lock.outputs.causal4d_ref" in workflow
    assert "Latest Causal4D main canary" in workflow
    assert "continue-on-error: true" in workflow
    assert "persist-credentials: false" in workflow


def test_three_repository_workflow_uses_all_lock_lanes() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "three-repository-golden-path.yml"
    ).read_text(encoding="utf-8")
    assert "Resolve committed ecosystem lock" in workflow
    assert "locked_bpt_ref" in workflow
    assert "needs.resolve-lock.outputs.prob4d_ref" in workflow
    assert "needs.resolve-lock.outputs.causal4d_ref" in workflow
    assert "Current BPT + selected Prob4D/Causal4D" in workflow
    assert "Reproduce exact locked trio" in workflow
    assert "Latest three-repository main canary" in workflow
    assert "continue-on-error: true" in workflow
    assert "THREE_REPOSITORY_REQUIRE_LOCKED_REVISIONS" in workflow
    assert "persist-credentials: false" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert not (
        root / ".github" / "workflows" / "ecosystem-locked-golden-path.yml"
    ).exists()
