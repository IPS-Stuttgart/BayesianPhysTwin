from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.ecosystem_validate import main as ecosystem_main
from bayesian_phystwin.ecosystem_compatibility import (
    load_ecosystem_compatibility_lock,
    validate_installed_ecosystem,
)


def _complete_versions() -> dict[str, str]:
    return {
        "bayesian_phystwin": "0.4.0",
        "prob4d": "0.3.1",
        "causal4d": "0.5.0",
    }


def test_bundled_ecosystem_lock_is_complete_and_content_addressed() -> None:
    lock = load_ecosystem_compatibility_lock()
    assert lock.lock_name == "three-repository-installed-wheel-v1"
    assert len(lock.lock_sha256) == 64
    assert tuple(component.component_id for component in lock.components) == (
        "bayesian_phystwin",
        "prob4d",
        "causal4d",
    )
    assert lock.component("bpt").revision == (
        "3c2c703f731a46019cf07b540474f25827dd5106"
    )
    assert lock.component("prob4d").revision == (
        "9ad07f89f9a85b68cf1375a4087ffa447b6af846"
    )
    assert lock.component("causal4d").revision == (
        "b0bf0c2de176b29534ef59484ad167b8f27d9dae"
    )


def test_optional_companions_and_require_all_have_distinct_semantics() -> None:
    lock = load_ecosystem_compatibility_lock()
    installed = {"bpt": "0.4.0"}

    optional = validate_installed_ecosystem(lock, installed_versions=installed)
    assert optional.compatible
    assert optional.missing_components == ("prob4d", "causal4d")
    assert optional.incompatible_components == ()

    required = validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=installed,
    )
    assert not required.compatible
    assert required.incompatible_components == ("prob4d", "causal4d")


def test_exact_versions_and_source_revisions_fail_closed() -> None:
    lock = load_ecosystem_compatibility_lock()
    compatible = validate_installed_ecosystem(
        lock,
        require_all=True,
        exact_versions=True,
        installed_versions=_complete_versions(),
        revisions={
            "prob4d": lock.component("prob4d").revision,
            "causal4d": lock.component("causal4d").revision,
        },
    )
    assert compatible.compatible

    wrong_version = dict(_complete_versions())
    wrong_version["prob4d"] = "0.4.0"
    assert not validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=wrong_version,
    ).compatible

    assert not validate_installed_ecosystem(
        lock,
        require_all=True,
        installed_versions=_complete_versions(),
        revisions={"causal4d": "0" * 40},
    ).compatible

    with pytest.raises(ValueError, match="lowercase 40-character"):
        validate_installed_ecosystem(
            lock,
            installed_versions={"bpt": "0.4.0"},
            revisions={"causal4d": "ABC"},
        )


def test_lock_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"first","schema":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict JSON"):
        load_ecosystem_compatibility_lock(path)


def test_ecosystem_cli_emits_machine_readable_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "ecosystem.json"
    assert ecosystem_main(["--json", "--output-json", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == persisted
    assert stdout["schema"] == "bayesian-phystwin.ecosystem-compatibility-report"
    assert stdout["components"]["bayesian_phystwin"]["installed"] is True


def test_compatibility_workflow_has_locked_and_canary_lanes() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "causal4d-provider-compatibility.yml"
    ).read_text(encoding="utf-8")
    assert "data/ecosystem_compatibility_v1.json" in workflow
    assert "needs.resolve-lock.outputs.causal4d_ref" in workflow
    assert "Latest Causal4D main canary" in workflow
    assert "continue-on-error: true" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
