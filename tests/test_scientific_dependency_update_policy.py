"""Keep policy-bearing dependencies outside routine dependency-bump PRs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DEPENDABOT = ROOT / ".github/dependabot.yml"
PR_TEMPLATE = ROOT / ".github/PULL_REQUEST_TEMPLATE.md"

FROZEN_SCIENTIFIC_RUNTIMES = {
    "mujoco": "3.9.0",
    "newton": "1.5.0",
    "pyrecest": "2.4.1",
}
MANUAL_DEPENDABOT_PIP_DEPENDENCIES = {"numpy", "pip"}


def _pip_update() -> dict[str, Any]:
    payload = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    matches = [
        update
        for update in payload["updates"]
        if update["package-ecosystem"] == "pip" and update["directory"] == "/"
    ]
    assert len(matches) == 1
    return dict(matches[0])


def _exact_project_pins() -> dict[str, str]:
    pattern = re.compile(
        r'^\s*"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^";]+)",\s*$',
        re.MULTILINE,
    )
    return {
        match.group("name").lower(): match.group("version")
        for match in pattern.finditer(PYPROJECT.read_text(encoding="utf-8"))
    }


def test_policy_bearing_dependencies_are_excluded_from_dependabot() -> None:
    assert _exact_project_pins() == FROZEN_SCIENTIFIC_RUNTIMES

    ignored = _pip_update().get("ignore")
    assert isinstance(ignored, list)
    assert all(set(entry) == {"dependency-name"} for entry in ignored)
    ignored_names = {entry["dependency-name"].lower() for entry in ignored}
    expected = set(FROZEN_SCIENTIFIC_RUNTIMES) | MANUAL_DEPENDABOT_PIP_DEPENDENCIES
    assert ignored_names == expected


def test_evidence_first_admission_is_part_of_every_pull_request() -> None:
    template = PR_TEMPLATE.read_text(encoding="utf-8")
    required = (
        "## Evidence-first admission",
        "registered evidence gate",
        "reproduced defect or interoperability failure",
        "reduces duplicated code, workflows, or public surface",
        "Owning issue, protocol, or reproduced defect:",
    )
    for marker in required:
        assert marker in template
