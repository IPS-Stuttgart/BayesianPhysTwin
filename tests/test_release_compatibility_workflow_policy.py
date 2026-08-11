from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/release-compatibility-evidence.yml"
RELEASE_INPUT = ROOT / "requirements/release-build-py312.txt"
NUMPY_FLOOR = ROOT / "requirements/numpy-floor-py310.txt"
ACTION_REFERENCE = re.compile(r"^\s*uses:\s+([^\s@]+)@([^\s#]+)", re.MULTILINE)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _requirements(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_release_compatibility_workflow_is_read_only_and_exact_head_bound() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "id-token: write" not in text
    assert "${{ secrets." not in text
    assert "persist-credentials: false" in text
    assert (
        "SOURCE_REVISION: ${{ github.event.pull_request.head.sha || github.sha }}"
        in text
    )
    assert "ref: ${{ env.SOURCE_REVISION }}" in text


def test_release_artifacts_are_built_and_installed_on_every_supported_python() -> None:
    text = _workflow_text()

    assert 'python-version: ["3.10", "3.12", "3.14"]' in text
    assert "python -m build --no-isolation" in text
    assert "python -m twine check --strict dist/*" in text
    assert "for kind in wheel sdist" in text
    assert "--no-build-isolation" in text
    assert "import bayesian_phystwin.v1 as api" in text
    assert 'find dist -maxdepth 1 -name \'*.whl\'' in text
    assert 'find dist -maxdepth 1 -name \'*.tar.gz\'' in text


def test_declared_numpy_floor_is_exact_and_runs_the_core_contracts() -> None:
    text = _workflow_text()

    assert _requirements(NUMPY_FLOOR) == ("numpy==1.23.0",)
    assert "Core contracts / Python 3.10 / NumPy 1.23.0" in text
    assert "--constraint requirements/numpy-floor-py310.txt" in text
    assert 'numpy.__version__ != "1.23.0"' in text
    assert "test_suite_manifest.py list core-contracts" in text
    assert 'python -m pytest -q "${floor_tests[@]}"' in text


def test_numerical_profile_binds_the_checked_in_resolver_input() -> None:
    text = _workflow_text()
    requirements = _requirements(RELEASE_INPUT)

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert "numpy==2.2.6" in requirements
    assert "pip==26.1.2" in requirements
    assert "--requirement requirements/release-build-py312.txt" in text
    assert "numerical_environment_v1 capture" in text
    assert "numerical_environment_v1 validate" in text
    assert "--dependency-lock requirements/release-build-py312.txt" in text
    assert "--require-dependency-lock" in text
    assert 'profile["dependency_lock"]["sha256"]' in text
    assert "release-resolved-environment.txt" in text
    assert "release-numerical-environment.json" in text
    assert "retention-days: 30" in text


def test_all_release_compatibility_actions_are_commit_pinned() -> None:
    references = ACTION_REFERENCE.findall(_workflow_text())

    assert references
    for action, revision in references:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{action} must use a full lowercase commit SHA, got {revision!r}"
        )
