from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release-candidate.yml"
ACTION_REFERENCE = re.compile(r"^\s*uses:\s+([^\s@]+)@([^\s#]+)", re.MULTILINE)


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _requirements(relative: str) -> tuple[str, ...]:
    path = ROOT / relative
    return tuple(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


def test_release_candidate_is_read_only_and_never_publishes() -> None:
    text = _text()

    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "id-token: write" not in text
    assert "${{ secrets." not in text
    assert "persist-credentials: false" in text
    assert "gh-action-pypi-publish" not in text
    assert "twine upload" not in text
    assert "gh release" not in text


def test_release_candidate_uses_exact_revision_and_tag_version_gate() -> None:
    text = _text()

    assert (
        "SOURCE_REVISION: ${{ github.event.pull_request.head.sha || github.sha }}"
        in text
    )
    assert "ref: ${{ env.SOURCE_REVISION }}" in text
    assert '      - "v*"' in text
    assert '--expected-tag "${GITHUB_REF_NAME}"' in text
    assert '--source-revision "${SOURCE_REVISION}"' in text
    assert 'git show -s --format=%ct "${SOURCE_REVISION}"' in text
    assert 'echo "SOURCE_DATE_EPOCH=${source_date_epoch}"' in text


def test_release_build_resolver_input_is_exact_and_strict() -> None:
    text = _text()

    assert _requirements("requirements/release-build.txt") == (
        "pip==26.1.2",
        "build==1.5.0",
        "pip-audit==2.10.1",
        "setuptools==83.0.0",
        "twine==6.2.0",
        "wheel==0.47.0",
    )
    assert "--requirement requirements/release-build.txt" in text
    for command in (
        "python -m build --no-isolation",
        "python -m twine check --strict dist/*",
        "--format cyclonedx-json",
        "--build-environment build-environment.json",
        "--output release-evidence-base.json",
        "bind_release_matrix_contracts.py",
        "--output release-evidence.json",
    ):
        assert command in text


def test_release_runtime_resolver_inputs_are_exact() -> None:
    assert _requirements("requirements/release-runtime-py310-floor.txt") == (
        "pip==26.1.2",
        "setuptools==83.0.0",
        "wheel==0.47.0",
        "numpy==1.23.5",
    )
    assert _requirements("requirements/release-runtime-py312.txt") == (
        "pip==26.1.2",
        "setuptools==83.0.0",
        "wheel==0.47.0",
        "numpy==2.2.6",
    )
    assert _requirements("requirements/release-runtime-py314.txt") == (
        "pip==26.1.2",
        "setuptools==83.0.0",
        "wheel==0.47.0",
        "numpy==2.5.2",
    )


def test_release_matrix_covers_both_artifacts_and_all_supported_pythons() -> None:
    text = _text()

    for lane in (
        "py310-wheel-floor",
        "py310-sdist-floor",
        "py312-wheel",
        "py312-sdist",
        "py314-wheel",
        "py314-sdist",
    ):
        assert f"lane: {lane}" in text
    for version in (
        'python_version: "3.10"',
        'python_version: "3.12"',
        'python_version: "3.14"',
    ):
        assert version in text
    assert text.count("artifact_kind: wheel") == 3
    assert text.count("artifact_kind: sdist") == 3
    assert "--no-build-isolation" in text
    assert "--no-deps" in text
    assert 'test "${actual_numpy}" = "${EXPECTED_NUMPY_VERSION}"' in text


def test_release_matrix_binds_numerical_environment_and_resolver_input() -> None:
    text = _text()

    assert "bayesian_phystwin.numerical_environment_v1" in text
    assert "--dependency-lock \"${RESOLVER_INPUT}\"" in text
    assert "--require-dependency-lock" in text
    assert "build_release_matrix_evidence.py" in text
    assert "validation-receipt-${LANE}.json" in text
    assert "release-matrix-evidence.json" in text
    assert "release-matrix-summary.md" in text
    assert "pattern: release-validation-*-${{ env.SOURCE_REVISION }}" in text
    assert "merge-multiple: true" in text


def test_release_candidate_retains_all_evidence_files() -> None:
    text = _text()

    for path in (
        "build-environment.json",
        "dist/",
        "release-evidence-base.json",
        "release-evidence.json",
        "release-sbom.cdx.json",
        "release-summary.md",
        "release-matrix-evidence.json",
        "release-matrix-summary.md",
        "release-validation/",
    ):
        assert path in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text


def test_all_release_actions_are_pinned_to_full_commit_shas() -> None:
    references = ACTION_REFERENCE.findall(_text())

    assert references
    assert any(action == "actions/download-artifact" for action, _ in references)
    for action, revision in references:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{action} must use a full lowercase commit SHA, got {revision!r}"
        )
