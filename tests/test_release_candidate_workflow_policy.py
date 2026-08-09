from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release-candidate.yml"
ACTION_REFERENCE = re.compile(r"^\s*uses:\s+([^\s@]+)@([^\s#]+)", re.MULTILINE)


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


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
        'SOURCE_REVISION: ${{ github.event.pull_request.head.sha || github.sha }}'
        in text
    )
    assert 'ref: ${{ env.SOURCE_REVISION }}' in text
    assert '      - "v*"' in text
    assert '--expected-tag "${GITHUB_REF_NAME}"' in text
    assert '--source-revision "${SOURCE_REVISION}"' in text
    assert 'git show -s --format=%ct "${SOURCE_REVISION}"' in text
    assert 'echo "SOURCE_DATE_EPOCH=${source_date_epoch}"' in text


def test_release_toolchain_and_checks_are_pinned_and_strict() -> None:
    text = _text()

    required = (
        '"pip==26.1.2"',
        '"build==1.5.0"',
        '"pip-audit==2.10.1"',
        '"setuptools==83.0.0"',
        '"twine==6.2.0"',
        '"wheel==0.47.0"',
        "python -m build --no-isolation",
        "python -m twine check --strict dist/*",
        "--format cyclonedx-json",
        "--build-environment build-environment.json",
        "--output release-evidence.json",
    )
    for value in required:
        assert value in text


def test_release_candidate_retains_all_evidence_files() -> None:
    text = _text()

    for path in (
        "build-environment.json",
        "dist/",
        "release-evidence.json",
        "release-sbom.cdx.json",
        "release-summary.md",
    ):
        assert path in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text


def test_all_release_actions_are_pinned_to_full_commit_shas() -> None:
    references = ACTION_REFERENCE.findall(_text())

    assert references
    for action, revision in references:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{action} must use a full lowercase commit SHA, got {revision!r}"
        )
