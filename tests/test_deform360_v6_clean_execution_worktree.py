from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"


def test_source_execution_uses_clean_exact_revision_worktree() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert ': "${BPT_SOURCE_SHA:?BPT_SOURCE_SHA is required}"' in text
    assert (
        'git worktree add --detach "${EXECUTION_REPO_ROOT}" "${BPT_SOURCE_SHA}"' in text
    )
    assert (
        'test "$(git -C "${EXECUTION_REPO_ROOT}" rev-parse HEAD)" = "${BPT_SOURCE_SHA}"'
    ) in text
    assert 'test -z "$(git -C "${EXECUTION_REPO_ROOT}" status --porcelain=v1)"' in text
    assert 'GITHUB_WORKSPACE="${EXECUTION_REPO_ROOT}" \\' in text
    assert 'git worktree remove --force "${EXECUTION_REPO_ROOT}"' in text
    assert text.index("git worktree add --detach") < text.index(
        'bash "${SELECTOR_WRAPPER}"'
    )


def test_clean_worktree_repair_does_not_modify_frozen_science_runner() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'SCIENCE_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"' in text
    assert (
        'SELECTOR_WRAPPER_BLOB_SHA="5958db6362917e6bc355b194abdac4736e39a5a4"' in text
    )
    assert 'RUNNER_WORKSPACE="${PHYSICAL_UPSTREAM_ROOT}" \\' in text
