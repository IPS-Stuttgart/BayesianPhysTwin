from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
GITIGNORE = ROOT / ".gitignore"
PHYSICAL_VALIDATOR = (
    ROOT / "src/bayesian_phystwin/deform360_joint_sparse_physical_source_v5.py"
)

DEPENDENCY_CHECKOUTS = {
    "_deform360_physical": "lhy0807/deform360",
    "_official_phystwin": "Jianghanxiao/PhysTwin",
    "_sam2": "facebookresearch/sam2",
    "_causal4d_discovery": "IPS-Stuttgart/Causal4D",
}


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_only_exact_pinned_dependency_roots_are_ignored() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    ignored = {
        line.strip()
        for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    expected = {f"/{path}/" for path in DEPENDENCY_CHECKOUTS}
    assert expected <= ignored
    assert "/_*/" not in ignored
    assert "_*" not in ignored

    for path, repository in DEPENDENCY_CHECKOUTS.items():
        assert f"repository: {repository}" in workflow
        assert f"path: {path}" in workflow


def test_nested_checkouts_do_not_weaken_parent_source_cleanliness(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / ".gitignore").write_text(
        "".join(f"/{path}/\n" for path in DEPENDENCY_CHECKOUTS),
        encoding="utf-8",
    )
    (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "source.py")
    _git(
        repository,
        "-c",
        "user.name=contract-test",
        "-c",
        "user.email=contract-test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )

    for path in DEPENDENCY_CHECKOUTS:
        nested = repository / path
        nested.mkdir()
        (nested / ".git").mkdir()
        (nested / "PINNED_REVISION").write_text("fixture\n", encoding="utf-8")

    assert _git(repository, "status", "--porcelain").stdout == ""

    unexpected = repository / "_unexpected_dependency"
    unexpected.mkdir()
    (unexpected / "payload.py").write_text("VALUE = 2\n", encoding="utf-8")
    status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout.splitlines()
    assert status == ["?? _unexpected_dependency/payload.py"]

    (repository / "source.py").write_text("VALUE = 3\n", encoding="utf-8")
    status = _git(repository, "status", "--porcelain").stdout.splitlines()
    assert " M source.py" in status

    validator = PHYSICAL_VALIDATOR.read_text(encoding="utf-8")
    assert '_git_output(repo, "status", "--porcelain")' in validator
    assert '"repository is dirty"' in validator
