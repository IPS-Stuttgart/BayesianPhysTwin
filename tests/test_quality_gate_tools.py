import json
import subprocess
from pathlib import Path

import pytest

from tools import check_ruff_baseline as ruff_baseline_module
from tools.check_changed_python_quality import changed_python_files
from tools.check_changed_semantic_coverage import (
    check_changed_semantic_coverage,
    coverage_failures,
    parse_added_lines,
)
from tools.check_ruff_baseline import RuffDiagnostic, check_ruff_regression


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
    ).strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Quality Gate Test")
    _git(repository, "config", "user.email", "quality@example.invalid")
    return repository


def test_changed_python_files_only_returns_existing_python_paths(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / "kept.py").write_text("VALUE = 1\n")
    (repository / "deleted.py").write_text("VALUE = 2\n")
    (repository / "notes.txt").write_text("initial\n")
    base = _commit(repository, "base")

    (repository / "kept.py").write_text("VALUE = 3\n")
    (repository / "added.pyi").write_text("VALUE: int\n")
    (repository / "deleted.py").unlink()
    (repository / "notes.txt").write_text("changed\n")
    head = _commit(repository, "head")

    assert changed_python_files(repository, base=base, head=head) == (
        "added.pyi",
        "kept.py",
    )


def test_parse_added_lines_understands_zero_context_hunks() -> None:
    diff = """@@ -2 +2,3 @@
-old
+new
+branch
+body
@@ -9,0 +12 @@
+tail
"""

    assert parse_added_lines(diff) == frozenset({2, 3, 4, 12})


def test_coverage_failures_report_changed_lines_and_branch_origins() -> None:
    failures = coverage_failures(
        frozenset({2, 3, 4}),
        {
            "executed_lines": [2, 3],
            "missing_lines": [4, 8],
            "excluded_lines": [],
            "missing_branches": [[2, 4], [9, 10]],
        },
    )

    assert failures == (
        "uncovered changed lines: [4]",
        "uncovered changed branches: [(2, 4)]",
    )


def test_changed_semantic_coverage_accepts_complete_diff_coverage(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    relative = Path("src/bayesian_phystwin/observation_belief.py")
    source = repository / relative
    source.parent.mkdir(parents=True)
    source.write_text("def choose(flag: bool) -> int:\n    return int(flag)\n")
    base = _commit(repository, "base")

    source.write_text(
        "def choose(flag: bool) -> int:\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 0\n"
    )
    head = _commit(repository, "head")
    coverage_json = repository / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "files": {
                    str(source.resolve()): {
                        "executed_lines": [1, 2, 3, 4],
                        "missing_lines": [],
                        "excluded_lines": [],
                        "executed_branches": [[2, 3], [2, 4]],
                        "missing_branches": [],
                    }
                }
            }
        )
    )

    changed = check_changed_semantic_coverage(
        repository,
        coverage_json,
        base=base,
        head=head,
    )

    assert set(changed) == {relative.as_posix()}


def test_changed_semantic_coverage_rejects_an_uncovered_new_branch(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    relative = Path("src/bayesian_phystwin/observation_belief.py")
    source = repository / relative
    source.parent.mkdir(parents=True)
    source.write_text("def choose(flag: bool) -> int:\n    return int(flag)\n")
    base = _commit(repository, "base")

    source.write_text(
        "def choose(flag: bool) -> int:\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 0\n"
    )
    head = _commit(repository, "head")
    coverage_json = repository / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "files": {
                    relative.as_posix(): {
                        "executed_lines": [1, 2, 3],
                        "missing_lines": [4],
                        "excluded_lines": [],
                        "executed_branches": [[2, 3]],
                        "missing_branches": [[2, 4]],
                    }
                }
            }
        )
    )

    with pytest.raises(SystemExit, match="uncovered changed branches"):
        check_changed_semantic_coverage(
            repository,
            coverage_json,
            base=base,
            head=head,
        )


def _ruff_diagnostics(repository_root: Path, count: int) -> list[RuffDiagnostic]:
    return [
        RuffDiagnostic(
            filename=str(repository_root / "src/example.py"),
            code="F401",
            message="unused import",
        )
        for _ in range(count)
    ]


def _ruff_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = _repository(tmp_path)
    source = repository / "src/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("BASE = True\n")
    base = _commit(repository, "base")
    source.write_text("HEAD = True\n")
    head = _commit(repository, "head")
    return repository, base, head


def test_repository_ruff_gate_accepts_removed_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, base, head = _ruff_repository(tmp_path)

    def fake_run_ruff(
        repository_root: Path,
        paths: tuple[str, ...],
    ) -> list[RuffDiagnostic]:
        del paths
        is_base = (repository_root / "src/example.py").read_text() == "BASE = True\n"
        return _ruff_diagnostics(repository_root, 2 if is_base else 1)

    monkeypatch.setattr(ruff_baseline_module, "_run_ruff", fake_run_ruff)
    baseline_report = tmp_path / "baseline-report.json"
    current_report = tmp_path / "current-report.json"

    assert (
        check_ruff_regression(
            repository,
            base=base,
            head=head,
            paths=("src",),
            baseline_report_path=baseline_report,
            current_report_path=current_report,
        )
        == 0
    )
    assert len(json.loads(baseline_report.read_text())) == 2
    assert len(json.loads(current_report.read_text())) == 1


def test_repository_ruff_gate_rejects_increased_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, base, head = _ruff_repository(tmp_path)

    def fake_run_ruff(
        repository_root: Path,
        paths: tuple[str, ...],
    ) -> list[RuffDiagnostic]:
        del paths
        is_base = (repository_root / "src/example.py").read_text() == "BASE = True\n"
        return _ruff_diagnostics(repository_root, 1 if is_base else 2)

    monkeypatch.setattr(ruff_baseline_module, "_run_ruff", fake_run_ruff)

    assert (
        check_ruff_regression(
            repository,
            base=base,
            head=head,
            paths=("src",),
        )
        == 1
    )
