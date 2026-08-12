from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci/check_changed_python.py"


def _load_preflight_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "bpt_check_changed_python",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


PREFLIGHT = _load_preflight_module()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _initialize_repository(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "preflight@example.invalid")
    _git(path, "config", "user.name", "Preflight Test")
    (path / "ruff.toml").write_text(
        'target-version = "py310"\n\n[lint]\n'
        'select = ["E4", "E7", "E9", "F", "I", "UP", "B"]\n',
        encoding="utf-8",
    )


def _run_preflight(
    repository: Path,
    *,
    base: str,
    head: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            base,
            "--head",
            head,
            "--repository-root",
            str(repository),
            "--chunk-size",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _scan_source(tmp_path: Path, source: str) -> tuple[object, ...]:
    relative = Path("src/bayesian_phystwin/example.py")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    return PREFLIGHT.scientific_boundary_violations(tmp_path, (relative,))


def test_checks_only_existing_changed_python_files(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    (repository / "changed.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "deleted.py").write_text("OLD = True\n", encoding="utf-8")
    (repository / "notes.txt").write_text("original\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")

    (repository / "changed.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repository / "new.py").write_text(
        "def answer() -> int:\n    return 42\n",
        encoding="utf-8",
    )
    (repository / "deleted.py").unlink()
    (repository / "notes.txt").write_text("changed\n", encoding="utf-8")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", "head")
    head = _git(repository, "rev-parse", "HEAD")

    result = _run_preflight(repository, base=base, head=head)

    assert result.returncode == 0, result.stderr
    assert "changed.py" in result.stdout
    assert "new.py" in result.stdout
    assert "deleted.py" not in result.stdout
    assert "notes.txt" not in result.stdout


def test_fails_before_expensive_work_on_unformatted_python(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")

    (repository / "module.py").write_text("VALUE=2\n", encoding="utf-8")
    _git(repository, "add", "module.py")
    _git(repository, "commit", "-m", "unformatted")
    head = _git(repository, "rev-parse", "HEAD")

    result = _run_preflight(repository, base=base, head=head)

    assert result.returncode != 0
    assert "module.py" in result.stdout


def test_rejects_unavailable_revision(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    head = _git(repository, "rev-parse", "HEAD")

    result = _run_preflight(repository, base="0" * 40, head=head)

    assert result.returncode != 0
    assert "base_revision is not an available commit" in result.stderr


def test_rejects_checkout_that_does_not_match_requested_head(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")

    (repository / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "add", "module.py")
    _git(repository, "commit", "-m", "head")
    head = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--detach", base)

    result = _run_preflight(repository, base=base, head=head)

    assert result.returncode != 0
    assert "repository HEAD does not match head_revision" in result.stderr


@pytest.mark.skipif(not hasattr(Path, "is_symlink"), reason="symlinks unsupported")
def test_rejects_changed_python_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    (repository / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")

    link = repository / "linked.py"
    try:
        link.symlink_to("target.py")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    _git(repository, "add", "linked.py")
    _git(repository, "commit", "-m", "symlink")
    head = _git(repository, "rev-parse", "HEAD")

    result = _run_preflight(repository, base=base, head=head)

    assert result.returncode != 0
    assert "must not be a symlink" in result.stderr


def test_rejects_truthy_boolean_mask_coercion(tmp_path: Path) -> None:
    violations = _scan_source(
        tmp_path,
        "import numpy as np\n\n"
        "def admit(available: object) -> np.ndarray:\n"
        "    return np.asarray(available, dtype=bool)\n",
    )

    assert [violation.code for violation in violations] == ["BPTQ001"]
    assert "truth-coerces NaN" in violations[0].message


def test_rejects_falsey_configuration_defaulting(tmp_path: Path) -> None:
    violations = _scan_source(
        tmp_path,
        "class ExampleConfig:\n"
        "    pass\n\n"
        "def select(config: object | None = None) -> object:\n"
        "    return config or ExampleConfig()\n",
    )

    assert [violation.code for violation in violations] == ["BPTQ002"]
    assert "branch on `is None`" in violations[0].message


def test_allows_explicit_mask_and_configuration_admission(tmp_path: Path) -> None:
    violations = _scan_source(
        tmp_path,
        "import numpy as np\n\n"
        "class ExampleConfig:\n"
        "    pass\n\n"
        "def admit(\n"
        "    available: object,\n"
        "    config: ExampleConfig | None = None,\n"
        ") -> tuple[np.ndarray, ExampleConfig]:\n"
        "    raw_mask = np.asarray(available)\n"
        "    if raw_mask.dtype != np.dtype(np.bool_):\n"
        "        raise ValueError('availability must contain only booleans')\n"
        "    mask = np.asarray(raw_mask, dtype=np.bool_)\n"
        "    cfg = ExampleConfig() if config is None else config\n"
        "    return mask, cfg\n",
    )

    assert violations == ()


def test_allows_audited_boolean_coercion_suppression(tmp_path: Path) -> None:
    violations = _scan_source(
        tmp_path,
        "import numpy as np\n\n"
        "def normalize(values: object) -> np.ndarray:\n"
        "    return np.asarray(\n"
        "        values, dtype=bool\n"
        "    )  # bpt-quality: allow BPTQ001 -- internal 0/1 sentinel\n",
    )

    assert violations == ()


def test_fast_preflight_reports_boundary_pattern_before_ruff(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_repository(repository)
    source = repository / "src/bayesian_phystwin/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")

    source.write_text(
        "import numpy as np\n\n"
        "def admit(available: object) -> np.ndarray:\n"
        "    return np.asarray(available, dtype=bool)\n",
        encoding="utf-8",
    )
    _git(repository, "add", "src/bayesian_phystwin/module.py")
    _git(repository, "commit", "-m", "unsafe mask coercion")
    head = _git(repository, "rev-parse", "HEAD")

    result = _run_preflight(repository, base=base, head=head)

    assert result.returncode != 0
    assert "BPTQ001" in result.stderr
    assert "unsafe scientific-boundary pattern" in result.stderr
