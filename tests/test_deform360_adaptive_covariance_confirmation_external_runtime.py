from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import bayesian_phystwin
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime as runtime
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock_module
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_probe(
    repository: Path,
    *arguments: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(repository / "scripts" / "probe.py"), *arguments],
        cwd=repository,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def _provenance_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "provenance-repository"
    config_root = repository / "configs" / "sota"
    config_root.mkdir(parents=True)
    (config_root / ".gitkeep").write_text("", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n",
        encoding="utf-8",
    )
    package = repository / "src" / "bayesian_phystwin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '"""Minimal provenance-test package."""\n',
        encoding="utf-8",
    )
    shutil.copyfile(
        Path(runtime.__file__),
        package / "deform360_adaptive_covariance_confirmation_external_runtime.py",
    )
    shutil.copyfile(
        Path(lock_module.__file__),
        package / "deform360_adaptive_covariance_confirmation_lock.py",
    )
    probe = repository / "scripts" / "probe.py"
    probe.parent.mkdir()
    probe.write_text(
        """
from __future__ import annotations
import argparse
from pathlib import Path
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_h1_lock_generation_entrypoint,
    validate_confirmation_h2_production_entrypoint,
)

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("h1", "h2"))
parser.add_argument("--repository", required=True)
parser.add_argument("--lock", required=True)
parser.add_argument("--h1", required=True)
parser.add_argument("--h2")
parser.add_argument("--entrypoint-file")
args = parser.parse_args()
entrypoint = args.entrypoint_file or __file__
if args.phase == "h1":
    result = validate_confirmation_h1_lock_generation_entrypoint(
        args.repository,
        args.lock,
        args.h1,
        entrypoint_file=entrypoint,
        entrypoint_repository_path="scripts/probe.py",
    )
else:
    result = validate_confirmation_h2_production_entrypoint(
        args.repository,
        args.lock,
        args.h2,
        expected_h1=args.h1,
        entrypoint_file=entrypoint,
        entrypoint_repository_path="scripts/probe.py",
    )
print(result["implementation_commit_h1"])
""".lstrip(),
        encoding="utf-8",
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Confirmation Test")
    _git(repository, "config", "user.email", "confirmation@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "Freeze H1")
    return repository, _git(repository, "rev-parse", "HEAD")


def _load_external_stage_wrapper() -> object:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_deform360_adaptive_confirmation_external_stage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_adaptive_confirmation_external_stage",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_outcome_stage_wrapper() -> object:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_deform360_adaptive_confirmation_outcome_stage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_adaptive_confirmation_outcome_stage",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cli_bootstrap() -> object:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_deform360_adaptive_confirmation_cli.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_adaptive_confirmation_cli_bootstrap",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_wrapper_rejects_abbreviations_of_bound_stage_paths() -> None:
    wrapper = _load_external_stage_wrapper()
    reserved = {"--protocol", "--repo", "--deform360-repo"}
    for argument in (
        "--prot=/tmp/other",
        "--rep=/tmp/other",
        "--deform360-r=/tmp/other",
    ):
        with pytest.raises(ValueError, match="reserved option abbreviation"):
            wrapper._reject_reserved_option_abbreviations([argument], reserved)
    wrapper._reject_reserved_option_abbreviations(
        ["--staged-case-dir", "/tmp/case"],
        reserved,
    )


def test_external_wrapper_requires_declared_h1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _load_external_stage_wrapper()
    arguments = [
        "external-wrapper",
        "--adapter-repo",
        "/adapter",
        "--execution-repo",
        "/execution",
        "--deform360-repo",
        "/deform360",
        "--lock",
        "/lock.json",
        "--h2-commit",
        "b" * 40,
        "--stage",
        "frame-zero",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        wrapper._parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [*arguments, "--expected-h1", "a" * 40],
    )
    parsed, remaining = wrapper._parse_args()
    assert parsed.expected_h1 == "a" * 40
    assert remaining == []


def test_outcome_wrapper_does_not_accept_outer_path_abbreviations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _load_outcome_stage_wrapper()
    required = (
        ("--adapter-repo", "/adapter"),
        ("--execution-repo", "/execution"),
        ("--deform360-repo", "/deform360"),
        ("--lock", "/lock.json"),
        ("--h2-commit", "b" * 40),
        ("--expected-h1", "a" * 40),
        ("--barrier", "/barrier.json"),
        ("--case-root", "/cases"),
        ("--measurement-root", "/measurements"),
        ("--compatibility-root", "/compatibility"),
        ("--stage", "authorized-future"),
    )
    for full, abbreviation in (
        ("--barrier", "--bar"),
        ("--case-root", "--case-r"),
        ("--compatibility-root", "--compatibility-r"),
    ):
        argv = ["outcome-wrapper"]
        for option, value in required:
            argv.extend([abbreviation if option == full else option, value])
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            wrapper._parse_args()


def test_direct_wrappers_reject_adapter_cache_before_import(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    (adapter / "src").mkdir(parents=True)
    (adapter / "scripts").mkdir()
    cache = adapter / "src" / "bayesian_phystwin" / "__pycache__"
    cache.mkdir(parents=True)
    cache.joinpath("forged.cpython-312.pyc").write_bytes(b"forged")

    for wrapper in (
        _load_external_stage_wrapper(),
        _load_outcome_stage_wrapper(),
        _load_cli_bootstrap(),
    ):
        with pytest.raises(ValueError, match="bytecode cache is forbidden"):
            wrapper._reject_adapter_python_caches(adapter)


def test_cli_bootstrap_requires_one_exact_adapter_repository_option(
    tmp_path: Path,
) -> None:
    bootstrap = _load_cli_bootstrap()
    assert (
        bootstrap._adapter_repository(["evaluate", "--adapter-repo", str(tmp_path)])
        == tmp_path.absolute()
    )
    with pytest.raises(ValueError, match="abbreviation is forbidden"):
        bootstrap._adapter_repository(["evaluate", f"--adapter-r={tmp_path}"])
    with pytest.raises(ValueError, match="exactly one"):
        bootstrap._adapter_repository(
            [
                "evaluate",
                "--adapter-repo",
                str(tmp_path),
                f"--adapter-repo={tmp_path}",
            ]
        )


def test_h1_and_h2_entrypoint_provenance_fail_closed(
    tmp_path: Path,
) -> None:
    repository, h1 = _provenance_repository(tmp_path)
    lock_path = repository / runtime.COHORT_LOCK_REPOSITORY_PATH
    common = (
        "--repository",
        str(repository),
        "--lock",
        str(lock_path),
        "--h1",
        h1,
    )

    h1_valid = _run_probe(repository, "h1", *common, check=True)
    assert h1_valid.stdout.strip() == h1

    noncanonical = _run_probe(
        repository,
        "h1",
        "--repository",
        str(repository),
        "--lock",
        str(repository / "not-the-canonical-lock.json"),
        "--h1",
        h1,
    )
    assert noncanonical.returncode != 0
    assert "not the canonical repository path" in noncanonical.stderr

    outside = tmp_path / "outside-probe.py"
    outside.write_text("# not the committed production entrypoint\n", encoding="utf-8")
    wrong_source = _run_probe(
        repository,
        "h1",
        *common,
        "--entrypoint-file",
        str(outside),
    )
    assert wrong_source.returncode != 0
    assert "does not match its direct caller" in wrong_source.stderr

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = lock_module.build_confirmation_cohort_lock(h1)
    lock_bytes = (
        json.dumps(lock_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    lock_path.write_bytes(lock_bytes)
    _git(repository, "add", runtime.COHORT_LOCK_REPOSITORY_PATH)
    _git(repository, "commit", "-q", "-m", "Freeze H2 lock")
    h2 = _git(repository, "rev-parse", "HEAD")

    h2_valid = _run_probe(
        repository,
        "h2",
        *common,
        "--h2",
        h2,
        check=True,
    )
    assert h2_valid.stdout.strip() == h1

    cache = (
        repository
        / "src"
        / "bayesian_phystwin"
        / "__pycache__"
        / "forged.cpython-312.pyc"
    )
    cache.parent.mkdir()
    cache.write_bytes(b"forged bytecode")
    cached = _run_probe(repository, "h2", *common, "--h2", h2)
    assert cached.returncode != 0
    assert "Python bytecode cache is forbidden" in cached.stderr
    shutil.rmtree(cache.parent)

    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = _run_probe(repository, "h2", *common, "--h2", h2)
    assert dirty.returncode != 0
    assert "adapter checkout is dirty" in dirty.stderr
    (repository / "untracked.txt").unlink()

    lock_path.write_bytes(lock_bytes + b" ")
    modified_h2 = _run_probe(repository, "h2", *common, "--h2", h2)
    assert modified_h2.returncode != 0
    assert "adapter checkout is dirty" in modified_h2.stderr


def test_h1_entrypoint_rejects_a_preexisting_committed_lock(
    tmp_path: Path,
) -> None:
    repository, _parent = _provenance_repository(tmp_path)
    lock_path = repository / runtime.COHORT_LOCK_REPOSITORY_PATH
    lock_path.write_text('{"premature":true}\n', encoding="utf-8")
    _git(repository, "add", runtime.COHORT_LOCK_REPOSITORY_PATH)
    _git(repository, "commit", "-q", "-m", "Invalid H1 with cohort lock")
    invalid_h1 = _git(repository, "rev-parse", "HEAD")

    rejected = _run_probe(
        repository,
        "h1",
        "--repository",
        str(repository),
        "--lock",
        str(lock_path),
        "--h1",
        invalid_h1,
    )
    assert rejected.returncode != 0
    assert "already exists at H1" in rejected.stderr


def test_two_commit_validator_requires_direct_single_lock_child(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Confirmation Test")
    _git(repository, "config", "user.email", "confirmation@example.invalid")
    (repository / "README.md").write_text("H1\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-q", "-m", "Freeze implementation")
    h1 = _git(repository, "rev-parse", "HEAD")

    lock = repository / runtime.COHORT_LOCK_REPOSITORY_PATH
    lock.parent.mkdir(parents=True)
    lock.write_text('{"lock":true}\n', encoding="utf-8")
    _git(repository, "add", runtime.COHORT_LOCK_REPOSITORY_PATH)
    _git(repository, "commit", "-q", "-m", "Freeze cohort")
    h2 = _git(repository, "rev-parse", "HEAD")

    result = runtime.validate_two_commit_execution_repository(
        repository,
        lock,
        h1_commit=h1,
        h2_commit=h2,
    )
    assert result["implementation_commit_h1"] == h1
    assert result["cohort_lock_commit_h2"] == h2
    assert (
        result["cohort_lock_file_sha256"]
        == hashlib.sha256(lock.read_bytes()).hexdigest()
    )

    lock.write_text('{"lock":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        runtime.validate_two_commit_execution_repository(
            repository,
            lock,
            h1_commit=h1,
            h2_commit=h2,
        )


def test_two_commit_validator_rejects_lock_modified_instead_of_added(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Confirmation Test")
    _git(repository, "config", "user.email", "confirmation@example.invalid")
    lock = repository / runtime.COHORT_LOCK_REPOSITORY_PATH
    lock.parent.mkdir(parents=True)
    lock.write_text('{"phase":"H1"}\n', encoding="utf-8")
    _git(repository, "add", runtime.COHORT_LOCK_REPOSITORY_PATH)
    _git(repository, "commit", "-q", "-m", "Freeze implementation")
    h1 = _git(repository, "rev-parse", "HEAD")

    lock.write_text('{"phase":"H2"}\n', encoding="utf-8")
    _git(repository, "add", runtime.COHORT_LOCK_REPOSITORY_PATH)
    _git(repository, "commit", "-q", "-m", "Modify cohort")
    h2 = _git(repository, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="must add only"):
        runtime.validate_two_commit_execution_repository(
            repository,
            lock,
            h1_commit=h1,
            h2_commit=h2,
        )


def test_activation_preempts_same_named_adapter_modules_and_restores_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_package = tmp_path / "src" / "bayesian_phystwin"
    external_package.mkdir(parents=True)
    suffixes = (
        "deform360_bias_aware_prospective_artifacts",
        "deform360_bias_aware_prospective_physical",
        "deform360_bias_aware_prospective_uncertainty",
    )
    for suffix in suffixes:
        body = "PROTOCOL_ID = 'external-original'\n"
        if suffix.endswith("artifacts"):
            body += (
                "def load_bias_aware_prospective_protocol(*args, **kwargs):\n"
                "    return 'external'\n"
                "def prospective_case_records(*args, **kwargs):\n"
                "    return ()\n"
                "def prospective_case_record(*args, **kwargs):\n"
                "    return {}\n"
            )
        (external_package / f"{suffix}.py").write_text(body, encoding="utf-8")

    monkeypatch.setattr(runtime, "EXTERNAL_MODULE_SUFFIXES", suffixes)
    monkeypatch.setattr(
        runtime,
        "validate_external_execution_repository",
        lambda _: {},
    )
    original_path = tuple(bayesian_phystwin.__path__)
    names = tuple(f"bayesian_phystwin.{suffix}" for suffix in suffixes)
    assert all(name not in sys.modules for name in names)

    with runtime.activate_confirmation_external_runtime(tmp_path) as modules:
        assert all(
            external_package in Path(str(module.__file__)).resolve().parents
            for module in modules.values()
        )
        assert all(module.PROTOCOL_ID == PROTOCOL_ID for module in modules.values())
        observed = runtime.validate_external_module_provenance(tmp_path)
        assert set(observed) == set(names)

    assert tuple(bayesian_phystwin.__path__) == original_path
    assert all(name not in sys.modules for name in names)
