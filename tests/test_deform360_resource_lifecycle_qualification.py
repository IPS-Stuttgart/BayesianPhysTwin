from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/development/qualify_deform360_resource_lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deform360_resource_lifecycle_qualification", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def _write_dataset(root: Path) -> Path:
    root.mkdir()
    (root / "images").mkdir()
    (root / "images/camera-a.png").write_bytes(b"camera-a")
    (root / "images/camera-b.png").write_bytes(b"camera-b")
    (root / "visual_hull_seed.ply").write_bytes(b"ply\nseed")
    (root / "outputs").mkdir()
    (root / "outputs/old-checkpoint.ckpt").write_bytes(b"must-not-copy")
    (root / "unreferenced.png").write_bytes(b"must-not-copy")
    transforms = {
        "camera_model": "OPENCV",
        "ply_file_path": str((root / "visual_hull_seed.ply").resolve()),
        "frames": [
            {"file_path": "./images/camera-a.png", "transform_matrix": [[1.0]]},
            {"file_path": "images/camera-b.png", "transform_matrix": [[2.0]]},
        ],
    }
    (root / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    return root


def test_formal_held_paths_are_rejected_before_resolution() -> None:
    formal = (
        qualification.FORMAL_HELD_PARENT
        / "held-v8-attempt-4-withdrawn-postbarrier/private-target.npz"
    )
    with pytest.raises(ValueError, match="formal held root"):
        qualification._assert_nonheld_path(formal, label="test path", must_exist=False)

    runtime_named_held = Path(
        "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/runtime/bin/python"
    )
    assert (
        qualification._assert_nonheld_path(
            runtime_named_held, label="runtime", must_exist=False
        )
        == runtime_named_held
    )


def test_materialize_dataset_copies_only_references_and_rewrites_seed(
    tmp_path: Path,
) -> None:
    source = _write_dataset(tmp_path / "source")
    destination = tmp_path / "materialized"

    audit = qualification._materialize_dataset(source, destination)

    rewritten = json.loads((destination / "transforms.json").read_text())
    assert rewritten["ply_file_path"] == str(
        (destination / "visual_hull_seed.ply").resolve()
    )
    assert (destination / "images/camera-a.png").read_bytes() == b"camera-a"
    assert (destination / "images/camera-b.png").read_bytes() == b"camera-b"
    assert (destination / "visual_hull_seed.ply").read_bytes() == b"ply\nseed"
    assert not (destination / "outputs").exists()
    assert not (destination / "unreferenced.png").exists()
    assert audit["rewritten_field"] == "ply_file_path"
    assert audit["frame_count"] == 2
    assert audit["copied_regular_file_count"] == 4
    assert audit["unreferenced_outputs_copied"] is False
    assert len(audit["portable_transforms_sha256"]) == 64
    assert set(audit["materialized_records"]) == {
        "images/camera-a.png",
        "images/camera-b.png",
        "transforms.json",
        "visual_hull_seed.ply",
    }


def test_materialize_dataset_rejects_frame_escape(tmp_path: Path) -> None:
    source = _write_dataset(tmp_path / "source")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    transforms = json.loads((source / "transforms.json").read_text())
    transforms["frames"][0]["file_path"] = "../outside.png"
    (source / "transforms.json").write_text(json.dumps(transforms))

    with pytest.raises(ValueError, match="escapes the dataset"):
        qualification._materialize_dataset(source, tmp_path / "materialized")
    assert not (tmp_path / "materialized").exists()


def test_nonheld_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    linked = tmp_path / "linked.bin"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not support symlinks")

    with pytest.raises(ValueError, match="is a symlink"):
        qualification._read_regular_nofollow(linked, label="linked input")


def test_materialized_identity_ignores_private_absolute_root(tmp_path: Path) -> None:
    source = _write_dataset(tmp_path / "source")
    first = qualification._materialize_dataset(source, tmp_path / "first")
    second = qualification._materialize_dataset(source, tmp_path / "second")

    assert (
        first["materialized_transforms_sha256"]
        != second["materialized_transforms_sha256"]
    )
    assert qualification._materialized_dataset_identity(
        first
    ) == qualification._materialized_dataset_identity(second)
    assert first["referenced_source_materialized_content_equal"] is True
    assert (
        first["referenced_source_content"] == first["referenced_materialized_content"]
    )
    assert qualification._materialized_inputs_stable(first) is True
    assert qualification._source_inputs_stable(first) is True
    changed = Path(first["materialized_seed_ply_path"])
    changed.chmod(0o644)
    changed.write_bytes(b"changed")
    assert qualification._materialized_inputs_stable(first) is False
    source_image = source / "images/camera-a.png"
    source_image.write_bytes(b"changed-source")
    assert qualification._source_inputs_stable(first) is False


def test_materialization_fails_closed_on_copy_or_source_transform_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_dataset(tmp_path / "source")
    write_regular = qualification._write_new_regular

    def corrupt_copy(path: Path, payload: bytes) -> Path:
        if Path(path).name == "camera-a.png":
            payload = b"corrupted-copy"
        return write_regular(path, payload)

    monkeypatch.setattr(qualification, "_write_new_regular", corrupt_copy)
    with pytest.raises(ValueError, match="source and materialized referenced"):
        qualification._materialize_dataset(source, tmp_path / "corrupted")

    monkeypatch.setattr(qualification, "_write_new_regular", write_regular)

    def mutate_transforms(path: Path, payload: bytes) -> Path:
        result = write_regular(path, payload)
        if Path(path).name == "transforms.json":
            (source / "transforms.json").write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(qualification, "_write_new_regular", mutate_transforms)
    with pytest.raises(ValueError, match="source transforms changed"):
        qualification._materialize_dataset(source, tmp_path / "mutated")


def test_canonical_run_parameters_reject_every_weakening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "canonical-dataset"
    dataset.mkdir()
    monkeypatch.setattr(qualification, "DEFAULT_PUBLIC_DEV_DATASET", dataset)
    canonical = {
        "phase": "all",
        "cuda_device": 1,
        "seed": 0,
        "ab_iterations": 250,
        "ab_repeat_count": 5,
        "soak_fit_count": 243,
        "soak_iterations": 1,
        "first_fit_fd_growth_limit": 32,
        "steady_fd_growth_limit": 4,
        "steady_task_growth_limit": 4,
        "fit_timeout_seconds": 3_600,
        "analyzer_timeout_seconds": 86_400,
        "soak_timeout_seconds": 86_400,
    }
    arguments = argparse.Namespace(**canonical)

    observed = qualification._assert_canonical_run_parameters(arguments, dataset)

    assert observed == {"dataset": str(dataset.resolve()), **canonical}
    weakenings = {
        "phase": "ab",
        "cuda_device": 0,
        "seed": 1,
        "ab_iterations": 249,
        "ab_repeat_count": 4,
        "soak_fit_count": 242,
        "soak_iterations": 2,
        "first_fit_fd_growth_limit": 33,
        "steady_fd_growth_limit": 5,
        "steady_task_growth_limit": 5,
        "fit_timeout_seconds": 3_599,
        "analyzer_timeout_seconds": 86_399,
        "soak_timeout_seconds": 86_399,
    }
    for name, weak_value in weakenings.items():
        weak = argparse.Namespace(**{**canonical, name: weak_value})
        with pytest.raises(ValueError, match=f"requires {name}="):
            qualification._assert_canonical_run_parameters(weak, dataset)
    substitute = tmp_path / "substitute-dataset"
    substitute.mkdir()
    with pytest.raises(ValueError, match="exact resolved public"):
        qualification._assert_canonical_run_parameters(arguments, substitute)


def test_python_runtime_binding_hashes_resolved_binary_and_normalized_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    binary = runtime / "resolved-python"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    binary.write_bytes(b"python-executable")
    binary.chmod(0o755)
    python.symlink_to("../resolved-python")
    (runtime / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    freeze = b"z-package==2\r\na-package==1\n\n"
    normalized = b"a-package==1\nz-package==2\n"
    frozen_inventory = tmp_path / "runtime.freeze.sorted.txt"
    tree_manifest = tmp_path / "runtime.tree-manifest.json"
    frozen_inventory.write_bytes(normalized)
    tree_manifest.write_bytes(b'{"files":[]}\n')
    frozen_inventory.chmod(0o400)
    tree_manifest.chmod(0o400)

    monkeypatch.setattr(qualification, "PINNED_PYTHON", python)
    monkeypatch.setattr(
        qualification, "PINNED_PYTHON_SYMLINK_TARGET", "../resolved-python"
    )
    monkeypatch.setattr(qualification, "PINNED_PYTHON_RESOLVED", binary)
    monkeypatch.setattr(
        qualification,
        "PINNED_PYTHON_RESOLVED_SHA256",
        hashlib.sha256(b"python-executable").hexdigest(),
    )
    monkeypatch.setattr(qualification, "PINNED_PYTHON_FREEZE", frozen_inventory)
    monkeypatch.setattr(
        qualification,
        "PINNED_PYTHON_FREEZE_SHA256",
        hashlib.sha256(normalized).hexdigest(),
    )
    monkeypatch.setattr(qualification, "PINNED_PYTHON_TREE_MANIFEST", tree_manifest)
    monkeypatch.setattr(
        qualification,
        "PINNED_PYTHON_TREE_MANIFEST_SHA256",
        hashlib.sha256(b'{"files":[]}\n').hexdigest(),
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert command[0] == str(python)
        assert command[-2:] == ["freeze", "--all"]
        assert kwargs["timeout"] == 120
        return subprocess.CompletedProcess(command, 0, stdout=freeze, stderr=b"")

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)

    binding = qualification._python_runtime_binding(python)

    assert binding["lexical_path"] == str(python)
    assert binding["lexical_symlink_target"] == "../resolved-python"
    assert binding["resolved_executable"]["path"] == str(binary)
    assert (
        binding["resolved_executable"]["sha256"]
        == hashlib.sha256(b"python-executable").hexdigest()
    )
    assert binding["resolved_executable"]["size_bytes"] == len(b"python-executable")
    assert binding["pip_freeze_all"] == {
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "normalized_line_count": 2,
        "normalized_size_bytes": len(normalized),
        "equals_frozen_package_inventory": True,
    }
    assert binding["frozen_package_inventory"]["mode_octal"] == "0400"
    assert binding["frozen_runtime_tree_manifest"]["mode_octal"] == "0400"
    tree_manifest.chmod(0o600)
    tree_manifest.write_bytes(b'{"files":["tampered"]}\n')
    tree_manifest.chmod(0o400)
    with pytest.raises(ValueError, match="tree manifest binding changed"):
        qualification._python_runtime_binding(python)


def test_canonical_parent_must_itself_run_through_pinned_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    launcher = runtime / "bin/python"
    base_executable = tmp_path / "usr/bin/python3.12"
    launcher.parent.mkdir(parents=True)
    base_executable.parent.mkdir(parents=True)
    monkeypatch.setattr(qualification, "PINNED_PYTHON", launcher)
    monkeypatch.setattr(qualification, "PINNED_PYTHON_RUNTIME", runtime)
    monkeypatch.setattr(qualification, "PINNED_PYTHON_RESOLVED", base_executable)
    monkeypatch.setattr(qualification, "PINNED_PYTHON_BASE_PREFIX", tmp_path / "usr")
    monkeypatch.setattr(qualification.sys, "executable", str(launcher))
    monkeypatch.setattr(qualification.sys, "_base_executable", str(base_executable))
    monkeypatch.setattr(qualification.sys, "prefix", str(runtime))
    monkeypatch.setattr(qualification.sys, "base_prefix", str(tmp_path / "usr"))

    binding = qualification._current_python_process_binding()

    assert binding["sys_executable"] == str(launcher)
    assert binding["sys_base_executable"] == str(base_executable)
    monkeypatch.setattr(qualification.sys, "executable", str(tmp_path / "other"))
    with pytest.raises(ValueError, match="not executing through the pinned"):
        qualification._current_python_process_binding()


def test_git_binding_rejects_ordinary_and_ignored_untracked_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", "-C", repository, *arguments],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("config", "user.email", "qualification@example.invalid")
    git("config", "user.name", "Qualification Test")
    (repository / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git("add", ".gitignore", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    binding = qualification._git_binding(repository)
    assert binding["ordinary_untracked_file_count"] == 0
    assert binding["ignored_untracked_file_count"] == 0

    ordinary = repository / "ordinary.txt"
    ordinary.write_text("ordinary\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repository is dirty"):
        qualification._git_binding(repository)
    ordinary.unlink()
    (repository / "cache.ignored").write_text("ignored\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ignored untracked files"):
        qualification._git_binding(repository)


def test_git_binding_rejects_replacement_refs_before_head_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "code-repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", repository, *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "qualification@example.invalid")
    git("config", "user.name", "Qualification Test")
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    replacement = git("commit-tree", "HEAD^{tree}", "-m", "replacement")
    git("replace", "HEAD", replacement)

    real_run = subprocess.run
    calls: list[tuple[str, ...]] = []

    def record_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        assert isinstance(command, list)
        assert command[:7] == [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            repository,
        ]
        calls.append(tuple(command[7:]))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        return real_run(*args, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "run", record_run)

    with pytest.raises(ValueError, match="replacement refs"):
        qualification._git_binding(repository)

    assert calls == [
        ("for-each-ref", "--format=%(refname)", "refs/replace"),
    ]
    assert ("rev-parse", "HEAD") not in calls
    assert ("rev-parse", "HEAD^{tree}") not in calls


def test_pinned_git_binding_rejects_grafts_before_head_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "pinned-deform360-repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", repository, *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "qualification@example.invalid")
    git("config", "user.name", "Qualification Test")
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    expected_head = git("rev-parse", "HEAD")
    (repository / ".git/info/grafts").write_text(
        f"{expected_head} {'0' * 40}\n",
        encoding="ascii",
    )

    real_run = subprocess.run
    calls: list[tuple[str, ...]] = []

    def record_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        assert isinstance(command, list)
        assert command[:7] == [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            repository,
        ]
        calls.append(tuple(command[7:]))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        return real_run(*args, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "run", record_run)

    with pytest.raises(ValueError, match="grafts file"):
        qualification._git_binding(repository, expected_head=expected_head)

    assert calls == [
        ("for-each-ref", "--format=%(refname)", "refs/replace"),
        ("rev-parse", "--absolute-git-dir"),
    ]
    assert ("rev-parse", "HEAD") not in calls
    assert ("rev-parse", "HEAD^{tree}") not in calls


@pytest.mark.parametrize(
    ("index_flag", "expected_tag"),
    (
        ("--assume-unchanged", "h"),
        ("--skip-worktree", "S"),
    ),
)
def test_git_binding_rejects_hidden_tracked_drift_before_identity_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
    expected_tag: str,
) -> None:
    repository = tmp_path / "hidden-drift-repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", repository, *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "qualification@example.invalid")
    git("config", "user.name", "Qualification Test")
    tracked = repository / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    git("update-index", index_flag, "tracked.txt")
    tracked.write_text("hidden drift\n", encoding="utf-8")
    assert git("status", "--porcelain=v1", "--untracked-files=all") == ""
    assert git("ls-files", "-v", "tracked.txt").startswith(f"{expected_tag} ")

    real_run = subprocess.run
    calls: list[tuple[str, ...]] = []

    def record_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        assert isinstance(command, list)
        assert command[:7] == [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            repository,
        ]
        calls.append(tuple(command[7:]))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        return real_run(*args, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "run", record_run)

    with pytest.raises(ValueError, match="non-ordinary tracked index entries"):
        qualification._git_binding(repository)

    assert calls[-1] == ("ls-files", "-v", "-z")
    assert ("rev-parse", "HEAD") not in calls
    assert ("rev-parse", "HEAD^{tree}") not in calls


def test_structured_comparison_requires_exact_finite_fields() -> None:
    dtype = np.dtype([("x", "<f4"), ("opacity", "<f4")])
    left = np.array([(1.0, 0.5), (2.0, -0.25)], dtype=dtype)
    exact = left.copy()

    result = qualification._compare_structured_arrays(left, exact)

    assert result["passed"] is True
    assert result["all_fields_exact"] is True
    assert result["fields"]["x"]["finite_max_abs_difference"] == 0.0
    assert result["fields"]["opacity"]["max_abs_difference_is_finite"] is True

    changed = exact.copy()
    changed[1]["x"] += np.float32(0.25)
    result = qualification._compare_structured_arrays(left, changed)
    assert result["passed"] is False
    assert result["fields"]["x"]["exact"] is False
    assert result["fields"]["x"]["finite_max_abs_difference"] == 0.25

    nonfinite = exact.copy()
    nonfinite[0]["opacity"] = np.nan
    result = qualification._compare_structured_arrays(left, nonfinite)
    assert result["passed"] is False
    assert result["fields"]["opacity"]["right_finite"] is False
    assert result["fields"]["opacity"]["finite_max_abs_difference"] is None


def test_structured_comparison_rejects_field_order_and_dtype_changes() -> None:
    left = np.array([(1.0, 2.0)], dtype=[("x", "<f4"), ("y", "<f4")])
    reordered = np.array([(2.0, 1.0)], dtype=[("y", "<f4"), ("x", "<f4")])
    wider = np.array([(1.0, 2.0)], dtype=[("x", "<f8"), ("y", "<f4")])

    assert qualification._compare_structured_arrays(left, reordered)["passed"] is False
    result = qualification._compare_structured_arrays(left, wider)
    assert result["passed"] is False
    assert result["fields"]["x"]["dtype_equal"] is False


def test_global_snapshot_detects_object_and_content_drift() -> None:
    prior_writer = object()
    prior_storage = object()
    prior_buffer = object()
    prior_profiler = object()
    pytorch_profiler = object()
    writer = SimpleNamespace(
        EVENT_WRITERS=[prior_writer],
        EVENT_STORAGE=[prior_storage],
        GLOBAL_BUFFER={"prior": prior_buffer},
    )
    profiler = SimpleNamespace(
        PROFILER=[prior_profiler], PYTORCH_PROFILER=pytorch_profiler
    )
    initial = qualification._global_state_snapshot(writer, profiler)

    assert qualification._global_state_snapshot(writer, profiler) == initial
    writer.EVENT_WRITERS.append(object())
    assert qualification._global_state_snapshot(writer, profiler) != initial
    writer.EVENT_WRITERS.pop()
    writer.GLOBAL_BUFFER = {"prior": prior_buffer}
    assert qualification._global_state_snapshot(writer, profiler) != initial


def _fit_boundary(
    index: int,
    *,
    fd: int,
    tasks: int,
    rss: int = 1_000,
    restored: bool = True,
) -> dict[str, object]:
    return {
        "fit_index": index,
        "trainer_reinitialized": (
            index % qualification.SOAK_TRAINER_REINITIALIZATION_INTERVAL == 0
        ),
        "output_created": True,
        "dataset_outputs_created": True,
        "cleanup_completed": True,
        "output_ply_absent_after_cleanup": True,
        "dataset_outputs_absent_after_cleanup": True,
        "resource_boundary_stage": "after_cleanup",
        "globals_restored": restored,
        "resource_boundary": {
            "file_descriptor_count": fd,
            "task_count": tasks,
            "rss_kib": rss,
            "rlimit_nofile_soft": 1024,
            "rlimit_nofile_hard": 1_048_576,
        },
    }


def test_soak_boundary_gate_uses_first_fit_and_steady_limits() -> None:
    before = {
        "file_descriptor_count": 10,
        "task_count": 2,
        "rss_kib": 900,
        "rlimit_nofile_soft": 1024,
        "rlimit_nofile_hard": 1_048_576,
    }
    fits = [
        _fit_boundary(0, fd=14, tasks=3),
        _fit_boundary(1, fd=16, tasks=4, rss=1_100),
        _fit_boundary(2, fd=18, tasks=7, rss=1_200),
    ]

    result = qualification._evaluate_soak_boundaries(
        before,
        fits,
        expected_fit_count=3,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )

    assert result["passed"] is True
    assert result["predicates"]["cleanup_completed_after_every_fit"] is True
    assert result["observed"]["maximum_fd_growth_from_first_post_fit"] == 4
    assert result["observed"]["maximum_task_growth_from_first_post_fit"] == 4

    fits[-1] = _fit_boundary(2, fd=19, tasks=7)
    assert (
        qualification._evaluate_soak_boundaries(
            before,
            fits,
            expected_fit_count=3,
            first_fd_growth_limit=32,
            steady_fd_growth_limit=4,
            steady_task_growth_limit=4,
        )["passed"]
        is False
    )
    fits[-1] = _fit_boundary(2, fd=18, tasks=7)
    fits[-1]["cleanup_completed"] = False
    result = qualification._evaluate_soak_boundaries(
        before,
        fits,
        expected_fit_count=3,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )
    assert result["predicates"]["cleanup_completed_after_every_fit"] is False
    fits[-1] = _fit_boundary(2, fd=18, tasks=7)
    fits[-1]["resource_boundary"]["rlimit_nofile_soft"] = 2048
    result = qualification._evaluate_soak_boundaries(
        before,
        fits,
        expected_fit_count=3,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )
    assert result["predicates"]["rlimit_nofile_unchanged"] is False
    fits[-1] = _fit_boundary(2, fd=18, tasks=7, restored=False)
    assert (
        qualification._evaluate_soak_boundaries(
            before,
            fits,
            expected_fit_count=3,
            first_fd_growth_limit=32,
            steady_fd_growth_limit=4,
            steady_task_growth_limit=4,
        )["predicates"]["globals_restored_after_every_fit"]
        is False
    )


def test_soak_gate_requires_formal_case_reinitialization_indices() -> None:
    before = {
        "file_descriptor_count": 10,
        "task_count": 2,
        "rss_kib": 900,
        "rlimit_nofile_soft": 1024,
        "rlimit_nofile_hard": 1_048_576,
    }
    fits = [_fit_boundary(index, fd=12, tasks=3) for index in range(243)]

    result = qualification._evaluate_soak_boundaries(
        before,
        fits,
        expected_fit_count=243,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )

    assert result["passed"] is True
    assert result["trainer_reinitialization"] == {
        "interval": 81,
        "expected_indices": [0, 81, 162],
        "observed_indices": [0, 81, 162],
    }
    fits[81]["trainer_reinitialized"] = False
    result = qualification._evaluate_soak_boundaries(
        before,
        fits,
        expected_fit_count=243,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )
    assert result["predicates"]["trainer_reinitialization_indices_exact"] is False


def test_signed_json_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    evidence = qualification._signed(
        {
            "artifact_kind": "SyntheticQualificationEvidence",
            "passed": True,
            "values": [1, 2, 3],
        }
    )
    path = qualification._write_new_json(tmp_path / "evidence.json", evidence)

    assert qualification._load_signed_json(path, label="evidence") == evidence
    changed = json.loads(path.read_text())
    changed["passed"] = False
    path.chmod(0o644)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="signature is invalid"):
        qualification._load_signed_json(path, label="evidence")


def test_missing_child_evidence_and_output_become_auditable_failures(
    tmp_path: Path,
) -> None:
    child, validation = qualification._load_optional_child_evidence(
        tmp_path / "missing.json", label="missing child"
    )

    assert child is None
    assert validation["loaded_and_signature_valid"] is False
    comparison = qualification._compare_optional_fit_outputs(None, None)
    assert comparison["passed"] is False
    assert comparison["error"]["type"] == "ValueError"


def test_ab_comparison_binds_child_outputs_before_and_after_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_path = tmp_path / "original.ply"
    wrapped_path = tmp_path / "wrapped.ply"
    original_path.write_bytes(b"original")
    wrapped_path.write_bytes(b"wrapped")
    original = {"output": qualification._bound_file(original_path)}
    wrapped = {"output": qualification._bound_file(wrapped_path)}
    monkeypatch.setattr(
        qualification,
        "_compare_ply",
        lambda *_args: {"passed": True, "synthetic_comparison": True},
    )

    comparison = qualification._compare_optional_fit_outputs(original, wrapped)

    assert comparison["passed"] is True
    assert all(comparison["output_integrity_predicates"].values())
    original_path.write_bytes(b"changed-before-parse")
    comparison = qualification._compare_optional_fit_outputs(original, wrapped)
    assert comparison["passed"] is False
    assert (
        comparison["output_integrity_predicates"][
            "original_binding_matches_child_evidence_before_parse"
        ]
        is False
    )

    original_path.write_bytes(b"original")

    def mutate_during_parse(*_args: object) -> dict[str, bool]:
        wrapped_path.write_bytes(b"changed-during-parse")
        return {"passed": True}

    monkeypatch.setattr(qualification, "_compare_ply", mutate_during_parse)
    comparison = qualification._compare_optional_fit_outputs(original, wrapped)
    assert comparison["passed"] is False
    assert (
        comparison["output_integrity_predicates"]["wrapped_binding_stable_across_parse"]
        is False
    )


def test_child_timeout_is_recorded_without_losing_log(tmp_path: Path) -> None:
    invocation = qualification._invoke_child(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        environment={"PATH": "/usr/bin:/bin"},
        log_path=tmp_path / "timed-out.log",
        timeout_seconds=0.01,
    )

    assert invocation["return_code"] is None
    assert invocation["timed_out"] is True
    assert invocation["timeout_error"]["type"] == "TimeoutExpired"
    assert invocation["timeout_seconds"] == 0.01
    assert invocation["environment"] == {"PATH": "/usr/bin:/bin"}
    assert invocation["log"]["size_bytes"] == 0


def test_child_environment_is_normalized_and_does_not_inherit_pythonpath(
    tmp_path: Path,
) -> None:
    environment = qualification._child_environment(1, tmp_path)

    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["CUDA_MODULE_LOADING"] == "LAZY"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == ("/nonexistent/bpt-held-v8-pycache")
    assert "PYTHONPATH" not in environment
    assert environment["TMPDIR"] == str(tmp_path)
    assert qualification._child_python_argv_prefix(
        Path("/runtime/bin/python"), Path("/code/qualify.py")
    ) == [
        "/runtime/bin/python",
        "-I",
        "-B",
        "-X",
        "pycache_prefix=/nonexistent/bpt-held-v8-pycache",
        "/code/qualify.py",
    ]


def test_owned_cleanup_records_bounded_no_follow_inventory(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    tree = parent / "outputs"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested/artifact.bin").write_bytes(b"artifact")

    cleanup = qualification._remove_owned_tree(
        tree, parent=parent, label="generated outputs"
    )

    assert cleanup["bounded_parent"] == str(parent)
    assert cleanup["pre_cleanup_inventory"]["entry_count"] == 2
    assert cleanup["pre_cleanup_inventory"]["regular_file_bytes"] == len(b"artifact")
    assert cleanup["post_cleanup_absent"] is True
    assert not tree.exists()

    generated = parent / "splat.ply"
    generated.write_bytes(b"ply")
    file_cleanup = qualification._remove_owned_file(
        generated, parent=parent, label="generated Ply"
    )
    assert file_cleanup["pre_cleanup_link_count"] == 1
    assert (
        file_cleanup["pre_cleanup_binding"]["sha256"]
        == hashlib.sha256(b"ply").hexdigest()
    )
    assert not generated.exists()


def test_owned_cleanup_rejects_escape_symlink_and_hardlink(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.bin").write_bytes(b"outside")
    with pytest.raises(ValueError, match="escaped its bounded parent"):
        qualification._remove_owned_tree(
            outside, parent=parent, label="escaped outputs"
        )

    linked_tree = parent / "linked-tree"
    try:
        linked_tree.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not support symlinks")
    with pytest.raises(ValueError, match="not a real directory"):
        qualification._remove_owned_tree(
            linked_tree, parent=parent, label="linked outputs"
        )
    linked_tree.unlink()

    target = parent / "target.bin"
    alias = parent / "alias.bin"
    target.write_bytes(b"shared")
    try:
        alias.hardlink_to(target)
    except OSError:
        pytest.skip("filesystem does not support hardlinks")
    with pytest.raises(ValueError, match="linked or not a regular file"):
        qualification._remove_owned_file(
            alias, parent=parent, label="hardlinked output"
        )
    assert target.read_bytes() == b"shared"
    assert alias.read_bytes() == b"shared"


def test_owned_file_cleanup_resists_ancestor_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "anchor"
    parent = anchor / "parent"
    parent.mkdir(parents=True)
    generated = parent / "generated.ply"
    generated.write_bytes(b"owned")
    moved_anchor = tmp_path / "moved-anchor"
    replacement = anchor / "parent/generated.ply"
    real_unlink = qualification.os.unlink
    substituted = False

    def substitute_ancestor_before_unlink(
        path: str | bytes,
        *args: object,
        dir_fd: int | None = None,
        **kwargs: object,
    ) -> None:
        nonlocal substituted
        if not substituted and path == "generated.ply" and dir_fd is not None:
            substituted = True
            anchor.rename(moved_anchor)
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"replacement")
        real_unlink(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(qualification.os, "unlink", substitute_ancestor_before_unlink)

    with pytest.raises(ValueError, match="bounded parent changed during cleanup"):
        qualification._remove_owned_file(
            generated,
            parent=parent,
            label="generated Ply",
        )

    assert substituted is True
    assert replacement.read_bytes() == b"replacement"
    assert not (moved_anchor / "parent/generated.ply").exists()


def test_owned_tree_cleanup_resists_ancestor_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "anchor"
    parent = anchor / "parent"
    original = parent / "outputs/nested/artifact.bin"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"owned")
    moved_anchor = tmp_path / "moved-anchor"
    replacement = anchor / "parent/outputs/nested/artifact.bin"
    real_unlink = qualification.os.unlink
    substituted = False

    def substitute_ancestor_before_unlink(
        path: str | bytes,
        *args: object,
        dir_fd: int | None = None,
        **kwargs: object,
    ) -> None:
        nonlocal substituted
        if not substituted and path == "artifact.bin" and dir_fd is not None:
            substituted = True
            anchor.rename(moved_anchor)
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"replacement")
        real_unlink(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(qualification.os, "unlink", substitute_ancestor_before_unlink)

    with pytest.raises(ValueError, match="bounded parent changed during cleanup"):
        qualification._remove_owned_tree(
            parent / "outputs",
            parent=parent,
            label="generated outputs",
        )

    assert substituted is True
    assert replacement.read_bytes() == b"replacement"
    assert not (moved_anchor / "parent/outputs").exists()


def test_gsplat_smoke_loads_exact_adapter_and_validates_signed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "code"
    adapter = code / "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# exact adapter\n", encoding="utf-8")
    unsigned = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
        "extension_loaded_and_retained": True,
        "target_or_outcome_path_accessed": False,
        "physical_gpu_index": 1,
    }
    smoke = {
        **unsigned,
        "artifact_sha256": qualification._gsplat_smoke_artifact_sha256(unsigned),
    }
    calls = 0

    def load_and_smoke() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return smoke

    runtime = SimpleNamespace(
        __file__=str(adapter), load_and_smoke_gsplat_runtime=load_and_smoke
    )
    monkeypatch.setattr(qualification.importlib, "import_module", lambda _name: runtime)

    binding = qualification._load_and_smoke_gsplat_runtime(code)

    assert calls == 1
    assert binding["adapter_source"]["path"] == str(adapter)
    assert binding["evidence"] == smoke
    assert binding["evidence_artifact_sha256"] == smoke["artifact_sha256"]
    smoke["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature is invalid"):
        qualification._load_and_smoke_gsplat_runtime(code)


class _FakeDelegate:
    def __init__(self, writer: SimpleNamespace, profiler: SimpleNamespace) -> None:
        self.writer = writer
        self.profiler = profiler

    def train(
        self,
        dataset: Path,
        output: Path,
        filename: str,
        _iterations: int,
    ) -> Path:
        self.writer.EVENT_WRITERS.append(object())
        self.writer.EVENT_STORAGE.append(object())
        self.writer.GLOBAL_BUFFER["transient"] = object()
        self.profiler.PROFILER.append(object())
        (dataset / "outputs").mkdir(exist_ok=True)
        produced = output / filename
        produced.write_bytes(b"synthetic-ply")
        return produced


class _FakeWrapper:
    def __init__(self, delegate: _FakeDelegate) -> None:
        self.delegate = delegate

    def train(
        self,
        dataset: Path,
        output: Path,
        filename: str,
        iterations: int,
    ) -> Path:
        writer = self.delegate.writer
        profiler = self.delegate.profiler
        prior_writers = list(writer.EVENT_WRITERS)
        prior_storage = list(writer.EVENT_STORAGE)
        prior_buffer = dict(writer.GLOBAL_BUFFER)
        prior_profilers = list(profiler.PROFILER)
        try:
            return self.delegate.train(dataset, output, filename, iterations)
        finally:
            writer.EVENT_WRITERS.clear()
            writer.EVENT_WRITERS.extend(prior_writers)
            writer.EVENT_STORAGE.clear()
            writer.EVENT_STORAGE.extend(prior_storage)
            writer.GLOBAL_BUFFER.clear()
            writer.GLOBAL_BUFFER.update(prior_buffer)
            profiler.PROFILER.clear()
            profiler.PROFILER.extend(prior_profilers)


def _fake_runtime_modules() -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(EVENT_WRITERS=[], EVENT_STORAGE=[], GLOBAL_BUFFER={}),
        SimpleNamespace(PROFILER=[], PYTORCH_PROFILER=None),
    )


def _fake_gsplat_smoke_binding() -> dict[str, object]:
    return {
        "adapter_source": {"path": "/code/adapter.py", "sha256": "a" * 64},
        "evidence": {
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
            "artifact_sha256": "b" * 64,
            "extension_loaded_and_retained": True,
        },
        "evidence_artifact_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("variant", "restored"), (("original", False), ("wrapped", True))
)
def test_fit_child_treats_original_leak_as_observation_and_wrapped_as_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    restored: bool,
) -> None:
    code = tmp_path / "code"
    deform360 = tmp_path / "deform360"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    for path in (code, deform360, dataset, output):
        path.mkdir()
    writer, profiler = _fake_runtime_modules()
    smoke_calls: list[Path] = []
    events: list[str] = []

    def smoke_runtime(root: Path) -> dict[str, object]:
        events.append("smoke")
        smoke_calls.append(root)
        return _fake_gsplat_smoke_binding()

    def import_trainers(*_args: object) -> tuple[object, ...]:
        events.append("trainers")
        return (
            lambda: _FakeDelegate(writer, profiler),
            _FakeWrapper,
            writer,
            profiler,
        )

    monkeypatch.setattr(
        qualification,
        "_load_and_smoke_gsplat_runtime",
        smoke_runtime,
    )
    monkeypatch.setattr(qualification, "_import_trainers", import_trainers)
    monkeypatch.setattr(qualification, "_seed_runtime", lambda _seed: {"seed": 0})
    monkeypatch.setattr(
        qualification,
        "_process_boundary",
        lambda: {
            "file_descriptor_count": 10,
            "task_count": 2,
            "rss_kib": 100,
            "rlimit_nofile_soft": 1024,
            "rlimit_nofile_hard": 1_048_576,
        },
    )
    result_path = tmp_path / f"{variant}.json"
    arguments = argparse.Namespace(
        code_root=code,
        deform360_repo=deform360,
        dataset=dataset,
        output_dir=output,
        result=result_path,
        iterations=1,
        seed=0,
        variant=variant,
    )

    assert qualification._child_fit(arguments) == 0
    result = qualification._load_signed_json(result_path, label="fit result")
    assert result["passed"] is True
    assert smoke_calls == [code]
    assert events == ["smoke", "trainers"]
    assert result["gsplat_runtime_smoke"] == _fake_gsplat_smoke_binding()
    assert result["predicates"]["gsplat_runtime_smoke_validated_and_retained"] is True
    assert result["global_state"]["restored"] is restored
    assert result["predicates"]["wrapped_fit_requires_global_restoration"] is True


def test_soak_child_runs_repeated_fits_in_one_wrapped_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = tmp_path / "code"
    deform360 = tmp_path / "deform360"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    for path in (code, deform360, dataset, output):
        path.mkdir()
    writer, profiler = _fake_runtime_modules()
    smoke_calls: list[Path] = []
    events: list[str] = []

    def smoke_runtime(root: Path) -> dict[str, object]:
        events.append("smoke")
        smoke_calls.append(root)
        return _fake_gsplat_smoke_binding()

    def import_trainers(*_args: object) -> tuple[object, ...]:
        events.append("trainers")
        return (
            lambda: _FakeDelegate(writer, profiler),
            _FakeWrapper,
            writer,
            profiler,
        )

    monkeypatch.setattr(
        qualification,
        "_load_and_smoke_gsplat_runtime",
        smoke_runtime,
    )
    monkeypatch.setattr(qualification, "_import_trainers", import_trainers)
    monkeypatch.setattr(qualification, "_seed_runtime", lambda _seed: {"seed": 0})
    boundaries = iter(
        [
            {
                "file_descriptor_count": 10,
                "task_count": 2,
                "rss_kib": 100,
                "rlimit_nofile_soft": 1024,
                "rlimit_nofile_hard": 1_048_576,
            },
            {
                "file_descriptor_count": 12,
                "task_count": 3,
                "rss_kib": 110,
                "rlimit_nofile_soft": 1024,
                "rlimit_nofile_hard": 1_048_576,
            },
            {
                "file_descriptor_count": 13,
                "task_count": 3,
                "rss_kib": 120,
                "rlimit_nofile_soft": 1024,
                "rlimit_nofile_hard": 1_048_576,
            },
            {
                "file_descriptor_count": 14,
                "task_count": 3,
                "rss_kib": 130,
                "rlimit_nofile_soft": 1024,
                "rlimit_nofile_hard": 1_048_576,
            },
        ]
    )
    monkeypatch.setattr(qualification, "_process_boundary", lambda: next(boundaries))
    result_path = tmp_path / "soak.json"
    arguments = argparse.Namespace(
        code_root=code,
        deform360_repo=deform360,
        dataset=dataset,
        output_dir=output,
        result=result_path,
        iterations=1,
        seed=0,
        fit_count=3,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )

    assert qualification._child_soak(arguments) == 0
    result = qualification._load_signed_json(result_path, label="soak result")
    assert result["passed"] is True
    assert smoke_calls == [code]
    assert events == ["smoke", "trainers"]
    assert result["gsplat_runtime_smoke"] == _fake_gsplat_smoke_binding()
    assert len(result["fits"]) == 3
    assert all(value["globals_restored"] for value in result["fits"])
    assert all(value["cleanup_completed"] for value in result["fits"])
    assert all(value["dataset_outputs_created"] for value in result["fits"])
    assert result["evaluation"]["trainer_reinitialization"]["observed_indices"] == [0]
    assert not (dataset / "outputs").exists()
    assert not tuple(output.iterdir())


def test_soak_cleanup_failure_is_signed_and_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "code"
    deform360 = tmp_path / "deform360"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    for path in (code, deform360, dataset, output):
        path.mkdir()
    writer, profiler = _fake_runtime_modules()
    smoke_calls: list[Path] = []
    monkeypatch.setattr(
        qualification,
        "_load_and_smoke_gsplat_runtime",
        lambda root: smoke_calls.append(root) or _fake_gsplat_smoke_binding(),
    )
    monkeypatch.setattr(
        qualification,
        "_import_trainers",
        lambda *_args: (
            lambda: _FakeDelegate(writer, profiler),
            _FakeWrapper,
            writer,
            profiler,
        ),
    )
    monkeypatch.setattr(qualification, "_seed_runtime", lambda _seed: {"seed": 0})
    boundary_calls = 0

    def process_boundary() -> dict[str, int]:
        nonlocal boundary_calls
        boundary_calls += 1
        return {
            "file_descriptor_count": 10,
            "task_count": 2,
            "rss_kib": 100,
            "rlimit_nofile_soft": 1024,
            "rlimit_nofile_hard": 1_048_576,
        }

    monkeypatch.setattr(qualification, "_process_boundary", process_boundary)

    def fail_cleanup(_path: Path, *, parent: Path, label: str) -> None:
        assert parent == dataset
        assert "generated outputs" in label
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(qualification, "_remove_owned_tree", fail_cleanup)
    result_path = tmp_path / "soak-cleanup-failure.json"
    arguments = argparse.Namespace(
        code_root=code,
        deform360_repo=deform360,
        dataset=dataset,
        output_dir=output,
        result=result_path,
        iterations=1,
        seed=0,
        fit_count=1,
        first_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
    )

    assert qualification._child_soak(arguments) == 2
    result = qualification._load_signed_json(result_path, label="failed soak")
    assert result["passed"] is False
    assert smoke_calls == [code]
    assert result["error"]["type"] == "OSError"
    assert "injected cleanup failure" in result["error"]["message"]
    assert boundary_calls == 1
    assert (dataset / "outputs").is_dir()
    assert not tuple(output.iterdir())


@pytest.mark.parametrize(
    "analyzer_case",
    (
        "accepted",
        "accepted-postcondition-failure",
        "scientific-no-go",
        "technical-exit",
        "timeout",
        "missing-result",
    ),
)
def test_run_analyzer_failures_consume_root_and_only_no_go_is_signed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, analyzer_case: str
) -> None:
    code = SCRIPT.parents[2].resolve()
    deform360 = tmp_path / "deform360"
    dataset = tmp_path / "dataset"
    deform360.mkdir()
    dataset.mkdir()
    (dataset / "transforms.json").write_text("{}\n", encoding="utf-8")
    python = tmp_path / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic-python")
    base = tmp_path / "qualification-base"
    base.mkdir()
    head = "b" * 40
    output = base / f"bpt-resource-lifecycle-qualification-{head}"
    monkeypatch.setattr(qualification, "PINNED_PYTHON", python)
    monkeypatch.setattr(qualification, "PINNED_DEFORM360", deform360)
    monkeypatch.setattr(qualification, "DEFAULT_PUBLIC_DEV_DATASET", dataset)
    monkeypatch.setattr(qualification, "QUALIFICATION_BASE", base)
    monkeypatch.setattr(qualification.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(
        qualification,
        "_current_python_process_binding",
        lambda: {"pinned": True},
    )
    monkeypatch.setattr(
        qualification,
        "_python_runtime_binding",
        lambda _python: {"runtime": "stable"},
    )

    def git_binding(
        root: Path, *, expected_head: str | None = None
    ) -> dict[str, object]:
        observed_head = expected_head if expected_head is not None else head
        return {
            "path": str(root),
            "head": observed_head,
            "tree": "c" * 40,
            "clean": True,
            "ordinary_untracked_file_count": 0,
            "ignored_untracked_file_count": 0,
        }

    monkeypatch.setattr(qualification, "_git_binding", git_binding)

    def materialize(_source: Path, destination: Path) -> dict[str, object]:
        destination.mkdir()
        (destination / "transforms.json").write_text("{}\n", encoding="utf-8")
        return {
            "destination": str(destination),
            "synthetic_content_identity": "same-public-input",
            "referenced_source_materialized_content_equal": True,
            "referenced_source_content": {"content": "same-public-input"},
            "referenced_materialized_content": {"content": "same-public-input"},
        }

    monkeypatch.setattr(qualification, "_materialize_dataset", materialize)
    monkeypatch.setattr(
        qualification,
        "_materialized_dataset_identity",
        lambda _audit: {"content": "same-public-input"},
    )
    monkeypatch.setattr(
        qualification,
        "_materialized_inputs_stable",
        lambda _audit: analyzer_case != "accepted-postcondition-failure",
    )
    monkeypatch.setattr(qualification, "_source_inputs_stable", lambda _audit: True)

    calls: list[str] = []

    def invoke(
        command: list[str],
        *,
        environment: dict[str, str],
        log_path: Path,
        timeout_seconds: int,
    ) -> dict[str, object]:
        log_path.write_bytes(b"")
        timed_out = False
        timeout_error = None
        if "_fit-child" in command:
            calls.append("fit")
            variant = command[command.index("--variant") + 1]
            materialized = Path(command[command.index("--dataset") + 1])
            export = Path(command[command.index("--output-dir") + 1])
            result = Path(command[command.index("--result") + 1])
            (materialized / "outputs").mkdir()
            (materialized / "outputs/checkpoint.bin").write_bytes(b"checkpoint")
            ply = export / "splat.ply"
            ply.write_bytes(f"{variant}-ply".encode())
            child = qualification._signed(
                {
                    "schema_version": 1,
                    "artifact_kind": "Deform360ResourceLifecycleFitChildEvidence",
                    "qualification_id": qualification.QUALIFICATION_ID,
                    "variant": variant,
                    "passed": True,
                    "parameters": {"iterations": 250, "seed": 0},
                    "runtime": {"synthetic": True},
                    "gsplat_runtime_smoke": {
                        "evidence": {
                            "physical_gpu_index": 1,
                            "logical_device": "cuda:0",
                            "target_or_outcome_path_accessed": False,
                        }
                    },
                    "dataset": str(materialized),
                    "output": qualification._bound_file(ply),
                    "resource_boundary": {"synthetic": True},
                    "global_state": {"synthetic": True},
                    "predicates": {"synthetic_pass": True},
                    "formal_held_path_supplied": False,
                }
            )
            qualification._write_new_json(result, child)
            return_code = 0
        elif "prepare-manifest" in command:
            calls.append("prepare-manifest")
            result = Path(command[command.index("--output") + 1])
            manifest = qualification._signed(
                {
                    "schema_version": 1,
                    "artifact_kind": qualification.ANALYSIS_MANIFEST_KIND,
                    "analysis_id": qualification.ANALYSIS_ID,
                    "expected_environment": {
                        "generator_profile": qualification.GENERATOR_PROFILE,
                        "physical_gpu_index": 1,
                    },
                }
            )
            qualification._write_new_json(result, manifest)
            return_code = 0
        elif "analyze" in command:
            calls.append("analyze")
            result = Path(command[command.index("--output") + 1])
            manifest_path = Path(command[command.index("--manifest") + 1])
            analyzer_accepts = analyzer_case in {
                "accepted",
                "accepted-postcondition-failure",
            }
            if analyzer_case in {
                "accepted",
                "accepted-postcondition-failure",
                "scientific-no-go",
                "technical-exit",
            }:
                manifest = qualification._load_signed_json(
                    manifest_path, label="synthetic manifest"
                )
                analysis = qualification._signed(
                    {
                        "schema_version": 1,
                        "artifact_kind": qualification.ANALYSIS_RESULT_KIND,
                        "analysis_id": qualification.ANALYSIS_ID,
                        "development_only": True,
                        "formal_path_accessed": False,
                        "generator_profile": qualification.GENERATOR_PROFILE,
                        "physical_gpu_index": 1,
                        "input_manifest": {
                            **qualification._bound_file(manifest_path),
                            "artifact_sha256": manifest["artifact_sha256"],
                        },
                        "decision": {
                            "exact_matched_structured_array_equality_primary_passed": (
                                False
                            ),
                            "exact_matched_file_bytes_equal": False,
                            "secondary_distributional_equivalence_passed": (
                                analyzer_accepts
                            ),
                            "accepted": analyzer_accepts,
                            "acceptance_basis": (
                                "secondary-distributional-envelope"
                                if analyzer_accepts
                                else "rejected"
                            ),
                        },
                    }
                )
                qualification._write_new_json(result, analysis)
            if analyzer_accepts:
                return_code = 0
            elif analyzer_case == "scientific-no-go":
                return_code = 3
            elif analyzer_case == "technical-exit":
                return_code = 2
            elif analyzer_case == "timeout":
                return_code = None
                timed_out = True
                timeout_error = {
                    "type": "TimeoutExpired",
                    "message": "synthetic analyzer timeout",
                    "timeout_seconds": timeout_seconds,
                }
            else:
                return_code = 0
        elif "_soak-child" in command:
            calls.append("soak")
            materialized = Path(command[command.index("--dataset") + 1])
            export = Path(command[command.index("--output-dir") + 1])
            result = Path(command[command.index("--result") + 1])
            fits = []
            for index in range(243):
                fits.append(
                    {
                        "fit_index": index,
                        "trainer_reinitialized": index % 81 == 0,
                        "output_created": True,
                        "dataset_outputs_created": True,
                        "output_size_bytes": 1,
                        "cleanup_completed": True,
                        "cleanup": {
                            "output_ply": {
                                "bounded_parent": str(export),
                                "pre_cleanup_binding": {
                                    "path": str(export / f"splat-{index:04d}.ply"),
                                    "size_bytes": 1,
                                    "sha256": "0" * 64,
                                    "mode_octal": "0644",
                                },
                                "pre_cleanup_link_count": 1,
                                "removed": True,
                                "post_cleanup_absent": True,
                            },
                            "dataset_outputs": {
                                "bounded_parent": str(materialized),
                                "pre_cleanup_inventory": {
                                    "root": str(materialized / "outputs"),
                                    "entry_count": 1,
                                    "regular_file_bytes": 1,
                                    "inventory_sha256": "1" * 64,
                                },
                                "removed": True,
                                "post_cleanup_absent": True,
                            },
                        },
                        "output_ply_absent_after_cleanup": True,
                        "dataset_outputs_absent_after_cleanup": True,
                        "resource_boundary_stage": "after_cleanup",
                        "resource_boundary": {"synthetic": True},
                        "globals_restored": True,
                        "global_state": {"synthetic": True},
                    }
                )
            soak = qualification._signed(
                {
                    "schema_version": 1,
                    "artifact_kind": ("Deform360ResourceLifecycleSoakChildEvidence"),
                    "qualification_id": qualification.QUALIFICATION_ID,
                    "passed": True,
                    "parameters": {
                        "fit_count": 243,
                        "iterations_per_fit": 1,
                        "seed": 0,
                        "trainer_reinitialization_interval": 81,
                    },
                    "runtime": {"seed": 0, "cuda_device_count": 1},
                    "gsplat_runtime_smoke": {
                        "evidence": {
                            "physical_gpu_index": 1,
                            "logical_device": "cuda:0",
                            "target_or_outcome_path_accessed": False,
                        }
                    },
                    "dataset": str(materialized),
                    "initial_global_state": {"synthetic": True},
                    "fits": fits,
                    "evaluation": {
                        "passed": True,
                        "predicates": {"all_soak_gates": True},
                    },
                    "formal_held_path_supplied": False,
                }
            )
            qualification._write_new_json(result, soak)
            return_code = 0
        else:  # pragma: no cover - documents the forbidden no-go branch
            raise AssertionError(f"unexpected child invocation: {command}")
        return {
            "command": command,
            "environment": environment,
            "return_code": return_code,
            "timed_out": timed_out,
            "timeout_error": timeout_error,
            "timeout_seconds": timeout_seconds,
            "log": qualification._bound_file(log_path),
        }

    monkeypatch.setattr(qualification, "_invoke_child", invoke)
    arguments = argparse.Namespace(
        code_root=code,
        python=python,
        deform360_repo=deform360,
        dataset=dataset,
        output_dir=output,
        phase="all",
        cuda_device=1,
        seed=0,
        ab_iterations=250,
        ab_repeat_count=5,
        soak_fit_count=243,
        soak_iterations=1,
        first_fit_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
        fit_timeout_seconds=3_600,
        analyzer_timeout_seconds=86_400,
        soak_timeout_seconds=86_400,
    )

    if analyzer_case == "accepted":
        assert qualification._run(arguments) == 0
    elif analyzer_case == "accepted-postcondition-failure":
        with pytest.raises(
            ValueError,
            match="qualification failed technically after canonical root creation",
        ):
            qualification._run(arguments)
    elif analyzer_case == "scientific-no-go":
        assert qualification._run(arguments) == 3
    else:
        with pytest.raises(ValueError, match="analyzer failed technically"):
            qualification._run(arguments)
    attempt = qualification._load_signed_json(
        output / "qualification-attempt.json", label="qualification attempt"
    )
    expected_calls = [*(["fit"] * 10), "prepare-manifest", "analyze"]
    if analyzer_case in {"accepted", "accepted-postcondition-failure"}:
        expected_calls.append("soak")
    assert calls == expected_calls
    assert attempt["state"] == "canonical-root-consumed-at-creation"
    assert (
        attempt["root_consumption_policy"] == qualification._root_consumption_policy()
    )
    assert attempt["frozen_analyzer_source"]["sha256"] == (
        qualification.FROZEN_ANALYZER_SOURCE_SHA256
    )
    evidence_path = output / "resource-lifecycle-qualification.json"
    if analyzer_case == "accepted":
        evidence = qualification._load_signed_json(
            evidence_path, label="qualification accepted"
        )
        assert evidence["status"] == "qualified"
        assert evidence["passed"] is True
        assert evidence["admission"]["decision"] == "admitted"
        assert evidence["ab"]["repeat_count_per_mode"] == 5
        assert evidence["ab"]["equivalence"]["decision"]["accepted"] is True
        assert len(evidence["soak"]["child_evidence"]["fits"]) == 243
        assert evidence["soak"]["passed"] is True
        assert not (output / "tmp").exists()
    elif analyzer_case == "scientific-no-go":
        evidence = qualification._load_signed_json(
            evidence_path, label="qualification no-go"
        )
        assert evidence["schema_version"] == 2
        assert evidence["status"] == "admission-inconclusive"
        assert evidence["passed"] is False
        assert evidence["admission"] == {
            "decision": "inconclusive",
            "terminal": True,
            "analyzer_outcome": "scientific-no-go",
            "analyzer_no_go_interpretation": (
                qualification.ANALYZER_NO_GO_INTERPRETATION
            ),
            "wrapper_inequivalence_proven": False,
            "retry_permitted": False,
            "in_place_reuse_permitted": False,
        }
        assert evidence["root_consumption_policy"] == (
            qualification._root_consumption_policy()
        )
        assert evidence["soak"] is None
        assert not (output / "tmp").exists()
    else:
        assert not evidence_path.exists()
    if analyzer_case not in {"accepted", "accepted-postcondition-failure"}:
        assert not (output / "soak").exists()
    assert all(
        not (output / "ab" / mode / repeat / "dataset/outputs").exists()
        for mode in ("original", "wrapped")
        for repeat in [f"repeat-{index:03d}" for index in range(5)]
    )
    with pytest.raises(ValueError, match="output already exists"):
        qualification._run(arguments)
