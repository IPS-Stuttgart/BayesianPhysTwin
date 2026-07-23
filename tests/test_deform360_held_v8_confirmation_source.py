from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import bayesian_phystwin.deform360_held_v8_confirmation_source as source


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_blob_id(payload: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - Git fixture identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def _payload(relative: str) -> bytes:
    return f"confirmation-source-fixture:{relative}\n".encode()


def _remote_record(relative: str) -> dict[str, Any]:
    payload = _payload(relative)
    return {
        "path": relative,
        "size": len(payload),
        "blob_id": _git_blob_id(payload),
        "lfs": {"sha256": hashlib.sha256(payload).hexdigest()},
    }


def _synthetic_inventories() -> tuple[
    tuple[source.ConfirmationSourceCase, ...],
    dict[str, list[dict[str, Any]]],
    dict[str, bytes],
]:
    cases: list[source.ConfirmationSourceCase] = []
    inventories: dict[str, list[dict[str, Any]]] = {}
    payloads: dict[str, bytes] = {}
    for template in source.CONFIRMATION_SOURCE_CASES:
        root = f"raw/{template.object_id}"
        records: list[dict[str, Any]] = []
        selected_camera_paths: list[str] = []
        selected_legacy_paths: list[str] = []
        for stream in (*template.cameras, *source.TACTILE_STREAMS):
            extensions = (
                ("npy", "txt")
                if stream in source.TACTILE_STREAMS
                else (
                    "mp4",
                    "txt",
                )
            )
            for episode in range(10):
                stem = f"{stream}_{episode:04d}"
                for extension in extensions:
                    relative = f"{root}/{stream}/{stem}.{extension}"
                    records.append(_remote_record(relative))
                    payloads[relative] = _payload(relative)
                    if episode == template.episode_id:
                        selected_legacy_paths.append(relative)
                        if stream not in source.TACTILE_STREAMS:
                            selected_camera_paths.append(relative)
        for relative_tail in source.SHARED_RELATIVE_PATHS:
            relative = f"{root}/{relative_tail}"
            records.append(_remote_record(relative))
            payloads[relative] = _payload(relative)
        by_path = {record["path"]: record for record in records}
        legacy = []
        for relative in sorted(selected_legacy_paths):
            record = by_path[relative]
            legacy.append(
                {
                    "path": record["path"],
                    "size": record["size"],
                    "sha256": record["lfs"]["sha256"],
                    "blob_id": record["blob_id"],
                }
            )
        processing_paths = sorted(
            [
                *selected_camera_paths,
                *(f"{root}/{relative}" for relative in source.SHARED_RELATIVE_PATHS),
            ]
        )
        processing = []
        for relative in processing_paths:
            record = by_path[relative]
            processing.append(
                {
                    "path": record["path"],
                    "size": record["size"],
                    "sha256": record["lfs"]["sha256"],
                    "blob_id": record["blob_id"],
                }
            )
        replacement_schema = [
            {
                "path": record["path"],
                "size_bytes": record["size"],
                "blob_id": record["blob_id"],
                "lfs_sha256": record["sha256"],
            }
            for record in processing
        ]
        cases.append(
            source.ConfirmationSourceCase(
                case_name=template.case_name,
                object_id=template.object_id,
                episode_id=template.episode_id,
                remote_inventory_sha256=_canonical_digest(legacy),
                remote_file_count=len(legacy),
                remote_total_bytes=sum(record["size"] for record in legacy),
                processing_inventory_sha256=_canonical_digest(replacement_schema),
                processing_total_bytes=sum(record["size"] for record in processing),
                bimanual=template.bimanual,
                cameras=template.cameras,
            )
        )
        inventories[template.object_id] = records
    return tuple(cases), inventories, payloads


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fake_episode(
    output_root: Path,
    *,
    cameras: list[str],
) -> None:
    episode = output_root / "episode_0000"
    episode.mkdir(parents=True)
    for name in source._EPISODE_OUTPUT_FILES:
        path = episode / name
        if name == "alignment.json":
            _write_json(
                path,
                {
                    "episode_index": 0,
                    "cameras": cameras,
                    "frame_count": 2,
                },
            )
        else:
            path.write_bytes(f"fixture:{name}".encode())
    for camera in cameras:
        directory = episode / camera
        directory.mkdir()
        for name in source._CAMERA_OUTPUT_FILES:
            path = directory / name
            if name.endswith(".json"):
                _write_json(path, {"camera": camera})
            else:
                path.write_bytes(f"fixture:{camera}:{name}".encode())


def _fake_robot(
    aligned_object: Path,
    *,
    episode_id: int,
    cameras: list[str],
    bimanual: bool,
) -> None:
    robot = aligned_object / f"episode_{episode_id:04d}" / "robot"
    robot.mkdir()
    (robot / "robot.npz").write_bytes(b"synthetic-robot-archive")
    _write_json(
        robot / "robot.meta.json",
        {
            "parameters": {
                "seed": 0,
                "bimanual": bimanual,
                "cameras": cameras,
            },
            "outputs": {"bimanual": bimanual, "num_frames": 2},
        },
    )


def _thaw_and_remove(root: Path) -> None:
    if not os.path.lexists(root):
        return
    for current, directories, files in os.walk(root, topdown=True):
        current_path = Path(current)
        current_path.chmod(0o700)
        for name in directories:
            (current_path / name).chmod(0o700)
        for name in files:
            (current_path / name).chmod(0o600)
    shutil.rmtree(root)


def _install_synthetic_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    cases, inventories, payloads = _synthetic_inventories()
    names = tuple(case.case_name for case in cases)
    contract = deepcopy(source.CONFIRMATION_SOURCE_CONTRACT)
    contract["ordered_case_names"] = list(names)
    monkeypatch.setattr(source, "CONFIRMATION_SOURCE_CASES", cases)
    monkeypatch.setattr(source, "CONFIRMATION_SOURCE_CASE_NAMES", names)
    monkeypatch.setattr(source, "CONFIRMATION_SOURCE_CONTRACT", contract)

    processing = tmp_path / "processing"
    processing.mkdir()
    python = tmp_path / "runtime" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic-python")
    python.chmod(0o700)
    python_identity = {
        "python_executable": str(python),
        "python_link_target": source.PINNED_PYTHON_LINK_TARGET,
        "python_resolved": str(source.PINNED_PYTHON_RESOLVED),
        "python_resolved_sha256": source.PINNED_PYTHON_TARGET_SHA256,
    }
    monkeypatch.setattr(
        source, "_validated_python", lambda _path: dict(python_identity)
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return source.PROCESSING_CODE_REVISION
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return source.PROCESSING_CODE_TREE
        if arguments[0] in {"status", "ls-files"}:
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(source, "_git_value", fake_git)
    touched = {"provider": 0, "downloader": 0, "consumer": 0}

    def provider(**kwargs: Any) -> list[dict[str, Any]]:
        touched["provider"] += 1
        object_id = str(kwargs["path_in_repo"]).split("/", 1)[1]
        return inventories[object_id]

    def downloader(**kwargs: Any) -> str:
        touched["downloader"] += 1
        patterns = list(kwargs["allow_patterns"])
        assert len(patterns) == 168
        assert not any(
            len(Path(path).parts) == 4 and Path(path).parts[2] in source.TACTILE_STREAMS
            for path in patterns
        )
        root = Path(kwargs["local_dir"])
        for relative in patterns:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payloads[relative])
        return str(root)

    case_by_object = {case.object_id: case for case in cases}

    def command_runner(
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> object:
        assert cwd.name.startswith(".confirmation-source.partial.")
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        if command[4] == source._UNDISTORT_SCRIPT:
            _fake_episode(
                Path(command[7]),
                cameras=json.loads(command[8]),
            )
        elif command[4] == source._ROBOT_SCRIPT:
            _fake_robot(
                Path(command[6]),
                episode_id=int(command[7]),
                bimanual=command[8] == "true",
                cameras=json.loads(command[9]),
            )
        else:
            raise AssertionError("unexpected processing command")
        return object()

    def load_robot(path: Path, *, expected_frame_count: int) -> SimpleNamespace:
        assert expected_frame_count == 2
        object_id = path.parents[2].name
        return SimpleNamespace(bimanual=case_by_object[object_id].bimanual)

    monkeypatch.setattr(source, "load_robot_kinematics_archive", load_robot)
    expected_permit = {
        "protocol_id": source.PROTOCOL_ID,
        "role": "confirmation",
        "operation": source.SOURCE_OPERATION,
        "ordered_case_names": list(names),
        "single_use_consumed": True,
    }

    def consume(
        permit: object,
        *,
        operation: str,
        ordered_case_names: tuple[str, ...],
    ) -> dict[str, Any]:
        touched["consumer"] += 1
        assert permit is permit_token
        assert operation == source.SOURCE_OPERATION
        assert tuple(ordered_case_names) == names
        return dict(expected_permit)

    permit_token = object()
    source_root = tmp_path / "held" / "confirmation-source"
    source_root.parent.mkdir()
    paths = source.ConfirmationSourcePaths(
        source_root=source_root,
        processing_code_root=processing,
        python_executable=python,
    )
    return SimpleNamespace(
        cases=cases,
        names=names,
        inventories=inventories,
        payloads=payloads,
        touched=touched,
        provider=provider,
        downloader=downloader,
        command_runner=command_runner,
        expected_permit=expected_permit,
        consume=consume,
        permit=permit_token,
        paths=paths,
        source_root=source_root,
    )


def _materialize(environment: SimpleNamespace) -> Path:
    return source.materialize_confirmation_source_cohort(
        environment.paths,
        source_permit=environment.permit,
        consume_source_permit=environment.consume,
        expected_source_permit=environment.expected_permit,
        inventory_provider=environment.provider,
        snapshot_downloader=environment.downloader,
        command_runner=environment.command_runner,
    )


def test_locked_exact_six_inventories_remain_frozen() -> None:
    expected = {
        "002-rope-silk-ep0001": (
            "b33791f6faa8d05717408d7b77cf1405083b614fe42ecef3a3538a0dc2008858",
            32,
            37_863_432,
        ),
        "081-stripe-rope-ep0005": (
            "6055375fb66ea1e0732e808d855e4eecb66687f14dfd6a6a604d5d9a39a194e0",
            32,
            61_222_868,
        ),
        "085-scarf-cloth-ep0002": (
            "cb9ee9be4c99244e94f676a329b31ecb629c0afef9b7ffbe6060a6b061b81249",
            32,
            31_710_094,
        ),
        "083-blanket-cloth-ep0007": (
            "102f9edd98b6d3703c3d98625a358c7588d87c79024c795d233771e76b10be84",
            32,
            53_947_570,
        ),
        "092-squirrel-ep0001": (
            "6f02afc8e8101fdc0e30ee171435162d1d6a4d648f5ee910070f711313d2b960",
            32,
            38_161_504,
        ),
        "170-spider-ep0006": (
            "c19cb57b087aa98c5e792e8dfcb2e889cb4b2a52653a78a2cba6591a0fdc80a7",
            32,
            47_269_453,
        ),
    }
    observed = {
        case.case_name: (
            case.remote_inventory_sha256,
            case.remote_file_count,
            case.remote_total_bytes,
        )
        for case in source.CONFIRMATION_SOURCE_CASES
    }
    assert observed == expected
    expected_processing = {
        "002-rope-silk-ep0001": (
            "e22d487e67bdfd9c80ce2fe17948034705cd8919bb3f3f06093f92f7dc04f821",
            35_071_469,
        ),
        "081-stripe-rope-ep0005": (
            "b4e3233b7b099a095598964ac908e7b31f22059671080152baf6adaa764cc523",
            57_750_695,
        ),
        "085-scarf-cloth-ep0002": (
            "9e6ad1ce079d56764b085a9a67525d14020319a67806d106dfd755d25abc3329",
            29_598_061,
        ),
        "083-blanket-cloth-ep0007": (
            "4c0416e1218d6a6579d2ea3f1c7b06e810e0d55230fa08b5530f7060ea71fe4f",
            50_787_356,
        ),
        "092-squirrel-ep0001": (
            "79fec8a5ff5cccae59af499129d5c572368e8aeea91003530b3e0d31e86d0bfa",
            35_012_467,
        ),
        "170-spider-ep0006": (
            "babf0e8e6e4555442e923ae5f0059480c940fcf8e859ed1e80b1bcc7f1227527",
            44_778_111,
        ),
    }
    assert {
        case.case_name: (
            case.processing_inventory_sha256,
            case.processing_total_bytes,
        )
        for case in source.CONFIRMATION_SOURCE_CASES
    } == expected_processing
    assert len(source.CONFIRMATION_SOURCE_CASES) == 6
    assert all(len(case.cameras) == 12 for case in source.CONFIRMATION_SOURCE_CASES)


def test_materializer_consumes_permit_before_any_path_or_provider_touch() -> None:
    touched = {"provider": False}

    class PoisonPaths:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"path touched before permit rejection: {name}")

    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("permit rejected")

    def provider(**_kwargs: Any) -> list[object]:
        touched["provider"] = True
        return []

    with pytest.raises(ValueError, match="permit rejected"):
        source.materialize_confirmation_source_cohort(
            PoisonPaths(),  # type: ignore[arg-type]
            source_permit=object(),
            consume_source_permit=reject,
            expected_source_permit={},
            inventory_provider=provider,
        )
    assert touched == {"provider": False}


def test_exact_six_case_materialization_is_metadata_only_for_tactile_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _install_synthetic_environment(monkeypatch, tmp_path)
    original_core = source._validate_confirmation_source_cohort_manifest_at_root
    validations: list[tuple[Path, Path]] = []

    def observe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        validations.append(
            (kwargs["actual_source_root"], kwargs["declared_source_root"])
        )
        return original_core(*args, **kwargs)

    monkeypatch.setattr(
        source, "_validate_confirmation_source_cohort_manifest_at_root", observe
    )
    try:
        manifest_path = _materialize(environment)
        assert environment.touched == {
            "provider": 6,
            "downloader": 1,
            "consumer": 1,
        }
        assert len(validations) == 2
        assert validations[0][0].name.startswith(".confirmation-source.partial.")
        assert validations[0][1] == environment.source_root
        assert validations[1] == (environment.source_root, environment.source_root)

        manifest = source.validate_confirmation_source_cohort_manifest(
            manifest_path,
            expected_source_permit=environment.expected_permit,
            verify_content=True,
        )
        assert len(validations) == 3
        assert manifest["ordered_case_names"] == list(environment.names)
        assert [case["case_name"] for case in manifest["cases"]] == list(
            environment.names
        )
        for case in manifest["cases"]:
            legacy_paths = [
                record["path"] for record in case["legacy_remote_inventory"]["records"]
            ]
            processing_paths = [
                record["path"]
                for record in case["processing_remote_inventory"]["records"]
            ]
            downloaded_paths = [
                record["path"] for record in case["downloaded_raw_records"]
            ]
            assert len(legacy_paths) == 32
            assert len(processing_paths) == len(downloaded_paths) == 28
            assert (
                sum(
                    len(Path(path).parts) == 4
                    and Path(path).parts[2] in source.TACTILE_STREAMS
                    for path in legacy_paths
                )
                == 8
            )
            assert not any(
                len(Path(path).parts) == 4
                and Path(path).parts[2] in source.TACTILE_STREAMS
                for path in [*processing_paths, *downloaded_paths]
            )

        expected_objects = {case.object_id for case in environment.cases}
        assert {
            path.name for path in (environment.source_root / "aligned").iterdir()
        } == expected_objects
        for entry in environment.source_root.rglob("*"):
            observed = os.lstat(entry)
            if stat.S_ISDIR(observed.st_mode):
                assert stat.S_IMODE(observed.st_mode) == 0o500
            else:
                assert stat.S_ISREG(observed.st_mode)
                assert observed.st_nlink == 1
                assert stat.S_IMODE(observed.st_mode) == 0o400

        first_record = manifest["cases"][0]["downloaded_raw_records"][0]
        changed = environment.source_root / "download" / first_record["path"]
        changed.chmod(0o600)
        changed.write_bytes(changed.read_bytes() + b"tamper")
        changed.chmod(0o400)
        with pytest.raises(ValueError, match="downloaded raw"):
            source.validate_confirmation_source_cohort_manifest(
                manifest_path,
                expected_source_permit=environment.expected_permit,
                verify_content=True,
            )
    finally:
        _thaw_and_remove(environment.source_root)


@pytest.mark.parametrize("failure_boundary", ["pre-rename", "post-rename"])
def test_validation_failure_never_leaves_a_published_or_partial_cohort(
    failure_boundary: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _install_synthetic_environment(monkeypatch, tmp_path)
    original_core = source._validate_confirmation_source_cohort_manifest_at_root
    call_count = 0

    def fail_at_boundary(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if (
            failure_boundary == "pre-rename"
            or failure_boundary == "post-rename"
            and call_count == 2
        ):
            raise ValueError(f"{failure_boundary} validation failure")
        return original_core(*args, **kwargs)

    monkeypatch.setattr(
        source,
        "_validate_confirmation_source_cohort_manifest_at_root",
        fail_at_boundary,
    )
    with pytest.raises(ValueError, match=failure_boundary):
        _materialize(environment)
    assert not os.path.lexists(environment.source_root)
    assert not list(
        environment.source_root.parent.glob(".confirmation-source.partial.*")
    )
    assert call_count == (1 if failure_boundary == "pre-rename" else 2)


def test_content_validation_rejects_symlink_wrong_blob_and_hash_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    linked = raw / "linked.bin"
    linked.symlink_to(target)
    with pytest.raises(ValueError, match="link"):
        source._content_records(
            raw,
            (
                {
                    "path": "linked.bin",
                    "size": len(b"outside"),
                    "sha256": hashlib.sha256(b"outside").hexdigest(),
                    "blob_id": "a" * 40,
                },
            ),
        )
    linked.unlink()

    payload = b"ordinary Git payload"
    ordinary = raw / "ordinary.bin"
    ordinary.write_bytes(payload)
    with pytest.raises(ValueError, match="Git blob changed"):
        source._content_records(
            raw,
            (
                {
                    "path": "ordinary.bin",
                    "size": len(payload),
                    "sha256": None,
                    "blob_id": "0" * 40,
                },
            ),
        )

    racing = tmp_path / "racing.bin"
    racing.write_bytes(b"x" * (2 * 1024 * 1024))
    original_read = source.os.read
    mutated = False

    def read_and_mutate(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        block = original_read(descriptor, count)
        if block and not mutated:
            mutated = True
            observed = os.lstat(racing)
            os.utime(
                racing,
                ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
            )
        return block

    monkeypatch.setattr(source.os, "read", read_and_mutate)
    with pytest.raises(ValueError, match="changed while hashing"):
        source._sha256_file(racing)
    assert mutated


def test_owned_tree_cleanup_refuses_a_replaced_root(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    observed = os.lstat(owned)
    identity = (observed.st_dev, observed.st_ino)
    owned.rename(tmp_path / "displaced")
    owned.mkdir()

    with pytest.raises(ValueError, match="replaced confirmation source root"):
        source._remove_owned_tree(owned, expected_identity=identity)
    assert owned.is_dir()


def _load_confirmation_wrapper() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "held"
        / "run_deform360_v8_confirmation_source.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_confirmation_source_wrapper", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def test_wrapper_deployment_census_explicitly_rejects_ignored_files(
    tmp_path: Path,
) -> None:
    wrapper = _load_confirmation_wrapper()
    stage = tmp_path / "stage"
    stage.mkdir()
    _git(stage, "init", "--quiet")
    _git(stage, "config", "user.email", "held-test@example.invalid")
    _git(stage, "config", "user.name", "Held Test")
    (stage / ".gitignore").write_text("ignored-runtime.py\n", encoding="utf-8")
    (stage / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(stage, "add", ".gitignore", "tracked.py")
    _git(stage, "commit", "--quiet", "-m", "fixture")
    _git(stage, "checkout", "--quiet", "--detach", "HEAD")
    head = _git(stage, "rev-parse", "HEAD").decode("ascii").strip()
    code = tmp_path / f"code-{head}"
    stage.rename(code)
    tree = wrapper._git_tree_records(_git(code, "ls-tree", "-r", "-z", "HEAD"))
    bindings = {
        "method_head_text_sha256": hashlib.sha256(head.encode()).hexdigest(),
        "method_deployed_snapshot_tree": hashlib.sha256(
            wrapper._canonical_bytes(tree)
        ).hexdigest(),
    }
    wrapper._validate_deployed_repository(code, bindings)

    (code / "ignored-runtime.py").write_text("VALUE = 'shadow'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean exact worktree"):
        wrapper._validate_deployed_repository(code, bindings)


def test_wrapper_rejects_camera_episode_or_bimanual_lineage_drift() -> None:
    wrapper = _load_confirmation_wrapper()
    repository = Path(__file__).parents[1]
    geometry_qa = json.loads(
        (
            repository
            / "milestones"
            / "deform360-replication-source-qa-v1"
            / "artifacts"
            / "source_geometry_qa.json"
        ).read_text(encoding="utf-8")
    )
    preregistration = json.loads(
        (
            repository / "configs" / "causal4d_public" / "deform360_replication_v1.json"
        ).read_text(encoding="utf-8")
    )
    wrapper._validate_confirmation_selection_lineage(
        geometry_qa=geometry_qa,
        preregistration=preregistration,
        source=source,
    )

    changed_qa = deepcopy(geometry_qa)
    changed_qa["objects"][0]["selected_cameras"][0:2] = reversed(
        changed_qa["objects"][0]["selected_cameras"][0:2]
    )
    with pytest.raises(ValueError, match="camera-selection QA differs"):
        wrapper._validate_confirmation_selection_lineage(
            geometry_qa=changed_qa,
            preregistration=preregistration,
            source=source,
        )

    changed_episode = deepcopy(preregistration)
    changed_episode["config"]["cohort"][0]["target_episode_id"] += 1
    with pytest.raises(ValueError, match="preregistration target differs"):
        wrapper._validate_confirmation_selection_lineage(
            geometry_qa=geometry_qa,
            preregistration=changed_episode,
            source=source,
        )

    changed_bimanual = deepcopy(preregistration)
    changed_bimanual["config"]["cohort"][0]["target_bimanual"] = not (
        changed_bimanual["config"]["cohort"][0]["target_bimanual"]
    )
    with pytest.raises(ValueError, match="preregistration target differs"):
        wrapper._validate_confirmation_selection_lineage(
            geometry_qa=geometry_qa,
            preregistration=changed_bimanual,
            source=source,
        )

    changed_order = deepcopy(preregistration)
    changed_order["config"]["cohort"][0:2] = reversed(
        changed_order["config"]["cohort"][0:2]
    )
    with pytest.raises(ValueError, match="cohort order differs"):
        wrapper._validate_confirmation_selection_lineage(
            geometry_qa=geometry_qa,
            preregistration=changed_order,
            source=source,
        )


def test_wrapper_cache_cleanup_failure_rolls_back_only_published_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _load_confirmation_wrapper()
    monkeypatch.setattr(wrapper, "HELD_ROOT", tmp_path)
    runtime = wrapper._runtime_root()
    runtime.mkdir()
    (runtime / "cache.bin").write_bytes(b"cache")
    runtime_state = os.lstat(runtime)
    runtime_identity = (runtime_state.st_dev, runtime_state.st_ino)

    published = tmp_path / "confirmation-source"
    published.mkdir()
    (published / "manifest.bin").write_bytes(b"published")
    published_state = os.lstat(published)
    published_identity = (published_state.st_dev, published_state.st_ino)
    monkeypatch.setattr(wrapper, "SOURCE_ROOT", published)
    monkeypatch.setattr(
        wrapper,
        "_remove_owned_runtime_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected runtime cleanup failure")
        ),
    )

    with pytest.raises(OSError, match="injected runtime cleanup failure"):
        wrapper._cleanup_runtime_or_rollback_source(
            runtime=runtime,
            runtime_identity=runtime_identity,
            published_source_identity=published_identity,
            source_module=source,
        )
    assert not os.path.lexists(published)
    assert runtime.is_dir()
    assert wrapper._runtime_root() == tmp_path / ".confirmation-source-runtime"
    assert wrapper._runtime_root().parent == wrapper.HELD_ROOT
