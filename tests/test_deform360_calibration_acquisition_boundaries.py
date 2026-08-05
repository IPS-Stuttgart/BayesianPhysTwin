# ruff: noqa: F403, F405
from deform360_calibration_acquisition_test_support import *

def test_official_processing_uses_local_zero_but_retains_source_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    object_id = "sheet-0"
    source_episode = 7
    raw_object = tmp_path / "raw" / object_id
    raw_object.mkdir(parents=True)
    (raw_object / "metadata.json").write_text(
        json.dumps({"sequences": {"7": {"bimanual": "no"}}}),
        encoding="utf-8",
    )

    calls: list[tuple[str, int]] = []

    def undistort_episode(
        object_dir: Path,
        output_dir: Path,
        episode_index: int,
        **_: Any,
    ) -> Path:
        assert object_dir == raw_object
        calls.append(("undistort", episode_index))
        episode = output_dir / "episode_0000"
        camera = episode / "brics-odroid-001_cam0"
        camera.mkdir(parents=True)
        (camera / "undistorted.mp4").write_bytes(b"video")
        (camera / "aligned_timestamps.txt").write_text(
            "0.0 0\n",
            encoding="utf-8",
        )
        (episode / "alignment.json").write_text(
            json.dumps({"frame_count": 1}),
            encoding="utf-8",
        )
        return episode

    def process_tactile_episode(
        object_dir: Path,
        aligned_dir: Path,
        episode_index: int,
        **_: Any,
    ) -> dict[str, Path]:
        assert object_dir == raw_object
        calls.append(("tactile", episode_index))
        output = aligned_dir / "episode_0000" / "brics-odroid_tactile0"
        output.mkdir(parents=True)
        path = output / "synced_tactile.npy"
        path.write_bytes(b"tactile")
        return {"brics-odroid_tactile0": path}

    def process_robot_episode(
        aligned_dir: Path,
        episode_index: int,
        **_: Any,
    ) -> Path:
        calls.append(("robot", episode_index))
        output = aligned_dir / "episode_0000" / "robot" / "robot.npz"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"robot")
        return output

    package = types.ModuleType("deform360")
    package.__dict__["__path__"] = []
    package.__dict__["tactile"] = types.SimpleNamespace(
        process_tactile_episode=process_tactile_episode
    )
    package.__dict__["undistort"] = types.SimpleNamespace(
        undistort_episode=undistort_episode
    )
    processing = types.ModuleType("deform360.processing")
    processing.__dict__["__path__"] = []
    processing.__dict__["robot_stage"] = types.SimpleNamespace(
        process_robot_episode=process_robot_episode
    )
    monkeypatch.setitem(sys.modules, "deform360", package)
    monkeypatch.setitem(sys.modules, "deform360.processing", processing)

    case = module._process_case(
        plan_id=_digest("plan"),
        object_id=object_id,
        episode_id=source_episode,
        stratum="sheet",
        data_root=tmp_path,
        raw_artifacts={
            f"raw/{object_id}/metadata.json": _digest("metadata")
        },
        failure_root=tmp_path / "failures",
    )

    assert case.status == "prepared"
    assert calls == [("undistort", 0), ("tactile", 0), ("robot", 0)]
    assert case.metadata["source_episode_id"] == source_episode
    assert case.metadata["official_processing_episode_index"] == 0
    assert any(
        f"source_episode_{source_episode:04d}/episode_0000" in path
        for path in case.output_artifacts
    )


def test_processing_failure_is_bound_to_a_redacted_failure_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    object_id = "sheet-0"
    raw_object = tmp_path / "raw" / object_id
    raw_object.mkdir(parents=True)
    (raw_object / "metadata.json").write_text(
        json.dumps({"sequences": {"0": {"bimanual": "no"}}}),
        encoding="utf-8",
    )

    def fail_undistort(*_: Any, **__: Any) -> Path:
        raise RuntimeError(f"failed below {tmp_path}")

    package = types.ModuleType("deform360")
    package.__dict__["__path__"] = []
    package.__dict__["tactile"] = types.SimpleNamespace(
        process_tactile_episode=lambda *args, **kwargs: {}
    )
    package.__dict__["undistort"] = types.SimpleNamespace(
        undistort_episode=fail_undistort
    )
    processing = types.ModuleType("deform360.processing")
    processing.__dict__["__path__"] = []
    processing.__dict__["robot_stage"] = types.SimpleNamespace(
        process_robot_episode=lambda *args, **kwargs: None
    )
    monkeypatch.setitem(sys.modules, "deform360", package)
    monkeypatch.setitem(sys.modules, "deform360.processing", processing)

    case = module._process_case(
        plan_id=_digest("plan"),
        object_id=object_id,
        episode_id=0,
        stratum="sheet",
        data_root=tmp_path,
        raw_artifacts={f"raw/{object_id}/metadata.json": _digest("metadata")},
        failure_root=tmp_path / "failures",
    )
    assert case.status == "technical_failure"
    log = tmp_path / "failures" / f"{object_id}-episode-0000.txt"
    assert log.is_file()
    assert str(tmp_path) not in log.read_text(encoding="utf-8")
    assert case.metadata["failure_log_sha256"] == module.file_sha256(log)
    expected_message = (
        "RuntimeError:failed below <CALIBRATION_DATA_ROOT>"
    )
    assert case.failure_message_sha256 == module._sha256_text(expected_message)
    assert case.output_artifacts == {}


def test_clean_checkout_rejects_untracked_source(tmp_path: Path) -> None:
    module = _script_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    module._require_clean(repository, name="fixture")
    nested = repository / "_deform360"
    nested.mkdir()
    (nested / "source.py").write_text("pinned = True\n", encoding="utf-8")
    module._require_clean(
        repository,
        name="fixture",
        allowed_untracked=("_deform360",),
    )
    (repository / "untracked.py").write_text("unsafe = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked or untracked"):
        module._require_clean(
            repository,
            name="fixture",
            allowed_untracked=("_deform360",),
        )
    with pytest.raises(ValueError, match="canonical relative paths"):
        module._require_clean(
            repository,
            name="fixture",
            allowed_untracked=(".",),
        )


def test_data_and_evidence_roots_must_be_separate_from_source(
    tmp_path: Path,
) -> None:
    module = _script_module()
    repository = tmp_path / "repository"
    deform360 = repository / "_deform360"
    data_root = tmp_path / "calibration-data"
    output = tmp_path / "compact-evidence"
    for path in (repository, deform360, data_root, output):
        path.mkdir(parents=True, exist_ok=True)
    module._require_separate_data_and_output_roots(
        repository=repository,
        deform360_checkout=deform360,
        data_root=data_root,
        output=output,
    )
    with pytest.raises(ValueError, match="data root overlaps"):
        module._require_separate_data_and_output_roots(
            repository=repository,
            deform360_checkout=deform360,
            data_root=repository / "data",
            output=output,
        )
    with pytest.raises(ValueError, match="compact evidence roots overlap"):
        module._require_separate_data_and_output_roots(
            repository=repository,
            deform360_checkout=deform360,
            data_root=data_root,
            output=data_root / "evidence",
        )


def test_download_rejects_symlinked_local_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    object_id = "sheet-0"
    selected = (f"raw/{object_id}/metadata.json",)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    def hf_hub_download(**arguments: Any) -> str:
        path = Path(arguments["local_dir"]) / arguments["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks are unavailable")
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    with pytest.raises(ValueError, match="ordinary file"):
        module._download_unit(
            repository="brownu/deform360",
            revision="a" * 40,
            object_id=object_id,
            episode_id=0,
            selected_paths=selected,
            expected_metadata_sha256=_digest("metadata"),
            data_root=tmp_path / "data",
            token=None,
        )


def test_downloaded_metadata_must_match_stage_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    object_id = "sheet-0"
    selected = (
        f"raw/{object_id}/metadata.json",
        f"raw/{object_id}/calibration_refined/dist.npy",
    )

    def hf_hub_download(**arguments: Any) -> str:
        path = Path(arguments["local_dir"]) / arguments["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arguments["filename"].encode("utf-8"))
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )
    with pytest.raises(ValueError, match="downloaded metadata changed"):
        module._download_unit(
            repository="brownu/deform360",
            revision="a" * 40,
            object_id=object_id,
            episode_id=0,
            selected_paths=selected,
            expected_metadata_sha256=_digest("different"),
            data_root=tmp_path,
            token=None,
        )
