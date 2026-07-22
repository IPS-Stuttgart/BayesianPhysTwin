from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "scripts" / "held" / "seal_deform360_v8_attempt4_outcome_failure.py"


def _module():
    spec = importlib.util.spec_from_file_location("v81_attempt4_withdrawal", OPERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: bytes, mode: int = 0o400) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _git(code: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(code), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _make_writable(root: Path) -> None:
    if not os.path.lexists(root):
        return
    state = os.lstat(root)
    if stat.S_ISREG(state.st_mode):
        os.chmod(root, 0o600, follow_symlinks=False)
        return
    if not stat.S_ISDIR(state.st_mode):
        return
    os.chmod(root, 0o700, follow_symlinks=False)
    for current, directories, files in os.walk(root):
        for name in directories:
            os.chmod(Path(current) / name, 0o700, follow_symlinks=False)
        for name in files:
            os.chmod(Path(current) / name, 0o600, follow_symlinks=False)


def _seal_code(code: Path) -> None:
    for current, directories, files in os.walk(code, topdown=False):
        for name in files:
            os.chmod(Path(current) / name, 0o444, follow_symlinks=False)
        for name in directories:
            os.chmod(Path(current) / name, 0o555, follow_symlinks=False)
    code.chmod(0o555)


def _make_code(module, monkeypatch: pytest.MonkeyPatch, active: Path) -> dict:
    staging = active / "code-staging"
    staging.mkdir()
    _git(staging, "init", "--quiet")
    _git(staging, "config", "user.name", "Held Test")
    _git(staging, "config", "user.email", "held@example.invalid")
    (staging / "method.txt").write_text("frozen method\n", encoding="utf-8")
    _git(staging, "add", "method.txt")
    _git(staging, "commit", "--quiet", "-m", "fixture")
    head = _git(staging, "rev-parse", "HEAD")
    code = active / f"code-{head}"
    staging.rename(code)

    rows = module._parse_git_tree(
        module._run_git(code, ["ls-tree", "-r", "-z", "HEAD"])
    )
    tree_sha256 = hashlib.sha256(module._canonical_bytes(rows)).hexdigest()
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_HEAD", head)
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_CODE_NAME", code.name)
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_TREE_RECORD_COUNT", len(rows))
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_TREE_SHA256", tree_sha256)
    binding = module._repository_binding(code)
    _seal_code(code)
    return binding


def _make_preoutcome_cases(module, active: Path) -> None:
    cases = active / "calibration" / "cases"
    for case in module.EXPECTED_CASES:
        for relative in (
            "frame-zero/frame_zero_bundle.manifest.json",
            "physical/physical_prior_seal.json",
            "prefix-authorization.json",
            "online/online_prediction_seal.json",
            "frozen-field/preoutcome-frozen-field-manifest.json",
        ):
            _write(cases / case / relative, b"opaque sealed pre-outcome metadata\n")


def _make_output_roots(module, calibration: Path) -> tuple[Path, Path, Path]:
    roots = tuple(
        calibration / name
        for name in ("private-targets", "query-inputs", "query-outputs")
    )
    for root in roots:
        root.mkdir()
        root.chmod(0o700)
        for case in module.EXPECTED_CASES:
            (root / case).mkdir()
            (root / case).chmod(0o700)
    return roots


def _make_completed_pairs(
    module, private: Path, query_inputs: Path, query_outputs: Path
) -> None:
    for case in module.COMPLETED_OUTCOME_CASES:
        reconstruction = private / case / "fresh-official-reconstruction"
        reconstruction.mkdir()
        _write(
            reconstruction / "held-v8-official-reconstruction-audit.json",
            b"opaque reconstruction audit, deliberately not JSON\x00",
        )
        episode = reconstruction / "staged-aligned" / "episode_0000"
        for camera in module.COMPLETED_CAMERAS[case]:
            camera_root = episode / camera
            for name in (
                "aligned_timestamps.txt",
                "mask_refined.h5",
                "metadata.json",
                "rendered_depth.h5",
                "rendered_depth.meta.json",
                "undistorted.mp4",
            ):
                _write(camera_root / name, b"opaque completed camera payload\x00")
            for name in ("tracking.meta.json", "vel.h5", "visibility.h5"):
                _write(
                    camera_root / "tracking" / name,
                    b"opaque completed tracking payload\x00",
                )
        _write(episode / "extrinsics.npy", b"opaque completed extrinsics\x00")
        _write(
            episode / "undistorted_intrinsics.npy",
            b"opaque completed intrinsics\x00",
        )
        for frame in range(module.COMPLETED_PCD_FRAME_COUNT):
            _write(
                episode / "pcd_clean" / f"{frame:06d}.npz",
                b"opaque completed clean point cloud\x00",
            )
        _write(
            episode / "pcd_clean/pcd_clean.meta.json",
            b"opaque completed clean point-cloud metadata\x00",
        )
        _write(
            episode / "robot/robot.meta.json", b"opaque completed robot metadata\x00"
        )
        _write(episode / "robot/robot.npz", b"opaque completed robot archive\x00")
        for frame in range(module.COMPLETED_SPLAT_FRAME_COUNT):
            _write(
                episode / "splatfacto" / f"splat_{frame}.ply",
                b"opaque completed point cloud\x00",
            )
        _write(
            episode / "splatfacto/splatfacto.meta.json",
            b"opaque completed splat metadata\x00",
        )
        for current, directories, _files in os.walk(reconstruction, topdown=False):
            for name in directories:
                (Path(current) / name).chmod(0o500)
        reconstruction.chmod(0o500)
        _write(
            private / case / "official-target-manifest.json",
            b"opaque protected target metadata\x00",
        )
        _write(
            private / case / "official-target.npz",
            b"not a numpy target archive\x00",
        )
        _write(
            query_inputs / case / "official-frame-zero-query-manifest.json",
            b"opaque protected query metadata\x00",
        )
        _write(
            query_inputs / case / "official-frame-zero-query.npz",
            b"not a numpy query archive\x00",
        )
        _write(
            query_outputs / case / "queried-prediction-seal.json",
            b"opaque protected output metadata\x00",
        )
        _write(
            query_outputs / case / "queried-prediction.npz",
            b"not a numpy prediction archive\x00",
        )


def _make_failed_reconstruction(module, private: Path) -> None:
    episode = (
        private
        / module.FAILED_CASE
        / "fresh-official-reconstruction/staged-aligned/episode_0000"
    )
    episode.mkdir(parents=True)
    for camera in module.FAILED_CAMERAS:
        camera_root = episode / camera
        for name in (
            "aligned_timestamps.txt",
            "mask_refined.h5",
            "metadata.json",
            "undistorted.mp4",
        ):
            _write(camera_root / name, b"opaque camera payload\x00")
    _write(episode / "extrinsics.npy", b"opaque extrinsics\x00")
    _write(episode / "undistorted_intrinsics.npy", b"opaque intrinsics\x00")
    _write(episode / "robot/robot.meta.json", b"opaque robot metadata\x00")
    _write(episode / "robot/robot.npz", b"opaque robot archive\x00")
    splat = episode / "splatfacto"
    for frame in range(81):
        _write(splat / f"splat_{frame}.ply", b"opaque point cloud\x00")
    timestamp = (
        splat
        / ".scratch_000080/outputs/splat_80/splatfacto"
        / module.FAILED_SCRATCH_TIMESTAMP
    )
    _write(timestamp / "config.yml", b"opaque config\x00")
    _write(timestamp / "dataparser_transforms.json", b"opaque transforms\x00")
    _write(
        timestamp / "nerfstudio_models/step-000000249.ckpt",
        b"opaque checkpoint\x00",
    )
    (private / module.FAILED_CASE / "fresh-official-reconstruction").chmod(0o700)


def _make_launcher(module, monkeypatch: pytest.MonkeyPatch, launcher: Path) -> None:
    launcher.mkdir(parents=True)
    launcher.chmod(0o700)
    lines: list[bytes] = []
    for marker, count in module.LOG_MARKERS.values():
        lines.extend(marker for _ in range(count))
    payload = b"attempt-4 fixture launcher\n" + b"\n".join(lines) + b"\n"
    log = _write(launcher / "output.log", payload)
    exit_path = _write(launcher / "exit.code", module.EXPECTED_EXIT_BYTES)
    monkeypatch.setattr(module, "EXPECTED_LAUNCHER_LOG_SIZE", len(payload))
    monkeypatch.setattr(
        module, "EXPECTED_LAUNCHER_LOG_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(module, "EXPECTED_EXIT_SIZE", exit_path.stat().st_size)
    monkeypatch.setattr(
        module,
        "EXPECTED_EXIT_SHA256",
        hashlib.sha256(exit_path.read_bytes()).hexdigest(),
    )
    assert log.stat().st_size == len(payload)


def _configure_fixture(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base = tmp_path / "formal"
    active = base / "held-v8"
    archive = base / "held-v8-attempt-4-withdrawn-postbarrier"
    pointer = base / "held-v8-attempt-4-withdrawal-pointer.json"
    completion = base / "held-v8-attempt-4-withdrawal-integrity-completion.json"
    launcher = tmp_path / "orchestration" / "attempt4"
    active.mkdir(parents=True)
    active.chmod(0o700)

    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "ACTIVE", active)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "POINTER", pointer)
    monkeypatch.setattr(module, "COMPLETION", completion)
    monkeypatch.setattr(module, "LAUNCHER", launcher)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(module, "_running_formal_processes", lambda: [])
    monkeypatch.setattr(module, "FAILED_CAMERAS", ("camera0",))
    monkeypatch.setattr(module, "FAILED_SCRATCH_TIMESTAMP", "fixture-timestamp")
    monkeypatch.setattr(
        module,
        "COMPLETED_CAMERAS",
        {case: ("completed-camera0",) for case in module.COMPLETED_OUTCOME_CASES},
    )
    monkeypatch.setattr(module, "COMPLETED_PCD_FRAME_COUNT", 2)
    monkeypatch.setattr(module, "COMPLETED_SPLAT_FRAME_COUNT", 2)

    binding = _make_code(module, monkeypatch, active)
    calibration = active / "calibration"
    _make_preoutcome_cases(module, active)
    for name, mode in (
        (".shard-0.claim", 0o700),
        (".shard-1.claim", 0o700),
        (".v8-outcome-phase.claim", 0o500),
    ):
        path = calibration / name
        path.mkdir()
        path.chmod(mode)
    (calibration / "logs").mkdir()
    _write(calibration / "shard-0.lock-verify.log", b"verified\n")
    _write(calibration / "shard-1.lock-verify.log", b"verified\n")
    private, query_inputs, query_outputs = _make_output_roots(module, calibration)
    _make_completed_pairs(module, private, query_inputs, query_outputs)
    staging_expectations = {}
    for case in module.COMPLETED_OUTCOME_CASES:
        staged = private / case / "fresh-official-reconstruction/staged-aligned"
        descendants = list(staged.rglob("*"))
        staging_expectations[case] = {
            "directory_count": sum(path.is_dir() for path in descendants),
            "regular_file_count": sum(path.is_file() for path in descendants),
            "regular_file_bytes": sum(
                path.stat().st_size for path in descendants if path.is_file()
            ),
        }
    monkeypatch.setattr(module, "COMPLETED_STAGING_EXPECTATIONS", staging_expectations)
    _make_failed_reconstruction(module, private)

    disclosure = active / "post-withdrawal-development-use-disclosure.json"
    _write(disclosure, b"opaque disclosure\n")
    replacement = active / "replacement-source"
    replacement.mkdir()
    _write(replacement / "source.json", b"opaque replacement-source evidence\n")

    lock = module._signed(
        {
            "schema_version": 1,
            "artifact_kind": "fixture lock",
            "protocol_id": module.PROTOCOL_ID,
            "stage": "calibration",
            "execution_attempt": module.EXECUTION_ATTEMPT,
            "held_root": str(active),
            "calibration_case_whitelist": list(module.EXPECTED_CASES),
            "immutable_bindings": {
                "method_deployed_snapshot_tree": binding["git_tree_manifest_sha256"],
                "method_head_text_sha256": binding["head_text_sha256"],
            },
        }
    )
    lock_payload = module._pretty_json_bytes(lock)
    _write(active / "calibration-lock.json", lock_payload)
    monkeypatch.setattr(
        module,
        "EXPECTED_LOCK_FILE_SHA256",
        hashlib.sha256(lock_payload).hexdigest(),
    )
    monkeypatch.setattr(
        module, "EXPECTED_LOCK_ARTIFACT_SHA256", lock["artifact_sha256"]
    )
    _make_launcher(module, monkeypatch, launcher)
    return active, archive, pointer, completion, launcher


@pytest.fixture
def fixture_root():
    temporary = tempfile.TemporaryDirectory(
        prefix="bpt-v81-attempt4-withdrawal-test-", dir="/tmp"
    )
    root = Path(temporary.name)
    yield root
    for child in root.iterdir():
        _make_writable(child)
    shutil.rmtree(root, ignore_errors=True)
    temporary.cleanup()


def test_attempt4_operator_archives_without_deserializing_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    active, archive, pointer, completion, launcher = _configure_fixture(
        module, monkeypatch, fixture_root
    )
    parsed_paths: list[Path] = []
    original_reader = module._read_metadata_json

    def guarded_reader(path: Path, *, role: str):
        parsed_paths.append(path)
        assert path.name in {
            "calibration-lock.json",
            module.REPORT_NAME,
            pointer.name,
            completion.name,
        }
        return original_reader(path, role=role)

    monkeypatch.setattr(module, "_read_metadata_json", guarded_reader)
    assert module.main() == 0
    first_stdout = json.loads(capsys.readouterr().out)
    assert not active.exists()
    assert archive.is_dir()
    assert pointer.is_file()
    assert completion.is_file()
    assert first_stdout["independent_post_rename_integrity_verified"] is True

    report_path = archive / module.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pointer_value = json.loads(pointer.read_text(encoding="utf-8"))
    completion_value = json.loads(completion.read_text(encoding="utf-8"))
    assert report["artifact_sha256"] == module._artifact_sha256(report)
    assert pointer_value["artifact_sha256"] == module._artifact_sha256(pointer_value)
    assert completion_value["artifact_sha256"] == module._artifact_sha256(
        completion_value
    )
    assert report["execution_boundary"]["official_target_archive_count"] == 2
    assert report["execution_boundary"]["queried_prediction_seal_count"] == 2
    assert report["execution_boundary"]["partial_reconstruction_count"] == 1
    assert report["execution_boundary"]["failed_case"] == module.FAILED_CASE
    assert report["execution_boundary"]["failed_case_evidence"] == {
        "staged_camera_count": len(module.FAILED_CAMERAS),
        "splatfacto_ply_count": 81,
        "partial_final_frame_scratch_present": True,
        "final_frame_checkpoint_present": True,
        "official_reconstruction_audit_present": False,
        "official_target_archive_present": False,
        "official_target_manifest_present": False,
    }
    assert "official_target_archive_present" not in report["execution_boundary"]
    assert "official_target_manifest_present" not in report["execution_boundary"]
    assert "official_reconstruction_audit_present" not in report["execution_boundary"]
    assert report["execution_boundary"]["score_evidence_count"] == 0
    assert (
        report["stable_noncode_inventory"]["payload_deserialization_performed"] is False
    )
    assert (
        report["information_boundary"]["second_complete_cohort_barrier_crossed"]
        is False
    )
    assert (
        pointer_value["withdrawal_report_file_sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    assert pointer_value["withdrawal_integrity_completion"]["sha256"] == (
        hashlib.sha256(completion.read_bytes()).hexdigest()
    )
    assert stat.S_IMODE(archive.stat().st_mode) == 0o500
    assert stat.S_IMODE(launcher.stat().st_mode) == 0o500
    assert not any(path.stat().st_mode & 0o222 for path in archive.rglob("*"))

    assert module.main() == 0
    assert json.loads(capsys.readouterr().out) == first_stdout
    assert any(path.name == "calibration-lock.json" for path in parsed_paths)
    assert not any("official-target" in str(path) for path in parsed_paths)
    assert not any("official-frame-zero-query" in str(path) for path in parsed_paths)
    assert not any("queried-prediction" in str(path) for path in parsed_paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_seal", "online_prediction_seal is incomplete"),
        ("later_private", "later private target exists"),
        ("score", "calibration root inventory changed"),
        ("barrier2", "launcher log identity or terminal markers changed"),
    ],
)
def test_attempt4_operator_rejects_wrong_boundary_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
    mutation: str,
    message: str,
) -> None:
    module = _module()
    active, archive, pointer, completion, launcher = _configure_fixture(
        module, monkeypatch, fixture_root
    )
    if mutation == "missing_seal":
        path = (
            active
            / "calibration/cases/002-rope-silk-ep0004/online/online_prediction_seal.json"
        )
        path.chmod(0o600)
        path.unlink()
    elif mutation == "later_private":
        _write(
            active
            / "calibration/private-targets/083-blanket-cloth-ep0000/unexpected.bin",
            b"forbidden\n",
        )
    elif mutation == "score":
        _write(active / "calibration/calibration-score-evidence.json", b"forbidden\n")
    else:
        log = launcher / "output.log"
        payload = log.read_bytes() + b'\n{"event": "SECOND_COHORT_BARRIER_VALIDATED"}\n'
        _write(log, payload)
        monkeypatch.setattr(module, "EXPECTED_LAUNCHER_LOG_SIZE", len(payload))
        monkeypatch.setattr(
            module,
            "EXPECTED_LAUNCHER_LOG_SHA256",
            hashlib.sha256(payload).hexdigest(),
        )

    with pytest.raises(RuntimeError, match=message):
        module.main()
    assert active.is_dir()
    assert not archive.exists()
    assert not pointer.exists()
    assert not completion.exists()
    assert not (active / module.REPORT_NAME).exists()


def test_attempt4_operator_rejects_hardlinked_payload(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
) -> None:
    module = _module()
    active, archive, pointer, completion, _launcher = _configure_fixture(
        module, monkeypatch, fixture_root
    )
    payload = (
        active
        / "calibration/private-targets/072-cotton-clohesline-ep0003/official-target.npz"
    )
    outside = fixture_root / "linked-target.npz"
    os.link(payload, outside)

    with pytest.raises(RuntimeError, match="regular single-link file required"):
        module.main()
    assert active.is_dir()
    assert not archive.exists()
    assert not pointer.exists()
    assert not completion.exists()


def test_attempt4_operator_recovers_after_atomic_rename(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
) -> None:
    module = _module()
    active, archive, pointer, completion, _launcher = _configure_fixture(
        module, monkeypatch, fixture_root
    )
    report = module._prepare_active_report()
    os.rename(active, archive)
    assert not active.exists() and not pointer.exists() and not completion.exists()

    assert module.main() == 0
    observed_report = module._validate_signed_metadata(
        archive / module.REPORT_NAME, role="test report"
    )
    observed_pointer = module._validate_signed_metadata(pointer, role="test pointer")
    observed_completion = module._validate_signed_metadata(
        completion, role="test completion"
    )
    assert observed_report == report
    assert (
        observed_pointer["withdrawal_report_artifact_sha256"]
        == report["artifact_sha256"]
    )
    assert (
        observed_pointer["withdrawal_integrity_completion"]["artifact_sha256"]
        == observed_completion["artifact_sha256"]
    )
    assert observed_pointer["independent_post_rename_integrity_verified"] is True
    assert (
        observed_completion["pointer_contract"]["pointer_must_bind_this_completion"]
        is True
    )


@pytest.mark.parametrize("artifact", ["report", "completion", "pointer"])
@pytest.mark.parametrize(
    "interruption", ["partial_pending", "complete_pending", "linked_pending"]
)
def test_attempt4_operator_recovers_interrupted_artifact_publication(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
    artifact: str,
    interruption: str,
) -> None:
    module = _module()
    active, archive, pointer, completion, _launcher = _configure_fixture(
        module, monkeypatch, fixture_root
    )
    targets = {
        "report": active / module.REPORT_NAME,
        "completion": completion,
        "pointer": pointer,
    }
    target = targets[artifact]
    original_writer = module._exclusive_json
    interrupted = False

    def interrupting_writer(path: Path, value: dict):
        nonlocal interrupted
        if path == target and not interrupted:
            interrupted = True
            pending = module._pending_json_path(path)
            assert pending.parent == module.BASE
            assert module.ACTIVE not in pending.parents
            if interruption == "partial_pending":
                _write(pending, b'{"partial":')
            else:
                _write(pending, module._pretty_json_bytes(value))
                if interruption == "linked_pending":
                    os.link(pending, path, follow_symlinks=False)
            raise RuntimeError("simulated abrupt interruption")
        return original_writer(path, value)

    monkeypatch.setattr(module, "_exclusive_json", interrupting_writer)
    with pytest.raises(RuntimeError, match="simulated abrupt interruption"):
        module.main()
    assert interrupted is True

    monkeypatch.setattr(module, "_exclusive_json", original_writer)
    assert module.main() == 0
    final_report = archive / module.REPORT_NAME
    for final in (final_report, completion, pointer):
        assert final.is_file()
        assert stat.S_IMODE(final.stat().st_mode) == 0o400
        assert not os.path.lexists(module._pending_json_path(final))
    assert not active.exists()


def test_attempt4_operator_never_overwrites_unexpected_final_json(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "BASE", fixture_root)
    monkeypatch.setattr(module, "ACTIVE", fixture_root / "held-v8")
    path = fixture_root / "final.json"
    original = b"unexpected immutable final\n"
    _write(path, original)

    with pytest.raises(RuntimeError, match="existing final JSON changed"):
        module._exclusive_json(path, module._signed({"value": "expected"}))
    assert path.read_bytes() == original
    assert not os.path.lexists(module._pending_json_path(path))


def test_attempt4_operator_loses_publication_race_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "BASE", fixture_root)
    monkeypatch.setattr(module, "ACTIVE", fixture_root / "held-v8")
    path = fixture_root / "final.json"
    unexpected = b"concurrent unexpected final\n"
    original_link = module.os.link

    def racing_link(source: Path, destination: Path, *, follow_symlinks: bool):
        _write(destination, unexpected)
        return original_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        module._exclusive_json(path, module._signed({"value": "expected"}))
    assert path.read_bytes() == unexpected
    pending = module._pending_json_path(path)
    assert pending.is_file()
    assert pending.stat().st_nlink == 1


def test_attempt4_operator_rejects_launcher_tamper(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
) -> None:
    module = _module()
    active, archive, pointer, completion, launcher = _configure_fixture(
        module, monkeypatch, fixture_root
    )
    log = launcher / "output.log"
    _write(log, log.read_bytes() + b"tamper\n")

    with pytest.raises(RuntimeError, match="launcher log identity"):
        module.main()
    assert active.is_dir()
    assert not archive.exists()
    assert not pointer.exists()
    assert not completion.exists()


def test_attempt4_source_has_no_payload_deserializer() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    for forbidden in (
        "np.load(",
        "numpy",
        "pickle.load(",
        "torch.load(",
        "h5py",
        "cv2",
        "PIL",
        "imageio",
        "VideoCapture",
    ):
        assert forbidden not in source
    assert "O_NOFOLLOW" in source
    assert "os.rename(ACTIVE, ARCHIVE)" in source


def test_attempt4_pins_exact_64_hex_lock_artifact_sha256() -> None:
    module = _module()
    expected = "da8ff292b80e64b1d235af3adcc11c5fd31e0b5109827b6cc7cc68113c954437"
    assert module.EXPECTED_LOCK_ARTIFACT_SHA256 == expected
    assert len(module.EXPECTED_LOCK_ARTIFACT_SHA256) == 64
    assert set(module.EXPECTED_LOCK_ARTIFACT_SHA256) <= set("0123456789abcdef")
