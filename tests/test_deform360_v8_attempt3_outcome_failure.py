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
OPERATOR = (
    ROOT
    / "scripts"
    / "held"
    / "seal_deform360_v8_attempt3_outcome_failure.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("v8_attempt3_withdrawal", OPERATOR)
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
    if root.is_file():
        os.chmod(root, 0o600, follow_symlinks=False)
        return
    os.chmod(root, 0o700, follow_symlinks=False)
    for current, directories, files in os.walk(root):
        for name in directories:
            os.chmod(Path(current) / name, 0o700, follow_symlinks=False)
        for name in files:
            os.chmod(Path(current) / name, 0o600, follow_symlinks=False)


def _configure_fixture(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base = tmp_path / "formal"
    active = base / "held-v8"
    archive = base / "held-v8-attempt-3-withdrawn-postbarrier"
    pointer = base / "held-v8-attempt-3-withdrawal-pointer.json"
    active.mkdir(parents=True)
    active.chmod(0o700)

    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "ACTIVE", active)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "POINTER", pointer)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(module, "_running_formal_processes", lambda: [])
    monkeypatch.setattr(module, "STAGED_CAMERAS", ("camera0",))
    monkeypatch.setattr(module, "PCD_CLEAN_FRAME_COUNT", 2)
    monkeypatch.setattr(module, "SPLATFACTO_FRAME_COUNT", 2)

    temporary_code = active / "code-staging"
    temporary_code.mkdir()
    _git(temporary_code, "init", "--quiet")
    _git(temporary_code, "config", "user.name", "Held Test")
    _git(temporary_code, "config", "user.email", "held@example.invalid")
    (temporary_code / "method.txt").write_text("frozen method\n", encoding="utf-8")
    _git(temporary_code, "add", "method.txt")
    _git(temporary_code, "commit", "--quiet", "-m", "fixture")
    head = _git(temporary_code, "rev-parse", "HEAD")
    code = active / f"code-{head}"
    temporary_code.rename(code)
    binding = module._repository_binding(code)
    monkeypatch.setattr(module, "EXPECTED_DEPLOYED_HEAD", head)
    monkeypatch.setattr(
        module,
        "EXPECTED_DEPLOYED_TREE_SHA256",
        binding["git_tree_manifest_sha256"],
    )

    cases_root = active / "calibration" / "cases"
    for case in module.EXPECTED_CASES:
        case_root = cases_root / case
        for relative in (
            "frame-zero/frame_zero_bundle.manifest.json",
            "physical/physical_prior_seal.json",
            "prefix-authorization.json",
            "online/online_prediction_seal.json",
            "frozen-field/preoutcome-frozen-field-manifest.json",
        ):
            _write(case_root / relative, b"sealed but deliberately not JSON\n")

    calibration = active / "calibration"
    claim = calibration / ".v8-outcome-phase.claim"
    claim.mkdir()
    claim.chmod(0o500)
    private = calibration / "private-targets"
    query_inputs = calibration / "query-inputs"
    query_outputs = calibration / "query-outputs"
    for output_root in (private, query_inputs, query_outputs):
        output_root.mkdir()
        output_root.chmod(0o700)
        for case in module.EXPECTED_CASES:
            (output_root / case).mkdir()
            (output_root / case).chmod(0o700)

    directories, files = module._expected_reconstruction_paths()
    for relative in sorted(directories, key=lambda value: value.count("/")):
        path = active / relative
        path.mkdir(exist_ok=True)
    for relative in files:
        _write(active / relative, b"opaque reconstruction payload\x00" + relative.encode())

    formal_payloads = {
        (
            "calibration/private-targets/072-cotton-clohesline-ep0003/"
            "fresh-official-reconstruction/held-v8-official-reconstruction-audit.json"
        ): b"not-json reconstruction audit metadata\x00",
        (
            "calibration/private-targets/072-cotton-clohesline-ep0003/"
            "official-target-manifest.json"
        ): b"not-json protected target metadata\x00",
        (
            "calibration/query-inputs/072-cotton-clohesline-ep0003/"
            "official-frame-zero-query-manifest.json"
        ): b"not-json protected query metadata\x00",
    }
    for relative, payload in formal_payloads.items():
        _write(active / relative, payload)
    expected_metadata = {
        relative: (len(payload), hashlib.sha256(payload).hexdigest())
        for relative, payload in formal_payloads.items()
    }
    monkeypatch.setattr(module, "EXPECTED_FORMAL_METADATA", expected_metadata)

    array_payloads = {
        (
            "calibration/private-targets/072-cotton-clohesline-ep0003/"
            "official-target.npz"
        ): b"not a numpy target archive\x00",
        (
            "calibration/query-inputs/072-cotton-clohesline-ep0003/"
            "official-frame-zero-query.npz"
        ): b"not a numpy query archive\x00",
    }
    for relative, payload in array_payloads.items():
        _write(active / relative, payload)
    monkeypatch.setattr(
        module,
        "EXPECTED_FORMAL_ARRAY_SIZES",
        {relative: len(payload) for relative, payload in array_payloads.items()},
    )
    for relative in sorted(
        directories, key=lambda value: value.count("/"), reverse=True
    ):
        (active / relative).chmod(0o500)

    lock = module._signed(
        {
            "schema_version": 1,
            "artifact_kind": "fixture lock",
            "protocol_id": module.PROTOCOL_ID,
            "stage": "calibration",
            "execution_attempt": 3,
            "held_root": str(active),
            "calibration_case_whitelist": list(module.EXPECTED_CASES),
            "immutable_bindings": {
                "method_deployed_snapshot_tree": binding[
                    "git_tree_manifest_sha256"
                ],
                "method_head_text_sha256": binding["head_text_sha256"],
            },
        }
    )
    lock_path = active / "calibration-lock.json"
    lock_payload = module._pretty_json_bytes(lock)
    _write(lock_path, lock_payload)
    monkeypatch.setattr(
        module,
        "EXPECTED_LOCK_FILE_SHA256",
        hashlib.sha256(lock_payload).hexdigest(),
    )
    monkeypatch.setattr(
        module, "EXPECTED_LOCK_ARTIFACT_SHA256", lock["artifact_sha256"]
    )

    for current, directories_in_code, files_in_code in os.walk(code, topdown=False):
        for name in files_in_code:
            os.chmod(Path(current) / name, 0o444, follow_symlinks=False)
        for name in directories_in_code:
            os.chmod(Path(current) / name, 0o555, follow_symlinks=False)
    code.chmod(0o555)
    return active, archive, pointer


@pytest.fixture
def fixture_root():
    temporary = tempfile.TemporaryDirectory(
        prefix="bpt-v8-attempt3-withdrawal-test-", dir="/tmp"
    )
    root = Path(temporary.name)
    yield root
    for child in root.iterdir():
        _make_writable(child)
    shutil.rmtree(root, ignore_errors=True)
    temporary.cleanup()


def test_attempt3_operator_archives_without_deserializing_payloads_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    active, archive, pointer = _configure_fixture(module, monkeypatch, fixture_root)
    parsed_paths: list[Path] = []
    original_reader = module._read_metadata_json

    def guarded_reader(path: Path, *, role: str):
        parsed_paths.append(path)
        assert path.name in {
            "calibration-lock.json",
            module.REPORT_NAME,
            pointer.name,
        }
        return original_reader(path, role=role)

    monkeypatch.setattr(module, "_read_metadata_json", guarded_reader)
    assert module.main() == 0
    first_stdout = json.loads(capsys.readouterr().out)
    assert not active.exists()
    assert archive.is_dir()
    assert pointer.is_file()
    assert first_stdout["independent_post_rename_integrity_verified"] is True

    report_path = archive / module.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pointer_value = json.loads(pointer.read_text(encoding="utf-8"))
    assert report["artifact_sha256"] == module._artifact_sha256(report)
    assert pointer_value["artifact_sha256"] == module._artifact_sha256(pointer_value)
    assert pointer_value["withdrawal_report_file_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert report["disposition"] == module.DISPOSITION
    assert report["terminal_failure"]["outer_outcome_driver_exit_code"] == 2
    assert report["terminal_failure"]["inner_x0_worker_exit_code"] == 1
    assert report["terminal_failure"]["inner_exception_message"] == (
        "an assimilation center has no query identity within the exclusion radius"
    )
    assert report["execution_boundary"]["queried_prediction_seal_count"] == 0
    assert report["execution_boundary"]["score_evidence_count"] == 0
    assert report["stable_noncode_inventory"]["payload_deserialization_performed"] is False
    assert stat.S_IMODE(archive.stat().st_mode) == 0o500
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(pointer.stat().st_mode) == 0o400
    assert not any(path.stat().st_mode & 0o222 for path in archive.rglob("*"))

    assert module.main() == 0
    second_stdout = json.loads(capsys.readouterr().out)
    assert second_stdout == first_stdout
    assert any(path.name == "calibration-lock.json" for path in parsed_paths)
    assert not any("official-target" in str(path) for path in parsed_paths)
    assert not any("official-frame-zero-query" in str(path) for path in parsed_paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_seal", "online_prediction_seal is incomplete"),
        ("queried_output", "queried prediction exists"),
        ("score", "score, decision, or confirmation evidence exists"),
    ],
)
def test_attempt3_operator_rejects_wrong_counts_or_paths_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
    mutation: str,
    message: str,
) -> None:
    module = _module()
    active, archive, pointer = _configure_fixture(module, monkeypatch, fixture_root)
    if mutation == "missing_seal":
        path = (
            active
            / "calibration/cases/002-rope-silk-ep0004/online/online_prediction_seal.json"
        )
        path.chmod(0o600)
        path.unlink()
    elif mutation == "queried_output":
        _write(
            active
            / "calibration/query-outputs/072-cotton-clohesline-ep0003/queried-prediction-seal.json",
            b"forbidden\n",
        )
    else:
        _write(active / "calibration/calibration-score-evidence.json", b"forbidden\n")

    with pytest.raises(RuntimeError, match=message):
        module.main()
    assert active.is_dir()
    assert not archive.exists()
    assert not pointer.exists()
    assert not (active / module.REPORT_NAME).exists()


def test_attempt3_operator_recovers_after_atomic_rename_and_verifies_hashes(
    monkeypatch: pytest.MonkeyPatch,
    fixture_root: Path,
) -> None:
    module = _module()
    active, archive, pointer = _configure_fixture(module, monkeypatch, fixture_root)
    report = module._prepare_active_report()
    os.rename(active, archive)
    assert not active.exists() and not pointer.exists()

    assert module.main() == 0
    report_path = archive / module.REPORT_NAME
    observed_report = module._validate_signed_metadata(
        report_path, role="test report"
    )
    observed_pointer = module._validate_signed_metadata(pointer, role="test pointer")
    assert observed_report == report
    assert observed_pointer["withdrawal_report_artifact_sha256"] == report[
        "artifact_sha256"
    ]
    assert observed_pointer["withdrawal_report_file_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert observed_pointer["independent_post_rename_integrity_verified"] is True
    assert observed_pointer["archive_fully_nonwritable"] is True


def test_attempt3_source_has_no_payload_deserializer() -> None:
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
