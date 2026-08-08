from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path("scripts/ci/diagnose_deform360_visual_production_failure.py")
SPEC = importlib.util.spec_from_file_location(
    "_diagnose_deform360_visual_production_failure",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert isinstance(MODULE, ModuleType)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ADMISSION_ID = "a" * 64
IMPLEMENTATION_REVISION = "b" * 40
ATTEMPT_ID = "31277475724-1"
FAILED_RUN_ID = 31277475724
BOUNDARY = {
    "calibration_robot_state_opened": False,
    "calibration_tactile_payloads_opened": False,
    "confirmation_payloads_opened": False,
    "motioncrafter_prediction_payloads_opened": True,
    "replacement_allowed": False,
    "reserved_evaluation_frames_opened": False,
    "retained_calibration_camera_payloads_opened": True,
    "target_outcomes_used": False,
}
TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/runner/work/project/source/module.py", line 7, in run\n'
    '    raise RuntimeError("boom")\n'
    "RuntimeError: boom\n"
).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, job_count: int = 2) -> tuple[Path, str, int]:
    visual_output_root = tmp_path / "deform360" / "results" / "visual-production"
    run_root = visual_output_root / ADMISSION_ID / IMPLEMENTATION_REVISION
    stderr_sha256 = hashlib.sha256(TRACEBACK).hexdigest()
    for index in range(job_count):
        job_id = f"{index + 1:064x}"
        stderr_relative = (
            f"logs/{ATTEMPT_ID}/{job_id}.motioncrafter-production.stderr.bin"
        )
        stderr_path = run_root / stderr_relative
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_bytes(TRACEBACK)
        receipt = {
            "schema": (
                "bayesian-phystwin.deform360-calibration-visual-"
                "technical-failure"
            ),
            "schema_version": 1,
            "admission_id": ADMISSION_ID,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "attempt_id": ATTEMPT_ID,
            "job_id": job_id,
            "stage": "motioncrafter-production",
            "status": "failed",
            "completion_kind": "technical_failure",
            "return_code": 1,
            "detail_sha256": "d" * 64,
            "stderr": {
                "path": stderr_relative,
                "sha256": stderr_sha256,
                "byte_count": len(TRACEBACK),
            },
            "information_boundary": BOUNDARY,
        }
        _write_json(run_root / "failures" / f"{job_id}.json", receipt)
    return visual_output_root, stderr_sha256, len(TRACEBACK)


def _diagnose(
    tmp_path: Path,
    *,
    visual_output_root: Path,
    stderr_sha256: str,
    stderr_bytes: int,
    job_count: int = 2,
    output_name: str = "diagnostic",
) -> dict[str, object]:
    return MODULE.diagnose_failure(
        visual_output_root=visual_output_root,
        admission_id=ADMISSION_ID,
        implementation_revision=IMPLEMENTATION_REVISION,
        attempt_id=ATTEMPT_ID,
        failed_workflow_run_id=FAILED_RUN_ID,
        expected_job_count=job_count,
        expected_stderr_sha256=stderr_sha256,
        expected_stderr_bytes=stderr_bytes,
        output_dir=tmp_path / output_name,
        path_replacements={"/home/runner/work": "<GITHUB_WORKSPACE>"},
    )


def test_diagnostic_verifies_receipts_and_sanitizes_one_traceback(
    tmp_path: Path,
) -> None:
    visual_output_root, stderr_sha256, stderr_bytes = _fixture(tmp_path)

    diagnostic = _diagnose(
        tmp_path,
        visual_output_root=visual_output_root,
        stderr_sha256=stderr_sha256,
        stderr_bytes=stderr_bytes,
    )

    assert diagnostic["failure_receipt_count"] == 2
    assert diagnostic["unique_job_count"] == 2
    assert diagnostic["exception_line"] == "RuntimeError: boom"
    assert diagnostic["common_stderr_sha256"] == stderr_sha256
    assert diagnostic["predecessor_information_boundary"] == BOUNDARY
    boundary = diagnostic["diagnostic_information_boundary"]
    assert boundary["retained_technical_logs_opened"] is True
    assert boundary["retained_calibration_camera_payloads_opened_by_diagnostic"] is False
    sanitized = (tmp_path / "diagnostic" / "sanitized-traceback.txt").read_text(
        encoding="utf-8"
    )
    assert "<GITHUB_WORKSPACE>/project/source/module.py" in sanitized
    assert "/home/runner/work" not in sanitized
    restored = json.loads(
        (tmp_path / "diagnostic" / "diagnostic.json").read_text(encoding="utf-8")
    )
    assert restored == diagnostic


def test_diagnostic_rejects_information_boundary_drift(tmp_path: Path) -> None:
    visual_output_root, stderr_sha256, stderr_bytes = _fixture(tmp_path)
    receipt_path = next(
        (visual_output_root / ADMISSION_ID / IMPLEMENTATION_REVISION / "failures").glob(
            "*.json"
        )
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["information_boundary"]["confirmation_payloads_opened"] = True
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="information boundary changed"):
        _diagnose(
            tmp_path,
            visual_output_root=visual_output_root,
            stderr_sha256=stderr_sha256,
            stderr_bytes=stderr_bytes,
        )


def test_diagnostic_rejects_changed_traceback_bytes(tmp_path: Path) -> None:
    visual_output_root, stderr_sha256, stderr_bytes = _fixture(tmp_path)
    log_path = next(
        (
            visual_output_root
            / ADMISSION_ID
            / IMPLEMENTATION_REVISION
            / "logs"
            / ATTEMPT_ID
        ).glob("*.bin")
    )
    log_path.write_bytes(TRACEBACK.replace(b"boom", b"boon"))

    with pytest.raises(ValueError, match="traceback digest changed"):
        _diagnose(
            tmp_path,
            visual_output_root=visual_output_root,
            stderr_sha256=stderr_sha256,
            stderr_bytes=stderr_bytes,
        )


def test_diagnostic_rejects_retained_log_path_escape(tmp_path: Path) -> None:
    visual_output_root, stderr_sha256, stderr_bytes = _fixture(tmp_path)
    receipt_path = sorted(
        (visual_output_root / ADMISSION_ID / IMPLEMENTATION_REVISION / "failures").glob(
            "*.json"
        )
    )[0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stderr"]["path"] = "../secret.bin"
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="outside the exact attempt"):
        _diagnose(
            tmp_path,
            visual_output_root=visual_output_root,
            stderr_sha256=stderr_sha256,
            stderr_bytes=stderr_bytes,
        )
