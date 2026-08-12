from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "scripts/ci/dispatch_deform360_v6_source_python.sh"


def _environment(tmp_path: Path, receipt: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "BPT_PRIMARY_PYTHON": sys.executable,
            "BPT_FRAME_ZERO_PYTHON": sys.executable,
            "BPT_FRAME_ZERO_RUNTIME_MARKER": str(tmp_path / "frame-zero.json"),
            "BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER": str(
                tmp_path / "fallback-config.json"
            ),
            "BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER": str(
                tmp_path / "official-runtime.json"
            ),
            "BPT_CASE_STDIN_ISOLATION_MARKER": str(tmp_path / "stdin.json"),
            "RECEIPT_PATH": str(receipt),
            "RUNNER_TEMP": str(tmp_path),
        }
    )
    return environment


def _receipt_id(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("receipt_id", None)
    canonical = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_dispatcher_scrubs_unobserved_legacy_receipt_probes(tmp_path: Path) -> None:
    receipt = tmp_path / "execution-receipt.json"
    program = """
import json
import os
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
receipt = {"status": "source-prediction-evidence-sealed"}
receipt["runtime_cuda_host_compiler_repair"] = {
    "probe_passed": os.environ["CUDA_HOST_COMPILER_PROBE_PASSED"] == "true",
    "version": os.environ["CUDA_HOST_COMPILER_VERSION"],
}
receipt["runtime_ninja_build_tool_repair"] = {
    "probe_passed": os.environ["NINJA_PYTORCH_PROBE_PASSED"] == "true",
    "version": os.environ["NINJA_DISTRIBUTION_VERSION"],
}
receipt["receipt_id"] = "stale"
path.write_text(json.dumps(receipt) + "\\n", encoding="utf-8")
"""

    completed = subprocess.run(
        ["bash", str(DISPATCHER), "-"],
        input=program,
        capture_output=True,
        check=False,
        env=_environment(tmp_path, receipt),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "source-prediction-evidence-sealed"
    assert "runtime_cuda_host_compiler_repair" not in payload
    assert "runtime_ninja_build_tool_repair" not in payload
    assert payload["receipt_id"] == _receipt_id(payload)


def test_dispatcher_does_not_inject_legacy_probes_into_other_stdin(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "unused.json"
    program = (
        "import os; print(os.environ.get('CUDA_HOST_COMPILER_PROBE_PASSED', 'missing'))"
    )

    completed = subprocess.run(
        ["bash", str(DISPATCHER), "-"],
        input=program,
        capture_output=True,
        check=False,
        env=_environment(tmp_path, receipt),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "missing"
    assert not receipt.exists()
