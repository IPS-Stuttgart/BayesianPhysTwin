from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "scripts/ci/dispatch_deform360_v6_source_python.sh"
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "source_plan_run_root_environment.json"
)
REPAIR_ID = "2637a1d95f0af46461a44e5f62e3264d07f75531c1fe490d8ad78a5d4404817d"
AMENDMENT_ID = "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
SOURCE_SHA = "e41b39b56388a41627657e5c4a7aee6dde991fd3"
MARKER = {
    "binding": "derived-exported-run-root",
    "repair_id": REPAIR_ID,
    "source_plan_inputs_present": True,
}


def _fake_python(tmp_path: Path) -> Path:
    path = tmp_path / "capture-python"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(\n"
        "    json.dumps({\n"
        "        'arguments': sys.argv[1:],\n"
        "        'run_root': os.environ.get('RUN_ROOT'),\n"
        "        'stdin': sys.stdin.read(),\n"
        "    }, sort_keys=True),\n"
        "    encoding='utf-8',\n"
        ")\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    runtime = _fake_python(tmp_path)
    results = tmp_path / "results"
    evidence = tmp_path / "evidence"
    capture = tmp_path / "capture.json"
    environment = os.environ.copy()
    environment.update(
        {
            "AMENDMENT_ID": AMENDMENT_ID,
            "BPT_CASE_STDIN_ISOLATION_MARKER": str(tmp_path / "case-marker.json"),
            "BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER": str(
                tmp_path / "fallback-marker.json"
            ),
            "BPT_FRAME_ZERO_PYTHON": str(runtime),
            "BPT_FRAME_ZERO_RUNTIME_MARKER": str(tmp_path / "frame-marker.json"),
            "BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER": str(
                tmp_path / "official-marker.json"
            ),
            "BPT_PRIMARY_PYTHON": str(runtime),
            "BPT_SOURCE_SHA": SOURCE_SHA,
            "CAPTURE_PATH": str(capture),
            "EVIDENCE_ROOT": str(evidence),
            "RESULTS_ROOT": str(results),
        }
    )
    run_root = (
        results
        / "bayesian-phystwin/deform360-v6-source-prediction"
        / AMENDMENT_ID
        / SOURCE_SHA
    )
    marker = (
        evidence
        / "deform360-v6-source-prediction-evidence"
        / "source-plan-run-root-environment-repair.json"
    )
    return environment, run_root, marker, capture


def _run(
    environment: dict[str, str],
    *,
    stdin: str = "print('source-plan')\n",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(DISPATCHER), "-"],
        check=False,
        capture_output=True,
        env=environment,
        input=stdin,
        text=True,
    )


def test_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert declared == REPAIR_ID == hashlib.sha256(canonical).hexdigest()
    failure = payload["failed_execution"]
    assert failure["workflow_run_id"] == 31566855876
    assert failure["artifact_id"] == 9130158355
    assert failure["execution_receipt_id"] == (
        "c9ea0e47203b0d4880a6937668f9ed81c2578bc1a6d566ec177634d1ee43a04f"
    )
    assert failure["terminal_stage"] == "materialize-source-plan"
    assert failure["physical_manifest_count"] == 10
    assert failure["source_prediction_seal_count"] == 0
    assert failure["source_plan_present"] is False
    assert not any(payload["information_boundary"].values())
    assert not any(payload["scientific_scope"].values())


def test_dispatcher_exports_derived_run_root_only_after_plan_inputs(
    tmp_path: Path,
) -> None:
    environment, run_root, marker, capture = _environment(tmp_path)
    run_root.mkdir(parents=True)
    (run_root / "source-plan-inputs.json").write_text("{}\n", encoding="utf-8")

    completed = _run(environment)

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(capture.read_text(encoding="utf-8"))
    assert observed == {
        "arguments": ["-"],
        "run_root": str(run_root),
        "stdin": "print('source-plan')\n",
    }
    assert json.loads(marker.read_text(encoding="utf-8")) == MARKER


def test_dispatcher_does_not_bind_before_plan_inputs(tmp_path: Path) -> None:
    environment, run_root, marker, capture = _environment(tmp_path)
    run_root.mkdir(parents=True)

    completed = _run(environment)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(capture.read_text(encoding="utf-8"))["run_root"] is None
    assert not marker.exists()


def test_dispatcher_preserves_an_existing_exported_run_root(tmp_path: Path) -> None:
    environment, run_root, marker, capture = _environment(tmp_path)
    run_root.mkdir(parents=True)
    (run_root / "source-plan-inputs.json").write_text("{}\n", encoding="utf-8")
    environment["RUN_ROOT"] = str(tmp_path / "caller-owned-run-root")

    completed = _run(environment)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(capture.read_text(encoding="utf-8"))["run_root"] == environment[
        "RUN_ROOT"
    ]
    assert not marker.exists()


def test_dispatcher_rejects_symlinked_plan_inputs(tmp_path: Path) -> None:
    environment, run_root, marker, capture = _environment(tmp_path)
    run_root.mkdir(parents=True)
    source = tmp_path / "foreign-plan-inputs.json"
    source.write_text("{}\n", encoding="utf-8")
    (run_root / "source-plan-inputs.json").symlink_to(source)

    completed = _run(environment)

    assert completed.returncode == 2
    assert "source-plan inputs path is not a real file" in completed.stderr
    assert not capture.exists()
    assert not marker.exists()


def test_dispatcher_rejects_changed_activation_marker(tmp_path: Path) -> None:
    environment, run_root, marker, capture = _environment(tmp_path)
    run_root.mkdir(parents=True)
    (run_root / "source-plan-inputs.json").write_text("{}\n", encoding="utf-8")
    marker.parent.mkdir(parents=True)
    marker.write_text('{"repair_id":"changed"}\n', encoding="utf-8")

    completed = _run(environment)

    assert completed.returncode == 2
    assert "source-plan run-root marker changed" in completed.stderr
    assert not capture.exists()


def test_dispatcher_declares_exact_repair_and_retained_source_boundary() -> None:
    text = DISPATCHER.read_text(encoding="utf-8")

    assert f'SOURCE_PLAN_RUN_ROOT_REPAIR_ID="{REPAIR_ID}"' in text
    assert '[[ "${1:-}" == "-" ]] || return 0' in text
    assert '[[ -z "${RUN_ROOT+x}" ]] || return 0' in text
    assert 'export RUN_ROOT="${computed_run_root}"' in text
    assert "source-plan-inputs.json" in text
    assert "source-plan.json" in text
    start = text.index("bind_source_plan_run_root() {")
    body = text[start : text.index("\nif [[", start)]
    assert "target" not in body.lower()
