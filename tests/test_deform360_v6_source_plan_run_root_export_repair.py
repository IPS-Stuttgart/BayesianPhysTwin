from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "source_plan_run_root_export.json"
)
ACTIVE_RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
ARCHIVED_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
LEGACY_WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence.yml"
)


def test_source_plan_run_root_export_repair_is_content_addressed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"] == {
        "artifact_digest": (
            "sha256:e5dda15cbff1850a92c8b6ab68d8257ca29b1b18fccc6198e974cd8257563bce"
        ),
        "artifact_id": 9130158355,
        "execution_receipt_id": (
            "c9ea0e47203b0d4880a6937668f9ed81c2578bc1a6d566ec177634d1ee43a04f"
        ),
        "exit_code": 1,
        "physical_manifest_count": 10,
        "source_plan_inputs_byte_count": 95838,
        "source_plan_inputs_sha256": (
            "5203e4baa5d3e51f490628c43949ab0f440a47ba9d23378dc90000e8c12f4961"
        ),
        "source_plan_materialized": False,
        "source_prediction_seal_count": 0,
        "source_revision": "e41b39b56388a41627657e5c4a7aee6dde991fd3",
        "terminal_stage": "materialize-source-plan",
        "workflow_run_attempt": 1,
        "workflow_run_id": 31566855876,
    }
    assert payload["repair_scope"]["source_plan_run_root_export_completed"]
    assert payload["repair_scope"]["duplicate_empirical_execution_prevented"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key
        not in {
            "duplicate_empirical_execution_prevented",
            "source_plan_run_root_export_completed",
        }
    )


def test_archived_extractor_requires_exported_run_root() -> None:
    archived = ARCHIVED_RUNNER.read_text(encoding="utf-8")

    assignment = 'RUN_ROOT="${RESULTS_ROOT}/bayesian-phystwin/'
    extractor = 'root = Path(os.environ["RUN_ROOT"])'
    assert archived.count(assignment) == 1
    assert archived.count(extractor) == 2
    assert "export RUN_ROOT PHYSICAL_WORK_ROOT" in archived
    assert archived.index(assignment) < archived.index(extractor)
    stage = archived.index('set_stage "materialize-source-plan"')
    main_extractor = archived.index(extractor, stage)
    assert "export RUN_ROOT" not in archived[stage:main_extractor]


def test_export_seed_survives_nested_shell_assignment(tmp_path: Path) -> None:
    inner = tmp_path / "inner.sh"
    middle = tmp_path / "middle.sh"
    output = tmp_path / "observed.txt"
    inner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RUN_ROOT="${RESULTS_ROOT}/frozen-run"\n'
        "\"${PYTHON_BIN}\" - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['OUTPUT']).write_text(\n"
        "    os.environ['RUN_ROOT'], encoding='utf-8'\n"
        ")\n"
        "PY\n",
        encoding="utf-8",
    )
    middle.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nbash "${INNER}"\n',
        encoding="utf-8",
    )
    inner.chmod(0o700)
    middle.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "INNER": str(inner),
            "OUTPUT": str(output),
            "PYTHON_BIN": sys.executable,
            "RESULTS_ROOT": str(tmp_path / "results"),
            "RUN_ROOT": "",
        }
    )

    completed = subprocess.run(
        ["bash", str(middle)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text(encoding="utf-8") == str(
        tmp_path / "results" / "frozen-run"
    )


def test_active_launcher_and_workflow_bind_and_attest_repair() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    amendment_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    launcher = ACTIVE_RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'RUN_ROOT="" \\\n' in launcher
    assert launcher.count('RUN_ROOT="" \\\n') == 1
    assert f"SOURCE_PLAN_RUN_ROOT_EXPORT_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"SOURCE_PLAN_RUN_ROOT_EXPORT_REPAIR_SHA256: {amendment_digest}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert '"runtime_source_plan_run_root_export_repair"' in workflow
    assert 'expected_run_root_export["activated"] = False' in workflow
    assert "source-plan run-root export receipt field changed" in workflow


def test_dual_runtime_workflow_is_the_only_empirical_executor() -> None:
    import yaml

    dual = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    legacy = yaml.safe_load(LEGACY_WORKFLOW.read_text(encoding="utf-8"))

    dual_condition = str(dual["jobs"]["evidence"]["if"])
    legacy_condition = str(legacy["jobs"]["evidence"]["if"])
    assert "github.event_name == 'push'" in dual_condition
    assert not dual_condition.lstrip().startswith("false &&")
    assert legacy_condition.lstrip().startswith("false &&")
    assert "github.event_name == 'push'" in legacy_condition
    assert "contracts" in legacy["jobs"]
