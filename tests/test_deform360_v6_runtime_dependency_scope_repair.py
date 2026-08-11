from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_runtime_dependency_scope.json"
)
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"


def test_runtime_dependency_scope_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert payload["correction"] == {
        "bayesian_phystwin_install_target": ".[graph,vision]",
        "exact_allowlisted_pip_check_line": (
            "pyrecest 2.4.3 has requirement numpy<2.5,>=2.0, but you have numpy 1.26.4."
        ),
        "full_pip_check_expected_exit_code": 1,
        "full_pip_check_expected_line_count": 1,
        "inherited_pyrecest_distribution_version": "2.4.3",
        "nerfstudio_version": "1.1.5",
        "nuscenes_devkit_numpy_constraint": "numpy<2.0.0,>=1.22.0",
        "nuscenes_devkit_version": "1.2.0",
        "other_dependency_conflicts_allowed": False,
        "pyrecest_extra_installed": False,
        "pyrecest_runtime_used": False,
        "runtime_numpy_version": "1.26.4",
    }
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["runtime_dependency_check_scoped"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "runtime_dependency_check_scoped"
    )


def test_runtime_records_the_prior_scope_repair_as_superseded() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    correction = payload["correction"]
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert f"RUNTIME_DEPENDENCY_SCOPE_REPAIR_ID: {payload['repair_id']}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert correction["exact_allowlisted_pip_check_line"] in workflow
    assert 'test "${pip_check_status}" -eq 1' not in workflow
    assert 'test "${pip_check_status}" -eq 0' in workflow
    assert 'test "${pip_check_output}" = "No broken requirements found."' in workflow
    assert 'version("numpy") != "1.26.4"' in workflow
    assert 'version("pyrecest") != "2.4.3"' not in workflow
    assert "PyRecEst entered the isolated source runtime" in workflow
    assert 'version("nuscenes-devkit") != "1.2.0"' in workflow
    assert "pip check || true" not in workflow
    assert f'RUNTIME_DEPENDENCY_SCOPE_REPAIR_ID="{payload["repair_id"]}"' in runner
    assert '"runtime_dependency_scope_repair"' in runner
    assert '"superseded": True' in runner


def test_candidate_does_not_install_the_optional_pyrecest_extra() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runtime = workflow.split("Build isolated GPU source runtime", 1)[1]

    assert '-e ".[graph,vision]"' in runtime
    assert ".[graph,vision,pyrecest]" not in runtime
