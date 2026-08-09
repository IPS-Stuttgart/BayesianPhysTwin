from __future__ import annotations

import json
from pathlib import Path

import yaml

from bayesian_phystwin._portable_contracts import content_id

WORKFLOW = Path(".github/workflows/deform360-robot-metric-prefix-smoke.yml")
LAUNCHER = Path(".github/workflows/launch-deform360-robot-metric-prefix-smoke-once.yml")
POLICY = Path(
    "protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json"
)


def test_robot_metric_smoke_is_public_source_only_and_one_shot() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert isinstance(yaml.load(text, Loader=yaml.BaseLoader), dict)
    assert isinstance(yaml.load(launcher, Loader=yaml.BaseLoader), dict)
    assert "workflow_call:" in text
    assert "workflow_dispatch:" not in text
    assert "runs-on: self-hosted" in text
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert 'PRODUCTION_RUN_ID: "31279398563"' in text
    assert 'PRODUCTION_ARTIFACT_ID: "9031215572"' in text
    assert "PRODUCTION_RESULT_ID: 146f885351b2af" in text
    assert 'succeeded_job_count": 324' in text
    assert 'technical_failure_job_count": 0' in text
    assert 'test ! -e "${output}"' in text
    assert "jobs = sorted(" in text
    assert "job = jobs[0]" in text
    assert "rendered_depth.h5" not in text
    assert "adaptive-confirmation" not in text
    assert 'confirmation_payloads_opened": False' in text
    assert 'target_outcomes_used": False' in text
    assert 'human_approval_required": False' in text
    assert "branches: [main]" in launcher
    assert "workflow_dispatch:" not in launcher
    assert "execute_authorized: true" in launcher
    assert "cancel-in-progress: false" in launcher


def test_robot_metric_policy_names_the_new_evidence_source() -> None:
    text = POLICY.read_text(encoding="utf-8")
    policy = json.loads(text)

    assert "released-deform360-robot-taxel-gauge-v1" in text
    assert '"full_sequence_rendered_depth_used": false' in text
    assert '"future_frames_used": false' in text
    assert '"human_approval_required": false' in text
    assert '"confirmation_payloads_opened": false' in text
    assert '"target_outcomes_used": false' in text
    declared = policy.pop("artifact_id")
    assert declared == content_id(policy)
