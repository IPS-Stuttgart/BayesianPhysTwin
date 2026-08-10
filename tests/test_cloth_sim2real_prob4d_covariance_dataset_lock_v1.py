from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from bayesian_phystwin.prob4d_covariance_ablation import (
    PROB4D_COVARIANCE_ABLATION_SCHEMA,
    Prob4DCovarianceAblationV1,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/science/run_cloth_sim2real_prob4d_covariance_locked_v1.py"
WORKFLOW = ROOT / (
    ".github/workflows/cloth-sim2real-prob4d-covariance-evidence-v1.yml"
)
EXPECTED_SHA256 = "268d07d94396f6f4ca277b6da0e8acf43512747fea6d40327eb33166da972c7f"
ACTION_REFERENCE = re.compile(r"^\s*uses:\s+([^\s@]+)@([^\s#]+)", re.MULTILINE)


def _module():
    spec = importlib.util.spec_from_file_location(
        "cloth_covariance_dataset_lock",
        LAUNCHER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_binds_authoritative_dataset_sha256() -> None:
    module = _module()

    assert module.DATASET_SHA256 == EXPECTED_SHA256
    assert len(module.DATASET_SHA256) == 64
    assert module._IMPLEMENTATION.DATASET_SHA256 == EXPECTED_SHA256
    assert module.CovariancePolicy.from_treatment("full_joint").construction == (
        "persistent"
    )


def test_authoritative_identity_differs_from_known_truncated_value() -> None:
    module = _module()

    assert module.LEGACY_MALFORMED_DATASET_SHA256 != EXPECTED_SHA256
    assert len(module.LEGACY_MALFORMED_DATASET_SHA256) == 63


def test_ablation_schema_version_rejects_integral_float() -> None:
    payload = {
        "schema": PROB4D_COVARIANCE_ABLATION_SCHEMA,
        "schema_version": 1.0,
        "ablation_id": "schema-version-control",
        "reference_treatment": "independent_rows",
        "locked_factors": {},
        "variants": [],
        "evidence": {},
    }

    with pytest.raises(ValueError, match="schema_version"):
        Prob4DCovarianceAblationV1.from_mapping(payload)


def test_workflow_pins_actions_and_records_checked_out_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'SOURCE_REVISION="$(git rev-parse HEAD)"' in text
    assert '"source_revision": os.environ["SOURCE_REVISION"]' in text
    assert '"source_revision": os.environ["GITHUB_SHA"]' not in text
    references = ACTION_REFERENCE.findall(text)
    assert references
    for action, revision in references:
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{action} must use a full lowercase commit SHA, got {revision!r}"
        )
