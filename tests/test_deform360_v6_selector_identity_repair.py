from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin._portable_contracts import content_id
from scripts.remote import run_deform360_v6_selector_identity_repair as repair

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_selector_identity_repair.json"
)
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
WRAPPER = ROOT / "scripts/remote/run_deform360_v6_selector_identity_repair.py"


def test_selector_identity_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == repair.REPAIR_ID == content_id(payload)
    assert repair.load_selector_identity_repair(REPAIR)["repair_id"] == declared
    assert payload["correction"] == {
        "corrected_byte_count": 17310,
        "corrected_sha256": (
            "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
        ),
        "field": "runtime_sources.generic_selector_sha256",
        "historical_registered_digest_found_in_repository_history": False,
        "model_id_prefix": (
            "causal4d_public/deformable-object-sam2.1-small-automatic-v1@"
        ),
        "path": "src/causal4d_public/deform360_object_sam2.py",
        "previous_sha256": (
            "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
        ),
        "repository": "IPS-Stuttgart/Causal4D",
        "repository_revision": "50e3682a5dbf976b20cc9115b6e7a975d0144ea5",
        "selector_class": "DeformableObjectSam2VideoPredictor",
    }
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["runtime_byte_identity_only"] is True
    assert payload["repair_scope"]["claim_authorized"] is False
    assert payload["repair_scope"]["model_family_changed"] is False
    assert payload["repair_scope"]["model_size_changed"] is False
    assert payload["repair_scope"]["selector_algorithm_changed"] is False


def test_selector_identity_repair_tampering_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    payload["correction"]["corrected_byte_count"] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selector identity repair changed"):
        repair.load_selector_identity_repair(changed)


def test_wrapper_patches_only_the_locked_stage_selector_digest() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert 'choices=("stage-prefix",)' in text
    assert "validate_joint_sparse_physical_execution_v5" in text
    assert "patch_joint_sparse_physical_stage_v5" in text
    assert 'setattr(module, "GENERIC_SELECTOR_SHA256"' in text
    assert "CORRECTED_SELECTOR_SHA256" in text
    assert 'getattr(module, "SAM2_REPOSITORY_REVISION"' in text
    assert "source_prediction" not in text
    assert "target_outcome" not in text


def test_runner_uses_the_explicit_repaired_selector_path() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert (
        'SELECTOR_REPAIR_ID="41f3580de5ca7e09bcd4c2623569c293'
        'e29ed796634c60c84ededdbd945af042"' in text
    )
    assert (
        'SELECTOR_SHA256="c10391578c73dde47fbce160312559a7e'
        '638007e9053ec89373fe575cc64d7e5"' in text
    )
    assert (
        'GENERIC_SELECTOR_REPOSITORY="${GITHUB_WORKSPACE}/_causal4d_discovery"'
        in text
    )
    assert (
        'GENERIC_SELECTOR_SOURCE="${GENERIC_SELECTOR_REPOSITORY}/src/'
        'causal4d_public/deform360_object_sam2.py"' in text
    )
    assert "run_deform360_v6_selector_identity_repair.py" in text
    assert '--runtime-repair "${SELECTOR_REPAIR_PATH}"' in text
    assert '--selector-repository "${GENERIC_SELECTOR_REPOSITORY}"' in text
    assert "find \\\n      \"${RUNNER_WORKSPACE:-/home/github-runner}\"" not in text
