from __future__ import annotations

import json
import re
from pathlib import Path

from bayesian_phystwin.deform360_exclusion_union import validate_exclusion_manifest
from bayesian_phystwin.deform360_tactile_features import file_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY_ROOT
    / "configs/sota/deform360_tactile_regret_guard_fresh_exclusion_v1.json"
)
SOURCE_FILES = {
    "Deform360TactileRegretGuardSourceDiagnostic": (
        REPOSITORY_ROOT
        / "results/sota/diagnostics/deform360_tactile_regret_guard_source_v1/result.json"
    ),
    "Deform360SelectiveVirtualSensingPredictionCohortSeal": (
        REPOSITORY_ROOT
        / "results/sota/deform360_selective_virtual_sensing_v1/prediction_cohort_seal.json"
    ),
    "Deform360BiasAwareProspectiveV2CalibrationCohortSeal": (
        REPOSITORY_ROOT
        / "results/sota/deform360_bias_aware_guarded_belief_prospective_v2"
        / "calibration_prediction_cohort_seal_v2.json"
    ),
}


def test_fresh_exclusion_union_is_sealed_and_hash_only() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_exclusion_manifest(payload)
    assert payload["accounting"] == {
        "independent_hash_count": 85,
        "opened_source_object_count": 43,
        "new_opened_source_hash_count": 12,
        "union_hash_count": 97,
    }
    text = MANIFEST.read_text(encoding="utf-8")
    assert re.search(r'"[0-9]{3}-[a-z0-9-]+"', text) is None
    assert payload["information_boundary"]["held_runtime_tree_accessed"] is False
    assert payload["information_boundary"]["object_ids_emitted"] is False


def test_opened_source_inputs_remain_byte_bound() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {
        str(row["artifact_kind"]): str(row["file_sha256"])
        for row in payload["opened_source_inputs"]
    }
    assert records == {
        kind: file_sha256(path) for kind, path in SOURCE_FILES.items()
    }
