from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    BUDGETS,
    CALIBRATION_COUNT,
    CONFIRMATION_COUNT,
    FIT_COUNT,
    OBJECT_COUNT,
    SELECTABLE_POSES,
    SOURCE_TEST_COUNT,
    SPLIT_DOMAIN,
    derive_poseit_object_cohort,
    load_poseit_real_decision_protocol,
    poseit_protocol_config_sha256,
    poseit_protocol_file_sha256,
)

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "poseit_real_decision_probe_v1.json"
)
PROTOCOL_FILE_SHA256 = (
    "221803b109a82d3a2d923d5e0c18284b965a8848bcd69e25addd97409d31c5d4"
)
PROTOCOL_CONFIG_SHA256 = (
    "fa49b7c2d20d02d554f9d38b6025839d583493bf2a28dbea29a17cb804e66504"
)


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_preoutcome_protocol_is_hash_locked_and_loadable() -> None:
    payload = load_poseit_real_decision_protocol(PROTOCOL_PATH)

    assert poseit_protocol_file_sha256(PROTOCOL_PATH) == PROTOCOL_FILE_SHA256
    assert poseit_protocol_config_sha256(payload) == PROTOCOL_CONFIG_SHA256
    assert payload["promotion"]["target_authorized"] is False
    assert payload["dataset"]["archive"]["expected_sha256"] is None
    assert payload["information_boundary"]["held_v8_access_allowed"] is False
    assert payload["method"]["model_family"]["prob4d_used"] is False


def test_hash_split_is_deterministic_disjoint_and_object_level() -> None:
    tokens = tuple(f"Object-{index:02d}" for index in range(OBJECT_COUNT))
    cohort = derive_poseit_object_cohort(tuple(reversed(tokens)))

    def key(token: str) -> tuple[str, str]:
        canonical = token.casefold()
        digest = hashlib.sha256(f"{SPLIT_DOMAIN}\0{canonical}".encode()).hexdigest()
        return digest, canonical

    expected = tuple(token.casefold() for token in sorted(tokens, key=key))
    assert cohort.all_objects == expected
    assert len(cohort.fit) == FIT_COUNT
    assert len(cohort.calibration) == CALIBRATION_COUNT
    assert len(cohort.source_test) == SOURCE_TEST_COUNT
    assert len(cohort.confirmation) == CONFIRMATION_COUNT
    assert len(set(cohort.all_objects)) == OBJECT_COUNT
    assert cohort.role(cohort.confirmation[0].upper()) == "confirmation"


def test_probe_outcomes_are_disjoint_from_observed_probe_fields() -> None:
    payload = load_poseit_real_decision_protocol(PROTOCOL_PATH)

    assert tuple(payload["evaluation"]["budgets"]["additional_probe_counts"]) == BUDGETS
    assert tuple(payload["method"]["selectable_probes"]["holding_poses"]) == (
        SELECTABLE_POSES
    )
    assert payload["method"]["selectable_probes"]["shake_outcome_revealed"] is False
    assert payload["decision"]["held_intervention"].startswith(
        "the manually annotated shake-phase outcome"
    )
    forbidden = payload["method"]["feature_boundary"]["forbidden"]
    assert any("shaking-phase sensor sample" in value for value in forbidden)


@pytest.mark.parametrize(
    "mutation",
    (
        "authorize_target",
        "set_archive_hash",
        "change_split",
        "change_probe",
        "reveal_probe_outcome",
        "allow_held_v8",
        "use_prob4d",
    ),
)
def test_preoutcome_protocol_rejects_boundary_drift(
    tmp_path: Path, mutation: str
) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    if mutation == "authorize_target":
        changed["promotion"]["target_authorized"] = True
    elif mutation == "set_archive_hash":
        changed["dataset"]["archive"]["expected_sha256"] = "a" * 64
    elif mutation == "change_split":
        changed["cohort"]["assignment"]["confirmation_count"] = 5
    elif mutation == "change_probe":
        changed["method"]["selectable_probes"]["holding_poses"][0] = 1
    elif mutation == "reveal_probe_outcome":
        changed["method"]["selectable_probes"]["shake_outcome_revealed"] = True
    elif mutation == "allow_held_v8":
        changed["information_boundary"]["held_v8_access_allowed"] = True
    elif mutation == "use_prob4d":
        changed["method"]["model_family"]["prob4d_used"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        load_poseit_real_decision_protocol(_write_payload(tmp_path, changed))
