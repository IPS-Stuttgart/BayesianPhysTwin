from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin.rct_real_decision_protocol import (
    CALIBRATION_MATERIALS,
    CONFIRMATION_MATERIALS,
    HELD_INTERVENTION,
    MANDATORY_ANCHOR,
    SELECTABLE_PROBES,
    SOURCE_TEST_MATERIALS,
    cohort_from_protocol,
    load_rct_real_decision_protocol,
    protocol_config_sha256,
    protocol_file_sha256,
)

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "rct_real_decision_probe_v1.json"
)
PROTOCOL_FILE_SHA256 = (
    "c6eac3371e379956c285fe0ea0743c2ba9b67eb40d09fe18a3642839188ba8bd"
)
PROTOCOL_CONFIG_SHA256 = (
    "6a6d0d0b52ed71cb530e0ad5cb5fe5898f202d6dd9ad099cab6b035fa063a140"
)


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_preoutcome_protocol_is_hash_locked_and_loadable() -> None:
    payload = load_rct_real_decision_protocol(PROTOCOL_PATH)

    assert protocol_file_sha256(PROTOCOL_PATH) == PROTOCOL_FILE_SHA256
    assert protocol_config_sha256(payload) == PROTOCOL_CONFIG_SHA256
    assert payload["promotion"]["target_authorized"] is False
    assert payload["dataset"]["archive"]["expected_sha256"] is None
    assert payload["information_boundary"]["held_v8_access_allowed"] is False


def test_material_roles_are_pairwise_disjoint_and_exhaust_the_reserved_sixty() -> None:
    payload = load_rct_real_decision_protocol(PROTOCOL_PATH)
    cohort = cohort_from_protocol(payload)

    assert cohort.calibration == CALIBRATION_MATERIALS
    assert cohort.source_test == SOURCE_TEST_MATERIALS
    assert cohort.confirmation == CONFIRMATION_MATERIALS
    assert len(cohort.reserved) == 60
    assert cohort.expected_fit_count == 62
    assert cohort.role(CALIBRATION_MATERIALS[0]) == "calibration"
    assert cohort.role(SOURCE_TEST_MATERIALS[0]) == "source_test"
    assert cohort.role(CONFIRMATION_MATERIALS[0]) == "confirmation"
    assert cohort.role("unregistered-fit-material") == "fit"


def test_probe_and_held_intervention_rosters_are_physically_disjoint() -> None:
    payload = load_rct_real_decision_protocol(PROTOCOL_PATH)
    method = payload["method"]
    decision = payload["decision"]

    anchor = (
        method["mandatory_anchor"]["position"],
        method["mandatory_anchor"]["sensor"],
    )
    probes = tuple(
        (record["position"], record["sensor"])
        for record in method["selectable_probes"]
    )
    held = (
        decision["held_intervention"]["position"],
        decision["held_intervention"]["sensor"],
    )
    assert anchor == MANDATORY_ANCHOR
    assert probes == SELECTABLE_PROBES
    assert held == HELD_INTERVENTION
    assert len({anchor, *probes, held}) == 5


@pytest.mark.parametrize(
    "mutation",
    (
        "authorize_target",
        "set_archive_hash",
        "change_confirmation_material",
        "change_probe",
        "allow_held_v8",
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
    elif mutation == "change_confirmation_material":
        changed["cohort"]["confirmation"]["material_ids"][0] = "different"
    elif mutation == "change_probe":
        changed["method"]["selectable_probes"][0]["sensor"] = 3
    elif mutation == "allow_held_v8":
        changed["information_boundary"]["held_v8_access_allowed"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        load_rct_real_decision_protocol(_write_payload(tmp_path, changed))
