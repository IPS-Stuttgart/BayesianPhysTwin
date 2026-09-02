from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    load_poseit_preaccess_mapping_constraints,
    poseit_mapping_constraints_config_sha256,
    poseit_mapping_constraints_file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/poseit_real_decision_probe_v1.json"
CONSTRAINTS = (
    ROOT
    / "protocols"
    / "poseit_real_decision_probe_v1_preaccess_mapping_constraints.json"
)
CONSTRAINTS_FILE_SHA256 = (
    "8bf66c087437d77589d5fcd35d74a47b2a4d8ba69b311041123d719da8445210"
)
CONSTRAINTS_CONFIG_SHA256 = (
    "75551bbdc63021d768bbc5f44ba9cc38a56592689306a11dff4912abf3f18682"
)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "mapping-constraints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_preaccess_mapping_constraints_are_hash_locked() -> None:
    payload = load_poseit_preaccess_mapping_constraints(
        CONSTRAINTS,
        parent_protocol_path=PROTOCOL,
    )

    assert (
        poseit_mapping_constraints_file_sha256(CONSTRAINTS) == CONSTRAINTS_FILE_SHA256
    )
    assert (
        poseit_mapping_constraints_config_sha256(payload) == CONSTRAINTS_CONFIG_SHA256
    )
    temporal = payload["experiment_fixed_constraints"]["temporal_sampling"]
    assert temporal["sample_count"] == 20
    assert temporal["shake_or_retract_samples_allowed"] is False
    assert payload["boundaries"]["phase_labels_opened"] is False
    assert payload["boundaries"]["confirmation_opened"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("boundaries", "phase_labels_opened", True),
        ("boundaries", "confirmation_opened", True),
        ("official_source_constraints", "repository_revision", "different"),
    ),
)
def test_mapping_constraints_reject_source_or_boundary_drift(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    payload = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed[section][key] = value

    with pytest.raises(ValueError):
        load_poseit_preaccess_mapping_constraints(_write(tmp_path, changed))


def test_mapping_constraints_reject_temporal_drift(tmp_path: Path) -> None:
    payload = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["experiment_fixed_constraints"]["temporal_sampling"]["sample_count"] = 21

    with pytest.raises(ValueError, match="sample count"):
        load_poseit_preaccess_mapping_constraints(_write(tmp_path, changed))


def test_mapping_constraints_reject_parent_protocol_drift(tmp_path: Path) -> None:
    parent = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent["promotion"]["target_authorized"] = True
    changed_parent = tmp_path / "changed-parent.json"
    changed_parent.write_text(json.dumps(parent), encoding="utf-8")

    with pytest.raises(ValueError):
        load_poseit_preaccess_mapping_constraints(
            CONSTRAINTS,
            parent_protocol_path=changed_parent,
        )
