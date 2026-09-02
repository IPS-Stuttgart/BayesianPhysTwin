from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    METHOD_LOCK_ID,
    load_poseit_real_decision_method_lock,
    poseit_method_lock_config_sha256,
    poseit_method_lock_file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "poseit_real_decision_probe_v1.json"
MAPPING = (
    ROOT
    / "protocols"
    / "poseit_real_decision_probe_v1_preaccess_mapping_constraints.json"
)
METHOD_LOCK = ROOT / "protocols" / "poseit_real_decision_probe_v1_method_lock.json"


def test_method_lock_is_exact_and_parent_bound() -> None:
    payload = load_poseit_real_decision_method_lock(
        METHOD_LOCK,
        parent_protocol_path=PROTOCOL,
        mapping_constraints_path=MAPPING,
    )

    assert payload["contract"] == METHOD_LOCK_ID
    assert (
        poseit_method_lock_file_sha256(METHOD_LOCK)
        == "4fa1ef3c96df28a67e13461b79c44690f53f5abb4c90e06200c4e90bcf8e1a1c"
    )
    assert (
        poseit_method_lock_config_sha256(payload)
        == "96ca3eb18ced1d01a42aeadc3ec71aa1042719be3b2f524d9b60df675eb5d148"
    )


def test_method_lock_rejects_source_independent_method_drift(tmp_path: Path) -> None:
    payload = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    payload["latent_response_twin"]["covariance"]["diagonal_shrinkage"] = 0.2
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="shrinkage changed"):
        load_poseit_real_decision_method_lock(changed)


def test_method_lock_records_all_outcome_boundaries_closed() -> None:
    payload = load_poseit_real_decision_method_lock(METHOD_LOCK)

    assert all(value is False for value in payload["boundaries"].values())
    assert payload["calibration"]["uses_confirmation"] is False
    assert payload["preprocessing"]["fit_only_standardization"]["uses_outcome"] is False
