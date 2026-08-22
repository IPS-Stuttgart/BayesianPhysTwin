from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rgbench_matphys_protocol_v1 import (
    MATPHYS_REVISION,
    RGBENCH_HF_REVISION,
    RGBENCH_REVISION,
    load_rgbench_matphys_preaccess_amendment_v1,
    load_rgbench_matphys_protocol_v1,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/locks/rgbench_matphys_selective_risk_v1.json"
AMENDMENT = (
    ROOT
    / "protocols/amendments/rgbench_matphys_selective_risk_v1_preaccess.json"
)


def _payload() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _reseal(value: dict[str, Any]) -> None:
    identity = dict(value)
    identity.pop("policy_id", None)
    value["policy_id"] = content_id(identity)


def _write(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_frozen_protocol_loads_with_target_closed() -> None:
    protocol = load_rgbench_matphys_protocol_v1(PROTOCOL)

    assert protocol.value["upstreams"]["rgbench_revision"] == RGBENCH_REVISION
    assert protocol.value["upstreams"]["rgbench_hf_revision"] == RGBENCH_HF_REVISION
    assert protocol.value["upstreams"]["matphys_revision"] == MATPHYS_REVISION
    assert len(protocol.source_cells) == 15
    assert len(protocol.target_cells) == 12
    assert {cell.garment_id for cell in protocol.source_cells} == {
        "beige_hoodie",
        "blue_dress",
        "green_tshirt",
        "grey_sunwear",
        "white_shirt",
    }
    assert {cell.garment_id for cell in protocol.target_cells} == {
        "brown_coat",
        "grey_pleat_skirt",
        "khaki_blazer",
        "white_cakeskirt",
    }
    assert protocol.target_execution_authorized is False
    assert protocol.value["information_boundary"] == {
        "public_metadata_read": True,
        "source_payload_download_allowed_after_lock": True,
        "source_payload_decode_allowed_after_lock": True,
        "source_outcomes_may_be_used_for_development": True,
        "target_payload_download_allowed": False,
        "target_payload_decode_allowed": False,
        "target_outcomes_opened": False,
        "target_execution_authorized": False,
        "replacement_allowed": False,
    }


def test_policy_id_binds_every_contract_field() -> None:
    value = _payload()
    declared = value.pop("policy_id")

    assert declared == content_id(value)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("upstreams", "matphys_revision", "0" * 40, "MatPhys revision changed"),
        (
            "selection",
            "cell_selection_salt",
            "post-outcome-salt",
            "cell selection salt changed",
        ),
        (
            "information_boundary",
            "target_payload_decode_allowed",
            True,
            "information boundary changed",
        ),
        (
            "information_boundary",
            "replacement_allowed",
            True,
            "information boundary changed",
        ),
    ],
)
def test_resealed_contract_mutations_still_fail_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _payload()
    payload[section][field] = value
    _reseal(payload)

    with pytest.raises(ValueError, match=match):
        load_rgbench_matphys_protocol_v1(_write(tmp_path, payload))


def test_cell_identity_and_selection_key_are_both_frozen(tmp_path: Path) -> None:
    payload = _payload()
    payload["selection"]["source_cells"][0]["sample_id"] = "99"
    _reseal(payload)

    with pytest.raises(ValueError, match="selection key changed"):
        load_rgbench_matphys_protocol_v1(_write(tmp_path, payload))

    payload = _payload()
    payload["selection"]["source_cells"][0]["selection_key_sha256"] = "0" * 64
    _reseal(payload)
    with pytest.raises(ValueError, match="selection key changed"):
        load_rgbench_matphys_protocol_v1(_write(tmp_path, payload))


def test_missing_action_or_cross_split_replacement_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["selection"]["source_cells"].pop()
    _reseal(payload)
    with pytest.raises(ValueError, match="source_cells action coverage changed"):
        load_rgbench_matphys_protocol_v1(_write(tmp_path, payload))

    payload = _payload()
    payload["selection"]["target_manifold_garments"][0] = "beige_hoodie"
    _reseal(payload)
    with pytest.raises(ValueError, match="target garments changed"):
        load_rgbench_matphys_protocol_v1(_write(tmp_path, payload))


def test_preaccess_amendment_moves_every_previously_registered_garment_to_source() -> None:
    amended = load_rgbench_matphys_preaccess_amendment_v1(PROTOCOL, AMENDMENT)

    assert len(amended.source_cells) == 21
    assert len(amended.target_cells) == 6
    assert {cell.garment_id for cell in amended.target_cells} == {
        "grey_sunwear",
        "khaki_blazer",
    }
    assert amended.target_execution_authorized is False
    assert amended.amendment["information_boundary"]["target_payload_read_allowed"] is False


def test_preaccess_amendment_is_content_addressed_and_fails_closed(
    tmp_path: Path,
) -> None:
    value: dict[str, Any] = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = value.pop("amendment_id")
    assert declared == content_id(value)

    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    value["information_boundary"]["target_payload_read_allowed"] = True
    identity = dict(value)
    identity.pop("amendment_id")
    value["amendment_id"] = content_id(identity)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="amended information boundary changed"):
        load_rgbench_matphys_preaccess_amendment_v1(PROTOCOL, path)


def test_preaccess_amendment_rejects_relabeling_an_exposed_garment(
    tmp_path: Path,
) -> None:
    value: dict[str, Any] = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    value["amended_roles"]["target_garments"][0] = "brown_coat"
    identity = dict(value)
    identity.pop("amendment_id")
    value["amendment_id"] = content_id(identity)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="amended target garments changed"):
        load_rgbench_matphys_preaccess_amendment_v1(PROTOCOL, path)
