import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    partition_deform_source_names,
    validate_deform_dlo2_fresh_parent,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_fresh_v1.json"


def _authorized_parent(protocol_payload: dict[str, object]) -> dict[str, object]:
    authorization = protocol_payload["authorization"]
    return {
        "contract": authorization["required_parent_contract"],
        "official_eval_read": False,
        "source_gate": {"passed": True},
        "checkpoint_posterior_authorized": True,
        "protocol": {
            "sha256": authorization["required_parent_protocol_sha256"],
        },
    }


def test_dlo2_fresh_protocol_is_disjoint_and_eval_closed() -> None:
    protocol = load_deform_dlo_source_protocol(PROTOCOL)

    assert protocol["dlo_types"] == ("DLO2",)
    assert protocol["data"]["expected_node_count"] == {"DLO2": 12}
    assert protocol["training"]["total_updates"] == 6400
    assert protocol["training"]["checkpoint_updates"][-2:] == [6040, 6400]
    assert protocol["source_gate"]["published_reference_l1_m"]["DLO2"] == 0.0097
    assert protocol["data"]["official_eval_metrics_opened"] is False
    assert protocol["future_stage"]["official_eval_allowed_before_source_gate"] is False

    split = partition_deform_source_names(
        [f"{index}.pkl" for index in range(56)],
        seed=protocol["source_split"]["seed"],
        fit_count=40,
        validation_count=8,
        source_test_count=8,
    )
    assert tuple(map(len, split.values())) == (40, 8, 8)
    assert len(set().union(*map(set, split.values()))) == 56


def test_dlo2_fresh_parent_gate_accepts_only_authorized_longrun() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = _authorized_parent(payload)

    authorized = validate_deform_dlo2_fresh_parent(payload, parent)

    assert authorized["source_gate_passed"] is True
    assert authorized["checkpoint_posterior_authorized"] is True

    parent["source_gate"]["passed"] = False
    with pytest.raises(ValueError, match="did not authorize"):
        validate_deform_dlo2_fresh_parent(payload, parent)


def test_dlo2_fresh_posterior_matches_dlo1_policy() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    posterior = json.loads(
        (
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "deform_dlo_longrun_posterior_v1.json"
        ).read_text(encoding="utf-8")
    )

    for key in (
        "operators",
        "softmax_temperature_m",
        "coordinate_variance_floor_m2",
        "coordinate_interval_nominal_coverage",
        "arms",
    ):
        assert payload["checkpoint_posterior"][key] == posterior[key]
