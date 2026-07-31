import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    validate_deform_dlo2_checkpoint_posterior,
    validate_deform_dlo2_fresh_posterior_parent,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    partition_deform_source_names,
    validate_deform_dlo2_fresh_parent,
    validate_deform_dlo2_stage_authorization,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_fresh_v2.json"
SUPERSEDED_PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_fresh_v1.json"


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
    assert protocol["model_initialization"] == "official-deform-dlo-initialization-v1"
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


def _posterior_parent() -> tuple[dict[str, object], dict[str, object]]:
    selected_spec = {
        "operator": "predictive_median",
        "weights": {"5200": 0.3, "6040": 0.3, "6400": 0.4},
    }
    selected_arm = "predictive_median::tail_3_uniform"
    selection = {
        "selected_arm": selected_arm,
        "fallback_used": False,
        "relative_improvement": 0.02,
    }
    seal = {
        "contract": "deform-dlo-longrun-posterior-selection-v1",
        "official_eval_read": False,
        "protocol": {
            "sha256": "f2ed821600b9d99ee8cd0d35a0c70ccbf65de75415324b833c75350cf9ab8eeb"
        },
        "selection": selection,
        "candidate_specs": {selected_arm: selected_spec},
    }
    result = {
        "contract": "deform-dlo-longrun-posterior-result-v1",
        "official_eval_read": False,
        "selection_seal": {"sha256": "a" * 64},
        "selection": selection,
        "selected_arm": selected_arm,
        "selected_spec": selected_spec,
        "exact_fallback": False,
        "source_test": {
            "transfer": {"relative_improvement": 0.02, "wins": 6},
        },
        "uncertainty": {
            "validation_fitted_variance_scale": 2.0,
            "source_test": {"coordinate_coverage": 0.85},
        },
        "fresh_dlo2_checkpoint_posterior_authorized": True,
    }
    return result, seal


def test_dlo2_fresh_requires_dlo1_posterior_transfer() -> None:
    payload = load_deform_dlo_source_protocol(PROTOCOL)
    parent, seal = _posterior_parent()

    authorization = validate_deform_dlo2_fresh_posterior_parent(
        payload,
        parent,
        seal,
        selection_seal_sha256="a" * 64,
    )

    assert authorization["selected_arm"] == "predictive_median::tail_3_uniform"
    assert authorization["source_transfer_wins"] == 6
    assert authorization["source_coordinate_coverage"] == 0.85

    parent["fresh_dlo2_checkpoint_posterior_authorized"] = False
    with pytest.raises(ValueError, match="did not authorize"):
        validate_deform_dlo2_fresh_posterior_parent(
            payload,
            parent,
            seal,
            selection_seal_sha256="a" * 64,
        )


def test_dlo2_source_stage_requires_the_two_parent_authorization() -> None:
    protocol = load_deform_dlo_source_protocol(PROTOCOL)
    protocol_sha256 = "67d2fddd82687a9b30fc1ae0284aa199bbffd74d2487fb2c799fcb8b4f6292c0"
    authorization = {
        "contract": "deform-dlo2-fresh-authorization-v2",
        "official_eval_read": False,
        "source_test_opened": False,
        "protocol": {"sha256": protocol_sha256},
        "parent_longrun_result": {
            "sha256": "a" * 64,
            "source_gate_passed": True,
            "checkpoint_posterior_authorized": True,
        },
        "parent_posterior_result": {
            "sha256": "b" * 64,
            "selected_arm": "predictive_median::tail_3_uniform",
            "selected_spec": {"operator": "predictive_median"},
            "source_transfer_relative_improvement": 0.02,
            "source_transfer_wins": 6,
            "validation_fitted_variance_scale": 2.0,
        },
    }

    validated = validate_deform_dlo2_stage_authorization(
        protocol,
        authorization,
        protocol_sha256=protocol_sha256,
    )

    assert validated["selected_arm"] == "predictive_median::tail_3_uniform"
    authorization["parent_posterior_result"]["source_transfer_wins"] = 4
    with pytest.raises(ValueError, match="authorization differs"):
        validate_deform_dlo2_stage_authorization(
            protocol,
            authorization,
            protocol_sha256=protocol_sha256,
        )


def test_dlo2_fresh_posterior_matches_dlo1_policy() -> None:
    payload = load_deform_dlo_source_protocol(PROTOCOL)
    policy = validate_deform_dlo2_checkpoint_posterior(payload)
    posterior = json.loads(
        (
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "deform_dlo_longrun_posterior_v2.json"
        ).read_text(encoding="utf-8")
    )

    for key in (
        "operators",
        "softmax_temperature_m",
        "source_transfer_improvement_min",
        "source_transfer_minimum_case_wins",
        "coordinate_variance_floor_m2",
        "coordinate_interval_nominal_coverage",
        "arms",
    ):
        expected = posterior[key]
        if key == "arms":
            expected = [
                {**arm, "updates": tuple(arm["updates"])} for arm in posterior["arms"]
            ]
        assert policy[key] == expected


def test_dlo2_fresh_superseded_posterior_bank_is_non_executable() -> None:
    payload = load_deform_dlo_source_protocol(SUPERSEDED_PROTOCOL)

    with pytest.raises(ValueError, match="operators differ"):
        validate_deform_dlo2_checkpoint_posterior(payload)


def test_dlo2_fresh_posterior_rejects_a_weakened_fallback() -> None:
    payload = load_deform_dlo_source_protocol(PROTOCOL)
    payload["checkpoint_posterior"]["fallback"] = "best_candidate"

    with pytest.raises(ValueError, match="exact fallback"):
        validate_deform_dlo2_checkpoint_posterior(payload)


def test_dlo2_fresh_posterior_rejects_an_incomplete_source_protocol() -> None:
    payload = load_deform_dlo_source_protocol(PROTOCOL)
    payload["source_split"] = None

    with pytest.raises(ValueError, match="incomplete"):
        validate_deform_dlo2_checkpoint_posterior(payload)


def test_dlo2_fresh_rejects_a_generic_model_initialization(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["model_initialization"] = "constructor-default"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="locked upstream initialization"):
        load_deform_dlo_source_protocol(changed)
