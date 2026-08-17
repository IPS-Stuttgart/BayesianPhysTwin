import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_alltrain import (
    DEFORM_DLO2_DEEP_ALLTRAIN_CONTRACT,
    load_deform_dlo2_deep_alltrain_protocol,
    validate_deform_dlo2_deep_alltrain_authorization,
)
from bayesian_phystwin.deform_dlo_source import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform_dlo2_deep_alltrain_refit_v1.json"
)


def _source_result(seed: int, update: int) -> dict[str, object]:
    return {
        "contract": "deform-dlo-source-reproduction-result-v1",
        "official_eval_read": False,
        "advancement_authorized": True,
        "source_gate": {"passed": True},
        "selected_checkpoint": {"update": update},
        "stage_authorization": {
            "contract": "deform-dlo2-deep-seed-authorization-v1",
            "seed": seed,
        },
    }


def _ensemble_parent() -> tuple[dict[str, object], dict[str, object]]:
    selected_arm = "equal_weight_predictive_mean"
    selected_spec = {
        "operator": "predictive_mean",
        "weights": {"42": 0.5, "43": 0.5},
        "selected_member_updates": {"42": 6400, "43": 6040},
    }
    selection = {
        "selected_arm": selected_arm,
        "fallback_used": False,
        "relative_improvement": 0.02,
    }
    seal = {
        "contract": "deform-dlo2-deep-ensemble-v1",
        "official_eval_read": False,
        "source_test_evaluated_by_this_stage": False,
        "protocol": {
            "sha256": "d8962edfa305356783a9d963aa221db1052956fcb1912af2ab45038363460a11"
        },
        "seed42_source_result": {"sha256": "b" * 64},
        "seed43_source_result": {"sha256": "c" * 64},
        "selection": selection,
        "candidate_specs": {selected_arm: selected_spec},
    }
    result = {
        "contract": "deform-dlo2-deep-ensemble-result-v1",
        "official_eval_read": False,
        "selection_seal": {"sha256": "a" * 64},
        "selection": selection,
        "selected_arm": selected_arm,
        "selected_spec": selected_spec,
        "comparison_baseline_seed": 42,
        "exact_fallback": False,
        "source_test": {
            "transfer": {"relative_improvement": 0.02, "wins": 6},
            "candidate_gate": {"passed": True},
        },
        "uncertainty": {
            "validation_fitted_variance_scale": 2.0,
            "variance_floor_m2": 0.000025,
            "nominal_coordinate_coverage": 0.9,
        },
        "alltrain_deep_ensemble_authorized": True,
        "alltrain_authorization_contract": {
            "seeds": [42, 43],
            "same_operator_and_weights": True,
            "no_source_retuning": True,
        },
    }
    return result, seal


def test_deep_alltrain_protocol_binds_both_fresh_source_members() -> None:
    protocol = load_deform_dlo2_deep_alltrain_protocol(PROTOCOL)

    assert protocol["contract"] == DEFORM_DLO2_DEEP_ALLTRAIN_CONTRACT
    assert protocol["training"]["random_seeds"] == [42, 43]
    assert protocol["training"]["total_updates"] == 6400
    assert protocol["data"]["trajectory_count"] == 56
    assert protocol["data"]["official_eval_read"] is False
    for identity in protocol["parents"].values():
        assert identity["sha256"] == sha256_file(
            REPOSITORY_ROOT / identity["repository_path"]
        )


def test_deep_alltrain_authorization_copies_weights_and_updates_exactly() -> None:
    protocol = load_deform_dlo2_deep_alltrain_protocol(PROTOCOL)
    ensemble, seal = _ensemble_parent()
    sources = {42: _source_result(42, 6400), 43: _source_result(43, 6040)}

    selected = validate_deform_dlo2_deep_alltrain_authorization(
        protocol,
        sources,
        ensemble,
        seal,
        source_protocol_sha256s={
            42: protocol["parents"]["seed42_source_protocol"]["sha256"],
            43: protocol["parents"]["seed43_source_protocol"]["sha256"],
        },
        source_result_sha256s={42: "b" * 64, 43: "c" * 64},
        ensemble_protocol_sha256=protocol["parents"]["ensemble_protocol"][
            "sha256"
        ],
        selection_seal_sha256="a" * 64,
    )

    assert selected["weights"] == {42: 0.5, 43: 0.5}
    assert selected["member_updates"] == {42: 6400, 43: 6040}
    assert selected["comparison_baseline_seed"] == 42


def test_deep_alltrain_rejects_fallback_or_changed_member_update() -> None:
    protocol = load_deform_dlo2_deep_alltrain_protocol(PROTOCOL)
    ensemble, seal = _ensemble_parent()
    sources = {42: _source_result(42, 6400), 43: _source_result(43, 6040)}
    kwargs = {
        "source_protocol_sha256s": {
            42: protocol["parents"]["seed42_source_protocol"]["sha256"],
            43: protocol["parents"]["seed43_source_protocol"]["sha256"],
        },
        "source_result_sha256s": {42: "b" * 64, 43: "c" * 64},
        "ensemble_protocol_sha256": protocol["parents"]["ensemble_protocol"][
            "sha256"
        ],
        "selection_seal_sha256": "a" * 64,
    }

    ensemble["exact_fallback"] = True
    with pytest.raises(ValueError, match="did not authorize"):
        validate_deform_dlo2_deep_alltrain_authorization(
            protocol,
            sources,
            ensemble,
            seal,
            **kwargs,
        )

    ensemble["exact_fallback"] = False
    sources[43]["selected_checkpoint"]["update"] = 5200
    with pytest.raises(ValueError, match="did not authorize"):
        validate_deform_dlo2_deep_alltrain_authorization(
            protocol,
            sources,
            ensemble,
            seal,
            **kwargs,
        )


def test_deep_alltrain_protocol_rejects_target_reselection(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["method_transfer"]["target_reselection"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="transfer contract"):
        load_deform_dlo2_deep_alltrain_protocol(changed)
