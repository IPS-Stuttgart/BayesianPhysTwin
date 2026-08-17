import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from bayesian_phystwin.deform_dlo_deep_ensemble import (
    DEFORM_DLO2_DEEP_ENSEMBLE_CONTRACT,
    DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT,
    build_deform_two_seed_weights,
    load_deform_dlo1_deep_ensemble_protocol,
    load_deform_dlo2_deep_ensemble_protocol,
    validate_deform_dlo2_deep_ensemble_parent,
    validate_deform_two_seed_manifests,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
    validate_deform_dlo2_stage_authorization,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo1_deep_ensemble_eval_v1.json"
)
RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo_deep_ensemble.py"
DLO2_SEED_PROTOCOLS = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_deep_seed42_v1.json",
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo2_deep_seed43_v1.json",
)
DLO2_ENSEMBLE_PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform_dlo2_deep_ensemble_eval_v1.json"
)


def _load_runner():
    scripts_root = str(RUNNER.parent)
    sys.path.insert(0, scripts_root)
    try:
        spec = importlib.util.spec_from_file_location("deform_deep_ensemble", RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts_root)


def test_deep_ensemble_protocol_binds_both_seeds_and_source_manifest() -> None:
    protocol = load_deform_dlo1_deep_ensemble_protocol(PROTOCOL)
    parents = protocol["parents"]
    policy = protocol["policy"]

    assert protocol["contract"] == DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT
    for identity in parents.values():
        assert identity["sha256"] == sha256_file(
            REPOSITORY_ROOT / identity["repository_path"]
        )
    seed43 = json.loads(
        (
            REPOSITORY_ROOT / parents["seed43_source_protocol"]["repository_path"]
        ).read_text(encoding="utf-8")
    )["deep_ensemble_candidate"]
    seed43.pop("companion_seed42_protocol")
    assert seed43 == policy
    assert policy["fallback"] == "comparison-baseline-exact"
    assert policy["validation_improvement_min"] == 0.01
    assert policy["source_transfer_improvement_min"] == 0.01
    assert policy["source_transfer_minimum_case_wins"] == 5


def test_deep_ensemble_weights_are_validation_only_and_normalized() -> None:
    policy = load_deform_dlo1_deep_ensemble_protocol(PROTOCOL)["policy"]

    weights = build_deform_two_seed_weights(
        {42: 0.010, 43: 0.012},
        policy,
    )

    assert weights["equal_weight_predictive_mean"] == {42: 0.5, 43: 0.5}
    softmax = weights["validation_softmax_predictive_mean"]
    assert sum(softmax.values()) == pytest.approx(1.0)
    assert softmax[42] > softmax[43]


def _manifest(*, digest: str = "a" * 64) -> dict[str, object]:
    return {
        "contract": "deform-dlo-source-reproduction-v1",
        "dlo_type": "DLO1",
        "official_eval_read": False,
        "split": {
            "fit": ["fit.pkl"],
            "validation": ["validation.pkl"],
            "source_test": ["source.pkl"],
        },
        "trajectories": {
            "fit.pkl": {"sha256": digest, "size_bytes": 10},
            "validation.pkl": {"sha256": "b" * 64, "size_bytes": 20},
            "source.pkl": {"sha256": "c" * 64, "size_bytes": 30},
        },
    }


def test_deep_ensemble_manifests_require_identical_bytes_and_split() -> None:
    seed42 = _manifest()
    seed43 = _manifest()

    validate_deform_two_seed_manifests(seed42, seed43)

    changed = _manifest(digest="d" * 64)
    with pytest.raises(ValueError, match="trajectory bytes"):
        validate_deform_two_seed_manifests(seed42, changed)


def test_deep_ensemble_runner_loads_only_the_selected_checkpoint(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model_state_dict": {"weight": torch.tensor([1.0])}},
        checkpoint,
    )
    result = {
        "selected_checkpoint": {
            "update": 6400,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "update": 6400,
            },
        }
    }

    update, state = runner._selected_state(result, torch=torch)

    assert update == 6400
    assert torch.equal(state["weight"], torch.tensor([1.0]))


def _authorized_ensemble_parent() -> tuple[dict[str, object], dict[str, object]]:
    selected_arm = "equal_weight_predictive_mean"
    selected_spec = {
        "operator": "predictive_mean",
        "weights": {"42": 0.5, "43": 0.5},
        "selected_member_updates": {"42": 6400, "43": 6400},
    }
    selection = {
        "selected_arm": selected_arm,
        "fallback_used": False,
        "relative_improvement": 0.02,
    }
    selection_seal = {
        "contract": "deform-dlo-deep-ensemble-v1",
        "official_eval_read": False,
        "protocol": {"sha256": sha256_file(PROTOCOL)},
        "candidate_specs": {selected_arm: selected_spec},
        "selection": selection,
        "source_test_evaluated_by_this_stage": False,
    }
    result = {
        "contract": "deform-dlo-deep-ensemble-result-v1",
        "official_eval_read": False,
        "selection_seal": {"sha256": "a" * 64},
        "selection": selection,
        "selected_arm": selected_arm,
        "selected_spec": selected_spec,
        "exact_fallback": False,
        "source_test": {
            "transfer": {"relative_improvement": 0.02, "wins": 6},
        },
        "uncertainty": {"validation_fitted_variance_scale": 2.0},
        "fresh_dlo2_deep_ensemble_authorized": True,
        "fresh_confirmation_contract": load_deform_dlo1_deep_ensemble_protocol(
            PROTOCOL
        )["policy"]["fresh_confirmation"],
    }
    return result, selection_seal


@pytest.mark.parametrize(
    ("path", "seed", "peer_seed"),
    ((DLO2_SEED_PROTOCOLS[0], 42, 43), (DLO2_SEED_PROTOCOLS[1], 43, 42)),
)
def test_dlo2_deep_seed_protocols_copy_the_frozen_dlo1_policy(
    path: Path,
    seed: int,
    peer_seed: int,
) -> None:
    protocol = load_deform_dlo_source_protocol(path)

    assert protocol["dlo_types"] == ("DLO2",)
    assert protocol["training"]["random_seed"] == seed
    assert protocol["training"]["total_updates"] == 6400
    assert protocol["deep_ensemble_role"] == {
        "seed": seed,
        "peer_seed": peer_seed,
        "operator_bank": "copy-dlo1-exactly",
        "no_dlo1_retuning": True,
    }
    required = protocol["authorization"]["required_parent_deep_ensemble"]
    assert required["protocol_sha256"] == sha256_file(PROTOCOL)
    assert protocol["data"]["official_eval_metrics_opened"] is False


def test_dlo2_deep_seed_requires_a_nonfallback_dlo1_ensemble() -> None:
    source = load_deform_dlo_source_protocol(DLO2_SEED_PROTOCOLS[0])
    parent_protocol = load_deform_dlo1_deep_ensemble_protocol(PROTOCOL)
    result, seal = _authorized_ensemble_parent()

    authorized = validate_deform_dlo2_deep_ensemble_parent(
        source,
        parent_protocol,
        result,
        seal,
        parent_protocol_sha256=sha256_file(PROTOCOL),
        selection_seal_sha256="a" * 64,
    )

    assert authorized["selected_arm"] == "equal_weight_predictive_mean"
    assert authorized["source_transfer_wins"] == 6
    result["exact_fallback"] = True
    with pytest.raises(ValueError, match="did not authorize"):
        validate_deform_dlo2_deep_ensemble_parent(
            source,
            parent_protocol,
            result,
            seal,
            parent_protocol_sha256=sha256_file(PROTOCOL),
            selection_seal_sha256="a" * 64,
        )


def test_dlo2_deep_stage_authorization_binds_seed_role_and_parent() -> None:
    protocol = load_deform_dlo_source_protocol(DLO2_SEED_PROTOCOLS[0])
    protocol_sha256 = sha256_file(DLO2_SEED_PROTOCOLS[0])
    result, _ = _authorized_ensemble_parent()
    parent = {
        "sha256": "b" * 64,
        "contract": result["contract"],
        "selection_contract": "deform-dlo-deep-ensemble-v1",
        "selected_arm": result["selected_arm"],
        "selected_spec": result["selected_spec"],
        "source_transfer_relative_improvement": 0.02,
        "source_transfer_wins": 6,
        "validation_fitted_variance_scale": 2.0,
        "fresh_dlo2_deep_ensemble_authorized": True,
    }
    authorization = {
        "contract": "deform-dlo2-deep-seed-authorization-v1",
        "official_eval_read": False,
        "source_test_opened": False,
        "protocol": {"sha256": protocol_sha256},
        "parent_deep_ensemble_result": parent,
    }

    validated = validate_deform_dlo2_stage_authorization(
        protocol,
        authorization,
        protocol_sha256=protocol_sha256,
    )

    assert validated["seed"] == 42
    assert validated["peer_seed"] == 43
    parent["source_transfer_wins"] = 4
    with pytest.raises(ValueError, match="deep-seed stage authorization"):
        validate_deform_dlo2_stage_authorization(
            protocol,
            authorization,
            protocol_sha256=protocol_sha256,
        )


def test_dlo2_deep_ensemble_protocol_is_a_fresh_exact_policy_copy() -> None:
    protocol = load_deform_dlo2_deep_ensemble_protocol(DLO2_ENSEMBLE_PROTOCOL)
    parent = load_deform_dlo1_deep_ensemble_protocol(PROTOCOL)

    assert protocol["contract"] == DEFORM_DLO2_DEEP_ENSEMBLE_CONTRACT
    assert protocol["evaluation"]["dlo_type"] == "DLO2"
    assert protocol["evaluation"]["node_count"] == 12
    assert protocol["evaluation"]["official_eval_read"] is False
    for label, path in zip(
        ("seed42_source_protocol", "seed43_source_protocol"),
        DLO2_SEED_PROTOCOLS,
        strict=True,
    ):
        assert protocol["parents"][label]["sha256"] == sha256_file(path)
    for key in (
        "operators",
        "validation_improvement_min",
        "source_transfer_improvement_min",
        "source_transfer_minimum_case_wins",
        "coordinate_variance_floor_m2",
        "coordinate_interval_nominal_coverage",
    ):
        assert protocol["policy"][key] == parent["policy"][key]
