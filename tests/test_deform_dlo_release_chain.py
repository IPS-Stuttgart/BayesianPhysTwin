import json
from pathlib import Path

from bayesian_phystwin.deform_dlo_source import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "sota"
SOURCE_RESULT = (
    REPOSITORY_ROOT / "results" / "sota" / "deform_dlo_source_v1" / "source_result.json"
)
INITIALIZATION_SMOKE = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo2_initialization_amendment_v1"
    / "construction_smoke.json"
)
REFERENCE_OPERATOR_AUDIT = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo2_official_eval_v2"
    / "reference_operator_audit.json"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _arm_signature(payload: dict[str, object]) -> list[tuple[object, ...]]:
    return [
        (arm["name"], tuple(arm["updates"]), arm["weighting"])
        for arm in payload["arms"]
    ]


def test_deform_release_chain_has_no_stale_parent_hash() -> None:
    longrun_path = CONFIG_ROOT / "deform_dlo_longrun_v2.json"
    posterior_path = CONFIG_ROOT / "deform_dlo_longrun_posterior_v2.json"
    fresh_path = CONFIG_ROOT / "deform_dlo2_fresh_v2.json"
    alltrain_path = CONFIG_ROOT / "deform_dlo2_alltrain_refit_v2.json"
    official_path = CONFIG_ROOT / "deform_dlo2_official_eval_v2.json"
    longrun = _load(longrun_path.name)
    posterior = _load(posterior_path.name)
    fresh = _load(fresh_path.name)
    alltrain = _load(alltrain_path.name)
    official = _load(official_path.name)

    assert longrun["source_result"]["sha256"] == sha256_file(SOURCE_RESULT)
    assert posterior["parent_longrun_protocol"]["sha256"] == sha256_file(longrun_path)
    assert posterior["parent_source_result"]["sha256"] == sha256_file(SOURCE_RESULT)
    assert fresh["authorization"]["required_parent_protocol_sha256"] == sha256_file(
        longrun_path
    )
    assert posterior["fresh_confirmation"]["protocol_path"] == str(
        fresh_path.relative_to(REPOSITORY_ROOT)
    )
    assert alltrain["parent_source_protocol"]["sha256"] == sha256_file(fresh_path)
    assert official["parent_alltrain_protocol"]["sha256"] == sha256_file(alltrain_path)


def test_dlo2_initialization_smoke_binds_its_implementation() -> None:
    smoke = json.loads(INITIALIZATION_SMOKE.read_text(encoding="utf-8"))

    assert smoke["passed"] is True
    assert smoke["dlo2_source_read"] is False
    assert smoke["official_eval_read"] is False
    assert smoke["implementation"]["parser_sha256"] == sha256_file(
        REPOSITORY_ROOT / "src" / "bayesian_phystwin" / "deform_dlo_upstream.py"
    )
    assert smoke["implementation"]["runner_sha256"] == sha256_file(
        REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo_source.py"
    )
    assert smoke["implementation"]["verifier_sha256"] == sha256_file(
        REPOSITORY_ROOT / "scripts" / "remote" / "check_deform_dlo_initialization.py"
    )


def test_dlo2_reference_operator_audit_is_target_free() -> None:
    audit = json.loads(REFERENCE_OPERATOR_AUDIT.read_text(encoding="utf-8"))
    boundary = audit["information_boundary"]

    assert audit["paper"]["metric"] == "average-l1-over-500-step-prediction"
    assert audit["released_loader"]["eval_draw"]["unique_index_count"] == 9
    assert (
        audit["training_budget_audit"]["released_upstream"]["nominal_total_updates"]
        == 69800
    )
    assert audit["training_budget_audit"]["locked_longrun_v2"]["total_updates"] == 6400
    assert all(value is False for value in boundary.values())


def test_deform_release_chain_preserves_the_selected_method_family() -> None:
    longrun = _load("deform_dlo_longrun_v2.json")
    posterior = _load("deform_dlo_longrun_posterior_v2.json")
    fresh = _load("deform_dlo2_fresh_v2.json")
    alltrain = _load("deform_dlo2_alltrain_refit_v2.json")
    official = _load("deform_dlo2_official_eval_v2.json")
    longrun_policy = longrun["checkpoint_posterior_if_source_gate_passes"]
    fresh_policy = fresh["checkpoint_posterior"]

    assert posterior["operators"] == fresh_policy["operators"]
    assert posterior["benchmark_point_loss"] == fresh_policy["benchmark_point_loss"]
    assert (
        posterior["predictive_median_definition"]
        == fresh_policy["predictive_median_definition"]
    )
    assert posterior["fallback"] == fresh_policy["fallback"]
    assert posterior["softmax_temperature_m"] == fresh_policy["softmax_temperature_m"]
    assert _arm_signature(posterior) == _arm_signature(fresh_policy)
    assert _arm_signature(longrun_policy) == _arm_signature(fresh_policy)
    assert fresh["training"]["total_updates"] == alltrain["training"]["total_updates"]
    assert (
        fresh["training"]["checkpoint_updates"]
        == alltrain["training"]["checkpoint_updates"]
    )
    assert fresh["training"]["batch_size"] == alltrain["training"]["batch_size"]
    assert (
        fresh["training"]["unroll_horizon_frames"]
        == alltrain["training"]["unroll_horizon_frames"]
    )
    assert (
        fresh["model_initialization"]
        == alltrain["model_initialization"]
        == official["model_initialization"]
        == "official-deform-dlo-initialization-v1"
    )
    assert alltrain["method_transfer"]["target_reselection"] is False
    assert official["methods"]["target_selection"] is False
    assert official["methods"]["target_calibration"] is False
    assert official["methods"]["target_retries"] is False
