import json
from pathlib import Path

from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

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


def test_dlo2_initialization_smoke_preserves_its_historical_implementation() -> None:
    smoke = json.loads(INITIALIZATION_SMOKE.read_text(encoding="utf-8"))

    assert smoke["passed"] is True
    assert smoke["dlo2_source_read"] is False
    assert smoke["official_eval_read"] is False
    assert smoke["implementation"]["parser_sha256"] == sha256_file(
        REPOSITORY_ROOT
        / "src"
        / "bayesian_phystwin_experiments"
        / "deform_dlo_upstream.py"
    )
    # The smoke sealed runner revision f8e9e3af and its companion verifier.
    # Historical evidence must not be rebound to later repository paths or bytes.
    assert smoke["implementation"]["runner_sha256"] == (
        "d5626377a6028133791b6f89b4aee02ba2444b1222177cc3599b743b55daae67"
    )
    assert smoke["implementation"]["verifier_sha256"] == (
        "311b8e7a84037b021e796ac34ff483ab8b72abbf27acfdd73a86d3d5490714c5"
    )


def test_dlo2_reference_operator_contract_is_public_and_target_blind() -> None:
    official = _load("deform_dlo2_official_eval_v2.json")
    evaluation = official["evaluation"]
    operator = evaluation["published_reference_operator"]

    assert evaluation["metric"] == "mean-coordinate-l1-m"
    assert evaluation["published_reference_l1_m"] == 0.0097
    assert operator["preceding_train_population"] == 56
    assert operator["preceding_train_draw_count"] == 56
    assert operator["eval_population"] == 14
    assert operator["eval_draw_count"] == 14
    assert operator["canonical_unique_index_count"] == 9
    assert operator["canonical_eval_indices"] == [
        1,
        7,
        9,
        7,
        11,
        7,
        13,
        8,
        8,
        6,
        8,
        5,
        8,
        4,
    ]
    assert operator["upstream_glob_order"] == "unspecified"
    assert official["methods"]["target_selection"] is False
    assert official["methods"]["target_calibration"] is False
    assert official["methods"]["target_retries"] is False
    assert official["methods"]["case_replacement"] is False


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
