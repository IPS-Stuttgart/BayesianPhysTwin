import json
from pathlib import Path

from bayesian_phystwin.deform_dlo_source import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "sota"
SOURCE_RESULT = (
    REPOSITORY_ROOT / "results" / "sota" / "deform_dlo_source_v1" / "source_result.json"
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
    posterior_path = CONFIG_ROOT / "deform_dlo_longrun_posterior_v1.json"
    fresh_path = CONFIG_ROOT / "deform_dlo2_fresh_v1.json"
    alltrain_path = CONFIG_ROOT / "deform_dlo2_alltrain_refit_v1.json"
    official_path = CONFIG_ROOT / "deform_dlo2_official_eval_v1.json"
    longrun = _load(longrun_path.name)
    posterior = _load(posterior_path.name)
    fresh = _load(fresh_path.name)
    alltrain = _load(alltrain_path.name)
    official = _load(official_path.name)

    assert longrun["source_result"]["sha256"] == sha256_file(SOURCE_RESULT)
    assert posterior["parent_longrun_protocol"]["sha256"] == sha256_file(
        longrun_path
    )
    assert posterior["parent_source_result"]["sha256"] == sha256_file(
        SOURCE_RESULT
    )
    assert fresh["authorization"]["required_parent_protocol_sha256"] == sha256_file(
        longrun_path
    )
    assert posterior["fresh_confirmation"]["protocol_path"] == str(
        fresh_path.relative_to(REPOSITORY_ROOT)
    )
    assert alltrain["parent_source_protocol"]["sha256"] == sha256_file(fresh_path)
    assert official["parent_alltrain_protocol"]["sha256"] == sha256_file(
        alltrain_path
    )


def test_deform_release_chain_preserves_the_selected_method_family() -> None:
    longrun = _load("deform_dlo_longrun_v2.json")
    posterior = _load("deform_dlo_longrun_posterior_v1.json")
    fresh = _load("deform_dlo2_fresh_v1.json")
    alltrain = _load("deform_dlo2_alltrain_refit_v1.json")
    official = _load("deform_dlo2_official_eval_v1.json")
    longrun_policy = longrun["checkpoint_posterior_if_source_gate_passes"]
    fresh_policy = fresh["checkpoint_posterior"]

    assert posterior["operators"] == fresh_policy["operators"]
    assert posterior["fallback"] == fresh_policy["fallback"]
    assert posterior["softmax_temperature_m"] == fresh_policy[
        "softmax_temperature_m"
    ]
    assert _arm_signature(posterior) == _arm_signature(fresh_policy)
    assert _arm_signature(longrun_policy) == _arm_signature(fresh_policy)
    assert fresh["training"]["total_updates"] == alltrain["training"][
        "total_updates"
    ]
    assert fresh["training"]["checkpoint_updates"] == alltrain["training"][
        "checkpoint_updates"
    ]
    assert fresh["training"]["batch_size"] == alltrain["training"]["batch_size"]
    assert fresh["training"]["unroll_horizon_frames"] == alltrain["training"][
        "unroll_horizon_frames"
    ]
    assert alltrain["method_transfer"]["target_reselection"] is False
    assert official["methods"]["target_selection"] is False
    assert official["methods"]["target_calibration"] is False
    assert official["methods"]["target_retries"] is False
