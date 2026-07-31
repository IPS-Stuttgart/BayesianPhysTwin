import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from bayesian_phystwin.deform_dlo_deep_ensemble import (
    DEFORM_DLO_DEEP_ENSEMBLE_CONTRACT,
    build_deform_two_seed_weights,
    load_deform_dlo1_deep_ensemble_protocol,
    validate_deform_two_seed_manifests,
)
from bayesian_phystwin.deform_dlo_source import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo1_deep_ensemble_eval_v1.json"
)
RUNNER = REPOSITORY_ROOT / "scripts" / "remote" / "run_deform_dlo_deep_ensemble.py"


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
