import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_longrun import (
    load_deform_dlo_longrun_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_longrun_v2.json"
SOURCE_RESULT = (
    REPOSITORY_ROOT / "results" / "sota" / "deform_dlo_source_v1" / "source_result.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_longrun_binds_failed_source_without_reopening_eval() -> None:
    protocol = load_deform_dlo_longrun_protocol(PROTOCOL)
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))

    assert protocol["source_result"]["sha256"] == _sha256(SOURCE_RESULT)
    assert protocol["source_result"]["required_advancement_authorized"] is False
    assert source_result["advancement_authorized"] is False
    assert (
        protocol["starting_checkpoint"]["sha256"]
        == source_result["selected_checkpoint"]["checkpoint"]["sha256"]
    )
    assert protocol["source_test_status"] == "post-open-exploratory-only"
    assert protocol["official_eval_policy"] == "forbidden"
    assert protocol["fresh_confirmation_dlo"] == "DLO2"


def test_longrun_update_budget_and_posterior_arms_are_frozen() -> None:
    protocol = load_deform_dlo_longrun_protocol(PROTOCOL)
    training = protocol["training"]

    assert (
        protocol["starting_checkpoint"]["global_update"]
        + training["continuation_updates"]
        == training["final_global_update"]
    )
    assert training["checkpoint_updates"][-2:] == (6040, 6400)
    posterior = protocol["checkpoint_posterior_if_source_gate_passes"]
    assert posterior["validation_improvement_min"] == pytest.approx(0.01)
    assert posterior["fallback"] == "selected_single_exact"
    assert posterior["arms"][0]["updates"] == [6040, 6400]


def test_longrun_rejects_official_eval_authorization(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["official_eval_policy"] = "allowed"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbid official evaluation"):
        load_deform_dlo_longrun_protocol(changed)


def test_longrun_rejects_dlo1_confirmation(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["fresh_confirmation_dlo"] = "DLO1"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh DLO2"):
        load_deform_dlo_longrun_protocol(changed)
