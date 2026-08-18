import json
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.deform_dlo_robustness import (
    assign_deform_dlo3_source_partitions,
    load_deform_dlo_robustness_v1_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "sota" / "deform_dlo_robustness_v1.json"


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_loads_locked_dlo_robustness_protocol() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)

    assert protocol["prob4d_used"] is False
    assert protocol["freshness"]["primary_dlo"] == "DLO3"
    assert protocol["custody"]["held_v8_access"] is False


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("freshness", "primary_dlo"), "DLO4", "data boundary"),
        (("physical_training", "primary_seed"), 43, "fixed recipe"),
        (("local_residual", "shrinkage"), 0.5, "fixed recipe"),
        (("source_gate", "minimum_case_wins"), 5, "source gates"),
        (("backend_portability", "version"), "latest", "backend contract"),
        (("target_evaluation", "target_retries"), True, "Bayesian or target"),
        (("custody", "held_v8_access"), True, "Bayesian or target"),
    ],
)
def test_rejects_protocol_mutation(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    payload = _payload()
    payload[path[0]][path[1]] = value
    mutated = tmp_path / "protocol.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_deform_dlo_robustness_v1_protocol(mutated)


def test_source_assignment_is_order_independent_and_disjoint() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    names = [f"trajectory_{index:02d}.pkl" for index in range(56)]

    forward = assign_deform_dlo3_source_partitions(names, protocol)
    reverse = assign_deform_dlo3_source_partitions(list(reversed(names)), protocol)

    assert forward == reverse
    assert forward["payload_read"] is False
    fit = set(forward["fit"])
    calibration = set(forward["calibration"])
    source_test = set(forward["source_test"])
    assert (len(fit), len(calibration), len(source_test)) == (39, 9, 8)
    assert fit.isdisjoint(calibration | source_test)
    assert calibration.isdisjoint(source_test)
    assert fit | calibration | source_test == set(names)


def test_source_assignment_rejects_non_basename_or_wrong_count() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    names = [f"trajectory_{index:02d}.pkl" for index in range(56)]

    with pytest.raises(ValueError, match="incomplete"):
        assign_deform_dlo3_source_partitions(names[:-1], protocol)
    names[0] = "nested/trajectory_00.pkl"
    with pytest.raises(ValueError, match="basename"):
        assign_deform_dlo3_source_partitions(names, protocol)
