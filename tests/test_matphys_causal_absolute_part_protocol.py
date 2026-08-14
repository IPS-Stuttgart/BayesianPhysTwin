import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "matphys_causal_absolute_part_competence_v1.json"
)


def test_absolute_part_competence_protocol_is_causal_and_single_run() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == "matphys-causal-absolute-part-competence-v1"
    assert protocol["case"]["fit_interval"] == [0, 34]
    assert protocol["case"]["released_test_interval"] == [46, 66]
    assert protocol["method"]["published_matphys_method"] is False
    assert protocol["method"]["teacher_centered"] is False
    assert protocol["method"]["prob4d"] == "unused"
    assert protocol["method"]["molmomotion_beta"] == 0
    assert protocol["stage_0_mechanical_smoke"] == {
        **protocol["stage_0_mechanical_smoke"],
        "epochs": 1,
        "maximum_runs": 1,
        "future_metrics_may_be_opened": False,
    }
    assert protocol["stage_1_causal_competence"]["epochs"] == 200
    assert protocol["stage_1_causal_competence"]["maximum_runs"] == 1
    forbidden = " ".join(protocol["forbidden"])
    assert "at or after frame 34" in forbidden
    assert "held-v8" in forbidden
    assert "fresh target" in forbidden
