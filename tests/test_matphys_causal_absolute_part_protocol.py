import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "matphys_causal_absolute_part_competence_v1.json"
)
AMENDMENT = (
    ROOT
    / "configs"
    / "sota"
    / "matphys_causal_absolute_part_competence_v1_amendment.json"
)
STAGE_ZERO_GATE = (
    ROOT
    / "results"
    / "sota"
    / "matphys_causal_absolute_part_stage0_v1_1"
    / "stage0_gate.json"
)
STAGE_ONE_DECISION = (
    ROOT
    / "results"
    / "sota"
    / "matphys_causal_absolute_part_competence_v1"
    / "decision.json"
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


def test_stage_zero_amendment_allows_one_mechanical_retry_only() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert amendment["protocol_id"] == (
        "matphys-causal-absolute-part-competence-v1.1"
    )
    assert amendment["base_protocol"]["sha256"] == (
        "91e17a9fb4fbd5fe85b456e074d68c7e206c55f3609e5fd57d97774c6e87a616"
    )
    failed = amendment["failed_attempt"]
    assert failed["checkpoint_created"] is False
    assert failed["future_metrics_opened"] is False
    assert failed["exit_code"] == 1
    retry = amendment["stage_0_retry"]
    assert retry["directory"] == "stage0-v1.1"
    assert retry["epochs"] == 1
    assert retry["maximum_runs"] == 1
    assert retry["future_metrics_may_be_opened"] is False
    assert "unauthorized" in amendment["stage_1_causal_competence"][
        "authorization"
    ]


def test_stage_zero_gate_binds_causal_mechanical_pass() -> None:
    gate = json.loads(STAGE_ZERO_GATE.read_text(encoding="utf-8"))

    assert gate["implementation_revision"] == (
        "4bef7408390ccd5b6533e490873ae5929366b270"
    )
    assert gate["custody"]["future_metrics_opened"] is False
    assert gate["causal_audit"]["maximum_accessed_frame"] == 33
    assert gate["causal_audit"]["evidence_end_frame_exclusive"] == 34
    assert gate["optimizer"]["attempted_steps"] == 33
    assert gate["optimizer"]["accepted_steps"] == 33
    assert gate["optimizer"]["rejected_pre_step"] == 0
    assert gate["optimizer"]["rejected_post_step"] == 0
    assert gate["spring_field"]["finite"] is True
    assert gate["spring_field"]["positive"] is True
    assert gate["spring_field"]["distinct_part_count"] == 5
    assert gate["gate"]["passed"] is True
    assert gate["gate"]["authorize_stage_1"] is True


def test_stage_one_decision_closes_failed_absolute_prefix_family() -> None:
    decision = json.loads(STAGE_ONE_DECISION.read_text(encoding="utf-8"))

    assert decision["prediction_seal"]["sha256"] == (
        "a39cbc7f61a714b7da10f59d73b63fa93e9482a83fe3594be5d72ebcf425be3a"
    )
    assert decision["registered_future_interval"] == [46, 66]
    assert decision["gates"]["physical_gate_pass"] is True
    assert decision["gates"]["metric_gate_pass"] is False
    assert decision["gates"]["competence_pass"] is False
    assert decision["metrics_m"]["chamfer_distance_m"][
        "improvement_percent"
    ] == -101.00801954281074
    assert decision["metrics_m"]["track_error_m"][
        "improvement_percent"
    ] == -43.14219277419853
    assert decision["sota_headroom_diagnostic"]["joint_8_15_pass"] is False
    assert decision["decision"] == (
        "close_absolute_prefix_family_without_tuning_on_case_outcome"
    )
