import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.tapnextpp_material_transport_staging import (
    PROVIDER_PROTOCOL_ID,
    plan_material_transport_case,
    validate_material_transport_provider_protocol,
)


def _protocol() -> dict:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_tapnextpp_material_transport_provider_source_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_provider_panel_is_disjoint_and_frozen() -> None:
    protocol = _protocol()

    validate_material_transport_provider_protocol(protocol)

    assert protocol["protocol_id"] == PROVIDER_PROTOCOL_ID
    assert len(protocol["fixed_source_cases"]) == 14
    assert not set(protocol["fixed_source_cases"]).intersection(
        protocol["excluded_prior_assimilation_cases"]
    )


def test_case_plan_uses_terminal_window_and_frame_zero_nodes() -> None:
    protocol = _protocol()
    case_name = protocol["fixed_source_cases"][0]
    tracks = np.zeros((30, 5, 3), dtype=float)
    tracks[:, :, 0] = np.arange(5)[None] * 0.01
    tracks[10:, :, 1] = np.arange(20)[:, None] * 0.001
    physical = np.zeros((40, 8, 3), dtype=float)
    physical[:, :, 0] = np.arange(8)[None] * 0.01

    plan = plan_material_transport_case(
        case_name,
        tracks,
        physical,
        train_end_frame_exclusive=30,
        protocol=protocol,
    )

    assert plan.tracker_config.source_frame_start == 10
    assert plan.tracker_config.source_frame_end_exclusive == 30
    assert len(plan.tracker_config.selected_identity_ids) == 4
    selected = np.asarray(plan.tracker_config.selected_identity_ids)
    np.testing.assert_array_equal(
        plan.material_node_indices,
        selected,
    )


def test_case_plan_rejects_identity_without_frame_zero_attachment() -> None:
    protocol = _protocol()
    protocol["selection"]["maximum_frame_zero_attachment_m"] = 0.001
    tracks = np.zeros((30, 4, 3), dtype=float)
    tracks[:, :, 0] = 0.1 + np.arange(4)[None] * 0.01
    physical = np.zeros((30, 4, 3), dtype=float)

    try:
        plan_material_transport_case(
            protocol["fixed_source_cases"][0],
            tracks,
            physical,
            train_end_frame_exclusive=30,
            protocol=protocol,
        )
    except ValueError as error:
        assert "too far" in str(error)
    else:
        raise AssertionError("invalid material attachment was accepted")
