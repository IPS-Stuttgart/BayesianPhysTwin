"""Contracts for the target-free Deform360 metric-gauge scale amendment."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from bayesian_phystwin.deform360_fresh_object_session_public_inputs_v6_1 import (
    METRIC_GAUGE_CLUSTER_SIZE_PIXELS_V6_1,
    METRIC_GAUGE_SCALE_AMENDMENT_ID_V6_1,
    prepare_deform360_disjoint_visual_window_v6_1,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = (
    ROOT / "protocols" / "amendments" / "deform360_metric_gauge_scale_amendment_v1.json"
)


def _canonical_id(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_amendment_identity_and_information_boundary() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("amendment_id")
    assert declared == _canonical_id(payload)
    assert declared == METRIC_GAUGE_SCALE_AMENDMENT_ID_V6_1
    boundary = payload["audit"]["information_boundary"]
    assert boundary["source_prefix_arrays_opened"]
    assert not boundary["source_suffix_opened"]
    assert not boundary["confirmation_payload_opened"]
    assert not boundary["target_outcome_opened"]
    assert not payload["next_stage"]["confirmation_authorized"]
    assert not payload["next_stage"]["paper_claim_authorized"]


def test_selected_scale_is_coarsest_universal_source_scale() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    summaries = payload["scale_summaries"]
    passing = sorted(
        int(size)
        for size, record in summaries.items()
        if record["all_cameras_pass"] and record["all_units_pass"]
    )
    assert passing == [4, 8]
    assert payload["selected_metric_cluster_size_pixels"] == max(passing) == 8
    assert summaries["32"]["camera_pass_count"] == 41
    assert summaries["32"]["unit_full_panel_pass_count"] == 0
    assert summaries["8"]["camera_pass_count"] == 80
    assert summaries["8"]["unit_full_panel_pass_count"] == 10


def test_v61_adapter_default_is_bound_to_the_amendment() -> None:
    parameter = inspect.signature(
        prepare_deform360_disjoint_visual_window_v6_1
    ).parameters["metric_cluster_size_pixels"]
    assert parameter.default == METRIC_GAUGE_CLUSTER_SIZE_PIXELS_V6_1 == 8
