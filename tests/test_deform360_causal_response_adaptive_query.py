from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    ABSTAINED_ARM,
    ARCHIVE_FILENAME,
    INFLATED_FALLBACK_ARM,
    REPORT_FILENAME,
    STRICT_ARM,
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
    select_adaptive_camera_panels,
    validate_adaptive_causal_response_query_artifacts,
    write_adaptive_causal_response_query_artifacts,
)


def _inputs(*, camera_count: int = 10, node_count: int = 24) -> tuple[object, ...]:
    height = width = 96
    coordinate = np.linspace(-1.0, 1.0, node_count)
    frame_zero = np.column_stack(
        (
            0.18 * coordinate,
            0.04 * np.sin(np.pi * coordinate),
            np.full(node_count, 2.0),
        )
    )
    graph_basis = np.zeros((node_count, 3, 16), dtype=np.float64)
    for mode in range(16):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    action_support = np.linspace(0.05, 1.0, node_count)
    intrinsics = np.repeat(np.eye(3)[None], camera_count, axis=0)
    intrinsics[:, 0, 0] = 60.0
    intrinsics[:, 1, 1] = 60.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    poses = np.repeat(np.eye(4)[None], camera_count, axis=0)
    angle = np.linspace(0.0, 2.0 * np.pi, camera_count, endpoint=False)
    poses[:, 0, 3] = 0.08 * np.cos(angle)
    poses[:, 1, 3] = 0.08 * np.sin(angle)
    depth = np.full((camera_count, height, width), 2.0)
    masks = np.ones_like(depth, dtype=bool)
    cameras = tuple(f"camera-{index:02d}" for index in range(camera_count))
    return (
        frame_zero,
        graph_basis,
        action_support,
        intrinsics,
        poses,
        depth,
        masks,
        cameras,
    )


def _build(
    inputs: tuple[object, ...],
    *,
    config: AdaptiveCausalResponseQueryConfig | None = None,
):
    return build_adaptive_causal_response_query_schedule(
        *inputs[:-1],
        camera_ids=inputs[-1],
        config=config
        or AdaptiveCausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )


def test_panel_selection_is_deterministic_and_prefers_strict_coverage() -> None:
    valid = np.zeros((10, 20), dtype=bool)
    valid[:8] = True
    probability = valid.astype(np.float64)
    action_support = np.ones(20)
    config = AdaptiveCausalResponseQueryConfig(query_count=8)

    first = select_adaptive_camera_panels(
        valid,
        probability,
        action_support,
        config=config,
    )
    second = select_adaptive_camera_panels(
        valid,
        probability,
        action_support,
        config=config,
    )

    np.testing.assert_array_equal(first.proposal_indices, second.proposal_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    assert first.strict_eligible_count == 20
    assert first.fallback_eligible_count == 20
    assert tuple(first.selected_indices) == tuple(range(8))


def test_strict_arm_is_preferred_when_it_fills_the_budget() -> None:
    schedule = _build(_inputs())

    assert schedule.admitted
    assert schedule.arm == STRICT_ARM
    assert schedule.covariance_inflation == 1.0
    assert len(schedule.selected_camera_ids) == 8
    assert not np.intersect1d(
        schedule.panels.proposal_indices,
        schedule.panels.validation_indices,
    ).size


def test_two_plus_two_fallback_is_explicitly_covariance_inflated() -> None:
    inputs = list(_inputs(camera_count=8, node_count=20))
    masks = np.asarray(inputs[6]).copy()
    masks[[2, 3, 6, 7]] = False
    inputs[6] = masks
    schedule = _build(
        tuple(inputs),
        config=AdaptiveCausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )

    assert schedule.admitted
    assert schedule.arm == INFLATED_FALLBACK_ARM
    assert schedule.covariance_inflation == 4.0
    assert schedule.descriptor()["shared_bias_variance_m2"] == 0.005**2
    assert schedule.panels.strict_eligible_count < 8
    assert schedule.panels.fallback_eligible_count >= 8


def test_insufficient_two_plus_two_support_abstains_exactly() -> None:
    inputs = list(_inputs(camera_count=8, node_count=20))
    masks = np.asarray(inputs[6]).copy()
    masks[[1, 2, 3, 5, 6, 7]] = False
    inputs[6] = masks
    schedule = _build(
        tuple(inputs),
        config=AdaptiveCausalResponseQueryConfig(
            query_count=8,
            graph_basis_rank=8,
        ),
    )

    assert not schedule.admitted
    assert schedule.arm == ABSTAINED_ARM
    assert not len(schedule.query_schedule.entity_ids)
    assert schedule.panels.fallback_eligible_count < 8


def test_fallback_cannot_claim_independent_strength() -> None:
    config = AdaptiveCausalResponseQueryConfig()

    with pytest.raises(ValueError, match="at least fourfold"):
        AdaptiveCausalResponseQueryConfig(fallback_covariance_inflation=3.99)
    assert config.fallback_covariance_inflation >= 4.0
    assert (
        config.fallback_minimum_support_per_panel
        < config.strict_minimum_support_per_panel
    )


def test_adaptive_artifacts_round_trip_and_bind_boundary(tmp_path: Path) -> None:
    schedule = _build(_inputs())
    protocol = tmp_path / "protocol.json"
    physical_manifest = tmp_path / "physical.json"
    physical_archive = tmp_path / "physical.npz"
    for path, content in (
        (protocol, "{}\n"),
        (physical_manifest, "{}\n"),
        (physical_archive, "physical"),
    ):
        path.write_text(content, encoding="utf-8")
    output = tmp_path / "adaptive"

    report = write_adaptive_causal_response_query_artifacts(
        output,
        schedule,
        case_id="opened-source-case",
        repository_revision="a" * 40,
        protocol_path=protocol,
        physical_manifest_path=physical_manifest,
        physical_archive_path=physical_archive,
        camera_certificate_sha256="b" * 64,
    )
    loaded, arrays = validate_adaptive_causal_response_query_artifacts(output)

    assert loaded == report
    assert (output / ARCHIVE_FILENAME).is_file()
    assert (output / REPORT_FILENAME).is_file()
    np.testing.assert_array_equal(
        arrays["entity_ids"],
        schedule.query_schedule.entity_ids,
    )
    assert loaded["information_boundary"]["future_metric_read"] is False
    assert loaded["information_boundary"]["state_update_constructed"] is False
    assert loaded["information_boundary"]["tactile_read"] is False


def test_adaptive_validator_detects_report_tampering(tmp_path: Path) -> None:
    schedule = _build(_inputs())
    sources = [tmp_path / name for name in ("protocol", "manifest", "archive")]
    for path in sources:
        path.write_text("source", encoding="utf-8")
    output = tmp_path / "adaptive"
    write_adaptive_causal_response_query_artifacts(
        output,
        schedule,
        case_id="opened-source-case",
        repository_revision="a" * 40,
        protocol_path=sources[0],
        physical_manifest_path=sources[1],
        physical_archive_path=sources[2],
        camera_certificate_sha256="b" * 64,
    )
    report_path = output / REPORT_FILENAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["information_boundary"]["future_metric_read"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        validate_adaptive_causal_response_query_artifacts(output)
