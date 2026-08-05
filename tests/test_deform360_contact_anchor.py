from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.deform360_contact_anchor import (
    DEFORM360_CONTACT_ANCHOR_SEMANTICS,
    DEFORM360_CONTACT_ANCHOR_UNITS,
    DEFORM360_TACTILE_SOURCE_UNITS,
    Deform360ContactAnchorV1,
    attach_deform360_contact_anchor,
)
from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareObservationBatch,
    update_gauge_aware_belief,
)


def _design(values: np.ndarray) -> np.ndarray:
    result = np.zeros((len(values), 3, 1), dtype=np.float64)
    result[:, 0, 0] = values
    return result


def _anchor(**updates: Any) -> Deform360ContactAnchorV1:
    count = 2
    values: dict[str, Any] = {
        "object_id": "200-fresh-object",
        "episode_id": 3,
        "causal_frame_stop": 8,
        "sensor_names": ("tactile-a", "tactile-a"),
        "frame_ids": np.asarray([4, 5], dtype=np.int64),
        "innovation_m": np.asarray([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        "covariance_m2": np.repeat(np.eye(3)[None] * 1e-8, count, axis=0),
        "state_jacobian": _design(np.ones(count)),
        "correlation_group_ids": ("tactile-a:contact-0",) * count,
        "source_revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
        "source_artifacts": {
            "raw/200-fresh-object/tactile.npy": "1" * 64,
            "raw/200-fresh-object/tactile.txt": "2" * 64,
        },
        "metadata": {
            "processing_revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
            "source_taxel_shape": [16, 32],
        },
    }
    values.update(updates)
    return Deform360ContactAnchorV1(**values)


def _visual_batch(*, mode: np.ndarray | None = None) -> GaugeAwareObservationBatch:
    visual_mode = np.ones(10) if mode is None else mode
    count = len(visual_mode)
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.03 * visual_mode
    empty = np.zeros((count, 3, 0), dtype=np.float64)
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.repeat(np.eye(3)[None] * 1e-6, count, axis=0),
        state_jacobian=_design(visual_mode),
        gauge_jacobian=_design(visual_mode),
        shared_bias_jacobian=empty,
        view_bias_jacobian=empty,
        query_state_jacobian=_design(visual_mode),
        gauge_prior_covariance=np.asarray([[0.01]]),
        correlation_group_ids=("camera-window",) * count,
        prior_reliability=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[0.01]]),
        metadata={
            "observation_artifact_id": "a" * 64,
            "observation_causal_frame_stop": 8,
        },
    )


def test_contact_anchor_is_content_addressed_owned_and_immutable() -> None:
    innovation = np.asarray([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
    metadata = {"nested": {"values": [1, 2]}}
    anchor = _anchor(innovation_m=innovation, metadata=metadata)
    artifact_id = anchor.artifact_id

    innovation[0, 0] = 99.0
    metadata["nested"]["values"].append(3)

    assert anchor.artifact_id == artifact_id
    assert anchor.innovation_m[0, 0] == pytest.approx(0.01)
    assert anchor.metadata["nested"]["values"] == [1, 2]
    assert not anchor.innovation_m.flags.writeable
    with pytest.raises(TypeError, match="immutable"):
        anchor.metadata["nested"]["values"].append(3)

    summary = anchor.summary()
    assert summary["artifact_id"] == artifact_id
    assert summary["semantics"] == DEFORM360_CONTACT_ANCHOR_SEMANTICS
    assert summary["source_units"] == DEFORM360_TACTILE_SOURCE_UNITS
    assert summary["anchor_units"] == DEFORM360_CONTACT_ANCHOR_UNITS
    assert summary["raw_taxels_used_as_independent_rows"] is False
    assert summary["camera_gauge_present_in_anchor"] is False
    assert anchor.row_count == 2
    assert anchor.state_count == 1


def test_contact_anchor_identity_changes_with_source_provenance() -> None:
    first = _anchor()
    second = _anchor(
        source_artifacts={
            "raw/200-fresh-object/tactile.npy": "3" * 64,
            "raw/200-fresh-object/tactile.txt": "2" * 64,
        }
    )

    assert first.artifact_id != second.artifact_id


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"object_id": ""}, "object_id"),
        ({"episode_id": True}, "episode_id"),
        ({"causal_frame_stop": 0}, "causal_frame_stop"),
        ({"frame_ids": np.asarray([4.0, 5.0])}, "frame_ids"),
        ({"frame_ids": np.asarray([4, 8])}, "post-cutoff"),
        ({"sensor_names": ("tactile-a",)}, "sensor_names"),
        ({"correlation_group_ids": ("group",)}, "correlation_group_ids"),
        ({"innovation_m": np.zeros((2, 2))}, "innovation_m"),
        ({"covariance_m2": np.zeros((2, 3, 2))}, "covariance_m2"),
        ({"state_jacobian": np.zeros((2, 3, 0))}, "state_jacobian"),
        (
            {
                "covariance_m2": np.asarray(
                    [np.diag([1e-8, 1e-8, 0.0]), np.eye(3) * 1e-8]
                )
            },
            "positive definite",
        ),
        ({"prior_reliability": np.asarray([1.1, 1.0])}, "prior_reliability"),
        ({"prior_nominal_probability": np.asarray([-0.1, 1.0])}, "nominal"),
        ({"composite_weight": np.asarray([0.0, 1.0])}, "composite_weight"),
        ({"source_revision": "not-a-revision"}, "source_revision"),
        ({"source_artifacts": {}}, "source_artifacts"),
        ({"source_artifacts": {"path": "bad"}}, "SHA-256"),
        ({"bias_prior_covariance": np.eye(1)}, "requires bias_jacobian"),
        ({"bias_jacobian": np.zeros((2, 3, 1))}, "requires bias_prior"),
        (
            {
                "bias_jacobian": np.zeros((2, 3, 1)),
                "bias_prior_covariance": np.zeros((2, 2)),
            },
            "changed shape",
        ),
        (
            {
                "bias_jacobian": np.zeros((2, 3, 1)),
                "bias_prior_covariance": np.asarray([[-1.0]]),
            },
            "positive semidefinite",
        ),
        ({"metadata": {"bad": float("nan")}}, "finite JSON"),
    ],
)
def test_contact_anchor_rejects_malformed_inputs(
    updates: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _anchor(**updates)


def test_optional_anchor_bias_is_bound_into_identity_and_summary() -> None:
    anchor = _anchor(
        bias_jacobian=_design(np.ones(2)),
        bias_prior_covariance=np.asarray([[0.01]]),
    )

    assert anchor.summary()["anchor_bias_parameter_count"] == 1
    assert not anchor.bias_jacobian.flags.writeable
    assert not anchor.bias_prior_covariance.flags.writeable


@dataclass(frozen=True)
class _FakeBatch:
    state_jacobian: np.ndarray
    metadata: dict[str, object]
    anchor_innovation_m: np.ndarray | None = None
    anchor_covariance_m2: np.ndarray | None = None
    anchor_state_jacobian: np.ndarray | None = None
    anchor_correlation_group_ids: tuple[str, ...] | None = None
    anchor_prior_reliability: np.ndarray | None = None
    anchor_prior_nominal_probability: np.ndarray | None = None
    anchor_composite_weight: np.ndarray | None = None
    anchor_bias_jacobian: np.ndarray | None = None
    anchor_bias_prior_covariance: np.ndarray | None = None


def test_attach_contact_anchor_preserves_lineage_and_fails_closed() -> None:
    anchor = _anchor()
    batch = _FakeBatch(
        state_jacobian=np.zeros((3, 3, 1)),
        metadata={"observation_causal_frame_stop": 8},
    )

    attached = attach_deform360_contact_anchor(batch, anchor)

    assert attached.anchor_innovation_m is anchor.innovation_m
    assert attached.anchor_correlation_group_ids == anchor.correlation_group_ids
    assert attached.metadata["deform360_contact_anchor"]["artifact_id"] == (
        anchor.artifact_id
    )

    with pytest.raises(ValueError, match="already contains anchors"):
        attach_deform360_contact_anchor(attached, anchor)
    with pytest.raises(ValueError, match="state dimension"):
        attach_deform360_contact_anchor(
            replace(batch, state_jacobian=np.zeros((3, 3, 2))),
            anchor,
        )
    with pytest.raises(ValueError, match="causal cutoffs differ"):
        attach_deform360_contact_anchor(
            replace(batch, metadata={"observation_causal_frame_stop": 7}),
            anchor,
        )
    with pytest.raises(ValueError, match="lineage is already present"):
        attach_deform360_contact_anchor(
            replace(batch, metadata={"deform360_contact_anchor": {}}),
            anchor,
        )


def test_independent_contact_anchor_breaks_visual_gauge_ambiguity() -> None:
    batch = attach_deform360_contact_anchor(_visual_batch(), _anchor())

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=10.0,
            effective_samples_per_anchor_correlation_group=2.0,
        ),
    )

    assert result.inference_admissible
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=3e-4)
    assert result.gauge_delta[0] == pytest.approx(0.02, abs=3e-4)
    assert (
        result.input_lineage["deform360_contact_anchor"]["artifact_id"]
        == (batch.metadata["deform360_contact_anchor"]["artifact_id"])
    )


def test_shared_contact_sensor_bias_does_not_fake_identifiability() -> None:
    anchor = _anchor(
        bias_jacobian=_design(np.ones(2)),
        bias_prior_covariance=np.asarray([[0.01]]),
    )
    batch = attach_deform360_contact_anchor(_visual_batch(), anchor)

    result = update_gauge_aware_belief(batch)

    assert not result.inference_admissible
    assert result.reason == "no-identifiable-query-state"
