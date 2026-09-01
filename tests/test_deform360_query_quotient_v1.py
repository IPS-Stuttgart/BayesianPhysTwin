from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deform360_query_quotient_v1.run import (
    analyze_sequences,
    EpisodeData,
    load_protocol,
    query_class_index,
    validate_result,
)


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "deform360_query_quotient_v1"
    / "protocol.json"
)


def _synthetic_episode(episode_id: int, action: str, rho: float) -> EpisodeData:
    frames = 96
    points = 16
    direction = np.linspace(0.5, 1.5, points)[:, None] * np.array(
        [0.0010, -0.0004, 0.0003]
    )
    positions = np.zeros((frames, points, 3), dtype=np.float64)
    positions[0] = np.linspace(-0.01, 0.01, points)[:, None] * np.array(
        [1.0, 0.5, -0.25]
    )
    velocity = direction.copy()
    phase = 0.17 * episode_id
    for frame in range(1, frames):
        forcing = 0.08 * direction * np.sin(0.23 * frame + phase)
        velocity = rho * velocity + forcing
        positions[frame] = positions[frame - 1] + velocity
    return EpisodeData(
        episode_id=episode_id,
        action=action,
        positions_m=positions,
        archive_metadata={"synthetic": True},
    )


def _sequences(protocol: dict[str, object]) -> list[EpisodeData]:
    rhos = (0.58, 0.66, 0.73, 0.81, 0.90, 0.98)
    return [
        _synthetic_episode(
            episode_id,
            str(protocol["source_episode_actions"][str(episode_id)]),  # type: ignore[index]
            rho,
        )
        for episode_id, rho in zip(
            protocol["source_episode_ids"], rhos, strict=True  # type: ignore[arg-type]
        )
    ]


def test_protocol_and_query_partition_are_frozen() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    grid = np.linspace(-0.5, 1.25, 71)
    classes = query_class_index(grid, protocol)

    np.testing.assert_array_equal(np.unique(classes), np.array([0, 1, 2]))
    assert tuple(protocol["source_episode_ids"]) == (0, 2, 5, 6, 7, 9)
    assert tuple(protocol["forbidden_episode_ids"]) == (1, 3, 4, 8)
    assert not set(protocol["source_episode_ids"]) & set(
        protocol["forbidden_episode_ids"]
    )
    assert protocol["information_boundary"]["paper_claim_authorized"] is False


def test_synthetic_source_episodes_preserve_quotient_and_expose_specificity() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    summary = analyze_sequences(_sequences(protocol), protocol)

    assert summary["episode_count"] == 6
    assert summary["reset_count"] == 30
    assert summary["query_class_count"] == 3
    assert summary["jeffrey_unsupported_specificity_nats"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert summary["mean_full_posterior_unsupported_specificity_nats"] > 0.0
    assert 0.0 <= summary["latent_decision_ambiguity_fraction"] <= 1.0
    assert 0.0 <= summary["complete_lift_decision_disagreement_fraction"] <= 1.0

    for reset in summary["reset_records"]:
        quotient = np.asarray(reset["posterior_quotient_weights"])
        assert np.sum(quotient) == pytest.approx(1.0)
        assert len(reset["lifts"]) == 5
        for lift in reset["lifts"]:
            np.testing.assert_allclose(lift["quotient_weights"], quotient, atol=1e-12)
        jeffrey = reset["lifts"][0]
        assert jeffrey["name"] == "jeffrey_i_projection"
        assert jeffrey["unsupported_specificity_nats"] == pytest.approx(
            0.0, abs=1e-10
        )


def test_protocol_rejects_any_opened_forbidden_episode(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["information_boundary"]["forbidden_episode_payloads_opened"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="widens a closed information boundary"):
        load_protocol(path)


def test_result_validator_rejects_self_authorized_claim() -> None:
    result = {
        "schema_version": 1,
        "artifact_kind": "Deform360QueryQuotientRealPilotV1",
        "information_boundary": {
            "source_only": True,
            "forbidden_episode_payloads_opened": False,
            "official_velocity_arrays_used": False,
            "dataset_modified": False,
            "raw_payload_uploaded": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "summary": {
            "episode_count": 6,
            "reset_count": 30,
            "jeffrey_unsupported_specificity_nats": 0.0,
        },
        "paper_claim_authorized": True,
        "result_sha256": "not-used-before-claim-check",
    }

    with pytest.raises(ValueError, match="self-authorized"):
        validate_result(result)
