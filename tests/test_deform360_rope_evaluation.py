from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_contact import contact_artifact_sha256
from causal4d_public.deform360_rope_evaluation import (
    evaluate_held_out_rope_predictions,
    seal_held_out_rope_predictions,
    validate_held_out_rope_prediction_seal,
)


def _contact_seal() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360TargetContactPredictionSeal",
        "protocol_id": "fixture",
        "target_episode_id": "001-rope/episode_0006",
        "information_boundary": {"target_tactile_oracle_read": False},
    }
    payload["result_sha256"] = contact_artifact_sha256(payload)
    return payload


def _predictions() -> dict[str, np.ndarray]:
    time = np.linspace(0.0, 1.0, 12)
    nodes = np.linspace(-0.2, 0.2, 7)
    reference = np.zeros((len(time), len(nodes), 3), dtype=np.float64)
    reference[:, :, 0] = nodes
    reference[:, :, 2] = time[:, None] * np.linspace(0.0, 0.08, len(nodes))[None]
    return {
        "visual_only": reference + np.asarray((0.0, 0.010, 0.0)),
        "tactile_conditioned_z": reference + np.asarray((0.0, 0.004, 0.0)),
        "reference": reference,
    }


def test_prediction_seal_is_written_before_target_evaluation(tmp_path: Path) -> None:
    trajectories = _predictions()
    seal = seal_held_out_rope_predictions(
        tmp_path / "predictions.npz",
        {
            "visual_only": trajectories["visual_only"],
            "tactile_conditioned_z": trajectories["tactile_conditioned_z"],
        },
        protocol_id="fixture",
        contact_prediction_seal=_contact_seal(),
        shared_dynamics_fit_sha256="a" * 64,
        target_prefix_geometry_sha256="b" * 64,
        future_start_frame=109,
    )

    validation = validate_held_out_rope_prediction_seal(seal)
    assert validation["future_geometry_unlock_authorized"] is True
    assert seal["information_boundary"]["target_future_geometry_read"] is False
    result = evaluate_held_out_rope_predictions(
        trajectories["reference"],
        held_out_prediction_seal=seal,
        target_future_geometry_sha256="c" * 64,
        additional_predictions={
            "constant_persistence": np.repeat(
                trajectories["reference"][:1],
                len(trajectories["reference"]),
                axis=0,
            )
        },
    )
    assert result["paired_primary_difference_m"]["track_error_m"] < 0.0
    assert result["methods"]["tactile_conditioned_z"]["track_error_m"][
        "mean_m"
    ] == pytest.approx(0.004)
    assert "constant_persistence" in result["methods"]
    assert "constant_persistence" in result["additional_prediction_sha256"]


def test_prediction_seal_rejects_an_oracle_method(tmp_path: Path) -> None:
    trajectories = _predictions()
    with pytest.raises(ValueError, match="exactly visual_only"):
        seal_held_out_rope_predictions(
            tmp_path / "predictions.npz",
            trajectories,
            protocol_id="fixture",
            contact_prediction_seal=_contact_seal(),
            shared_dynamics_fit_sha256="a" * 64,
            target_prefix_geometry_sha256="b" * 64,
            future_start_frame=109,
        )


def test_prediction_archive_tampering_is_detected(tmp_path: Path) -> None:
    trajectories = _predictions()
    archive = tmp_path / "predictions.npz"
    seal = seal_held_out_rope_predictions(
        archive,
        {name: trajectories[name] for name in ("visual_only", "tactile_conditioned_z")},
        protocol_id="fixture",
        contact_prediction_seal=_contact_seal(),
        shared_dynamics_fit_sha256="a" * 64,
        target_prefix_geometry_sha256="b" * 64,
        future_start_frame=109,
    )
    archive.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="archive checksum mismatch"):
        validate_held_out_rope_prediction_seal(seal)
