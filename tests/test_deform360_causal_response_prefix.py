from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_prefix import (
    ARCHIVE_FILENAME,
    CausalResponsePrefixConfig,
    CausalResponsePrefixInputs,
    validate_causal_response_prefix_artifacts,
    write_causal_response_prefix_artifacts,
)


def _inputs(*, frame_count: int = 11) -> CausalResponsePrefixInputs:
    camera_count = 6
    height = width = 8
    return CausalResponsePrefixInputs(
        config=CausalResponsePrefixConfig(
            prefix_frame_count=11,
            minimum_camera_count=6,
        ),
        camera_ids=tuple(f"camera-{index}" for index in range(camera_count)),
        intrinsics=np.repeat(np.eye(3)[None], camera_count, axis=0),
        camera_to_world=np.repeat(
            np.eye(4)[None],
            camera_count,
            axis=0,
        ),
        depths_m=np.ones(
            (camera_count, frame_count, height, width),
            dtype=np.float32,
        ),
        object_masks=np.ones(
            (camera_count, frame_count, height, width),
            dtype=bool,
        ),
        tactile_contact_probability=np.linspace(
            0.0,
            1.0,
            frame_count,
        ),
        measured_actuator_positions_m=np.zeros((frame_count, 2, 3)),
    )


def test_prefix_artifact_round_trips_without_future_or_rgb(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prefix"

    report = write_causal_response_prefix_artifacts(
        output,
        inputs,
        case_id="fresh-source",
        protocol_path=protocol,
        source_sha256={"released_episode": "a" * 64},
    )
    validated, loaded = validate_causal_response_prefix_artifacts(output)

    assert validated["result_sha256"] == report["result_sha256"]
    assert validated["information_boundary"]["rgb_included"] is False
    assert validated["information_boundary"]["maximum_observation_frame"] == 10
    np.testing.assert_array_equal(loaded.depths_m, inputs.depths_m)
    np.testing.assert_array_equal(
        loaded.tactile_contact_probability,
        inputs.tactile_contact_probability,
    )


def test_prefix_rejects_a_future_length_carrier() -> None:
    with pytest.raises(ValueError, match="camera arrays changed shape"):
        _inputs(frame_count=12)


def test_prefix_validator_detects_archive_tampering(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prefix"
    write_causal_response_prefix_artifacts(
        output,
        _inputs(),
        case_id="fresh-source",
        protocol_path=protocol,
        source_sha256={"released_episode": "a" * 64},
    )

    with (output / ARCHIVE_FILENAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="archive checksum"):
        validate_causal_response_prefix_artifacts(output)
