import json
import zipfile
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_independent_depth_staging import (
    stage_pokeflex_independent_depth_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "pokeflex_independent_depth_development_v1.json"
)


def _archive(path: Path, take_id: str) -> Path:
    archive = path / f"{take_id}.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        for camera in ("0", "1"):
            payload.writestr(
                f"{take_id}/realsense/{camera}/camera_parameters.json",
                json.dumps({"camera": camera}),
            )
            payload.writestr(
                f"{take_id}/realsense/{camera}/depth/00001.png",
                f"depth-{camera}".encode(),
            )
        payload.writestr(f"{take_id}/meshes/mesh-f00001.obj", "forbidden outcome")
        payload.writestr(f"{take_id}/robot_data.json", "forbidden side channel")
    return archive


def test_source_stage_extracts_only_realsense_anchor_members(tmp_path) -> None:
    archive = _archive(tmp_path, "FoamDice_T3")

    result = stage_pokeflex_independent_depth_source(
        archive, tmp_path / "stage", PROTOCOL
    )

    take = tmp_path / "stage" / "FoamDice_T3"
    assert result["outcome_members_read"] is False
    assert (take / "realsense/0/depth/00001.png").is_file()
    assert not (take / "meshes").exists()
    assert not (take / "robot_data.json").exists()


def test_source_stage_rejects_prospective_T2(tmp_path) -> None:
    archive = _archive(tmp_path, "FoamDice_T2")

    with pytest.raises(ValueError, match="not outcome-open"):
        stage_pokeflex_independent_depth_source(
            archive, tmp_path / "stage", PROTOCOL
        )


def test_source_stage_is_immutable(tmp_path) -> None:
    archive = _archive(tmp_path, "FoamDice_T3")
    destination = tmp_path / "stage"
    stage_pokeflex_independent_depth_source(archive, destination, PROTOCOL)
    staged = destination / "FoamDice_T3/realsense/0/depth/00001.png"
    staged.write_bytes(b"changed")

    with pytest.raises(ValueError, match="staged file differs"):
        stage_pokeflex_independent_depth_source(archive, destination, PROTOCOL)
