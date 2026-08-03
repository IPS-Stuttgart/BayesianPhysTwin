from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "held" / "run_pokeflex_baseline_relative_guard_target.py"
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_baseline_relative_guard_public_paired_v2.json"
)


def test_stage_materializes_only_causal_inputs_and_one_template(
    tmp_path: Path,
) -> None:
    take_id = "3dPrintedCylinder_T3"
    archive_path = tmp_path / f"{take_id}.zip"
    robot = [
        {
            "frame": frame,
            "forces": [0.0, 4.0 if frame >= 2 else 0.0, 0.0],
        }
        for frame in range(1, 8)
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{take_id}/robot_data.json", json.dumps(robot))
        archive.writestr(f"{take_id}/meshes/mesh-f00001.obj", "template")
        archive.writestr(f"{take_id}/meshes/mesh-f00006.obj", "forbidden-target")
        for camera in (0, 1):
            parameters = {
                "depth_intrinsics": np.eye(3).tolist(),
                "depth_extrinsics": np.eye(4).tolist(),
            }
            archive.writestr(
                f"{take_id}/kinect/{camera}/camera_parameters.json",
                json.dumps(parameters),
            )
            archive.writestr(
                f"{take_id}/volucam/{camera}/camera_parameters.json", "wrong"
            )
            for frame in range(1, 7):
                archive.writestr(
                    f"{take_id}/kinect/{camera}/depth/{frame:05d}.png",
                    f"kinect-depth-{camera}-{frame}",
                )
                archive.writestr(
                    f"{take_id}/realsense/{camera}/depth/{frame:05d}.png",
                    f"wrong-depth-{camera}-{frame}",
                )
    output = tmp_path / take_id

    subprocess.run(
        (
            sys.executable,
            str(RUNNER),
            "--protocol",
            str(PROTOCOL),
            "stage",
            str(archive_path),
            str(output),
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(
        (output / "causal_input_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["future_target_mesh_read_count"] == 0
    assert manifest["target_frame_observation_read_count"] == 0
    assert manifest["template_mesh_read_count"] == 1
    assert (output / "meshes" / "mesh-f00001.obj").read_text() == "template"
    assert not (output / "meshes" / "mesh-f00006.obj").exists()
    assert len(list((output / "kinect" / "0" / "depth").glob("*.png"))) == 6
    assert (
        output / "kinect" / "0" / "depth" / "00001.png"
    ).read_text() == "kinect-depth-0-1"
