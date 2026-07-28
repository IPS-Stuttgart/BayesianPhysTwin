from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.causal4d_artifacts_v2 as artifact_api
from bayesian_phystwin.causal4d_artifacts_v2 import (
    ReleasedPhysTwinVisualInputsV2,
    load_released_phystwin_visual_inputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pickle(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)
    return _sha256(path)


def _valid_values(tmp_path: Path) -> dict[str, object]:
    raw = tmp_path / "raw"
    track = raw / "cotracker" / "camera0.npz"
    return {
        "raw_case_dir": raw,
        "final_data_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "pcd_sha256": "c" * 64,
        "calibration_sha256": "d" * 64,
        "cotracker_sha256": (("cotracker/camera0.npz", "e" * 64),),
        "initial_match_tolerance_m": 1e-5,
        "object_points_m": np.zeros((3, 2, 3)),
        "object_visibility": np.ones((3, 2), dtype=bool),
        "object_motion_valid": np.ones((3, 2), dtype=bool),
        "track_paths": (track,),
        "tracks_by_camera": (np.zeros((3, 2, 2)),),
        "visibility_by_camera": (np.ones((3, 2), dtype=bool),),
        "source_camera": np.asarray((0, 0)),
        "source_track": np.asarray((0, 1)),
        "source_world_points_m": np.zeros((2, 3)),
        "initial_match_distance_m": np.asarray((0.0, 1e-7)),
        "intrinsics": np.eye(3)[None],
        "camera_to_world": np.eye(4)[None],
        "source_fps": 30.0,
        "image_width": 2,
        "image_height": 2,
    }


def _artifact(tmp_path: Path, **overrides: object) -> ReleasedPhysTwinVisualInputsV2:
    values = _valid_values(tmp_path)
    values.update(overrides)
    return ReleasedPhysTwinVisualInputsV2(**values)


def test_valid_visual_artifact_is_immutable_and_content_addressed(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    assert len(artifact.artifact_id) == 64
    assert artifact.artifact_id == artifact.artifact_id
    assert artifact.input_digests() == {
        "final_data.pkl": "a" * 64,
        "metadata.json": "b" * 64,
        "pcd/0.npz": "c" * 64,
        "calibrate.pkl": "d" * 64,
        "cotracker/camera0.npz": "e" * 64,
    }
    arrays = (
        artifact.object_points_m,
        artifact.object_visibility,
        artifact.object_motion_valid,
        artifact.tracks_by_camera[0],
        artifact.visibility_by_camera[0],
        artifact.source_camera,
        artifact.source_track,
        artifact.source_world_points_m,
        artifact.initial_match_distance_m,
        artifact.intrinsics,
        artifact.camera_to_world,
    )
    assert all(not value.flags.writeable for value in arrays)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"final_data_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"final_data_sha256": "a" * 63}, "lowercase SHA-256"),
        ({"final_data_sha256": "g" * 64}, "lowercase SHA-256"),
        ({"cotracker_sha256": ()}, "unique nonempty"),
        (
            {
                "cotracker_sha256": (
                    ("cotracker/camera0.npz", "e" * 64),
                    ("cotracker/camera0.npz", "f" * 64),
                )
            },
            "unique nonempty",
        ),
        (
            {"cotracker_sha256": (("wrong/camera0.npz", "e" * 64),)},
            "cotracker/<archive>",
        ),
        (
            {"cotracker_sha256": (("cotracker/", "e" * 64),)},
            "cotracker/<archive>",
        ),
        ({"initial_match_tolerance_m": 0.0}, "positive and finite"),
        ({"initial_match_tolerance_m": np.nan}, "positive and finite"),
        ({"object_points_m": np.zeros((3, 2))}, "shape"),
        ({"object_points_m": np.zeros((3, 2, 2))}, "shape"),
        ({"object_visibility": np.ones((3, 1), dtype=bool)}, "visibility"),
        ({"object_motion_valid": np.ones((3, 1), dtype=bool)}, "motion"),
        (
            {
                "object_points_m": np.asarray(
                    [[[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 3
                )
            },
            "finite",
        ),
        ({"track_paths": ()}, "identify each camera"),
        ({"tracks_by_camera": ()}, "identify each camera"),
        ({"visibility_by_camera": ()}, "identify each camera"),
        ({"track_paths": (Path("outside/camera0.npz"),)}, "subpath"),
        (
            {"cotracker_sha256": (("cotracker/other.npz", "e" * 64),)},
            "order must match",
        ),
        ({"tracks_by_camera": (np.zeros((3, 2)),)}, "shape"),
        ({"tracks_by_camera": (np.zeros((2, 2, 2)),)}, "frame count"),
        ({"visibility_by_camera": (np.ones((3, 1), dtype=bool),)}, "visibility"),
        ({"tracks_by_camera": (np.full((3, 2, 2), np.nan),)}, "finite"),
        ({"source_camera": np.asarray((0,))}, "identify every object point"),
        ({"source_track": np.asarray((0,))}, "identify every object point"),
        ({"source_world_points_m": np.zeros((2, 2))}, "shape"),
        ({"initial_match_distance_m": np.zeros(1)}, "shape"),
        (
            {"source_world_points_m": np.asarray([[np.nan, 0.0, 0.0]] * 2)},
            "finite",
        ),
        ({"initial_match_distance_m": np.asarray((np.nan, 0.0))}, "finite"),
        ({"initial_match_distance_m": np.asarray((-1.0, 0.0))}, "tolerance"),
        ({"initial_match_distance_m": np.asarray((1.0, 0.0))}, "tolerance"),
        ({"source_camera": np.asarray((-1, 0))}, "unavailable camera"),
        ({"source_camera": np.asarray((1, 0))}, "unavailable camera"),
        ({"source_track": np.asarray((-1, 0))}, "unavailable raw track"),
        ({"source_track": np.asarray((2, 0))}, "unavailable raw track"),
        ({"intrinsics": np.eye(3)}, "intrinsics"),
        ({"camera_to_world": np.eye(4)}, "camera_to_world"),
        ({"intrinsics": np.full((1, 3, 3), np.nan)}, "calibration"),
        ({"camera_to_world": np.full((1, 4, 4), np.nan)}, "calibration"),
        ({"source_fps": 0.0}, "source_fps"),
        ({"source_fps": np.nan}, "source_fps"),
        ({"image_width": 0}, "dimensions"),
        ({"image_height": 0}, "dimensions"),
    ),
)
def test_visual_artifact_rejects_invalid_contract(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _artifact(tmp_path, **overrides)


def _loader_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    final_path = tmp_path / "final_data.pkl"
    final_digest = _write_pickle(
        final_path,
        {
            "object_points": np.zeros((3, 2, 3)),
            "object_visibilities": np.ones((3, 2), dtype=bool),
            "object_motions_valid": np.ones((3, 2), dtype=bool),
        },
    )
    raw = tmp_path / "raw"
    (raw / "pcd").mkdir(parents=True)
    (raw / "cotracker").mkdir()
    metadata = raw / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "WH": [2, 2],
                "intrinsics": np.eye(3)[None].tolist(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pcd = raw / "pcd" / "0.npz"
    np.savez_compressed(pcd, points=np.zeros((1, 2, 2, 3)))
    calibration = raw / "calibrate.pkl"
    calibration_digest = _write_pickle(calibration, np.eye(4)[None])
    track = raw / "cotracker" / "camera0.npz"
    np.savez_compressed(
        track,
        tracks=np.zeros((3, 2, 2)),
        visibility=np.ones((3, 2), dtype=bool),
    )
    return final_path, raw, {
        "final_data_sha256": final_digest,
        "metadata_sha256": _sha256(metadata),
        "pcd_sha256": _sha256(pcd),
        "calibration_sha256": calibration_digest,
        "cotracker/camera0.npz": _sha256(track),
    }


def _fake_module(
    final_path: Path,
    raw: Path,
    *,
    final_points: np.ndarray | None = None,
    final_visible: np.ndarray | None = None,
):
    class FakeConfig:
        def __init__(self, *, initial_match_tolerance_m: float) -> None:
            assert initial_match_tolerance_m == 1e-5

    class FakeMapping:
        track_paths = (raw / "cotracker" / "camera0.npz",)
        tracks_by_camera = (np.zeros((3, 2, 2)),)
        visibility_by_camera = (np.ones((3, 2), dtype=bool),)
        source_camera = np.asarray((0, 0))
        source_track = np.asarray((0, 1))
        source_world_points = np.zeros((2, 3))
        initial_match_distance_m = np.asarray((0.0, 1e-7))

    FakeMapping.final_points = (
        np.zeros((3, 2, 3)) if final_points is None else final_points
    )
    FakeMapping.final_visible = (
        np.ones((3, 2), dtype=bool) if final_visible is None else final_visible
    )

    class FakeModule:
        PhysTwinRawCueConfig = FakeConfig

        @staticmethod
        def load_phystwin_raw_track_map(final_data_path, raw_case_dir, *, config):
            assert Path(final_data_path) == final_path
            assert Path(raw_case_dir) == raw
            assert isinstance(config, FakeConfig)
            return FakeMapping()

    return FakeModule


def _load(
    final_path: Path,
    raw: Path,
    digests: dict[str, str],
    **overrides: object,
) -> ReleasedPhysTwinVisualInputsV2:
    arguments: dict[str, object] = {
        "final_data_sha256": digests["final_data_sha256"],
        "metadata_sha256": digests["metadata_sha256"],
        "pcd_sha256": digests["pcd_sha256"],
        "calibration_sha256": digests["calibration_sha256"],
        "cotracker_sha256": {
            "cotracker/camera0.npz": digests["cotracker/camera0.npz"]
        },
        "initial_match_tolerance_m": 1e-5,
    }
    arguments.update(overrides)
    return load_released_phystwin_visual_inputs(final_path, raw, **arguments)


def test_loader_accepts_complete_verified_fixture(tmp_path: Path, monkeypatch) -> None:
    final_path, raw, digests = _loader_fixture(tmp_path)
    monkeypatch.setattr(
        artifact_api,
        "import_module",
        lambda name: _fake_module(final_path, raw),
    )
    artifact = _load(final_path, raw, digests)
    assert artifact.input_digests()["metadata.json"] == digests["metadata_sha256"]


def test_loader_rejects_missing_track_directory(tmp_path: Path) -> None:
    final_path = tmp_path / "final.pkl"
    final_path.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="no cotracker"):
        load_released_phystwin_visual_inputs(
            final_path,
            tmp_path / "raw",
            final_data_sha256="a" * 64,
            metadata_sha256="b" * 64,
            pcd_sha256="c" * 64,
            calibration_sha256="d" * 64,
            cotracker_sha256={},
        )


def test_loader_rejects_missing_and_unexpected_track_inventory(tmp_path: Path) -> None:
    final_path, raw, digests = _loader_fixture(tmp_path)
    with pytest.raises(ValueError, match="digest inventory differs"):
        _load(final_path, raw, digests, cotracker_sha256={})
    with pytest.raises(ValueError, match="digest inventory differs"):
        _load(
            final_path,
            raw,
            digests,
            cotracker_sha256={
                "cotracker/camera0.npz": digests["cotracker/camera0.npz"],
                "cotracker/extra.npz": "f" * 64,
            },
        )


def test_loader_rejects_missing_file_and_digest_mismatch(tmp_path: Path) -> None:
    final_path, raw, digests = _loader_fixture(tmp_path)
    (raw / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError):
        _load(final_path, raw, digests)

    final_path, raw, digests = _loader_fixture(tmp_path / "other")
    with pytest.raises(ValueError, match="metadata.json SHA-256 mismatch"):
        _load(final_path, raw, digests, metadata_sha256="0" * 64)


def test_loader_rejects_nonmapping_metadata(tmp_path: Path) -> None:
    final_path, raw, digests = _loader_fixture(tmp_path)
    metadata = raw / "metadata.json"
    metadata.write_text("[]", encoding="utf-8")
    digests["metadata_sha256"] = _sha256(metadata)
    with pytest.raises(ValueError, match="must contain an object"):
        _load(final_path, raw, digests)


def test_loader_rejects_mapping_disagreement(tmp_path: Path, monkeypatch) -> None:
    final_path, raw, digests = _loader_fixture(tmp_path)
    monkeypatch.setattr(
        artifact_api,
        "import_module",
        lambda name: _fake_module(
            final_path,
            raw,
            final_points=np.ones((3, 2, 3)),
        ),
    )
    with pytest.raises(ValueError, match="final points differ"):
        _load(final_path, raw, digests)

    monkeypatch.setattr(
        artifact_api,
        "import_module",
        lambda name: _fake_module(
            final_path,
            raw,
            final_visible=np.zeros((3, 2), dtype=bool),
        ),
    )
    with pytest.raises(ValueError, match="visibility differs"):
        _load(final_path, raw, digests)


def test_loader_rejects_postload_mutation(tmp_path: Path, monkeypatch) -> None:
    final_path, raw, digests = _loader_fixture(tmp_path)
    fake = _fake_module(final_path, raw)
    original = fake.load_phystwin_raw_track_map

    def mutate(final_data_path, raw_case_dir, *, config):
        result = original(final_data_path, raw_case_dir, config=config)
        (raw / "metadata.json").write_text("{}", encoding="utf-8")
        return result

    fake.load_phystwin_raw_track_map = staticmethod(mutate)
    monkeypatch.setattr(artifact_api, "import_module", lambda name: fake)
    with pytest.raises(ValueError, match="metadata.json SHA-256 mismatch"):
        _load(final_path, raw, digests)
