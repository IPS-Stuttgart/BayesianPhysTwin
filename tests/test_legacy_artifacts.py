from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.causal4d_artifacts_v1 as artifact_api
import bayesian_phystwin.legacy_artifacts as legacy_artifacts
from bayesian_phystwin.causal4d_artifacts_v1 import (
    ReleasedPhysTwinRawTrackMapV1,
    causal4d_artifact_provider_manifest,
    load_released_phystwin_raw_track_map,
    load_trusted_legacy_phystwin_pickle,
)


def _write_pickle(path: Path, value) -> str:
    with path.open("wb") as stream:
        pickle.dump(value, stream)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provider_advertises_trusted_legacy_loading() -> None:
    manifest = causal4d_artifact_provider_manifest()
    assert manifest["provider_api_version"] == 1
    assert "digest_preflight_before_pickle" in manifest["capabilities"]
    assert "released_raw_track_map" in manifest["capabilities"]
    assert manifest["new_artifact_policy"] == "json-npz-only"


def test_loads_hash_locked_mapping_and_validates_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    digest = _write_pickle(path, {"object_points": np.arange(3), "surface_points": []})

    loaded = load_trusted_legacy_phystwin_pickle(
        path,
        expected_sha256=digest,
        artifact_kind="mapping",
        required_keys=("object_points", "surface_points"),
    )

    np.testing.assert_array_equal(loaded["object_points"], np.arange(3))


def test_rejects_digest_mismatch_before_deserialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "artifact.pkl"
    _write_pickle(path, {"value": 1})
    monkeypatch.setattr(
        legacy_artifacts.pickle,
        "load",
        lambda stream: pytest.fail("pickle must not be opened after digest mismatch"),
    )

    with pytest.raises(ValueError, match="refusing to deserialize"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256="0" * 64,
            artifact_kind="mapping",
        )


def test_rejects_top_level_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    digest = _write_pickle(path, [1, 2, 3])

    with pytest.raises(TypeError, match="mapping"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256=digest,
            artifact_kind="mapping",
        )


def test_rejects_missing_mapping_keys_and_invalid_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pkl"
    digest = _write_pickle(path, {"value": 1})

    with pytest.raises(ValueError, match="missing required keys"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256=digest,
            artifact_kind="mapping",
            required_keys=("other",),
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_trusted_legacy_phystwin_pickle(
            path,
            expected_sha256=digest.upper(),
            artifact_kind="mapping",
        )


def test_released_raw_track_map_is_hash_locked_and_immutable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_path = tmp_path / "final_data.pkl"
    digest = _write_pickle(
        final_path,
        {
            "object_points": np.zeros((2, 2, 3)),
            "object_visibilities": np.ones((2, 2), dtype=bool),
        },
    )
    raw_case = tmp_path / "raw"
    raw_case.mkdir()

    class FakeConfig:
        def __init__(self, *, initial_match_tolerance_m: float) -> None:
            assert initial_match_tolerance_m == 1e-5

    class FakeMapping:
        track_paths = (raw_case / "camera0.npz",)
        tracks_by_camera = (np.zeros((2, 2, 2)),)
        visibility_by_camera = (np.ones((2, 2), dtype=bool),)
        source_camera = np.asarray((0, 0))
        source_track = np.asarray((0, 1))

    class FakeModule:
        PhysTwinRawCueConfig = FakeConfig

        @staticmethod
        def load_phystwin_raw_track_map(final_data_path, raw_case_dir, *, config):
            assert Path(final_data_path) == final_path
            assert Path(raw_case_dir) == raw_case
            assert isinstance(config, FakeConfig)
            return FakeMapping()

    monkeypatch.setattr(artifact_api, "import_module", lambda name: FakeModule)
    mapping = load_released_phystwin_raw_track_map(
        final_path,
        raw_case,
        final_data_sha256=digest,
        initial_match_tolerance_m=1e-5,
    )

    assert isinstance(mapping, ReleasedPhysTwinRawTrackMapV1)
    assert mapping.final_data_sha256 == digest
    assert not mapping.tracks_by_camera[0].flags.writeable
    assert not mapping.visibility_by_camera[0].flags.writeable
    assert not mapping.source_camera.flags.writeable
    assert not mapping.source_track.flags.writeable


def test_released_raw_track_map_rejects_digest_before_internal_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_path = tmp_path / "final_data.pkl"
    _write_pickle(final_path, {"object_points": [], "object_visibilities": []})
    monkeypatch.setattr(
        artifact_api,
        "import_module",
        lambda name: pytest.fail("raw-track implementation must not load"),
    )

    with pytest.raises(ValueError, match="refusing to deserialize"):
        load_released_phystwin_raw_track_map(
            final_path,
            tmp_path,
            final_data_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"final_data_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"track_paths": ()}, "identify each camera"),
        ({"tracks_by_camera": (np.zeros((2, 2)),)}, "shape"),
        ({"visibility_by_camera": (np.ones((2, 3), dtype=bool),)}, "visibility"),
        ({"tracks_by_camera": (np.full((2, 2, 2), np.nan),)}, "finite"),
        (
            {
                "track_paths": (Path("a"), Path("b")),
                "tracks_by_camera": (np.zeros((2, 2, 2)), np.zeros((3, 2, 2))),
                "visibility_by_camera": (
                    np.ones((2, 2), dtype=bool),
                    np.ones((3, 2), dtype=bool),
                ),
            },
            "same frame count",
        ),
        ({"source_track": np.asarray((0,))}, "matching vectors"),
        ({"source_camera": np.asarray((1, 0))}, "unavailable camera"),
        ({"source_track": np.asarray((2, 0))}, "unavailable raw track"),
    ),
)
def test_released_raw_track_map_rejects_invalid_contract(
    kwargs,
    message: str,
) -> None:
    values = {
        "final_data_sha256": "a" * 64,
        "raw_case_dir": Path("raw"),
        "track_paths": (Path("tracks.npz"),),
        "tracks_by_camera": (np.zeros((2, 2, 2)),),
        "visibility_by_camera": (np.ones((2, 2), dtype=bool),),
        "source_camera": np.asarray((0, 0)),
        "source_track": np.asarray((0, 1)),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ReleasedPhysTwinRawTrackMapV1(**values)
