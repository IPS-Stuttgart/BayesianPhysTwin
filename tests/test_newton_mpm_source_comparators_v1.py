from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli import newton_mpm_backend as newton_cli
from bayesian_phystwin.newton_mpm_source_comparators_v1 import (
    COMPARATOR_MANIFEST_FILENAME,
    INCUMBENT_PHYSICAL_FILENAME,
    MATPHYS_PHYSICAL_FILENAME,
    materialize_source_comparators,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive


def _pickle(path: Path, value: object) -> Path:
    with path.open("wb") as stream:
        pickle.dump(value, stream)
    return path


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], np.ndarray, np.ndarray]:
    objects = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    surface = np.array([[0.0, 0.01, 0.0]], dtype=np.float32)
    interior = np.array([[0.0, 0.0, 0.01]], dtype=np.float32)
    frame_zero = np.concatenate((objects, surface, interior), axis=0)
    final_data = {
        "object_points": np.repeat(objects[None], 5, axis=0),
        "surface_points": surface,
        "interior_points": interior,
    }
    incumbent = np.repeat(frame_zero[None], 5, axis=0)
    matphys = incumbent.copy()
    for frame in range(1, 5):
        incumbent[frame, :, 0] += np.float32(0.001 * frame)
        matphys[frame, :, 0] += np.float32(0.0008 * frame)
    return (
        {
            "final_data": _pickle(tmp_path / "final-data.pkl", final_data),
            "incumbent": _pickle(tmp_path / "incumbent.pkl", incumbent),
            "matphys": _pickle(tmp_path / "matphys.pkl", matphys),
        },
        incumbent,
        matphys,
    )


def test_materializes_matched_comparators_with_explicit_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, incumbent, matphys = _fixture(tmp_path)
    monkeypatch.setattr(
        "bayesian_phystwin.newton_mpm_source_comparators_v1._implementation_provenance",
        lambda: {
            "git_head": "a" * 40,
            "git_worktree_clean": True,
            "source_files": {},
        },
    )
    output = tmp_path / "comparators"

    manifest = materialize_source_comparators(
        final_data_path=paths["final_data"],
        incumbent_trajectory_path=paths["incumbent"],
        matphys_trajectory_path=paths["matphys"],
        output_dir=output,
    )

    incumbent_archive = load_physical_rollout_archive(
        output / INCUMBENT_PHYSICAL_FILENAME,
        expected_frame_count=5,
    )
    matphys_archive = load_physical_rollout_archive(
        output / MATPHYS_PHYSICAL_FILENAME,
        expected_frame_count=5,
    )
    np.testing.assert_array_equal(incumbent_archive["prediction_m"], incumbent)
    np.testing.assert_array_equal(matphys_archive["prediction_m"], matphys)
    np.testing.assert_array_equal(
        incumbent_archive["action_support"],
        matphys_archive["action_support"],
    )
    np.testing.assert_array_equal(
        incumbent_archive["zero_action_readout_m"],
        incumbent_archive["persistence_m"],
    )
    assert manifest["information_boundary"]["object_geometry_frames_used"] == [0]
    assert "placeholder" in manifest["semantics"]["zero_action"]
    stored = json.loads(
        (output / COMPARATOR_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert stored["materialization_id"] == manifest["materialization_id"]
    with pytest.raises(FileExistsError):
        materialize_source_comparators(
            final_data_path=paths["final_data"],
            incumbent_trajectory_path=paths["incumbent"],
            matphys_trajectory_path=paths["matphys"],
            output_dir=output,
        )


def test_rejects_changed_frame_zero_and_no_action_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, incumbent, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        "bayesian_phystwin.newton_mpm_source_comparators_v1._implementation_provenance",
        lambda: {},
    )
    changed = incumbent.copy()
    changed[0, 0, 0] += 0.001
    changed_path = _pickle(tmp_path / "changed.pkl", changed)
    with pytest.raises(ValueError, match="frame-zero"):
        materialize_source_comparators(
            final_data_path=paths["final_data"],
            incumbent_trajectory_path=changed_path,
            matphys_trajectory_path=paths["matphys"],
            output_dir=tmp_path / "changed-output",
        )

    persistence = np.repeat(incumbent[0][None], len(incumbent), axis=0)
    persistence_path = _pickle(tmp_path / "persistence.pkl", persistence)
    with pytest.raises(ValueError, match="no action response"):
        materialize_source_comparators(
            final_data_path=paths["final_data"],
            incumbent_trajectory_path=persistence_path,
            matphys_trajectory_path=paths["matphys"],
            output_dir=tmp_path / "persistence-output",
        )


def test_comparator_cli_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_materialize(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "materialized"}

    monkeypatch.setattr(newton_cli, "materialize_source_comparators", fake_materialize)
    argv = [
        "source-materialize-comparators",
        str(tmp_path / "final.pkl"),
        str(tmp_path / "incumbent.pkl"),
        str(tmp_path / "matphys.pkl"),
        str(tmp_path / "output"),
    ]

    assert newton_cli.main(argv) == 0
    assert calls == [
        {
            "final_data_path": tmp_path / "final.pkl",
            "incumbent_trajectory_path": tmp_path / "incumbent.pkl",
            "matphys_trajectory_path": tmp_path / "matphys.pkl",
            "output_dir": tmp_path / "output",
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "materialized"}
