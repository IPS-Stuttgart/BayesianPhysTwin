import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.matphys_causal_bridge import sha256_file
from bayesian_phystwin.matphys_loo_strength_sweep import (
    _refit_command,
    _resolve_released_artifact,
    _resolve_released_checkpoint,
    _validate_replay_cache,
    _write_stability_control,
    build_strength_external_manifest,
    strength_family_name,
)


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def test_strength_family_name_is_order_preserving() -> None:
    assert [strength_family_name(value) for value in (0.0, 0.25, 0.5, 1.0)] == [
        "alpha_0000",
        "alpha_0250",
        "alpha_0500",
        "alpha_1000",
    ]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        strength_family_name(1.1)


def test_refit_command_seals_future_scoring(tmp_path: Path) -> None:
    command = _refit_command(
        tmp_path / "python",
        tmp_path / "official",
        tmp_path / "case",
        tmp_path / "cues.npz",
        tmp_path / "checkpoint.pt",
        tmp_path / "output",
        train_end=8,
        fit_end=6,
    )

    assert "--selection-only" in command
    assert "--released-trajectory" in command
    assert "--gt-track-3d" not in command
    assert command[command.index("--fit-end-frame") + 1] == "6"


def test_refit_command_accepts_split_upstream_inputs(tmp_path: Path) -> None:
    optimal = tmp_path / "upstream" / "optimal.pkl"
    trajectory = tmp_path / "upstream" / "inference.pkl"
    command = _refit_command(
        tmp_path / "python",
        tmp_path / "official",
        tmp_path / "case",
        tmp_path / "cues.npz",
        tmp_path / "checkpoint.pt",
        tmp_path / "output",
        train_end=8,
        fit_end=6,
        optimal_params=optimal,
        released_trajectory=trajectory,
    )

    assert str(optimal) in command
    assert command[command.index("--released-trajectory") + 1] == str(trajectory)


def test_released_checkpoint_prefers_extracted_copy(tmp_path: Path) -> None:
    case_data = tmp_path / "data" / "case_a"
    case_data.mkdir(parents=True)
    extracted = case_data / "checkpoint.pth"
    extracted.write_bytes(b"extracted")
    upstream = tmp_path / "official" / "experiments" / "case_a" / "train"
    upstream.mkdir(parents=True)
    (upstream / "best_99.pth").write_bytes(b"upstream")

    assert (
        _resolve_released_checkpoint(case_data, tmp_path / "official", "case_a")
        == extracted
    )


def test_released_checkpoint_accepts_one_pinned_upstream_checkpoint(
    tmp_path: Path,
) -> None:
    case_data = tmp_path / "data" / "case_a"
    case_data.mkdir(parents=True)
    upstream = tmp_path / "official" / "experiments" / "case_a" / "train"
    upstream.mkdir(parents=True)
    checkpoint = upstream / "best_199.pth"
    checkpoint.write_bytes(b"upstream")

    assert (
        _resolve_released_checkpoint(case_data, tmp_path / "official", "case_a")
        == checkpoint
    )


def test_released_checkpoint_rejects_ambiguous_upstream_matches(
    tmp_path: Path,
) -> None:
    case_data = tmp_path / "data" / "case_a"
    case_data.mkdir(parents=True)
    upstream = tmp_path / "official" / "experiments" / "case_a" / "train"
    upstream.mkdir(parents=True)
    (upstream / "best_80.pth").write_bytes(b"first")
    (upstream / "best_99.pth").write_bytes(b"second")

    with pytest.raises(FileNotFoundError, match="exactly one upstream"):
        _resolve_released_checkpoint(case_data, tmp_path / "official", "case_a")


@pytest.mark.parametrize(
    ("filename", "relative"),
    [
        ("inference.pkl", "experiments/case_a/inference.pkl"),
        (
            "optimal_params.pkl",
            "experiments_optimization/case_a/optimal_params.pkl",
        ),
    ],
)
def test_released_artifact_accepts_compact_and_upstream_layouts(
    tmp_path: Path,
    filename: str,
    relative: str,
) -> None:
    case_data = tmp_path / "data" / "case_a"
    case_data.mkdir(parents=True)
    upstream = tmp_path / "official" / relative
    upstream.parent.mkdir(parents=True)
    upstream.write_bytes(b"upstream")

    assert (
        _resolve_released_artifact(case_data, tmp_path / "official", "case_a", filename)
        == upstream
    )
    extracted = case_data / filename
    extracted.write_bytes(b"extracted")
    assert (
        _resolve_released_artifact(case_data, tmp_path / "official", "case_a", filename)
        == extracted
    )


def test_stability_control_resolves_upstream_released_trajectory(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    (data_root / "case_a").mkdir(parents=True)
    official = tmp_path / "official"
    trajectory = official / "experiments" / "case_a" / "inference.pkl"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_bytes(b"trajectory")
    destination = tmp_path / "control.json"

    _write_stability_control(
        "alpha_0250",
        ["case_a"],
        data_root,
        official,
        destination,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["future_observations_used"] is False
    assert payload["cases"][0]["trajectory"] == _identity(trajectory)


def test_replay_cache_requires_sealed_summary_and_exact_overlay(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    candidate = tmp_path / "candidate.npy"
    output_checkpoint = tmp_path / "overlay.pt"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")
    output_checkpoint.write_bytes(b"overlay")
    overlay_summary = tmp_path / "overlay.json"
    overlay_summary.write_text(
        json.dumps(
            {
                "proposal_strength": 0.5,
                "source_checkpoint": _identity(source),
                "candidate_spring_y": _identity(candidate),
                "output_checkpoint": _identity(output_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    trajectory = replay_root / "trajectory.pkl"
    trajectory.write_bytes(b"trajectory")
    summary = {
        "future_metrics_opened": False,
        "config": {"evaluate_future": False},
        "inputs": {"checkpoint": _identity(output_checkpoint)},
        "outputs": {"trajectory": str(trajectory)},
    }
    (replay_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert _validate_replay_cache(
        replay_root,
        overlay_summary,
        source_checkpoint=source,
        candidate_field=candidate,
        strength=0.5,
    )
    summary["future_metrics_opened"] = True
    (replay_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not _validate_replay_cache(
        replay_root,
        overlay_summary,
        source_checkpoint=source,
        candidate_field=candidate,
        strength=0.5,
    )


def test_strength_manifest_binds_future_sealed_replays(tmp_path: Path) -> None:
    source_manifest = tmp_path / "fields.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    replay_root = tmp_path / "replays"
    case_root = replay_root / "alpha_0500" / "cases" / "case_a"
    replay = case_root / "replay"
    replay.mkdir(parents=True)
    trajectory = np.zeros((4, 2, 3), dtype=np.float32)
    with (replay / "trajectory.pkl").open("wb") as handle:
        pickle.dump(trajectory, handle)
    (replay / "summary.json").write_text(
        json.dumps({"future_metrics_opened": False}), encoding="utf-8"
    )
    (case_root / "overlay.json").write_text("{}\n", encoding="utf-8")
    fields = {
        "manifest": _identity(source_manifest),
        "backbone": {
            "source_repository": "https://example.test/matphys",
            "source_commit": "a" * 40,
            "proxy_contract": "test-proxy",
        },
        "cases": [
            {
                "name": "case_a",
                "evidence_end_frame_exclusive": 3,
            }
        ],
    }

    result = build_strength_external_manifest(
        fields,
        replay_root,
        tmp_path / "external.json",
        strength=0.5,
    )

    assert result["backbone"]["future_observations_used"] is False
    assert result["backbone"]["proposal_strength"] == pytest.approx(0.5)
    assert result["cases"][0]["sha256"] == sha256_file(replay / "trajectory.pkl")
