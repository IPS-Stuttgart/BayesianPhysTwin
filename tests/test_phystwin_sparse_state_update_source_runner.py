import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "run_phystwin_sparse_state_update_source.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_phystwin_sparse_state_update_source",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _protocol() -> dict:
    return {
        "inputs": {
            "final_data": {"sha256": "final"},
            "optimal_params": {"sha256": "optimal"},
            "checkpoint": {"sha256": "checkpoint"},
        },
        "source_replay": {
            "code_commit": "commit",
            "runtime": {
                "torch_version": "2.9.1+cu128",
                "warp_version": "1.11.1",
            },
            "config": {
                "epochs": 0,
                "deterministic_spring_forces": True,
            },
        },
    }


def _summary() -> dict:
    return {
        "code_commit": "commit",
        "runtime": {
            "torch_version": "2.9.1+cu128",
            "warp_version": "1.11.1",
        },
        "inputs": {
            "final_data": {"sha256": "final"},
            "optimal_params": {"sha256": "optimal"},
            "checkpoint": {"sha256": "checkpoint"},
        },
        "config": {
            "epochs": 0,
            "deterministic_spring_forces": True,
        },
        "released_trajectory_parity": {
            "vector_rmse_m": 0.0,
            "max_norm_m": 0.0,
        },
        "selected_baseline_trajectory_parity": {
            "vector_rmse_m": 0.0,
            "max_norm_m": 0.0,
        },
    }


def _write_summary(tmp_path: Path, summary: dict) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_source_replay_verifier_accepts_exact_provenance(tmp_path: Path) -> None:
    summary = _summary()
    path = _write_summary(tmp_path, summary)

    result = RUNNER._verify_source_replay(
        _protocol(),
        {"source_replay_summary": path},
    )

    assert result == summary


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("code_commit", "different", "code commit"),
        ("runtime", {"torch_version": "different"}, "runtime"),
    ),
)
def test_source_replay_verifier_rejects_changed_provenance(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    summary = _summary()
    summary[field] = value
    path = _write_summary(tmp_path, summary)

    with pytest.raises(ValueError, match=message):
        RUNNER._verify_source_replay(
            _protocol(),
            {"source_replay_summary": path},
        )


def test_source_replay_verifier_rejects_nonzero_parity(tmp_path: Path) -> None:
    summary = _summary()
    summary["selected_baseline_trajectory_parity"]["vector_rmse_m"] = 1e-6
    path = _write_summary(tmp_path, summary)

    with pytest.raises(ValueError, match="exact trajectory parity"):
        RUNNER._verify_source_replay(
            _protocol(),
            {"source_replay_summary": path},
        )
