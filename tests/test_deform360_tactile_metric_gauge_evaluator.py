from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin._portable_contracts import content_id

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "evaluate_deform360_tactile_metric_gauge_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_deform360_tactile_metric_gauge_evaluator",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _camera(name: str, admitted: tuple[bool, bool]) -> dict[str, object]:
    hypotheses = []
    for value in admitted:
        hypotheses.append(
            {
                "admitted": value,
                "covariance_m2": (
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    if value
                    else None
                ),
            }
        )
    return {"camera": name, "assignment_hypotheses": hypotheses}


def test_gate_requires_three_cameras_that_pass_both_assignments() -> None:
    records = [
        _camera("a", (True, True)),
        _camera("b", (True, True)),
        _camera("c", (True, False)),
    ]

    rejected = MODULE._gate_summary(
        records,
        minimum_admitted_cameras=3,
        assignment_probabilities=(0.5, 0.5),
    )
    assert not rejected["metric_gauge_authorized"]
    assert not rejected["contact_anchor_authorized"]
    assert rejected["jointly_admitted_cameras"] == ["a", "b"]

    records[2] = _camera("c", (True, True))
    accepted = MODULE._gate_summary(
        records,
        minimum_admitted_cameras=3,
        assignment_probabilities=(0.5, 0.5),
    )
    assert accepted["metric_gauge_authorized"]
    assert not accepted["contact_anchor_authorized"]
    assert all(
        item["covariance_intersection_m2"] is not None
        for item in accepted["assignment_mixture"]
    )


def test_camera_dictionary_loader_accepts_first_party_npy(tmp_path: Path) -> None:
    path = tmp_path / "intrinsics.npy"
    np.save(path, {"camera": np.eye(3)}, allow_pickle=True)

    loaded = MODULE._load_camera_dictionary(path)

    assert list(loaded) == ["camera"]
    assert np.array_equal(loaded["camera"], np.eye(3))


def test_run_report_loader_checks_content_identity(tmp_path: Path) -> None:
    descriptor = {"schema": "test", "status": "complete"}
    path = tmp_path / "run_report.json"
    path.write_text(
        json.dumps({"run_id": content_id(descriptor), **descriptor}),
        encoding="utf-8",
    )

    assert MODULE._load_content_addressed_report(path, id_field="run_id") == {
        "run_id": content_id(descriptor),
        **descriptor,
    }

    path.write_text(
        json.dumps({"run_id": "0" * 64, **descriptor}),
        encoding="utf-8",
    )
    try:
        MODULE._load_content_addressed_report(path, id_field="run_id")
    except ValueError as error:
        assert "identity changed" in str(error)
    else:
        raise AssertionError("mutated run report was accepted")
