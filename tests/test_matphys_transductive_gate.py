import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _load_gate():
    scripts = Path(__file__).parents[1] / "scripts"
    aggregate_spec = importlib.util.spec_from_file_location(
        "aggregate_matphys_transductive_sweep",
        scripts / "aggregate_matphys_transductive_sweep.py",
    )
    assert aggregate_spec is not None and aggregate_spec.loader is not None
    aggregate = importlib.util.module_from_spec(aggregate_spec)
    aggregate_spec.loader.exec_module(aggregate)

    import sys

    sys.modules[aggregate_spec.name] = aggregate
    gate_spec = importlib.util.spec_from_file_location(
        "evaluate_matphys_transductive_gate",
        scripts / "evaluate_matphys_transductive_gate.py",
    )
    assert gate_spec is not None and gate_spec.loader is not None
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    return gate


def _write_result(path: Path, case: str, cd_m: float, track_m: float) -> None:
    provenance = {}
    for key in ("checkpoint", "trajectory", "training_audit"):
        bound_path = path.parent / f"{case}-{key}.bin"
        bound_path.write_bytes(f"{case}-{key}".encode())
        provenance[key] = {
            "path": str(bound_path.resolve()),
            "sha256": hashlib.sha256(bound_path.read_bytes()).hexdigest(),
            "size_bytes": bound_path.stat().st_size,
        }
    path.write_text(
        json.dumps(
            {
                "contract": "matphys-offline-all-frame-reconstruction-v1",
                "claim_boundary": "offline reconstruction only",
                "future_observations_used": True,
                "released_test_outcomes_used_in_objective": True,
                "case_name": case,
                **provenance,
                "official_evaluation": {
                    "evaluation": {
                        "test": {
                            "frame_count": 3,
                            "chamfer_distance_m": cd_m,
                            "track_error_m": track_m,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_gate_passes_only_when_means_and_case_caps_pass(tmp_path: Path) -> None:
    gate = _load_gate()
    paths = []
    for case, baseline in gate.GATE_BASELINES.items():
        path = tmp_path / f"{case}.json"
        _write_result(
            path,
            case,
            0.9 * baseline["chamfer_distance_m"],
            0.9 * baseline["track_error_m"],
        )
        paths.append(path)

    decision = gate.evaluate_gate(paths)

    assert decision["passed"] is True
    assert decision["decision"] == "continue_full_22"


def test_gate_rejects_mean_regression(tmp_path: Path) -> None:
    gate = _load_gate()
    paths = []
    for case, baseline in gate.GATE_BASELINES.items():
        path = tmp_path / f"{case}.json"
        _write_result(
            path,
            case,
            0.9 * baseline["chamfer_distance_m"],
            1.01 * baseline["track_error_m"],
        )
        paths.append(path)

    decision = gate.evaluate_gate(paths)

    assert decision["passed"] is False
    assert decision["mean_checks"]["track_error_m"]["improves"] is False


def test_gate_rejects_large_case_regression_despite_better_mean(tmp_path: Path) -> None:
    gate = _load_gate()
    cases = sorted(gate.GATE_BASELINES)
    first, second = cases
    paths = []
    factors = {first: 0.5, second: 1.11}
    for case in cases:
        baseline = gate.GATE_BASELINES[case]
        path = tmp_path / f"{case}.json"
        _write_result(
            path,
            case,
            factors[case] * baseline["chamfer_distance_m"],
            factors[case] * baseline["track_error_m"],
        )
        paths.append(path)

    decision = gate.evaluate_gate(paths)

    assert all(item["improves"] for item in decision["mean_checks"].values())
    assert decision["passed"] is False
    assert decision["per_case_checks"][second]["track_error_m"][
        "within_regression_cap"
    ] is False


def test_gate_rejects_incomplete_cohort(tmp_path: Path) -> None:
    gate = _load_gate()
    case = next(iter(gate.GATE_BASELINES))
    baseline = gate.GATE_BASELINES[case]
    path = tmp_path / f"{case}.json"
    _write_result(
        path,
        case,
        baseline["chamfer_distance_m"],
        baseline["track_error_m"],
    )

    with pytest.raises(ValueError, match="gate cohort mismatch"):
        gate.evaluate_gate([path])


def test_gate_rejects_changed_bound_file(tmp_path: Path) -> None:
    gate = _load_gate()
    paths = []
    for case, baseline in gate.GATE_BASELINES.items():
        path = tmp_path / f"{case}.json"
        _write_result(
            path,
            case,
            baseline["chamfer_distance_m"],
            baseline["track_error_m"],
        )
        paths.append(path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    Path(payload["trajectory"]["path"]).write_bytes(b"mutated")

    with pytest.raises(ValueError, match="trajectory provenance hash changed"):
        gate.evaluate_gate(paths)
