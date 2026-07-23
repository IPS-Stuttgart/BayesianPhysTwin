import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _load_aggregator():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "aggregate_matphys_transductive_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("matphys_transductive_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_result(
    path: Path,
    case_name: str,
    *,
    frames: int,
    cd_m: float,
    track_m: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for key in ("checkpoint", "trajectory", "training_audit"):
        bound_path = path.parent / f"{case_name}-{key}.bin"
        bound_path.write_bytes(f"{case_name}-{key}".encode())
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
                "case_name": case_name,
                **provenance,
                "official_evaluation": {
                    "evaluation": {
                        "test": {
                            "frame_count": frames,
                            "chamfer_distance_m": cd_m,
                            "track_error_m": track_m,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_reports_case_and_frame_weighted_means(tmp_path: Path) -> None:
    aggregator = _load_aggregator()
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write_result(first, "a", frames=1, cd_m=0.004, track_m=0.010)
    _write_result(second, "b", frames=3, cd_m=0.012, track_m=0.020)

    summary = aggregator.aggregate_results(
        [first, second], expected_cases={"a", "b"}
    )

    assert summary["case_count"] == 2
    assert summary["test_frame_count"] == 4
    cd = summary["metrics"]["chamfer_distance_m"]
    assert cd["case_balanced_mean_m"] == pytest.approx(0.008)
    assert cd["frame_weighted_mean_m"] == pytest.approx(0.010)
    assert cd["case_balanced_percent_vs_reference"] == pytest.approx(0.0)


def test_aggregate_rejects_incomplete_cohort(tmp_path: Path) -> None:
    aggregator = _load_aggregator()
    result = tmp_path / "a.json"
    _write_result(result, "a", frames=1, cd_m=0.004, track_m=0.010)

    with pytest.raises(ValueError, match=r"missing=\['b'\]"):
        aggregator.aggregate_results([result], expected_cases={"a", "b"})


def test_aggregate_rejects_undisclosed_future_use(tmp_path: Path) -> None:
    aggregator = _load_aggregator()
    result = tmp_path / "a.json"
    _write_result(result, "a", frames=1, cd_m=0.004, track_m=0.010)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["future_observations_used"] = False
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="future-observation disclosure"):
        aggregator.aggregate_results([result], expected_cases={"a"})


def test_aggregate_rejects_changed_bound_file(tmp_path: Path) -> None:
    aggregator = _load_aggregator()
    result = tmp_path / "a.json"
    _write_result(result, "a", frames=1, cd_m=0.004, track_m=0.010)
    payload = json.loads(result.read_text(encoding="utf-8"))
    Path(payload["training_audit"]["path"]).write_bytes(b"mutated")

    with pytest.raises(ValueError, match="training_audit provenance hash changed"):
        aggregator.aggregate_results([result], expected_cases={"a"})
