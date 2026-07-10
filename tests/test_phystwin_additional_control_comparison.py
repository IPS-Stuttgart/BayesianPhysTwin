import json
from pathlib import Path

from bayesian_phystwin.phystwin_additional_control_comparison import (
    compare_additional_anchor_controls,
)


def _write_run(
    root: Path,
    *,
    mode: str,
    corrected: list[float],
) -> None:
    protocol = {
        "protocol_id": f"protocol-{mode}",
        "specification": {"spatial_mode": mode},
    }
    (root / "locked_protocol.json").parent.mkdir(parents=True)
    (root / "locked_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for case in ("cloth_blue_fold", "cloth_red_lift"):
        path = root / "cases" / case / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "future": {
                        "baseline_chamfer_by_frame_m": [1.0, 1.0, 1.0],
                        "corrected_chamfer_by_frame_m": corrected,
                    }
                }
            ),
            encoding="utf-8",
        )


def test_compare_additional_anchor_controls_is_direct_and_paired(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    _write_run(candidate, mode="per_point", corrected=[0.8, 0.8, 0.8])
    _write_run(reference, mode="se3", corrected=[1.0, 1.0, 1.0])

    result = compare_additional_anchor_controls(
        candidate,
        [reference],
        bootstrap_samples=20,
        bootstrap_block_length=2,
    )

    comparison = result["comparisons"]["se3"]
    macro = comparison["bootstrap"]["macro"]["chamfer_distance_m"]
    assert comparison["candidate_better_case_count"] == 2
    assert abs(macro["observed_macro_percent_change"] + 20.0) < 1e-12
    assert result["candidate"]["spatial_mode"] == "per_point"
