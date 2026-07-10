import json
from pathlib import Path

from bayesian_phystwin.phystwin_baseline_confirmation import _comparison_manifest


def test_baseline_comparison_manifest_uses_method_trajectory(tmp_path: Path):
    case = tmp_path / "data" / "case_a"
    case.mkdir(parents=True)
    (case / "split.json").write_text(
        json.dumps({"frame_len": 10, "train": [0, 7], "test": [7, 10]})
    )

    manifest = _comparison_manifest(
        tmp_path / "data", tmp_path / "output", ("case_a",), "dmdc"
    )

    entry = manifest["cases"][0]
    assert entry["start_frame"] == 7
    assert entry["candidate_trajectory"].endswith(
        "/output/cases/case_a/dmdc/trajectory.pkl"
    )
