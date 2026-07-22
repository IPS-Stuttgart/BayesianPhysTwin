from pathlib import Path


def test_force_depth_runner_keeps_target_objects_outside_opened_cohort() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_pokeflex_force_depth_regret_development.py"
    ).read_text(encoding="utf-8")

    assert "3dPrintedPyramid_T2" in source
    assert "PlushMoon_T2" in source
    assert "target_objects_opened\": False" in source
    assert "record_online_observation_regret=False" in source
    assert "record_independent_anchor_regret=True" in source
    assert "force_action_plane_local_state" in source
