from bayesian_phystwin.cli.deform360_raw_camera_observation import build_parser


def test_build_cohort_center_count_default_remains_frozen() -> None:
    args = build_parser().parse_args(
        ["build-cohort", "panel", "processed", "output", "tracker", "checkpoint"]
    )

    assert args.center_count == 16


def test_build_cohort_accepts_larger_causal_observation_pool() -> None:
    args = build_parser().parse_args(
        [
            "build-cohort",
            "panel",
            "processed",
            "output",
            "tracker",
            "checkpoint",
            "--center-count",
            "64",
        ]
    )

    assert args.center_count == 64
