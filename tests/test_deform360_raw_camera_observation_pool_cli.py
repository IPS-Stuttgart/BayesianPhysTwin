import sys

import pytest

from bayesian_phystwin.cli.deform360_raw_camera_observation_pool import (
    build_parser,
    main,
)


def test_pool_cli_defaults_to_frozen_64_centers() -> None:
    args = build_parser().parse_args(
        ["panel", "processed", "output", "tracker", "checkpoint"]
    )

    assert args.center_count == 64


def test_pool_cli_rejects_nonfrozen_center_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deform360_raw_camera_observation_pool",
            "panel",
            "processed",
            "output",
            "tracker",
            "checkpoint",
            "--center-count",
            "32",
        ],
    )

    with pytest.raises(ValueError, match="requires 64"):
        main()
