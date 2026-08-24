from __future__ import annotations

import re
from pathlib import Path

from bayesian_phystwin.mujoco_flex_source_v1 import MUJOCO_VERSION


def test_mujoco_flex_extra_matches_frozen_native_runtime() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r'^mujoco-flex\s*=\s*\[\s*"mujoco==([^"]+)"\s*,?\s*\]',
        pyproject,
        flags=re.MULTILINE,
    )
    assert match is not None, "mujoco-flex must contain one exact MuJoCo pin"
    assert match.group(1) == MUJOCO_VERSION
