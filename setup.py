"""Setuptools command customizations for the stable source distribution."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Final

from setuptools import find_namespace_packages, setup
from setuptools.command.sdist import sdist as _sdist

ROOT: Final = Path(__file__).resolve().parent
CONTRACT_PATH: Final = ROOT / "release" / "stable_distribution_contract_v1.json"
PACKAGE_INCLUDE: Final = ("bayesian_phystwin", "bayesian_phystwin.*")
PACKAGE_EXCLUDE: Final = (
    "bayesian_phystwin.experiments",
    "bayesian_phystwin.experiments.*",
    "bayesian_phystwin_experiments",
    "bayesian_phystwin_experiments.*",
)


def _supported_sdist_self_tests() -> frozenset[str]:
    """Read the exact supported self-test boundary from the release contract."""

    try:
        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        raw_tests = payload["sdist"]["supported_self_test_files"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("cannot read the stable sdist self-test boundary") from exc
    if (
        not isinstance(raw_tests, list)
        or not raw_tests
        or any(type(item) is not str or not item for item in raw_tests)
    ):
        raise RuntimeError("stable sdist self-test boundary is malformed")
    tests = frozenset(raw_tests)
    if len(tests) != len(raw_tests):
        raise RuntimeError("stable sdist self-test boundary contains duplicates")
    return tests


SUPPORTED_SDIST_SELF_TESTS: Final = _supported_sdist_self_tests()
DISCOVERED_PACKAGES: Final = tuple(
    find_namespace_packages(
        where="src",
        include=PACKAGE_INCLUDE,
        exclude=PACKAGE_EXCLUDE,
    )
)
if "bayesian_phystwin" not in DISCOVERED_PACKAGES:
    raise RuntimeError("stable package discovery did not find bayesian_phystwin")


def _retain_sdist_file(raw_path: str) -> bool:
    """Keep repository tests only when the stable contract names them."""

    normalized = PurePosixPath(raw_path.replace("\\", "/")).as_posix()
    return (
        not normalized.startswith("tests/") or normalized in SUPPORTED_SDIST_SELF_TESTS
    )


class StableSdist(_sdist):
    """Exclude the implicit setuptools test corpus from release source archives."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(
            base_dir,
            [path for path in files if _retain_sdist_file(path)],
        )


setup(
    cmdclass={"sdist": StableSdist},
    package_dir={"": "src"},
    packages=DISCOVERED_PACKAGES,
)
