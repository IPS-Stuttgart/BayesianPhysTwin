from __future__ import annotations

import importlib.util
import io
import json
import stat
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "release" / "check_stable_distribution.py"


def _load_checker() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "bpt_stable_distribution_checker",
        CHECKER,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


checker = _load_checker()
StableDistributionError = checker.StableDistributionError


def _contract(root: Path) -> Path:
    api = root / "api.json"
    api.write_text(json.dumps({"symbols": ["A"]}) + "\n", encoding="utf-8")
    payload = {
        "schema": "bayesian-phystwin.stable-distribution-contract",
        "schema_version": 1,
        "compatibility_line": "0.4",
        "wheel": {
            "maximum_size_bytes": 200_000,
            "maximum_member_count": 20,
            "required_members": ["bayesian_phystwin/__init__.py"],
            "forbidden_member_prefixes": ["tests/", "tools/"],
            "console_scripts": {"bpt": "bayesian_phystwin:main"},
            "isolated_imports": [
                {
                    "module": "bayesian_phystwin",
                    "api_manifest": "api.json",
                    "forbidden_external_modules": ["xmlrpc"],
                    "forbidden_package_prefixes": [
                        "bayesian_phystwin.experiments"
                    ],
                }
            ],
        },
        "sdist": {
            "maximum_size_bytes": 200_000,
            "maximum_regular_member_count": 20,
            "required_members": [
                "release/stable_distribution_contract_v1.json",
                "docs/stable_distribution_contract_v1.md",
                "tools/release/check_stable_distribution.py",
            ],
            "supported_self_test_files": [
                "tests/test_stable_distribution_contract.py"
            ],
        },
    }
    path = root / "contract.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _wheel(
    root: Path,
    *,
    package_source: str = '__all__ = ["A"]\n\ndef main():\n    return 0\n',
    extra: dict[str, bytes | str | zipfile.ZipInfo] | None = None,
) -> Path:
    path = root / "fixture.whl"
    members: dict[str, bytes | str | zipfile.ZipInfo] = {
        "bayesian_phystwin/__init__.py": package_source,
        "fixture-0.4.0.dist-info/entry_points.txt": (
            "[console_scripts]\n"
            "bpt = bayesian_phystwin:main\n"
        ),
    }
    members.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            if isinstance(value, zipfile.ZipInfo):
                archive.writestr(value, b"link target")
            else:
                archive.writestr(name, value)
    return path


def _sdist(
    root: Path,
    *,
    link: bool = False,
    special: bool = False,
) -> Path:
    path = root / "fixture.tar.gz"
    prefix = "fixture-0.4.0"
    members = [
        "release/stable_distribution_contract_v1.json",
        "docs/stable_distribution_contract_v1.md",
        "tools/release/check_stable_distribution.py",
        "tests/test_stable_distribution_contract.py",
    ]
    with tarfile.open(path, "w:gz") as archive:
        for relative in members:
            data = relative.encode()
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(f"{prefix}/linked")
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            archive.addfile(info)
        if special:
            info = tarfile.TarInfo(f"{prefix}/fifo")
            info.type = tarfile.FIFOTYPE
            archive.addfile(info)
    return path


def _validate(root: Path, **wheel_kwargs: Any) -> dict[str, object]:
    return checker.validate_distributions(
        _contract(root),
        wheel_path=_wheel(root, **wheel_kwargs),
        sdist_path=_sdist(root),
        project_root=root,
        python_executable=sys.executable,
    )


def test_valid_distribution_contract_reports_stable_surface(tmp_path: Path) -> None:
    report = _validate(tmp_path)

    assert report["schema"] == "bayesian-phystwin.stable-distribution-report"
    assert report["compatibility_line"] == "0.4"
    wheel = report["wheel"]
    assert isinstance(wheel, dict)
    assert wheel["console_scripts"] == {"bpt": "bayesian_phystwin:main"}
    imports = wheel["isolated_imports"]
    assert isinstance(imports, list)
    assert imports[0]["module"] == "bayesian_phystwin"
    assert imports[0]["symbol_count"] == 1
    assert "numpy_version" in imports[0]["runtime"]


def test_wheel_rejects_repository_only_member(tmp_path: Path) -> None:
    with pytest.raises(StableDistributionError, match="repository-only"):
        _validate(tmp_path, extra={"tests/leak.py": ""})


def test_isolated_import_rejects_api_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    (tmp_path / "api.json").write_text(
        json.dumps({"symbols": ["B"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(StableDistributionError, match="public API differs"):
        checker.validate_distributions(
            contract,
            wheel_path=_wheel(tmp_path),
            sdist_path=_sdist(tmp_path),
            project_root=tmp_path,
            python_executable=sys.executable,
        )


def test_isolated_import_rejects_optional_dependency_leak(tmp_path: Path) -> None:
    with pytest.raises(StableDistributionError, match="forbidden external"):
        _validate(
            tmp_path,
            package_source=(
                "import xmlrpc.client\n"
                '__all__ = ["A"]\n'
                "def main():\n"
                "    return 0\n"
            ),
        )


def test_wheel_rejects_link_member(tmp_path: Path) -> None:
    link = zipfile.ZipInfo("bayesian_phystwin/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(StableDistributionError, match="link or special"):
        _validate(tmp_path, extra={"ignored": link})


@pytest.mark.parametrize("kind", ["link", "special"])
def test_sdist_rejects_links_and_special_members(
    tmp_path: Path,
    kind: str,
) -> None:
    with pytest.raises(StableDistributionError, match="link|special"):
        checker.validate_distributions(
            _contract(tmp_path),
            wheel_path=_wheel(tmp_path),
            sdist_path=_sdist(
                tmp_path,
                link=kind == "link",
                special=kind == "special",
            ),
            project_root=tmp_path,
            python_executable=sys.executable,
        )


def test_contract_rejects_relaxed_or_malformed_limits(tmp_path: Path) -> None:
    path = _contract(tmp_path)
    payload = json.loads(path.read_text())
    payload["wheel"]["maximum_member_count"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StableDistributionError, match="positive integer"):
        checker.load_contract(path)
