from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/release/build_release_evidence.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_release_evidence",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _write_project(root: Path, *, version: str = "0.4.0") -> None:
    files = {
        "pyproject.toml": (
            "[build-system]\n"
            'requires = ["setuptools"]\n\n'
            "[project]\n"
            'name = "bayesian-phystwin"\n'
            f'version = "{version}"\n'
        ),
        "CITATION.cff": f'version: "{version}"\n',
        "CHANGELOG.md": "# Changelog\n",
        "SUPPORT.md": "# Support\n",
        "THIRD_PARTY_NOTICES.md": "# Notices\n",
        f"api/root-public-api-v{'.'.join(version.split('.')[:2])}.json": "{}\n",
        "docs/public_api_policy.md": "# API policy\n",
        "docs/releasing.md": "# Releasing\n",
        "tools/quality/check_public_api.py": "# checker\n",
        "tools/release/build_release_evidence.py": TOOL_PATH.read_text(
            encoding="utf-8"
        ),
        "LICENSE": "MIT\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _metadata(version: str) -> bytes:
    return (
        f"Metadata-Version: 2.4\nName: bayesian-phystwin\nVersion: {version}\n\n"
    ).encode()


def _write_wheel(
    path: Path,
    *,
    version: str = "0.4.0",
    include_typed_marker: bool = True,
) -> None:
    dist_info = f"bayesian_phystwin-{version}.dist-info"
    members = {
        f"{dist_info}/METADATA": _metadata(version),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\n",
        f"{dist_info}/entry_points.txt": b"[console_scripts]\nbpt = example:main\n",
        f"{dist_info}/RECORD": b"",
        f"{dist_info}/licenses/LICENSE": b"MIT\n",
        f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md": b"Notices\n",
    }
    if include_typed_marker:
        members["bayesian_phystwin/py.typed"] = b""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _write_sdist(
    path: Path,
    project_root: Path,
    *,
    version: str = "0.4.0",
    add_link: bool = False,
) -> None:
    root = f"bayesian_phystwin-{version}"
    compatibility = ".".join(version.split(".")[:2])
    files = {
        "PKG-INFO": _metadata(version),
        "pyproject.toml": (project_root / "pyproject.toml").read_bytes(),
        "CITATION.cff": (project_root / "CITATION.cff").read_bytes(),
        "CHANGELOG.md": (project_root / "CHANGELOG.md").read_bytes(),
        "LICENSE": (project_root / "LICENSE").read_bytes(),
        "SUPPORT.md": (project_root / "SUPPORT.md").read_bytes(),
        "THIRD_PARTY_NOTICES.md": (
            project_root / "THIRD_PARTY_NOTICES.md"
        ).read_bytes(),
        f"api/root-public-api-v{compatibility}.json": (
            project_root / f"api/root-public-api-v{compatibility}.json"
        ).read_bytes(),
        "docs/public_api_policy.md": (
            project_root / "docs/public_api_policy.md"
        ).read_bytes(),
        "docs/releasing.md": (project_root / "docs/releasing.md").read_bytes(),
        "tools/quality/check_public_api.py": (
            project_root / "tools/quality/check_public_api.py"
        ).read_bytes(),
        "tools/release/build_release_evidence.py": (
            project_root / "tools/release/build_release_evidence.py"
        ).read_bytes(),
    }
    with tarfile.open(path, "w:gz") as archive:
        for relative, content in files.items():
            _add_tar_bytes(archive, f"{root}/{relative}", content)
        if add_link:
            link = tarfile.TarInfo(f"{root}/unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)


def _write_sbom(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "version": 1,
                "components": [{"type": "library", "name": "numpy"}],
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )


def _write_build_environment(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "bayesian-phystwin.release-build-environment",
                "schema_version": 1,
                "python_implementation": "CPython",
                "python_version": "3.12.12",
                "source_date_epoch": 1_754_700_000,
                "packages": {
                    "build": "1.5.0",
                    "pip": "26.1.2",
                    "pip-audit": "2.10.1",
                    "setuptools": "83.0.0",
                    "twine": "6.2.0",
                    "wheel": "0.47.0",
                },
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    dist = tmp_path / "dist"
    project.mkdir()
    dist.mkdir()
    _write_project(project)
    _write_wheel(dist / "bayesian_phystwin-0.4.0-py3-none-any.whl")
    _write_sdist(dist / "bayesian_phystwin-0.4.0.tar.gz", project)
    sbom = tmp_path / "release-sbom.cdx.json"
    _write_sbom(sbom)
    build_environment = tmp_path / "build-environment.json"
    _write_build_environment(build_environment)
    return project, dist, sbom, build_environment


def test_builds_content_addressed_release_evidence(tmp_path: Path) -> None:
    project, dist, sbom, build_environment = _fixture(tmp_path)

    evidence = tool.build_release_evidence(
        dist,
        sbom_path=sbom,
        build_environment_path=build_environment,
        source_revision="a" * 40,
        project_root=project,
        expected_tag="v0.4.0",
    )

    assert evidence["project_version"] == "0.4.0"
    assert evidence["release_tag"] == "v0.4.0"
    assert evidence["tag_validated"] is True
    assert evidence["artifacts"]["wheel"]["project_version"] == "0.4.0"
    assert evidence["artifacts"]["sdist"]["archive_root"] == ("bayesian_phystwin-0.4.0")
    assert evidence["artifacts"]["sbom"]["bom_format"] == "CycloneDX"

    supplied_id = evidence["evidence_id"]
    descriptor = dict(evidence)
    descriptor.pop("evidence_id")
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert supplied_id == hashlib.sha256(canonical).hexdigest()

    output = tmp_path / "release-evidence.json"
    tool.write_evidence(output, evidence)
    assert json.loads(output.read_text(encoding="utf-8")) == evidence
    with pytest.raises(tool.ReleaseEvidenceError, match="refusing to overwrite"):
        tool.write_evidence(output, evidence)


def test_release_tag_must_match_project_version(tmp_path: Path) -> None:
    project, dist, sbom, build_environment = _fixture(tmp_path)

    with pytest.raises(tool.ReleaseEvidenceError, match="release tag must be"):
        tool.build_release_evidence(
            dist,
            sbom_path=sbom,
            build_environment_path=build_environment,
            source_revision="b" * 40,
            project_root=project,
            expected_tag="v0.4.1",
        )


def test_wheel_must_retain_typing_and_license_members(tmp_path: Path) -> None:
    project, dist, sbom, build_environment = _fixture(tmp_path)
    wheel = next(dist.glob("*.whl"))
    wheel.unlink()
    _write_wheel(wheel, include_typed_marker=False)

    with pytest.raises(tool.ReleaseEvidenceError, match="missing required members"):
        tool.build_release_evidence(
            dist,
            sbom_path=sbom,
            build_environment_path=build_environment,
            source_revision="c" * 40,
            project_root=project,
        )


def test_sdist_rejects_links_even_without_extraction(tmp_path: Path) -> None:
    project, dist, sbom, build_environment = _fixture(tmp_path)
    sdist = next(dist.glob("*.tar.gz"))
    sdist.unlink()
    _write_sdist(sdist, project, add_link=True)

    with pytest.raises(tool.ReleaseEvidenceError, match="must not contain links"):
        tool.build_release_evidence(
            dist,
            sbom_path=sbom,
            build_environment_path=build_environment,
            source_revision="d" * 40,
            project_root=project,
        )


def test_sbom_and_source_revision_fail_closed(tmp_path: Path) -> None:
    project, dist, sbom, build_environment = _fixture(tmp_path)
    sbom.write_text('{"bomFormat": "SPDX"}', encoding="utf-8")

    with pytest.raises(tool.ReleaseEvidenceError, match="full lowercase"):
        tool.build_release_evidence(
            dist,
            sbom_path=sbom,
            build_environment_path=build_environment,
            source_revision="not-a-revision",
            project_root=project,
        )
    with pytest.raises(tool.ReleaseEvidenceError, match="not CycloneDX"):
        tool.build_release_evidence(
            dist,
            sbom_path=sbom,
            build_environment_path=build_environment,
            source_revision="e" * 40,
            project_root=project,
        )


def test_duplicate_project_versions_are_rejected(tmp_path: Path) -> None:
    project, dist, sbom, build_environment = _fixture(tmp_path)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8") + 'version = "0.4.1"\n',
        encoding="utf-8",
    )

    with pytest.raises(tool.ReleaseEvidenceError, match="one literal"):
        tool.build_release_evidence(
            dist,
            sbom_path=sbom,
            build_environment_path=build_environment,
            source_revision="f" * 40,
            project_root=project,
        )
