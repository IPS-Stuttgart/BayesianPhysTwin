#!/usr/bin/env python3
"""Build a content-addressed evidence record for BayesianPhysTwin artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

SCHEMA: Final = "bayesian-phystwin.release-candidate-evidence"
SCHEMA_VERSION: Final = 1
PROJECT_NAME: Final = "bayesian-phystwin"
PACKAGE_NAME: Final = "bayesian_phystwin"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$")
_PROJECT_SECTION = re.compile(r"^\s*\[project\]\s*(?:#.*)?$")
_SECTION = re.compile(r"^\s*\[[^]]+\]\s*(?:#.*)?$")
_PROJECT_VALUE = re.compile(
    r"^\s*(name|version)\s*=\s*(['\"])([^'\"]+)\2\s*(?:#.*)?$"
)
_CITATION_VERSION = re.compile(
    r"^version:\s*(?:\"([^\"]+)\"|'([^']+)'|([^#\n]+?))\s*(?:#.*)?$",
    re.MULTILINE,
)


class ReleaseEvidenceError(ValueError):
    """Raised when a release candidate is incomplete or inconsistent."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReleaseEvidenceError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ReleaseEvidenceError(f"{name} must be a nonempty canonical string")
    return value


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _archive_stem(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).lower()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ReleaseEvidenceError(f"cannot read {path}") from error
    return {
        "path": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _source_record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    record = _file_record(path)
    record["path"] = relative
    return record


def _canonical_archive_path(value: str, *, name: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ReleaseEvidenceError(f"{name} contains a nonportable path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ReleaseEvidenceError(f"{name} contains a noncanonical path")
    canonical = PurePosixPath(value).as_posix()
    if canonical != value:
        raise ReleaseEvidenceError(f"{name} contains a noncanonical path")
    return canonical


def _project_metadata(path: Path) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ReleaseEvidenceError("cannot read pyproject.toml") from error

    in_project = False
    values: dict[str, list[str]] = {"name": [], "version": []}
    for line in lines:
        if _PROJECT_SECTION.fullmatch(line):
            in_project = True
            continue
        if _SECTION.fullmatch(line):
            in_project = False
            continue
        if not in_project:
            continue
        match = _PROJECT_VALUE.fullmatch(line)
        if match is not None:
            values[match.group(1)].append(match.group(3))

    if len(values["name"]) != 1 or len(values["version"]) != 1:
        raise ReleaseEvidenceError(
            "pyproject.toml must declare one literal project name and version"
        )
    name = values["name"][0]
    version = values["version"][0]
    if _normalized_distribution_name(name) != PROJECT_NAME:
        raise ReleaseEvidenceError("unexpected project name")
    if _VERSION.fullmatch(version) is None:
        raise ReleaseEvidenceError("project version is not a supported literal version")
    return name, version


def _citation_version(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseEvidenceError("cannot read CITATION.cff") from error
    matches = list(_CITATION_VERSION.finditer(text))
    if len(matches) != 1:
        raise ReleaseEvidenceError("CITATION.cff must declare one version")
    version = next(group for group in matches[0].groups() if group is not None).strip()
    if _VERSION.fullmatch(version) is None:
        raise ReleaseEvidenceError("CITATION.cff version is invalid")
    return version


def _message_metadata(value: bytes, *, name: str) -> dict[str, str]:
    try:
        text = value.decode("utf-8")
    except UnicodeError as error:
        raise ReleaseEvidenceError(f"{name} is not UTF-8") from error
    message = Parser().parsestr(text)
    project = message.get("Name")
    version = message.get("Version")
    if project is None or version is None:
        raise ReleaseEvidenceError(f"{name} lacks Name or Version metadata")
    return {"name": project, "version": version}


def _validate_metadata(
    metadata: Mapping[str, str],
    *,
    expected_name: str,
    expected_version: str,
    name: str,
) -> None:
    if _normalized_distribution_name(metadata["name"]) != (
        _normalized_distribution_name(expected_name)
    ):
        raise ReleaseEvidenceError(f"{name} project name does not match pyproject.toml")
    if metadata["version"] != expected_version:
        raise ReleaseEvidenceError(f"{name} version does not match pyproject.toml")


def _wheel_record(path: Path, *, project: str, version: str) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            canonical = [
                _canonical_archive_path(name, name="wheel") for name in names
            ]
            if len(canonical) != len(set(canonical)):
                raise ReleaseEvidenceError("wheel contains duplicate members")
            metadata_members = [
                name for name in canonical if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_members) != 1:
                raise ReleaseEvidenceError("wheel must contain exactly one METADATA")
            metadata_member = metadata_members[0]
            dist_info = metadata_member.removesuffix("/METADATA")
            expected_dist_info = f"{_archive_stem(project)}-{version}.dist-info"
            if dist_info != expected_dist_info:
                raise ReleaseEvidenceError("wheel dist-info directory is inconsistent")
            required = {
                metadata_member,
                f"{dist_info}/WHEEL",
                f"{dist_info}/entry_points.txt",
                f"{dist_info}/RECORD",
                f"{dist_info}/licenses/LICENSE",
                f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md",
                f"{PACKAGE_NAME}/py.typed",
            }
            missing = required - set(canonical)
            if missing:
                raise ReleaseEvidenceError(
                    f"wheel is missing required members: {sorted(missing)}"
                )
            metadata_bytes = archive.read(metadata_member)
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ReleaseEvidenceError("cannot inspect wheel") from error

    metadata = _message_metadata(metadata_bytes, name="wheel METADATA")
    _validate_metadata(
        metadata,
        expected_name=project,
        expected_version=version,
        name="wheel",
    )
    record = _file_record(path)
    record.update(
        {
            "archive_kind": "wheel",
            "member_count": len(canonical),
            "metadata_member": metadata_member,
            "metadata_sha256": _sha256_bytes(metadata_bytes),
            "project_name": metadata["name"],
            "project_version": metadata["version"],
        }
    )
    return record


def _sdist_record(
    path: Path,
    *,
    project: str,
    version: str,
    api_manifest: str,
) -> dict[str, object]:
    expected_root = f"{_archive_stem(project)}-{version}"
    required_relative = {
        "PKG-INFO",
        "pyproject.toml",
        "CITATION.cff",
        "CHANGELOG.md",
        "LICENSE",
        "SUPPORT.md",
        "THIRD_PARTY_NOTICES.md",
        api_manifest,
        "docs/public_api_policy.md",
        "docs/releasing.md",
        "tools/quality/check_public_api.py",
        "tools/release/build_release_evidence.py",
    }
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            regular: dict[str, tarfile.TarInfo] = {}
            for item in archive.getmembers():
                raw_name = (
                    item.name[:-1]
                    if item.isdir() and item.name.endswith("/")
                    else item.name
                )
                canonical = _canonical_archive_path(raw_name, name="sdist")
                if item.issym() or item.islnk():
                    raise ReleaseEvidenceError("sdist must not contain links")
                if item.isfile():
                    if canonical in regular:
                        raise ReleaseEvidenceError("sdist contains duplicate members")
                    regular[canonical] = item
            expected = {f"{expected_root}/{relative}" for relative in required_relative}
            missing = expected - set(regular)
            if missing:
                raise ReleaseEvidenceError(
                    f"sdist is missing required members: {sorted(missing)}"
                )
            pkg_info_member = f"{expected_root}/PKG-INFO"
            stream = archive.extractfile(regular[pkg_info_member])
            if stream is None:
                raise ReleaseEvidenceError("cannot read sdist PKG-INFO")
            pkg_info_bytes = stream.read()
    except (OSError, tarfile.TarError) as error:
        raise ReleaseEvidenceError("cannot inspect sdist") from error

    metadata = _message_metadata(pkg_info_bytes, name="sdist PKG-INFO")
    _validate_metadata(
        metadata,
        expected_name=project,
        expected_version=version,
        name="sdist",
    )
    record = _file_record(path)
    record.update(
        {
            "archive_kind": "sdist",
            "archive_root": expected_root,
            "regular_member_count": len(regular),
            "pkg_info_sha256": _sha256_bytes(pkg_info_bytes),
            "project_name": metadata["name"],
            "project_version": metadata["version"],
        }
    )
    return record


def _sbom_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("cannot read CycloneDX SBOM") from error
    sbom = _mapping(value, name="CycloneDX SBOM")
    if sbom.get("bomFormat") != "CycloneDX":
        raise ReleaseEvidenceError("SBOM is not CycloneDX")
    specification = _literal(sbom.get("specVersion"), name="SBOM specVersion")
    if re.fullmatch(r"[0-9]+\.[0-9]+", specification) is None:
        raise ReleaseEvidenceError("SBOM specVersion is invalid")
    components = _sequence(sbom.get("components", []), name="SBOM components")
    dependencies = _sequence(sbom.get("dependencies", []), name="SBOM dependencies")
    record = _file_record(path)
    record.update(
        {
            "archive_kind": "cyclonedx-json",
            "bom_format": "CycloneDX",
            "spec_version": specification,
            "component_count": len(components),
            "dependency_record_count": len(dependencies),
        }
    )
    return record


def _build_environment_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError("cannot read build environment") from error
    environment = _mapping(value, name="build environment")
    expected_fields = {
        "schema",
        "schema_version",
        "python_implementation",
        "python_version",
        "source_date_epoch",
        "packages",
    }
    if set(environment) != expected_fields:
        raise ReleaseEvidenceError("build environment fields changed")
    if environment["schema"] != "bayesian-phystwin.release-build-environment":
        raise ReleaseEvidenceError("build environment schema changed")
    if environment["schema_version"] != 1:
        raise ReleaseEvidenceError("build environment version changed")
    implementation = _literal(
        environment["python_implementation"],
        name="python implementation",
    )
    python_version = _literal(
        environment["python_version"],
        name="python version",
    )
    epoch = environment["source_date_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ReleaseEvidenceError(
            "source_date_epoch must be a nonnegative integer"
        )
    raw_packages = _mapping(environment["packages"], name="build packages")
    packages: dict[str, str] = {}
    for raw_name, raw_version in raw_packages.items():
        name = _literal(raw_name, name="build package name")
        version = _literal(raw_version, name=f"build package {name} version")
        normalized = _normalized_distribution_name(name)
        if normalized in packages:
            raise ReleaseEvidenceError(
                "build package names collide after normalization"
            )
        packages[normalized] = version
    required = {"build", "pip", "pip-audit", "setuptools", "twine", "wheel"}
    missing = required - set(packages)
    if missing:
        raise ReleaseEvidenceError(
            f"build environment is missing required packages: {sorted(missing)}"
        )
    record = _file_record(path)
    record.update(
        {
            "schema": "bayesian-phystwin.release-build-environment",
            "schema_version": 1,
            "python_implementation": implementation,
            "python_version": python_version,
            "source_date_epoch": epoch,
            "packages": dict(sorted(packages.items())),
        }
    )
    return record


def _artifact_paths(dist_dir: Path) -> tuple[Path, Path]:
    try:
        wheels = sorted(path for path in dist_dir.iterdir() if path.suffix == ".whl")
        sdists = sorted(
            path for path in dist_dir.iterdir() if path.name.endswith(".tar.gz")
        )
    except OSError as error:
        raise ReleaseEvidenceError("cannot enumerate distribution directory") from error
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseEvidenceError(
            "distribution directory must contain exactly one wheel and one sdist"
        )
    return wheels[0], sdists[0]


def _evidence_id(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def build_release_evidence(
    dist_dir: str | Path,
    *,
    sbom_path: str | Path,
    build_environment_path: str | Path,
    source_revision: str,
    project_root: str | Path,
    expected_tag: str | None = None,
) -> dict[str, object]:
    """Validate release artifacts and return their content-addressed evidence."""

    root = Path(project_root).resolve()
    dist = Path(dist_dir).resolve()
    sbom = Path(sbom_path).resolve()
    build_environment = Path(build_environment_path).resolve()
    if _GIT_SHA.fullmatch(source_revision) is None:
        raise ReleaseEvidenceError(
            "source_revision must be a full lowercase 40-character Git SHA"
        )

    project, version = _project_metadata(root / "pyproject.toml")
    citation_version = _citation_version(root / "CITATION.cff")
    if citation_version != version:
        raise ReleaseEvidenceError("CITATION.cff version does not match pyproject.toml")
    release_tag = f"v{version}"
    if expected_tag is not None and expected_tag != release_tag:
        raise ReleaseEvidenceError(
            f"release tag must be {release_tag!r}, got {expected_tag!r}"
        )

    compatibility_line = ".".join(version.split(".")[:2])
    api_manifest = f"api/root-public-api-v{compatibility_line}.json"
    api_path = root / api_manifest
    if not api_path.is_file():
        raise ReleaseEvidenceError(
            f"release lacks the {compatibility_line} root API snapshot"
        )

    wheel, sdist = _artifact_paths(dist)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "project_name": project,
        "project_version": version,
        "release_tag": release_tag,
        "tag_validated": expected_tag is not None,
        "source_revision": source_revision,
        "build_environment": _build_environment_record(build_environment),
        "artifacts": {
            "wheel": _wheel_record(wheel, project=project, version=version),
            "sdist": _sdist_record(
                sdist,
                project=project,
                version=version,
                api_manifest=api_manifest,
            ),
            "sbom": _sbom_record(sbom),
        },
        "source_contracts": {
            "pyproject": _source_record(root, "pyproject.toml"),
            "citation": _source_record(root, "CITATION.cff"),
            "changelog": _source_record(root, "CHANGELOG.md"),
            "support": _source_record(root, "SUPPORT.md"),
            "public_api_manifest": _source_record(root, api_manifest),
        },
        "claim_boundary": (
            "Build, metadata, archive-membership, SBOM, and provenance evidence "
            "only. This record is not a PyPI publication, scientific result, "
            "deployment approval, or empirical accuracy claim."
        ),
    }
    payload["evidence_id"] = _evidence_id(payload)
    return payload


def write_evidence(path: str | Path, evidence: Mapping[str, object]) -> None:
    """Publish evidence atomically without overwriting an existing record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ReleaseEvidenceError(
                f"refusing to overwrite {destination}"
            ) from error
        except OSError as error:
            raise ReleaseEvidenceError(
                f"cannot publish release evidence to {destination}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--build-environment", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-tag")
    parser.add_argument("--output", type=Path, default=Path("release-evidence.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        evidence = build_release_evidence(
            arguments.dist_dir,
            sbom_path=arguments.sbom,
            build_environment_path=arguments.build_environment,
            source_revision=arguments.source_revision,
            project_root=arguments.project_root,
            expected_tag=arguments.expected_tag,
        )
        write_evidence(arguments.output, evidence)
    except ReleaseEvidenceError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
