#!/usr/bin/env python3
"""Fail closed when built BayesianPhysTwin distributions exceed the stable surface."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

SCHEMA: Final = "bayesian-phystwin.stable-distribution-contract"
SCHEMA_VERSION: Final = 1
REPORT_SCHEMA: Final = "bayesian-phystwin.stable-distribution-report"
REPORT_SCHEMA_VERSION: Final = 1


class StableDistributionError(ValueError):
    """Raised when an archive or contract violates the stable release surface."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StableDistributionError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StableDistributionError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _literal(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise StableDistributionError(f"{name} must be a canonical nonempty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StableDistributionError(f"{name} must be a positive integer")
    return value


def _string_tuple(value: object, name: str, *, paths: bool = False) -> tuple[str, ...]:
    values = tuple(_literal(item, f"{name} item") for item in _sequence(value, name))
    if len(values) != len(set(values)):
        raise StableDistributionError(f"{name} contains duplicates")
    if paths:
        for item in values:
            _archive_path(item, name)
    return values


def _archive_path(value: str, name: str) -> str:
    if value.startswith("/") or "\\" in value:
        raise StableDistributionError(f"{name} contains a nonportable path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise StableDistributionError(f"{name} contains a noncanonical path")
    if PurePosixPath(value).as_posix() != value:
        raise StableDistributionError(f"{name} contains a noncanonical path")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise StableDistributionError(f"cannot read {path}") from error
    return digest.hexdigest()


def load_contract(path: str | Path) -> dict[str, object]:
    """Load and validate the version-1 distribution contract."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StableDistributionError(
            "cannot read stable distribution contract"
        ) from error
    contract = _mapping(raw, "contract")
    expected = {"schema", "schema_version", "compatibility_line", "wheel", "sdist"}
    if set(contract) != expected:
        raise StableDistributionError("stable distribution contract fields changed")
    if contract["schema"] != SCHEMA or contract["schema_version"] != SCHEMA_VERSION:
        raise StableDistributionError("stable distribution contract identity changed")
    compatibility = _literal(contract["compatibility_line"], "compatibility_line")
    if re.fullmatch(r"[0-9]+\.[0-9]+", compatibility) is None:
        raise StableDistributionError("compatibility_line is invalid")

    wheel = _mapping(contract["wheel"], "wheel")
    expected_wheel = {
        "maximum_size_bytes",
        "maximum_member_count",
        "required_members",
        "forbidden_member_prefixes",
        "console_scripts",
        "isolated_imports",
    }
    if set(wheel) != expected_wheel:
        raise StableDistributionError("wheel contract fields changed")
    _positive_integer(wheel["maximum_size_bytes"], "wheel maximum_size_bytes")
    _positive_integer(wheel["maximum_member_count"], "wheel maximum_member_count")
    _string_tuple(wheel["required_members"], "wheel required_members", paths=True)
    prefixes = _string_tuple(wheel["forbidden_member_prefixes"], "forbidden prefixes")
    if any(not prefix.endswith("/") for prefix in prefixes):
        raise StableDistributionError("forbidden wheel prefixes must end in '/'")
    scripts = _mapping(wheel["console_scripts"], "console_scripts")
    if not scripts:
        raise StableDistributionError("console_scripts must not be empty")
    for alias, target in scripts.items():
        _literal(alias, "console script name")
        if ":" not in _literal(target, "console script target"):
            raise StableDistributionError("console script target must contain ':'")
    imports = _sequence(wheel["isolated_imports"], "isolated_imports")
    if not imports:
        raise StableDistributionError("isolated_imports must not be empty")
    for index, item in enumerate(imports):
        specification = _mapping(item, f"isolated_imports[{index}]")
        expected_import = {
            "module",
            "api_manifest",
            "forbidden_external_modules",
            "forbidden_package_prefixes",
        }
        if set(specification) != expected_import:
            raise StableDistributionError("isolated import fields changed")
        _literal(specification["module"], "isolated module")
        _archive_path(
            _literal(specification["api_manifest"], "API manifest"), "API manifest"
        )
        _string_tuple(
            specification["forbidden_external_modules"], "forbidden externals"
        )
        _string_tuple(
            specification["forbidden_package_prefixes"], "forbidden package prefixes"
        )

    sdist = _mapping(contract["sdist"], "sdist")
    expected_sdist = {
        "maximum_size_bytes",
        "maximum_regular_member_count",
        "required_members",
        "supported_self_test_files",
    }
    if set(sdist) != expected_sdist:
        raise StableDistributionError("sdist contract fields changed")
    _positive_integer(sdist["maximum_size_bytes"], "sdist maximum_size_bytes")
    _positive_integer(sdist["maximum_regular_member_count"], "sdist member limit")
    _string_tuple(sdist["required_members"], "sdist required_members", paths=True)
    _string_tuple(
        sdist["supported_self_test_files"], "supported self tests", paths=True
    )
    return dict(contract)


def _zip_members(path: Path) -> tuple[dict[str, zipfile.ZipInfo], bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            entry_points: list[str] = []
            for item in archive.infolist():
                if item.is_dir():
                    continue
                name = _archive_path(item.filename, "wheel member")
                if name in members:
                    raise StableDistributionError("wheel contains duplicate members")
                mode = (item.external_attr >> 16) & 0o177777
                file_type = stat.S_IFMT(mode)
                if file_type and file_type != stat.S_IFREG:
                    raise StableDistributionError(
                        "wheel contains a link or special member"
                    )
                members[name] = item
                if name.endswith(".dist-info/entry_points.txt"):
                    entry_points.append(name)
            if len(entry_points) != 1:
                raise StableDistributionError("wheel must contain one entry_points.txt")
            return members, archive.read(entry_points[0])
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise StableDistributionError("cannot inspect wheel") from error


def _console_scripts(value: bytes) -> dict[str, str]:
    try:
        text = value.decode("utf-8")
    except UnicodeError as error:
        raise StableDistributionError("entry_points.txt is not UTF-8") from error
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read_string(text)
    except configparser.Error as error:
        raise StableDistributionError("entry_points.txt is invalid") from error
    return (
        dict(parser.items("console_scripts"))
        if parser.has_section("console_scripts")
        else {}
    )


def _isolated_import(
    wheel: Path,
    specification: Mapping[str, Any],
    *,
    project_root: Path,
    python_executable: str,
) -> dict[str, object]:
    module = cast(str, specification["module"])
    manifest_path = project_root / cast(str, specification["api_manifest"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StableDistributionError(
            f"cannot read API manifest for {module}"
        ) from error
    expected_symbols = sorted(
        _string_tuple(_mapping(manifest, "API manifest").get("symbols"), "symbols")
    )
    payload = {
        "wheel": str(wheel.resolve()),
        "module": module,
        "expected_symbols": expected_symbols,
        "forbidden_external_modules": list(specification["forbidden_external_modules"]),
        "forbidden_package_prefixes": list(specification["forbidden_package_prefixes"]),
    }
    script = r"""
import importlib, json, platform, sys
p=json.loads(sys.argv[1]); sys.path.insert(0,p["wheel"])
m=importlib.import_module(p["module"])
actual=sorted(getattr(m,"__all__",()))
if actual!=p["expected_symbols"]: raise SystemExit("public API differs from manifest")
loaded=set(sys.modules)
for name in p["forbidden_external_modules"]:
    if name in loaded or any(x.startswith(name+".") for x in loaded):
        raise SystemExit("forbidden external module loaded: "+name)
for prefix in p["forbidden_package_prefixes"]:
    if any(x==prefix or x.startswith(prefix+".") for x in loaded):
        raise SystemExit("forbidden package prefix loaded: "+prefix)
origin=str(getattr(m,"__file__", ""))
if p["wheel"] not in origin: raise SystemExit("module was not loaded from candidate wheel")
import numpy
print(json.dumps({"module":p["module"],"symbol_count":len(actual),"module_member":origin.split(p["wheel"]+"/",1)[-1],"runtime":{"python_implementation":platform.python_implementation(),"python_version":platform.python_version(),"numpy_version":numpy.__version__}},sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="bpt-stable-import-") as directory:
        process = subprocess.run(
            [
                python_executable,
                "-I",
                "-c",
                script,
                json.dumps(payload, sort_keys=True),
            ],
            cwd=directory,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown failure"
        raise StableDistributionError(f"isolated import {module} failed: {detail}")
    try:
        return cast(dict[str, object], json.loads(process.stdout))
    except json.JSONDecodeError as error:
        raise StableDistributionError(
            f"isolated import {module} returned invalid JSON"
        ) from error


def _sdist_members(path: Path) -> tuple[str, set[str]]:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            roots: set[str] = set()
            regular: set[str] = set()
            for item in archive.getmembers():
                raw = (
                    item.name[:-1]
                    if item.isdir() and item.name.endswith("/")
                    else item.name
                )
                name = _archive_path(raw, "sdist member")
                roots.add(name.split("/", 1)[0])
                if item.issym() or item.islnk():
                    raise StableDistributionError(
                        "sdist contains a symbolic or hard link"
                    )
                if not item.isdir() and not item.isfile():
                    raise StableDistributionError("sdist contains a special member")
                if item.isfile():
                    regular.add(name)
    except (OSError, tarfile.TarError) as error:
        raise StableDistributionError("cannot inspect sdist") from error
    if len(roots) != 1:
        raise StableDistributionError("sdist must contain one canonical archive root")
    return next(iter(roots)), regular


def validate_distributions(
    contract_path: str | Path,
    *,
    wheel_path: str | Path,
    sdist_path: str | Path,
    project_root: str | Path,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    """Validate both release archives and return a deterministic report."""

    contract_source = Path(contract_path).resolve(strict=True)
    wheel_source = Path(wheel_path).resolve(strict=True)
    sdist_source = Path(sdist_path).resolve(strict=True)
    root = Path(project_root).resolve(strict=True)
    contract = load_contract(contract_source)
    wheel_contract = cast(Mapping[str, Any], contract["wheel"])
    sdist_contract = cast(Mapping[str, Any], contract["sdist"])

    wheel_size = wheel_source.stat().st_size
    if wheel_size > cast(int, wheel_contract["maximum_size_bytes"]):
        raise StableDistributionError("wheel exceeds its size budget")
    wheel_members, entry_points = _zip_members(wheel_source)
    if len(wheel_members) > cast(int, wheel_contract["maximum_member_count"]):
        raise StableDistributionError("wheel exceeds its member-count budget")
    missing_wheel = set(
        cast(tuple[str, ...], wheel_contract["required_members"])
    ) - set(wheel_members)
    if missing_wheel:
        raise StableDistributionError(
            f"wheel is missing required members: {sorted(missing_wheel)}"
        )
    forbidden = [
        member
        for member in wheel_members
        if member.startswith(
            tuple(cast(Sequence[str], wheel_contract["forbidden_member_prefixes"]))
        )
    ]
    if forbidden:
        raise StableDistributionError(
            f"wheel contains repository-only members: {forbidden}"
        )
    scripts = _console_scripts(entry_points)
    expected_scripts = dict(cast(Mapping[str, str], wheel_contract["console_scripts"]))
    if scripts != expected_scripts:
        raise StableDistributionError(f"wheel console scripts changed: {scripts}")
    imports = [
        _isolated_import(
            wheel_source,
            cast(Mapping[str, Any], item),
            project_root=root,
            python_executable=python_executable,
        )
        for item in cast(
            Sequence[Mapping[str, Any]], wheel_contract["isolated_imports"]
        )
    ]

    sdist_size = sdist_source.stat().st_size
    if sdist_size > cast(int, sdist_contract["maximum_size_bytes"]):
        raise StableDistributionError("sdist exceeds its size budget")
    sdist_root, sdist_members = _sdist_members(sdist_source)
    if len(sdist_members) > cast(int, sdist_contract["maximum_regular_member_count"]):
        raise StableDistributionError("sdist exceeds its member-count budget")
    required_relative = set(
        cast(tuple[str, ...], sdist_contract["required_members"])
    ) | set(cast(tuple[str, ...], sdist_contract["supported_self_test_files"]))
    missing_sdist = {
        item
        for item in required_relative
        if f"{sdist_root}/{item}" not in sdist_members
    }
    if missing_sdist:
        raise StableDistributionError(
            f"sdist is missing stable members: {sorted(missing_sdist)}"
        )

    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "compatibility_line": contract["compatibility_line"],
        "contract": {
            "path": contract_source.relative_to(root).as_posix(),
            "sha256": _sha256(contract_source),
        },
        "wheel": {
            "path": wheel_source.name,
            "sha256": _sha256(wheel_source),
            "size_bytes": wheel_size,
            "maximum_size_bytes": wheel_contract["maximum_size_bytes"],
            "member_count": len(wheel_members),
            "maximum_member_count": wheel_contract["maximum_member_count"],
            "console_scripts": expected_scripts,
            "isolated_imports": imports,
        },
        "sdist": {
            "path": sdist_source.name,
            "sha256": _sha256(sdist_source),
            "size_bytes": sdist_size,
            "maximum_size_bytes": sdist_contract["maximum_size_bytes"],
            "archive_root": sdist_root,
            "regular_member_count": len(sdist_members),
            "maximum_regular_member_count": sdist_contract[
                "maximum_regular_member_count"
            ],
            "supported_self_test_files": list(
                sdist_contract["supported_self_test_files"]
            ),
        },
        "claim_boundary": (
            "Distribution membership, size, entry-point, public-API, and isolated-import "
            "conformance only; not scientific evidence or deployment approval."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = validate_distributions(
            arguments.contract,
            wheel_path=arguments.wheel,
            sdist_path=arguments.sdist,
            project_root=arguments.project_root,
            python_executable=arguments.python,
        )
    except StableDistributionError as error:
        raise SystemExit(str(error)) from error
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
