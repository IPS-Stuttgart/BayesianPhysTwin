"""Create or validate a content-addressed Bayesian-PhysTwin run manifest."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin.run_manifest import (
    RunManifestV1,
    artifact_digest,
    installed_package_versions,
    load_run_manifest,
    verify_run_manifest_artifacts,
    write_run_manifest,
)


def _load_json_mapping(path: Path | None, *, name: str) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} JSON must contain an object")
    return dict(value)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    create = subparsers.add_parser("create", help="hash artifacts and write a manifest")
    create.add_argument("manifest", type=Path)
    create.add_argument("--run-id", required=True)
    create.add_argument(
        "--repository",
        default="FlorianPfaff/Bayesian-PhysTwin",
    )
    create.add_argument("--revision", required=True)
    create.add_argument("--dirty", action="store_true")
    create.add_argument(
        "--classification",
        choices=(
            "controlled",
            "exploratory",
            "confirmatory",
            "diagnostic",
            "infrastructure",
        ),
        required=True,
    )
    create.add_argument("--statistical-unit", required=True)
    create.add_argument("--command-line", required=True)
    create.add_argument("--configuration-json", type=Path)
    create.add_argument("--information-boundary-json", type=Path)
    create.add_argument("--seed", type=int, action="append", default=[])
    create.add_argument("--input", type=_named_path, action="append", default=[])
    create.add_argument(
        "--output-artifact",
        type=_named_path,
        action="append",
        default=[],
    )
    create.add_argument("--artifact-root", type=Path, default=Path.cwd())
    create.add_argument(
        "--package",
        action="append",
        default=["bayesian-phystwin", "numpy"],
    )
    create.add_argument("--notes", default="")

    validate = subparsers.add_parser("validate", help="validate a saved manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--artifact-root", type=Path)
    return parser


def _create(args: argparse.Namespace) -> int:
    root = args.artifact_root.resolve()
    inputs = tuple(
        artifact_digest(path, name=name, role="input", root=root)
        for name, path in args.input
    )
    outputs = tuple(
        artifact_digest(path, name=name, role="output", root=root)
        for name, path in args.output_artifact
    )
    manifest = RunManifestV1(
        run_id=args.run_id,
        repository=args.repository,
        revision=args.revision,
        dirty=args.dirty,
        command=tuple(shlex.split(args.command_line)),
        classification=args.classification,
        statistical_unit=args.statistical_unit,
        information_boundary=_load_json_mapping(
            args.information_boundary_json,
            name="information boundary",
        ),
        configuration=_load_json_mapping(
            args.configuration_json,
            name="configuration",
        ),
        seeds=tuple(args.seed),
        inputs=inputs,
        outputs=outputs,
        package_versions=installed_package_versions(args.package),
        notes=args.notes,
    )
    write_run_manifest(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "manifest_id": manifest.manifest_id,
                "input_count": len(manifest.inputs),
                "output_count": len(manifest.outputs),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    manifest = load_run_manifest(args.manifest)
    if args.artifact_root is not None:
        verify_run_manifest_artifacts(manifest, root=args.artifact_root)
    print(
        json.dumps(
            {
                "status": "valid",
                "manifest_id": manifest.manifest_id,
                "run_id": manifest.run_id,
                "classification": manifest.classification,
                "artifacts_verified": args.artifact_root is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_name == "create":
        return _create(args)
    if args.command_name == "validate":
        return _validate(args)
    raise AssertionError(f"unhandled command: {args.command_name}")


if __name__ == "__main__":
    raise SystemExit(main())
