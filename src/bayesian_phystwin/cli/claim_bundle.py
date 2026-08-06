"""Build or validate a portable claim-bearing evidence bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.claim_bundle_v1 import (
    ClaimBundleArtifactKind,
    ClaimBundleArtifactV1,
    build_claim_bundle,
    claim_bundle_artifact,
    load_claim_bundle,
    verify_claim_bundle_artifacts,
    write_claim_bundle,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name.strip(), Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    build = subparsers.add_parser(
        "build",
        help="validate inputs and write a content-addressed ClaimBundleV1",
    )
    build.add_argument("bundle", type=Path)
    build.add_argument("--run-manifest", type=Path, required=True)
    build.add_argument("--evidence-summary", type=Path, required=True)
    build.add_argument("--claim-binding", type=Path)
    build.add_argument("--figure", type=_named_path, action="append", default=[])
    build.add_argument("--table-data", type=_named_path, action="append", default=[])
    build.add_argument("--supporting", type=_named_path, action="append", default=[])
    build.add_argument(
        "--artifact-root",
        type=Path,
        default=Path.cwd(),
        help="root used for portable artifact paths and digests",
    )

    validate = subparsers.add_parser(
        "validate",
        help="validate bundle identity and optionally re-hash every artifact",
    )
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--artifact-root", type=Path)
    validate.add_argument(
        "--require-claim-binding",
        action="store_true",
        help="fail unless one claim-binding JSON artifact is present",
    )
    return parser


def _artifacts_for(
    values: Sequence[tuple[str, Path]],
    *,
    kind: ClaimBundleArtifactKind,
    root: Path,
) -> tuple[ClaimBundleArtifactV1, ...]:
    return tuple(
        claim_bundle_artifact(
            path,
            name=name,
            kind=kind,
            root=root,
        )
        for name, path in values
    )


def _additional_artifacts(
    args: argparse.Namespace,
    *,
    root: Path,
) -> tuple[ClaimBundleArtifactV1, ...]:
    return (
        *_artifacts_for(args.figure, kind="figure", root=root),
        *_artifacts_for(args.table_data, kind="table_data", root=root),
        *_artifacts_for(args.supporting, kind="supporting", root=root),
    )


def _build(args: argparse.Namespace) -> int:
    root = args.artifact_root.resolve()
    bundle = build_claim_bundle(
        run_manifest_path=args.run_manifest,
        evidence_summary_path=args.evidence_summary,
        artifact_root=root,
        claim_binding_path=args.claim_binding,
        additional_artifacts=_additional_artifacts(args, root=root),
    )
    write_claim_bundle(args.bundle, bundle)
    print(
        json.dumps(
            {
                "bundle": str(args.bundle.resolve()),
                "bundle_id": bundle.bundle_id,
                "schema_version": 1,
                "run_manifest_id": bundle.run_manifest_id,
                "evidence_fingerprint": bundle.evidence_fingerprint,
                "claim_count": len(bundle.claim_ids),
                "repository_count": len(bundle.repositories),
                "artifact_count": len(bundle.artifacts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    bundle = load_claim_bundle(args.bundle)
    claim_binding_present = any(
        artifact.kind == "claim_binding" for artifact in bundle.artifacts
    )
    if args.require_claim_binding and not claim_binding_present:
        raise ValueError("claim bundle has no claim-binding artifact")
    if args.artifact_root is not None:
        verify_claim_bundle_artifacts(bundle, root=args.artifact_root)
    print(
        json.dumps(
            {
                "status": "valid",
                "schema_version": 1,
                "bundle_id": bundle.bundle_id,
                "run_id": bundle.run_id,
                "classification": bundle.classification,
                "claim_count": len(bundle.claim_ids),
                "claim_binding": (
                    "present" if claim_binding_present else "absent"
                ),
                "artifacts_verified": args.artifact_root is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_name == "build":
        return _build(args)
    if args.command_name == "validate":
        return _validate(args)
    raise AssertionError(f"unhandled command: {args.command_name}")


if __name__ == "__main__":
    raise SystemExit(main())
