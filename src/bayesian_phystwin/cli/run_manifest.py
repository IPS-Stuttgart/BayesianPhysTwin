"""Create or validate a content-addressed Bayesian-PhysTwin run manifest."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin.paper_evidence_v1 import (
    PAPER_EVIDENCE_PROFILE_KEY,
    embed_paper_evidence_bindings,
    load_paper_evidence_bindings,
    validate_paper_evidence_manifest,
)
from bayesian_phystwin.repository_provenance import (
    RepositoryRole,
    RepositoryState,
    default_runtime_environment,
    discover_git_repository_state,
)
from bayesian_phystwin.run_manifest import (
    artifact_digest,
    installed_package_versions,
)
from bayesian_phystwin.run_manifest_v2 import (
    RunManifestV2,
    load_run_manifest,
    verify_run_manifest_artifacts,
    write_run_manifest,
)

_DEFAULT_REPOSITORY = "FlorianPfaff/Bayesian-PhysTwin"
_REPOSITORY_FIELDS = frozenset({"repository", "revision", "dirty", "role"})
_RELATED_REPOSITORY_ROLES = frozenset(
    {
        "upstream",
        "observation",
        "downstream",
        "paper",
        "environment",
        "dependency",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_value(path: Path, *, name: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} JSON cannot be read as UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} JSON is malformed") from error


def _load_json_mapping(path: Path | None, *, name: str) -> dict[str, Any]:
    if path is None:
        return {}
    value = _load_json_value(path, name=name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} JSON must contain an object")
    return dict(value)


def _load_repository_states(path: Path | None) -> tuple[RepositoryState, ...]:
    if path is None:
        return ()
    value = _load_json_value(path, name="related repositories")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("related repositories JSON must contain an array")
    states: list[RepositoryState] = []
    for position, raw_record in enumerate(value):
        if not isinstance(raw_record, Mapping):
            raise ValueError(
                f"related repository record {position} must contain an object"
            )
        actual = frozenset(raw_record)
        if actual != _REPOSITORY_FIELDS:
            raise ValueError(
                "related repository records require exactly "
                f"{sorted(_REPOSITORY_FIELDS)}"
            )
        repository = raw_record["repository"]
        revision = raw_record["revision"]
        role = raw_record["role"]
        dirty = raw_record["dirty"]
        if type(repository) is not str:
            raise ValueError("related repository name must be a genuine string")
        if type(revision) is not str:
            raise ValueError("related repository revision must be a genuine string")
        if type(role) is not str or role not in _RELATED_REPOSITORY_ROLES:
            raise ValueError("related repository role is unsupported")
        if type(dirty) is not bool:
            raise ValueError("related repository dirty field must be boolean")
        states.append(
            RepositoryState(
                repository=repository,
                revision=revision,
                dirty=dirty,
                role=cast(RepositoryRole, role),
            )
        )
    return tuple(states)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def _resolve_artifact_path(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _primary_repository_state(args: argparse.Namespace) -> RepositoryState:
    if args.revision is not None:
        state = RepositoryState(
            repository=args.repository or _DEFAULT_REPOSITORY,
            revision=args.revision,
            dirty=bool(args.dirty),
            role="primary",
        )
    else:
        if args.dirty:
            raise ValueError("--dirty requires an explicit --revision")
        state = discover_git_repository_state(
            args.repository_root,
            repository=args.repository,
        )
    if state.dirty and not args.allow_dirty:
        raise ValueError(
            "repository checkout is dirty; pass --allow-dirty to record and "
            "acknowledge that state"
        )
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    create = subparsers.add_parser("create", help="hash artifacts and write a manifest")
    create.add_argument("manifest", type=Path)
    create.add_argument("--run-id", required=True)
    create.add_argument(
        "--repository",
        help="override the GitHub owner/name inferred from --repository-root",
    )
    create.add_argument(
        "--revision",
        help="explicit exact revision; otherwise discover it from Git",
    )
    create.add_argument(
        "--dirty",
        action="store_true",
        help="record a dirty checkout when --revision is supplied explicitly",
    )
    create.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Git checkout used for automatic repository-state discovery",
    )
    create.add_argument(
        "--allow-dirty",
        action="store_true",
        help="acknowledge a dirty primary checkout instead of failing closed",
    )
    create.add_argument(
        "--related-repositories-json",
        type=Path,
        help="JSON array of exact participating repository states",
    )
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
    create.add_argument(
        "--paper-evidence-json",
        type=Path,
        help=(
            "strict paper-evidence profile binding provider, stream, "
            "artifact, and distribution identities"
        ),
    )
    create.add_argument("--runtime-json", type=Path)
    create.add_argument(
        "--environment-variable",
        action="append",
        default=[],
        help="record one explicitly named environment variable",
    )
    create.add_argument("--claim-id", action="append", default=[])
    create.add_argument("--method-freeze-id", default="")
    create.add_argument("--protocol-id", default="")
    create.add_argument("--split-id", default="")
    create.add_argument("--baseline-id", default="")
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
    validate.add_argument(
        "--require-paper-evidence",
        action="store_true",
        help="fail unless the manifest has a valid paper-evidence profile",
    )
    return parser


def _create(args: argparse.Namespace) -> int:
    root = args.artifact_root.resolve()
    primary = _primary_repository_state(args)
    inputs = tuple(
        artifact_digest(
            _resolve_artifact_path(path, root=root),
            name=name,
            role="input",
            root=root,
        )
        for name, path in args.input
    )
    outputs = tuple(
        artifact_digest(
            _resolve_artifact_path(path, root=root),
            name=name,
            role="output",
            root=root,
        )
        for name, path in args.output_artifact
    )
    information_boundary = _load_json_mapping(
        args.information_boundary_json,
        name="information boundary",
    )
    paper_evidence_requested = args.paper_evidence_json is not None
    if paper_evidence_requested:
        information_boundary = embed_paper_evidence_bindings(
            information_boundary,
            load_paper_evidence_bindings(args.paper_evidence_json),
        )

    manifest = RunManifestV2(
        run_id=args.run_id,
        repository=primary.repository,
        revision=primary.revision,
        dirty=primary.dirty,
        related_repositories=_load_repository_states(args.related_repositories_json),
        command=tuple(shlex.split(args.command_line)),
        classification=args.classification,
        statistical_unit=args.statistical_unit,
        information_boundary=information_boundary,
        configuration=_load_json_mapping(
            args.configuration_json,
            name="configuration",
        ),
        seeds=tuple(args.seed),
        inputs=inputs,
        outputs=outputs,
        package_versions=installed_package_versions(args.package),
        runtime_environment=default_runtime_environment(
            overrides=_load_json_mapping(args.runtime_json, name="runtime"),
            environment_variables=args.environment_variable,
        ),
        claim_ids=tuple(args.claim_id),
        method_freeze_id=args.method_freeze_id,
        protocol_id=args.protocol_id,
        split_id=args.split_id,
        baseline_id=args.baseline_id,
        notes=args.notes,
    )
    if paper_evidence_requested:
        validate_paper_evidence_manifest(manifest)
    write_run_manifest(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "schema_version": 2,
                "manifest_id": manifest.manifest_id,
                "evidence_fingerprint": manifest.evidence_fingerprint,
                "repository_count": 1 + len(manifest.related_repositories),
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
    summary: dict[str, object] = {
        "status": "valid",
        "manifest_id": manifest.manifest_id,
        "run_id": manifest.run_id,
        "classification": manifest.classification,
        "artifacts_verified": args.artifact_root is not None,
    }
    if isinstance(manifest, RunManifestV2):
        summary["schema_version"] = 2
        summary["evidence_fingerprint"] = manifest.evidence_fingerprint
        paper_evidence_present = (
            PAPER_EVIDENCE_PROFILE_KEY in manifest.information_boundary
        )
        if paper_evidence_present:
            validate_paper_evidence_manifest(manifest)
        elif args.require_paper_evidence:
            raise ValueError("run manifest has no paper-evidence profile")
        summary["paper_evidence_profile"] = (
            "valid" if paper_evidence_present else "absent"
        )
    else:
        if args.require_paper_evidence:
            raise ValueError("paper-evidence profile requires RunManifestV2")
        summary["schema_version"] = 1
    print(json.dumps(summary, indent=2, sort_keys=True))
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
