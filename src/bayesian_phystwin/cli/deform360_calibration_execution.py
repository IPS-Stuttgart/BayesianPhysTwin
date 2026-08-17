"""Seal one complete Deform360 calibration execution before confirmation access."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from bayesian_phystwin._portable_contracts import (
    exact_revision,
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    save_deform360_calibration_bundle,
)
from bayesian_phystwin.deform360_calibration_execution import (
    build_deform360_calibration_execution_seal,
    file_sha256,
    load_deform360_calibration_artifact_ref,
    load_deform360_stage0_selection,
    save_deform360_calibration_execution_seal,
    verify_deform360_calibration_execution_artifacts,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    load_deform360_visual_provider_lock,
    save_deform360_visual_calibration_lock,
)
from bayesian_phystwin.evidence_use_ledger import (
    load_evidence_use_ledger,
)

_COMMITTED_SELECTION = (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
_REPOSITORY_SOURCES = (
    ".github/workflows/deform360-calibration-seal.yml",
    "protocols/deform360_official_hub_visuotactile_v1.json",
    _COMMITTED_SELECTION,
    (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    ),
    (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    ),
    "src/bayesian_phystwin/_canonical_contracts.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/deform360_calibration_execution.py",
    "src/bayesian_phystwin/deform360_calibration_bundle.py",
    "src/bayesian_phystwin/deform360_visual_provider_lock.py",
    "src/bayesian_phystwin/evidence_use_ledger.py",
    "src/bayesian_phystwin/cli/deform360_calibration_execution.py",
    "src/bayesian_phystwin/cli/experiments.py",
)
_RUNTIME_MODULE_SOURCES = {
    "bayesian_phystwin._canonical_contracts": (
        "src/bayesian_phystwin/_canonical_contracts.py"
    ),
    "bayesian_phystwin._portable_contracts": (
        "src/bayesian_phystwin/_portable_contracts.py"
    ),
    "bayesian_phystwin.deform360_calibration_execution": (
        "src/bayesian_phystwin/deform360_calibration_execution.py"
    ),
    "bayesian_phystwin.deform360_calibration_bundle": (
        "src/bayesian_phystwin/deform360_calibration_bundle.py"
    ),
    "bayesian_phystwin.deform360_visual_provider_lock": (
        "src/bayesian_phystwin/deform360_visual_provider_lock.py"
    ),
    "bayesian_phystwin.evidence_use_ledger": (
        "src/bayesian_phystwin/evidence_use_ledger.py"
    ),
    "bayesian_phystwin.cli.deform360_calibration_execution": (
        "src/bayesian_phystwin/cli/deform360_calibration_execution.py"
    ),
}


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def _artifact_path(value: str) -> tuple[str, Path]:
    role, path = _named_path(value)
    if role not in DEFORM360_CALIBRATION_ROLES:
        raise argparse.ArgumentTypeError(
            "ROLE must be one of " + ", ".join(DEFORM360_CALIBRATION_ROLES)
        )
    return role, path


def _logical_name(value: str) -> str:
    if "\\" in value:
        raise ValueError("source name must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("source name must be a confined relative POSIX path")
    return path.as_posix()


def _ordinary_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"cannot verify Git repository state at {repository}"
        ) from error


def _verify_repository(
    repository: Path,
    *,
    expected_revision: str,
) -> str:
    root = repository.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    expected = exact_revision(
        expected_revision,
        name="implementation_revision",
    )
    observed = _git_output(root, "rev-parse", "HEAD")
    if observed != expected:
        raise ValueError(
            "implementation revision differs from repository HEAD: "
            f"{expected} != {observed}"
        )
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError("repository checkout must be clean before sealing")
    return observed


def _verify_runtime_sources(repository: Path) -> None:
    """Require the imported sealer code to match the reviewed checkout."""

    for module_name, relative_path in _RUNTIME_MODULE_SOURCES.items():
        module = importlib.import_module(module_name)
        runtime_name = getattr(module, "__file__", None)
        if type(runtime_name) is not str or not runtime_name:
            raise ValueError(f"cannot identify runtime source for module {module_name}")
        runtime = _ordinary_file(
            Path(runtime_name),
            name=f"runtime source for {module_name}",
        )
        reviewed = _ordinary_file(
            repository / relative_path,
            name=f"reviewed source for {module_name}",
        )
        if file_sha256(runtime) != file_sha256(reviewed):
            raise ValueError(
                f"runtime source bytes differ from reviewed checkout: {module_name}"
            )


def _verify_committed_selection_lock(
    repository: Path,
    supplied_selection: Path,
) -> None:
    """Reject a structurally valid but unreviewed Stage-0 cohort."""

    committed = _ordinary_file(
        repository / _COMMITTED_SELECTION,
        name="committed Stage-0 selection",
    )
    supplied = _ordinary_file(
        supplied_selection,
        name="supplied Stage-0 selection",
    )
    if file_sha256(committed) != file_sha256(supplied):
        raise ValueError("supplied Stage-0 selection bytes differ from committed lock")


def _copy_source(source: Path, destination: Path) -> str:
    ordinary = _ordinary_file(source, name="source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ordinary.open("rb") as source_handle:
        data = source_handle.read()
    with destination.open("wb") as destination_handle:
        destination_handle.write(data)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    source_digest = file_sha256(ordinary)
    if file_sha256(destination) != source_digest:
        raise ValueError(f"copied source bytes changed: {source}")
    return source_digest


def _write_status(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(
        "\n".join(
            (
                "# Deform360 calibration execution seal",
                "",
                f"- Seal ID: `{summary['seal_id']}`",
                (
                    "- Confirmation opening token: "
                    f"`{summary['confirmation_opening_token']}`"
                ),
                (f"- Visual-provider lock: `{summary['visual_provider_lock_id']}`"),
                (
                    "- Stage-1 calibration lock: "
                    f"`{summary['visual_calibration_lock_id']}`"
                ),
                (f"- Calibration bundle: `{summary['calibration_bundle_id']}`"),
                "- Calibration payloads opened: `true`",
                "- Confirmation payloads opened: `false`",
                "- Target outcomes used: `false`",
                "",
                str(summary["claim_boundary"]),
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}" for path in paths
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_inputs(
    *,
    temporary: Path,
    repository_root: Path,
    selection_lock: Path,
    visual_provider_lock: Path,
    evidence_ledger: Path,
    artifacts: Mapping[str, Path],
    additional_sources: Sequence[tuple[str, Path]],
) -> dict[str, str]:
    sources: dict[str, str] = {}

    def copy(logical_path: str, source: Path) -> Path:
        logical = _logical_name(logical_path)
        if logical in sources:
            raise ValueError(f"duplicate source path: {logical}")
        destination = temporary / logical
        sources[logical] = _copy_source(source, destination)
        return destination

    copy("sources/stage0/selection.json", selection_lock)
    copy("sources/locks/visual-provider-lock.json", visual_provider_lock)
    copy(
        "sources/calibration/evidence-use-ledger.json",
        evidence_ledger,
    )
    for role in DEFORM360_CALIBRATION_ROLES:
        copy(
            f"sources/calibration/artifacts/{role}.json",
            artifacts[role],
        )
    for relative in _REPOSITORY_SOURCES:
        copy(
            f"sources/repository/{relative}",
            repository_root / relative,
        )
    for name, source in additional_sources:
        copy(f"sources/additional/{_logical_name(name)}", source)
    return sources


def _artifact_mapping(
    values: Sequence[tuple[str, Path]],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for role, path in values:
        if role in result:
            raise ValueError(f"duplicate calibration artifact role: {role}")
        result[role] = path
    missing = sorted(set(DEFORM360_CALIBRATION_ROLES) - set(result))
    extra = sorted(set(result) - set(DEFORM360_CALIBRATION_ROLES))
    if missing or extra:
        raise ValueError(
            f"calibration artifact roles changed: missing={missing}, extra={extra}"
        )
    return result


def _metadata(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    return load_strict_json_object(path, label="calibration execution metadata")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--evidence-ledger", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        type=_artifact_path,
        default=[],
        metavar="ROLE=PATH",
        help="Selected calibration artifact reference; repeat for all roles",
    )
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument(
        "--additional-source",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-payloads-opened",
        action="store_true",
        help=("Acknowledge that only the locked calibration payloads were opened"),
    )
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.calibration_payloads_opened:
        raise ValueError("--calibration-payloads-opened is required for a Stage-1 seal")
    repository_root = args.repository_root.resolve()
    revision = _verify_repository(
        repository_root,
        expected_revision=args.implementation_revision,
    )
    _verify_runtime_sources(repository_root)
    _verify_committed_selection_lock(repository_root, args.selection_lock)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(repository_root):
        raise ValueError("output directory must be outside the Git checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_paths = _artifact_mapping(args.artifact)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        source_artifacts = _copy_inputs(
            temporary=temporary,
            repository_root=repository_root,
            selection_lock=args.selection_lock,
            visual_provider_lock=args.visual_provider_lock,
            evidence_ledger=args.evidence_ledger,
            artifacts=artifact_paths,
            additional_sources=args.additional_source,
        )
        stage0 = load_deform360_stage0_selection(
            temporary / "sources/stage0/selection.json",
            protocol_path=(
                temporary / "sources/repository/protocols/"
                "deform360_official_hub_visuotactile_v1.json"
            ),
        )
        provider = load_deform360_visual_provider_lock(
            temporary / "sources/locks/visual-provider-lock.json"
        )
        ledger = load_evidence_use_ledger(
            temporary / "sources/calibration/evidence-use-ledger.json"
        )
        artifacts = tuple(
            load_deform360_calibration_artifact_ref(
                temporary / f"sources/calibration/artifacts/{role}.json"
            )
            for role in DEFORM360_CALIBRATION_ROLES
        )
        products = build_deform360_calibration_execution_seal(
            stage0_selection=stage0,
            visual_provider_lock=provider,
            evidence_use_ledger=ledger,
            calibration_artifacts=artifacts,
            implementation_revision=revision,
            source_artifacts=source_artifacts,
            metadata=_metadata(args.metadata_json),
        )
        verify_deform360_calibration_execution_artifacts(
            products,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            evidence_use_ledger=ledger,
        )
        save_deform360_visual_calibration_lock(
            temporary / "visual-calibration-lock.json",
            products.visual_calibration_lock,
        )
        save_deform360_calibration_bundle(
            products.calibration_bundle,
            temporary / "calibration-bundle.json",
        )
        save_deform360_calibration_execution_seal(
            products.execution_seal,
            temporary / "calibration-execution-seal.json",
        )
        summary = products.execution_seal.summary()
        write_atomic_json(
            summary,
            temporary / "summary.json",
            overwrite=False,
        )
        _write_status(temporary / "STATUS.md", summary)
        _write_checksums(temporary)
        os.replace(temporary, output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = _run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
