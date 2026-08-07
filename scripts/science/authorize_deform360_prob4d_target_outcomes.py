#!/usr/bin/env python3
"""Authorize Deform360 target-outcome access after Prob4D target admission.

The command copies the exact BayesianPhysTwin Stage-0, visual-provider, and
confirmation-opening artifacts together with the Prob4D cohort binding,
promotion lock, and target-provider admission into one private bundle. It then
validates their complete cross-repository lineage while target outcomes remain
closed, emits a content-addressed authorization, and writes the exact metadata
required on the later BayesianPhysTwin query-result stream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_execution import (
    file_sha256,
    load_deform360_stage0_selection,
)
from bayesian_phystwin.deform360_calibration_observability_binding import (
    load_deform360_confirmation_opening_authorization,
)
from bayesian_phystwin.deform360_prob4d_target_outcome_authorization import (
    BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY,
    BPT_STAGE0_SOURCE_KEY,
    BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY,
    PROB4D_COHORT_BINDING_SOURCE_KEY,
    PROB4D_PROMOTION_LOCK_SOURCE_KEY,
    PROB4D_TARGET_ADMISSION_SOURCE_KEY,
    build_deform360_prob4d_target_outcome_authorization,
    save_deform360_prob4d_target_outcome_authorization,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    load_deform360_visual_provider_lock,
)


def _ordinary_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symbolic links: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _copy_source(source: Path, destination: Path, *, name: str) -> str:
    ordinary = _ordinary_file(source, name=name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ordinary.open("rb") as source_handle:
        payload = source_handle.read()
    with destination.open("xb") as destination_handle:
        destination_handle.write(payload)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    source_sha256 = file_sha256(ordinary)
    if file_sha256(destination) != source_sha256:
        raise ValueError(f"copied {name} bytes changed")
    return source_sha256


def _metadata(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {}
    return load_strict_json_object(path, label="target-outcome authorization metadata")


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument(
        "--confirmation-opening-authorization",
        type=Path,
        required=True,
    )
    parser.add_argument("--prob4d-cohort-binding", type=Path, required=True)
    parser.add_argument("--prob4d-promotion-lock", type=Path, required=True)
    parser.add_argument("--target-provider-admission", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmation-provider-inputs-opened",
        action="store_true",
        help=(
            "acknowledge that only predictor-side confirmation inputs were opened "
            "to produce the admitted provider manifests"
        ),
    )
    return parser


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not args.confirmation_provider_inputs_opened:
        raise ValueError(
            "--confirmation-provider-inputs-opened is required after target manifests exist"
        )
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.prob4d-outcome-authorization.",
            dir=output.parent,
        )
    )
    staged = workspace / "authorized"
    try:
        staged.mkdir()
        input_paths = {
            BPT_STAGE0_SOURCE_KEY: args.selection_lock,
            BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY: args.visual_provider_lock,
            BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY: (
                args.confirmation_opening_authorization
            ),
            PROB4D_COHORT_BINDING_SOURCE_KEY: args.prob4d_cohort_binding,
            PROB4D_PROMOTION_LOCK_SOURCE_KEY: args.prob4d_promotion_lock,
            PROB4D_TARGET_ADMISSION_SOURCE_KEY: args.target_provider_admission,
        }
        source_artifacts = {
            logical_path: _copy_source(
                source,
                staged / logical_path,
                name=logical_path,
            )
            for logical_path, source in input_paths.items()
        }
        stage0 = load_deform360_stage0_selection(
            staged / BPT_STAGE0_SOURCE_KEY,
        )
        visual_provider = load_deform360_visual_provider_lock(
            staged / BPT_VISUAL_PROVIDER_LOCK_SOURCE_KEY,
        )
        confirmation_authorization = (
            load_deform360_confirmation_opening_authorization(
                staged / BPT_CONFIRMATION_AUTHORIZATION_SOURCE_KEY,
            )
        )
        cohort_binding = load_strict_json_object(
            staged / PROB4D_COHORT_BINDING_SOURCE_KEY,
            label="Prob4D cohort binding",
        )
        promotion_lock = load_strict_json_object(
            staged / PROB4D_PROMOTION_LOCK_SOURCE_KEY,
            label="Prob4D promotion lock",
        )
        target_admission = load_strict_json_object(
            staged / PROB4D_TARGET_ADMISSION_SOURCE_KEY,
            label="Prob4D target-provider admission",
        )
        authorization = build_deform360_prob4d_target_outcome_authorization(
            stage0_selection=stage0,
            visual_provider_lock=visual_provider,
            confirmation_opening_authorization=confirmation_authorization,
            prob4d_cohort_binding=cohort_binding,
            prob4d_promotion_lock=promotion_lock,
            target_provider_admission=target_admission,
            source_artifacts=source_artifacts,
            confirmation_provider_inputs_opened=True,
            metadata=_metadata(args.metadata_json),
        )
        save_deform360_prob4d_target_outcome_authorization(
            authorization,
            staged / "target-outcome-authorization.json",
        )
        write_atomic_json(
            authorization.query_result_metadata(),
            staged / "query-result-metadata.json",
            overwrite=False,
        )
        summary = authorization.summary()
        write_atomic_json(
            summary,
            staged / "target-outcome-authorization-summary.json",
            overwrite=False,
        )
        _write_checksums(staged)
        if output.exists():
            raise FileExistsError(output)
        os.replace(staged, output)
        return summary
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = _run(arguments)
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"authorized": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
