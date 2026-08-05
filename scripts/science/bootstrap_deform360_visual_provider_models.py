#!/usr/bin/env python3
"""Populate the exact model snapshots needed by the Deform360 producer freeze.

The bootstrap reads only a target-blind producer specification and public model
repositories. It does not accept or open a Deform360 dataset path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_THIS_DIRECTORY = Path(__file__).resolve().parent
_PREFLIGHT_SOURCE = _THIS_DIRECTORY / "build_deform360_visual_provider_freeze.py"


def _load_preflight_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "deform360_visual_provider_freeze_for_bootstrap",
        _PREFLIGHT_SOURCE,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the Deform360 visual-provider preflight")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _exact_source_groups(spec: Mapping[str, Any]) -> dict[tuple[str, str], tuple[str, ...]]:
    motion = spec.get("motioncrafter")
    if not isinstance(motion, Mapping):
        raise ValueError("preflight motioncrafter specification must be an object")
    sources = motion.get("model_sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "unet",
        "vae",
        "image_vae",
        "base_pipeline",
    }:
        raise ValueError("preflight model source roles changed")

    helper = _load_preflight_module()
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for role in ("unet", "vae", "image_vae", "base_pipeline"):
        source = sources[role]
        if not isinstance(source, Mapping):
            raise ValueError(f"model source {role} must be an object")
        repository = helper._require_literal_string(
            source.get("repository"),
            name=f"{role} repository",
        )
        revision = helper._require_revision(
            source.get("expected_revision"),
            name=f"{role} expected_revision",
        )
        members = source.get("required_members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"model source {role} required_members must be nonempty")
        for index, member in enumerate(members):
            relative = helper._require_literal_string(
                member,
                name=f"{role} required member {index}",
            )
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ValueError(f"{role} required member is not a canonical relative path")
            grouped[(repository, revision)].add(relative)
    return {
        key: tuple(sorted(members))
        for key, members in sorted(grouped.items())
    }


def bootstrap_exact_model_snapshots(
    *,
    spec_path: str | Path,
    cache_directory: str | Path,
    token: str | None = None,
) -> dict[str, Any]:
    """Download missing members from exact immutable model revisions."""

    helper = _load_preflight_module()
    spec = helper.load_preflight_spec(spec_path)
    groups = _exact_source_groups(spec)
    cache = Path(cache_directory).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("model bootstrap requires huggingface_hub") from error

    records: list[dict[str, Any]] = []
    for (repository, revision), members in groups.items():
        repository_directory = "models--" + repository.replace("/", "--")
        expected_snapshot = cache / repository_directory / "snapshots" / revision
        complete_before = expected_snapshot.is_dir() and all(
            (expected_snapshot / relative).exists() for relative in members
        )
        if not complete_before:
            observed = Path(
                snapshot_download(
                    repo_id=repository,
                    revision=revision,
                    cache_dir=cache,
                    allow_patterns=list(members),
                    token=token or None,
                )
            ).resolve()
            if observed.name != revision:
                raise ValueError(
                    f"{repository} resolved to {observed.name}, expected {revision}"
                )
            expected_snapshot = observed
        missing = [
            relative
            for relative in members
            if not (expected_snapshot / relative).exists()
        ]
        if missing:
            raise ValueError(
                f"{repository}@{revision} remains incomplete after bootstrap: {missing}"
            )
        records.append(
            {
                "repository": repository,
                "revision": revision,
                "required_members": list(members),
                "download_performed": not complete_before,
                "snapshot_path_recorded": False,
            }
        )

    return {
        "schema": "bayesian-phystwin/deform360-model-bootstrap-result-v1",
        "schema_version": 1,
        "preflight_spec_id": spec["artifact_id"],
        "source_count": len(records),
        "sources": records,
        "selected_raw_payloads_opened": False,
        "calibration_payloads_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "claim_boundary": spec["claim_boundary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--token-environment-variable",
        default="HF_TOKEN",
        help="environment variable holding an optional Hugging Face token",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    token_name = str(arguments.token_environment_variable)
    result = bootstrap_exact_model_snapshots(
        spec_path=arguments.spec,
        cache_directory=arguments.cache_dir,
        token=os.environ.get(token_name),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
