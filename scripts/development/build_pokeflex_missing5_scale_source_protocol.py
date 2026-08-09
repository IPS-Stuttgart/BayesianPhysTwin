#!/usr/bin/env python3
"""Freeze the 30-action missing-five PokeFlex source-scale protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_missing5_scale import (  # noqa: E402
    build_source_protocol,
    file_sha256,
    object_name,
    source_take_ids,
)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--locked-at-utc", required=True)
    parser.add_argument("--implementation-revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace source protocol: {args.output}")
    source_root = args.source_root.resolve()
    inventory = {}
    for take_id in source_take_ids():
        relative = Path(object_name(take_id)) / f"{take_id}.zip"
        archive = source_root / relative
        if not archive.is_file():
            raise FileNotFoundError(f"missing source archive: {archive}")
        inventory[take_id] = {
            "relative_path": relative.as_posix(),
            "sha256": file_sha256(archive),
            "bytes": archive.stat().st_size,
        }

    projection_runner = (
        ROOT / "scripts" / "remote" / "stage_pokeflex_missing5_scale_source_archive.py"
    )
    source_runner = (
        ROOT / "scripts" / "remote" / "run_pokeflex_missing5_scale_source_take.py"
    )
    legacy_runner = (
        ROOT / "scripts" / "remote" / "run_pokeflex_checkpoint_registration_smoke.py"
    )
    registration_protocol = (
        ROOT / "configs" / "sota" / "pokeflex_bayesian_registration_v1.json"
    )
    payload = build_source_protocol(
        inventory,
        locked_at_utc=args.locked_at_utc,
        implementation_revision=args.implementation_revision or _git_revision(),
        source_projection_runner_file_sha256=file_sha256(projection_runner),
        source_runner_file_sha256=file_sha256(source_runner),
        legacy_runner_file_sha256=file_sha256(legacy_runner),
        registration_protocol_file_sha256=file_sha256(registration_protocol),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "protocol_sha256": payload["protocol_sha256"],
                "source_take_count": len(inventory),
                "selected_total_bytes": payload["archive_inventory"][
                    "selected_total_bytes"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
