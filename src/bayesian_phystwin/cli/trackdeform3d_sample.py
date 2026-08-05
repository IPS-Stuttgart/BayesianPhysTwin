"""Inspect the public TrackDeform3D sample without decoding observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bayesian_phystwin.trackdeform3d_adapter import inspect_trackdeform3d_chunk

_SAMPLE_CHUNKS = (
    ("dlo", "chunk_1"),
    ("bdlo", "chunk_7"),
    ("fabric", "chunk_14"),
    ("cloth", "chunk_0"),
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_sample_manifest(
    dataset_root: str | Path,
    *,
    upstream_revision: str,
) -> dict[str, Any]:
    root = Path(dataset_root)
    admissions = []
    for object_kind, chunk_name in _SAMPLE_CHUNKS:
        admission = inspect_trackdeform3d_chunk(
            root / object_kind / chunk_name,
            root / object_kind / "calibration" / "transform_ee_cam_world.npz",
            object_kind=object_kind,  # type: ignore[arg-type]
        )
        admissions.append(admission.to_dict())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "trackdeform3d-public-smoke-v1",
        "upstream_revision": upstream_revision,
        "admissions": admissions,
        "admitted_count": len(admissions),
        "information_boundary": {
            "observation_values_decoded": False,
            "keypoint_trajectories_read": False,
            "future_outcomes_read": False,
            "claim_boundary": "interface_and_capacity_smoke_only",
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-revision", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_sample_manifest(
        args.dataset_root,
        upstream_revision=args.upstream_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
