#!/usr/bin/env python3
"""Export one decoded-uniform Prob4D bundle for Bayesian-PhysTwin."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prob4d.benchmark import fuse_prediction_bundle_methods
from prob4d.io import load_prediction_bundle, pack_symmetric_covariance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--prob4d-root", type=Path, required=True)
    parser.add_argument("--expected-prob4d-revision", required=True)
    args = parser.parse_args()

    prob4d_root = args.prob4d_root.resolve()
    revision = _git_revision(prob4d_root)
    if revision != args.expected_prob4d_revision:
        raise RuntimeError(
            f"Prob4D revision {revision} does not match {args.expected_prob4d_revision}"
        )
    bundle = load_prediction_bundle(args.manifest)
    sequence = fuse_prediction_bundle_methods(bundle, method_names={"prob4d_uniform"})[
        "prob4d_uniform"
    ]
    if (
        sequence.scene_flow is None
        or sequence.deform_mask is None
        or sequence.flow_covariance is None
    ):
        raise RuntimeError(
            "decoded-uniform fusion must provide scene flow, deform mask, and "
            "flow covariance"
        )
    payload: dict[str, np.ndarray] = {
        "point_map": sequence.point_map.astype(np.float16),
        "valid_mask": sequence.valid_mask,
        "frame_indices": sequence.frame_indices.astype(np.int64),
        "point_covariance_packed": pack_symmetric_covariance(
            sequence.point_covariance
        ).astype(np.float32),
        "contributors": sequence.contributors,
        "scene_flow": sequence.scene_flow.astype(np.float16),
        "deform_mask": sequence.deform_mask,
        "flow_covariance_packed": pack_symmetric_covariance(
            sequence.flow_covariance
        ).astype(np.float32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **payload)
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "decoded uniform overlap fusion",
        "fixed_prob4d_vggt_blend": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "prob4d_root": str(prob4d_root),
        "prob4d_revision": revision,
        "output_npz": str(args.output_npz.resolve()),
        "output_npz_sha256": _sha256(args.output_npz),
        "frame_count": int(len(sequence.frame_indices)),
        "overlap_pixel_fraction": float(np.mean(sequence.contributors > 1)),
        "maximum_contributors": int(np.max(sequence.contributors)),
        "covariance_units": "m^2 after Prob4D gauge alignment",
    }
    report_path = args.output_npz.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
