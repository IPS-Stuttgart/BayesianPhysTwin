#!/usr/bin/env python3
"""Extract provenance-bound causal tactile features from a locked case manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_tactile_features import (  # noqa: E402
    build_tactile_feature_artifact,
    canonical_artifact_sha256,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    _require(
        manifest.get("artifact_kind") == "Deform360TactileFeatureCaseManifestV1",
        "unexpected tactile case manifest",
    )
    _require(
        manifest.get("artifact_sha256") == canonical_artifact_sha256(manifest),
        "tactile case manifest checksum changed",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("outcomes_read_for_selection") is False
        and boundary.get("held_v8_read") is False,
        "tactile case manifest crossed its information boundary",
    )
    payload = build_tactile_feature_artifact(
        manifest.get("cases", []),
        window_root=args.window_root,
        raw_root=args.raw_root,
    )
    payload["inputs"] = {
        "case_manifest_path": args.case_manifest.name,
        "case_manifest_artifact_sha256": manifest["artifact_sha256"],
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "case_count": len(payload["cases"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
