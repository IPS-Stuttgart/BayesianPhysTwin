"""Create the target-free author-source admission manifest for PokeFlex V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (
    build_author_source_manifest,
    load_official18_v4_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sota/pokeflex_action_robust_official18_v4.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {args.output}")
    protocol = load_official18_v4_protocol(args.protocol)
    manifest = build_author_source_manifest(args.source_root, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "take_count": len(manifest["takes"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
