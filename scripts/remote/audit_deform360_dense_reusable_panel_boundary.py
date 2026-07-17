#!/usr/bin/env python3
"""Verify that no sealed dense-panel target entered a derived-data tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causal4d_public.deform360_dense_reusable_panel import (
    audit_dense_panel_target_boundary,
    load_dense_reusable_panel_config,
)


def _result_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
    )
    config = load_dense_reusable_panel_config(config_path)
    result = audit_dense_panel_target_boundary(
        config,
        replication_root=args.replication_root,
    )
    result["result_sha256"] = _result_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"boundary audit already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "record_count": len(result["records"]),
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
