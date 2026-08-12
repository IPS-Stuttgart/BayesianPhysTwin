#!/usr/bin/env python3
"""Generate or verify the historical-root import migration map."""

from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "api/root-public-api-migration-v0.5.json.gz"
ROOT_MANIFEST = ROOT / "api/root-public-api-v0.4.json"
V1_MANIFEST = ROOT / "api/versioned-public-api-v1.json"
LEGACY_SOURCE = ROOT / "src/bayesian_phystwin/_legacy_root_eager.py"


def _export_modules() -> dict[str, str]:
    tree = ast.parse(
        LEGACY_SOURCE.read_text(encoding="utf-8"),
        filename=str(LEGACY_SOURCE),
    )
    exports: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        for alias in node.names:
            public_name = alias.asname or alias.name
            if public_name in exports:
                raise ValueError(f"duplicate historical export: {public_name}")
            exports[public_name] = f"bayesian_phystwin.{node.module}"
    return exports


def build_manifest() -> dict[str, Any]:
    """Build the deterministic root-to-module migration manifest."""

    root_payload = json.loads(ROOT_MANIFEST.read_text(encoding="utf-8"))
    v1_payload = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    root_symbols = root_payload["symbols"]
    v1_symbols = frozenset(v1_payload["symbols"])
    export_modules = _export_modules()
    if set(export_modules) != set(root_symbols):
        missing = sorted(set(root_symbols) - set(export_modules))
        unexpected = sorted(set(export_modules) - set(root_symbols))
        raise ValueError(
            "historical export source differs from the root manifest; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    symbols: list[dict[str, str]] = []
    for name in root_symbols:
        defining_module = export_modules[name]
        stable = name in v1_symbols
        symbols.append(
            {
                "name": name,
                "defining_module": defining_module,
                "preferred_import": (
                    "bayesian_phystwin.v1" if stable else defining_module
                ),
                "support": "stable-v1" if stable else "legacy-root-compatibility",
            }
        )
    return {
        "schema": "bayesian-phystwin.root-api-migration",
        "schema_version": 1,
        "source_package": "bayesian_phystwin",
        "source_compatibility_line": "0.4",
        "target_compatibility_line": "0.5",
        "stable_namespace": "bayesian_phystwin.v1",
        "symbols": symbols,
    }


def render_json() -> bytes:
    """Return canonical, review-friendly UTF-8 JSON."""

    return (json.dumps(build_manifest(), indent=2, sort_keys=False) + "\n").encode()


def render_gzip() -> bytes:
    """Return deterministic gzip-compressed JSON."""

    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        compresslevel=9,
        mtime=0,
    ) as archive:
        archive.write(render_json())
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed migration map is not current",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the uncompressed JSON instead of writing a file",
    )
    args = parser.parse_args(argv)

    if args.stdout:
        print(render_json().decode(), end="")
        return 0

    rendered = render_gzip()
    if args.check:
        try:
            current = args.output.read_bytes()
        except FileNotFoundError as error:
            raise SystemExit(f"missing migration map: {args.output}") from error
        if current != rendered:
            raise SystemExit(
                "root API migration map is stale; run "
                "python tools/quality/generate_root_api_migration.py"
            )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
