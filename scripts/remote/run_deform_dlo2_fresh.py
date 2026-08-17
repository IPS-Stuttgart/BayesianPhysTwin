#!/usr/bin/env python3
"""Authorize and run the fresh DLO2 source reproduction after the DLO1 gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    validate_deform_dlo2_fresh_posterior_parent,
)
from bayesian_phystwin.deform_dlo_source import (
    load_deform_dlo_source_protocol,
    sha256_file,
    validate_deform_dlo2_fresh_parent,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-longrun-result", type=Path, required=True)
    parser.add_argument("--parent-posterior-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "run"),
        default="run",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    protocol = load_deform_dlo_source_protocol(args.protocol)
    if protocol["dlo_types"] != ("DLO2",):
        raise ValueError("fresh confirmation wrapper requires DLO2 only")
    protocol_payload = _read_json(args.protocol)
    parent_path = args.parent_longrun_result.resolve()
    parent = _read_json(parent_path)
    parent_authorization = validate_deform_dlo2_fresh_parent(
        protocol_payload,
        parent,
    )
    posterior_path = args.parent_posterior_result.resolve()
    posterior_result = _read_json(posterior_path)
    selection_identity = posterior_result.get("selection_seal")
    if not isinstance(selection_identity, dict):
        raise ValueError("DLO1 posterior result omits its selection seal")
    selection_path = Path(str(selection_identity.get("path", ""))).resolve()
    if not selection_path.is_file() or sha256_file(
        selection_path
    ) != selection_identity.get("sha256"):
        raise ValueError("DLO1 posterior selection seal does not verify")
    selection_seal = _read_json(selection_path)
    posterior_authorization = validate_deform_dlo2_fresh_posterior_parent(
        protocol_payload,
        posterior_result,
        selection_seal,
        selection_seal_sha256=sha256_file(selection_path),
    )

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_output = output_root / "source_run"
    authorization = {
        "schema_version": 2,
        "contract": "deform-dlo2-fresh-authorization-v2",
        "mode": args.mode,
        "official_eval_read": False,
        "source_test_opened": False,
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "parent_longrun_result": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            **parent_authorization,
        },
        "parent_posterior_result": {
            "path": str(posterior_path),
            "sha256": sha256_file(posterior_path),
            "selection_seal": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            },
            **posterior_authorization,
        },
        "source_output": str(source_output),
    }
    authorization_path = output_root / "authorization.json"
    _write_json(authorization_path, authorization)

    runner = Path(__file__).resolve().with_name("run_deform_dlo_source.py")
    command = [
        sys.executable,
        str(runner),
        "--protocol",
        str(args.protocol.resolve()),
        "--upstream-root",
        str(args.upstream_root.resolve()),
        "--output-root",
        str(source_output),
        "--stage-authorization",
        str(authorization_path),
        "--dlo-type",
        "DLO2",
        "--device",
        args.device,
        "--mode",
        args.mode,
    ]
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = str(
        protocol["training"]["cublas_workspace_config"]
    )
    completed = subprocess.run(
        command,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
