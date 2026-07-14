#!/usr/bin/env python3
"""Seal the pre-target Deform360 source-backend admission decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_replication_backend import (
    build_source_backend_decision_artifact,
    load_backend_policy,
    validate_source_backend_decision_artifact,
    write_source_backend_decision_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_deform360_replication_protocol(args.protocol)
    backend_policy = load_backend_policy(args.backend_policy)
    fits = [
        json.loads(
            (
                args.data_root
                / "fits"
                / record["object_id"]
                / "pooled_source_fit.json"
            ).read_text(encoding="utf-8")
        )
        for record in protocol["config"]["cohort"]
    ]
    payload = build_source_backend_decision_artifact(
        protocol, fits, backend_policy
    )
    write_source_backend_decision_artifact(args.output, payload)
    print(
        json.dumps(
            validate_source_backend_decision_artifact(payload), sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
