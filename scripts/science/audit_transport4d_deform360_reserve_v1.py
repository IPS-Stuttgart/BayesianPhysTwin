#!/usr/bin/env python3
"""Reserve remaining public Deform360 object namespaces for Transport4D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin_experiments.transport4d_public_reserve_v1 import (
    audit_deform360_transport_reserve,
    render_reserve_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--action-kernel-protocol", type=Path, required=True)
    parser.add_argument("--untouched-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit_deform360_transport_reserve(
        data_root=args.data_root,
        reserve_protocol_path=args.protocol,
        action_kernel_protocol_path=args.action_kernel_protocol,
        untouched_protocol_path=args.untouched_protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_reserve_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
