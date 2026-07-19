#!/usr/bin/env python3
"""Prepare immutable compact-proxy inputs for object-disjoint MatPhys folds."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_loo_protocol import prepare_matphys_loo_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol")
    parser.add_argument("output_root")
    parser.add_argument("--proxy-summary", action="append", required=True)
    args = parser.parse_args()
    result = prepare_matphys_loo_workspace(
        args.protocol,
        args.proxy_summary,
        args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
