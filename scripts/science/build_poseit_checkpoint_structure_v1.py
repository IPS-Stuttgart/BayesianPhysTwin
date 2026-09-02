#!/usr/bin/env python3
"""Use a separately frozen structure authorization after checkpoint acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import acquire_poseit_checkpointed_range_hash_v1 as transport

from bayesian_phystwin_experiments import poseit_checkpoint_acquisition as acquisition
from bayesian_phystwin_experiments import poseit_checkpoint_structure as structure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-result-sha256")
    args = parser.parse_args(argv)
    if args.mode == "verify" and args.expected_result_sha256 is None:
        parser.error("verify requires the exact published result SHA-256")
    if args.mode == "run" and args.expected_result_sha256 is not None:
        parser.error("a result digest is accepted only for offline verification")
    spec, engine = transport.load_context(
        args.amendment.absolute(),
        expected_amendment_sha256=args.expected_amendment_sha256,
    )
    if args.mode == "run":
        result = structure.run_checkpointed_structure(
            spec,
            engine,
            args.authorization.absolute(),
            expected_authorization_sha256=args.expected_authorization_sha256,
            opener=acquisition._default_open,
        )
    else:
        assert args.expected_result_sha256 is not None
        result = structure.verify_checkpointed_structure(
            spec,
            engine,
            args.authorization.absolute(),
            expected_authorization_sha256=args.expected_authorization_sha256,
            expected_result_sha256=args.expected_result_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
