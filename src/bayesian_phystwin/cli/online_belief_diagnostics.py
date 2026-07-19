"""Reproduce the four retrospective online-belief diagnostics.

The subcommands expose repository-owned versions of the original read-only
development runners.  They intentionally retain the legacy protocols and
output schemas so that the published JSON checksums can be verified against
the same sealed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from bayesian_phystwin import causal_continuation_diagnostic
from bayesian_phystwin import deform360_corruption_diagnostic
from bayesian_phystwin import deform360_tail_gate_diagnostic
from bayesian_phystwin import residual_velocity_diagnostic


DEFAULT_BPT_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
DEFAULT_DEFORM_ROOT = Path(
    "/mnt/corsair/florianpfaff/deform360-dense-reusable-panel-v1/independent-source-v1"
)
DEFAULT_DEFORM_V1_RUN = DEFAULT_BPT_ROOT / "runs/deform360-online-belief-open27-v1"
DEFAULT_PHYSTWIN_V3_RUN = (
    DEFAULT_BPT_ROOT / "runs/online-belief-original22-observation-gated-v3"
)
DEFAULT_PHYSTWIN_V3_CONFIG = (
    DEFAULT_BPT_ROOT
    / "configs/sota/phystwin_online_belief_v3_original22_development.json"
)


def _write_json(result: dict, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce read-only, post-hoc online-belief diagnostics; outputs "
            "are development evidence and not SOTA confirmation."
        )
    )
    subparsers = parser.add_subparsers(dest="diagnostic", required=True)

    corruption = subparsers.add_parser(
        "corruption-stress",
        help="Run the frozen Deform360 Gaussian/mismatch corruption stress.",
    )
    corruption.add_argument("--root", type=Path, default=DEFAULT_DEFORM_ROOT)
    corruption.add_argument("--output", type=Path, required=True)

    tail = subparsers.add_parser(
        "tail-gates",
        help="Run the post-hoc p90 and 13-of-16 correspondence gate stress.",
    )
    tail.add_argument("--root", type=Path, default=DEFAULT_DEFORM_ROOT)
    tail.add_argument("--output", type=Path, required=True)
    tail.add_argument(
        "--include-full-stream",
        action="store_true",
        help="Also corrupt the causal threshold-calibration history.",
    )

    causal = subparsers.add_parser(
        "causal-continuation",
        help="Run the retrospective continuation-vs-freeze diagnostic.",
    )
    causal.add_argument("--phys-run", type=Path, default=DEFAULT_PHYSTWIN_V3_RUN)
    causal.add_argument("--phys-config", type=Path, default=DEFAULT_PHYSTWIN_V3_CONFIG)
    causal.add_argument("--deform-run", type=Path, default=DEFAULT_DEFORM_V1_RUN)
    causal.add_argument("--output", type=Path, required=True)

    velocity = subparsers.add_parser(
        "residual-velocity",
        help="Run the retrospective residual-velocity field diagnostic.",
    )
    velocity.add_argument("--bpt-root", type=Path, default=DEFAULT_BPT_ROOT)
    velocity.add_argument("--deform-root", type=Path, default=DEFAULT_DEFORM_ROOT)
    velocity.add_argument("--deform-run", type=Path, default=DEFAULT_DEFORM_V1_RUN)
    velocity.add_argument("--phystwin-run", type=Path, default=DEFAULT_PHYSTWIN_V3_RUN)
    velocity.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.diagnostic == "corruption-stress":
        result = deform360_corruption_diagnostic.run(args.root.resolve())
    elif args.diagnostic == "tail-gates":
        result = deform360_tail_gate_diagnostic.run(
            args.root.resolve(), include_full_stream=args.include_full_stream
        )
    elif args.diagnostic == "causal-continuation":
        result = causal_continuation_diagnostic.run_diagnostic(
            phys_run=args.phys_run.resolve(),
            phys_config=args.phys_config.resolve(),
            deform_run=args.deform_run.resolve(),
        )
    elif args.diagnostic == "residual-velocity":
        result = residual_velocity_diagnostic.run_diagnostic(
            bpt_root=args.bpt_root.resolve(),
            deform_root=args.deform_root.resolve(),
            deform_run=args.deform_run.resolve(),
            phystwin_run=args.phystwin_run.resolve(),
        )
    else:  # pragma: no cover - argparse enforces the subcommand choices.
        raise AssertionError(args.diagnostic)

    digest = _write_json(result, args.output)
    print(json.dumps({"output": str(args.output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
