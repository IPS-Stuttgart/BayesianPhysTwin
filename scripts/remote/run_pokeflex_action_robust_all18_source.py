#!/usr/bin/env python3
"""Run one frozen source case for the all-18 PokeFlex scale extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "remote"))

from bayesian_phystwin.pokeflex_action_robust_all18 import (  # noqa: E402
    SOURCE_FIELD,
    load_all18_source_protocol,
    validate_all18_source_protocol,
)
from bayesian_phystwin.pokeflex_action_robust_scale import (  # noqa: E402
    BASE_EFFECTIVE_SCALE,
    CANDIDATE_MULTIPLIERS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--source-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_all18_source_v4.json"
        ),
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    args = parser.parse_args()

    from run_pokeflex_checkpoint_registration_smoke import run_smoke

    source_protocol = load_all18_source_protocol(args.source_protocol)
    validation = validate_all18_source_protocol(source_protocol)
    take_root = args.take_root.resolve()
    if take_root.name not in validation["selected_take_ids"]:
        raise ValueError("take is outside the frozen all18 source selection")
    scales = (0.0,) + tuple(
        BASE_EFFECTIVE_SCALE * value for value in CANDIDATE_MULTIPLIERS
    )
    result = run_smoke(
        take_root,
        args.registration_protocol.resolve(),
        args.upstream_checkout.resolve(),
        args.checkpoint_root.resolve(),
        correction_scales=scales,
        correction_fields=(SOURCE_FIELD,),
        residual_geometry="point_to_point",
        maximum_frame=None,
        include_frozen_action_guard=False,
        record_online_observation_regret=False,
        additional_authorized_take_ids=tuple(validation["selected_take_ids"]),
    )
    result["all18_source_protocol_sha256"] = source_protocol["protocol_sha256"]
    result["source_prediction_role"] = (
        "opened non-target source interaction; never an official target result"
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing source artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "take_id": take_root.name,
                "all18_source_protocol_sha256": source_protocol["protocol_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
