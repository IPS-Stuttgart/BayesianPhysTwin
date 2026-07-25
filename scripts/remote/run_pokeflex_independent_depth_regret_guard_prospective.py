#!/usr/bin/env python3
"""Generate the frozen candidate bank for one prospective PokeFlex take."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_independent_depth_protocol import (  # noqa: E402
    load_pokeflex_independent_depth_protocol,
)
from bayesian_phystwin.pokeflex_independent_depth_regret_guard_protocol import (  # noqa: E402
    load_pokeflex_regret_guard_prospective_protocol,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    run_smoke,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--prospective-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_regret_guard_prospective_v1.json"
        ),
    )
    parser.add_argument(
        "--independent-depth-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_source_validation_v2.json"
        ),
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    args = parser.parse_args()
    take_root = args.take_root.resolve()
    prospective = load_pokeflex_regret_guard_prospective_protocol(
        args.prospective_protocol.resolve()
    )
    if take_root.name not in prospective["take_ids"]:
        raise ValueError("take is outside the prospective cohort")
    candidate_runner = (
        _repository_root()
        / "scripts"
        / "remote"
        / "run_pokeflex_checkpoint_registration_independent_depth.py"
    )
    expected_runner_hash = prospective["payload"]["parent_method"][
        "candidate_runner_sha256"
    ]
    if _sha256(candidate_runner) != expected_runner_hash:
        raise ValueError("frozen candidate runner checksum changed")
    independent = load_pokeflex_independent_depth_protocol(
        args.independent_depth_protocol.resolve()
    )
    method = independent["payload"]["method_lock"]
    result = run_smoke(
        take_root,
        args.registration_protocol.resolve(),
        args.upstream_checkout.resolve(),
        args.checkpoint_root.resolve(),
        correction_scales=tuple(map(float, method["correction_scales"])),
        correction_fields=tuple(map(str, method["correction_fields"])),
        residual_geometry="point_to_point",
        maximum_frame=None,
        include_frozen_action_guard=False,
        record_online_observation_regret=False,
        record_independent_anchor_regret=True,
        independent_depth_protocol_path=args.independent_depth_protocol.resolve(),
        independent_anchor_maximum_template_distance_m=(
            float(method["static_template_support_radius_mm"]) / 1000.0
        ),
    )
    result["prospective_regret_guard"] = {
        "protocol_id": prospective["payload"]["protocol_id"],
        "protocol_sha256": prospective["protocol_sha256"],
        "candidate_runner_sha256": expected_runner_hash,
        "outcome_opened_after_protocol_lock": True,
        "candidate_bank_refit": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing prospective artifact differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "take_id": take_root.name,
                "target_count": len(result["targets"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
