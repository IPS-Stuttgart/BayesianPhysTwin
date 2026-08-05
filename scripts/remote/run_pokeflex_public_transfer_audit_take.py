#!/usr/bin/env python3
"""Run one frozen take in the retrospective 78-action PokeFlex audit."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "remote"))

from bayesian_phystwin.pokeflex_action_robust_all18 import (  # noqa: E402
    SOURCE_FIELD,
)
from bayesian_phystwin.pokeflex_public_transfer_audit import (  # noqa: E402
    AUDIT_RUNNER_FILE_SHA256,
    BASE_EFFECTIVE_SCALE,
    LEGACY_RUNNER_FILE_SHA256,
    file_sha256,
    validate_public_transfer_protocol,
)


def _run_smoke(
    runner_module,
    *,
    take_root: Path,
    registration_protocol: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
    effective_scale: float,
):
    """Authorize the fixed retrospective cohort without editing the legacy runner."""

    original_loader = runner_module.load_pokeflex_registration_protocol

    def load_with_public_authorization(path: Path):
        protocol = deepcopy(original_loader(path))
        protocol["payload"]["cohort"]["development_objects"] = sorted(
            set(protocol["payload"]["cohort"].get("development_objects", ()))
            | {take_root.name.rpartition("_T")[0]}
        )
        return protocol

    runner_module.load_pokeflex_registration_protocol = load_with_public_authorization
    try:
        return runner_module.run_smoke(
            take_root,
            registration_protocol,
            upstream_checkout,
            checkpoint_root,
            correction_scales=tuple(sorted({0.0, BASE_EFFECTIVE_SCALE, effective_scale})),
            correction_fields=(SOURCE_FIELD,),
            residual_geometry="point_to_point",
            maximum_frame=None,
            include_frozen_action_guard=False,
            record_online_observation_regret=False,
        )
    finally:
        runner_module.load_pokeflex_registration_protocol = original_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_public78_retrospective_v6.json"
        ),
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(ROOT / "configs" / "sota" / "pokeflex_bayesian_registration_v1.json"),
    )
    args = parser.parse_args()

    import run_pokeflex_checkpoint_registration_smoke as runner_module

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validation = validate_public_transfer_protocol(protocol)
    if file_sha256(Path(__file__)) != AUDIT_RUNNER_FILE_SHA256:
        raise ValueError("audit runner bytes changed")
    take_root = args.take_root.resolve()
    take_id = take_root.name
    if take_id not in validation["retrospective_take_ids"]:
        raise ValueError("take is outside the frozen retrospective cohort")
    expected_archive = protocol["archive_inventory"]["takes"][take_id]
    source_archive = take_root.parent / f"{take_id}.zip"
    if source_archive.exists() and file_sha256(source_archive) != expected_archive["sha256"]:
        raise ValueError("source archive bytes changed")
    object_name = take_id.rpartition("_T")[0]
    effective_scale = validation["effective_scales"][object_name]
    legacy_path = ROOT / "scripts" / "remote" / "run_pokeflex_checkpoint_registration_smoke.py"
    if file_sha256(legacy_path) != LEGACY_RUNNER_FILE_SHA256:
        raise ValueError("legacy runner bytes changed")

    result = _run_smoke(
        runner_module,
        take_root=take_root,
        registration_protocol=args.registration_protocol.resolve(),
        upstream_checkout=args.upstream_checkout.resolve(),
        checkpoint_root=args.checkpoint_root.resolve(),
        effective_scale=effective_scale,
    )
    result["public_transfer_protocol_sha256"] = protocol["protocol_sha256"]
    result["legacy_runner_file_sha256"] = LEGACY_RUNNER_FILE_SHA256
    result["retrospective_prediction_role"] = (
        "previously exposed public action; fixed all18 scale; never prospective evidence"
    )
    result["candidate_effective_scale"] = effective_scale
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing audit artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "take_id": take_id,
                "protocol_sha256": protocol["protocol_sha256"],
                "effective_scale": effective_scale,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
