#!/usr/bin/env python3
"""Run the frozen PokeFlex independent-depth source-validation panel."""

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
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    run_smoke,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_source_protocol() -> Path:
    return (
        _repository_root()
        / "configs"
        / "sota"
        / "pokeflex_independent_depth_source_validation_v2.json"
    )


def _default_parent_protocol() -> Path:
    return (
        _repository_root()
        / "configs"
        / "sota"
        / "pokeflex_bayesian_registration_v1.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, default=_default_source_protocol())
    parser.add_argument("--parent-protocol", type=Path, default=_default_parent_protocol())
    parser.add_argument(
        "--take-id",
        action="append",
        help="Run only selected source-validation take ids; repeat as needed",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    protocol = load_pokeflex_independent_depth_protocol(args.source_protocol)
    payload = protocol["payload"]
    boundary = payload["evidence_boundary"]
    method = payload["method_lock"]
    expected = [
        f"{object_name}_{take}"
        for object_name in boundary["development_objects"]
        for take in boundary["source_validation_takes"]
    ]
    selected = args.take_id or expected
    unknown = sorted(set(selected) - set(expected))
    if unknown:
        raise ValueError(f"takes are outside the frozen source panel: {unknown}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for take_id in selected:
        output = args.output_root / f"{take_id}_full_template15mm_v2.json"
        if output.exists() and args.skip_existing:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("independent_depth_anchor", {}).get("protocol_sha256")
                != protocol["protocol_sha256"]
            ):
                raise ValueError(f"existing output uses another protocol: {output}")
            records.append(
                {
                    "take_id": take_id,
                    "status": "existing",
                    "output": str(output.resolve()),
                    "sha256": _sha256(output),
                }
            )
            continue
        try:
            result = run_smoke(
                (args.dataset_root / take_id).resolve(),
                args.parent_protocol.resolve(),
                args.upstream_checkout.resolve(),
                args.checkpoint_root.resolve(),
                correction_scales=tuple(map(float, method["correction_scales"])),
                correction_fields=tuple(map(str, method["correction_fields"])),
                residual_geometry="point_to_point",
                maximum_frame=None,
                include_frozen_action_guard=False,
                record_online_observation_regret=False,
                record_independent_anchor_regret=True,
                independent_depth_protocol_path=args.source_protocol.resolve(),
                independent_anchor_maximum_template_distance_m=(
                    float(method["static_template_support_radius_mm"]) / 1000.0
                ),
            )
            rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if output.exists() and output.read_text(encoding="utf-8") != rendered:
                raise ValueError(f"existing source output differs: {output}")
            output.write_text(rendered, encoding="utf-8")
            records.append(
                {
                    "take_id": take_id,
                    "status": "completed",
                    "output": str(output.resolve()),
                    "sha256": _sha256(output),
                }
            )
        except Exception as error:
            records.append(
                {
                    "take_id": take_id,
                    "status": "failed-no-replacement",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        progress = {
            "schema_version": 1,
            "artifact_kind": "PokeFlexIndependentDepthSourceValidationProgress",
            "protocol_sha256": protocol["protocol_sha256"],
            "replacement_allowed": False,
            "records": records,
        }
        progress_path = args.output_root / "source_validation_progress_v2.json"
        progress_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
