"""Build one source-only Deform360 rope dynamics observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_contact import load_contact_artifact
from causal4d_public.deform360_rope_observations import (
    build_source_rope_observation,
    validate_source_rope_observation_artifact,
    write_source_rope_observation_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("contact_model_json")
    parser.add_argument("rope_sequence_json")
    parser.add_argument("output_archive_npz")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        contact_model = load_contact_artifact(
            args.contact_model_json, expected_kind="Deform360ContactModel"
        )
        rope_sequence = json.loads(
            Path(args.rope_sequence_json).read_text(encoding="utf-8")
        )
        result = build_source_rope_observation(
            args.processed_root,
            protocol,
            contact_model,
            rope_sequence,
            args.output_archive_npz,
        )
        write_source_rope_observation_artifact(args.output_audit_json, result)
        validation = validate_source_rope_observation_artifact(result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {**validation, "output": args.output_audit_json},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
