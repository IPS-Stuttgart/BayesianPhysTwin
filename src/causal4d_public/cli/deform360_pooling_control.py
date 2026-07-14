"""Seal pooled and single-source candidate identities from source scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_replication_controls import (
    build_pooling_control_selection_artifact,
    validate_pooling_control_selection_artifact,
    write_pooling_control_selection_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_fit_json")
    parser.add_argument("output_json")
    args = parser.parse_args()
    try:
        source_fit = json.loads(Path(args.source_fit_json).read_text(encoding="utf-8"))
        payload = build_pooling_control_selection_artifact(source_fit)
        output = write_pooling_control_selection_artifact(args.output_json, payload)
        result = validate_pooling_control_selection_artifact(payload)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                **result,
                "output": str(output.resolve()),
                "selection": payload["selection"],
                "sealed_candidate_indices": payload["sealed_candidate_indices"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
