"""Validate the locked public Deform360 replication protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_replication import (
    validate_deform360_replication_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol_json")
    args = parser.parse_args()
    try:
        payload = json.loads(Path(args.protocol_json).read_text(encoding="utf-8"))
        result = validate_deform360_replication_protocol(payload)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
