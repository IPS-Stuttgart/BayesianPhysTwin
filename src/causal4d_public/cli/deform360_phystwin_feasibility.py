"""Run the locked source-only official-PhysTwin/Warp feasibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_phystwin_feasibility import (
    run_official_warp_feasibility_gate,
    validate_official_warp_feasibility_artifact,
    write_official_warp_feasibility_artifact,
)


def _source_paths(directory: Path) -> list[Path]:
    return [
        directory / f"deform360_001_rope_source{episode}_observation_v5.json"
        for episode in (0, 3, 4, 5, 8)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--source-observation-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-archive", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        payload = run_official_warp_feasibility_gate(
            args.protocol,
            args.official_repo,
            _source_paths(Path(args.source_observation_dir)),
            args.output_archive,
            device=args.device,
        )
        output = write_official_warp_feasibility_artifact(args.output_json, payload)
        stored = json.loads(output.read_text(encoding="utf-8"))
        result = validate_official_warp_feasibility_artifact(stored)
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                **result,
                "output": str(output.resolve()),
                "source_competence": stored["source_competence"],
                "numerical_audit": stored["numerical_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if stored["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
