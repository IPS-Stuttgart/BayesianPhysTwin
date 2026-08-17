#!/usr/bin/env python3
"""Build the pre-target PokeFlex missing-five V6 execution lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (  # noqa: E402
    load_official18_v4_protocol,
)
from bayesian_phystwin.pokeflex_missing5_completion_v5 import (  # noqa: E402
    validate_completion_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    file_sha256,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    load_execution_protocol as load_v5_execution_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v6 import (  # noqa: E402
    IMPLEMENTATION_FILE_PATHS,
    build_execution_protocol,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v5-execution-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v5.json"
        ),
    )
    parser.add_argument(
        "--completion-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_missing5_scale_completion_v5.json"
        ),
    )
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_official18_v4.json"
        ),
    )
    parser.add_argument(
        "--causal-scale-model",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_missing5_causal_scale_v6.json"
        ),
    )
    parser.add_argument(
        "--source-result",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "sota"
            / "pokeflex_missing5_causal_scale_v6"
            / "source_result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v6.json"
        ),
    )
    parser.add_argument("--locked-at-utc", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace execution lock: {args.output}")
    parent = load_official18_v4_protocol(args.parent_protocol)
    completion = _load_json(args.completion_protocol)
    validate_completion_protocol(completion)
    v5_execution = load_v5_execution_protocol(
        args.v5_execution_protocol,
        completion,
        parent,
    )
    model = _load_json(args.causal_scale_model)
    source_result = _load_json(args.source_result)
    hashes = {
        relative: file_sha256(REPOSITORY_ROOT / relative)
        for relative in IMPLEMENTATION_FILE_PATHS
    }
    protocol = build_execution_protocol(
        v5_execution,
        completion,
        parent,
        model,
        source_result,
        locked_at_utc=args.locked_at_utc,
        model_file_sha256=file_sha256(args.causal_scale_model),
        source_result_file_sha256=file_sha256(args.source_result),
        implementation_file_sha256s=hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "execution_protocol_sha256": protocol["execution_protocol_sha256"],
                "implementation_file_count": len(hashes),
                "official_target_outcomes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
