#!/usr/bin/env python3
"""Fit graph-local action support on source train frames only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from causal4d_public.deform360_action_support import (
    GraphActionSupportEpisode,
    fit_source_graph_action_support,
    graph_contact_distance_m,
)
from causal4d_public.deform360_phystwin_trust import (
    load_official_phystwin_readout_trust_episode,
)
from causal4d_public.deform360_reusable_graph import (
    load_canonical_deform360_graph,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode(value: str) -> tuple[str, tuple[Path, ...]]:
    fields = value.split("=", 7)
    if len(fields) != 8:
        raise argparse.ArgumentTypeError(
            "episode must be LABEL=TARGET=SIM=READOUT=GRAPH=DRIVEN=ZERO=SPLIT"
        )
    return fields[0], tuple(Path(field) for field in fields[1:])


def _float_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(field) for field in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid must contain floats") from error
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=_episode, required=True)
    parser.add_argument("--transfer-episode", action="append", type=_episode)
    parser.add_argument("--length-scales-m", type=_float_grid, required=True)
    parser.add_argument("--action-responses", type=_float_grid, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = []
    transfer = []
    input_sha256 = {}
    for role, specifications, destination in (
        ("selection", args.episode, source),
        ("held_out_source_transfer", args.transfer_episode or (), transfer),
    ):
        for label, paths in specifications:
            target, simulation, readout, graph_path, driven, zero, split = paths
            episode = load_official_phystwin_readout_trust_episode(
                label,
                target,
                simulation,
                readout,
                driven,
                zero,
                split,
            )
            graph = load_canonical_deform360_graph(graph_path)
            with np.load(readout, allow_pickle=False) as archive:
                readout_weights = np.asarray(
                    archive["readout_weights"], dtype=np.float64
                )
            destination.append(
                GraphActionSupportEpisode(
                    episode=episode,
                    readout_weights=readout_weights,
                    node_contact_distance_m=graph_contact_distance_m(graph),
                )
            )
            input_sha256[label] = {
                "role": role,
                **{
                    name: _sha256_file(path)
                    for name, path in zip(
                        (
                            "target",
                            "simulation",
                            "readout",
                            "graph",
                            "driven",
                            "zero",
                            "split",
                        ),
                        paths,
                    )
                },
            }

    result = fit_source_graph_action_support(
        source,
        length_scale_grid_m=args.length_scales_m,
        action_response_grid=args.action_responses,
        transfer_episodes=transfer,
    )
    result.pop("result_sha256")
    result["input_sha256"] = input_sha256
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["selected"], indent=2, sort_keys=True))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
