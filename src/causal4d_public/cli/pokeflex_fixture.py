"""Generate a tiny PokeFlex-shaped fixture containing no restricted data."""

from __future__ import annotations

import argparse
import json

from causal4d_public.pokeflex import write_synthetic_pokeflex_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--objects", type=int, default=1)
    parser.add_argument("--takes-per-object", type=int, default=5)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--mutate-last-mesh-topology", action="store_true")
    parser.add_argument("--include-material-identity", action="store_true")
    args = parser.parse_args()
    result = write_synthetic_pokeflex_fixture(
        args.output_root,
        object_count=args.objects,
        takes_per_object=args.takes_per_object,
        frame_count=args.frames,
        mutate_last_mesh_topology=args.mutate_last_mesh_topology,
        include_material_identity=args.include_material_identity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
