#!/usr/bin/env python3
"""Audit finite movement of the zero-initialized MatPhys part adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bayesian_phystwin.matphys_reconstruction_control import (
    MATPHYS_RECONSTRUCTION_ADAPTER_AUDIT_CONTRACT,
)


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output_json")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output_json).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint omits its model state")
    names = (
        "part_feature_encoder.1.weight",
        "part_feature_encoder.1.bias",
    )
    tensors = {}
    finite = True
    for name in names:
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint omits {name}")
        value = value.detach().float().cpu()
        value_finite = bool(torch.isfinite(value).all())
        finite = finite and value_finite
        tensors[name] = {
            "shape": list(value.shape),
            "finite": value_finite,
            "l2_norm": float(value.norm()),
            "maximum_absolute": float(value.abs().max()),
            "nonzero_count": int(torch.count_nonzero(value)),
        }
    moved = tensors[names[0]]["nonzero_count"] > 0 and tensors[names[0]]["l2_norm"] > 0.0
    result = {
        "schema_version": 1,
        "contract": MATPHYS_RECONSTRUCTION_ADAPTER_AUDIT_CONTRACT,
        "checkpoint": _identity(checkpoint_path),
        "finite": finite,
        "adapter_moved_from_zero": moved,
        "zero_initialized_parameters": list(names),
        "tensors": tensors,
    }
    if not finite or not moved:
        raise RuntimeError("terminal MatPhys part adapter did not move finitely")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "audit_path": str(output_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
