#!/usr/bin/env python3
"""Audit stricter subsets of the opened PokeFlex v2 guarded predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_guard_postopen_audit import (  # noqa: E402
    audit_sealed_guard_rows,
)

RESULT_SHA256 = "aa2680cbe0d7a6c9e342c9093ff4045e25a6952191fc9033376516d468329685"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_result", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace audit output: {args.output}")
    if _sha256(args.target_result) != RESULT_SHA256:
        raise ValueError("registered target result bytes changed")
    target = json.loads(args.target_result.read_text(encoding="utf-8"))
    scored_rows = []
    prediction_hashes = {}
    for object_row in target["objects"]:
        take_id = str(object_row["take_id"])
        prediction_path = args.prediction_root / take_id / "prediction.npz"
        prediction_hashes[take_id] = _sha256(prediction_path)
        with np.load(prediction_path, allow_pickle=False) as archive:
            frames = np.asarray(archive["target_frames"], dtype=np.int64)
            accepted = np.asarray(archive["guard_accepted"], dtype=np.bool_)
            upper = np.asarray(archive["guard_upper_regret_mm"], dtype=np.float64)
        by_frame = {int(frame): index for index, frame in enumerate(frames)}
        for frame_row in object_row["frames"]:
            frame = int(frame_row["target_frame"])
            index = by_frame[frame]
            if bool(frame_row["update_supported"]) != bool(accepted[index]):
                raise ValueError(f"scored support differs from seal for {take_id}/{frame}")
            scored_rows.append(
                {
                    "take_id": take_id,
                    "object_name": str(object_row["object_name"]),
                    "target_frame": frame,
                    "baseline_error_mm": float(frame_row["baseline_CD_UL1_mm"]),
                    "candidate_error_mm": float(frame_row["candidate_CD_UL1_mm"]),
                    "accepted": bool(accepted[index]),
                    "upper_regret_mm": (
                        float(upper[index]) if accepted[index] else None
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBaselineRelativeGuardPostOpenAudit",
        "target_result_sha256": RESULT_SHA256,
        "prediction_npz_sha256": prediction_hashes,
        "audit": audit_sealed_guard_rows(scored_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(payload["audit"], indent=2))


if __name__ == "__main__":
    main()
