#!/usr/bin/env python3
"""Import only fit/validation cached forecasts from a checksum-bound sibling export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CHECKPOINT = "b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65"
MANIFEST = "7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98"
TRAINING_ROOT = Path(
    "/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7/training_run"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = TRAINING_ROOT / "source_manifest.json"
    checkpoint_path = TRAINING_ROOT / "checkpoints/update_6400.pt"
    if sha(manifest_path) != MANIFEST or sha(checkpoint_path) != CHECKPOINT:
        raise ValueError("Original manifest or checkpoint changed")
    manifest = json.loads(manifest_path.read_text())
    arrays, receipts = [], {}
    for split, count in [("fit", 40), ("validation", 8)]:
        path = args.cache_root / (split + ".npz")
        receipt_path = args.cache_root / (split + ".json")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["checkpoint_sha256"] != CHECKPOINT
            or receipt["manifest_sha256"] != MANIFEST
            or receipt["sha256"] != sha(path)
            or receipt["official_eval_read"] is not False
        ):
            raise ValueError("Sibling export receipt failed verification")
        with np.load(path, allow_pickle=False) as archive:
            data = {
                key: archive[key].copy()
                for key in (
                    "initial",
                    "action",
                    "predictions",
                    "targets",
                    "persistence",
                    "names",
                )
            }
        names = data["names"].tolist()
        if (
            names != manifest["split"][split]
            or len(names) != count
            or receipt["names"] != names
        ):
            raise ValueError("Cached trajectory identities/order differ")
        if set(names).intersection(manifest["split"]["source_test"]):
            raise ValueError("Forbidden source-test overlap")
        if data["predictions"].shape != data["targets"].shape or data[
            "predictions"
        ].shape != (count, 498, 12, 3):
            raise ValueError("Unexpected cached forecast shape")
        if not all(
            np.isfinite(value).all() for key, value in data.items() if key != "names"
        ):
            raise ValueError("Nonfinite cached numeric array")
        loss = float(
            np.mean(np.abs(data["predictions"] - data["targets"]), dtype=np.float64)
        )
        if abs(loss - receipt["baseline_l1_m"]) > 1e-8:
            raise ValueError("Cached baseline does not reproduce its receipt")
        if split == "validation" and abs(loss - 0.007912029745057225) > 1e-7:
            raise ValueError("Historical baseline failed the original parity tolerance")
        receipts[split] = dict(
            receipt=receipt, receipt_sha256=sha(receipt_path), path=str(path)
        )
        arrays.append(data)
    combined = {
        key: np.concatenate([part[key] for part in arrays]) for key in arrays[0]
    }
    np.savez_compressed(
        out / "development.npz",
        initial=combined["initial"],
        action=combined["action"],
        baseline=combined["predictions"],
        target=combined["targets"],
        persistence=combined["persistence"],
        names=combined["names"],
        fit_count=np.asarray([40]),
    )
    meta = dict(
        schema="material-gp-development-cache-import-v1",
        scope="retrospective development only; imported identical permitted fit/validation partitions",
        npz_sha256=sha(out / "development.npz"),
        checkpoint_sha256=CHECKPOINT,
        manifest_sha256=MANIFEST,
        source_test_read=False,
        official_eval_read=False,
        import_sha256=sha(__file__),
        implementation_sha=os.environ.get("GITHUB_SHA"),
        producer_implementation="IPS-Stuttgart/BayesianPhysTwin@8429018c1e779b9dd217c1bfa93547eb47b06025:experiments/deform_gp_residual_pilot_v1/export.py",
        import_only_no_raw_trajectory_reads=True,
        upstream_receipts=receipts,
        baseline_validation_l1_m=loss,
        baseline_reference_l1_m=0.007912029745057225,
        fit_names=arrays[0]["names"].tolist(),
        validation_names=arrays[1]["names"].tolist(),
    )
    (out / "export.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("VERIFIED_CACHE", json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
