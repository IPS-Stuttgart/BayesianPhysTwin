"""Prepare the frozen GP pilot from verified fit/validation rollout caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.deform_gp_residual_dev_v1.run import ROOT, TRAIN_ROOT, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "configs/sota/deform_dlo2_local_residual_v6.json"
    protocol = json.loads(protocol_path.read_text())
    identities = {
        "training_result": TRAIN_ROOT / "training_validation_result.json",
        "source_manifest": TRAIN_ROOT / "source_manifest.json",
        "selected_checkpoint": TRAIN_ROOT / "checkpoints/update_6400.pt",
    }
    for key, path in identities.items():
        if sha(path) != protocol[key]["sha256"]:
            raise ValueError("Frozen identity mismatch: " + key)
    manifest = json.loads(identities["source_manifest"].read_text())
    training = json.loads(identities["training_result"].read_text())
    if manifest["partition"] != "train" or manifest["dlo_type"] != "DLO2":
        raise ValueError("Only original DLO2 training partition is permitted")
    receipts, query_hashes = {}, {}
    for split, count in (("fit", 40), ("validation", 8)):
        names = manifest["split"][split]
        if len(names) != count or set(names) & set(manifest["split"]["source_test"]):
            raise ValueError("Development roster changed")
        path = args.cache / (split + ".npz")
        receipt_path = args.cache / (split + ".json")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["checkpoint_sha256"] != protocol["selected_checkpoint"]["sha256"]
            or receipt["manifest_sha256"] != protocol["source_manifest"]["sha256"]
            or receipt["sha256"] != sha(path)
            or receipt["official_eval_read"] is not False
            or receipt["names"] != names
        ):
            raise ValueError("Cached rollout receipt does not verify")
        with np.load(path, allow_pickle=False) as archive:
            data = {
                key: archive[key].copy()
                for key in ("initial", "action", "predictions", "targets", "names")
            }
        if data["names"].tolist() != names:
            raise ValueError("Cached trajectory order changed")
        if (
            data["predictions"].shape != (count, 498, 12, 3)
            or data["targets"].shape != (count, 498, 12, 3)
            or data["initial"].shape != (count, 2, 12, 3)
            or data["action"].shape != (count, 498, 4, 3)
        ):
            raise ValueError("Unexpected development array shape")
        for key in ("initial", "action", "predictions", "targets"):
            data[key] = np.asarray(data[key], dtype=np.float64)
            if not np.isfinite(data[key]).all():
                raise ValueError("Nonfinite development value")
        loss = float(np.mean(np.abs(data["predictions"] - data["targets"])))
        if abs(loss - receipt["baseline_l1_m"]) > 1e-8:
            raise ValueError("Cache does not reproduce its baseline receipt")
        if split == "validation":
            expected = float(training["selected_checkpoint"]["validation_l1_m"])
            if abs(loss - expected) > 1e-7:
                raise ValueError("Historical baseline reproduction failed")
        prepared = {
            "initial": data["initial"],
            "action": data["action"],
            "baseline": data["predictions"],
            "names": data["names"],
        }
        if split == "fit":
            prepared["targets"] = data["targets"]
        else:
            np.savez_compressed(output / "validation_truth.npz", targets=data["targets"])
        np.savez_compressed(output / (split + ".npz"), **prepared)
        query_hashes[split] = [
            hashlib.sha256(
                data["initial"][i].tobytes()
                + data["action"][i].tobytes()
                + data["predictions"][i].tobytes()
            ).hexdigest()
            for i in range(count)
        ]
        receipts[split] = {
            "receipt": receipt,
            "receipt_sha256": sha(receipt_path),
            "path": str(path),
            "baseline_recomputed_l1_m": loss,
        }
    if set(query_hashes["fit"]) & set(query_hashes["validation"]):
        raise ValueError("Duplicate causal query crosses partitions")
    if set(manifest["split"]["fit"]) & set(manifest["split"]["validation"]):
        raise ValueError("Overlapping development names")
    write_json(
        output / "preparation.json",
        {
            "training_result_sha256": sha(identities["training_result"]),
            "source_manifest_sha256": sha(identities["source_manifest"]),
            "checkpoint_sha256": sha(identities["selected_checkpoint"]),
            "fit_names": manifest["split"]["fit"],
            "validation_names": manifest["split"]["validation"],
            "causal_query_digests": query_hashes,
            "source_test_read": False,
            "official_eval_read": False,
            "code_revision": os.environ.get("EXPERIMENT_REVISION"),
            "numpy_version": np.__version__,
            "upstream_receipts": receipts,
            "preparation_mode": "verified-sibling-rollout-cache",
            "producer_revision": "8429018c1e779b9dd217c1bfa93547eb47b06025",
            "producer_path": "experiments/deform_gp_residual_pilot_v1/export.py",
            "same_rollout_operator": "run_deform_dlo_longrun_posterior._evaluate_state",
        },
    )
    print("VERIFIED 40 fit / 8 validation; fixed checkpoint and baseline parity", flush=True)


if __name__ == "__main__":
    main()
