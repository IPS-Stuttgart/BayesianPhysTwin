"""Read-only source-partition rollout export for the GP pilot; no official eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7/training_run"
)
UPSTREAM = Path("/home/florianpfaff/source-only/deform-bayesian-v1/DEFORM-b73b8b8")
MANIFEST_HASH = "7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98"
CHECKPOINT_HASH = "b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(cache: Path, partitions: list[str]) -> dict:
    if not set(partitions).issubset({"fit", "validation", "source_test"}):
        raise ValueError("only named original training-side partitions are permitted")
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "source_manifest.json"
    checkpoint = ROOT / "checkpoints/update_6400.pt"
    if digest(manifest_path) != MANIFEST_HASH or digest(checkpoint) != CHECKPOINT_HASH:
        raise ValueError("frozen manifest/checkpoint identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["partition"] != "train" or manifest["dlo_type"] != "DLO2":
        raise ValueError("not the registered DLO2 training manifest")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import run_deform_dlo_longrun_posterior as posterior
    import run_deform_dlo_source as source
    import torch

    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        deform_causal_inputs,
    )

    for dlo in ("DLO1", "DLO2", "DLO3", "DLO4", "DLO5"):
        source._install_eval_read_guard(UPSTREAM / "data_set" / dlo / "eval")
    source._seed_everything(torch, 42)
    modules = source._load_upstream(UPSTREAM)
    state = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
    metadata = {
        "checkpoint_sha256": CHECKPOINT_HASH,
        "manifest_sha256": MANIFEST_HASH,
        "source_revision": os.environ.get("GITHUB_SHA"),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "official_eval_read": False,
        "partitions": {},
    }
    print("EXPORT_RUNTIME", json.dumps(metadata), flush=True)
    for partition in partitions:
        names = list(manifest["split"][partition])
        output = cache / f"{partition}.npz"
        receipt = cache / f"{partition}.json"
        if output.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            if (
                record["checkpoint_sha256"] != CHECKPOINT_HASH
                or record["manifest_sha256"] != MANIFEST_HASH
                or record["sha256"] != digest(output)
            ):
                raise ValueError("stale or altered rollout cache")
            print("CACHE_VERIFIED", partition, record["sha256"], flush=True)
            metadata["partitions"][partition] = record
            continue
        for name in names:
            p = Path(manifest["trajectories"][name]["path"])
            if p.parent != UPSTREAM / "data_set/DLO2/train":
                raise ValueError("trajectory outside training path")
        started = time.monotonic()
        print("ROLLOUT_BEGIN", partition, len(names), flush=True)
        trajectories = source._load_named_trajectories(
            manifest, names, frame_count=500, node_count=12
        )
        rollout = posterior._evaluate_state(
            state,
            trajectories,
            modules=modules,
            torch=torch,
            device="cuda:0",
            dlo_type="DLO2",
            node_count=12,
        )
        if rollout["names"] != names:
            raise ValueError("rollout changed trajectory order")
        array = np.stack([trajectories[name] for name in names])
        initial, action = deform_causal_inputs(array)
        values = {
            key: np.asarray(rollout[key])
            for key in ("predictions", "targets", "persistence")
        }
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("nonfinite frozen hybrid rollout")
        np.savez_compressed(
            output, names=np.asarray(names), initial=initial, action=action, **values
        )
        l1 = np.mean(np.abs(values["predictions"] - values["targets"]), axis=(1, 2, 3))
        record = {
            "checkpoint_sha256": CHECKPOINT_HASH,
            "manifest_sha256": MANIFEST_HASH,
            "sha256": digest(output),
            "names": names,
            "shape": list(values["predictions"].shape),
            "baseline_l1_m": float(l1.mean()),
            "case_l1_m": l1.tolist(),
            "elapsed_seconds": time.monotonic() - started,
            "official_eval_read": False,
        }
        if partition == "validation":
            expected = 0.007912029745057225
            record["historical_baseline_l1_m"] = expected
            record["baseline_absolute_parity_error_m"] = abs(
                float(l1.mean()) - expected
            )
            if record["baseline_absolute_parity_error_m"] > 1e-6:
                raise ValueError("frozen hybrid validation parity failed")
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        metadata["partitions"][partition] = record
        print("ROLLOUT_END", partition, json.dumps(record), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", default=["validation", "fit"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = export(args.cache, args.partitions)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
