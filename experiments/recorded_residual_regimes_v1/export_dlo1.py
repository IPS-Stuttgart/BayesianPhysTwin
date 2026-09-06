"""Export existing DLO1 development recordings; never open official evaluation.

Research diagnostic owned by #951. Reuses frozen DEFORM and local-residual
implementations without changing them. The source-test partition is already-open
retrospective development data, not new independent confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts/remote")]
    manifest_path = repo / "results/sota/deform_dlo_source_v1/source_manifest.json"
    if digest(manifest_path) != "570e8f65a4a9c5b5bbfeb923fcb8714885896ec041d862548b91ef6ff1599a8a":
        raise ValueError("DLO1 source manifest identity changed")
    manifest = json.loads(manifest_path.read_text())
    splits = manifest["split"]
    assert [len(splits[k]) for k in ("fit", "validation", "source_test")] == [40, 8, 8]
    names = sum((splits[k] for k in ("fit", "validation", "source_test")), [])
    assert len(set(names)) == 56
    checkpoint = Path("/home/florianpfaff/source-only/deform-bayesian-v2/runs/longrun-v2-2060c71/checkpoints/update_6400.pt")
    expected = "ea2b9e83d09a05bf94eae25aa1dafb449868d4a31961b06ec7b4420216d726e0"
    if digest(checkpoint) != expected:
        raise ValueError("DLO1 frozen checkpoint identity changed")
    upstream = Path(manifest["trajectories"][names[0]]["path"]).parents[3]
    allowed = upstream / "data_set/DLO1/train"
    for name in names:
        path = Path(manifest["trajectories"][name]["path"]).resolve()
        if path.parent != allowed.resolve() or path.name != name:
            raise ValueError("Non-development trajectory path")
        if digest(path) != manifest["trajectories"][name]["sha256"]:
            raise ValueError("Trajectory identity changed: " + name)
    import run_deform_dlo_source as source
    import run_deform_dlo_longrun_posterior as replay
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        deform_causal_inputs,
        fit_deform_local_residual,
        predict_deform_local_residual,
        serialize_deform_local_residual_model,
    )
    upstream_identity = source._assert_upstream(upstream, "b73b8b8ecc033caefa693fab7898741d4e6dbeff")
    for dlo in ("DLO1", "DLO2", "DLO3", "DLO4", "DLO5"):
        source._install_eval_read_guard(upstream / "data_set" / dlo / "eval")
    for dlo in ("DLO2", "DLO3", "DLO4", "DLO5"):
        source._install_eval_read_guard(upstream / "data_set" / dlo)
    record = {
        "schema": "recorded-residual-dlo1-export-v1", "issue": 951,
        "evidence_class": "retrospective-development-only",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "manifest_sha256": digest(manifest_path), "checkpoint_sha256": expected,
        "upstream": upstream_identity, "split": splits,
        "official_eval_read": False, "dlo2_to_dlo5_read": False,
        "source_test_previously_opened": True,
        "future_free_node_truth_used_for_prediction": False,
        "python": sys.version, "hostname": platform.node(),
    }
    (out / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    source._seed_everything(torch, 42)
    modules = source._load_upstream(upstream)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(bundle["global_update"]) != 6400:
        raise ValueError("Unexpected checkpoint update")
    arrays = source._load_named_trajectories(manifest, names, frame_count=500, node_count=13)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Replaying 56 DLO1 development recordings on " + device, flush=True)
    rollout = replay._evaluate_state(bundle["model_state_dict"], arrays, modules=modules, torch=torch, device=device)
    assert rollout["names"] == names
    full = np.stack([arrays[name] for name in names])
    initial, action = deform_causal_inputs(full)
    base = np.asarray(rollout["predictions"], dtype=np.float64)
    truth = np.asarray(rollout["targets"], dtype=np.float64)
    validation_l1 = float(np.mean(np.abs(base[40:48] - truth[40:48])))
    if abs(validation_l1 - 0.009256238292437047) > 1e-7:
        raise ValueError(f"Frozen hybrid replay drifted: {validation_l1}")
    model = fit_deform_local_residual(initial[:40], action[:40], base[:40], truth[:40], names[:40], ridge=1.0, variance_floor_m2=1e-6)
    local = predict_deform_local_residual(model, initial, action, base, shrinkage=0.5)["predictions"]
    features, frames = build_deform_local_residual_features(initial, action, base)
    # Compact causal exogenous features; no response residual enters this array.
    features = np.concatenate((features.mean(axis=2), features[:, :, :, 12:15].reshape(56, base.shape[1], -1)), axis=-1)
    np.savez_compressed(out / "local_model.npz", **serialize_deform_local_residual_model(model))
    for key, start, stop in (("fit", 0, 40), ("validation", 40, 48), ("source_test", 48, 56)):
        np.savez_compressed(out / (key + ".npz"), names=np.asarray(names[start:stop]), hybrid=base[start:stop], local=local[start:stop], truth=truth[start:stop], features=features[start:stop], frames=frames[start:stop])
    record.update(torch=torch.__version__, numpy=np.__version__, device=device, validation_hybrid_l1_m=validation_l1,
        source_hybrid_l1_m=float(np.mean(np.abs(base[48:] - truth[48:]))),
        source_local_l1_m=float(np.mean(np.abs(local[48:] - truth[48:]))),
        artifacts={p.name: digest(p) for p in sorted(out.glob("*.npz"))})
    (out / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
