"""Prepare frozen DEFORM DLO1 development rollouts, never official/source-test data.

This is exploratory reuse of already-open fit/validation trajectories. No
historical scientific protocol is rerun, promoted, modified, or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "remote"))
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol_path = ROOT / "configs/sota/deform_dlo_local_residual_v4.json"
    protocol = json.loads(protocol_path.read_text())
    longrun_path = Path(protocol["longrun_result"]["path"])
    manifest_path = ROOT / protocol["source_manifest"]["repository_path"]
    for path, expected in (
        (longrun_path, protocol["longrun_result"]["sha256"]),
        (manifest_path, protocol["source_manifest"]["sha256"]),
    ):
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Frozen parent unavailable or changed: {path}")
    longrun = json.loads(longrun_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest["dlo_type"] != "DLO1" or manifest["partition"] != "train":
        raise RuntimeError("Only DLO1/train development is authorized")
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    if len(fit_names) != 40 or len(validation_names) != 8:
        raise RuntimeError("Historical development roster changed")
    names = fit_names + validation_names
    if len(set(names)) != 48:
        raise RuntimeError("Development split names overlap")
    allowed = [Path(manifest["trajectories"][n]["path"]).resolve() for n in names]
    upstream = allowed[0].parents[3]
    if not all(p.parent == upstream / "data_set/DLO1/train" for p in allowed):
        raise RuntimeError("Development paths do not share the authorized partition")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import run_deform_dlo_action_residual as common
    import run_deform_dlo_longrun_posterior as posterior
    import run_deform_dlo_source as source
    import torch

    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        deform_causal_inputs,
    )

    source._assert_upstream(upstream, longrun["upstream"]["commit"])
    source._install_eval_read_guard(upstream / "data_set/DLO1/eval")
    for dlo in ("DLO2", "DLO3", "DLO4", "DLO5"):
        source._install_eval_read_guard(upstream / "data_set" / dlo)
    for name in manifest["split"]["source_test"]:
        source._install_eval_read_guard(Path(manifest["trajectories"][name]["path"]))
    selected = longrun["selected_checkpoint"]
    if selected["checkpoint"]["sha256"] != protocol["baseline"]["checkpoint_sha256"]:
        raise RuntimeError("Upstream hybrid checkpoint differs")
    source._seed_everything(torch, 42)
    modules = source._load_upstream(upstream)
    state = posterior._checkpoint_states(longrun, {6400}, torch=torch)[6400]
    data = source._load_named_trajectories(
        manifest, names, frame_count=500, node_count=13
    )
    print(
        "Preparing 40 fit and 8 validation trajectories; no source-test/eval access",
        flush=True,
    )
    rollout = common._rollout(
        state, data, modules=modules, torch=torch, device="cuda:0"
    )
    order = [list(rollout["names"]).index(name) for name in names]
    baseline = np.asarray(rollout["predictions"])[order]
    targets = np.asarray(rollout["targets"])[order]
    persistence = np.asarray(rollout["persistence"])[order]
    initial, action = deform_causal_inputs(np.stack([data[name] for name in names]))
    if targets.shape != (48, 498, 13, 3) or baseline.shape != targets.shape:
        raise RuntimeError(f"Unexpected recursive rollout shape: {targets.shape}")
    error = float(np.mean(np.abs(baseline[40:] - targets[40:])))
    expected = float(protocol["baseline"]["validation_l1_m"])
    if abs(error - expected) > float(protocol["baseline"]["reproduction_tolerance_m"]):
        raise RuntimeError(f"Frozen baseline parity failed: {error} versus {expected}")
    bundle = out / "development.npz"
    np.savez_compressed(
        bundle,
        names=np.asarray(names),
        initial=initial,
        action=action,
        baseline=baseline,
        targets=targets,
        persistence=persistence,
        split=np.asarray(["fit"] * 40 + ["validation"] * 8),
    )
    record = {
        "contract": "dp-residual-dlo1-development-bundle-v1",
        "claim_boundary": "Exploratory historical DLO1 fit/validation only; not independent confirmation.",
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "bundle_sha256": digest(bundle),
        "protocol_sha256": digest(protocol_path),
        "source_manifest_sha256": digest(manifest_path),
        "longrun_result_sha256": digest(longrun_path),
        "checkpoint_sha256": selected["checkpoint"]["sha256"],
        "upstream_commit": longrun["upstream"]["commit"],
        "baseline_description": "Frozen retrained DEFORM hybrid including its own GCN correction",
        "validation_baseline_l1_mm": 1000 * error,
        "fit_count": 40,
        "validation_count": 8,
        "source_test_read": False,
        "official_eval_read": False,
        "other_dlos_read": False,
        "query_contract": "two observed states, known clamped action, baseline rollout only",
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (out / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
