#!/usr/bin/env python3
"""Cache DLO2 fit/validation forecasts only; never open source-test or eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/remote")]
TRAINING_ROOT = Path(
    "/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7/training_run"
)
UPSTREAM = Path("/home/florianpfaff/source-only/deform-bayesian-v1/DEFORM-b73b8b8")
EXPECTED_TRAINING = "1f8d092bc38b03f6cdd68ef38abcb7d403d914e38ba483698579deaeea8c2572"
EXPECTED_CHECKPOINT = "b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "export.json").exists():
        meta = json.loads((out / "export.json").read_text())
        if meta["npz_sha256"] != sha(out / "development.npz"):
            raise ValueError("Cached development arrays changed")
        if meta["checkpoint_sha256"] != EXPECTED_CHECKPOINT:
            raise ValueError("Cached checkpoint differs")
        print(json.dumps(meta, indent=2), flush=True)
        return 0
    if (out / "development.npz").exists():
        raise RuntimeError("Incomplete cache must not be silently reused")
    import run_deform_dlo2_local_residual as v5
    import run_deform_dlo_local_residual as local
    import run_deform_dlo_source as source

    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        load_deform_dlo2_local_residual_protocol,
    )

    protocol_path = ROOT / "configs/sota/deform_dlo2_local_residual_v5.json"
    protocol = load_deform_dlo2_local_residual_protocol(protocol_path)
    training_path = TRAINING_ROOT / "training_validation_result.json"
    if sha(training_path) != EXPECTED_TRAINING:
        raise ValueError("Frozen training record changed")
    training, manifest, manifest_path = v5._verify_training_result(
        training_path, protocol=protocol, protocol_path=protocol_path
    )
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    names = fit_names + validation_names
    if len(fit_names) != 40 or len(validation_names) != 8 or len(set(names)) != 48:
        raise ValueError("Unexpected development split")
    if set(names).intersection(manifest["split"]["source_test"]):
        raise ValueError("Development/source-test overlap")
    for dlo in ("DLO1", "DLO3", "DLO4", "DLO5"):
        source._install_eval_read_guard(UPSTREAM / "data_set" / dlo)
    source._install_eval_read_guard(UPSTREAM / "data_set/DLO2/eval")
    allowed = set(names)
    reads = set()

    def guard(event, values):
        if event == "open" and values and isinstance(values[0], (str, bytes)):
            p = Path(os.fsdecode(values[0]))
            if p.suffix == ".pkl":
                if p.name not in allowed or "eval" in p.parts:
                    raise RuntimeError("Forbidden trajectory read: " + str(p))
                reads.add(str(p.resolve()))

    sys.addaudithook(guard)
    source._assert_upstream(UPSTREAM, protocol["upstream"]["commit"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    source._seed_everything(torch, 42)
    torch.set_num_threads(4)
    modules = source._load_upstream(UPSTREAM)
    state, checkpoint = v5._checkpoint_state(training, torch=torch)
    if sha(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("Frozen checkpoint changed")
    print(
        "Loading 40 fit + 8 development-validation trajectories; no source-test/eval",
        flush=True,
    )
    trajectories = source._load_named_trajectories(
        manifest, names, frame_count=500, node_count=12
    )
    print("Replaying frozen 6400-update DEFORM hybrid", flush=True)
    rollout = v5._rollout(
        state, trajectories, modules=modules, torch=torch, device="cuda:0"
    )
    if list(rollout["names"]) != names:
        raise ValueError("Rollout order mismatch")
    initial, action = local._causal_inputs(trajectories, names)
    prediction = np.asarray(rollout["predictions"])
    target = np.asarray(rollout["targets"])
    observed = float(np.mean(np.abs(prediction[40:] - target[40:]), dtype=np.float64))
    expected = float(training["selected_checkpoint"]["validation_l1_m"])
    if abs(observed - expected) > 1e-7:
        raise ValueError(f"Baseline failed reproduction: {observed} versus {expected}")
    np.savez_compressed(
        out / "development.npz",
        initial=initial,
        action=action,
        baseline=prediction,
        target=target,
        names=np.asarray(names),
        fit_count=np.asarray([40]),
        persistence=np.asarray(rollout["persistence"]),
    )
    meta = dict(
        schema="material-gp-development-export-v1",
        scope="retrospective development only; no independent confirmation",
        npz_sha256=sha(out / "development.npz"),
        checkpoint_sha256=sha(checkpoint),
        training_sha256=sha(training_path),
        manifest_sha256=sha(manifest_path),
        implementation_sha=os.environ.get("GITHUB_SHA", "local"),
        exporter_sha256=sha(Path(__file__)),
        runtime={
            k: sha(ROOT / "scripts/remote" / k)
            for k in (
                "run_deform_dlo2_local_residual.py",
                "run_deform_dlo_longrun_posterior.py",
                "run_deform_dlo_local_residual.py",
                "run_deform_dlo_source.py",
            )
        },
        python=sys.version,
        numpy=np.__version__,
        torch=torch.__version__,
        fit_names=fit_names,
        validation_names=validation_names,
        trajectory_reads=sorted(reads),
        official_eval_read=False,
        source_test_read=False,
        baseline_validation_l1_m=observed,
        baseline_reference_l1_m=expected,
    )
    (out / "export.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
