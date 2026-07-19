#!/usr/bin/env python3
"""Time the pinned MatPhys model's distributed initialization stages."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from bayesian_phystwin.matphys_part_model import install_part_aware_simple_model
from scripts.remote.run_matphys_causal import (
    _configure_matphys_imports,
    _install_torchvision_nms_stub,
)


def _stage(name: str, started: float) -> None:
    rank = int(os.environ.get("RANK", "0"))
    elapsed = time.monotonic() - started
    print(f"rank={rank} stage={name} elapsed_s={elapsed:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matphys_root")
    parser.add_argument("--part-feature-dim", type=int, default=1024)
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--skip-matphys-imports", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    _stage("process_group", started)

    training = None
    if not args.skip_matphys_imports:
        root = Path(args.matphys_root).resolve()
        os.chdir(root)
        _install_torchvision_nms_stub()
        _configure_matphys_imports(root)
        import train_model_video_material_simple as training

        install_part_aware_simple_model(
            training,
            part_feature_dim=args.part_feature_dim,
            part_feature_scale=1.0,
        )
    _stage("imports", started)
    if training is None and not args.linear_only:
        raise ValueError("--skip-matphys-imports requires --linear-only")
    model = (
        torch.nn.Linear(4, 4)
        if args.linear_only
        else training.SimpleVideoMaterialPhysicsModel(
            part_feature_dim=args.part_feature_dim,
            part_feature_scale=1.0,
        )
    )
    _stage("model_constructed", started)
    model = model.to(device)
    _stage("model_on_device", started)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
    )
    _stage("ddp_wrapped", started)
    dist.barrier()
    _stage("barrier", started)
    del model
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
