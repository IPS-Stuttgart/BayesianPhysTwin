import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402


def _uneven_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
) -> None:
    from scripts.remote.run_matphys_causal import _install_uneven_ddp_training

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        training = SimpleNamespace()
        training._is_distributed = lambda: True
        training.forward_case = lambda wrapped, value: wrapped(value)

        def train_epoch(
            wrapped,
            train_loader,
            active_optimizer,
            device,
            runtimes,
            args,
            epoch,
        ):
            del device, runtimes, args, epoch
            total = 0.0
            for value in train_loader:
                active_optimizer.zero_grad(set_to_none=True)
                prediction = training.forward_case(wrapped, value)
                loss = prediction.square().mean()
                loss.backward()
                active_optimizer.step()
                total += float(loss.detach())
            count = len(train_loader)
            mean = total / count
            return {
                "loss": mean,
                "track": mean,
                "geo": mean,
                "render": 0.0,
                "acc": 0.0,
                "phys_part": 0.0,
                "phys_global": 0.0,
                "teacher_log_k": 0.0,
                "teacher_global": 0.0,
                "num_graphs": count,
            }

        training.train_epoch = train_epoch
        _install_uneven_ddp_training(training)
        loader = [torch.tensor([[1.0]])] if rank == 0 else [
            torch.tensor([[2.0]]),
            torch.tensor([[3.0]]),
        ]
        stats = training.train_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            {},
            SimpleNamespace(),
            0,
        )
        parameter = next(model.parameters())
        state = optimizer.state[parameter]
        local = torch.tensor(
            [
                float(parameter.detach().item()),
                float(state["exp_avg"].item()),
                float(state["exp_avg_sq"].item()),
            ]
        )
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        if rank == 0:
            Path(result_file).write_text(
                json.dumps(
                    {
                        "states": [value.tolist() for value in gathered],
                        "num_graphs": stats["num_graphs"],
                    }
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


def test_uneven_ddp_syncs_model_and_optimizer_state(tmp_path: Path) -> None:
    init_file = tmp_path / "gloo_init"
    result_file = tmp_path / "result.json"

    mp.spawn(
        _uneven_worker,
        args=(2, str(init_file), str(result_file)),
        nprocs=2,
        join=True,
    )

    result = json.loads(result_file.read_text())
    assert result["num_graphs"] == 3
    assert result["states"][0] == pytest.approx(result["states"][1])
