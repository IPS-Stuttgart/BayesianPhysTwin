#!/usr/bin/env python3
"""Frozen standard optimizer for nominal source-task competence, CPU only."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_benchmark import (
    source_identity,
    write_native_bundle,
)
from bayesian_phystwin_experiments.dlolab_native import array_digest, file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    clean_revision,
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_slingshot_batch import run_batch, split_batch
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import (
    final_checks,
    protocol,
    task_metrics,
    verify_inputs,
    verify_retained_failure,
    worker_environment,
)
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    run_native,
    runtime,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_cmaes.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_controls.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_native.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "scripts/remote/run_dlolab_slingshot_cmaes.py",
    "tests/test_dlolab_slingshot_cmaes.py",
    "docs/dlolab_slingshot_cmaes_source_v1.md",
    "docs/dlolab_slingshot_cmaes_runtime_v1_1.md",
)


def directory_for(output: Path, index: int) -> Path:
    if type(index) is not int or index not in range(-1, 8):
        raise ValueError("unregistered optimizer execution")
    return output / ("best-replay" if index == -1 else f"batch-{index:02d}")


def worker(output: Path, index: int) -> None:
    lock = read_record(output / "lock.json")
    if (
        lock["source_revision"] != clean_revision(ROOT)
        or lock["source_sha256"] != {name: file_digest(ROOT / name) for name in SOURCES}
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(output.resolve())
    ):
        raise ValueError("optimizer source/lock changed")
    if (
        verify_retained_failure(Path(lock["retained_failure"]["path"]))
        != lock["retained_failure"]
    ):
        raise ValueError("retained failure binding changed")
    if (
        runtime() != lock["verified"]["qualification"]["runtime"]
        or importlib.metadata.version("cma") != protocol()["cma_version"]
    ):
        raise ValueError("optimizer runtime changed")
    assets = Path(lock["assets_root"])
    if (
        source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != lock["verified"]["qualification"]["native_source"]
    ):
        raise ValueError("native source changed")
    directory = directory_for(output, index)
    plan = read_record(directory / "plan.json")
    if plan["lock_id"] != lock["artifact_id"] or plan["index"] != index:
        raise ValueError("optimizer plan binding changed")
    if index == -1:
        selection = read_record(output / "selection.json")
        if (
            plan["selection_id"] != selection["artifact_id"]
            or len(selection["generation_ids"]) != 4
        ):
            raise ValueError("completed optimizer selection required")
        for batch in range(8):
            read_record(directory_for(output, batch) / "output/seal.json")
    inputs = load_native_bundle(directory / "input", plan["input_bundle"])
    if set(inputs) != {"controls"} or inputs["controls"].shape != (
        (1 if index == -1 else 8),
        3,
        6,
    ):
        raise ValueError("optimizer control layout changed")
    target = directory / "output"
    target.mkdir(exist_ok=False)
    claim = write_record(
        target / "claim.json",
        {
            "schema": "dlolab-slingshot-cmaes-claim-v1",
            "plan_id": plan["artifact_id"],
            "retry_authorized": False,
        },
    )
    try:
        execute = run_native if index == -1 else run_batch
        arrays, native = execute(assets / "upstream", target, inputs["controls"])
        bundle = write_native_bundle(target, arrays)
        write_record(
            target / "seal.json",
            {
                "schema": "dlolab-slingshot-cmaes-seal-v1",
                "lock_id": lock["artifact_id"],
                "plan_id": plan["artifact_id"],
                "claim_id": claim["artifact_id"],
                "bundle": bundle,
                "native": native,
            },
        )
    except Exception as error:
        write_record(
            target / "failure.json",
            {
                "schema": "dlolab-slingshot-cmaes-failure-v1",
                "claim_id": claim["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
            },
        )
        raise


def execute_plan(
    output: Path,
    lock: dict[str, Any],
    index: int,
    controls: np.ndarray,
    selection_id: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = directory_for(output, index)
    directory.mkdir(exist_ok=False)
    (directory / "input").mkdir()
    bundle = write_native_bundle(directory / "input", {"controls": controls})
    plan = write_record(
        directory / "plan.json",
        {
            "schema": "dlolab-slingshot-cmaes-plan-v1",
            "lock_id": lock["artifact_id"],
            "index": index,
            "input_bundle": bundle,
            "selection_id": selection_id,
        },
    )
    with (directory / "execution.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output.resolve()),
                "--worker-index",
                str(index),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=worker_environment(lock["verified"]["qualification"]["runtime"]),
        )
    seal = read_record(directory / "output/seal.json")
    if seal["lock_id"] != lock["artifact_id"] or seal["plan_id"] != plan["artifact_id"]:
        raise ValueError("optimizer execution seal changed")
    arrays = load_native_bundle(directory / "output", seal["bundle"])
    if array_digest(arrays["controls"]) != array_digest(controls):
        raise ValueError("executed controls changed")
    return arrays, seal


def run(
    output: Path,
    assets: Path,
    batch_result: Path,
    source_result: Path,
    retained_failure: Path,
) -> dict[str, Any]:
    revision = clean_revision(ROOT)
    retained = verify_retained_failure(retained_failure)
    verified, x0, warm_arrays = verify_inputs(batch_result, source_result, ROOT)
    if (
        runtime() != verified["qualification"]["runtime"]
        or importlib.metadata.version("cma") != protocol()["cma_version"]
    ):
        raise ValueError("frozen optimizer runtime required")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-cmaes-lock-v1",
            "source_revision": revision,
            "source_sha256": {name: file_digest(ROOT / name) for name in SOURCES},
            "protocol": protocol(),
            "verified": verified,
            "retained_failure": retained,
            "assets_root": str(assets.resolve()),
            "output_root": str(output.resolve()),
            "protected_data_read": False,
        },
    )
    try:
        sys.path.insert(0, str(assets / "upstream/experiments"))
        import cma
        from trajopt.cmaes import project_deltas

        bound = np.tile([0.1, 0.1, 0.1, 1, 1, 1], 3)
        es = cma.CMAEvolutionStrategy(
            x0.ravel(),
            0.02,
            {
                "seed": 260829,
                "popsize": 16,
                "verbose": -9,
                "verb_log": 0,
                "bounds": [-bound, bound],
            },
        )
        best_arrays = warm_arrays
        best: dict[str, Any] = {
            "source": "warm_start",
            "index": -1,
            **task_metrics(warm_arrays),
        }
        generation_ids, evaluations = [], []
        for generation in range(4):
            proposals = np.asarray(es.ask(), dtype=np.float64)
            controls = np.stack(
                [
                    project_deltas(
                        x.copy().reshape(3, 6),
                        np.asarray([0.1, 0.1, 0.1, 1, 1, 1]),
                        np.ones(3),
                        0.1,
                    )
                    for x in proposals
                ]
            )
            fitness, seals = [], []
            print(f"CMA-ES source generation {generation + 1}/4", flush=True)
            for chunk in range(2):
                index = 2 * generation + chunk
                arrays, seal = execute_plan(
                    output, lock, index, controls[8 * chunk : 8 * chunk + 8]
                )
                seals.append(seal["artifact_id"])
                for local, row in enumerate(split_batch(arrays, 8)):
                    metrics = task_metrics(row)
                    if (
                        metrics["native_reward"]
                        != seal["native"]["native_cumulative_reward"][local]
                    ):
                        raise ValueError("optimizer native reward does not reproduce")
                    fitness.append(-metrics["native_reward"])
                    item = {
                        "source": "optimizer",
                        "index": 8 * index + local,
                        "batch_index": index,
                        "local_index": local,
                        **metrics,
                    }
                    evaluations.append(item)
                    if metrics["native_reward"] > best["native_reward"]:
                        best, best_arrays = item, row
            es.tell(list(proposals), fitness)
            saved = write_record(
                output / f"generation-{generation}.json",
                {
                    "schema": "dlolab-slingshot-cmaes-generation-v1",
                    "lock_id": lock["artifact_id"],
                    "index": generation,
                    "proposals": proposals.tolist(),
                    "controls": controls.tolist(),
                    "fitness": fitness,
                    "batch_seals": seals,
                    "optimizer_mean": es.mean.tolist(),
                    "optimizer_sigma": float(es.sigma),
                },
            )
            generation_ids.append(saved["artifact_id"])
        selection = write_record(
            output / "selection.json",
            {
                "schema": "dlolab-slingshot-cmaes-selection-v1",
                "lock_id": lock["artifact_id"],
                "generation_ids": generation_ids,
                "evaluations": evaluations,
                "best": best,
                "selected_controls_sha256": array_digest(best_arrays["controls"]),
            },
        )
        replay, replay_seal = execute_plan(
            output, lock, -1, best_arrays["controls"], selection["artifact_id"]
        )
        result = write_record(
            output / "result.json",
            {
                "schema": "dlolab-slingshot-cmaes-result-v1",
                "lock_id": lock["artifact_id"],
                "selection_id": selection["artifact_id"],
                "replay_seal_id": replay_seal["artifact_id"],
                "evaluated_candidates": len(evaluations),
                "best": best,
                **final_checks(best_arrays, replay, verified["zero_reward"]),
            },
        )
        print(
            f"controller competence={result['controller_competence_passed']}; reward={result['best']['native_reward']}; id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-cmaes-failure-v1",
                "lock_id": lock["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--batch-result", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--retained-failure", type=Path)
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args.output, args.worker_index)
    elif any(
        value is None
        for value in (
            args.assets,
            args.batch_result,
            args.source_result,
            args.retained_failure,
        )
    ):
        parser.error("all frozen source input paths are required")
    else:
        run(
            args.output,
            args.assets,
            args.batch_result,
            args.source_result,
            args.retained_failure,
        )
