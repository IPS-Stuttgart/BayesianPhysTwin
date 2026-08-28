#!/usr/bin/env python3
"""Write-once native Slingshot belief/control experiment using public assets."""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration
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
from bayesian_phystwin_experiments.dlolab_slingshot_batch import TRACE_NAMES
from bayesian_phystwin_experiments.dlolab_slingshot_belief import (
    ARMS,
    BASELINE,
    COUNTS,
    MODES,
    POSITION_ENVELOPE_M,
    calibrate,
    commands_for_decisions,
    controls,
    decide,
    infer,
    native_qa,
    particle_worlds,
    prefix_observations,
    protocol,
    sample_worlds,
    score,
    sensor_errors,
)
from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    run_registered_worlds,
)
from bayesian_phystwin_experiments.dlolab_slingshot_cmaes import worker_environment
from bayesian_phystwin_experiments.dlolab_slingshot_process import (
    load_native_bundle,
    runtime,
)
from bayesian_phystwin_experiments.dlolab_slingshot_value import (
    decision_value,
    verify_source,
    world_metrics,
    worlds,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    "/home/fpfaff/source-only/dlolab-benchmark-source-v1/belief-control-source-v1"
)
SCREEN_SHA256 = "e4097d1be73321573ef3dd1ecb309e9d77101207cbc0ffbdc90c8eed3b5d165b"
SOURCES = (
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_belief_native.py",
    "scripts/remote/run_dlolab_slingshot_belief.py",
    "tests/test_dlolab_slingshot_belief.py",
    "tests/test_dlolab_slingshot_belief_custody.py",
    "docs/dlolab_slingshot_belief_control_source_v1.md",
    "src/bayesian_phystwin_experiments/coupled_action_regret.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_batch.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_value.py",
    "src/bayesian_phystwin_experiments/dlolab_slingshot_process.py",
    "src/bayesian_phystwin_experiments/dlolab_benchmark.py",
    "src/bayesian_phystwin_experiments/dlolab_regret_artifacts.py",
    "src/bayesian_phystwin/guard_harm_risk.py",
)
POSITION_FIELDS = ("rod_pos_m", "sphere_pos_m", "cube_pos_m", "gripper_pos_m")


def key(world: dict) -> tuple:
    return tuple(world[name] for name in ("x_offset_m", "bending_E", "stretching_K"))


def verify_screen(path: Path, root: Path):
    if file_digest(path) != SCREEN_SHA256:
        raise ValueError("exact passing source screen required")
    result = read_record(path)
    lock = read_record(path.parent / "lock.json")
    if result["lock_id"] != lock["artifact_id"]:
        raise ValueError("source screen lock changed")
    for name, sha in lock["source_sha256"].items():
        if file_digest(root / name) != sha:
            raise ValueError("source screen implementation changed")
    verified, reference = verify_source(
        Path(lock["source"]["controller"]["path"]),
        Path(lock["source"]["mechanism"]["path"]),
        root,
    )
    if verified != lock["source"]:
        raise ValueError("source lineage changed")
    generation = read_record(path.parent / "generation.json")
    if (
        generation["artifact_id"] != result["generation_id"]
        or generation["lock_id"] != lock["artifact_id"]
    ):
        raise ValueError("source generation changed")
    rows, arrays = [], []
    for i, world in enumerate(worlds()):
        directory = path.parent / f"world-{i:02d}"
        seal, claim = (
            read_record(directory / "seal.json"),
            read_record(directory / "claim.json"),
        )
        if (
            seal["artifact_id"] != generation["world_seals"][i]
            or seal["claim_id"] != claim["artifact_id"]
            or seal["world"] != world
            or claim["world"] != world
            or seal["lock_id"] != lock["artifact_id"]
        ):
            raise ValueError("source native seal changed")
        data = load_native_bundle(directory, seal["bundle"])
        computed = world_metrics(data, seal["native"], world, reference)
        if result["worlds"][i] != computed:
            raise ValueError("source world arithmetic changed")
        rows.append(computed)
        arrays.append(data)
    computed = decision_value(rows)
    if not computed["source_decision_value_passed"] or any(
        result[k] != v for k, v in computed.items()
    ):
        raise ValueError("source decision-value gate not reproduced")
    return (
        {
            "path": str(path.resolve()),
            "sha256": SCREEN_SHA256,
            "artifact_id": result["artifact_id"],
            "source": verified,
        },
        reference,
        arrays,
    )


def task(kind: str, index: int) -> dict[str, Any]:
    if type(index) is not int or index < 0:
        raise ValueError("invalid task index")
    if kind == "qualification" and index == 0:
        return {
            "name": "prefix-qualification",
            "kind": kind,
            "index": index,
            "worlds": worlds()[:8],
            "prefix_only": True,
            "case_indices": list(range(8)),
        }
    if kind == "particle" and index < 27:
        world = particle_worlds()[index]
        if key(world) in {key(w) for w in worlds()}:
            raise ValueError(
                "already opened particles must be reused without rerunning"
            )
        return {
            "name": f"particle-{index:02d}",
            "kind": kind,
            "index": index,
            "worlds": [world] * 8,
            "prefix_only": False,
            "case_indices": [index],
        }
    for role in COUNTS:
        roster = sample_worlds(role)
        if kind == f"{role}-prefix" and index < (len(roster) + 7) // 8:
            indices = list(range(8 * index, min(8 * index + 8, len(roster))))
            selected = [roster[i] for i in indices]
            padded = selected + [selected[-1]] * (8 - len(selected))
            return {
                "name": f"{kind}-{index:02d}",
                "kind": kind,
                "index": index,
                "worlds": padded,
                "prefix_only": True,
                "case_indices": indices,
            }
        if kind == f"{role}-future" and index < len(roster):
            return {
                "name": f"{kind}-{index:02d}",
                "kind": kind,
                "index": index,
                "worlds": [roster[index]] * 8,
                "prefix_only": False,
                "case_indices": [index],
            }
    raise ValueError("unregistered native task")


def validate_lock(output: Path):
    if output.resolve() != OUTPUT_ROOT:
        raise ValueError("only the registered write-once root is authorized")
    lock = read_record(output / "lock.json")
    if (
        lock["source_revision"] != clean_revision(ROOT)
        or lock["source_sha256"] != {p: file_digest(ROOT / p) for p in SOURCES}
        or lock["protocol"] != protocol()
        or lock["output_root"] != str(output.resolve())
    ):
        raise ValueError("frozen source/lock changed")
    verified, reference, opened = verify_screen(Path(lock["screen"]["path"]), ROOT)
    if (
        verified != lock["screen"]
        or runtime() != verified["source"]["controller"]["runtime"]
    ):
        raise ValueError("frozen screen/runtime changed")
    if array_digest(np.asarray(lock["controls"], dtype=np.float64)) != array_digest(
        controls(reference["controls"])
    ):
        raise ValueError("source-selected control bank changed")
    assets = Path(lock["assets_root"])
    if (
        source_identity(
            assets / "upstream", assets / "mushroom-rl", assets / "dlo-lab.zip"
        )
        != verified["source"]["controller"]["native_source"]
    ):
        raise ValueError("public native assets changed")
    return lock, reference, opened


def load_task(output: Path, lock: dict, spec: dict):
    directory = output / spec["name"]
    seal, claim = (
        read_record(directory / "seal.json"),
        read_record(directory / "claim.json"),
    )
    if (
        claim["task"] != spec
        or seal["task"] != spec
        or claim["lock_id"] != lock["artifact_id"]
        or seal["lock_id"] != lock["artifact_id"]
        or seal["claim_id"] != claim["artifact_id"]
    ):
        raise ValueError("native task/claim binding changed")
    arrays = load_native_bundle(directory, seal["bundle"])
    expected_controls = np.asarray(lock["controls"], dtype=np.float64)
    if spec["prefix_only"]:
        expected_controls = np.repeat(
            expected_controls[BASELINE : BASELINE + 1], 8, axis=0
        )
    if array_digest(arrays["controls"]) != array_digest(expected_controls):
        raise ValueError("registered task controls changed")
    native = seal["native"]
    realization = native["world_realization"]
    expected_e = [[w["bending_E"] for w in spec["worlds"]]]
    expected_k = [[w["stretching_K"] for w in spec["worlds"]]]
    if realization["bending"] != expected_e or realization["stretching"] != expected_k:
        raise ValueError("realized material parameters changed")
    for name, y, z in (("sphere", 0.06, 0.2), ("cube", 0.23, 0.22)):
        expected = np.asarray([[0.12 + w["x_offset_m"], y, z] for w in spec["worlds"]])
        if not np.allclose(
            realization[f"{name}_initial_position_m"], expected, rtol=0, atol=1e-15
        ):
            raise ValueError("realized placement changed")
    if spec["prefix_only"]:
        if (
            native["native_steps"] != 300
            or native["future_simulated"] is not False
            or native["reward_scored"] is not False
        ):
            raise ValueError("prefix execution crossed information boundary")
        if set(arrays) != set(TRACE_NAMES + ("controls",)) or any(
            arrays[name].shape[:2] != (300, 8) for name in TRACE_NAMES
        ):
            raise ValueError("prefix artifact contains undeclared or future arrays")
        prefix_observations(arrays)
    elif native["native_steps"] != 900:
        raise ValueError("incomplete native future")
    return seal, arrays


def qualification(output: Path, lock: dict, opened: list[dict]):
    seal, values = load_task(output, lock, task("qualification", 0))
    errors = [
        max(
            float(np.max(np.abs(values[name][:, i] - opened[i][name][:300, 0])))
            for name in POSITION_FIELDS
        )
        for i in range(8)
    ]
    return {
        "schema": "dlolab-slingshot-belief-prefix-qualification-v1",
        "lock_id": lock["artifact_id"],
        "seal_id": seal["artifact_id"],
        "position_errors_m": errors,
        "passed": max(errors) <= POSITION_ENVELOPE_M,
        "future_simulated": False,
    }


def require_qualification(output: Path, lock: dict, opened: list[dict]) -> None:
    stored = read_record(output / "qualification.json")
    computed = qualification(output, lock, opened)
    if not computed["passed"] or any(stored[k] != v for k, v in computed.items()):
        raise ValueError("prefix qualification gate failed")


def load_bank(output: Path, lock: dict):
    seal = read_record(output / "model-bank/seal.json")
    if (
        seal["lock_id"] != lock["artifact_id"]
        or seal["worlds"] != particle_worlds()
        or len(seal["parents"]) != 27
    ):
        raise ValueError("model-bank custody changed")
    arrays = load_native_bundle(output / "model-bank", seal["bundle"])
    if arrays["prefix"].shape != (27, 3, 4, 3) or arrays["reward"].shape != (27, 7):
        raise ValueError("model bank incomplete")
    return seal, arrays


def load_calibrator(output: Path, lock: dict):
    record = read_record(output / "calibrator.json")
    if (
        record["lock_id"] != lock["artifact_id"]
        or record["count"] != 19
        or record["evaluation_futures_read"] is not False
    ):
        raise ValueError("calibrator boundary changed")
    values = {k: RegretCalibration(**record["calibrations"][k]) for k in MODES}
    rewards, parents, qas = future_table(output, lock, "calibration")
    parts = [load_prediction(output, lock, "calibration", i)[1] for i in range(19)]
    recomputed = calibrate(parts, rewards)
    if (
        values != recomputed
        or record["future_seals"] != parents
        or record["native_qa"] != qas
        or not all(q["qa_passed"] for q in qas)
    ):
        raise ValueError("calibration arithmetic or native gate changed")
    return record, values


def prediction(output: Path, lock: dict, role: str, index: int):
    if role not in COUNTS or index not in range(COUNTS[role]):
        raise ValueError("unregistered prediction")
    spec = task(f"{role}-prefix", index // 8)
    prefix_seal, prefix = load_task(output, lock, spec)
    bank_seal, bank = load_bank(output, lock)
    observed = prefix_observations(prefix)[index % 8] + sensor_errors(role)[index]
    parts = infer(observed, bank["prefix"], bank["reward"])
    arrays = {"observation": observed, **parts}
    metadata: dict[str, Any] = {
        "schema": "dlolab-slingshot-belief-prefix-prediction-v1",
        "lock_id": lock["artifact_id"],
        "role": role,
        "index": index,
        "prefix_seal_id": prefix_seal["artifact_id"],
        "bank_seal_id": bank_seal["artifact_id"],
        "future_simulated": False,
        "future_read": False,
        "calibrator_id": None,
    }
    if role == "evaluation":
        record, calibrations = load_calibrator(output, lock)
        decision = decide(parts, calibrations)
        arrays["decisions"] = decision
        bank_controls = np.asarray(lock["controls"], dtype=np.float64)
        commands = commands_for_decisions(bank_controls, decision)
        arrays["selected_commands"] = np.concatenate(commands, axis=0)
        metadata["calibrator_id"] = record["artifact_id"]
        metadata["command_sha256"] = [array_digest(c) for c in commands]
        metadata["arms"] = list(ARMS)
    return metadata, arrays


def load_prediction(output: Path, lock: dict, role: str, index: int):
    directory = output / f"{role}-predictions" / f"case-{index:02d}"
    record = read_record(directory / "seal.json")
    metadata, expected = prediction(output, lock, role, index)
    if any(record[k] != v for k, v in metadata.items()):
        raise ValueError("prediction source binding changed")
    arrays = load_native_bundle(directory, record["bundle"])
    if set(arrays) != set(expected) or any(
        array_digest(arrays[k]) != array_digest(v) for k, v in expected.items()
    ):
        raise ValueError("sealed prefix prediction does not reproduce")
    return record, arrays


def barrier_contents(output: Path, lock: dict, role: str) -> dict[str, Any]:
    records = [load_prediction(output, lock, role, i)[0] for i in range(COUNTS[role])]
    return {
        "schema": "dlolab-slingshot-belief-prediction-barrier-v1",
        "lock_id": lock["artifact_id"],
        "role": role,
        "count": COUNTS[role],
        "prediction_seals": [r["artifact_id"] for r in records],
        "future_simulated": False,
        "future_read": False,
    }


def require_barrier(output: Path, lock: dict, role: str) -> dict[str, Any]:
    recorded = read_record(output / f"{role}-prediction-barrier.json")
    expected = barrier_contents(output, lock, role)
    if any(recorded[k] != value for k, value in expected.items()):
        raise ValueError("complete prediction barrier not reproduced")
    return recorded


def authorize_task(
    output: Path, lock: dict, spec: dict, opened: list[dict]
) -> dict[str, Any]:
    if spec["kind"] == "qualification":
        return {"gate": "registered_source_prefix_qualification"}
    require_qualification(output, lock, opened)
    if spec["kind"] == "particle":
        return {"gate": "qualified_model_bank"}
    role = spec["kind"].split("-")[0]
    bank_seal, _ = load_bank(output, lock)
    binding = {"gate": "permitted_prefix", "bank_seal_id": bank_seal["artifact_id"]}
    if role == "evaluation":
        calibrator, _ = load_calibrator(output, lock)
        binding["calibrator_id"] = calibrator["artifact_id"]
    if not spec["prefix_only"]:
        barrier = require_barrier(output, lock, role)
        binding["gate"] = "all_prefix_predictions_sealed"
        binding["barrier_id"] = barrier["artifact_id"]
    return binding


def worker(output: Path, kind: str, index: int) -> None:
    lock, _, opened = validate_lock(output)
    spec = task(kind, index)
    authorization = authorize_task(output, lock, spec, opened)
    directory = output / spec["name"]
    directory.mkdir(exist_ok=False)
    claim = write_record(
        directory / "claim.json",
        {
            "schema": "dlolab-slingshot-belief-native-claim-v1",
            "lock_id": lock["artifact_id"],
            "task": spec,
            "authorization": authorization,
            "retry_authorized": False,
        },
    )
    try:
        bank = np.asarray(lock["controls"], dtype=np.float64)
        native_controls = (
            np.repeat(bank[BASELINE : BASELINE + 1], 8, axis=0)
            if spec["prefix_only"]
            else bank
        )
        arrays, native = run_registered_worlds(
            Path(lock["assets_root"]) / "upstream",
            directory,
            native_controls,
            spec["worlds"],
            prefix_only=spec["prefix_only"],
        )
        bundle = write_native_bundle(directory, arrays)
        write_record(
            directory / "seal.json",
            {
                "schema": "dlolab-slingshot-belief-native-seal-v1",
                "lock_id": lock["artifact_id"],
                "claim_id": claim["artifact_id"],
                "task": spec,
                "native": native,
                "bundle": bundle,
            },
        )
    except Exception as error:
        write_record(
            directory / "failure.json",
            {
                "schema": "dlolab-slingshot-belief-native-failure-v1",
                "claim_id": claim["artifact_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
            },
        )
        raise


def execute(output: Path, lock: dict, kind: str, index: int) -> None:
    spec = task(kind, index)
    print(f"native belief stage: {spec['name']}", flush=True)
    with (output / f"{spec['name']}.log").open("x") as stream:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output.resolve()),
                "--worker-kind",
                kind,
                "--worker-index",
                str(index),
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=worker_environment(lock["screen"]["source"]["controller"]["runtime"]),
            check=True,
        )


def build_bank(output: Path, lock: dict, opened: list[dict]):
    parents, prefixes, rewards = [], [], []
    old = {key(w): i for i, w in enumerate(worlds())}
    source_result = read_record(Path(lock["screen"]["path"]))
    bank_controls = np.asarray(lock["controls"], dtype=np.float64)
    for i, world in enumerate(particle_worlds()):
        if key(world) in old:
            index = old[key(world)]
            data = opened[index]
            metrics = source_result["worlds"][index]["metrics"]
            parent = {
                "kind": "reused_open_source",
                "world_index": index,
                "screen_id": source_result["artifact_id"],
            }
        else:
            execute(output, lock, "particle", i)
            seal, data = load_task(output, lock, task("particle", i))
            qa = native_qa(data, seal["native"], bank_controls)
            if not qa["qa_passed"]:
                raise ValueError("particle native QA failed")
            metrics = qa["metrics"]
            parent = {
                "kind": "fresh_model_particle",
                "seal_id": seal["artifact_id"],
                "qa": qa,
            }
        prefixes.append(
            prefix_observations(
                {
                    "rod_pos_m": data["rod_pos_m"][:300],
                    "sphere_pos_m": data["sphere_pos_m"][:300],
                }
            )[0]
        )
        rewards.append([m["native_reward"] for m in metrics[:7]])
        parents.append(parent)
    directory = output / "model-bank"
    directory.mkdir(exist_ok=False)
    bundle = write_native_bundle(
        directory,
        {"prefix": np.stack(prefixes), "reward": np.asarray(rewards, dtype=np.float64)},
    )
    return write_record(
        directory / "seal.json",
        {
            "schema": "dlolab-slingshot-belief-bank-v1",
            "lock_id": lock["artifact_id"],
            "worlds": particle_worlds(),
            "parents": parents,
            "bundle": bundle,
        },
    )


def prepare_predictions(output: Path, lock: dict, role: str):
    for group in range((COUNTS[role] + 7) // 8):
        execute(output, lock, f"{role}-prefix", group)
    for index in range(COUNTS[role]):
        metadata, arrays = prediction(output, lock, role, index)
        directory = output / f"{role}-predictions" / f"case-{index:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        bundle = write_native_bundle(directory, arrays)
        write_record(directory / "seal.json", {**metadata, "bundle": bundle})
    return write_record(
        output / f"{role}-prediction-barrier.json", barrier_contents(output, lock, role)
    )


def future_table(output: Path, lock: dict, role: str):
    barrier = require_barrier(output, lock, role)
    rewards, parents, qas = [], [], []
    bank_controls = np.asarray(lock["controls"], dtype=np.float64)
    for index in range(COUNTS[role]):
        spec = task(f"{role}-future", index)
        seal, arrays = load_task(output, lock, spec)
        claim = read_record(output / spec["name"] / "claim.json")
        if claim["authorization"]["barrier_id"] != barrier["artifact_id"]:
            raise ValueError("future was not bound to the complete prediction barrier")
        _, prefix = load_task(output, lock, task(f"{role}-prefix", index // 8))
        row = {name: prefix[name][:, index % 8] for name in POSITION_FIELDS}
        qa = native_qa(arrays, seal["native"], bank_controls, row)
        qas.append(qa)
        rewards.append([m["native_reward"] for m in qa["metrics"][:7]])
        parents.append(seal["artifact_id"])
    return np.asarray(rewards), parents, qas


def run(output: Path, assets: Path, screen: Path):
    if output.resolve() != OUTPUT_ROOT:
        raise ValueError("only the registered write-once output root is authorized")
    revision = clean_revision(ROOT)
    verified, reference, opened = verify_screen(screen, ROOT)
    if runtime() != verified["source"]["controller"]["runtime"]:
        raise ValueError("qualified runtime required")
    output.mkdir(parents=True, exist_ok=False)
    lock = write_record(
        output / "lock.json",
        {
            "schema": "dlolab-slingshot-belief-control-lock-v1",
            "source_revision": revision,
            "source_sha256": {p: file_digest(ROOT / p) for p in SOURCES},
            "protocol": protocol(),
            "screen": verified,
            "controls": controls(reference["controls"]).tolist(),
            "assets_root": str(assets.resolve()),
            "output_root": str(output.resolve()),
            "protected_data_read": False,
        },
    )
    stage = "prefix_qualification"
    try:
        execute(output, lock, "qualification", 0)
        q = write_record(
            output / "qualification.json", qualification(output, lock, opened)
        )
        if not q["passed"]:
            raise ValueError("native mixed-world prefix qualification failed")
        stage = "model_bank"
        build_bank(output, lock, opened)
        stage = "calibration_prefix_predictions"
        prepare_predictions(output, lock, "calibration")
        stage = "calibration_futures"
        for index in range(COUNTS["calibration"]):
            execute(output, lock, "calibration-future", index)
        reward, parents, qas = future_table(output, lock, "calibration")
        if not all(q["qa_passed"] for q in qas):
            raise ValueError("calibration native QA failed")
        parts = [load_prediction(output, lock, "calibration", i)[1] for i in range(19)]
        calibrated = calibrate(parts, reward)
        stage = "calibrator"
        write_record(
            output / "calibrator.json",
            {
                "schema": "dlolab-slingshot-belief-calibrator-v1",
                "lock_id": lock["artifact_id"],
                "count": 19,
                "calibrations": {
                    k: dataclasses.asdict(v) for k, v in calibrated.items()
                },
                "future_seals": parents,
                "native_qa": qas,
                "evaluation_futures_read": False,
            },
        )
        stage = "evaluation_prefix_decisions"
        prepare_predictions(output, lock, "evaluation")
        stage = "evaluation_futures"
        for index in range(COUNTS["evaluation"]):
            execute(output, lock, "evaluation-future", index)
        reward, parents, qas = future_table(output, lock, "evaluation")
        generation = write_record(
            output / "evaluation-generation.json",
            {
                "schema": "dlolab-slingshot-belief-generation-v1",
                "lock_id": lock["artifact_id"],
                "count": 32,
                "ordinary_native_worlds": 32,
                "technical_failures": 0,
                "future_seals": parents,
            },
        )
        stage = "score"
        parts = [load_prediction(output, lock, "evaluation", i)[1] for i in range(32)]
        decisions = np.stack([p["decisions"] for p in parts])
        result = write_record(
            output / "result.json",
            {
                **score(
                    decisions,
                    parts,
                    reward,
                    calibrated,
                    all_native_qa=all(q["qa_passed"] for q in qas),
                ),
                "lock_id": lock["artifact_id"],
                "generation_id": generation["artifact_id"],
                "native_qa": qas,
            },
        )
        print(
            f"belief source gate={result['source_gate_passed']}; id={result['artifact_id']}",
            flush=True,
        )
        return result
    except Exception as error:
        counts = {
            role: sum(
                (output / f"{role}-future-{i:02d}/seal.json").is_file()
                for i in range(COUNTS[role])
            )
            for role in COUNTS
        }
        write_record(
            output / "failure.json",
            {
                "schema": "dlolab-slingshot-belief-failure-v1",
                "lock_id": lock["artifact_id"],
                "terminal_stage": stage,
                "completed_native_future_worlds": counts,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--worker-kind")
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.worker_kind is not None and args.worker_index is not None:
        worker(args.output, args.worker_kind, args.worker_index)
    elif args.worker_kind is not None or args.worker_index is not None:
        parser.error("both registered worker arguments are required")
    elif args.assets is None or args.screen is None:
        parser.error("assets and the passing source screen are required")
    else:
        run(args.output, args.assets, args.screen)
