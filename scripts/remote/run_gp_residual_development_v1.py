#!/usr/bin/env python3
"""Nested residual-development comparison on existing DLO2 fit/validation only.

Never trains or modifies DEFORM. Never reads the eight source-test recordings,
any official-evaluation payload, or any other DLO. This is retrospective
method development, not a fresh confirmation of an accuracy/calibration claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments.gaussian_residual_v1 import (
    GaussianResidual,
    apply_canonical_correction,
    grouped_support,
    require_disjoint_groups,
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_json(path: Path, expected: str) -> dict:
    if digest(path) != expected:
        raise ValueError(f"input digest mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source_manifest(manifest: dict) -> tuple[list[str], list[str]]:
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO2"
        or manifest.get("partition") != "train"
        or manifest.get("official_eval_read") is not False
    ):
        raise ValueError("only the existing DLO2 training manifest is allowed")
    fit = list(manifest["split"]["fit"])
    validation = list(manifest["split"]["validation"])
    source_test = list(manifest["split"]["source_test"])
    require_disjoint_groups(fit, validation, source_test)
    if (len(fit), len(validation), len(source_test)) != (40, 8, 8):
        raise ValueError("unexpected historical split counts")
    return fit, validation


def install_payload_guard(allowed_paths: set[Path]) -> None:
    """Fail closed on non-allowlisted pickle payloads and eval-directory opens."""
    allowed = {str(path.resolve()) for path in allowed_paths}

    def audit(event: str, arguments: tuple) -> None:
        if (
            event != "open"
            or not arguments
            or not isinstance(arguments[0], (str, bytes))
        ):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if "eval" in path.parts or (path.suffix == ".pkl" and str(path) not in allowed):
            raise PermissionError(f"non-development payload denied: {path}")

    sys.addaudithook(audit)


def load_source(protocol: dict, output: Path, device: str) -> tuple:
    import run_deform_dlo2_local_residual as runtime
    import run_deform_dlo_source as source_runtime

    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        deform_causal_inputs,
    )

    inputs = protocol["inputs"]
    training_root = Path(inputs["training_root"])
    candidates = [
        training_root / "training_run" / "training_validation_result.json",
        training_root / "training_validation_result.json",
    ]
    training_path = next((path for path in candidates if path.is_file()), None)
    if training_path is None:
        raise FileNotFoundError(
            f"retained training result not found under {training_root}"
        )
    training = checked_json(training_path, inputs["training_result_sha256"])
    if (
        training.get("source_test_opened") is not False
        or training.get("official_eval_read") is not False
    ):
        raise ValueError("retained checkpoint training crossed a held-out boundary")
    manifest_path = Path(training["source_manifest"]["path"])
    manifest = checked_json(manifest_path, inputs["source_manifest_sha256"])
    fit_names, validation_names = validate_source_manifest(manifest)
    checkpoint_path = Path(training["selected_checkpoint"]["checkpoint"]["path"])
    if digest(checkpoint_path) != inputs["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint identity mismatch")
    upstream = Path(inputs["upstream_root"])
    source_runtime._assert_upstream(upstream, inputs["upstream_commit"])
    # Resolve exact manifest names, without enumerating the dataset tree.
    allowed_names = fit_names + validation_names
    data_directory = (upstream / "data_set" / "DLO2" / "train").resolve()
    allowed_paths = {(data_directory / name).resolve() for name in allowed_names}
    for name in allowed_names:
        path = (data_directory / name).resolve()
        if path.parent != data_directory or path.name != name:
            raise ValueError("invalid development recording path")
    install_payload_guard(allowed_paths)
    write_json(
        output / "preflight.json",
        {
            "protocol": protocol,
            "fit_names": fit_names,
            "validation_names": validation_names,
            "training_result_sha256": digest(training_path),
            "source_manifest_sha256": digest(manifest_path),
            "checkpoint_sha256": digest(checkpoint_path),
            "source_test_read": False,
            "official_eval_read": False,
            "upstream_training": False,
            "claim_boundary": "retrospective residual development only",
        },
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(upstream)
    state, _ = runtime._checkpoint_state(training, torch=torch)
    trajectories = source_runtime._load_named_trajectories(
        manifest, allowed_names, frame_count=500, node_count=12
    )
    trajectories = {name: trajectories[name] for name in allowed_names}
    print("SOURCE: frozen-checkpoint rollout on 48 development recordings", flush=True)
    rollout = runtime._rollout(
        state, trajectories, modules=modules, torch=torch, device=device
    )
    full = np.stack([trajectories[name] for name in allowed_names])
    initial, action = deform_causal_inputs(full)
    baseline = np.asarray(rollout["predictions"], dtype=np.float64)
    targets = np.asarray(rollout["targets"], dtype=np.float64)
    expected = float(training["selected_checkpoint"]["validation_l1_m"])
    observed = float(np.mean(np.abs(baseline[40:] - targets[40:])))
    if not np.isclose(observed, expected, rtol=0, atol=1e-7):
        raise ValueError(
            f"frozen validation baseline mismatch: {observed} vs {expected}"
        )
    np.savez_compressed(
        output / "development_rollouts.npz",
        initial=initial,
        action=action,
        baseline=baseline,
        targets=targets,
        names=np.asarray(allowed_names),
    )
    return initial, action, baseline, targets, allowed_names


def build_splits(
    names: list[str], seed: int, budget: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(names) != 48 or budget not in (8, 16, 32):
        raise ValueError("invalid residual-development roster or budget")
    order = sorted(
        range(40),
        key=lambda i: hashlib.sha256(f"gp-v1:{seed}:{names[i]}".encode()).hexdigest(),
    )
    train = np.asarray(order[8 : 8 + budget])
    inner = np.asarray(order[:8])
    outer = np.arange(40, 48)
    require_disjoint_groups(
        *[[names[index] for index in split] for split in (train, inner, outer)]
    )
    return train, inner, outer


def select_strength(predictions: list[np.ndarray], truth: np.ndarray) -> int:
    """Only inner-validation outcomes may be supplied here."""
    if not predictions or not np.isfinite(truth).all():
        raise ValueError("invalid selection inputs")
    scores = []
    for value in predictions:
        if value.shape != truth.shape or not np.isfinite(value).all():
            raise ValueError("invalid candidate prediction")
        scores.append(float(np.mean(np.abs(value - truth))))
    return int(np.argmin(scores))


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error = np.abs(prediction - truth)
    return {
        "coordinate_l1_mm": float(error.mean() * 1000),
        "free_node_coordinate_l1_mm": float(error[:, :, 2:-2].mean() * 1000),
        "per_recording_l1_mm": (error.mean(axis=(1, 2, 3)) * 1000).tolist(),
        "horizon_quartile_l1_mm": [
            float(block.mean() * 1000) for block in np.array_split(error, 4, axis=1)
        ],
    }


def mlp_correction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    hidden: int,
    weight_decay: float,
    seed: int,
    per_group: int,
) -> np.ndarray:
    """Independent-node MLPs with matched GP support rows and fit inputs."""
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    count, horizon, nodes, feature_count = train_x.shape
    rows = grouped_support(np.repeat(np.arange(count), horizon), per_group)
    location = train_x.mean(axis=(0, 1))
    scale = train_x.std(axis=(0, 1))
    scale = np.where(scale > 1e-10, scale, 1.0)
    output_scale = np.sqrt(np.mean(train_y**2, axis=(0, 1)))
    output_scale = np.where(output_scale > 1e-10, output_scale, 1.0)
    x = ((train_x - location) / scale).reshape(-1, nodes, feature_count)[rows]
    y = (train_y / output_scale).reshape(-1, nodes, 3)[rows]
    query = ((query_x - location) / scale).reshape(-1, nodes, feature_count)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w1 = torch.nn.Parameter(
        torch.randn(nodes, feature_count, hidden) / np.sqrt(feature_count)
    )
    b1 = torch.nn.Parameter(torch.zeros(nodes, hidden))
    w2 = torch.nn.Parameter(torch.randn(nodes, hidden, 3) / np.sqrt(hidden))
    b2 = torch.nn.Parameter(torch.zeros(nodes, 3))
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=0.003, weight_decay=weight_decay)

    def forward(value):
        hidden_values = torch.tanh(torch.einsum("bnf,nfh->bnh", value, w1) + b1)
        return torch.einsum("bnh,nhc->bnc", hidden_values, w2) + b2

    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.mean((forward(x_t) - y_t) ** 2)
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite MLP training loss")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        values = [
            forward(torch.tensor(part, dtype=torch.float32)).numpy()
            for part in np.array_split(query, max(1, len(query) // 1024))
        ]
    return np.concatenate(values).reshape(*query_x.shape[:3], 3) * output_scale


def compare(protocol: dict, arrays: tuple, output: Path) -> dict:
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    initial, action, baseline, targets, names = arrays
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    residual = np.einsum("ntvi,nij->ntvj", targets - baseline, frames)[:, :, 2:-2]
    identities = [
        hashlib.sha256(
            initial[i].tobytes() + action[i].tobytes() + baseline[i].tobytes()
        ).hexdigest()
        for i in range(len(names))
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate causal-query recordings need protocol amendment")
    anchor_model = fit_deform_local_residual(
        initial[:40],
        action[:40],
        baseline[:40],
        targets[:40],
        names[:40],
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    anchor = predict_deform_local_residual(
        anchor_model,
        initial[40:],
        action[40:],
        baseline[40:],
        shrinkage=0.25,
    )["predictions"]
    experiments = []
    for seed in protocol["seeds"]:
        for budget in protocol["budgets"]:
            train, inner, outer = build_splits(names, seed, budget)
            query = np.concatenate((inner, outer))
            subdir = output / f"seed-{seed}-budget-{budget}"
            subdir.mkdir()
            print(f"FIT: seed={seed} budget={budget}", flush=True)
            ridge = fit_deform_local_residual(
                initial[train],
                action[train],
                baseline[train],
                targets[train],
                [names[i] for i in train],
                ridge=1.0,
                variance_floor_m2=1e-6,
            )
            candidates: dict[str, list[tuple[dict, np.ndarray]]] = {
                "ridge": [],
                "gp": [],
                "mlp": [],
            }
            for strength in protocol["strengths"]:
                pred = predict_deform_local_residual(
                    ridge,
                    initial[query],
                    action[query],
                    baseline[query],
                    shrinkage=strength,
                )["predictions"]
                candidates["ridge"].append(({"strength": strength}, pred))
            fixed_ridge = predict_deform_local_residual(
                ridge,
                initial[query],
                action[query],
                baseline[query],
                shrinkage=0.25,
            )["predictions"]
            groups = np.repeat(np.asarray([names[i] for i in train]), features.shape[1])
            for length in protocol["length_scales"]:
                for noise in protocol["noise_variances"]:
                    corrections = []
                    for node in range(features.shape[2]):
                        model = GaussianResidual.fit(
                            features[train, :, node].reshape(-1, features.shape[-1]),
                            residual[train, :, node].reshape(-1, 3),
                            groups,
                            length_scale=length,
                            noise_variance=noise,
                            per_group=protocol["support_per_recording"],
                        )
                        value = model.predict_mean(
                            features[query, :, node].reshape(-1, features.shape[-1])
                        )
                        corrections.append(
                            value.reshape(len(query), features.shape[1], 3)
                        )
                    correction = np.stack(corrections, axis=2)
                    for strength in protocol["strengths"]:
                        pred = apply_canonical_correction(
                            baseline[query],
                            correction,
                            frames[query],
                            strength,
                        )
                        candidates["gp"].append(
                            (
                                {
                                    "length_scale": length,
                                    "noise_variance": noise,
                                    "strength": strength,
                                },
                                pred,
                            )
                        )
            for width in protocol["mlp_widths"]:
                correction = mlp_correction(
                    features[train],
                    residual[train],
                    features[query],
                    hidden=width,
                    weight_decay=0.001,
                    seed=seed,
                    per_group=protocol["support_per_recording"],
                )
                for strength in protocol["strengths"]:
                    pred = apply_canonical_correction(
                        baseline[query],
                        correction,
                        frames[query],
                        strength,
                    )
                    candidates["mlp"].append(
                        (
                            {
                                "hidden": width,
                                "strength": strength,
                            },
                            pred,
                        )
                    )
            selection = {}
            selected_predictions = {
                "backbone": baseline[outer],
                "ridge_fixed_0p25": fixed_ridge[8:],
                "ridge_full40_anchor": anchor,
            }
            for family, bank in candidates.items():
                index = select_strength(
                    [value[:8] for _, value in bank], targets[inner]
                )
                parameters, prediction = bank[index]
                selection[family] = {
                    "parameters": parameters,
                    "inner_l1_mm": float(
                        np.mean(np.abs(prediction[:8] - targets[inner])) * 1000
                    ),
                    "all_inner_candidates": [
                        {
                            **spec,
                            "l1_mm": float(
                                np.mean(np.abs(value[:8] - targets[inner])) * 1000
                            ),
                        }
                        for spec, value in bank
                    ],
                }
                selected_predictions[family] = prediction[8:]
            # Freeze selected forecasts before scoring outer-validation outcomes.
            prediction_path = subdir / "selected_predictions.npz"
            np.savez_compressed(prediction_path, **selected_predictions)
            write_json(
                subdir / "selection.json",
                {
                    "train": [names[i] for i in train],
                    "inner": [names[i] for i in inner],
                    "outer": [names[i] for i in outer],
                    "selection": selection,
                    "prediction_sha256": digest(prediction_path),
                    "outer_used_for_selection": False,
                },
            )
            scores = {
                family: metrics(value, targets[outer])
                for family, value in selected_predictions.items()
            }
            base_errors = np.asarray(scores["ridge_fixed_0p25"]["per_recording_l1_mm"])
            for value in scores.values():
                errors = np.asarray(value["per_recording_l1_mm"])
                value["wins_against_fixed_ridge"] = int(np.sum(errors < base_errors))
                value["relative_improvement_against_fixed_ridge"] = float(
                    1 - errors.mean() / base_errors.mean()
                )
            record = {
                "seed": seed,
                "budget": budget,
                "scores": scores,
                "selection": selection,
                "outer_names": [names[i] for i in outer],
                "prediction_sha256": digest(prediction_path),
            }
            write_json(subdir / "result.json", record)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "budget": budget,
                        "l1_mm": {
                            key: value["coordinate_l1_mm"]
                            for key, value in scores.items()
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            experiments.append(record)
    return {
        "contract": "gp-residual-development-v1",
        "experiments": experiments,
        "source_test_read": False,
        "official_eval_read": False,
        "upstream_training": False,
        "fresh_confirmation": False,
        "paper_claim_authorized": False,
        "interpretation": (
            "Retrospective residual adaptation on a previously selected checkpoint "
            "and previously used validation recordings. GP/MLP use matched "
            "group-balanced support; ridge uses all permitted fit rows. "
            "No calibrated-uncertainty claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if protocol.get("contract") != "gp-residual-development-v1":
            raise ValueError("invalid development protocol")
        arrays = load_source(protocol, output, args.device)
        result = compare(protocol, arrays, output)
        result.update(
            {
                "protocol_sha256": digest(args.protocol),
                "source_sha": os.environ.get("GITHUB_SHA"),
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(output / "result.json", result)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "paper_claim_authorized": False,
                "source_sha": os.environ.get("GITHUB_SHA"),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
