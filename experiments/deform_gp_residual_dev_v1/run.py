"""Source-only nonlinear GP residual pilot; no official or source-test reads.

The GP uses a finite-rank Nystrom Matern-3/2 kernel and all fitting rows.
Its mean equals kernel ridge regression. Raw variances are diagnostic only.
Preparation, prediction, and scoring are separate processes.
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
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    _collapse_duplicate_queries,
    build_deform_local_residual_features,
    fit_deform_local_residual,
    predict_deform_local_residual,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path(
    "/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/"
    "train-8cc85de7/training_run"
)
UPSTREAM = Path("/home/florianpfaff/source-only/DEFORM-b73b8b8")
LENGTHS = (0.5, 1.5)
NOISES = (0.05, 0.5)
SHRINKAGES = (0.25, 0.5, 1.0)
RANK_PER_NODE = 64
BATCH = 2048


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def kernel(x, z, length, material=False):
    """Positive-semidefinite product of Euclidean and material Matern kernels."""
    if not np.isfinite(length) or length <= 0:
        raise ValueError("kernel length must be positive")
    a = x[:, :-1] if material else x
    b = z[:, :-1] if material else z
    d2 = np.maximum(
        np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T,
        0.0,
    ) / max(1, a.shape[1])
    distance = np.sqrt(3 * d2) / length
    result = (1 + distance) * np.exp(-distance)
    if material:
        spatial = np.sqrt(3.0) * np.abs(x[:, -1, None] - z[None, :, -1]) / 0.5
        result *= (1 + spatial) * np.exp(-spatial)
    return result


def normalize_inputs(x, material):
    # Arc and squared arc occupy columns 4/5 in the existing feature contract.
    keep = np.asarray([i for i in range(x.shape[1]) if i not in (4, 5)])
    location = x[:, keep].mean(axis=0)
    scale = x[:, keep].std(axis=0)
    scale[scale < 1e-10] = 1.0
    state = (x[:, keep] - location) / scale
    if material:
        state = np.column_stack((state, x[:, 4]))
    return state, keep, location, scale


def transform(x, model):
    state = (x[:, model["keep"]] - model["location"]) / model["scale"]
    if model["material"]:
        state = np.column_stack((state, x[:, 4]))
    return state


def basis(x, model):
    cross = kernel(x, model["inducing"], model["length"], model["material"])
    return solve_triangular(model["chol"], cross.T, lower=True, check_finite=False).T


def fit_gp_basis(x, y, rank, length, material):
    if x.ndim != 2 or y.shape != (len(x), 3):
        raise ValueError("GP training shapes do not align")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("GP training values must be finite")
    state, keep, location, scale = normalize_inputs(x, material)
    # Deterministic outcome-blind subsampling selects inducing locations only;
    # EVERY fitting row contributes to the likelihood sufficient statistics.
    rng = np.random.default_rng(620260907)
    indices = np.sort(rng.choice(len(x), min(rank, len(x)), replace=False))
    inducing = state[indices]
    gram = kernel(inducing, inducing, length, material)
    chol = np.linalg.cholesky(gram + np.eye(len(indices)) * 1e-6)
    response_scale = np.maximum(y.std(axis=0), 1e-4)
    model = dict(
        keep=keep,
        location=location,
        scale=scale,
        inducing=inducing,
        chol=chol,
        length=length,
        material=material,
        response_scale=response_scale,
        inducing_indices=indices,
    )
    normal = np.zeros_like(gram)
    rhs = np.zeros((len(indices), 3))
    for start in range(0, len(x), BATCH):
        phi = basis(state[start : start + BATCH], model)
        normal += phi.T @ phi
        rhs += phi.T @ (y[start : start + BATCH] / response_scale)
    model["normal"] = (normal + normal.T) / 2
    model["rhs"] = rhs
    return model


def solve_gp(model, noise):
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("GP noise variance must be positive")
    factor = cho_factor(model["normal"] + noise * np.eye(len(model["rhs"])), lower=True)
    return {
        **model,
        "noise": noise,
        "factor": factor,
        "weights": cho_solve(factor, model["rhs"]),
    }


def predict_gp(x, model, variance=False):
    state = transform(x, model)
    means, variances = [], []
    for start in range(0, len(x), BATCH):
        phi = basis(state[start : start + BATCH], model)
        means.append((phi @ model["weights"]) * model["response_scale"])
        if variance:
            quadratic = np.sum(phi * cho_solve(model["factor"], phi.T).T, axis=1)
            predictive = model["noise"] * (1 + np.maximum(quadratic, 0))
            variances.append(predictive[:, None] * model["response_scale"] ** 2)
    return np.concatenate(means), np.concatenate(variances) if variance else None


def canonical(data):
    features, frames = build_deform_local_residual_features(
        data["initial"], data["action"], data["baseline"]
    )
    return features, frames


def world_prediction(baseline, local, frames, shrinkage):
    prediction = baseline.copy()
    correction = np.einsum("ntvc,nkc->ntvk", local, frames)
    prediction[:, :, 2:-2] += shrinkage * correction
    if not np.array_equal(
        prediction[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]]
    ):
        raise AssertionError("clamped-node parity failed")
    return prediction


def family_models(x, y, length, family):
    count = x.shape[2]
    if family == "material_gp":
        return [
            fit_gp_basis(
                x.reshape(-1, x.shape[-1]),
                y.reshape(-1, 3),
                RANK_PER_NODE * count,
                length,
                True,
            )
        ]
    return [
        fit_gp_basis(
            x[:, :, node].reshape(-1, x.shape[-1]),
            y[:, :, node].reshape(-1, 3),
            RANK_PER_NODE,
            length,
            False,
        )
        for node in range(count)
    ]


def family_predict(models, x, noise, family, variance=False):
    shape = x.shape[:3] + (3,)
    solved = [solve_gp(model, noise) for model in models]
    if family == "material_gp":
        mean, var = predict_gp(x.reshape(-1, x.shape[-1]), solved[0], variance)
        return mean.reshape(shape), None if var is None else var.reshape(shape), solved
    predictions = [
        predict_gp(x[:, :, i].reshape(-1, x.shape[-1]), model, variance)
        for i, model in enumerate(solved)
    ]
    mean = np.stack(
        [item[0].reshape(x.shape[:2] + (3,)) for item in predictions], axis=2
    )
    var = None
    if variance:
        var = np.stack(
            [item[1].reshape(x.shape[:2] + (3,)) for item in predictions], axis=2
        )
    return mean, var, solved


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def ridge_prediction(train, query, shrinkage):
    model = fit_deform_local_residual(
        train["initial"],
        train["action"],
        train["baseline"],
        train["targets"],
        train["names"].tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    result = predict_deform_local_residual(
        model, query["initial"], query["action"], query["baseline"], shrinkage=shrinkage
    )
    return result["predictions"]


def prepare(output):
    import run_deform_dlo2_local_residual as runtime
    import run_deform_dlo_local_residual as local_runtime
    import run_deform_dlo_source as source_runtime
    import torch

    protocol = json.loads(
        (ROOT / "configs/sota/deform_dlo2_local_residual_v6.json").read_text()
    )
    training_path = TRAIN_ROOT / "training_validation_result.json"
    manifest_path = TRAIN_ROOT / "source_manifest.json"
    checkpoint_path = TRAIN_ROOT / "checkpoints/update_6400.pt"
    for path, key in [
        (training_path, "training_result"),
        (manifest_path, "source_manifest"),
        (checkpoint_path, "selected_checkpoint"),
    ]:
        if sha(path) != protocol[key]["sha256"]:
            raise ValueError(f"frozen {key} identity mismatch")
    training = json.loads(training_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest["dlo_type"] != "DLO2" or manifest["partition"] != "train":
        raise ValueError("only DLO2 training-partition data are allowed")
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    if len(fit_names) != 40 or len(validation_names) != 8:
        raise ValueError("development roster changed")
    if set(fit_names) & set(validation_names):
        raise ValueError("fitting and validation trajectories overlap")
    source_runtime._assert_upstream(UPSTREAM, protocol["upstream"]["commit"])
    data_root = UPSTREAM / "data_set"
    for dlo in ("DLO1", "DLO2", "DLO3", "DLO4", "DLO5"):
        source_runtime._install_eval_read_guard(data_root / dlo / "eval")
    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(UPSTREAM)
    state, restored_path = runtime._checkpoint_state(training, torch=torch)
    if sha(restored_path) != sha(checkpoint_path):
        raise ValueError("restored checkpoint differs")
    identities = []
    for label, names in [("fit", fit_names), ("validation", validation_names)]:
        trajectories = source_runtime._load_named_trajectories(
            manifest, names, frame_count=500, node_count=12
        )
        rollout = runtime._rollout(
            state, trajectories, modules=modules, torch=torch, device="cuda:0"
        )
        initial, action = local_runtime._causal_inputs(trajectories, names)
        baseline = np.asarray(rollout["predictions"], dtype=np.float64)
        targets = np.asarray(rollout["targets"], dtype=np.float64)
        if label == "validation":
            actual = float(np.mean(np.abs(baseline - targets)))
            expected = float(training["selected_checkpoint"]["validation_l1_m"])
            if abs(actual - expected) > 1e-7:
                raise ValueError(
                    f"baseline reproduction failed: {actual} vs {expected}"
                )
        data = dict(
            initial=initial, action=action, baseline=baseline, names=np.asarray(names)
        )
        if label == "fit":
            data["targets"] = targets
        else:
            np.savez_compressed(output / "validation_truth.npz", targets=targets)
        np.savez_compressed(output / f"{label}.npz", **data)
        for index in range(len(names)):
            digest = hashlib.sha256(
                initial[index].tobytes()
                + action[index].tobytes()
                + baseline[index].tobytes()
            ).hexdigest()
            identities.append((label, names[index], digest))
    fit_hashes = {item[2] for item in identities if item[0] == "fit"}
    if any(item[2] in fit_hashes for item in identities if item[0] == "validation"):
        raise ValueError("duplicate causal query crosses development partitions")
    write_json(
        output / "preparation.json",
        dict(
            training_result_sha256=sha(training_path),
            source_manifest_sha256=sha(manifest_path),
            checkpoint_sha256=sha(checkpoint_path),
            fit_names=fit_names,
            validation_names=validation_names,
            causal_query_digests=identities,
            source_test_read=False,
            official_eval_read=False,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            code_revision=os.environ.get("EXPERIMENT_REVISION"),
        ),
    )
    print(
        "PREPARATION_COMPLETE: 40 fit and 8 validation trajectories; baseline reproduced",
        flush=True,
    )


def predict(output):
    started = time.monotonic()
    data = read_npz(output / "fit.npz")
    query = read_npz(output / "validation.npz")  # contains no validation truth
    initial, action, baseline, targets, groups = _collapse_duplicate_queries(
        data["initial"],
        data["action"],
        data["baseline"],
        data["targets"],
        data["names"],
    )
    train = dict(
        initial=initial,
        action=action,
        baseline=baseline,
        targets=targets,
        names=np.asarray([group[0] for group in groups]),
    )
    ordered = sorted(
        range(len(groups)),
        key=lambda i: hashlib.sha256(train["names"][i].encode()).hexdigest(),
    )
    inner_validation = np.asarray(ordered[:8])
    inner_fit = np.asarray(ordered[8:])
    development = subset(train, inner_fit)
    selection = subset(train, inner_validation)
    fit_x, fit_frames = canonical(development)
    selection_x, selection_frames = canonical(selection)
    response = np.einsum(
        "ntvi,nij->ntvj", development["targets"] - development["baseline"], fit_frames
    )[:, :, 2:-2]
    bank = []
    selected = {}
    for family in ("independent_gp", "material_gp"):
        for length in LENGTHS:
            models = family_models(fit_x, response, length, family)
            for noise in NOISES:
                local, _, _ = family_predict(models, selection_x, noise, family)
                for shrinkage in SHRINKAGES:
                    prediction = world_prediction(
                        selection["baseline"], local, selection_frames, shrinkage
                    )
                    loss = float(np.mean(np.abs(prediction - selection["targets"])))
                    entry = dict(
                        family=family,
                        length=length,
                        noise_variance=noise,
                        shrinkage=shrinkage,
                        inner_l1_m=loss,
                    )
                    bank.append(entry)
                    print("INNER", json.dumps(entry), flush=True)
        selected[family] = min(
            [row for row in bank if row["family"] == family],
            key=lambda row: (
                row["inner_l1_m"],
                row["length"],
                row["noise_variance"],
                row["shrinkage"],
            ),
        )
    ridge_bank = []
    for shrinkage in (0.125, 0.25, 0.5, 1.0):
        prediction = ridge_prediction(development, selection, shrinkage)
        ridge_bank.append(
            dict(
                shrinkage=shrinkage,
                inner_l1_m=float(np.mean(np.abs(prediction - selection["targets"]))),
            )
        )
    selected_ridge = min(ridge_bank, key=lambda row: row["inner_l1_m"])
    write_json(
        output / "selection.json",
        dict(
            gp_bank=bank,
            selected=selected,
            ridge_bank=ridge_bank,
            selected_ridge=selected_ridge,
            inner_fit_names=development["names"].tolist(),
            inner_validation_names=selection["names"].tolist(),
            duplicate_groups=groups,
            outer_validation_truth_read=False,
        ),
    )
    all_x, all_frames = canonical(train)
    query_x, query_frames = canonical(query)
    all_response = np.einsum(
        "ntvi,nij->ntvj", train["targets"] - train["baseline"], all_frames
    )[:, :, 2:-2]
    predictions = {
        "baseline": query["baseline"],
        "ridge_fixed": ridge_prediction(train, query, 0.25),
        "ridge_inner_selected": ridge_prediction(
            train, query, selected_ridge["shrinkage"]
        ),
    }
    for family, configuration in selected.items():
        models = family_models(all_x, all_response, configuration["length"], family)
        local, variance, solved = family_predict(
            models, query_x, configuration["noise_variance"], family, True
        )
        shrinkage = configuration["shrinkage"]
        predictions[family] = world_prediction(
            query["baseline"], local, query_frames, shrinkage
        )
        # Raw diagonal predictive variance: model-based only, never claimed calibrated.
        global_variance = np.einsum("ntvc,nkc->ntvk", variance, query_frames**2)
        predictions[family + "_raw_variance"] = np.maximum(
            shrinkage**2 * global_variance, 1e-6
        )
        arrays = {}
        for index, model in enumerate(solved):
            for key in (
                "keep",
                "location",
                "scale",
                "inducing",
                "chol",
                "response_scale",
                "normal",
                "rhs",
                "weights",
                "inducing_indices",
            ):
                arrays[f"model_{index}_{key}"] = model[key]
        np.savez_compressed(output / f"{family}_model.npz", **arrays)
    np.savez_compressed(output / "predictions.npz", **predictions)
    write_json(
        output / "prediction_seal.json",
        dict(
            predictions_sha256=sha(output / "predictions.npz"),
            selection_sha256=sha(output / "selection.json"),
            outer_validation_truth_read=False,
            elapsed_seconds=time.monotonic() - started,
            model_family="finite-rank-Nystrom-Matern32-GP",
            rank_per_node=RANK_PER_NODE,
            all_fit_rows_used=True,
            covariance_calibrated=False,
        ),
    )
    print("PREDICTIONS_SEALED", flush=True)


def score(output):
    seal = json.loads((output / "prediction_seal.json").read_text())
    if sha(output / "predictions.npz") != seal["predictions_sha256"]:
        raise ValueError("prediction seal mismatch")
    predictions = read_npz(output / "predictions.npz")
    targets = read_npz(output / "validation_truth.npz")["targets"]
    names = read_npz(output / "validation.npz")["names"].tolist()
    methods = [
        "baseline",
        "ridge_fixed",
        "ridge_inner_selected",
        "independent_gp",
        "material_gp",
    ]
    per_case = {
        name: np.mean(np.abs(predictions[name] - targets), axis=(1, 2, 3))
        for name in methods
    }
    rng = np.random.default_rng(720260907)
    draws = rng.integers(0, len(names), (10000, len(names)))
    rows = []
    for name in methods:
        loss = per_case[name]
        difference = loss - per_case["ridge_fixed"]
        entry = dict(
            method=name,
            mean_coordinate_l1_mm=float(loss.mean() * 1000),
            case_l1_mm=(loss * 1000).tolist(),
            improvement_vs_ridge_percent=float(
                (1 - loss.mean() / per_case["ridge_fixed"].mean()) * 100
            ),
            wins_vs_ridge=int(np.sum(difference < -1e-12)),
            paired_difference_vs_ridge_mm=float(difference.mean() * 1000),
            paired_bootstrap_95_mm=(
                np.quantile(difference[draws].mean(axis=1), [0.025, 0.975]) * 1000
            ).tolist(),
        )
        if name.endswith("gp"):
            variance = predictions[name + "_raw_variance"]
            errors = predictions[name][:, :, 2:-2] - targets[:, :, 2:-2]
            entry["raw_coordinate_coverage90"] = float(
                np.mean(np.abs(errors) <= 1.6448536269514722 * np.sqrt(variance))
            )
            entry["raw_coordinate_nll"] = float(
                np.mean(0.5 * (np.log(2 * np.pi * variance) + errors**2 / variance))
            )
        rows.append(entry)
    result = dict(
        status="completed",
        claim_boundary="Retrospective DLO2 development pilot, not fresh confirmation",
        methods=rows,
        validation_names=names,
        trajectory_count=len(names),
        statistical_unit="complete-trajectory",
        selection=json.loads((output / "selection.json").read_text()),
        preparation=json.loads((output / "preparation.json").read_text()),
        prediction_seal=seal,
        source_test_read=False,
        official_eval_read=False,
        robot_data_collected=False,
        point_mean_has_kernel_ridge_equivalent=True,
        uncertainty_boundary="IID working likelihood; raw marginal variances uncalibrated; no joint-covariance or Bayesian-advantage claim",
    )
    write_json(output / "result.json", result)
    lines = [
        "# DLO2 GP residual development pilot",
        "",
        "Retrospective development only. No source-test or official evaluation read.",
        "",
        "| Method | L1 (mm) | Improvement vs fixed ridge | Wins / 8 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mean_coordinate_l1_mm']:.6f} | "
            f"{row['improvement_vs_ridge_percent']:.2f}% | {row['wins_vs_ridge']} |"
        )
    lines += [
        "",
        "GP hyperparameters were selected using 8 trajectories inside the original 40-trajectory fitting set.",
        "The physical checkpoint is unchanged. The original validation set was already used in historical development.",
        "Finite-rank GP means have a deterministic kernel-ridge equivalent. Raw covariance is not calibrated.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "predict", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare" and any(args.output.iterdir()):
        raise FileExistsError("preparation requires a new output directory")
    # Import historical runtime helpers only via the explicitly checked-out source.
    sys.path.insert(0, str(ROOT / "scripts/remote"))
    {"prepare": prepare, "predict": predict, "score": score}[args.stage](args.output)


if __name__ == "__main__":
    main()
