"""Retrospective DLO1 development feasibility; no source-test/official data.

A prepared, hash-bound bundle is reused read-only. Nothing is promoted into an
existing paper claim. This prefix-assimilation task is NOT the original two-
state DEFORM benchmark. Local residual = historical ridge 1 / shrinkage 0.5.
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
from model import fit, forecast

PROTOCOL = {
    "contract": "sticky-hdp-residual-dlo1-development-v1",
    "claim_boundary": "Retrospective source-development feasibility only; not fresh confirmation or the official DEFORM task.",
    "source_bundle_contract": "dp-residual-dlo1-development-bundle-v1",
    "source_bundle_sha256": "70894781f89351bb25fc3b53f7b1453d9e6b36aeebc03caa7cb341ac4023ad3d",
    "source_preparation_commit": "9b49a0bac80e698d508c597794d70d007b6f6b87",
    "source_preparation_run": "34052781162",
    "checkpoint_sha256": "ea2b9e83d09a05bf94eae25aa1dafb449868d4a31961b06ec7b4420216d726e0",
    "fit_count": 40,
    "inner_fit_count": 32,
    "inner_validation_count": 8,
    "diagnostic_validation_count": 8,
    "split_rule": "first 32 and final 8 historical fit entries; final diagnostic is historical validation 8",
    "origins": [29, 99, 199, 299],
    "horizons": [10, 50],
    "primary_horizon": 50,
    "metric": "equal-recording free-node coordinate L1, averaged over fixed origins and future frames",
    "point_forecast": "posterior mean for all Bayesian arms; common convention, not an L1-optimality claim",
    "residual_dimension": 6,
    "finite_k": [2, 4, 8],
    "hdp_truncation": 8,
    "alpha": 5.0,
    "sticky": 25.0,
    "gamma": 1.0,
    "ridge": 5.0,
    "iterations": 60,
    "burn": 30,
    "thin": 5,
    "seeds": [17, 29],
    "local_baseline_ridge": 1.0,
    "local_baseline_shrinkage": 0.5,
    "complement": "persist observed residual outside source-fitted PCA subspace",
    "controls": "known corrected-baseline velocity in residual subspace and mean clamped-node velocity, source-standardized",
    "finite_selection": "minimum inner-validation primary L1, ties favor smaller K",
    "future_observation_access_in_forecast": False,
    "raw_trajectory_reads": False,
    "source_test_read": False,
    "official_eval_read": False,
    "other_dlos_read": False,
    "mcmc_convergence_established": False,
    "fresh_confirmation_authorized": False,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


class Representation:
    def __init__(self, errors, baseline, dimension):
        flat = errors[..., 2:-2, :].reshape(*errors.shape[:2], -1)
        self.center = flat.mean(axis=(0, 1))
        matrix = (flat - self.center).reshape(-1, flat.shape[-1])
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        self.basis = vt[:dimension].T
        self.scale = np.maximum((matrix @ self.basis).std(axis=0), 1e-4)
        c = self.raw_controls(baseline)
        self.control_center = c.mean(axis=(0, 1))
        self.control_scale = np.maximum(c.std(axis=(0, 1)), 1e-4)
        self.retained_fraction = float(
            np.sum((matrix @ self.basis) ** 2) / max(np.sum(matrix**2), 1e-30)
        )

    def encode(self, errors):
        flat = errors[..., 2:-2, :].reshape(*errors.shape[:2], -1)
        return ((flat - self.center) @ self.basis) / self.scale

    def decode(self, d):
        return (d * self.scale) @ self.basis.T + self.center

    def raw_controls(self, baseline):
        delta = np.diff(baseline, axis=1, prepend=baseline[:, :1])
        free = delta[..., 2:-2, :].reshape(*delta.shape[:2], -1)
        clamp = delta[:, :, [0, 1, -2, -1], :].mean(axis=2)
        return np.concatenate((free @ self.basis, clamp), axis=-1)

    def controls(self, baseline):
        return (self.raw_controls(baseline) - self.control_center) / self.control_scale

    def save(self, path):
        np.savez_compressed(
            path,
            center=self.center,
            basis=self.basis,
            scale=self.scale,
            control_center=self.control_center,
            control_scale=self.control_scale,
        )


def local_predictions(bundle, fit_indices):
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    f = np.asarray(fit_indices)
    model = fit_deform_local_residual(
        bundle["initial"][f],
        bundle["action"][f],
        bundle["baseline"][f],
        bundle["targets"][f],
        bundle["names"][f].tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    pred = predict_deform_local_residual(
        model, bundle["initial"], bundle["action"], bundle["baseline"], shrinkage=0.5
    )["predictions"]
    if not np.array_equal(
        pred[:, :, [0, 1, -2, -1]], bundle["baseline"][:, :, [0, 1, -2, -1]]
    ):
        raise RuntimeError("Local baseline changed a clamped node")
    return np.asarray(pred, dtype=np.float64)


def train_arm(states, controls, *, k, hdp, out, name):
    draws = []
    traces = []
    start = time.monotonic()
    for seed in PROTOCOL["seeds"]:
        dr, tr = fit(
            states,
            controls,
            k=k,
            hdp=hdp,
            seed=seed,
            iterations=PROTOCOL["iterations"],
            burn=PROTOCOL["burn"],
            thin=PROTOCOL["thin"],
            alpha=PROTOCOL["alpha"],
            sticky=PROTOCOL["sticky"],
            gamma=PROTOCOL["gamma"],
            ridge=PROTOCOL["ridge"],
        )
        draws.extend(dr)
        traces.append(tr)
    write(out / f"{name}-trace.json", traces)
    np.savez_compressed(
        out / f"{name}-posterior.npz",
        coef=np.asarray([d.coef for d in draws]),
        covariance=np.asarray([d.covariance for d in draws]),
        transition=np.asarray([d.transition for d in draws]),
        initial=np.asarray([d.initial for d in draws]),
        beta=np.asarray([d.beta for d in draws]),
    )
    print(
        json.dumps(
            {
                "stage": "trained",
                "arm": name,
                "seconds": time.monotonic() - start,
                "occupied": [d.occupied for d in draws],
            }
        ),
        flush=True,
    )
    return draws


def make_forecast(
    draws, rep, baseline, prefix_observations, origin, horizon, *, hard=False
):
    """Only observed prefix supplied: this function cannot inspect target future."""
    if (
        len(prefix_observations.shape) != 4
        or prefix_observations.shape[1] != origin + 1
    ):
        raise ValueError("Forecast requires exactly the allowed observation prefix")
    error_prefix = prefix_observations - baseline[:, : origin + 1]
    dp = rep.encode(error_prefix)
    controls = rep.controls(baseline)
    latent = forecast(
        draws,
        dp,
        controls[:, : origin + 1],
        controls[:, origin + 1 : origin + 1 + horizon],
        hard=hard,
    )
    last = error_prefix[:, -1, 2:-2, :].reshape(len(baseline), -1)
    complement = last - rep.decode(dp[:, -1])
    residual = rep.decode(latent) + complement[:, None]
    output = baseline[:, origin + 1 : origin + 1 + horizon].copy()
    output[:, :, 2:-2, :] += residual.reshape(len(output), horizon, -1, 3)
    return output


def summarize_predictions(predictions, targets, names):
    # [origin,recording,horizon,node,3]; statistical units are recordings.
    e = np.abs(predictions[:, :, :, 2:-2, :] - targets[:, :, :, 2:-2, :])
    case = e.mean(axis=(0, 2, 3, 4)) * 1000
    return {
        "mean_l1_mm": float(case.mean()),
        "case_l1_mm": {str(n): float(v) for n, v in zip(names, case, strict=False)},
    }


def evaluate(bundle, local, indices, rep, arms, out, label):
    ix = np.asarray(indices)
    y = bundle["targets"][ix]
    base = local[ix]
    raw = bundle["baseline"][ix]
    names = bundle["names"][ix]
    output = {}
    for horizon in PROTOCOL["horizons"]:
        predicted = {
            key: []
            for key in [
                "hybrid",
                "local_residual",
                "last_residual_hybrid",
                "last_residual_local",
                "damped_residual_velocity",
            ]
            + list(arms)
            + ["hdp_hard_regime"]
        }
        actual = []
        for origin in PROTOCOL["origins"]:
            sl = slice(origin + 1, origin + 1 + horizon)
            # All methods see the same recorded controls; assimilation controls see the same prefix.
            rawlast = raw[:, sl].copy()
            rawlast[:, :, 2:-2] += (
                y[:, origin : origin + 1, 2:-2] - raw[:, origin : origin + 1, 2:-2]
            )
            err = y[:, : origin + 1] - base[:, : origin + 1]
            last = base[:, sl].copy()
            last[:, :, 2:-2] += err[:, -1:, 2:-2]
            velocity = last.copy()
            factors = np.cumsum(0.8 ** np.arange(1, horizon + 1))
            velocity[:, :, 2:-2] += factors[None, :, None, None] * (
                err[:, -1:, 2:-2] - err[:, -2:-1, 2:-2]
            )
            for name, p in [
                ("hybrid", raw[:, sl]),
                ("local_residual", base[:, sl]),
                ("last_residual_hybrid", rawlast),
                ("last_residual_local", last),
                ("damped_residual_velocity", velocity),
            ]:
                predicted[name].append(p)
            for name, draws in arms.items():
                predicted[name].append(
                    make_forecast(draws, rep, base, y[:, : origin + 1], origin, horizon)
                )
            predicted["hdp_hard_regime"].append(
                make_forecast(
                    arms["hdp"],
                    rep,
                    base,
                    y[:, : origin + 1],
                    origin,
                    horizon,
                    hard=True,
                )
            )
            actual.append(y[:, sl])
        truths = np.asarray(actual)
        summaries = {
            name: summarize_predictions(np.asarray(p), truths, names)
            for name, p in predicted.items()
        }
        # Paired bootstrap over recordings; origins/coordinates do not count as independent units.
        rng = np.random.default_rng(72026)
        boot = rng.integers(len(names), size=(4000, len(names)))
        candidate = np.array(list(summaries["hdp"]["case_l1_mm"].values()))
        comparisons = {}
        for name, item in summaries.items():
            if name == "hdp":
                continue
            reference = np.array(list(item["case_l1_mm"].values()))
            diff = candidate - reference
            ci = np.quantile(diff[boot].mean(axis=1), [0.025, 0.975])
            comparisons[name] = {
                "delta_mm": float(diff.mean()),
                "relative_improvement": float(1 - candidate.mean() / reference.mean()),
                "wins": int(np.sum(diff < 0)),
                "recordings": len(names),
                "bootstrap_95_delta_mm": ci.tolist(),
            }
        output[str(horizon)] = {"methods": summaries, "hdp_comparisons": comparisons}
        np.savez_compressed(
            out / f"{label}-h{horizon}-predictions.npz",
            names=names,
            origins=np.asarray(PROTOCOL["origins"]),
            targets=truths,
            **{name: np.asarray(p) for name, p in predicted.items()},
        )
    write(out / f"{label}-scores.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # This record is written BEFORE a development outcome payload is loaded.
    write(out / "protocol.json", PROTOCOL)
    meta = json.loads(args.preparation.read_text())
    if (
        meta.get("contract") != PROTOCOL["source_bundle_contract"]
        or meta.get("checkpoint_sha256") != PROTOCOL["checkpoint_sha256"]
        or meta.get("commit") != PROTOCOL["source_preparation_commit"]
        or meta.get("run_id") != PROTOCOL["source_preparation_run"]
        or meta.get("bundle_sha256") != PROTOCOL["source_bundle_sha256"]
    ):
        raise RuntimeError("Wrong prepared source/checkpoint")
    if any(
        meta.get(k) is not False
        for k in ("source_test_read", "official_eval_read", "other_dlos_read")
    ):
        raise RuntimeError("Prepared bundle does not attest development-only data")
    if (
        meta.get("fit_count") != 40
        or meta.get("validation_count") != 8
        or digest(args.bundle) != meta.get("bundle_sha256")
    ):
        raise RuntimeError("Prepared source bundle identity/count mismatch")
    write(
        out / "input_binding.json",
        {
            "preparation": meta,
            "preparation_sha256": digest(args.preparation),
            "bundle_sha256": digest(args.bundle),
            "code_commit": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
        },
    )
    with np.load(args.bundle, allow_pickle=False) as f:
        bundle = {k: f[k] for k in f.files}
    if (
        bundle["targets"].shape != (48, 498, 13, 3)
        or bundle["baseline"].shape != bundle["targets"].shape
    ):
        raise RuntimeError("Unexpected DLO1 source shape")
    if (
        bundle["split"].tolist() != ["fit"] * 40 + ["validation"] * 8
        or len(set(bundle["names"].tolist())) != 48
    ):
        raise RuntimeError("Wrong development partition")
    print(
        "Verified read-only DLO1 fit/validation bundle; no raw trajectory paths opened.",
        flush=True,
    )
    started = time.monotonic()
    # Inner validation selects finite K only. Representation/local correction are refitted on inner training.
    local = local_predictions(bundle, np.arange(32))
    errors = bundle["targets"][:32] - local[:32]
    rep = Representation(errors, local[:32], PROTOCOL["residual_dimension"])
    rep.save(out / "inner-representation.npz")
    states = rep.encode(errors)
    controls = rep.controls(local[:32])
    arms = {}
    for k in PROTOCOL["finite_k"]:
        arms[f"finite_k{k}"] = train_arm(
            states, controls, k=k, hdp=False, out=out, name=f"inner-finite-k{k}"
        )
    arms["pooled_ar"] = train_arm(
        states, controls, k=1, hdp=False, out=out, name="inner-pooled"
    )
    arms["hdp"] = train_arm(
        states,
        controls,
        k=PROTOCOL["hdp_truncation"],
        hdp=True,
        out=out,
        name="inner-hdp",
    )
    inner = evaluate(
        bundle, local, np.arange(32, 40), rep, arms, out, "inner-validation"
    )
    primary = inner[str(PROTOCOL["primary_horizon"])]["methods"]
    selected = min(
        PROTOCOL["finite_k"], key=lambda k: (primary[f"finite_k{k}"]["mean_l1_mm"], k)
    )
    write(
        out / "selection.json",
        {
            "finite_k": selected,
            "rule": PROTOCOL["finite_selection"],
            "final_diagnostic_scored": False,
            "inner_primary": primary,
        },
    )
    print(json.dumps({"stage": "finite-K-selected", "k": selected}), flush=True)
    # Refit on 40 source-fit recordings; historical validation 8 is diagnostic only.
    local = local_predictions(bundle, np.arange(40))
    errors = bundle["targets"][:40] - local[:40]
    rep = Representation(errors, local[:40], PROTOCOL["residual_dimension"])
    rep.save(out / "final-representation.npz")
    states = rep.encode(errors)
    controls = rep.controls(local[:40])
    arms = {}
    for name, k, hdp in [
        ("pooled_ar", 1, False),
        ("finite_selected", selected, False),
        ("hdp", PROTOCOL["hdp_truncation"], True),
    ]:
        arms[name] = train_arm(
            states, controls, k=k, hdp=hdp, out=out, name=f"final-{name}"
        )
    final = evaluate(
        bundle, local, np.arange(40, 48), rep, arms, out, "diagnostic-validation"
    )
    comp = final[str(PROTOCOL["primary_horizon"])]["hdp_comparisons"]
    strongest = [
        "last_residual_hybrid",
        "last_residual_local",
        "damped_residual_velocity",
        "pooled_ar",
        "finite_selected",
    ]
    positive = all(comp[name]["bootstrap_95_delta_mm"][1] < 0 for name in strongest)
    result = {
        "contract": PROTOCOL["contract"],
        "claim_boundary": PROTOCOL["claim_boundary"],
        "selected_finite_k": selected,
        "result": final,
        "retained_residual_variance_fraction": rep.retained_fraction,
        "hdp_occupied_states": [d.occupied for d in arms["hdp"]],
        "hdp_truncation_saturated": any(
            d.occupied == PROTOCOL["hdp_truncation"] for d in arms["hdp"]
        ),
        "development_positive_against_all_strong_controls": positive,
        "fresh_confirmation_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
        "source_test_read": False,
        "official_eval_read": False,
        "other_dlos_read": False,
        "mcmc_convergence_established": False,
        "versions": {"python": sys.version, "numpy": np.__version__},
    }
    write(out / "result.json", result)
    write(
        out / "artifact_hashes.json",
        {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    )
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print("RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
