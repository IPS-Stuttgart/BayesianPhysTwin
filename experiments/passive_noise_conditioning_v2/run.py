"""Passive visibility/noise transfer with exact noise-integrated risk.

Real recorded trajectories; artificial masks and Gaussian measurement perturbations.
This is a predictive discrepancy study, not a simulator-parameter posterior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np

HORIZONS = (0, 5, 25)
CUTS = np.arange(25, 451, 25)
D = 24
ARMS = (
    "mean_only",
    "fixed_ridge",
    "pooled_noise_ridge",
    "adaptive_empirical",
    "adaptive_rod",
    "diagonal_R",
    "direct_noise_aware_ridge",
)
PRIMARY = ("clean", "iid2", "iid10", "hetero2_10", "shared10")
CONDITIONS = PRIMARY + ("iid30", "shared10_Rquarter", "shared10_Rfour")
REGS = (0.0001, 0.01, 0.1, 1.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def masks(budget: int, source: bool) -> list[np.ndarray]:
    if budget not in (2, 4):
        raise ValueError("Unregistered observation budget")
    values = itertools.combinations(range(8), budget)
    return [np.array(v) for v in values if (v[-1] - v[0] + 1 == budget) == source]


def coords(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes)
    if nodes.ndim != 1 or nodes.dtype.kind not in "iu" or len(set(nodes)) != len(nodes):
        raise ValueError("Nodes must be unique integer indices")
    if len(nodes) == 0 or nodes.min() < 0 or nodes.max() >= 8:
        raise ValueError("Nodes outside free-node roster")
    return (3 * nodes[:, None] + np.arange(3)).ravel()


def hidden(nodes: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            h * D + coords(np.setdiff1d(np.arange(8), nodes))
            for h in range(len(HORIZONS))
        ]
    )


def noise(label: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
    size = budget * 3
    if label == "clean":
        r = np.zeros((size, size))
    elif label.startswith("iid"):
        sd = float(label[3:]) / 1000
        r = sd**2 * np.eye(size)
    elif label == "hetero2_10":
        sds = np.resize([0.002, 0.010], budget)
        r = np.diag(np.repeat(sds**2, 3))
    elif label.startswith("shared10"):
        r = 0.002**2 * np.eye(size) + 0.010**2 * np.kron(
            np.ones((budget, budget)), np.eye(3)
        )
    else:
        raise ValueError("Unknown noise condition")
    scale = 0.25 if label.endswith("Rquarter") else 4 if label.endswith("Rfour") else 1
    return r, scale * r


def fit(samples: np.ndarray, rank: int = 0) -> dict:
    x = np.asarray(samples, dtype=np.float64).reshape(-1, D * len(HORIZONS))
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Insufficient or nonfinite source data")
    mean = x.mean(0)
    centered = x - mean
    cov = centered.T @ centered / len(x)
    marginal = np.maximum(np.diag(cov), 1e-12)
    cov[np.diag_indices_from(cov)] = marginal
    if rank:
        if rank not in (2, 4):
            raise ValueError("Unregistered rod rank")
        basis = np.sin(np.pi * np.arange(1, 9)[:, None] * np.arange(1, rank + 1) / 9)
        q, _ = np.linalg.qr(basis)
        p = np.kron(np.eye(3), np.kron(q @ q.T, np.eye(3)))
        cov = p @ cov @ p.T
        scale = np.sqrt(marginal / np.maximum(np.diag(cov), 1e-18))
        cov = 0.95 * cov * np.outer(scale, scale) + 0.05 * np.diag(marginal)
    return {
        "mean": mean,
        "cov": cov,
        "centered": centered,
        "reg_scale": float(marginal[:D].mean()),
    }


def gain(
    model: dict, nodes: np.ndarray, arm: str, reg: float, supplied_r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    o = coords(nodes)
    c = model["cov"]
    r = np.asarray(supplied_r, dtype=np.float64)
    if r.shape != (len(o), len(o)) or not np.isfinite(r).all():
        raise ValueError("Invalid supplied observation covariance")
    if not np.allclose(r, r.T, atol=1e-14) or np.linalg.eigvalsh(r).min() < -1e-14:
        raise ValueError("Observation covariance is not symmetric PSD")
    if arm == "mean_only":
        return np.zeros((len(c), len(o))), np.diag(c).copy()
    if reg <= 0 or not np.isfinite(reg):
        raise ValueError("Regularizer must be positive")
    if arm == "fixed_ridge":
        r = np.zeros_like(r)
    elif arm == "pooled_noise_ridge":
        r = np.eye(len(o)) * ((0**2 + 0.005**2 + 0.015**2) / 3)
    elif arm == "diagonal_R":
        r = np.diag(np.diag(r))
    elif arm not in ("adaptive_empirical", "adaptive_rod", "direct_noise_aware_ridge"):
        raise ValueError("Unknown estimator arm")
    regularizer = reg * model["reg_scale"] * np.eye(len(o))
    if arm == "direct_noise_aware_ridge":
        # Independently solved noise-augmented least squares: an equivalence control.
        x = model["centered"][:, o]
        targets = model["centered"]
        g = np.linalg.solve(
            x.T @ x / len(x) + r + regularizer, x.T @ targets / len(x)
        ).T
    else:
        g = np.linalg.solve(c[np.ix_(o, o)] + r + regularizer, c[o]).T
    v = np.diag(c) - np.sum(g * c[:, o], axis=1)
    if v.min() < -1e-10 or not np.isfinite(g).all():
        raise ArithmeticError("Invalid conditional variance or gain")
    return g, np.maximum(v, 1e-12)


def predict(
    model: dict, g: np.ndarray, observed_residual: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """Only visible current innovations enter; no hidden or future argument exists."""
    y = np.asarray(observed_residual, dtype=np.float64)
    if y.shape[-1] != g.shape[1] or not np.isfinite(y).all():
        raise ValueError("Invalid visible innovation")
    return model["mean"] + (y - model["mean"][coords(nodes)]) @ g.T


def expected_error2(
    predicted: np.ndarray, truth: np.ndarray, g: np.ndarray, actual_r: np.ndarray
) -> np.ndarray:
    # E[(prediction(y+epsilon)-truth)^2] for epsilon~N(0,R), exactly, without seeds.
    noise_variance = np.einsum("ij,jk,ik->i", g, actual_r, g)
    return (predicted - truth) ** 2 + np.maximum(noise_variance, 0)


def source_cv(
    source: np.ndarray, budget: int, arm: str, param: dict, calibration: bool = False
) -> float | dict:
    losses, static_sum, counts = [], np.zeros(72), np.zeros(72)
    ratios, ratio_counts = np.zeros(3), np.zeros(3)
    for k in range(len(source)):
        m = fit(np.delete(source, k, axis=0), param.get("rank", 0))
        val = source[k]
        mask_losses = []
        for nodes in masks(budget, True):
            hid = hidden(nodes)
            for label in ("clean", "iid5", "iid15"):
                actual, supplied = noise(label, budget)
                g, v = gain(m, nodes, arm, param["reg"], supplied)
                pred = predict(m, g, val[:, coords(nodes)], nodes)
                e2 = expected_error2(pred, val, g, actual)
                first_two = hid[: 2 * len(hid) // 3]
                mask_losses.append(float(e2[:, first_two].mean()))
                if calibration:
                    static_sum[hid] += e2[:, hid].sum(0)
                    counts[hid] += len(val)
                    for j, idx in enumerate(hid.reshape(3, -1)):
                        ratios[j] += (e2[:, idx] / v[idx]).sum()
                        ratio_counts[j] += e2[:, idx].size
        losses.append(float(np.mean(mask_losses)))
    if not calibration:
        return float(np.mean(losses))
    return {
        "static_variance": (static_sum / counts).tolist(),
        "posterior_scale_by_horizon": (ratios / ratio_counts).tolist(),
        "source_expected_mse": float(np.mean(losses)),
    }


def select(source: np.ndarray) -> dict:
    selections = {}
    for budget in (2, 4):
        selections[budget] = {}
        for arm in ARMS:
            if arm == "mean_only":
                params = [{"reg": 0.01}]
            elif arm == "adaptive_rod":
                params = [{"reg": r, "rank": k} for r in REGS for k in (2, 4)]
            else:
                params = [{"reg": r} for r in REGS]
            options = [
                {
                    "parameters": p,
                    "source_expected_mse": source_cv(source, budget, arm, p),
                }
                for p in params
            ]
            chosen = min(options, key=lambda v: v["source_expected_mse"])
            selections[budget][arm] = {
                "parameters": chosen["parameters"],
                "options": options,
                "calibration": source_cv(
                    source, budget, arm, chosen["parameters"], True
                ),
            }
            print("SOURCE", budget, arm, chosen, flush=True)
        # These arms are algebraically equivalent, not independent competitors.
        if (
            selections[budget]["adaptive_empirical"]["parameters"]
            != selections[budget]["direct_noise_aware_ridge"]["parameters"]
        ):
            raise ArithmeticError(
                "Independent equivalent fits chose different regularizers"
            )
    return selections


def load_panel(parent: Path, data: Path, dlo: str, stage: str, pins: dict) -> tuple:
    folder = parent / (dlo.lower() + "-" + stage)
    pred_path = folder / (stage + "_predictions.npz")
    manifest_path = folder / (
        "source_manifest.json" if stage == "source" else "eval_manifest.json"
    )
    if sha(pred_path) != pins["predictions"] or sha(manifest_path) != pins["manifest"]:
        raise ValueError("Parent forecast or manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    with np.load(pred_path, allow_pickle=False) as z:
        names = list(map(str, z["names"]))
        base = np.asarray(z["candidate"], dtype=np.float64)
    expected = (
        manifest["partitions"]["source_test"]
        if stage == "source"
        else manifest["ordered_names"]
    )
    if names != expected or base.shape != (8 if stage == "source" else 14, 498, 12, 3):
        raise ValueError("Parent name order or shape mismatch")
    arrays, identities = [], []
    for name in names:
        if Path(name).name != name:
            raise ValueError("Invalid dataset basename")
        p = data / dlo / ("train" if stage == "source" else "eval") / name
        identity = manifest["trajectories"][name]
        if sha(p) != identity["sha256"] or p.stat().st_size != identity["size_bytes"]:
            raise ValueError("Dataset checksum mismatch")
        # Only checksum-bound trusted official NumPy pickles, never arbitrary user data.
        with p.open("rb") as f:
            x = np.asarray(pickle.load(f), dtype=np.float32)
        if x.shape != (500, 3, 12) or not np.isfinite(x).all():
            raise ValueError("Invalid trajectory")
        x = x.transpose(0, 2, 1).copy()
        x[:, :, 2] = np.clip(x[:, :, 2], 0.002001, 10000)
        arrays.append(x[2:].astype(np.float64))
        identities.append({"name": name, "sha256": identity["sha256"]})
    residual = np.stack(arrays)[:, :, 2:10] - base[:, :, 2:10]
    samples = np.stack([residual[:, CUTS + h] for h in HORIZONS], axis=2).reshape(
        len(names), len(CUTS), 72
    )
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite residual")
    return names, samples, identities


def evaluate(
    dlo: str, names: list, target: np.ndarray, source: np.ndarray, choices: dict
) -> tuple[list, float]:
    rows, max_parity = [], 0.0
    for budget in (2, 4):
        models = {
            arm: fit(source, choices[budget][arm]["parameters"].get("rank", 0))
            for arm in ARMS
        }
        fixed_arm = min(
            ("fixed_ridge", "pooled_noise_ridge"),
            key=lambda a: choices[budget][a]["calibration"]["source_expected_mse"],
        )
        for condition in CONDITIONS:
            actual, supplied = noise(condition, budget)
            accum = {a: np.zeros((len(names), 3, 3)) for a in ARMS}
            for nodes in masks(budget, False):
                hid = hidden(nodes).reshape(3, -1)
                preds = {}
                for arm in ARMS:
                    choice, m = choices[budget][arm], models[arm]
                    g, v = gain(m, nodes, arm, choice["parameters"]["reg"], supplied)
                    # Prediction takes only visible residuals; full target is scoring-only.
                    pred = predict(m, g, target[:, :, coords(nodes)], nodes)
                    preds[arm] = pred
                    e2 = expected_error2(pred, target, g, actual)
                    scales = choice["calibration"]["posterior_scale_by_horizon"]
                    static = np.array(choice["calibration"]["static_variance"])
                    for j, idx in enumerate(hid):
                        adaptive_v = np.maximum(v[idx] * scales[j], 1e-12)
                        static_v = np.maximum(static[idx], 1e-12)
                        accum[arm][:, j, 0] += e2[:, :, idx].mean((1, 2))
                        accum[arm][:, j, 1] += 0.5 * (
                            np.log(2 * np.pi * adaptive_v) + e2[:, :, idx] / adaptive_v
                        ).mean((1, 2))
                        accum[arm][:, j, 2] += 0.5 * (
                            np.log(2 * np.pi * static_v) + e2[:, :, idx] / static_v
                        ).mean((1, 2))
                max_parity = max(
                    max_parity,
                    float(
                        np.max(
                            np.abs(
                                preds["adaptive_empirical"]
                                - preds["direct_noise_aware_ridge"]
                            )
                        )
                    ),
                )
            count = len(masks(budget, False))
            accum["source_selected_fixed"] = accum[fixed_arm].copy()
            for arm, vals in accum.items():
                vals = vals / count
                for k, name in enumerate(names):
                    for j, horizon in enumerate(HORIZONS):
                        rows.append(
                            {
                                "dlo": dlo,
                                "trajectory": name,
                                "budget": budget,
                                "condition": condition,
                                "arm": arm,
                                "horizon": horizon,
                                "expected_mse_m2": float(vals[k, j, 0]),
                                "root_expected_mse_mm": float(
                                    1000 * np.sqrt(vals[k, j, 0])
                                ),
                                "posterior_expected_nll": float(vals[k, j, 1]),
                                "static_expected_nll_same_mean": float(vals[k, j, 2]),
                                "mask_count": count,
                                "cut_count": len(CUTS),
                                "source_selected_fixed": fixed_arm,
                            }
                        )
    if max_parity > 1e-10:
        raise ArithmeticError("Gaussian/direct ridge parity failed")
    return rows, max_parity


def summarize(rows: list, bootstrap_reps: int) -> tuple[list, list]:
    summary, contrasts = [], []
    rng = np.random.default_rng(6009)
    arms = ARMS + ("source_selected_fixed",)
    for budget in (2, 4):
        for horizon in HORIZONS:
            for condition in ("primary_mix",) + CONDITIONS:
                use = PRIMARY if condition == "primary_mix" else (condition,)
                vectors = {}
                for arm in arms:
                    grouped = {}
                    for row in rows:
                        if (
                            row["budget"] == budget
                            and row["horizon"] == horizon
                            and row["condition"] in use
                            and row["arm"] == arm
                        ):
                            grouped.setdefault(
                                (row["dlo"], row["trajectory"]), []
                            ).append(row)
                    if len(grouped) != 28 or any(
                        len(v) != len(use) for v in grouped.values()
                    ):
                        raise ValueError("Incomplete evaluation panel")
                    mse = {
                        key: np.mean([r["expected_mse_m2"] for r in v])
                        for key, v in grouped.items()
                    }
                    vec = {
                        dlo: np.array(
                            [1000 * np.sqrt(mse[k]) for k in sorted(mse) if k[0] == dlo]
                        )
                        for dlo in ("DLO4", "DLO5")
                    }
                    vectors[arm] = vec
                    summary.append(
                        {
                            "budget": budget,
                            "horizon": horizon,
                            "condition": condition,
                            "arm": arm,
                            "DLO4_mm": float(vec["DLO4"].mean()),
                            "DLO5_mm": float(vec["DLO5"].mean()),
                            "equal_object_mm": float(
                                np.mean([v.mean() for v in vec.values()])
                            ),
                            "posterior_expected_nll": float(
                                np.mean(
                                    [
                                        r["posterior_expected_nll"]
                                        for v in grouped.values()
                                        for r in v
                                    ]
                                )
                            ),
                            "static_expected_nll_same_mean": float(
                                np.mean(
                                    [
                                        r["static_expected_nll_same_mean"]
                                        for v in grouped.values()
                                        for r in v
                                    ]
                                )
                            ),
                        }
                    )
                for a, b in (
                    ("adaptive_empirical", "source_selected_fixed"),
                    ("adaptive_rod", "source_selected_fixed"),
                    ("adaptive_rod", "direct_noise_aware_ridge"),
                    ("adaptive_empirical", "diagonal_R"),
                    ("adaptive_empirical", "mean_only"),
                ):
                    differences = {d: vectors[a][d] - vectors[b][d] for d in vectors[a]}
                    boot = (
                        sum(
                            v[rng.integers(0, 14, (bootstrap_reps, 14))].mean(1)
                            for v in differences.values()
                        )
                        / 2
                    )
                    delta = float(np.mean([v.mean() for v in differences.values()]))
                    contrasts.append(
                        {
                            "budget": budget,
                            "horizon": horizon,
                            "condition": condition,
                            "candidate": a,
                            "comparator": b,
                            "difference_mm": delta,
                            "gain_pct": float(
                                -100
                                * delta
                                / np.mean([v.mean() for v in vectors[b].values()])
                            ),
                            "ci95_mm": np.quantile(boot, [0.025, 0.975]).tolist(),
                            "per_dlo_difference_mm": {
                                d: float(v.mean()) for d, v in differences.items()
                            },
                            "wins": sum(
                                int((v < -1e-9).sum()) for v in differences.values()
                            ),
                        }
                    )
    return summary, contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    try:
        if (
            protocol["horizons"] != list(HORIZONS)
            or protocol["cutoffs"] != CUTS.tolist()
            or protocol["target_conditions"] != list(CONDITIONS)
        ):
            raise ValueError("Protocol/implementation schedule mismatch")
        sources, choices, source_ids = {}, {}, {}
        for dlo in ("DLO4", "DLO5"):
            _, sources[dlo], source_ids[dlo] = load_panel(
                args.parent,
                args.data,
                dlo,
                "source",
                protocol["data_pins"][dlo]["source"],
            )
            choices[dlo] = select(sources[dlo])
        save(args.output / "source_choices.json", choices)
        save(
            args.output / "source_seal.json",
            {
                "source_sha256": sha(Path(__file__)),
                "protocol_sha256": sha(protocol_path),
                "source_choices_sha256": sha(args.output / "source_choices.json"),
                "source_identities": source_ids,
                "target_loaded": False,
                "git_sha": os.environ.get("GITHUB_SHA"),
            },
        )
        rows, parity, target_ids = [], [], {}
        for dlo in ("DLO4", "DLO5"):
            names, target, target_ids[dlo] = load_panel(
                args.parent,
                args.data,
                dlo,
                "target",
                protocol["data_pins"][dlo]["target"],
            )
            r, p = evaluate(dlo, names, target, sources[dlo], choices[dlo])
            rows.extend(r)
            parity.append(p)
        with (args.output / "per_trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary, contrasts = summarize(rows, protocol["bootstrap_replicates"])
        primary = [
            c
            for c in contrasts
            if (c["budget"], c["horizon"], c["condition"]) == (2, 0, "primary_mix")
        ]
        for c in primary:
            threshold = 1 if c["comparator"] == "direct_noise_aware_ridge" else 2
            c["passed"] = (
                c["gain_pct"] >= threshold
                and c["ci95_mm"][1] < 0
                and all(v < 0 for v in c["per_dlo_difference_mm"].values())
            )
        result = {
            "contract": protocol["contract"],
            "status": "complete",
            "scope": protocol["scope"],
            "noise_assumption": protocol["noise"],
            "primary": primary,
            "summary": summary,
            "contrasts": contrasts,
            "max_independent_ridge_parity_m": max(parity),
            "target_identities": target_ids,
            "source_seal_sha256": sha(args.output / "source_seal.json"),
            "rows_sha256": sha(args.output / "per_trajectory.csv"),
            "runtime": {
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "git_sha": os.environ.get("GITHUB_SHA"),
                "runner": os.environ.get("RUNNER_NAME"),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "seconds": time.monotonic() - start,
            },
        }
        save(args.output / "result.json", result)
        lines = [
            "# Passive visibility and noise transfer",
            "",
            protocol["scope"],
            "",
            protocol["noise"],
            "",
            "Primary: current hidden geometry; 2 of 8 free nodes observed; 21 unseen non-contiguous masks; equal five-condition mix.",
            "",
            "| Arm | DLO4 mm | DLO5 mm | Equal-object mm |",
            "|---|---:|---:|---:|",
        ]
        for row in summary:
            if (row["budget"], row["horizon"], row["condition"]) == (
                2,
                0,
                "primary_mix",
            ):
                lines.append(
                    f"| {row['arm']} | {row['DLO4_mm']:.4f} | {row['DLO5_mm']:.4f} | {row['equal_object_mm']:.4f} |"
                )
        lines += [
            "",
            "Primary contrasts (candidate minus comparator, mm):",
            "",
            json.dumps(primary, indent=2),
            "",
            "Gaussian/noise-aware direct ridge parity is an equivalence control, not a claim of Bayesian exclusivity.",
            "Intervals resample complete trajectories within two fixed objects; not unseen-object generalization.",
        ]
        (args.output / "report.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        print("RESULT_SHA256", sha(args.output / "result.json"), flush=True)
    except Exception as error:
        save(
            args.output / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
