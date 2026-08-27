"""Audit frozen noisy predictions with vectorized, independently written metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)


def metric_arrays(points: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    if points.ndim != 5 or points.shape[1:] != truth.shape or points.shape[-1] != 3:
        raise ValueError("expected repetitions, cases, time, identities, xyz")
    if not np.isfinite(points).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite forecasts cannot be dropped")
    error = points.astype(np.float64) - truth.astype(np.float64)[None]
    squared_distance = np.einsum("rbthc,rbthc->rbth", error, error)
    return {
        "coordinate_l1_mm": 1000 * np.mean(np.abs(error), axis=(2, 3, 4)),
        "point_rmse_mm": 1000 * np.sqrt(squared_distance.mean(axis=(2, 3))),
        "fde_mm": 1000 * np.sqrt(squared_distance[:, :, -1]).mean(axis=2),
    }


def verify(
    run: Path, archive: Path, parent_protocol: Path, noise_protocol: Path
) -> dict[str, Any]:
    base_config = json.loads(parent_protocol.read_text())
    config = json.loads(noise_protocol.read_text())
    barrier = json.loads((run / "prediction_barrier.json").read_text())
    preflight = json.loads((run / "preflight.json").read_text())
    result = json.loads((run / "result.json").read_text())
    if file_digest(archive) != base_config["source_archive_sha256"]:
        raise ValueError("reference archive identity differs")
    if preflight["protocol_sha256"] != file_digest(noise_protocol):
        raise ValueError("noise protocol changed")
    if barrier["preflight_sha256"] != file_digest(run / "preflight.json"):
        raise ValueError("preflight identity differs")
    if result["prediction_barrier_sha256"] != file_digest(
        run / "prediction_barrier.json"
    ):
        raise ValueError("barrier identity differs")
    if (
        preflight["native_parent_replay_byte_identical"] is not True
        or preflight["zero_update_byte_identical"] is not True
    ):
        raise ValueError("native replay controls were not passed")
    with np.load(archive, allow_pickle=False) as source:
        names = source["names"].tolist()
        start, end = base_config["prefix_length"], base_config["forecast_end"]
        mean = source["candidate_predictions"][:, start:end].copy()
        prefix = source["candidate_predictions"][:, :start].copy()
        observed = source["targets"][:, base_config["observation_frames"]][
            :, :, base_config["observed_nodes"]
        ].copy()
        truth = source["targets"][:, start:end, base_config["hidden_nodes"]].copy()
    if names != base_config["expected_names"] or barrier["case_count"] != len(names):
        raise ValueError("fixed opened roster differs")
    repeats = config["repetitions"]
    noise_lookup = {
        (x["condition"], x["repetition"]): x["noise_sha256"]
        for x in barrier["noise_hashes"]
    }
    if len(noise_lookup) != len(config["conditions"]) * repeats:
        raise ValueError("missing noise identity")
    for repeat in range(repeats):
        rng = np.random.default_rng(config["seed"] + repeat)
        independent = rng.normal(0, config["independent_noise_std_m"], observed.shape)
        bias = rng.normal(0, config["shared_bias_std_m"], (len(names), 1, 1, 3))
        for condition, values in zip(
            config["conditions"], (independent, independent + bias), strict=True
        ):
            if array_digest(values) != noise_lookup[condition, repeat]:
                raise ValueError("simulated measurement noise changed")
    keep = [i for i, name in enumerate(names) if name != base_config["design_case"]]
    indices = np.random.default_rng(base_config["seed"]).integers(
        0,
        len(keep),
        size=(base_config["bootstrap_replicates"], len(keep)),
    )
    verified = 0
    for condition in config["conditions"]:
        member = run / f"{condition}.npz"
        declared = barrier["conditions"][condition]
        if file_digest(member) != declared["file_sha256"]:
            raise ValueError("noise prediction file changed")
        with np.load(member, allow_pickle=False) as source:
            predictions = {arm: source[arm].copy() for arm in source.files}
        if list(predictions) != config["arms"]:
            raise ValueError("noise arm roster differs")
        for repeated_mean in predictions["incumbent"]:
            if array_digest(repeated_mean) != array_digest(mean):
                raise ValueError("unchanged incumbent mean bytes differ")
        # Verify the cheap matched readout separately from the native propagation.
        nodes = base_config["observed_nodes"]
        knots = sorted(nodes + base_config["clamped_nodes"])
        for repeat in range(repeats):
            rng = np.random.default_rng(config["seed"] + repeat)
            noise = rng.normal(0, config["independent_noise_std_m"], observed.shape)
            shared = rng.normal(0, config["shared_bias_std_m"], (len(names), 1, 1, 3))
            if condition == "independent_1mm_shared_5mm":
                noise += shared
            residual = observed[:, -1] + noise[:, -1] - prefix[:, -1, nodes]
            update = np.zeros((len(names), base_config["node_count"], 3))
            for case in range(len(names)):
                for axis in range(3):
                    values = [
                        residual[case, nodes.index(k), axis] if k in nodes else 0.0
                        for k in knots
                    ]
                    update[case, :, axis] = np.interp(
                        np.arange(base_config["node_count"]), knots, values
                    )
            np.testing.assert_allclose(
                predictions["readout_sparse_pose"][repeat],
                mean + update[:, None],
                atol=1e-14,
                rtol=1e-13,
            )
        baseline = metric_arrays(
            predictions["incumbent"][:, :, :, base_config["hidden_nodes"]], truth
        )
        for arm, points in predictions.items():
            if array_digest(points) != declared["array_digests"][arm]:
                raise ValueError("noise prediction array changed")
            metrics = metric_arrays(points[:, :, :, base_config["hidden_nodes"]], truth)
            recorded = result["conditions"][condition]
            for metric, values in metrics.items():
                expected_rows = np.array(
                    [
                        [
                            recorded["per_case"][arm][i]["noise_repetition_metrics"][r][
                                metric
                            ]
                            for i in range(len(names))
                        ]
                        for r in range(repeats)
                    ]
                )
                np.testing.assert_allclose(
                    values, expected_rows, rtol=1e-12, atol=1e-12
                )
                case_means = values.mean(axis=0)[keep]
                base_means = baseline[metric].mean(axis=0)[keep]
                summary = recorded["summaries"][arm]
                np.testing.assert_allclose(
                    case_means.mean(), summary[metric], rtol=1e-12
                )
                interval = np.quantile(
                    (case_means - base_means)[indices].mean(axis=1), [0.025, 0.975]
                )
                np.testing.assert_allclose(
                    interval, summary[metric + "_delta_ci95"], rtol=1e-11, atol=1e-12
                )
            verified += repeats * len(names)
    return {
        "schema": "deform-state-restart-noise-verification-v1",
        "passed": True,
        "case_predictions_verified": verified,
        "reported_trajectory_count": len(keep),
        "noise_repetitions_are_not_independent_cases": True,
        "result_sha256": file_digest(run / "result.json"),
        "barrier_sha256": file_digest(run / "prediction_barrier.json"),
        "independent_native_physics_reexecution": False,
        "protected_cohort_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--noise-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.archive, args.parent_protocol, args.noise_protocol)
    write_json_once(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
