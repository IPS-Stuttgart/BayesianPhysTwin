"""Matched opened-object validation of a fixed native-state/readout coupling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform_state_restart import RestartConfig, file_digest, prediction_metrics

METRICS = ("coordinate_l1_mm", "point_rmse_mm", "fde_mm")
HORIZONS = ("early", "middle", "late")


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "deform-multiobject-state-restart-v1":
        raise ValueError("unknown multi-object protocol")
    expected = {
        "prefix_length": 50,
        "forecast_end": 170,
        "dataset_frame_offset": 2,
        "dt_s": 0.01,
        "observation_frames": [41, 49],
        "observed_nodes": [2, 4, 6, 8],
        "hidden_nodes": [3, 5, 7, 9],
        "sparse_budget_point_observations": 8,
        "gains": {"primary": 1.0, "secondary": 0.25},
        "transfer_objects": ["DLO1", "DLO3"],
        "discovery_object": "DLO2",
        "prediction_case_count": 30,
        "analysis_case_count": 29,
        "transfer_case_count": 16,
        "protected_data_access": False,
        "checkpoint_or_readout_refitting": False,
        "gain_or_seed_selection": False,
        "new_official_evaluation": False,
        "future_free_node_truth_is_model_input": False,
        "all_objects_and_noise_predictions_sealed_before_metrics": True,
    }
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("matched sensing, denominator, or boundary contract changed")
    if [item["object"] for item in value["objects"]] != ["DLO1", "DLO2", "DLO3"]:
        raise ValueError("only the three opened objects are allowed")
    if [len(item["names"]) for item in value["objects"]] != [8, 14, 8]:
        raise ValueError("the complete opened rosters must be retained")
    for item in value["objects"]:
        if len(set(item["names"])) != len(item["names"]):
            raise ValueError("duplicate trajectory identity")
        if any(Path(n).name != n or not n.endswith(".pkl") for n in item["names"]):
            raise ValueError("noncanonical trajectory identity")
        excluded = "103.pkl" if item["object"] == "DLO2" else ""
        if item["excluded_design_case"] != excluded:
            raise ValueError("only the pre-existing DLO2 design case is excluded")
        config_for_object(value, item)
    return value


def config_for_object(
    protocol: Mapping[str, Any], item: Mapping[str, Any]
) -> RestartConfig:
    node_count = item["node_count"]
    if node_count != (13 if item["object"] == "DLO1" else 12):
        raise ValueError("object topology changed")
    if item["clamped_nodes"] != [0, 1, node_count - 2, node_count - 1]:
        raise ValueError("actuator-node contract changed")
    return RestartConfig(
        prefix_length=protocol["prefix_length"],
        forecast_end=protocol["forecast_end"],
        observation_frames=tuple(protocol["observation_frames"]),
        observed_nodes=tuple(protocol["observed_nodes"]),
        hidden_nodes=tuple(protocol["hidden_nodes"]),
        clamped_nodes=tuple(item["clamped_nodes"]),
        node_count=node_count,
        dt_s=protocol["dt_s"],
        design_case=item["excluded_design_case"],
        bootstrap_replicates=protocol["bootstrap_replicates"],
        seed=protocol["bootstrap_seed"],
    )


def validate_manifest(item: Mapping[str, Any]) -> dict[str, Any]:
    spec = item["manifest"]
    path = Path(spec["path"])
    if file_digest(path) != spec["sha256"]:
        raise ValueError("opened source manifest changed")
    manifest = json.loads(path.read_text())
    names = (
        manifest["ordered_names"]
        if spec["roster_key"] == "ordered_names"
        else manifest["split"][spec["roster_key"]]
    )
    if (
        names != item["names"]
        or manifest.get("dlo_type", item["object"]) != item["object"]
    ):
        raise ValueError("manifest roster or object differs")
    expected_partition = "eval" if item["object"] == "DLO2" else "train"
    for name in names:
        entry = manifest["trajectories"][name]
        raw_path = Path(entry["path"])
        if raw_path.parts[-3:] != (item["object"], expected_partition, name):
            raise ValueError("raw trajectory is outside the allowed opened partition")
        if file_digest(raw_path) != entry["sha256"]:
            raise ValueError("opened trajectory hash changed")
    return manifest


def summarize_predictions(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    names: list[str],
    config: RestartConfig,
) -> dict[str, Any]:
    """Average simulation noise first; resample whole trajectories, never points."""
    expected = (
        len(names),
        config.forecast_end - config.prefix_length,
        config.node_count,
        3,
    )
    if truth.shape != expected or "incumbent" not in predictions:
        raise ValueError("truth shape or incumbent differs")
    values = {k: v[None] if v.ndim == 4 else v for k, v in predictions.items()}
    shape = values["incumbent"].shape
    if (
        len(shape) != 5
        or shape[1:] != expected
        or any(v.shape != shape for v in values.values())
    ):
        raise ValueError("forecasts must align in repetition/case/time/identity")
    if not np.isfinite(truth).all() or any(
        not np.isfinite(v).all() for v in values.values()
    ):
        raise ValueError("failed predictions cannot be dropped from the denominator")
    keep = [i for i, name in enumerate(names) if name != config.design_case]
    if len(keep) < 2:
        raise ValueError("at least two complete non-design trajectories are required")
    frames = dict(zip(HORIZONS, np.array_split(np.arange(expected[1]), 3), strict=True))
    per_case: dict[str, list[dict[str, Any]]] = {}
    for arm, repetitions in values.items():
        rows = []
        for case, name in enumerate(names):
            target = truth[case][:, config.hidden_nodes]
            metrics = [
                prediction_metrics(rep[case][:, config.hidden_nodes], target)
                for rep in repetitions
            ]
            row: dict[str, Any] = {"case": name}
            row.update(
                {key: float(np.mean([m[key] for m in metrics])) for key in METRICS}
            )
            for label, indices in frames.items():
                partial = [
                    prediction_metrics(
                        rep[case][:, config.hidden_nodes][indices], target[indices]
                    )
                    for rep in repetitions
                ]
                row[label] = {
                    key: float(np.mean([m[key] for m in partial])) for key in METRICS
                }
            rows.append(row)
        per_case[arm] = rows
    rng = np.random.default_rng(config.seed)
    draws = rng.integers(0, len(keep), size=(config.bootstrap_replicates, len(keep)))
    summaries: dict[str, Any] = {}
    for arm, rows in per_case.items():
        summary: dict[str, Any] = {
            "case_count": len(keep),
            "noise_repetitions": shape[0],
        }
        for key in METRICS:
            candidate = np.array([rows[i][key] for i in keep])
            base = np.array([per_case["incumbent"][i][key] for i in keep])
            if np.any(base <= 0):
                raise ValueError("relative effects require positive reference error")
            delta = candidate - base
            summary[key] = float(candidate.mean())
            summary[key + "_change_percent"] = float(
                100 * (candidate.mean() / base.mean() - 1)
            )
            summary[key + "_delta_ci95"] = np.quantile(
                delta[draws].mean(axis=1), [0.025, 0.975]
            ).tolist()
            summary[key + "_wins"] = int(np.sum(delta < -1e-10))
            summary[key + "_worst_case_ratio"] = float(np.max(candidate / base))
        summary["joint_wins"] = sum(
            rows[i][METRICS[0]] < per_case["incumbent"][i][METRICS[0]]
            and rows[i][METRICS[1]] < per_case["incumbent"][i][METRICS[1]]
            for i in keep
        )
        summary.update(
            {
                label: {
                    key: float(np.mean([rows[i][label][key] for i in keep]))
                    for key in METRICS
                }
                for label in HORIZONS
            }
        )
        summaries[arm] = summary
    return {
        "per_case": per_case,
        "summaries": summaries,
        "excluded_design_case": config.design_case or None,
        "bootstrap_unit": "whole-trajectory-after-averaging-noise-repetitions",
        "interval_scope": "conditional-on-this-opened-object-not-population-confirmation",
    }


def transfer_assessment(
    protocol: Mapping[str, Any], results: Mapping[str, Any]
) -> dict[str, Any]:
    if set(results) != {item["object"] for item in protocol["objects"]}:
        raise ValueError("all registered objects are required")
    primary = protocol["primary_arm"]
    checks: dict[str, dict[str, bool]] = {}
    for name in protocol["transfer_objects"]:
        summaries = results[name]["clean"]["summaries"]
        base, candidate, readout = (
            summaries[x] for x in ("incumbent", primary, "readout_sparse_pose")
        )
        checks[name] = {
            "coordinate_l1_improves": candidate[METRICS[0]] < base[METRICS[0]],
            "point_rmse_improves": candidate[METRICS[1]] < base[METRICS[1]],
            "beats_matched_readout_on_both": all(
                candidate[k] < readout[k] for k in METRICS[:2]
            ),
            "late_point_rmse_nonincreasing": candidate["late"][METRICS[1]]
            <= base["late"][METRICS[1]],
            "joint_wins": candidate["joint_wins"]
            >= protocol["transfer_gate"]["minimum_joint_wins_per_transfer_object"],
        }
    aggregate: dict[str, Any] = {}
    for label, objects in (
        ("transfer_only", protocol["transfer_objects"]),
        ("all_three_including_discovery", list(results)),
    ):
        conditions = ("clean", *protocol["noise"]["conditions"])
        aggregate[label] = {}
        for condition in conditions:
            arms = (
                protocol["clean_arms"]
                if condition == "clean"
                else protocol["noise_arms"]
            )
            aggregate[label][condition] = {}
            for arm in arms:
                rows = [results[o][condition]["summaries"][arm] for o in objects]
                aggregate[label][condition][arm] = {
                    **{k: float(np.mean([r[k] for r in rows])) for k in METRICS},
                    **{
                        k + "_mean_object_change_percent": float(
                            np.mean([r[k + "_change_percent"] for r in rows])
                        )
                        for k in METRICS
                    },
                    "object_count": len(objects),
                    "case_count": sum(r["case_count"] for r in rows),
                    "joint_wins": sum(r["joint_wins"] for r in rows),
                }
    return {
        "primary_arm": primary,
        "primary_transfer_gate_passed": all(all(c.values()) for c in checks.values()),
        "checks": checks,
        "object_balanced": aggregate,
        "secondary_gain_cannot_rescue_primary_gate": True,
        "fresh_or_population_confirmation": False,
    }
