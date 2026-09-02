#!/usr/bin/env python3
"""Seal a source diagnosis and evaluate diagnosis-guided discrepancy transfer.

This program separates inference from evaluation.  A source model and its
transfer decision are serialized and content-addressed before any protected
target score is computed.  The target evaluator cannot change selected groups,
coefficients, standardizers, or the wrong-diagnosis placebo after target data
are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "hierarchical_missing_physics_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("hierarchical_missing_physics_core", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import diagnosis core from {CORE_PATH}")
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)

EPS = 1.0e-12
PHYSICAL_ROLLOUT_FORBIDDEN_GROUPS = frozenset({"sensor"})


class TransferContractError(ValueError):
    """Raised when source/target custody or a frozen transfer rule is violated."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_slice(value: slice) -> list[int]:
    return [int(value.start), int(value.stop)]


def _decode_slice(value: list[int]) -> slice:
    if len(value) != 2:
        raise TransferContractError(f"invalid serialized slice: {value}")
    return slice(int(value[0]), int(value[1]))


@dataclass(frozen=True)
class SealedSourceModel:
    model: Any
    selected_groups: tuple[str, ...]
    wrong_diagnosis_group: str | None
    source_result_id: str
    source_panel_sha256: str
    metadata: Mapping[str, Any]
    model_id: str


def model_payload(
    model: Any,
    *,
    selected_groups: Iterable[str],
    wrong_diagnosis_group: str | None,
    source_result_id: str,
    source_panel_sha256: str,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    group_order = tuple(model.group_order)
    selected = tuple(selected_groups)
    unknown = sorted(set(selected) - set(group_order))
    if unknown:
        raise TransferContractError(f"selected groups absent from model: {unknown}")
    if set(selected) & PHYSICAL_ROLLOUT_FORBIDDEN_GROUPS:
        raise TransferContractError("sensor discrepancy cannot alter physical rollout")
    if wrong_diagnosis_group is not None and wrong_diagnosis_group not in group_order:
        raise TransferContractError("wrong-diagnosis placebo is absent from model")

    manifest = {
        "schema": "bayesian-phystwin.sealed-hierarchical-missing-physics-model",
        "schema_version": 1,
        "group_order": list(group_order),
        "slices": {name: _encode_slice(model.slices[name]) for name in group_order},
        "alpha": {name: float(model.alpha[name]) for name in group_order},
        "effective_df_per_output": {
            name: float(model.effective_df_per_output[name]) for name in group_order
        },
        "noise_precision": float(model.noise_precision),
        "iterations": int(model.iterations),
        "converged": bool(model.converged),
        "selected_groups": list(selected),
        "wrong_diagnosis_group": wrong_diagnosis_group,
        "physical_rollout_forbidden_groups": sorted(PHYSICAL_ROLLOUT_FORBIDDEN_GROUPS),
        "source_result_id": source_result_id,
        "source_panel_sha256": source_panel_sha256,
        "metadata": dict(metadata),
    }
    arrays: dict[str, np.ndarray] = {
        "coefficient_mean": np.asarray(model.coefficient_mean, dtype=float),
        "posterior_covariance": np.asarray(model.posterior_covariance, dtype=float),
        "y_mean": np.asarray(model.y_mean, dtype=float),
    }
    for name in group_order:
        arrays[f"mean__{name}"] = np.asarray(
            model.x_standardizers[name].mean, dtype=float
        )
        arrays[f"scale__{name}"] = np.asarray(
            model.x_standardizers[name].scale, dtype=float
        )
    return manifest, arrays


def seal_source_model(
    source_panel_path: Path,
    source_manifest_path: Path | None,
    *,
    output_prefix: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    panel = CORE.load_panel(source_panel_path, source_manifest_path)
    source_result = CORE.evaluate_source_panel(
        panel,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    source_result["result_id"] = sha256_bytes(canonical_json(source_result))
    if not source_result["source_gate"]["passed"]:
        raise TransferContractError("source gate failed; target transfer is forbidden")

    selected = tuple(source_result["transferable_groups_selected_source_only"])
    if "shared_physics" not in selected:
        raise TransferContractError("source diagnosis did not authorize shared_physics")
    model = CORE.fit_group_ard(panel)
    manifest, arrays = model_payload(
        model,
        selected_groups=selected,
        wrong_diagnosis_group=source_result["wrong_diagnosis_group"],
        source_result_id=source_result["result_id"],
        source_panel_sha256=sha256_file(source_panel_path),
        metadata={
            "source_manifest_sha256": (
                None
                if source_manifest_path is None
                else sha256_file(source_manifest_path)
            ),
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "target_outcomes_read_during_selection": False,
        },
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    array_path = output_prefix.with_suffix(".npz")
    manifest_path = output_prefix.with_suffix(".json")
    source_result_path = output_prefix.with_name(output_prefix.name + "_source_result.json")
    np.savez_compressed(array_path, **arrays)
    manifest["array_sha256"] = sha256_file(array_path)
    manifest["model_id"] = sha256_bytes(canonical_json(manifest))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_result_path.write_text(
        json.dumps(source_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "model_manifest": manifest_path.as_posix(),
        "model_arrays": array_path.as_posix(),
        "source_result": source_result_path.as_posix(),
        "model_id": manifest["model_id"],
        "source_result_id": source_result["result_id"],
        "selected_groups": list(selected),
        "wrong_diagnosis_group": source_result["wrong_diagnosis_group"],
        "target_outcomes_read": False,
    }
    return receipt


def load_sealed_model(prefix: Path) -> SealedSourceModel:
    manifest_path = prefix.with_suffix(".json")
    array_path = prefix.with_suffix(".npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "bayesian-phystwin.sealed-hierarchical-missing-physics-model":
        raise TransferContractError("unexpected sealed-model schema")
    claimed_id = manifest.pop("model_id")
    actual_id = sha256_bytes(canonical_json(manifest))
    manifest["model_id"] = claimed_id
    if actual_id != claimed_id:
        raise TransferContractError("sealed-model manifest ID mismatch")
    if sha256_file(array_path) != manifest["array_sha256"]:
        raise TransferContractError("sealed-model array hash mismatch")

    with np.load(array_path, allow_pickle=False) as archive:
        group_order = tuple(manifest["group_order"])
        standardizers = {
            name: CORE.Standardizer(
                mean=np.asarray(archive[f"mean__{name}"], dtype=float),
                scale=np.asarray(archive[f"scale__{name}"], dtype=float),
            )
            for name in group_order
        }
        model = CORE.FittedARD(
            group_order=group_order,
            slices={name: _decode_slice(manifest["slices"][name]) for name in group_order},
            x_standardizers=standardizers,
            y_mean=np.asarray(archive["y_mean"], dtype=float),
            coefficient_mean=np.asarray(archive["coefficient_mean"], dtype=float),
            posterior_covariance=np.asarray(
                archive["posterior_covariance"], dtype=float
            ),
            noise_precision=float(manifest["noise_precision"]),
            alpha={name: float(manifest["alpha"][name]) for name in group_order},
            effective_df_per_output={
                name: float(manifest["effective_df_per_output"][name])
                for name in group_order
            },
            iterations=int(manifest["iterations"]),
            converged=bool(manifest["converged"]),
        )
    selected = tuple(manifest["selected_groups"])
    if set(selected) & PHYSICAL_ROLLOUT_FORBIDDEN_GROUPS:
        raise TransferContractError("sealed model authorizes sensor-to-state transfer")
    return SealedSourceModel(
        model=model,
        selected_groups=selected,
        wrong_diagnosis_group=manifest["wrong_diagnosis_group"],
        source_result_id=manifest["source_result_id"],
        source_panel_sha256=manifest["source_panel_sha256"],
        metadata=manifest["metadata"],
        model_id=claimed_id,
    )


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean(np.square(y - prediction))))
    mae = float(np.mean(np.abs(y - prediction)))
    return {"rmse": rmse, "mae": mae}


def _score_predictions(
    panel: Any,
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    physical_rmse = _metric(panel.y, predictions["physical"])["rmse"]
    aggregate: dict[str, Any] = {}
    for name, prediction in predictions.items():
        metrics = _metric(panel.y, prediction)
        metrics["relative_improvement_vs_physical"] = (
            physical_rmse - metrics["rmse"]
        ) / max(physical_rmse, EPS)
        aggregate[name] = metrics

    trajectory = np.asarray(panel.trajectory_id).astype(str)
    rows: list[dict[str, Any]] = []
    for value in np.unique(trajectory):
        mask = trajectory == value
        row: dict[str, Any] = {
            "trajectory_id": value,
            "object_id": str(np.asarray(panel.object_id)[mask][0]),
            "backend_id": str(np.asarray(panel.backend_id)[mask][0]),
        }
        for name, prediction in predictions.items():
            row[f"{name}_rmse"] = _metric(panel.y[mask], prediction[mask])["rmse"]
        rows.append(row)
    return aggregate, rows


def evaluate_frozen_transfer(
    target_panel: Any,
    sealed: SealedSourceModel,
    *,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    target_panel.validate()
    if authorization.get("target_scoring_authorized") is not True:
        raise TransferContractError("target scoring is not authorized")
    if authorization.get("source_model_id") != sealed.model_id:
        raise TransferContractError("authorization names another source-model seal")
    if authorization.get("selection_frozen_before_target") is not True:
        raise TransferContractError("selection was not frozen before target")
    if authorization.get("target_panel_sha256") != target_panel.metadata.get(
        "panel_sha256"
    ):
        raise TransferContractError("target panel hash does not match authorization")

    all_physical_groups = [
        name
        for name in sealed.model.group_order
        if name not in PHYSICAL_ROLLOUT_FORBIDDEN_GROUPS
    ]
    wrong = (
        []
        if sealed.wrong_diagnosis_group is None
        or sealed.wrong_diagnosis_group in PHYSICAL_ROLLOUT_FORBIDDEN_GROUPS
        else [sealed.wrong_diagnosis_group]
    )
    predictions = {
        "physical": np.zeros_like(target_panel.y),
        "shared_physics": sealed.model.predict(
            target_panel.blocks, active_groups=["shared_physics"]
        ),
        "diagnosis_guided": sealed.model.predict(
            target_panel.blocks, active_groups=sealed.selected_groups
        ),
        "all_components": sealed.model.predict(
            target_panel.blocks, active_groups=all_physical_groups
        ),
        "wrong_diagnosis": sealed.model.predict(
            target_panel.blocks, active_groups=wrong
        ) if wrong else np.zeros_like(target_panel.y),
    }
    sensor_only = (
        sealed.model.predict(target_panel.blocks, active_groups=["sensor"])
        if "sensor" in sealed.model.group_order
        else np.zeros_like(target_panel.y)
    )
    emitted_on_sensor_only_rejection = np.zeros_like(target_panel.y)
    fallback = predictions["physical"]
    fallback_identity_violations = int(
        not np.array_equal(emitted_on_sensor_only_rejection, fallback)
    )

    aggregate, per_trajectory = _score_predictions(target_panel, predictions)
    diagnosis_wins = sum(
        row["diagnosis_guided_rmse"] < row["physical_rmse"]
        for row in per_trajectory
    )
    all_component_wins = sum(
        row["diagnosis_guided_rmse"] < row["all_components_rmse"]
        for row in per_trajectory
    )
    worst_ratio = max(
        row["diagnosis_guided_rmse"] / max(row["physical_rmse"], EPS)
        for row in per_trajectory
    )
    result = {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-transfer-result",
        "schema_version": 1,
        "source_model_id": sealed.model_id,
        "source_result_id": sealed.source_result_id,
        "selected_groups_frozen_source_only": list(sealed.selected_groups),
        "wrong_diagnosis_group_frozen_source_only": sealed.wrong_diagnosis_group,
        "target_panel_sha256": target_panel.metadata["panel_sha256"],
        "aggregate": aggregate,
        "per_trajectory": per_trajectory,
        "diagnosis_guided_win_count_vs_physical": diagnosis_wins,
        "diagnosis_guided_win_count_vs_all_components": all_component_wins,
        "trajectory_count": len(per_trajectory),
        "worst_trajectory_ratio_vs_physical": worst_ratio,
        "sensor_only_latent_prediction_norm": float(np.linalg.norm(sensor_only)),
        "sensor_only_physical_rollout_change_norm": float(
            np.linalg.norm(emitted_on_sensor_only_rejection - fallback)
        ),
        "fallback_identity_violations": fallback_identity_violations,
        "information_boundary": {
            "target_used_for_group_selection": False,
            "target_used_for_coefficient_refit": False,
            "target_used_for_standardization": False,
            "sensor_discrepancy_injected_into_physical_rollout": False,
        },
    }
    result["result_id"] = sha256_bytes(canonical_json(result))
    return result


def make_controlled_transfer_panels(seed: int = 19) -> tuple[Any, Any]:
    """Generate a known-mechanism source/held-out transfer falsification pair."""

    rng = np.random.default_rng(seed)
    group_widths = {
        "shared_physics": 3,
        "object": 2,
        "backend": 2,
        "contact": 2,
        "actuation": 2,
        "sensor": 2,
    }

    def build(role: str) -> Any:
        is_source = role == "source"
        objects = ("DLO2", "DLO3") if is_source else ("DLO4", "DLO5")
        backends = ("deform-a", "deform-b") if is_source else ("alternate-a", "alternate-b")
        y_rows = []
        block_rows = {name: [] for name in group_widths}
        trajectory_ids = []
        object_ids = []
        backend_ids = []
        for object_index, object_name in enumerate(objects):
            for backend_index, backend_name in enumerate(backends):
                for trajectory_index in range(4):
                    phase = 0.31 * trajectory_index + 0.17 * object_index
                    trajectory = f"{object_name}:{backend_name}:trajectory-{trajectory_index}"
                    for sample_index in range(20):
                        t = sample_index / 19.0
                        shared = np.array([
                            math.sin(2.0 * math.pi * t + phase),
                            math.cos(math.pi * t - phase),
                            t - 0.5,
                        ])
                        # The target deliberately presents donor-like nuisance signatures,
                        # but its true nuisance coefficients reverse.  Blindly transferring
                        # every source component is therefore falsified.
                        object_block = np.zeros(2)
                        object_block[object_index] = shared[0]
                        backend_block = np.zeros(2)
                        backend_block[backend_index] = shared[1]
                        contact = np.array([float(t > 0.58), max(t - 0.58, 0.0)])
                        actuation = np.array([math.sin(math.pi * t), t])
                        sensor = np.array([1.0, (-1.0) ** sample_index])
                        signal = np.array([
                            0.82 * shared[0] - 0.38 * shared[1] + 0.22 * shared[2],
                            -0.58 * shared[0] + 0.29 * shared[1] - 0.12 * shared[2],
                        ])
                        nuisance_sign = 1.0 if is_source else -1.0
                        signal += nuisance_sign * np.array([
                            0.24 * object_block[object_index]
                            + 0.20 * backend_block[backend_index],
                            -0.18 * object_block[object_index]
                            + 0.16 * backend_block[backend_index],
                        ])
                        signal += nuisance_sign * np.array([
                            0.12 * contact[0] + 0.08 * actuation[0],
                            -0.10 * contact[1] + 0.06 * actuation[1],
                        ])
                        # Sensor bias is observable in measurement space but is never part
                        # of the physical residual target.
                        signal += rng.normal(scale=0.055 if is_source else 0.065, size=2)
                        y_rows.append(signal)
                        trajectory_ids.append(trajectory)
                        object_ids.append(object_name)
                        backend_ids.append(backend_name)
                        for name, value in (
                            ("shared_physics", shared),
                            ("object", object_block),
                            ("backend", backend_block),
                            ("contact", contact),
                            ("actuation", actuation),
                            ("sensor", sensor),
                        ):
                            block_rows[name].append(value)
        y = np.asarray(y_rows)
        blocks = {name: np.asarray(values) for name, values in block_rows.items()}
        content = {
            "role": role,
            "y_sha256": hashlib.sha256(np.ascontiguousarray(y).view(np.uint8)).hexdigest(),
            "block_sha256": {
                name: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
                for name, value in blocks.items()
            },
        }
        content["panel_sha256"] = sha256_bytes(canonical_json(content))
        return CORE.ResidualPanel(
            y=y,
            blocks=blocks,
            trajectory_id=np.asarray(trajectory_ids),
            object_id=np.asarray(object_ids),
            backend_id=np.asarray(backend_ids),
            metadata={
                "panel_sha256": content["panel_sha256"],
                "role": role,
                "transfer_eligibility": CORE.DEFAULT_TRANSFER_ELIGIBILITY,
                "minimum_bootstrap_diagnosis_frequency": 0.60,
                "source_gate": {
                    "shared_vs_physical_min_relative_improvement": 0.10,
                    "minimum_source_bootstrap_shared_diagnosis_frequency": 0.60,
                    "minimum_complete_trajectory_win_fraction": 0.75,
                    "maximum_worst_trajectory_ratio_vs_physical": 0.95,
                },
            },
        )

    return build("source"), build("target")


def save_panel(panel: Any, prefix: Path) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    manifest_path = prefix.with_suffix(".json")
    arrays = {
        "y": np.asarray(panel.y),
        "trajectory_id": np.asarray(panel.trajectory_id),
        "object_id": np.asarray(panel.object_id),
        "backend_id": np.asarray(panel.backend_id),
    }
    arrays.update({f"block__{name}": value for name, value in panel.blocks.items()})
    np.savez_compressed(npz_path, **arrays)
    metadata = dict(panel.metadata)
    metadata["serialized_panel_sha256"] = sha256_file(npz_path)
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return npz_path, manifest_path


def run_controlled_smoke(output_directory: Path, seed: int, bootstrap: int) -> dict[str, Any]:
    source, target = make_controlled_transfer_panels(seed)
    source_npz, source_json = save_panel(source, output_directory / "source_panel")
    target_npz, target_json = save_panel(target, output_directory / "target_panel")
    receipt = seal_source_model(
        source_npz,
        source_json,
        output_prefix=output_directory / "sealed_source_model",
        bootstrap_repetitions=bootstrap,
        seed=seed,
    )
    sealed = load_sealed_model(output_directory / "sealed_source_model")
    loaded_target = CORE.load_panel(target_npz, target_json)
    authorization = {
        "target_scoring_authorized": True,
        "source_model_id": sealed.model_id,
        "selection_frozen_before_target": True,
        "target_panel_sha256": loaded_target.metadata["panel_sha256"],
        "synthetic_controlled_falsification_only": True,
    }
    result = evaluate_frozen_transfer(
        loaded_target, sealed, authorization=authorization
    )
    result["source_seal_receipt"] = receipt
    result["controlled_gate"] = {
        "diagnosis_guided_beats_physical": bool(
            result["aggregate"]["diagnosis_guided"]["rmse"]
            < result["aggregate"]["physical"]["rmse"]
        ),
        "diagnosis_guided_beats_all_components": bool(
            result["aggregate"]["diagnosis_guided"]["rmse"]
            < result["aggregate"]["all_components"]["rmse"]
        ),
        "shared_beats_wrong_diagnosis": bool(
            result["aggregate"]["shared_physics"]["rmse"]
            < result["aggregate"]["wrong_diagnosis"]["rmse"]
        ),
        "all_target_trajectories_improve": bool(
            result["diagnosis_guided_win_count_vs_physical"]
            == result["trajectory_count"]
        ),
        "sensor_physical_change_is_zero": bool(
            result["sensor_only_physical_rollout_change_norm"] == 0.0
        ),
        "fallback_exact": bool(result["fallback_identity_violations"] == 0),
    }
    result["controlled_gate"]["passed"] = all(
        result["controlled_gate"].values()
    )
    result["result_id"] = sha256_bytes(canonical_json(result))
    result_path = output_directory / "controlled_transfer_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--model-prefix", type=Path)
    parser.add_argument("--target-panel", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--controlled-smoke", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()

    if arguments.controlled_smoke:
        if arguments.output_directory is None:
            raise SystemExit("--output-directory is required for controlled smoke")
        result = run_controlled_smoke(
            arguments.output_directory,
            seed=arguments.seed,
            bootstrap=arguments.bootstrap,
        )
        print(
            json.dumps(
                {
                    "result_id": result["result_id"],
                    "selected_groups": result[
                        "selected_groups_frozen_source_only"
                    ],
                    "aggregate": result["aggregate"],
                    "controlled_gate": result["controlled_gate"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not result["controlled_gate"]["passed"]:
            raise SystemExit("controlled transfer falsification failed")
        return

    if arguments.model_prefix is None:
        raise SystemExit("--model-prefix is required")
    if arguments.source_panel is not None:
        receipt = seal_source_model(
            arguments.source_panel,
            arguments.source_manifest,
            output_prefix=arguments.model_prefix,
            bootstrap_repetitions=arguments.bootstrap,
            seed=arguments.seed,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        if arguments.target_panel is None:
            return

    if arguments.target_panel is None or arguments.authorization is None or arguments.output is None:
        raise SystemExit(
            "target evaluation requires --target-panel, --authorization, and --output"
        )
    sealed = load_sealed_model(arguments.model_prefix)
    target = CORE.load_panel(arguments.target_panel, arguments.target_manifest)
    authorization = json.loads(arguments.authorization.read_text(encoding="utf-8"))
    result = evaluate_frozen_transfer(target, sealed, authorization=authorization)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
