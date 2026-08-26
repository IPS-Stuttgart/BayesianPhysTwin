from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_covariance_source_producer_v1 as producer
import bayesian_phystwin_experiments.deform360_covariance_source_scoring_v1 as scoring
from bayesian_phystwin.deform360_registered_residual_history_v1 import (
    ResidualHistorySourceProvenanceV1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz
from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
    SOURCE_ROSTER,
)

REVISION = "1" * 40
ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts/science/run_deform360_covariance_source_scoring_v1.py"


def _sha(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _runtime() -> dict[str, Any]:
    return {
        "implementation_revision": REVISION,
        "distribution": {"name": "bayesian-phystwin", "version": "test"},
        "environment": {
            "byteorder": "little",
            "machine": "test",
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "system": "Linux",
        },
        "numerical_runtime": {
            "float64_epsilon": 2.220446049250313e-16,
            "numpy_version": "2.0.0",
        },
        "runtime_id": _sha("runtime"),
    }


def _unit(tmp_path: Path, index: int) -> producer.CovarianceSourceUnitInputsV1:
    object_id, episode, stratum = SOURCE_ROSTER[index]
    physical = tmp_path / f"physical-{index}.npz"
    visual_a = tmp_path / f"visual-a-{index}.npz"
    visual_b = tmp_path / f"visual-b-{index}.npz"
    metric_a = tmp_path / f"metric-a-{index}.npz"
    metric_b = tmp_path / f"metric-b-{index}.npz"
    for path in (physical, visual_a, visual_b, metric_a, metric_b):
        path.write_bytes(path.name.encode("ascii"))
    return producer.CovarianceSourceUnitInputsV1(
        object_id=object_id,
        episode=episode,
        stratum=stratum,
        raw_prefix_range_half_open=(100, 158),
        physical_mode="warp_twin",
        physical_archive_path=physical,
        visual_inputs=(
            ("provider-a", visual_a, metric_a),
            ("provider-b", visual_b, metric_b),
        ),
        reserved_scoring_camera_ids=("score-a", "score-b"),
        source_artifacts={
            f"prefix/{object_id}.npz": {
                "path": f"source/{object_id}/prefix.npz",
                "sha256": _sha(f"prefix-{index}"),
                "size_bytes": 1,
            }
        },
    )


def _provenance(index: int) -> ResidualHistorySourceProvenanceV1:
    return ResidualHistorySourceProvenanceV1(
        source_inventory_id=_sha("inventory"),
        provider_reconstruction_id=_sha(f"provider-reconstruction-{index}"),
        scoring_reconstruction_id=_sha(f"scoring-reconstruction-{index}"),
        provider_implementation_revision=REVISION,
        scoring_implementation_revision=REVISION,
        provider_configuration_id=_sha(f"provider-configuration-{index}"),
        scoring_configuration_id=_sha(f"scoring-configuration-{index}"),
        provider_camera_family_ids=("provider-a", "provider-b"),
        scoring_camera_family_ids=("score-a", "score-b"),
        provider_input_artifact_ids=tuple(
            sorted(
                (
                    _sha(f"provider-{index}-a"),
                    _sha(f"provider-{index}-b"),
                )
            )
        ),
        scoring_input_artifact_ids=tuple(
            sorted(
                (
                    _sha(f"scoring-plan-{index}-a"),
                    _sha(f"scoring-plan-{index}-b"),
                )
            )
        ),
        metadata={"source_suffix_used": False},
    )


def _prediction_arrays(
    *, exact_fallback: bool
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mean = np.zeros((18, 128, 3), dtype=np.float64)
    covariance = np.zeros((18, 128, 3, 3), dtype=np.float64)
    if not exact_fallback:
        covariance[:] = 1e-4 * np.eye(3, dtype=np.float64)
    arrays = {
        "mean_m": mean,
        "covariance_m2": covariance,
        "residual_history_m": np.zeros((58, 128, 3), dtype=np.float64),
        "residual_valid": np.zeros((58, 128), dtype=np.bool_),
        "observation_covariance_m2": np.zeros((58, 128, 3, 3), dtype=np.float64),
        "prior_reliability": np.zeros((58, 128), dtype=np.float64),
    }
    return arrays, {
        "accepted": not exact_fallback,
        "decision": {
            "fallback_reasons": (
                ["insufficient-per-track-support"] if exact_fallback else []
            )
        },
        "diagnostic_code": (
            "insufficient-per-track-support" if exact_fallback else "accepted"
        ),
        "exact_fallback": exact_fallback,
    }


def _build_panel(
    tmp_path: Path,
    *,
    fallback_indices: frozenset[int] = frozenset(),
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    runtime = _runtime()
    panel = tmp_path / "panel"
    (panel / "unit-artifacts").mkdir(parents=True)
    manifests: list[dict[str, Any]] = []
    for index in range(len(SOURCE_ROSTER)):
        unit = _unit(tmp_path, index)
        arrays, metadata = _prediction_arrays(exact_fallback=index in fallback_indices)
        metadata["provenance"] = _provenance(index).descriptor()
        directory = (
            panel
            / "unit-artifacts"
            / f"{index:02d}-{unit.object_id}-ep{unit.episode:04d}"
        )
        manifests.append(
            producer._publish_unit_artifact(
                directory,
                unit=unit,
                arrays=arrays,
                metadata=metadata,
                runtime=runtime,
                source_inventory_id=_sha("inventory"),
            )
        )
    batch, record_digests = producer._publish_records_and_batch(
        panel,
        unit_manifests=manifests,
        runtime=runtime,
    )
    manifest_digests = {
        f"{index:02d}": producer._sha256_file(
            panel
            / "unit-artifacts"
            / (f"{index:02d}-{SOURCE_ROSTER[index][0]}-ep{SOURCE_ROSTER[index][1]:04d}")
            / "prediction-manifest.json"
        )
        for index in range(len(SOURCE_ROSTER))
    }
    receipt_identity = {
        "schema": producer.PANEL_RECEIPT_SCHEMA,
        "schema_version": producer.PANEL_RECEIPT_VERSION,
        "status": "source-prediction-barrier-sealed",
        "software_protocol_id": producer.SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": producer.PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": producer.CROSSREPO_BINDING_ID,
        "source_inventory_id": _sha("inventory"),
        "runtime_id": runtime["runtime_id"],
        "implementation_revision": REVISION,
        "upstream_execution_receipt_id": producer.UPSTREAM_EXECUTION_RECEIPT_ID,
        "prediction_batch_id": batch["batch_id"],
        "prediction_batch_file_sha256": producer._sha256_file(
            panel / "source-prediction-batch.json"
        ),
        "prediction_record_count": 100,
        "unit_artifact_count": 10,
        "candidate_unit_count": 10 - len(fallback_indices),
        "exact_fallback_unit_count": len(fallback_indices),
        "technical_failure_count": 0,
        "unit_manifest_file_sha256": manifest_digests,
        "record_file_sha256": record_digests,
        "information_boundary": dict(producer._INFORMATION_BOUNDARY),
        "source_suffix_scoring_authorized": False,
        "confirmation_prediction_authorized": False,
        "confirmation_outcome_opening_authorized": False,
        "claim_authorized": False,
    }
    receipt = {
        **receipt_identity,
        "receipt_id": producer.content_id(receipt_identity),
    }
    producer._write_json_once(panel / "source-panel-receipt.json", receipt)
    assert producer.validate_covariance_source_panel_v1(panel) == receipt
    return panel, batch, manifests


def _array_descriptor(
    value: np.ndarray,
    *,
    member: str,
    units: str,
) -> dict[str, Any]:
    return {
        "member": member,
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": scoring._array_sha256(value),
        "units": units,
    }


def _build_observations(
    tmp_path: Path,
    *,
    batch: dict[str, Any],
    manifests: list[dict[str, Any]],
    technical_indices: frozenset[int] = frozenset(),
    error_m: float = 0.010,
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "source-observations"
    (root / "inputs").mkdir(parents=True)
    (root / "outcomes").mkdir()
    rows: list[dict[str, Any]] = []
    selected = batch["scoring_prediction_by_source_unit"]
    for index, ((object_id, episode, stratum), manifest) in enumerate(
        zip(SOURCE_ROSTER, manifests, strict=True)
    ):
        provenance = manifest["provenance"]
        suffix_inputs: list[dict[str, Any]] = []
        for camera in ("score-a", "score-b"):
            path = root / "inputs" / f"{index:02d}-{camera}.bin"
            path.write_bytes(f"score-{index}-{camera}".encode("ascii"))
            suffix_inputs.append(
                {
                    "role": f"{camera}-future-rgb",
                    "path": path.relative_to(root).as_posix(),
                    "sha256": scoring._sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        suffix_inputs.sort(key=lambda row: row["role"])
        common = {
            "object_id": object_id,
            "episode": episode,
            "stratum": stratum,
            "prediction_id": selected[f"{object_id}#{episode}"],
            "unit_manifest_id": manifest["manifest_id"],
            "scoring_reconstruction_id": provenance["scoring_reconstruction_id"],
            "scoring_configuration_id": provenance["scoring_configuration_id"],
            "scoring_camera_family_ids": provenance["scoring_camera_family_ids"],
            "scoring_plan_artifact_ids": provenance["scoring_input_artifact_ids"],
            "source_suffix_input_artifacts": suffix_inputs,
            "future_range_half_open": [58, 76],
        }
        if index in technical_indices:
            rows.append(
                {
                    **common,
                    "disposition": "technical_failure",
                    "technical_failure_reason": "scoring-reconstruction-failed",
                    "artifact": None,
                    "arrays": None,
                }
            )
            continue
        mean = np.zeros((18, 128, 3), dtype=np.float64)
        observation = np.full_like(mean, error_m)
        valid = np.ones(mean.shape[:2], dtype=np.bool_)
        archive = root / "outcomes" / f"{index:02d}.npz"
        write_deterministic_npz(
            archive,
            {"observation_m": observation, "valid": valid},
        )
        rows.append(
            {
                **common,
                "disposition": "observed",
                "technical_failure_reason": None,
                "artifact": {
                    "path": archive.relative_to(root).as_posix(),
                    "sha256": scoring._sha256_file(archive),
                    "size_bytes": archive.stat().st_size,
                },
                "arrays": {
                    "observation_m": _array_descriptor(
                        observation,
                        member="observation_m",
                        units="m",
                    ),
                    "valid": _array_descriptor(
                        valid,
                        member="valid",
                        units="dimensionless",
                    ),
                },
            }
        )
    observations = scoring.seal_source_observations_v1(
        {
            "schema": scoring.OBSERVATIONS_SCHEMA,
            "schema_version": scoring.SCHEMA_VERSION,
            "software_protocol_id": scoring.SOFTWARE_PROTOCOL_ID,
            "paper_protocol_id": scoring.PAPER_PROTOCOL_ID,
            "prediction_batch_id": batch["batch_id"],
            "scoring_implementation_revision": REVISION,
            "information_boundary": dict(scoring._OBSERVATION_BOUNDARY),
            "rows": rows,
        }
    )
    path = root / "source-observations.json"
    scoring._write_json_once(path, observations)
    return root, path, observations


def _score(
    *,
    panel: Path,
    observations_path: Path,
    observation_root: Path,
    forbidden: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return scoring.score_covariance_source_panel_v1(
        panel_root=panel,
        source_observations_path=observations_path,
        source_observation_root=observation_root,
        forbidden_confirmation_root=forbidden,
    )


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "covariance_source_scoring_cli", CLI_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_source_panel_is_positive_with_exact_point_identity(
    tmp_path: Path,
) -> None:
    panel, batch, manifests = _build_panel(tmp_path)
    root, observations_path, _ = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    scores, decision, receipt = _score(
        panel=panel,
        observations_path=observations_path,
        observation_root=root,
        forbidden=forbidden,
    )
    assert decision["status"] == "source-positive"
    assert decision["mean_candidate_minus_reference_nll"] < 0.0
    assert decision["supported_or_exact_fallback_count"] == 10
    assert decision["all_point_means_identical"] is True
    assert decision["all_point_metrics_identical"] is True
    assert decision["confirmation_prediction_authorized"] is True
    assert decision["confirmation_outcome_opening_authorized"] is False
    assert receipt["claim_authorized"] is False
    assert all(
        row["aggregation"]
        == "equal-valid-3d-event-within-horizon-then-equal-three-horizon-mean"
        for row in scores["rows"]
    )


def test_exact_fallback_reproduces_reference_score(tmp_path: Path) -> None:
    panel, batch, manifests = _build_panel(
        tmp_path,
        fallback_indices=frozenset({0}),
    )
    root, observations_path, _ = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    scores, decision, _ = _score(
        panel=panel,
        observations_path=observations_path,
        observation_root=root,
        forbidden=forbidden,
    )
    fallback = scores["rows"][0]
    assert fallback["disposition"] == "exact_fallback"
    assert fallback["candidate_nll"] == fallback["reference_nll"]
    assert (
        fallback["horizon_scores"]["late"]["candidate_nll"]
        == (fallback["horizon_scores"]["late"]["reference_nll"])
    )
    assert decision["status"] == "source-positive"


def test_retained_scoring_failure_is_a_complete_technical_negative(
    tmp_path: Path,
) -> None:
    panel, batch, manifests = _build_panel(tmp_path)
    root, observations_path, _ = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
        technical_indices=frozenset({0}),
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    scores, decision, _ = _score(
        panel=panel,
        observations_path=observations_path,
        observation_root=root,
        forbidden=forbidden,
    )
    assert scores["rows"][0]["candidate_nll"] is None
    assert decision["status"] == "source-technical-negative"
    assert decision["confirmation_prediction_authorized"] is False
    assert decision["technical_failure_count"] == 1


def test_horizons_receive_equal_weight_independent_of_valid_event_count() -> None:
    mean = np.zeros((18, 4, 3), dtype=np.float64)
    observation = np.zeros_like(mean)
    observation[:6] = 0.002
    observation[6:12] = 0.010
    observation[12:] = 0.020
    covariance = np.repeat(
        (1e-4 * np.eye(3, dtype=np.float64))[None, None],
        18 * 4,
        axis=0,
    ).reshape(18, 4, 3, 3)
    valid = np.zeros((18, 4), dtype=np.bool_)
    valid[0, 0] = True
    valid[6:12, :2] = True
    valid[12:, :] = True
    result = scoring._score_unit(
        mean,
        covariance,
        observation,
        valid,
        exact_fallback=False,
    )
    horizon_scores = result["horizon_scores"]
    expected = (
        sum(
            horizon_scores[name]["candidate_nll"]
            for name in ("early", "middle", "late")
        )
        / 3.0
    )
    assert result["candidate_nll"] == pytest.approx(expected)
    assert [
        horizon_scores[name]["valid_event_count"]
        for name in ("early", "middle", "late")
    ] == [1, 12, 24]


def test_source_observation_archive_and_suffix_inputs_are_rehashed(
    tmp_path: Path,
) -> None:
    panel, batch, manifests = _build_panel(tmp_path)
    root, observations_path, observations = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    archive = root / observations["rows"][0]["artifact"]["path"]
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="observation archive identity changed"):
        _score(
            panel=panel,
            observations_path=observations_path,
            observation_root=root,
            forbidden=forbidden,
        )

    root2, observations_path2, observations2 = _build_observations(
        tmp_path / "second",
        batch=batch,
        manifests=manifests,
    )
    suffix = (
        root2 / observations2["rows"][0]["source_suffix_input_artifacts"][0]["path"]
    )
    suffix.write_bytes(suffix.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="suffix input artifact identity changed"):
        _score(
            panel=panel,
            observations_path=observations_path2,
            observation_root=root2,
            forbidden=forbidden,
        )


def test_provider_overlap_and_scoring_provenance_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    panel, batch, manifests = _build_panel(tmp_path)
    root, _observations_path, observations = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()

    overlap = copy.deepcopy(observations)
    row = overlap["rows"][0]
    suffix_record = row["source_suffix_input_artifacts"][0]
    suffix_path = root / suffix_record["path"]
    suffix_path.write_bytes(b"provider-0-a")
    suffix_record["sha256"] = scoring._sha256_file(suffix_path)
    suffix_record["size_bytes"] = suffix_path.stat().st_size
    overlap = scoring.seal_source_observations_v1(overlap)
    overlap_path = root / "overlap.json"
    scoring._write_json_once(overlap_path, overlap)
    with pytest.raises(ValueError, match="provider and scoring.*overlap"):
        _score(
            panel=panel,
            observations_path=overlap_path,
            observation_root=root,
            forbidden=forbidden,
        )

    root2, _path2, observations2 = _build_observations(
        tmp_path / "second",
        batch=batch,
        manifests=manifests,
    )
    mismatch = copy.deepcopy(observations2)
    mismatch["rows"][0]["scoring_reconstruction_id"] = _sha("wrong")
    mismatch = scoring.seal_source_observations_v1(mismatch)
    mismatch_path = root2 / "mismatch.json"
    scoring._write_json_once(mismatch_path, mismatch)
    with pytest.raises(ValueError, match="provenance differs"):
        _score(
            panel=panel,
            observations_path=mismatch_path,
            observation_root=root2,
            forbidden=forbidden,
        )


def test_missing_horizon_and_manifest_outside_root_fail_closed(tmp_path: Path) -> None:
    panel, batch, manifests = _build_panel(tmp_path)
    root, observations_path, observations = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    first = observations["rows"][0]
    archive = root / first["artifact"]["path"]
    with np.load(archive, allow_pickle=False) as stored:
        observation = np.asarray(stored["observation_m"])
        valid = np.asarray(stored["valid"])
    valid[:6] = False
    archive.unlink()
    write_deterministic_npz(
        archive,
        {"observation_m": observation, "valid": valid},
    )
    first["artifact"]["sha256"] = scoring._sha256_file(archive)
    first["artifact"]["size_bytes"] = archive.stat().st_size
    first["arrays"]["valid"] = _array_descriptor(
        valid,
        member="valid",
        units="dimensionless",
    )
    observations = scoring.seal_source_observations_v1(observations)
    observations_path.unlink()
    scoring._write_json_once(observations_path, observations)
    with pytest.raises(ValueError, match="no valid early events"):
        _score(
            panel=panel,
            observations_path=observations_path,
            observation_root=root,
            forbidden=forbidden,
        )
    outside = tmp_path / "outside.json"
    outside.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="outside its admitted root"):
        _score(
            panel=panel,
            observations_path=outside,
            observation_root=root,
            forbidden=forbidden,
        )


def test_observation_units_are_metric_and_content_addressed(tmp_path: Path) -> None:
    _panel, batch, manifests = _build_panel(tmp_path)
    _root, _path, observations = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    changed = copy.deepcopy(observations)
    changed["rows"][0]["arrays"]["observation_m"]["units"] = "mm"
    with pytest.raises(ValueError, match="units changed"):
        scoring.seal_source_observations_v1(changed)


def test_low_level_contract_rejects_noncanonical_values() -> None:
    with pytest.raises(TypeError, match="string-keyed mapping"):
        scoring._mapping([], name="value")
    with pytest.raises(TypeError, match="must be a sequence"):
        scoring._sequence("rows", name="value")
    with pytest.raises(ValueError, match="canonical nonempty"):
        scoring._literal(" value", name="value")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        scoring._sha256("a" * 63, name="value")
    with pytest.raises(ValueError, match="lowercase Git SHA-1"):
        scoring._revision("a" * 39, name="value")
    with pytest.raises(TypeError, match="must be an integer"):
        scoring._integer(True, name="value")
    with pytest.raises(ValueError, match="canonical relative POSIX"):
        scoring._canonical_relative_path("../value", name="value")
    with pytest.raises(ValueError, match="forbidden target path"):
        scoring._canonical_relative_path("target/value", name="value")
    descriptor = {
        "member": "value",
        "dtype": "<f8",
        "shape": [1],
        "sha256": _sha("value"),
        "units": "m",
    }
    changed = {**descriptor, "extra": True}
    with pytest.raises(ValueError, match="fields changed"):
        scoring._validate_array_descriptor(
            changed,
            name="value",
            expected_dtype="<f8",
            expected_shape=[1],
            expected_units="m",
        )
    changed = {**descriptor, "dtype": "<f4"}
    with pytest.raises(ValueError, match="representation changed"):
        scoring._validate_array_descriptor(
            changed,
            name="value",
            expected_dtype="<f8",
            expected_shape=[1],
            expected_units="m",
        )


def test_observation_manifest_validation_is_fail_closed(tmp_path: Path) -> None:
    _panel, batch, manifests = _build_panel(tmp_path)
    _root, _path, observations = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )

    def reject(changed: dict[str, Any], match: str) -> None:
        with pytest.raises((TypeError, ValueError), match=match):
            scoring.seal_source_observations_v1(changed)

    changed = copy.deepcopy(observations)
    changed["schema"] = "wrong"
    reject(changed, "schema changed")
    changed = copy.deepcopy(observations)
    changed["software_protocol_id"] = _sha("wrong")
    reject(changed, "protocol identity changed")
    changed = copy.deepcopy(observations)
    changed["information_boundary"]["target_informed_selection_used"] = True
    reject(changed, "information boundary changed")
    changed = copy.deepcopy(observations)
    changed["rows"].pop()
    reject(changed, "all ten source units")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["extra"] = True
    reject(changed, "row 0 fields changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["episode"] = 99
    reject(changed, "roster or order changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["scoring_camera_family_ids"] = ["score-a"]
    reject(changed, "camera family roster changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["scoring_plan_artifact_ids"] = [_sha("one")]
    reject(changed, "plan artifact roster changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["source_suffix_input_artifacts"] = []
    reject(changed, "artifact roster is empty")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["source_suffix_input_artifacts"][0]["extra"] = True
    reject(changed, "artifact fields changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["source_suffix_input_artifacts"][0]["size_bytes"] = -1
    reject(changed, "size must be nonnegative")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["source_suffix_input_artifacts"][1]["role"] = changed["rows"][0][
        "source_suffix_input_artifacts"
    ][0]["role"]
    reject(changed, "duplicated or unordered")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["future_range_half_open"] = [58, 75]
    reject(changed, "future range changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["disposition"] = "missing"
    reject(changed, "disposition changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["disposition"] = "technical_failure"
    changed["rows"][0]["technical_failure_reason"] = "failed"
    reject(changed, "carries invented arrays")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["technical_failure_reason"] = "failed"
    reject(changed, "observed source row")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["artifact"]["extra"] = True
    reject(changed, "observation artifact fields changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["artifact"]["size_bytes"] = 0
    reject(changed, "artifact is empty")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["arrays"]["extra"] = True
    reject(changed, "array roster changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["arrays"]["observation_m"]["dtype"] = "<f4"
    reject(changed, "dtype changed")
    changed = copy.deepcopy(observations)
    changed["rows"][0]["arrays"]["observation_m"]["shape"] = [17, 128, 3]
    reject(changed, "shape changed")

    changed = copy.deepcopy(observations)
    changed["observation_set_id"] = _sha("wrong")
    with pytest.raises(ValueError, match="does not match document content"):
        scoring.validate_source_observations_v1(changed)


def test_publication_is_write_once_rehashable_and_disjoint_from_confirmation(
    tmp_path: Path,
) -> None:
    panel, batch, manifests = _build_panel(tmp_path)
    root, observations_path, _ = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    output = tmp_path / "results" / "source-score"
    output.parent.mkdir()
    receipt = scoring.publish_covariance_source_scores_v1(
        panel_root=panel,
        source_observations_path=observations_path,
        source_observation_root=root,
        forbidden_confirmation_root=forbidden,
        output_root=output,
    )
    assert scoring.validate_covariance_source_scores_v1(output) == receipt
    with pytest.raises(ValueError, match="overwrite"):
        scoring.publish_covariance_source_scores_v1(
            panel_root=panel,
            source_observations_path=observations_path,
            source_observation_root=root,
            forbidden_confirmation_root=forbidden,
            output_root=output,
        )
    with pytest.raises(ValueError, match="output overlaps"):
        scoring.publish_covariance_source_scores_v1(
            panel_root=panel,
            source_observations_path=observations_path,
            source_observation_root=root,
            forbidden_confirmation_root=forbidden,
            output_root=forbidden / "scores",
        )
    for immutable_root in (panel, root):
        with pytest.raises(ValueError, match="immutable input root"):
            scoring.publish_covariance_source_scores_v1(
                panel_root=panel,
                source_observations_path=observations_path,
                source_observation_root=root,
                forbidden_confirmation_root=forbidden,
                output_root=immutable_root / "scores",
            )


def test_cli_seals_observations_and_scores_a_complete_negative(tmp_path: Path) -> None:
    cli = _load_cli_module()
    panel, batch, manifests = _build_panel(
        tmp_path,
        fallback_indices=frozenset(range(10)),
    )
    root, _observations_path, observations = _build_observations(
        tmp_path,
        batch=batch,
        manifests=manifests,
    )
    unsealed = copy.deepcopy(observations)
    unsealed.pop("observation_set_id")
    unsealed_path = root / "unsealed.json"
    unsealed_path.write_text(json.dumps(unsealed), encoding="utf-8")
    sealed_path = root / "sealed.json"
    assert (
        cli.main(
            [
                "seal-observations",
                "--input",
                str(unsealed_path),
                "--output",
                str(sealed_path),
            ]
        )
        == 0
    )
    assert cli.main(["validate-observations", str(sealed_path)]) == 0
    forbidden = tmp_path / "confirmation"
    forbidden.mkdir()
    output = tmp_path / "cli-results" / "scores"
    output.parent.mkdir()
    assert (
        cli.main(
            [
                "score",
                "--panel-root",
                str(panel),
                "--source-observations",
                str(sealed_path),
                "--source-observation-root",
                str(root),
                "--forbidden-confirmation-root",
                str(forbidden),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert cli.main(["validate-result", str(output)]) == 0
    decision = json.loads((output / "source-decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "source-negative"
    assert decision["confirmation_prediction_authorized"] is False
