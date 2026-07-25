#!/usr/bin/env python3
"""Audit exact-persistence integration for frame-zero reconstruction failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    PHYSICAL_ARRAY_NAMES,
    array_sha256,
    canonical_sha256,
    file_sha256,
    validate_prospective_backbone_seal,
)
from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    frame_zero_physical_policy,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_config_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = _load_json(path)
    config = payload.get("config")
    _require(isinstance(config, dict), "integration config is missing")
    digest = _canonical_config_sha256(config)
    _require(payload.get("config_sha256") == digest, "integration config changed")
    return config, digest


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]).copy() for name in stored.files}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prospective-protocol", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--original-staged-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    protocol_path = args.prospective_protocol.resolve()
    staged_root = args.staged_root.resolve()
    original_staged_root = args.original_staged_root.resolve()
    physical_root = args.physical_root.resolve()
    backbone_root = args.backbone_root.resolve()
    output = args.output.resolve()
    config, config_sha256 = _load_config(config_path)

    parity_case = str(config["admitted_path_parity_case"])
    parity_new = _load_arrays(staged_root / parity_case / "frame_zero_points.npz")
    parity_old = _load_arrays(
        original_staged_root / parity_case / "frame_zero_points.npz"
    )
    parity_manifest = _load_json(
        staged_root / parity_case / "frame_zero_reconstruction_manifest.json"
    )
    admitted_parity = bool(
        set(parity_new) == set(parity_old)
        and all(
            np.array_equal(parity_new[name], parity_old[name])
            for name in parity_new
        )
        and frame_zero_physical_policy(parity_manifest) == "automatic_twin"
        and parity_manifest.get("material_point_source") == "original-splat"
    )

    case_reports: list[dict[str, Any]] = []
    for case in config["known_postopen_failures"]:
        staged = staged_root / str(case)
        physical = physical_root / str(case)
        backbone = backbone_root / str(case)
        frame_manifest_path = staged / "frame_zero_reconstruction_manifest.json"
        frame_manifest = _load_json(frame_manifest_path)
        _require(
            frame_manifest.get("result_sha256")
            == canonical_sha256(frame_manifest, digest_key="result_sha256"),
            f"frame-zero manifest changed: {case}",
        )
        _require(
            frame_zero_physical_policy(frame_manifest) == "persistence_only",
            f"fallback policy changed: {case}",
        )
        geometry = _load_arrays(staged / "frame_zero_points.npz")
        manifest_path = physical / "physical_prediction_manifest.json"
        prediction_path = physical / "prediction.npz"
        manifest = _load_json(manifest_path)
        _require(
            manifest.get("result_sha256")
            == canonical_sha256(manifest, digest_key="result_sha256"),
            f"physical manifest changed: {case}",
        )
        arrays = _load_arrays(prediction_path)
        _require(set(arrays) == PHYSICAL_ARRAY_NAMES, f"array contract changed: {case}")
        persistence = arrays["persistence_m"]
        exact = bool(
            np.array_equal(arrays["prediction_m"], persistence)
            and np.array_equal(arrays["driven_readout_m"], persistence)
            and np.array_equal(arrays["zero_action_readout_m"], persistence)
            and np.array_equal(arrays["frame_zero_points_m"], geometry["points_m"])
            and np.count_nonzero(arrays["action_support"]) == 0
        )
        diagnostics = manifest.get("fallback_diagnostics", {})
        no_physical_artifacts = bool(
            not any(
                (physical / name).exists()
                for name in (
                    "episode_graph.npz",
                    "simulator_final_data.pkl",
                    "state_artifact.npz",
                    "warp_driven",
                    "warp_zero_action",
                )
            )
        )
        _require(
            manifest.get("physical_mode") == "persistence_fallback"
            and manifest.get("physical_admitted") is False
            and diagnostics.get("automatic_twin_attempted") is False
            and diagnostics.get("warp_attempted") is False
            and diagnostics.get("state_update_available") is False,
            f"physical fallback claim changed: {case}",
        )
        seal = _load_json(backbone / "prediction_seal.json")
        validate_prospective_backbone_seal(
            seal,
            protocol_path=protocol_path,
            case_dir=backbone,
        )
        passed = exact and no_physical_artifacts
        case_reports.append(
            {
                "case": case,
                "passed": passed,
                "material_point_count": int(len(geometry["points_m"])),
                "material_identity_sha256": array_sha256(geometry["points_m"]),
                "physical_policy": "persistence_only",
                "physical_mode": manifest["physical_mode"],
                "exact_persistence": exact,
                "action_support_nonzero_count": int(
                    np.count_nonzero(arrays["action_support"])
                ),
                "automatic_twin_or_warp_artifacts_absent": no_physical_artifacts,
                "frame_zero_manifest_sha256": file_sha256(frame_manifest_path),
                "physical_manifest_sha256": file_sha256(manifest_path),
                "physical_archive_sha256": file_sha256(prediction_path),
                "backbone_seal_sha256": file_sha256(
                    backbone / "prediction_seal.json"
                ),
            }
        )

    passed = bool(
        admitted_parity
        and len(case_reports) == 4
        and all(record["passed"] for record in case_reports)
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReconstructionFailurePersistenceFallbackAudit",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "passed": passed,
        "admitted_path": {
            "case": parity_case,
            "point_and_color_array_parity": admitted_parity,
            "physical_policy": frame_zero_physical_policy(parity_manifest),
        },
        "fallback_case_count": len(case_reports),
        "fallback_pass_count": int(sum(row["passed"] for row in case_reports)),
        "cases": case_reports,
        "scoring_boundary": config["scoring_boundary"],
        "information_boundary": config["information_boundary"],
        "inputs_sha256": {
            "config": file_sha256(config_path),
            "prospective_protocol": file_sha256(protocol_path),
        },
    }
    result["result_sha256"] = canonical_sha256(result, digest_key="result_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
