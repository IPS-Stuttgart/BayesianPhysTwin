#!/usr/bin/env python3
"""Test fallback geometry with the frozen automatic twin and official Warp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    array_sha256,
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import PROTOCOL_ID


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), "Bayesian-PhysTwin repository is not clean")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prospective-protocol", type=Path, required=True)
    parser.add_argument("--postopen-root", type=Path, required=True)
    parser.add_argument("--original-staged-root", type=Path, required=True)
    parser.add_argument("--robust-staged-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "config schema changed")
    config = payload.get("config")
    _require(isinstance(config, dict), "config is missing")
    expected = _canonical_hash(config)
    _require(payload.get("config_sha256") == expected, "config checksum changed")
    return config, expected


def _load_sealed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("result_sha256")
        == canonical_sha256(payload, digest_key="result_sha256"),
        f"JSON checksum changed: {path}",
    )
    return payload


def _write_robust_stage(
    record: dict[str, Any],
    *,
    original_staged_root: Path,
    postopen_root: Path,
    robust_staged_root: Path,
    prospective_protocol: Path,
    prospective_config_sha256: str,
    candidate: dict[str, Any],
) -> Path:
    source = original_staged_root / record["case"]
    destination = robust_staged_root / record["case"]
    _require(not destination.exists(), "robust staged case already exists")
    destination.mkdir(parents=True)
    prefix_path = source / "prediction_prefix_manifest.json"
    prefix = _load_sealed_json(prefix_path)
    _require(
        prefix.get("protocol_id") == PROTOCOL_ID
        and prefix.get("protocol_config_sha256") == prospective_config_sha256
        and all(prefix.get(key) == value for key, value in record.items()),
        "prediction prefix identity changed",
    )
    authorization = {
        key: prefix[key]
        for key in (
            "case",
            "object_id",
            "episode_id",
            "episode_key",
            "stratum",
            "role",
        )
    }
    shutil.copy2(prefix_path, destination / prefix_path.name)
    shutil.copytree(source / "known-action", destination / "known-action")
    fallback = (
        postopen_root / record["case"] / candidate["fallback_archive_name"]
    )
    _require(fallback.is_file(), "post-open fallback archive is missing")
    geometry = destination / "frame_zero_points.npz"
    shutil.copy2(fallback, geometry)
    with np.load(geometry, allow_pickle=False) as stored:
        points = np.asarray(stored["points_m"])
        colors = np.asarray(stored["colors"])
    _require(
        points.ndim == 2
        and points.shape[1:] == (3,)
        and colors.shape == points.shape
        and len(points) >= 128
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors)),
        "fallback geometry is invalid",
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareFrameZeroReconstruction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": prospective_config_sha256,
        **authorization,
        "initializer": {
            "method": "strict-multiview-visual-hull-surface",
            "source_result_sha256": candidate["source_result_sha256"],
            "postopen_result_sha256": candidate["postopen_result_sha256"],
            "initializer_module_sha256": candidate["initializer_module_sha256"],
        },
        "material_point_count": len(points),
        "material_identity_sha256": array_sha256(points),
        "inputs_sha256": {
            "prospective_protocol": file_sha256(prospective_protocol),
            "prediction_prefix_manifest": file_sha256(prefix_path),
            "postopen_fallback": file_sha256(fallback),
        },
        "outputs_sha256": {"frame_zero_points": file_sha256(geometry)},
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest,
        digest_key="result_sha256",
    )
    path = destination / "frame_zero_reconstruction_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _run_physical_prior(
    staged_case: Path,
    *,
    repository: Path,
    prospective_protocol: Path,
    work_root: Path,
    backbone_root: Path,
    upstream_repo: Path,
    official_phystwin_repo: Path,
    official_config: Path,
    deform360_repo: Path,
    python: Path,
    device: str,
) -> tuple[int, Path]:
    log = work_root.parent / "logs" / f"{staged_case.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(repository / "scripts/remote/run_deform360_bias_aware_physical_prior.py"),
        "--repo",
        str(repository),
        "--protocol",
        str(prospective_protocol),
        "--staged-case-dir",
        str(staged_case),
        "--work-root",
        str(work_root),
        "--backbone-root",
        str(backbone_root),
        "--upstream-repo",
        str(upstream_repo),
        "--official-phystwin-repo",
        str(official_phystwin_repo),
        "--official-config",
        str(official_config),
        "--deform360-repo",
        str(deform360_repo),
        "--python",
        str(python),
        "--device",
        device,
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(deform360_repo))
    )
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode, log


def _evaluate_physical_case(
    record: dict[str, Any],
    *,
    work_root: Path,
    gate: dict[str, Any],
) -> dict[str, Any]:
    root = work_root / record["case"]
    manifest = _load_sealed_json(root / "physical_prediction_manifest.json")
    archive_path = Path(
        manifest["physical_prediction_archive"]["path"]
    ).resolve()
    _require(
        file_sha256(archive_path)
        == manifest["physical_prediction_archive"]["file_sha256"],
        "physical archive changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        initial = np.asarray(stored["frame_zero_points_m"])
        prediction = np.asarray(stored["prediction_m"])
        zero = np.asarray(stored["zero_action_readout_m"])
        driven = np.asarray(stored["driven_readout_m"])
    all_finite = bool(
        np.all(np.isfinite(prediction))
        and np.all(np.isfinite(zero))
        and np.all(np.isfinite(driven))
    )
    identity_exact = bool(np.array_equal(prediction[0], initial))
    zero_displacement = np.linalg.norm(zero - zero[:1], axis=2)
    p99 = float(np.quantile(zero_displacement, 0.99))
    maximum = float(np.max(zero_displacement))
    twin = _load_sealed_json(root / "twin_summary.json")
    passed = bool(
        manifest.get("physical_mode") == "warp_twin"
        and manifest.get("physical_admitted") is True
        and twin.get("passed") is True
        and identity_exact
        and all_finite
        and p99 <= gate["maximum_zero_action_p99_displacement_m"]
        and maximum <= gate["maximum_zero_action_displacement_m"]
    )
    return {
        **record,
        "passed": passed,
        "physical_mode": manifest.get("physical_mode"),
        "automatic_twin_passed": twin.get("passed"),
        "frame_zero_identity_exact": identity_exact,
        "all_rollout_values_finite": all_finite,
        "material_point_count": len(initial),
        "graph_vertex_count": int(twin["capacity_diagnostic"]["effective_canonical_node_count"]),
        "zero_action_p99_displacement_m": p99,
        "zero_action_maximum_displacement_m": maximum,
        "driven_endpoint_median_displacement_m": float(
            np.median(np.linalg.norm(driven[-1] - driven[0], axis=1))
        ),
        "physical_manifest_result_sha256": manifest["result_sha256"],
        "automatic_twin_result_sha256": twin["result_sha256"],
    }


def main() -> int:
    args = _parse_args()
    repository = Path(__file__).resolve().parents[2]
    code_revision = _require_clean_repository(repository)
    config, config_sha256 = _load_config(args.config.resolve())
    candidate = config["candidate"]
    module_path = (
        repository
        / "src"
        / "bayesian_phystwin"
        / "deform360_frame_zero_initializer.py"
    )
    _require(
        file_sha256(module_path) == candidate["initializer_module_sha256"],
        "source-frozen initializer changed",
    )
    postopen = _load_sealed_json(args.postopen_root / "postopen_audit.json")
    _require(
        postopen.get("postopen_gate_passed") is True
        and postopen.get("result_sha256") == candidate["postopen_result_sha256"],
        "post-open fallback result changed",
    )
    prospective = json.loads(
        args.prospective_protocol.read_text(encoding="utf-8")
    )
    prospective_sha = str(prospective["config_sha256"])
    records = sorted(config["cases"], key=lambda row: row["case"])
    _require(
        len(records) == config["gate"]["required_case_count"],
        "physical case count changed",
    )
    for root in (args.robust_staged_root, args.work_root, args.backbone_root):
        _require(not root.exists(), f"output root already exists: {root}")
        root.mkdir(parents=True)
    results = []
    for record in records:
        print(f"running {record['case']}", flush=True)
        try:
            staged = _write_robust_stage(
                record,
                original_staged_root=args.original_staged_root.resolve(),
                postopen_root=args.postopen_root.resolve(),
                robust_staged_root=args.robust_staged_root.resolve(),
                prospective_protocol=args.prospective_protocol.resolve(),
                prospective_config_sha256=prospective_sha,
                candidate=candidate,
            )
            returncode, log = _run_physical_prior(
                staged,
                repository=repository,
                prospective_protocol=args.prospective_protocol.resolve(),
                work_root=args.work_root.resolve(),
                backbone_root=args.backbone_root.resolve(),
                upstream_repo=args.upstream_repo.resolve(),
                official_phystwin_repo=args.official_phystwin_repo.resolve(),
                official_config=args.official_config.resolve(),
                deform360_repo=args.deform360_repo.resolve(),
                python=args.python.resolve(),
                device=args.device,
            )
            _require(returncode == 0, f"physical runner failed; see {log}")
            result = _evaluate_physical_case(
                record,
                work_root=args.work_root.resolve(),
                gate=config["gate"],
            )
            result["runner_log_sha256"] = file_sha256(log)
        except Exception as error:  # noqa: BLE001 - account for every case
            result = {
                **record,
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        results.append(result)
    admitted_count = sum(
        row.get("physical_mode") == "warp_twin" for row in results
    )
    passing_count = sum(bool(row["passed"]) for row in results)
    gate_passed = bool(
        admitted_count == config["gate"]["required_warp_twin_count"]
        and passing_count == len(results)
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FrameZeroFallbackPhysicalAudit",
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": config_sha256,
        "physical_gate_passed": gate_passed,
        "summary": {
            "case_count": len(results),
            "warp_twin_count": admitted_count,
            "passing_case_count": passing_count,
        },
        "cases": results,
        "provenance": {
            "bayesian_phystwin_revision": code_revision,
            "initializer_module_sha256": file_sha256(module_path),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "postopen_result_sha256": postopen["result_sha256"],
            "upstream_revision": _git_revision(args.upstream_repo.resolve()),
            "official_phystwin_revision": _git_revision(
                args.official_phystwin_repo.resolve()
            ),
            "official_config_sha256": file_sha256(args.official_config.resolve()),
            "deform360_revision": _git_revision(args.deform360_repo.resolve()),
        },
        "information_boundary": dict(config["information_boundary"]),
        "claim_boundary": config["claim_boundary"],
    }
    output["result_sha256"] = canonical_sha256(
        output,
        digest_key="result_sha256",
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"physical_gate_passed={gate_passed}")
    print(f"result_sha256={output['result_sha256']}")
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
