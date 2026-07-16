#!/usr/bin/env python3
"""Run trust-aligned point-MAP controls on already-open calibration episodes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    reusable_dynamics_result_sha256,
    validate_reusable_dynamics_calibration_request,
)
from causal4d_public.deform360_reusable_ensemble import (
    load_reusable_ensemble_config,
    reusable_ensemble_result_sha256,
)


def _parameter_label(parameters: Mapping[str, float]) -> str:
    return (
        f"y{int(parameters['init_spring_Y'])}"
        f"-drag{int(parameters['drag_damping'])}"
        f"-dash{int(parameters['dashpot_damping'])}"
    )


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--point-map", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _controller_data(
    root: Path,
    *,
    episode_id: int,
    parent: Mapping[str, Any],
) -> tuple[Path, Path]:
    candidates = sorted(
        (root / f"ep{episode_id}" / "controller_bundle").glob("*.meta.json")
    )
    if len(candidates) != 1:
        raise ValueError("exploratory episode needs one controller bundle")
    metadata_path = candidates[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = validate_reusable_dynamics_calibration_request(
        parent,
        object_id=parent["config"]["object_id"],
        episode_id=episode_id,
        operation="one-shot-scoring",
    )
    if metadata.get("reusable_dynamics_request") != expected:
        raise ValueError("exploratory controller request changed")
    data_path = Path(metadata["output_final_data"])
    if sha256_file(data_path) != metadata["output_final_data_sha256"]:
        raise ValueError("exploratory controller bundle checksum changed")
    return metadata_path, data_path


def _run_arm(
    *,
    runner: Path,
    official_repo: Path,
    data_path: Path,
    config_path: Path,
    split_path: Path,
    output_dir: Path,
    parameters: Mapping[str, float],
    displacement_scale: float,
    device: str,
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"exploratory rollout exists: {output_dir}")
    command = [
        sys.executable,
        str(runner),
        "--official-phystwin-repo",
        str(official_repo),
        "--data",
        str(data_path),
        "--config",
        str(config_path),
        "--split-json",
        str(split_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--controller-radius-m",
        str(fixed["controller_radius"]),
        "--controller-max-neighbours",
        str(fixed["controller_max_neighbours"]),
        "--init-spring-y",
        str(parameters["init_spring_Y"]),
        "--drag-damping",
        str(parameters["drag_damping"]),
        "--dashpot-damping",
        str(parameters["dashpot_damping"]),
        "--controller-displacement-scale",
        str(displacement_scale),
        "--support-dynamics",
        "official-ground",
        "--reusable-dynamics-calibration",
        "--report-edge-strain",
    ]
    command.append("--reverse-z" if fixed["reverse_z"] else "--no-reverse-z")
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"exploratory Warp arm failed ({completed.returncode})\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )
    result_path = output_dir / "official_phystwin_smoke.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        result.get("passed") is True
        and result.get("source_only_smoke") is False
        and result.get("reusable_dynamics_calibration") is True
    ):
        raise ValueError("exploratory rollout has another evidence scope")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "trajectory_sha256": result["trajectory_sha256"],
    }


def main() -> int:
    args = _parse_args()
    parent_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    ensemble_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_ensemble_081_v1.json"
    )
    parent = load_reusable_dynamics_config(parent_path)
    ensemble = load_reusable_ensemble_config(ensemble_path)
    validate_reusable_dynamics_calibration_request(
        parent,
        object_id=parent["config"]["object_id"],
        episode_id=args.episode,
        operation="one-shot-scoring",
    )
    point_map = json.loads(args.point_map.read_text(encoding="utf-8"))
    if not (
        point_map.get("artifact_kind")
        == "Deform360ReusableTwinSourceTrustedPointMapControl"
        and point_map.get("config_sha256") == ensemble["config_sha256"]
        and point_map.get("result_sha256")
        == reusable_ensemble_result_sha256(point_map)
        and point_map.get("information_boundary", {}).get(
            "sealed_target_episode_read"
        )
        is False
    ):
        raise ValueError("trusted point-MAP artifact is invalid")
    official = parent["config"]["official_phystwin"]
    if _git_revision(args.official_phystwin_repo) != official["upstream_revision"]:
        raise ValueError("official PhysTwin revision changed")
    real_config = args.official_phystwin_repo / "configs" / "real.yaml"
    if sha256_file(real_config) != official["real_config_sha256"]:
        raise ValueError("official PhysTwin config changed")
    if sha256_file(args.split_json) != official["source_split_sha256"]:
        raise ValueError("exploratory split changed")
    metadata_path, data_path = _controller_data(
        args.controller_root, episode_id=args.episode, parent=parent
    )
    parameters_by_label = {}
    for parameters in (
        point_map["selected_pooled_physical_parameters"],
        *point_map["selected_single_source_physical_parameters"].values(),
    ):
        parameters_by_label[_parameter_label(parameters)] = parameters
    runner = args.repo / "scripts/remote/run_deform360_official_phystwin_smoke.py"
    arms: dict[str, Any] = {}
    for label, parameters in sorted(parameters_by_label.items()):
        arms[label] = {}
        for arm, scale in (
            ("driven", float(official["driven_control_scale"])),
            ("zero", float(official["zero_control_scale"])),
        ):
            arms[label][arm] = _run_arm(
                runner=runner,
                official_repo=args.official_phystwin_repo,
                data_path=data_path,
                config_path=real_config,
                split_path=args.split_json,
                output_dir=(
                    args.output_root / f"ep{args.episode}" / label / arm
                ),
                parameters=parameters,
                displacement_scale=scale,
                device=args.device,
                fixed=official["fixed_overrides"],
            )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTrustedMapExploratoryRollouts",
        "ensemble_config_sha256": ensemble["config_sha256"],
        "parent_config_sha256": parent["config_sha256"],
        "point_map_result_sha256": point_map["result_sha256"],
        "episode_id": args.episode,
        "controller_metadata_sha256": sha256_file(metadata_path),
        "arms": arms,
        "information_boundary": {
            "previously_opened_calibration_read": True,
            "confirmatory_claim_allowed": False,
            "sealed_target_read": False,
        },
        "claim_boundary": "post hoc mechanism-development evidence only",
    }
    manifest["result_sha256"] = reusable_dynamics_result_sha256(manifest)
    manifest_path = (
        args.output_root
        / f"ep{args.episode}"
        / "trusted_map_exploratory_rollouts.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_id": args.episode,
                "physical_tuple_count": len(parameters_by_label),
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
