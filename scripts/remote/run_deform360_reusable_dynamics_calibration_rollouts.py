#!/usr/bin/env python3
"""Run the frozen matched official-Warp calibration arms for one episode."""

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
    validate_reusable_dynamics_source_selection,
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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _controller_bundle(
    root: Path,
    *,
    episode_id: int,
    config: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    candidates = sorted(
        (root / f"ep{episode_id}" / "controller_bundle").glob("*.meta.json")
    )
    if len(candidates) != 1:
        raise ValueError("calibration episode needs exactly one controller bundle")
    metadata_path = candidates[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    canonical = dict(metadata)
    observed = canonical.pop("result_sha256", None)
    if observed != reusable_dynamics_result_sha256(canonical):
        raise ValueError("controller metadata checksum mismatch")
    if metadata.get("source_only") is not False:
        raise ValueError("controller bundle is not independent calibration data")
    expected = validate_reusable_dynamics_calibration_request(
        config,
        object_id=config["config"]["object_id"],
        episode_id=episode_id,
        operation="one-shot-scoring",
    )
    if metadata.get("reusable_dynamics_request") != expected:
        raise ValueError("controller bundle uses another calibration request")
    data_path = Path(metadata["output_final_data"])
    if sha256_file(data_path) != metadata["output_final_data_sha256"]:
        raise ValueError("controller bundle checksum mismatch")
    return data_path, metadata_path, metadata


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
        raise FileExistsError(f"calibration rollout already exists: {output_dir}")
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
            f"official Warp arm failed ({completed.returncode})\n"
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
        raise ValueError("official Warp arm has the wrong evidence scope")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "trajectory_sha256": result["trajectory_sha256"],
        "controller_displacement_scale": displacement_scale,
        "p99_relative_edge_strain": result["object_edge_strain"][
            "p99_absolute_relative_strain"
        ],
    }


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    config = load_reusable_dynamics_config(config_path)
    frozen = config["config"]
    validate_reusable_dynamics_calibration_request(
        config,
        object_id=frozen["object_id"],
        episode_id=args.episode,
        operation="one-shot-scoring",
    )
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    validate_reusable_dynamics_source_selection(selection, config=config)
    official = frozen["official_phystwin"]
    if _git_revision(args.official_phystwin_repo) != official["upstream_revision"]:
        raise ValueError("official PhysTwin revision changed")
    real_config_path = args.official_phystwin_repo / "configs" / "real.yaml"
    if sha256_file(real_config_path) != official["real_config_sha256"]:
        raise ValueError("official PhysTwin real config changed")
    if sha256_file(args.split_json) != official["source_split_sha256"]:
        raise ValueError("registered frame split changed")
    data_path, controller_meta_path, controller_meta = _controller_bundle(
        args.controller_root,
        episode_id=args.episode,
        config=config,
    )
    parameters_by_label = {}
    for parameters in (
        selection["selected_pooled_physical_parameters"],
        *selection["selected_single_source_physical_parameters"].values(),
    ):
        parameters_by_label[_parameter_label(parameters)] = parameters
    pooled_label = _parameter_label(
        selection["selected_pooled_physical_parameters"]
    )
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
                config_path=real_config_path,
                split_path=args.split_json,
                output_dir=(
                    args.output_root / f"ep{args.episode}" / label / arm
                ),
                parameters=parameters,
                displacement_scale=scale,
                device=args.device,
                fixed=official["fixed_overrides"],
            )
    repeats = {}
    pooled_parameters = parameters_by_label[pooled_label]
    for arm, scale in (
        ("driven", float(official["driven_control_scale"])),
        ("zero", float(official["zero_control_scale"])),
    ):
        repeats[arm] = _run_arm(
            runner=runner,
            official_repo=args.official_phystwin_repo,
            data_path=data_path,
            config_path=real_config_path,
            split_path=args.split_json,
            output_dir=(
                args.repeat_root / f"ep{args.episode}" / pooled_label / arm
            ),
            parameters=pooled_parameters,
            displacement_scale=scale,
            device=args.device,
            fixed=official["fixed_overrides"],
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsCalibrationRollouts",
        "protocol_id": frozen["protocol_id"],
        "config_sha256": config["config_sha256"],
        "episode_id": args.episode,
        "source_selection_result_sha256": selection["result_sha256"],
        "controller_metadata_path": str(controller_meta_path),
        "controller_metadata_sha256": sha256_file(controller_meta_path),
        "controller_metadata_result_sha256": controller_meta["result_sha256"],
        "official_phystwin_revision": official["upstream_revision"],
        "real_config_sha256": official["real_config_sha256"],
        "split_sha256": official["source_split_sha256"],
        "arms": arms,
        "deterministic_repeats": repeats,
        "information_boundary": {
            "independent_calibration_outcomes_generated": True,
            "method_or_hyperparameter_changes_allowed": False,
            "target_episode_read": False,
        },
        "claim_boundary": "calibration rollouts only; no target or SOTA claim",
    }
    manifest["result_sha256"] = reusable_dynamics_result_sha256(manifest)
    manifest_path = (
        args.output_root
        / f"ep{args.episode}"
        / "reusable_dynamics_calibration_rollouts.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"calibration manifest already exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_id": args.episode,
                "parameter_tuple_count": len(parameters_by_label),
                "matched_arm_count": 2 * len(parameters_by_label),
                "deterministic_repeat_count": len(repeats),
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
