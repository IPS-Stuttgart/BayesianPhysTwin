#!/usr/bin/env python3
"""Complete matched zero-action source rollouts for the Gibbs twin bank."""

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
    validate_reusable_dynamics_source_selection,
)
from causal4d_public.deform360_reusable_ensemble import (
    load_reusable_ensemble_config,
)


def _parameter_label(parameters: Mapping[str, float]) -> str:
    return (
        f"y{int(parameters['init_spring_Y'])}"
        f"-drag{int(parameters['drag_damping'])}"
        f"-dash{int(parameters['dashpot_damping'])}"
    )


def _data_path(root: Path, object_id: str, episode_id: int) -> Path:
    object_dir = (
        object_id if episode_id == 1 else f"{object_id}-ep{episode_id:04d}"
    )
    return (
        root
        / object_dir
        / (
            "final_data_sparse_controller_k1_per_group_"
            "support-positive-y-to-positive-z.pkl"
        )
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
    parser.add_argument("--driven-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate_existing_zero(
    path: Path,
    *,
    driven: Mapping[str, Any],
    data_path: Path,
    split_path: Path,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not (
        result.get("passed") is True
        and result.get("source_only_smoke") is True
        and result.get("data_sha256") == sha256_file(data_path)
        and result.get("split_sha256") == sha256_file(split_path)
        and result.get("official_phystwin_revision")
        == driven.get("official_phystwin_revision")
        and result.get("config_sha256") == driven.get("config_sha256")
        and result.get("config_overrides") == driven.get("config_overrides")
    ):
        raise ValueError("existing zero-action source rollout is incompatible")
    scale = result.get("realized_actuation", {}).get(
        "controller_displacement_scale"
    )
    if scale is not None and float(scale) != 0.0:
        raise ValueError("existing source zero-action rollout moves the controller")
    trajectory_path = path.with_name("official_phystwin_trajectory.npz")
    if result.get("trajectory_sha256") != sha256_file(trajectory_path):
        raise ValueError("existing source zero-action trajectory checksum changed")
    return {
        "generated": False,
        "result_sha256": sha256_file(path),
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
    frozen = parent["config"]
    if args.episode not in ensemble["config"]["source_fit"]["episode_ids"]:
        raise ValueError("episode is outside the ensemble source partition")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    validate_reusable_dynamics_source_selection(selection, config=parent)
    official = frozen["official_phystwin"]
    if _git_revision(args.official_phystwin_repo) != official["upstream_revision"]:
        raise ValueError("official PhysTwin revision changed")
    real_config = args.official_phystwin_repo / "configs" / "real.yaml"
    if sha256_file(real_config) != official["real_config_sha256"]:
        raise ValueError("official PhysTwin config changed")
    if sha256_file(args.split_json) != official["source_split_sha256"]:
        raise ValueError("source split changed")
    data_path = _data_path(args.controller_root, frozen["object_id"], args.episode)
    eligible = [row for row in selection["candidate_table"] if row["eligible"]]
    if len(eligible) != 18:
        raise ValueError("eligible source candidate set changed")
    runner = args.repo / "scripts/remote/run_deform360_official_phystwin_smoke.py"
    records: dict[str, Any] = {}
    for row in eligible:
        parameters = row["physical_parameters"]
        label = _parameter_label(parameters)
        if label != row["candidate_label"]:
            raise ValueError("source candidate label changed")
        driven_path = (
            args.driven_root
            / f"ep{args.episode}"
            / label
            / "official_phystwin_smoke.json"
        )
        driven = json.loads(driven_path.read_text(encoding="utf-8"))
        if not (
            driven.get("passed") is True
            and driven.get("source_only_smoke") is True
            and driven.get("data_sha256") == sha256_file(data_path)
        ):
            raise ValueError("source driven candidate is incompatible")
        output_dir = args.zero_root / f"ep{args.episode}" / label
        output_path = output_dir / "official_phystwin_smoke.json"
        if output_path.exists():
            records[label] = _validate_existing_zero(
                output_path,
                driven=driven,
                data_path=data_path,
                split_path=args.split_json,
            )
            continue
        command = [
            sys.executable,
            str(runner),
            "--official-phystwin-repo",
            str(args.official_phystwin_repo),
            "--data",
            str(data_path),
            "--config",
            str(real_config),
            "--split-json",
            str(args.split_json),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--controller-radius-m",
            str(official["fixed_overrides"]["controller_radius"]),
            "--controller-max-neighbours",
            str(official["fixed_overrides"]["controller_max_neighbours"]),
            "--init-spring-y",
            str(parameters["init_spring_Y"]),
            "--drag-damping",
            str(parameters["drag_damping"]),
            "--dashpot-damping",
            str(parameters["dashpot_damping"]),
            "--controller-displacement-scale",
            "0.0",
            "--support-dynamics",
            "official-ground",
        ]
        command.append(
            "--reverse-z"
            if official["fixed_overrides"]["reverse_z"]
            else "--no-reverse-z"
        )
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"source zero-action rollout failed for {label}\n"
                f"STDOUT:\n{completed.stdout[-4000:]}\n"
                f"STDERR:\n{completed.stderr[-4000:]}"
            )
        record = _validate_existing_zero(
            output_path,
            driven=driven,
            data_path=data_path,
            split_path=args.split_json,
        )
        record["generated"] = True
        records[label] = record
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableEnsembleSourceZeroBank",
        "ensemble_config_sha256": ensemble["config_sha256"],
        "parent_config_sha256": parent["config_sha256"],
        "source_selection_result_sha256": selection["result_sha256"],
        "episode_id": args.episode,
        "candidate_count": len(records),
        "generated_count": sum(int(record["generated"]) for record in records.values()),
        "records": records,
        "information_boundary": {
            "source_rollouts_generated": True,
            "source_untouched_tails_used_for_selection": False,
            "calibration_outcomes_read": False,
            "sealed_target_read": False,
        },
        "claim_boundary": "source zero-action bank only; no transfer claim",
    }
    manifest["result_sha256"] = reusable_dynamics_result_sha256(manifest)
    manifest_path = (
        args.zero_root
        / f"ep{args.episode}"
        / "reusable_ensemble_zero_manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"source zero-bank manifest exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "episode_id": args.episode,
                "candidate_count": len(records),
                "generated_count": manifest["generated_count"],
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
