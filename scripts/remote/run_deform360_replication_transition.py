#!/usr/bin/env python3
"""Build and fit the locked Deform360 causal contact-transition control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from causal4d_public.deform360_phystwin_feasibility import (
    WarpRopeCandidate,
    WarpRopeFeasibilityConfig,
)
from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_replication_case import (
    build_replication_warp_observation,
)
from causal4d_public.deform360_replication_contact import (
    ReplicationOpeningContactModel,
    contact_state_by_robot_axis,
    load_replication_contact_episode,
    visual_contact_schedule,
)
from causal4d_public.deform360_replication_controls import (
    ContactTransitionEpisode,
    fit_causal_contact_transition,
)
from causal4d_public.deform360_replication_fit import (
    validate_pooled_source_warp_fit,
)
from causal4d_public.deform360_replication_geometry import (
    load_replication_hull_archive,
)
from causal4d_public.deform360_replication_transition import (
    build_transition_episode_artifact,
    build_transition_fit_artifact,
    load_transition_episode_artifact,
    write_transition_artifact,
)
from causal4d_public.deform360_replication_warp import (
    OfficialWarpSparseGraphRunner,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("rollout", "fit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--split", choices=("source", "calibration"))
    parser.add_argument("--episode-id", type=int, action="append")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _cohort(protocol: dict, object_id: str) -> dict:
    matches = [
        record
        for record in protocol["config"]["cohort"]
        if record["object_id"] == object_id
    ]
    if len(matches) != 1:
        raise ValueError(f"object is not unique in the cohort: {object_id}")
    return matches[0]


def _contact_episode(root: Path, cohort: dict, object_id: str, episode_id: int):
    metadata = cohort["episodes"][str(episode_id)]
    return load_replication_contact_episode(
        root / "aligned" / object_id / f"episode_{episode_id:04d}",
        episode_id=f"{object_id}/episode_{episode_id:04d}",
        bimanual=metadata["bimanual"] == "yes",
        nonprehensile=metadata["nonprehensile"] == "yes",
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_rollouts(
    root: Path,
    cohort: dict,
    object_id: str,
    split: str,
    episode_ids: list[int],
    official_repo: Path,
    device: str,
) -> None:
    contact_model = ReplicationOpeningContactModel(
        **_load_json(root / "observations" / object_id / "contact_model.json")
    )
    pooled_fit = _load_json(root / "fits" / object_id / "pooled_source_fit.json")
    validate_pooled_source_warp_fit(pooled_fit)
    pooled_index = int(pooled_fit["selection"]["pooled_candidate_index"])
    candidate = WarpRopeCandidate(
        **pooled_fit["sealed_candidate_parameters"][str(pooled_index)]
    )
    config = WarpRopeFeasibilityConfig()
    for episode_id in episode_ids:
        output_dir = root / "transitions" / object_id
        output_json = output_dir / f"{split}_episode_{episode_id:04d}.json"
        if output_json.is_file():
            load_transition_episode_artifact(_load_json(output_json))
            print(f"{object_id} {split} episode {episode_id}: rollout valid", flush=True)
            continue
        episode = _contact_episode(root, cohort, object_id, episode_id)
        hull_payload = _load_json(
            root
            / "observations"
            / object_id
            / f"episode_{episode_id:04d}"
            / "sampled_hulls.json"
        )
        frames, hulls = load_replication_hull_archive(hull_payload)
        visual_schedule = visual_contact_schedule(episode, contact_model)
        observation = build_replication_warp_observation(
            root / "aligned" / object_id / f"episode_{episode_id:04d}",
            episode.episode_id,
            cohort["stratum"],
            frames[:1],
            hulls[:1],
            visual_schedule,
        )
        runner = OfficialWarpSparseGraphRunner(
            official_repo, observation.case, config, device=device
        )
        prediction = runner.rollout(candidate)
        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                f"pooled visual rollout is nonfinite: {object_id} episode {episode_id}"
            )
        prefix = observation.prefix_endpoint_frame
        oracle_contact = contact_state_by_robot_axis(
            episode, contact_model.tactile_group_to_robot_axis
        )[prefix:]
        transition_episode = ContactTransitionEpisode(
            episode_id=episode.episode_id,
            openings_m=episode.openings_m[prefix:],
            controller_positions_m=observation.case.controller_positions_m,
            predicted_object_positions_m=prediction,
            contact_active=oracle_contact,
            dt_seconds=observation.case.dt_seconds,
        )
        payload = build_transition_episode_artifact(
            transition_episode,
            output_dir / f"{split}_episode_{episode_id:04d}.npz",
            object_id=object_id,
            split=split,
            pooled_fit_result_sha256=pooled_fit["result_sha256"],
            pooled_candidate_index=pooled_index,
            prefix_geometry_result_sha256=hull_payload["result_sha256"],
            visual_contact_model=asdict(contact_model),
        )
        write_transition_artifact(output_json, payload)
        print(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "object_id": object_id,
                    "result_sha256": payload["result_sha256"],
                    "split": split,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _fit_transition(root: Path, protocol: dict) -> None:
    source_payloads = []
    calibration_payloads = []
    for cohort in protocol["config"]["cohort"]:
        object_id = cohort["object_id"]
        for split, destination in (
            ("source", source_payloads),
            ("calibration", calibration_payloads),
        ):
            for episode_id in cohort[f"{split}_episode_ids"]:
                destination.append(
                    _load_json(
                        root
                        / "transitions"
                        / object_id
                        / f"{split}_episode_{int(episode_id):04d}.json"
                    )
                )
    fit = fit_causal_contact_transition(
        [load_transition_episode_artifact(value) for value in source_payloads],
        [load_transition_episode_artifact(value) for value in calibration_payloads],
    )
    payload = build_transition_fit_artifact(
        fit, source_payloads, calibration_payloads
    )
    output = root / "fits" / "causal_contact_transition_fit.json"
    write_transition_artifact(output, payload)
    print(
        json.dumps(
            {
                "calibration_metrics": payload["calibration_metrics"],
                "result_sha256": payload["result_sha256"],
                "source_metrics": payload["source_metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    protocol = load_deform360_replication_protocol(args.protocol)
    root = args.data_root.resolve()
    if args.phase == "fit":
        if args.object_id is not None or args.split is not None or args.episode_id:
            raise ValueError("global transition fit does not accept episode selectors")
        _fit_transition(root, protocol)
        return
    if args.object_id is None or args.split is None:
        raise ValueError("transition rollout requires --object-id and --split")
    if args.official_phystwin_repo is None:
        raise ValueError("transition rollout requires --official-phystwin-repo")
    cohort = _cohort(protocol, args.object_id)
    allowed = list(map(int, cohort[f"{args.split}_episode_ids"]))
    episode_ids = args.episode_id or allowed
    if not set(episode_ids).issubset(allowed):
        raise ValueError(f"episode lies outside the {args.split} split")
    _build_rollouts(
        root,
        cohort,
        args.object_id,
        args.split,
        episode_ids,
        args.official_phystwin_repo,
        args.device,
    )


if __name__ == "__main__":
    main()
