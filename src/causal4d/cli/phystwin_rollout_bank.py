"""Build a real PhysTwin rollout bank for Causal4D inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causal4d.phystwin_backend import (
    OfficialPhysTwinBackend,
    OfficialPhysTwinBackendConfig,
    PhysTwinActionProposal,
    PhysTwinHypothesisConfig,
    hidden_action_proposals,
    known_action_proposal,
    save_rollout_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Causal4D action/contact hypotheses through the official "
            "PhysTwin simulator under Bayesian-PhysTwin parameter particles."
        )
    )
    parser.add_argument("official_repo")
    parser.add_argument("case_dir")
    parser.add_argument("profile_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_npz")
    parser.add_argument(
        "--action-setting",
        choices=("known", "hidden", "ambiguous"),
        default="hidden",
    )
    parser.add_argument("--train-end-frame", type=int)
    parser.add_argument("--parameter-particles", type=int, default=4)
    parser.add_argument("--maximum-contact-states", type=int, default=12)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--num-substeps", type=int, default=667)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nondeterministic-spring-forces", action="store_true")
    return parser


def _train_end(case_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
    return int(split["train"][1])


def _action_proposals(
    setting: str,
    controller_points,
    *,
    train_end_frame: int,
) -> tuple[PhysTwinActionProposal, ...]:
    known = known_action_proposal(controller_points)
    hidden = hidden_action_proposals(
        controller_points,
        start_frame=train_end_frame,
    )
    if setting == "known":
        return (known,)
    if setting == "hidden":
        return hidden
    ambiguous_known = PhysTwinActionProposal(
        proposal_id=known.proposal_id,
        controller_points_m=known.controller_points_m,
        prior_weight=1.0,
        future_action_observed=True,
        provenance="candidate in an ambiguous finite action library",
    )
    return (ambiguous_known, *hidden)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    train_end = _train_end(case_dir, args.train_end_frame)
    backend = OfficialPhysTwinBackend(
        official_repo=args.official_repo,
        final_data_path=case_dir / "final_data.pkl",
        optimal_params_path=case_dir / "optimal_params.pkl",
        checkpoint_path=args.checkpoint_path,
        baseline_trajectory_path=case_dir / "inference.pkl",
        profile_path=args.profile_path,
        train_end_frame=train_end,
        parameter_particle_count=args.parameter_particles,
        config=OfficialPhysTwinBackendConfig(
            dt=args.dt,
            num_substeps=args.num_substeps,
            deterministic_spring_forces=not args.nondeterministic_spring_forces,
            device=args.device,
        ),
    )
    proposals = _action_proposals(
        args.action_setting,
        backend.controller_points,
        train_end_frame=train_end,
    )
    bank, manifest = backend.build_rollout_bank(
        proposals,
        hypothesis_config=PhysTwinHypothesisConfig(
            maximum_contact_states=args.maximum_contact_states
        ),
    )
    manifest["action_setting"] = args.action_setting
    save_rollout_bank(args.output_npz, bank, manifest)
    print(
        json.dumps(
            {
                "output": str(Path(args.output_npz).resolve()),
                "case": backend.case_name,
                "action_setting": args.action_setting,
                "rollout_shape": list(bank.trajectories.shape),
                "retained_parameter_mass": backend.particles.retained_probability_mass,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

