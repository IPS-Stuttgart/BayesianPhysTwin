"""Export full Bayesian-PhysTwin endpoint particles for Causal4D."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causal4d.bpt_belief import export_official_phystwin_twin_belief
from causal4d.contracts import save_contract
from causal4d.phystwin_backend import (
    OfficialPhysTwinBackend,
    OfficialPhysTwinBackendConfig,
    hidden_action_proposals,
    known_action_proposal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay every selected Bayesian-PhysTwin theta particle through O- "
            "and export particle-specific state and discrepancy beliefs."
        )
    )
    parser.add_argument("official_repo")
    parser.add_argument("case_dir")
    parser.add_argument("profile_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_npz")
    parser.add_argument("--train-end-frame", type=int)
    parser.add_argument("--parameter-particles", type=int, default=4)
    parser.add_argument(
        "--counterfactual-action-id",
        choices=(
            "known_action",
            "history_continue",
            "history_persist",
            "history_reverse",
            "history_orthogonal",
        ),
        default="known_action",
    )
    parser.add_argument("--protocol-id", default="causal4d_phystwin_v1")
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
    proposals = {
        proposal.proposal_id: proposal
        for proposal in (
            known_action_proposal(backend.controller_points),
            *hidden_action_proposals(
                backend.controller_points,
                start_frame=train_end,
            ),
        )
    }
    proposal = proposals[args.counterfactual_action_id]
    context = backend.causal_context(
        (proposal,),
        protocol_id=args.protocol_id,
    )
    belief = export_official_phystwin_twin_belief(backend, context=context)
    save_contract(args.output_npz, belief)
    print(
        json.dumps(
            {
                "artifact_id": belief.artifact_id,
                "case": backend.case_name,
                "counterfactual_action_id": proposal.proposal_id,
                "endpoint_frame": belief.endpoint_frame,
                "maximum_pairwise_endpoint_rmse_m": belief.metadata[
                    "maximum_pairwise_endpoint_rmse_m"
                ],
                "output": str(Path(args.output_npz).resolve()),
                "particle_count": len(belief.weights),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
