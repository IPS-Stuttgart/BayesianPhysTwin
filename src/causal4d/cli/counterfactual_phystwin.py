"""Run an explicit PhysTwin ``do(u_cf)`` counterfactual."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    TwinBelief,
    load_contract,
    save_contract,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.phystwin_backend import (
    OfficialPhysTwinBackend,
    OfficialPhysTwinBackendConfig,
    PhysTwinHypothesisConfig,
    hidden_action_proposals,
    known_action_proposal,
    save_rollout_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer the factual theta/phi posterior, apply do(u_cf), and either "
            "reuse the same grasp or infer a fresh counterfactual contact event."
        )
    )
    parser.add_argument("official_repo")
    parser.add_argument("case_dir")
    parser.add_argument("profile_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("twin_belief_npz")
    parser.add_argument("factual_intervention_npz")
    parser.add_argument("output_physical_npz")
    parser.add_argument(
        "--counterfactual-action-id",
        choices=(
            "known_action",
            "history_continue",
            "history_persist",
            "history_reverse",
            "history_orthogonal",
        ),
        default="history_continue",
    )
    parser.add_argument(
        "--contact-policy",
        choices=("new_contact", "same_grasp"),
        default="new_contact",
    )
    parser.add_argument("--maximum-contact-states", type=int, default=9)
    parser.add_argument("--query-output")
    parser.add_argument("--rollout-bank-output")
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--num-substeps", type=int, default=667)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nondeterministic-spring-forces", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    belief_artifact = load_contract(args.twin_belief_npz)
    factual_artifact = load_contract(args.factual_intervention_npz)
    if not isinstance(belief_artifact, TwinBelief):
        raise TypeError("twin_belief_npz must contain a TwinBelief")
    if not isinstance(factual_artifact, FactualIntervention):
        raise TypeError("factual_intervention_npz must contain a FactualIntervention")
    train_end = belief_artifact.endpoint_frame + 1
    case_dir = Path(args.case_dir)
    backend = OfficialPhysTwinBackend(
        official_repo=args.official_repo,
        final_data_path=case_dir / "final_data.pkl",
        optimal_params_path=case_dir / "optimal_params.pkl",
        checkpoint_path=args.checkpoint_path,
        baseline_trajectory_path=case_dir / "inference.pkl",
        profile_path=args.profile_path,
        train_end_frame=train_end,
        parameter_particle_count=len(belief_artifact.weights),
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
        protocol_id=belief_artifact.context.protocol_id,
    )
    query = CounterfactualQuery(
        context=context,
        controller_points_m=proposal.controller_points_m[train_end:],
        horizon_frames=backend.frame_count - train_end,
        contact_policy=args.contact_policy,
        source_factual_intervention_id=factual_artifact.artifact_id,
        metadata={
            "action_provenance": proposal.provenance,
            "future_action_observed": proposal.future_action_observed,
        },
    )
    bank, manifest = backend.build_rollout_bank(
        (proposal,),
        twin_belief=belief_artifact,
        hypothesis_config=PhysTwinHypothesisConfig(
            maximum_contact_states=args.maximum_contact_states
        ),
    )
    posterior = apply_counterfactual_operator(
        bank,
        manifest,
        belief_artifact,
        factual_artifact,
        query,
    )
    output_path = Path(args.output_physical_npz)
    query_path = (
        Path(args.query_output)
        if args.query_output
        else output_path.with_name(output_path.stem + ".query.npz")
    )
    save_contract(query_path, query)
    save_contract(output_path, posterior)
    if args.rollout_bank_output:
        save_rollout_bank(args.rollout_bank_output, bank, manifest)
    effective_components = 1.0 / float(np.sum(np.square(posterior.weights)))
    print(
        json.dumps(
            {
                "action": proposal.proposal_id,
                "contact_policy": query.contact_policy,
                "effective_components": effective_components,
                "factual_kappa_reused": posterior.metadata[
                    "factual_kappa_reused"
                ],
                "fresh_kappa_cf_sampled": posterior.metadata[
                    "fresh_kappa_cf_sampled"
                ],
                "physical_posterior": str(output_path.resolve()),
                "physical_posterior_id": posterior.artifact_id,
                "query": str(query_path.resolve()),
                "query_id": query.artifact_id,
                "represented_factual_mass": posterior.metadata[
                    "represented_factual_mass_before_renormalization"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
