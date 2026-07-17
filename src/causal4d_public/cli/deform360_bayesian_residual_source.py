from __future__ import annotations

import argparse
from pathlib import Path
import sys

from causal4d_public.deform360_bayesian_residual import (
    BayesianResidualModelConfig,
)
from causal4d_public.deform360_bayesian_residual_experiment import (
    ResidualTrainingConfig,
    load_source_residual_panel,
    load_cross_fitted_trust_scales,
    run_leave_one_object_out_smoke,
    validate_source_protocol_before_experiment,
    write_source_smoke_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one source-only Deform360 residual LOO smoke fold."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--held-object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--message-steps", type=int, default=3)
    parser.add_argument(
        "--deform360-code-root",
        type=Path,
        help="Optional Deform360 checkout providing official gripper taxel geometry.",
    )
    parser.add_argument("--controller-points-per-gripper", type=int, default=32)
    parser.add_argument(
        "--physics-root",
        type=Path,
        help="Optional root containing prediction-first PhysTwin source artifacts.",
    )
    parser.add_argument(
        "--trust-diagnosis",
        type=Path,
        help="Frozen same-object trust diagnosis supplying cross-fitted scales.",
    )
    parser.add_argument(
        "--closure-diagnosis",
        type=Path,
        help="Frozen source failure diagnosis supplying the closure gate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    validate_source_protocol_before_experiment(args.protocol)
    controller_surface_provider = None
    if args.deform360_code_root is not None:
        sys.path.insert(0, str(args.deform360_code_root.resolve()))
        from deform360.processing.control_points_stage import gripper_taxel_points

        controller_surface_provider = gripper_taxel_points
    if (args.trust_diagnosis is None) != (args.closure_diagnosis is None):
        raise ValueError(
            "trust and closure diagnoses must be supplied together"
        )
    trust_scales = None
    if args.trust_diagnosis is not None:
        trust_scales = load_cross_fitted_trust_scales(
            args.trust_diagnosis,
            args.closure_diagnosis,
        )
    episodes = load_source_residual_panel(
        args.source_root,
        maximum_node_count=args.nodes,
        neighbor_count=args.neighbors,
        controller_surface_provider=controller_surface_provider,
        controller_points_per_gripper=args.controller_points_per_gripper,
        physics_root=args.physics_root,
        physics_response_scale_by_episode=trust_scales,
    )
    result = run_leave_one_object_out_smoke(
        episodes,
        held_object_id=args.held_object,
        model_config=BayesianResidualModelConfig(
            hidden_dim=args.hidden,
            message_steps=args.message_steps,
        ),
        training_config=ResidualTrainingConfig(
            steps=args.steps,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
        ),
        device=args.device,
    )
    write_source_smoke_result(args.output, result)
    print(result["result_sha256"])


if __name__ == "__main__":
    main()
