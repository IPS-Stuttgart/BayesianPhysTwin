"""Controlled counterfactual benchmarks for intervention-ready world models."""

from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.contact_evaluation import run_latent_contact_benchmark
from causal4d.contact_inference import LatentContactConfig
from causal4d.evaluation import run_counterfactual_benchmark
from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence

__all__ = [
    "CounterfactualBenchmarkConfig",
    "LatentContactConfig",
    "JointRolloutBank",
    "SparseTrajectoryEvidence",
    "build_protocol",
    "run_counterfactual_benchmark",
    "run_latent_contact_benchmark",
]

__version__ = "0.2.0"
