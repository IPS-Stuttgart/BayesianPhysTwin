"""Controlled counterfactual benchmarks for intervention-ready world models."""

from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.contact_evaluation import run_latent_contact_benchmark
from causal4d.contact_inference import LatentContactConfig
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TaskPosterior,
    TwinBelief,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.evaluation import run_counterfactual_benchmark
from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence
from causal4d.semantic_freshness import (
    SEMANTIC_TIMING_SCHEMA_VERSION,
    SEMANTIC_TIMING_SCOPE,
    SemanticFreshnessDecision,
    SemanticFreshnessLimits,
    SemanticTimingMetadata,
    apply_semantic_freshness_gate,
)

__all__ = [
    "CounterfactualBenchmarkConfig",
    "CounterfactualQuery",
    "FactualIntervention",
    "LatentContactConfig",
    "JointRolloutBank",
    "PhysicalPosterior",
    "SEMANTIC_TIMING_SCHEMA_VERSION",
    "SEMANTIC_TIMING_SCOPE",
    "SemanticFreshnessDecision",
    "SemanticFreshnessLimits",
    "SparseTrajectoryEvidence",
    "TaskPosterior",
    "SemanticTimingMetadata",
    "TwinBelief",
    "apply_counterfactual_operator",
    "apply_semantic_freshness_gate",
    "build_protocol",
    "run_counterfactual_benchmark",
    "run_latent_contact_benchmark",
]

__version__ = "0.3.0"
