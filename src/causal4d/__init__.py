"""Controlled counterfactual benchmarks for intervention-ready world models."""

from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.evaluation import run_counterfactual_benchmark

__all__ = [
    "CounterfactualBenchmarkConfig",
    "build_protocol",
    "run_counterfactual_benchmark",
]

__version__ = "0.1.0"
