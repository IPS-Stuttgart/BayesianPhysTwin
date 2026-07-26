"""Registry-backed access to non-stable Bayesian-PhysTwin commands."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One lazily imported research command outside the stable CLI surface."""

    experiment_id: str
    module: str
    function_name: str = "main"
    stability: str = "experimental"

    @property
    def summary(self) -> str:
        return self.experiment_id.replace("-", " ")

    @property
    def canonical_command(self) -> str:
        return f"bpt experiment run {self.experiment_id}"


_EXPERIMENT_MODULES: Final[dict[str, str]] = {
    "build-phystwin-cues": "bayesian_phystwin.cli.phystwin_cues",
    "calibrate-phystwin-discrepancy": "bayesian_phystwin.cli.phystwin_discrepancy",
    "export-phystwin-residuals": "bayesian_phystwin.cli.phystwin_export",
    "phystwin-refit": "bayesian_phystwin.cli.phystwin_refit",
    "build-phystwin-prefix": "bayesian_phystwin.cli.phystwin_prefix_artifact",
    "evaluate-phystwin-official": "bayesian_phystwin.cli.phystwin_official_evaluation",
    "evaluate-phystwin-priors": "bayesian_phystwin.cli.phystwin_prior_evaluation",
    "combine-phystwin-profiles": "bayesian_phystwin.cli.phystwin_joint_profile",
    "confirm-phystwin-residual": "bayesian_phystwin.cli.phystwin_confirmatory",
    "confirm-phystwin-residual-baselines": "bayesian_phystwin.cli.phystwin_baseline_confirmation",
    "confirm-phystwin-bayesian-anchor": "bayesian_phystwin.cli.phystwin_bayesian_confirmation",
    "audit-phystwin-calibration": "bayesian_phystwin.cli.phystwin_calibration",
    "confirm-phystwin-combined": "bayesian_phystwin.cli.phystwin_combined_confirmation",
    "confirm-phystwin-additional-anchor": "bayesian_phystwin.cli.phystwin_additional_confirmation",
    "confirm-phystwin-additional-bayesian": "bayesian_phystwin.cli.phystwin_additional_bayesian_confirmation",
    "compare-phystwin-additional-controls": "bayesian_phystwin.cli.phystwin_additional_control_comparison",
    "analyze-phystwin-horizon": "bayesian_phystwin.cli.phystwin_horizon_analysis",
    "analyze-phystwin-controller-sensitivity": "bayesian_phystwin.cli.phystwin_controller_sensitivity",
    "infer-phystwin-controller-bias": "bayesian_phystwin.cli.phystwin_controller_inference",
    "analyze-phystwin-spatial-modes": "bayesian_phystwin.cli.phystwin_spatial_mode_analysis",
    "compare-phystwin-graph-anchors": "bayesian_phystwin.cli.phystwin_graph_anchor_comparison",
    "evaluate-phystwin-state-injection": "bayesian_phystwin.cli.phystwin_state_injection",
    "fit-phystwin-residual-dynamics": "bayesian_phystwin.cli.phystwin_residual_dynamics",
    "fit-phystwin-residual-velocity": "bayesian_phystwin.cli.phystwin_residual_velocity",
    "fit-phystwin-shared-residual-velocity": "bayesian_phystwin.cli.phystwin_shared_residual_velocity",
    "evaluate-phystwin-pgrd": "bayesian_phystwin.cli.phystwin_pgrd_adapter",
    "calibrate-phystwin-pgrd": "bayesian_phystwin.cli.phystwin_pgrd_calibrated",
    "train-phystwin-pgrd": "bayesian_phystwin.cli.phystwin_pgrd_native",
    "train-phystwin-pgrd-unrolled": "bayesian_phystwin.cli.phystwin_pgrd_unrolled",
    "fit-phystwin-hierarchical-residual": "bayesian_phystwin.cli.phystwin_residual_shrinkage",
    "compare-phystwin-residual-scales": "bayesian_phystwin.cli.phystwin_residual_scale_comparison",
    "fit-phystwin-residual-baselines": "bayesian_phystwin.cli.phystwin_residual_baselines",
    "build-phystwin-raw-cues": "bayesian_phystwin.cli.phystwin_raw_cues",
    "build-phystwin-cotracker3-cues": "bayesian_phystwin.cli.phystwin_cotracker3_cues",
    "evaluate-phystwin-perception-cues": "bayesian_phystwin.cli.phystwin_perception_evaluation",
    "associate-phystwin-motioncrafter": "bayesian_phystwin.cli.phystwin_motioncrafter_association",
    "assimilate-phystwin-motioncrafter": "bayesian_phystwin.cli.phystwin_motioncrafter_assimilation",
    "evaluate-phystwin-motioncrafter-assimilation": "bayesian_phystwin.cli.phystwin_motioncrafter_assimilation_evaluation",
    "select-phystwin-motioncrafter-view": "bayesian_phystwin.cli.phystwin_motioncrafter_selection",
    "fit-phystwin-bayesian-anchor": "bayesian_phystwin.cli.phystwin_bayesian_anchor",
    "compare-phystwin-trajectories": "bayesian_phystwin.cli.phystwin_comparison",
    "compare-phystwin-sota": "bayesian_phystwin.cli.phystwin_sota_comparison",
    "overlay-phystwin-external-backbone": "bayesian_phystwin.cli.phystwin_external_backbone",
    "gate-phystwin-backbone-family": "bayesian_phystwin.cli.phystwin_backbone_family_gate",
    "open-phystwin-backbone-family-future": "bayesian_phystwin.cli.phystwin_backbone_family_future",
    "report-matphys-loo-sota": "bayesian_phystwin.cli.matphys_loo_sota_report",
    "gate-matphys-part-family": "bayesian_phystwin.cli.matphys_part_family_gate",
    "open-matphys-part-family-future": "bayesian_phystwin.cli.matphys_part_family_future",
    "gate-phystwin-shared-nonlinear-residual": "bayesian_phystwin.cli.phystwin_shared_nonlinear_residual",
    "gate-phystwin-canonical-triplane-residual": "bayesian_phystwin.cli.phystwin_canonical_triplane_residual",
    "build-phystwin-spring-overlay": "bayesian_phystwin.cli.phystwin_spring_overlay",
    "gate-phystwin-part-pair-source": "bayesian_phystwin.cli.phystwin_part_pair_source_gate",
    "build-phystwin-piecewise-topology": "bayesian_phystwin.cli.phystwin_piecewise_topology",
    "gate-phystwin-sparse-topology-source": "bayesian_phystwin.cli.phystwin_sparse_topology_source_gate",
    "search-phystwin-topology-field": "bayesian_phystwin.cli.phystwin_zero_order_topology",
    "gate-phystwin-zero-order-source": "bayesian_phystwin.cli.phystwin_zero_order_source_gate",
    "diagnose-phystwin-bias": "bayesian_phystwin.cli.phystwin_bias_diagnostic",
    "evaluate-deform360-online-belief": "bayesian_phystwin.cli.deform360_online_belief",
    "build-deform360-raw-camera": "bayesian_phystwin.cli.deform360_raw_camera_observation",
    "build-deform360-crossview-supplement": "bayesian_phystwin.cli.deform360_crossview_observation",
    "predict-deform360-crossview-guard": "bayesian_phystwin.cli.deform360_crossview_guard",
    "diagnose-deform360-raw-pairwise": "bayesian_phystwin.cli.deform360_raw_pairwise_correspondence_diagnostic",
    "download-deform360-selective-virtual-sensing": "bayesian_phystwin.cli.deform360_selective_virtual_sensing_download",
    "benchmark-bias-aware-belief": "bayesian_phystwin.cli.bias_aware_belief_benchmark",
    "develop-deform360-bias-aware-belief": "bayesian_phystwin.cli.deform360_bias_aware_belief_development",
    "deform360-bias-aware-prospective": "bayesian_phystwin.cli.deform360_bias_aware_prospective",
    "deform360-bias-aware-result": "bayesian_phystwin.cli.deform360_bias_aware_prospective_result",
    "fetch-phystwin-eval-data": "bayesian_phystwin.cli.phystwin_data",
    "structural-recovery-benchmark": "bayesian_phystwin.cli.structural_benchmark",
    "diagnose-phystwin-structure": "bayesian_phystwin.cli.phystwin_structural_diagnostic",
    "aggregate-phystwin-structure": "bayesian_phystwin.cli.structural_diagnostic_aggregate",
    "audit-phystwin-state-decay": "bayesian_phystwin.cli.phystwin_state_decay",
    "audit-phystwin-state-modes": "bayesian_phystwin.cli.phystwin_state_modes",
    "aggregate-phystwin-state-modes": "bayesian_phystwin.cli.phystwin_state_mode_aggregate",
}

EXPERIMENTS: Final[dict[str, ExperimentSpec]] = {
    experiment_id: ExperimentSpec(experiment_id, module)
    for experiment_id, module in _EXPERIMENT_MODULES.items()
}


def experiment_ids() -> tuple[str, ...]:
    """Return registered experiment identifiers in deterministic order."""

    return tuple(sorted(EXPERIMENTS))


def get_experiment(experiment_id: str) -> ExperimentSpec:
    """Resolve one experiment identifier or raise ``KeyError``."""

    return EXPERIMENTS[experiment_id]


def _render_help() -> str:
    return "\n".join(
        [
            "usage: bpt experiment <list|describe|run> [arguments]",
            "",
            "Registry-backed access to non-stable research commands.",
            "",
            "commands:",
            "  list                 list registered experiment identifiers",
            "  describe ID          show the module and canonical invocation",
            "  run ID [arguments]   invoke an experiment with forwarded arguments",
            "",
            "Only the top-level `bpt` executable is installed.",
        ]
    ) + "\n"


def _render_list() -> str:
    lines = ["registered experiments:"]
    lines.extend(f"  {experiment_id}" for experiment_id in experiment_ids())
    return "\n".join(lines) + "\n"


def _render_description(spec: ExperimentSpec) -> str:
    return "\n".join(
        [
            f"id: {spec.experiment_id}",
            f"stability: {spec.stability}",
            f"module: {spec.module}:{spec.function_name}",
            f"command: {spec.canonical_command}",
        ]
    ) + "\n"


def _resolve(experiment_id: str) -> ExperimentSpec | None:
    return EXPERIMENTS.get(experiment_id)


def _load_function(spec: ExperimentSpec) -> Callable[..., Any]:
    module = importlib.import_module(spec.module)
    function = getattr(module, spec.function_name)
    if not callable(function):
        raise TypeError(
            f"registered experiment target is not callable: "
            f"{spec.module}:{spec.function_name}"
        )
    return function


def _invoke(spec: ExperimentSpec, arguments: Sequence[str]) -> int:
    function = _load_function(spec)
    parameters = inspect.signature(function).parameters
    if not parameters:
        previous_argv = sys.argv
        sys.argv = [spec.canonical_command, *arguments]
        try:
            result = function()
        finally:
            sys.argv = previous_argv
    elif len(parameters) == 1:
        result = function(list(arguments))
    else:
        raise TypeError(
            f"registered experiment target has unsupported signature: "
            f"{spec.module}:{spec.function_name}"
        )
    return 0 if result is None else int(result)


def _unknown_experiment(experiment_id: str) -> int:
    print(f"unknown experiment: {experiment_id}", file=sys.stderr)
    print("run `bpt experiment list` to inspect registered identifiers", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_render_help(), end="")
        return 0

    command, *remaining = arguments
    if command == "list":
        if remaining:
            print(_render_help(), file=sys.stderr, end="")
            return 2
        print(_render_list(), end="")
        return 0

    if command == "describe":
        if len(remaining) != 1:
            print(_render_help(), file=sys.stderr, end="")
            return 2
        spec = _resolve(remaining[0])
        if spec is None:
            return _unknown_experiment(remaining[0])
        print(_render_description(spec), end="")
        return 0

    if command == "run":
        if not remaining:
            print(_render_help(), file=sys.stderr, end="")
            return 2
        experiment_id, *experiment_arguments = remaining
        spec = _resolve(experiment_id)
        if spec is None:
            return _unknown_experiment(experiment_id)
        return _invoke(spec, experiment_arguments)

    print(_render_help(), file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
