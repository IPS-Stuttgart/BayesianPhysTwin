"""Registry-backed access to non-stable Bayesian-PhysTwin commands."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, cast


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One lazily imported research command outside the stable CLI surface."""

    experiment_id: str
    module: str
    function_name: str = "main"
    stability: str = "experimental"

    @property
    def canonical_command(self) -> str:
        return f"bpt experiment run {self.experiment_id}"


_EXPERIMENT_TARGETS: Final[str] = """
build-phystwin-cues phystwin_cues
calibrate-phystwin-discrepancy phystwin_discrepancy
export-phystwin-residuals phystwin_export
phystwin-refit phystwin_refit
build-phystwin-prefix phystwin_prefix_artifact
evaluate-phystwin-official phystwin_official_evaluation
evaluate-phystwin-priors phystwin_prior_evaluation
combine-phystwin-profiles phystwin_joint_profile
confirm-phystwin-residual phystwin_confirmatory
confirm-phystwin-residual-baselines phystwin_baseline_confirmation
confirm-phystwin-bayesian-anchor phystwin_bayesian_confirmation
audit-phystwin-calibration phystwin_calibration
confirm-phystwin-combined phystwin_combined_confirmation
confirm-phystwin-additional-anchor phystwin_additional_confirmation
confirm-phystwin-additional-bayesian phystwin_additional_bayesian_confirmation
compare-phystwin-additional-controls phystwin_additional_control_comparison
analyze-phystwin-horizon phystwin_horizon_analysis
analyze-phystwin-controller-sensitivity phystwin_controller_sensitivity
infer-phystwin-controller-bias phystwin_controller_inference
analyze-phystwin-spatial-modes phystwin_spatial_mode_analysis
compare-phystwin-graph-anchors phystwin_graph_anchor_comparison
evaluate-phystwin-state-injection phystwin_state_injection
fit-phystwin-residual-dynamics phystwin_residual_dynamics
fit-phystwin-residual-velocity phystwin_residual_velocity
fit-phystwin-shared-residual-velocity phystwin_shared_residual_velocity
evaluate-phystwin-pgrd phystwin_pgrd_adapter
calibrate-phystwin-pgrd phystwin_pgrd_calibrated
train-phystwin-pgrd phystwin_pgrd_native
train-phystwin-pgrd-unrolled phystwin_pgrd_unrolled
fit-phystwin-hierarchical-residual phystwin_residual_shrinkage
compare-phystwin-residual-scales phystwin_residual_scale_comparison
fit-phystwin-residual-baselines phystwin_residual_baselines
build-phystwin-raw-cues phystwin_raw_cues
build-phystwin-cotracker3-cues phystwin_cotracker3_cues
evaluate-phystwin-perception-cues phystwin_perception_evaluation
associate-phystwin-motioncrafter phystwin_motioncrafter_association
assimilate-phystwin-motioncrafter phystwin_motioncrafter_assimilation
evaluate-phystwin-motioncrafter-assimilation phystwin_motioncrafter_assimilation_evaluation
select-phystwin-motioncrafter-view phystwin_motioncrafter_selection
fit-phystwin-bayesian-anchor phystwin_bayesian_anchor
compare-phystwin-trajectories phystwin_comparison
compare-phystwin-sota phystwin_sota_comparison
overlay-phystwin-external-backbone phystwin_external_backbone
gate-phystwin-backbone-family phystwin_backbone_family_gate
open-phystwin-backbone-family-future phystwin_backbone_family_future
report-matphys-loo-sota matphys_loo_sota_report
gate-matphys-part-family matphys_part_family_gate
open-matphys-part-family-future matphys_part_family_future
gate-phystwin-shared-nonlinear-residual phystwin_shared_nonlinear_residual
gate-phystwin-canonical-triplane-residual phystwin_canonical_triplane_residual
build-phystwin-spring-overlay phystwin_spring_overlay
gate-phystwin-part-pair-source phystwin_part_pair_source_gate
build-phystwin-piecewise-topology phystwin_piecewise_topology
gate-phystwin-sparse-topology-source phystwin_sparse_topology_source_gate
search-phystwin-topology-field phystwin_zero_order_topology
gate-phystwin-zero-order-source phystwin_zero_order_source_gate
diagnose-phystwin-bias phystwin_bias_diagnostic
evaluate-deform360-online-belief deform360_online_belief
build-deform360-raw-camera deform360_raw_camera_observation
build-deform360-crossview-supplement deform360_crossview_observation
predict-deform360-crossview-guard deform360_crossview_guard
diagnose-deform360-raw-pairwise deform360_raw_pairwise_correspondence_diagnostic
download-deform360-selective-virtual-sensing deform360_selective_virtual_sensing_download
benchmark-bias-aware-belief bias_aware_belief_benchmark
develop-deform360-bias-aware-belief deform360_bias_aware_belief_development
deform360-bias-aware-prospective deform360_bias_aware_prospective
deform360-bias-aware-result deform360_bias_aware_prospective_result
fetch-phystwin-eval-data phystwin_data
structural-recovery-benchmark structural_benchmark
diagnose-phystwin-structure phystwin_structural_diagnostic
aggregate-phystwin-structure structural_diagnostic_aggregate
audit-phystwin-state-decay phystwin_state_decay
audit-phystwin-state-modes phystwin_state_modes
aggregate-phystwin-state-modes phystwin_state_mode_aggregate
""".strip()


def _build_experiment_registry() -> dict[str, ExperimentSpec]:
    registry: dict[str, ExperimentSpec] = {}
    for line in _EXPERIMENT_TARGETS.splitlines():
        experiment_id, module_suffix = line.split()
        if experiment_id in registry:
            raise RuntimeError(f"duplicate experiment identifier: {experiment_id}")
        registry[experiment_id] = ExperimentSpec(
            experiment_id=experiment_id,
            module=f"bayesian_phystwin.cli.{module_suffix}",
        )
    return registry


EXPERIMENTS: Final[dict[str, ExperimentSpec]] = _build_experiment_registry()


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


def _load_function(spec: ExperimentSpec) -> Callable[..., object]:
    module = importlib.import_module(spec.module)
    function = getattr(module, spec.function_name)
    if not callable(function):
        raise TypeError(
            f"registered experiment target is not callable: "
            f"{spec.module}:{spec.function_name}"
        )
    return cast(Callable[..., object], function)


def _exit_code(result: object) -> int:
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    raise TypeError(
        "registered experiment target must return int or None, "
        f"received {type(result).__name__}"
    )


def _invoke(spec: ExperimentSpec, arguments: Sequence[str]) -> int:
    function = _load_function(spec)
    parameters = tuple(inspect.signature(function).parameters.values())
    if not parameters:
        previous_argv = sys.argv
        sys.argv = [spec.canonical_command, *arguments]
        try:
            return _exit_code(function())
        finally:
            sys.argv = previous_argv

    if len(parameters) == 1 and parameters[0].kind in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        return _exit_code(function(list(arguments)))

    raise TypeError(
        f"registered experiment target has unsupported signature: "
        f"{spec.module}:{spec.function_name}"
    )


def _unknown_experiment(experiment_id: str) -> int:
    print(f"unknown experiment: {experiment_id}", file=sys.stderr)
    print(
        "run `bpt experiment list` to inspect registered identifiers",
        file=sys.stderr,
    )
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
