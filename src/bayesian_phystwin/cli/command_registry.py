"""Declarative registry for the supported Bayesian-PhysTwin command surface.

Historical ``bpt-*`` scripts remain frozen compatibility aliases. New commands
must be grouped-only entries and must not add another console script.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

CommandStatus = Literal["stable", "experiment", "diagnostic", "archived"]
Route: TypeAlias = tuple[str, ...]
STATUS_ORDER: Final[tuple[CommandStatus, ...]] = (
    "stable",
    "experiment",
    "diagnostic",
    "archived",
)
VISIBLE_STATUSES: Final[frozenset[CommandStatus]] = frozenset(
    {"stable", "experiment"}
)
FROZEN_LEGACY_ALIAS_SHA256: Final[str] = (
    "aead4f765e1f3a6ccd255a94ea3bc37f0dc9373efc1d75f680b1a4492af602cc"
)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One grouped command and its compatibility metadata."""

    command_id: str
    route: Route
    module: str
    function: str
    description: str
    status: CommandStatus
    milestone: str
    extras: tuple[str, ...] = ()
    legacy_alias: str | None = None

    @property
    def target(self) -> str:
        """Return the packaging entry-point target."""

        return f"{self.module}:{self.function}"


_LEGACY_TARGET_ROWS: Final[str] = """\
s|run-manifest|run_manifest
s|provider-manifest|provider_manifest
s|validate-observation-belief|observation_belief
e|build-phystwin-cues|phystwin_cues
d|calibrate-phystwin-discrepancy|phystwin_discrepancy
e|export-phystwin-residuals|phystwin_export
a|phystwin-refit|phystwin_refit
e|build-phystwin-prefix|phystwin_prefix_artifact
e|evaluate-phystwin-official|phystwin_official_evaluation
d|evaluate-phystwin-priors|phystwin_prior_evaluation
d|combine-phystwin-profiles|phystwin_joint_profile
e|confirm-phystwin-residual|phystwin_confirmatory
e|confirm-phystwin-residual-baselines|phystwin_baseline_confirmation
e|confirm-phystwin-bayesian-anchor|phystwin_bayesian_confirmation
d|audit-phystwin-calibration|phystwin_calibration
e|confirm-phystwin-combined|phystwin_combined_confirmation
a|confirm-phystwin-additional-anchor|phystwin_additional_confirmation
a|confirm-phystwin-additional-bayesian|phystwin_additional_bayesian_confirmation
a|compare-phystwin-additional-controls|phystwin_additional_control_comparison
d|analyze-phystwin-horizon|phystwin_horizon_analysis
d|analyze-phystwin-controller-sensitivity|phystwin_controller_sensitivity
d|infer-phystwin-controller-bias|phystwin_controller_inference
d|analyze-phystwin-spatial-modes|phystwin_spatial_mode_analysis
d|compare-phystwin-graph-anchors|phystwin_graph_anchor_comparison
d|evaluate-phystwin-state-injection|phystwin_state_injection
d|fit-phystwin-residual-dynamics|phystwin_residual_dynamics
a|fit-phystwin-residual-velocity|phystwin_residual_velocity
a|fit-phystwin-shared-residual-velocity|phystwin_shared_residual_velocity
a|evaluate-phystwin-pgrd|phystwin_pgrd_adapter
a|calibrate-phystwin-pgrd|phystwin_pgrd_calibrated
a|train-phystwin-pgrd|phystwin_pgrd_native
a|train-phystwin-pgrd-unrolled|phystwin_pgrd_unrolled
d|fit-phystwin-hierarchical-residual|phystwin_residual_shrinkage
d|compare-phystwin-residual-scales|phystwin_residual_scale_comparison
d|fit-phystwin-residual-baselines|phystwin_residual_baselines
e|build-phystwin-raw-cues|phystwin_raw_cues
e|build-phystwin-cotracker3-cues|phystwin_cotracker3_cues
e|evaluate-phystwin-perception-cues|phystwin_perception_evaluation
a|associate-phystwin-motioncrafter|phystwin_motioncrafter_association
a|assimilate-phystwin-motioncrafter|phystwin_motioncrafter_assimilation
a|evaluate-phystwin-motioncrafter-assimilation|phystwin_motioncrafter_assimilation_evaluation
a|select-phystwin-motioncrafter-view|phystwin_motioncrafter_selection
e|fit-phystwin-bayesian-anchor|phystwin_bayesian_anchor
d|compare-phystwin-trajectories|phystwin_comparison
d|compare-phystwin-sota|phystwin_sota_comparison
e|overlay-phystwin-external-backbone|phystwin_external_backbone
e|gate-phystwin-backbone-family|phystwin_backbone_family_gate
e|open-phystwin-backbone-family-future|phystwin_backbone_family_future
a|report-matphys-loo-sota|matphys_loo_sota_report
a|gate-matphys-part-family|matphys_part_family_gate
a|open-matphys-part-family-future|matphys_part_family_future
a|gate-phystwin-shared-nonlinear-residual|phystwin_shared_nonlinear_residual
a|gate-phystwin-canonical-triplane-residual|phystwin_canonical_triplane_residual
a|build-phystwin-spring-overlay|phystwin_spring_overlay
a|gate-phystwin-part-pair-source|phystwin_part_pair_source_gate
a|build-phystwin-piecewise-topology|phystwin_piecewise_topology
a|gate-phystwin-sparse-topology-source|phystwin_sparse_topology_source_gate
a|search-phystwin-topology-field|phystwin_zero_order_topology
a|gate-phystwin-zero-order-source|phystwin_zero_order_source_gate
d|diagnose-phystwin-bias|phystwin_bias_diagnostic
e|evaluate-deform360-online-belief|deform360_online_belief
e|build-deform360-raw-camera|deform360_raw_camera_observation
e|build-deform360-crossview-supplement|deform360_crossview_observation
e|predict-deform360-crossview-guard|deform360_crossview_guard
d|diagnose-deform360-raw-pairwise|deform360_raw_pairwise_correspondence_diagnostic
e|download-deform360-selective-virtual-sensing|deform360_selective_virtual_sensing_download
e|benchmark-bias-aware-belief|bias_aware_belief_benchmark
e|develop-deform360-bias-aware-belief|deform360_bias_aware_belief_development
e|deform360-bias-aware-prospective|deform360_bias_aware_prospective
e|deform360-bias-aware-result|deform360_bias_aware_prospective_result
e|fetch-phystwin-eval-data|phystwin_data
s|replay-residuals|residual_replay
s|synthetic-benchmark|synthetic_benchmark
d|structural-recovery-benchmark|structural_benchmark
d|diagnose-phystwin-structure|phystwin_structural_diagnostic
d|aggregate-phystwin-structure|structural_diagnostic_aggregate
d|audit-phystwin-state-decay|phystwin_state_decay
d|audit-phystwin-state-modes|phystwin_state_modes
d|aggregate-phystwin-state-modes|phystwin_state_mode_aggregate
"""
_STATUS_CODES: Final[Mapping[str, CommandStatus]] = MappingProxyType(
    {"s": "stable", "e": "experiment", "d": "diagnostic", "a": "archived"}
)


def _legacy_rows() -> Iterator[tuple[CommandStatus, str, str]]:
    for row in _LEGACY_TARGET_ROWS.splitlines():
        code, name, module = row.split("|", maxsplit=2)
        yield _STATUS_CODES[code], name, module


def _legacy_entry_points() -> dict[str, str]:
    return {
        f"bpt-{name}": f"bayesian_phystwin.cli.{module}:main"
        for _, name, module in _legacy_rows()
    }


LEGACY_ENTRY_POINTS: Final[Mapping[str, str]] = MappingProxyType(
    _legacy_entry_points()
)
_STABLE_ROUTES: Final[Mapping[str, Route]] = MappingProxyType(
    {
        "run-manifest": ("run", "manifest"),
        "provider-manifest": ("provider", "manifest"),
        "validate-observation-belief": ("observation", "validate"),
        "replay-residuals": ("residual", "replay"),
        "synthetic-benchmark": ("benchmark", "synthetic"),
    }
)
_STABLE_MILESTONES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "run-manifest": "run-manifest-v2",
        "provider-manifest": "causal4d-provider-v1",
        "validate-observation-belief": "observation-belief-v1",
        "replay-residuals": "reliability-model-v1",
        "synthetic-benchmark": "synthetic-benchmark-v1",
    }
)
_STABLE_DESCRIPTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "run-manifest": "create or validate content-addressed run provenance",
        "provider-manifest": "print the versioned Causal4D provider manifest",
        "validate-observation-belief": (
            "validate or summarize an ObservationBeliefV1 artifact"
        ),
        "replay-residuals": "replay residuals through the robust likelihood",
        "synthetic-benchmark": "run the controlled fixed-graph benchmark",
    }
)


def _domain(name: str) -> str:
    if "deform360" in name or name == "benchmark-bias-aware-belief":
        return "deform360"
    if "matphys" in name:
        return "matphys"
    if "phystwin" in name or name == "structural-recovery-benchmark":
        return "phystwin"
    return "misc"


def _route(name: str, status: CommandStatus) -> Route:
    stable = _STABLE_ROUTES.get(name)
    if stable is not None:
        return stable
    return status, "run", name


def _milestone(name: str, status: CommandStatus) -> str:
    stable = _STABLE_MILESTONES.get(name)
    if stable is not None:
        return stable
    domain = _domain(name)
    if domain == "deform360":
        return "deform360-bias-aware-v1"
    if status == "experiment":
        return "phystwin-full22-v1"
    if status == "archived":
        if domain == "matphys":
            return "matphys-causal-backbone-v1"
        return "historical-compatibility"
    if "structure" in name or name == "structural-recovery-benchmark":
        return "phystwin-structural-calibration-v1"
    if "state-" in name or "discrepancy" in name:
        return "phystwin-discrepancy-localization-v1"
    return "phystwin-full22-diagnostic"


def _extras(name: str) -> tuple[str, ...]:
    vision_tokens = (
        "raw-camera",
        "raw-cues",
        "cotracker",
        "crossview",
        "perception",
        "motioncrafter",
    )
    graph_tokens = (
        "matphys",
        "pgrd",
        "structure",
        "topology",
        "spring-overlay",
        "part-pair",
        "nonlinear-residual",
        "triplane-residual",
    )
    if any(token in name for token in vision_tokens):
        return ("vision",)
    if name.startswith(("fetch-", "download-")):
        return ("data",)
    if any(token in name for token in graph_tokens):
        return ("graph",)
    return ()


def _legacy_commands() -> tuple[CommandSpec, ...]:
    commands: list[CommandSpec] = []
    for status, name, module_name in _legacy_rows():
        commands.append(
            CommandSpec(
                command_id=name,
                route=_route(name, status),
                module=f"bayesian_phystwin.cli.{module_name}",
                function="main",
                description=_STABLE_DESCRIPTIONS.get(
                    name, name.replace("-", " ")
                ),
                status=status,
                milestone=_milestone(name, status),
                extras=_extras(name),
                legacy_alias=f"bpt-{name}",
            )
        )
    return tuple(commands)


COMMANDS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        command_id="commands",
        route=("commands",),
        module="bayesian_phystwin.cli.main",
        function="_commands_main",
        description="list registry metadata and compatibility aliases",
        status="stable",
        milestone="command-surface-v1",
    ),
    CommandSpec(
        command_id="decisive-evidence",
        route=("evidence", "summarize"),
        module="bayesian_phystwin.cli.decisive_evidence",
        function="main",
        description="summarize matched guarded prospective evidence",
        status="stable",
        milestone="decisive-evidence-v1",
    ),
    *_legacy_commands(),
)


def _validate_registry(commands: Iterable[CommandSpec]) -> None:
    routes: set[Route] = set()
    command_ids: set[tuple[CommandStatus, str]] = set()
    aliases: set[str] = set()
    for command in commands:
        invalid_route = any(
            not token or token.startswith("-") for token in command.route
        )
        identity = (command.status, command.command_id)
        if not command.route or invalid_route or command.route in routes:
            raise RuntimeError(f"invalid or duplicate route: {command.route!r}")
        if identity in command_ids:
            raise RuntimeError(f"duplicate command identity: {identity!r}")
        routes.add(command.route)
        command_ids.add(identity)
        if not command.description.strip() or not command.milestone.strip():
            raise RuntimeError(f"incomplete metadata for {command.route!r}")
        if len(set(command.extras)) != len(command.extras):
            raise RuntimeError(f"duplicate extras for {command.route!r}")
        alias = command.legacy_alias
        if alias is None:
            continue
        if alias == "bpt" or not alias.startswith("bpt-") or alias in aliases:
            raise RuntimeError(f"invalid or duplicate legacy alias: {alias!r}")
        aliases.add(alias)


def legacy_alias_fingerprint(
    commands: Iterable[CommandSpec] | None = None,
) -> str:
    """Return the review-visible fingerprint of the frozen alias set."""

    selected = COMMANDS if commands is None else tuple(commands)
    aliases = sorted(
        command.legacy_alias
        for command in selected
        if command.legacy_alias is not None
    )
    return sha256("\n".join(aliases).encode()).hexdigest()


def iter_commands(
    *,
    statuses: Iterable[CommandStatus] | None = None,
) -> Iterator[CommandSpec]:
    """Iterate commands in lifecycle and route order."""

    selected = None if statuses is None else frozenset(statuses)
    rank = {status: index for index, status in enumerate(STATUS_ORDER)}
    ordered = sorted(COMMANDS, key=lambda item: (rank[item.status], item.route))
    for command in ordered:
        if selected is None or command.status in selected:
            yield command


_validate_registry(COMMANDS)
if legacy_alias_fingerprint() != FROZEN_LEGACY_ALIAS_SHA256:
    raise RuntimeError(
        "the frozen bpt-* alias set changed; new commands must be grouped-only"
    )

ROUTES: Final[Mapping[Route, CommandSpec]] = MappingProxyType(
    {command.route: command for command in COMMANDS}
)
