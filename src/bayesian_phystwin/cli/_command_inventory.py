"""Lifecycle and ownership metadata for the Bayesian-PhysTwin command catalog."""

from __future__ import annotations

from typing import Final

STABLE_ROUTES: Final[dict[str, tuple[str, ...]]] = {
    "provider-manifest": ("provider", "manifest"),
    "validate-observation-belief": ("observation", "validate"),
    "replay-residuals": ("residual", "replay"),
    "synthetic-benchmark": ("benchmark", "synthetic"),
    "decisive-evidence": ("evidence", "summarize"),
    "claim-bundle": ("evidence", "bundle"),
    "run-manifest": ("run", "manifest"),
}

ARCHIVED_IDS: Final = frozenset(
    """
    evaluate-phystwin-state-injection
    fit-phystwin-residual-velocity
    fit-phystwin-shared-residual-velocity
    evaluate-phystwin-pgrd
    calibrate-phystwin-pgrd
    train-phystwin-pgrd
    train-phystwin-pgrd-unrolled
    associate-phystwin-motioncrafter
    assimilate-phystwin-motioncrafter
    evaluate-phystwin-motioncrafter-assimilation
    select-phystwin-motioncrafter-view
    report-matphys-loo-sota
    gate-matphys-part-family
    open-matphys-part-family-future
    gate-phystwin-shared-nonlinear-residual
    gate-phystwin-canonical-triplane-residual
    build-phystwin-spring-overlay
    gate-phystwin-part-pair-source
    build-phystwin-piecewise-topology
    gate-phystwin-sparse-topology-source
    search-phystwin-topology-field
    gate-phystwin-zero-order-source
    structural-recovery-benchmark
    diagnose-phystwin-structure
    aggregate-phystwin-structure
    """.split()
)

DIAGNOSTIC_IDS: Final = frozenset(
    """
    evaluate-phystwin-priors
    audit-prob4d-covariance-ablation
    audit-phystwin-calibration
    compare-phystwin-additional-controls
    analyze-phystwin-horizon
    analyze-phystwin-controller-sensitivity
    infer-phystwin-controller-bias
    analyze-phystwin-spatial-modes
    compare-phystwin-graph-anchors
    compare-phystwin-residual-scales
    evaluate-phystwin-perception-cues
    compare-phystwin-trajectories
    compare-phystwin-sota
    diagnose-phystwin-bias
    diagnose-deform360-raw-pairwise
    diagnose-provider-failures
    select-discrepancy-candidate
    audit-phystwin-state-decay
    audit-phystwin-state-modes
    aggregate-phystwin-state-modes
    """.split()
)

DESCRIPTION_OVERRIDES: Final[dict[str, str]] = {
    "run-manifest": "create or validate a content-addressed run manifest",
    "provider-manifest": "print the Causal4D provider manifest",
    "validate-observation-belief": (
        "validate or score an ObservationBeliefV1 artifact"
    ),
    "replay-residuals": "replay exported residuals through the robust likelihood",
    "synthetic-benchmark": "run the controlled synthetic benchmark",
    "decisive-evidence": "summarize matched guarded prospective evidence",
    "claim-bundle": "build or validate a content-addressed claim bundle",
    "audit-prob4d-covariance-ablation": (
        "verify and compare a controlled five-way Prob4D covariance ablation"
    ),
    "confirm-phystwin-bayesian-anchor": (
        "evaluate the frozen Bayesian anchor on the official PhysTwin cohort"
    ),
    "evaluate-phystwin-official": "evaluate released PhysTwin trajectories",
    "deform360-bias-aware-prospective": (
        "run the sealed Deform360 bias-aware prospective protocol"
    ),
    "deform360-bias-aware-result": (
        "aggregate the sealed Deform360 bias-aware prospective result"
    ),
    "diagnose-provider-failures": (
        "attribute source-only provider and guarded-update rejection causes"
    ),
    "select-discrepancy-candidate": (
        "select one matched discrepancy belief on source-only groups"
    ),
    "seal-deform360-calibration": (
        "seal all target-blind Deform360 calibration choices before confirmation"
    ),
    "fetch-phystwin-eval-data": "fetch the released PhysTwin evaluation subset",
}

EXACT_OWNERS: Final[dict[str, str]] = {
    "run-manifest": "run-manifest-v2",
    "provider-manifest": "causal4d-provider-v1",
    "validate-observation-belief": "observation-belief-v1",
    "replay-residuals": "residual-replay-v1",
    "synthetic-benchmark": "synthetic-benchmark-v3",
    "decisive-evidence": "bayesian-phystwin-decisive-evidence-v1",
    "claim-bundle": "claim-bundle-v1",
    "audit-prob4d-covariance-ablation": "prob4d-covariance-ablation-v1",
    "combine-phystwin-profiles": "phystwin-profile-pooling-v1",
    "calibrate-phystwin-discrepancy": "phystwin-discrepancy-calibration-v1",
    "phystwin-refit": "phystwin-refit-v1",
    "evaluate-deform360-online-belief": "deform360-online-belief-v1",
    "seal-deform360-calibration": "deform360-official-hub-visuotactile-v1",
    "diagnose-phystwin-bias": "phystwin-bias-audit-v1",
    "diagnose-provider-failures": "provider-failure-decomposition-v1",
    "select-discrepancy-candidate": "discrepancy-candidate-tournament-v1",
}


def status_name(command_id: str) -> str:
    if command_id in STABLE_ROUTES:
        return "stable"
    if command_id in ARCHIVED_IDS:
        return "archived"
    if command_id in DIAGNOSTIC_IDS:
        return "diagnostic"
    return "experiment"


def description(command_id: str) -> str:
    return DESCRIPTION_OVERRIDES.get(command_id, command_id.replace("-", " "))


def owner(command_id: str) -> str:
    exact = EXACT_OWNERS.get(command_id)
    if exact is not None:
        return exact
    rules = (
        (("pokeflex",), "pokeflex-public-evaluation-v1"),
        (("deform360",), "deform360-bias-aware-v1"),
        (("matphys",), "matphys-causal-backbone-v1"),
        (("motioncrafter",), "phystwin-motioncrafter-v1"),
        (("pgrd",), "phystwin-pgrd-v1"),
        (("nonlinear-residual", "triplane"), "phystwin-nonlinear-residual-v1"),
        (
            ("topology", "spring-overlay", "part-pair", "zero-order"),
            "phystwin-topology-experiments-v1",
        ),
        (("structure", "structural"), "phystwin-structural-calibration-v1"),
        (
            ("state-decay", "state-modes"),
            "phystwin-discrepancy-localization-v1",
        ),
        (("state-injection",), "phystwin-state-injection-v1"),
        (
            ("residual-dynamics", "residual-velocity"),
            "phystwin-residual-dynamics-v1",
        ),
        (
            ("hierarchical-residual", "residual-scales", "residual-baselines"),
            "phystwin-residual-inference-v1",
        ),
        (
            ("raw-cues", "cotracker3", "perception-cues"),
            "phystwin-perception-cues-v1",
        ),
        (
            ("external-backbone", "backbone-family"),
            "phystwin-external-backbone-v1",
        ),
        (("additional",), "phystwin-additional-cohort-v1"),
        (("controller-sensitivity",), "phystwin-controller-sensitivity-v1"),
        (("controller-bias",), "phystwin-controller-bias-v1"),
    )
    for markers, protocol in rules:
        if any(marker in command_id for marker in markers):
            return protocol
    if command_id == "benchmark-bias-aware-belief":
        return "deform360-bias-aware-v1"
    return "phystwin-full22-v1"


def optional_dependencies(command_id: str) -> tuple[str, ...]:
    if command_id == "seal-deform360-calibration":
        return ()
    dependencies: list[str] = []
    if command_id == "evaluate-pokeflex-public":
        dependencies.extend(("graph", "vision"))
    if command_id in {
        "download-deform360-selective-virtual-sensing",
        "fetch-phystwin-eval-data",
    }:
        dependencies.append("data")
    graph = (
        "graph",
        "spring",
        "topology",
        "structure",
        "structural",
        "state-injection",
        "state-decay",
        "state-modes",
        "pgrd",
        "bayesian-anchor",
        "online-belief",
        "hierarchical-residual",
        "combined",
        "calibration",
        "controller-sensitivity",
    )
    vision = (
        "raw-camera",
        "crossview-supplement",
        "raw-pairwise",
        "cotracker3",
        "raw-cues",
        "perception-cues",
        "motioncrafter",
    )
    if any(marker in command_id for marker in graph):
        dependencies.append("graph")
    if any(marker in command_id for marker in vision):
        dependencies.append("vision")
    if command_id == "infer-phystwin-controller-bias":
        dependencies.append("pyrecest")
    return tuple(dependencies)
