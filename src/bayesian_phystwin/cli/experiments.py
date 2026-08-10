"""Registry-backed access to non-stable Bayesian-PhysTwin commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    module: str
    function_name: str = "main"


_EXPERIMENT_MODULES: Final[dict[str, str]] = {
    "aggregate-phystwin-state-modes": (
        "bayesian_phystwin.cli.phystwin_state_mode_aggregate"
    ),
    "aggregate-phystwin-structure": (
        "bayesian_phystwin.cli.structural_diagnostic_aggregate"
    ),
    "analyze-phystwin-controller-sensitivity": (
        "bayesian_phystwin.cli.phystwin_controller_sensitivity"
    ),
    "analyze-phystwin-horizon": "bayesian_phystwin.cli.phystwin_horizon_analysis",
    "analyze-phystwin-spatial-modes": (
        "bayesian_phystwin.cli.phystwin_spatial_mode_analysis"
    ),
    "assimilate-phystwin-motioncrafter": (
        "bayesian_phystwin.cli.phystwin_motioncrafter_assimilation"
    ),
    "associate-phystwin-motioncrafter": (
        "bayesian_phystwin.cli.phystwin_motioncrafter_association"
    ),
    "audit-prob4d-covariance-ablation": (
        "bayesian_phystwin.cli.prob4d_covariance_ablation"
    ),
    "audit-phystwin-calibration": "bayesian_phystwin.cli.phystwin_calibration",
    "audit-phystwin-state-decay": "bayesian_phystwin.cli.phystwin_state_decay",
    "audit-phystwin-state-modes": "bayesian_phystwin.cli.phystwin_state_modes",
    "benchmark-bias-aware-belief": "bayesian_phystwin.cli.bias_aware_belief_benchmark",
    "build-deform360-crossview-supplement": (
        "bayesian_phystwin.cli.deform360_crossview_observation"
    ),
    "build-deform360-raw-camera": (
        "bayesian_phystwin.cli.deform360_raw_camera_observation"
    ),
    "build-phystwin-cotracker3-cues": "bayesian_phystwin.cli.phystwin_cotracker3_cues",
    "build-phystwin-cues": "bayesian_phystwin.cli.phystwin_cues",
    "build-phystwin-piecewise-topology": (
        "bayesian_phystwin.cli.phystwin_piecewise_topology"
    ),
    "build-phystwin-prefix": "bayesian_phystwin.cli.phystwin_prefix_artifact",
    "build-phystwin-raw-cues": "bayesian_phystwin.cli.phystwin_raw_cues",
    "build-phystwin-spring-overlay": "bayesian_phystwin.cli.phystwin_spring_overlay",
    "calibrate-phystwin-discrepancy": "bayesian_phystwin.cli.phystwin_discrepancy",
    "calibrate-phystwin-pgrd": "bayesian_phystwin.cli.phystwin_pgrd_calibrated",
    "combine-phystwin-profiles": "bayesian_phystwin.cli.phystwin_joint_profile",
    "compare-phystwin-additional-controls": (
        "bayesian_phystwin.cli.phystwin_additional_control_comparison"
    ),
    "compare-phystwin-graph-anchors": (
        "bayesian_phystwin.cli.phystwin_graph_anchor_comparison"
    ),
    "compare-phystwin-residual-scales": (
        "bayesian_phystwin.cli.phystwin_residual_scale_comparison"
    ),
    "compare-phystwin-sota": "bayesian_phystwin.cli.phystwin_sota_comparison",
    "compare-phystwin-trajectories": "bayesian_phystwin.cli.phystwin_comparison",
    "confirm-phystwin-additional-anchor": (
        "bayesian_phystwin.cli.phystwin_additional_confirmation"
    ),
    "confirm-phystwin-additional-bayesian": (
        "bayesian_phystwin.cli.phystwin_additional_bayesian_confirmation"
    ),
    "confirm-phystwin-bayesian-anchor": (
        "bayesian_phystwin.cli.phystwin_bayesian_confirmation"
    ),
    "confirm-phystwin-combined": "bayesian_phystwin.cli.phystwin_combined_confirmation",
    "confirm-phystwin-residual": "bayesian_phystwin.cli.phystwin_confirmatory",
    "confirm-phystwin-residual-baselines": (
        "bayesian_phystwin.cli.phystwin_baseline_confirmation"
    ),
    "deform360-bias-aware-prospective": (
        "bayesian_phystwin.cli.deform360_bias_aware_prospective"
    ),
    "deform360-bias-aware-result": (
        "bayesian_phystwin.cli.deform360_bias_aware_prospective_result"
    ),
    "develop-deform360-bias-aware-belief": (
        "bayesian_phystwin.cli.deform360_bias_aware_belief_development"
    ),
    "diagnose-deform360-raw-pairwise": (
        "bayesian_phystwin.cli.deform360_raw_pairwise_correspondence_diagnostic"
    ),
    "diagnose-phystwin-bias": "bayesian_phystwin.cli.phystwin_bias_diagnostic",
    "diagnose-phystwin-structure": (
        "bayesian_phystwin.cli.phystwin_structural_diagnostic"
    ),
    "diagnose-provider-failures": (
        "bayesian_phystwin.cli.provider_failure_decomposition"
    ),
    "select-discrepancy-candidate": (
        "bayesian_phystwin.cli.discrepancy_candidate_tournament"
    ),
    "download-deform360-selective-virtual-sensing": (
        "bayesian_phystwin.cli.deform360_selective_virtual_sensing_download"
    ),
    "evaluate-deform360-online-belief": "bayesian_phystwin.cli.deform360_online_belief",
    "evaluate-phystwin-motioncrafter-assimilation": (
        "bayesian_phystwin.cli.phystwin_motioncrafter_assimilation_evaluation"
    ),
    "evaluate-phystwin-official": "bayesian_phystwin.cli.phystwin_official_evaluation",
    "evaluate-phystwin-perception-cues": (
        "bayesian_phystwin.cli.phystwin_perception_evaluation"
    ),
    "evaluate-phystwin-pgrd": "bayesian_phystwin.cli.phystwin_pgrd_adapter",
    "evaluate-phystwin-priors": "bayesian_phystwin.cli.phystwin_prior_evaluation",
    "evaluate-phystwin-state-injection": (
        "bayesian_phystwin.cli.phystwin_state_injection"
    ),
    "evaluate-pokeflex-public": ("bayesian_phystwin.cli.pokeflex_public_evaluation"),
    "export-phystwin-residuals": "bayesian_phystwin.cli.phystwin_export",
    "fetch-phystwin-eval-data": "bayesian_phystwin.cli.phystwin_data",
    "fit-phystwin-bayesian-anchor": "bayesian_phystwin.cli.phystwin_bayesian_anchor",
    "fit-phystwin-hierarchical-residual": (
        "bayesian_phystwin.cli.phystwin_residual_shrinkage"
    ),
    "fit-phystwin-residual-baselines": (
        "bayesian_phystwin.cli.phystwin_residual_baselines"
    ),
    "fit-phystwin-residual-dynamics": (
        "bayesian_phystwin.cli.phystwin_residual_dynamics"
    ),
    "fit-phystwin-residual-velocity": (
        "bayesian_phystwin.cli.phystwin_residual_velocity"
    ),
    "fit-phystwin-shared-residual-velocity": (
        "bayesian_phystwin.cli.phystwin_shared_residual_velocity"
    ),
    "gate-matphys-part-family": "bayesian_phystwin.cli.matphys_part_family_gate",
    "gate-phystwin-backbone-family": (
        "bayesian_phystwin.cli.phystwin_backbone_family_gate"
    ),
    "gate-phystwin-canonical-triplane-residual": (
        "bayesian_phystwin.cli.phystwin_canonical_triplane_residual"
    ),
    "gate-phystwin-part-pair-source": (
        "bayesian_phystwin.cli.phystwin_part_pair_source_gate"
    ),
    "gate-phystwin-shared-nonlinear-residual": (
        "bayesian_phystwin.cli.phystwin_shared_nonlinear_residual"
    ),
    "gate-phystwin-sparse-topology-source": (
        "bayesian_phystwin.cli.phystwin_sparse_topology_source_gate"
    ),
    "gate-phystwin-zero-order-source": (
        "bayesian_phystwin.cli.phystwin_zero_order_source_gate"
    ),
    "infer-phystwin-controller-bias": (
        "bayesian_phystwin.cli.phystwin_controller_inference"
    ),
    "open-matphys-part-family-future": (
        "bayesian_phystwin.cli.matphys_part_family_future"
    ),
    "open-phystwin-backbone-family-future": (
        "bayesian_phystwin.cli.phystwin_backbone_family_future"
    ),
    "overlay-phystwin-external-backbone": (
        "bayesian_phystwin.cli.phystwin_external_backbone"
    ),
    "phystwin-refit": "bayesian_phystwin.cli.phystwin_refit",
    "predict-deform360-crossview-guard": (
        "bayesian_phystwin.cli.deform360_crossview_guard"
    ),
    "report-matphys-loo-sota": "bayesian_phystwin.cli.matphys_loo_sota_report",
    "search-phystwin-topology-field": (
        "bayesian_phystwin.cli.phystwin_zero_order_topology"
    ),
    "seal-deform360-calibration": (
        "bayesian_phystwin.cli.deform360_calibration_execution"
    ),
    "select-phystwin-motioncrafter-view": (
        "bayesian_phystwin.cli.phystwin_motioncrafter_selection"
    ),
    "structural-recovery-benchmark": "bayesian_phystwin.cli.structural_benchmark",
    "train-phystwin-pgrd": "bayesian_phystwin.cli.phystwin_pgrd_native",
    "train-phystwin-pgrd-unrolled": "bayesian_phystwin.cli.phystwin_pgrd_unrolled",
}

EXPERIMENTS: Final[dict[str, ExperimentSpec]] = {
    experiment_id: ExperimentSpec(experiment_id, module)
    for experiment_id, module in _EXPERIMENT_MODULES.items()
}
