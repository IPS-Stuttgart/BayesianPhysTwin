"""Calibration-support repair protocol for the frozen bias-aware candidate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_bias_aware_prospective_protocol import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXPECTED_CALIBRATION_COHORT,
    EXPECTED_STRATA,
    EXPECTED_TARGET_COHORT,
    SOURCE_LOCK_GROUP_COUNT,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "deform360-bias-aware-guarded-belief-prospective-v2"
SELECTION_SEED = PROTOCOL_ID
BASE_PROTOCOL_ID = "deform360-bias-aware-guarded-belief-prospective-v1"
BASE_PROTOCOL_FILE_SHA256 = (
    "9e933a93e28869cc67101300cd0990feb148841355f6a2416dc8cb92f595fa01"
)
BASE_PROTOCOL_CONFIG_SHA256 = (
    "b6b19be5eaadf830a77f36cccddd38f5b7a35527ca21f7743d2ef147fceabbce"
)
BASE_CALIBRATION_COHORT_FILE_SHA256 = (
    "78d8bebee63b01ba10d4d90897f30534f472110312ddfd5c8533173f28e3e047"
)
BASE_CALIBRATION_COHORT_RESULT_SHA256 = (
    "581bfb211d312d0cadfdd5d1a0005d4cba1369f0eb70015438371a07a0503dc3"
)
BASE_SUPPORT_REJECTION_FILE_SHA256 = (
    "21aea093584561370b47f375df869191dfe163e20835e4da9d5065b931c1a339"
)
BASE_SUPPORT_REJECTION_RESULT_SHA256 = (
    "eeee3907be8404935e32a43c9d11e5d2ffb29f2af0a7d0af3b0faa92429771e0"
)
SOURCE_LOCK_SHA256 = (
    "5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6"
)
FALLBACK_PROTOCOL_FILE_SHA256 = (
    "240554ed41986cf5b330225d759c8df24de99d8642f9c1dcd6114185ee16fc0d"
)
FALLBACK_INTEGRATION_FILE_SHA256 = (
    "5c06156459204e4a02e1625fea13428848bcdccc3d05b5a673df60de3ae0ed51"
)
FALLBACK_INTEGRATION_RESULT_SHA256 = (
    "8da661c0452a482f9aef3c90676740bcf31a6f7cc24d5ffb63df218687b266ec"
)
LOCK_COMMIT = "fb959e523b289efdb6c6a6fc5de64a5a000bd134"

FRESH_FILAMENT_POOL = (
    "078-fishing-line",
    "088-snake",
    "161-tube",
    "181-belt",
)
FRESH_CALIBRATION_OBJECT_COUNT = 3
EXPECTED_FRESH_CALIBRATION = {
    "078-fishing-line": (4,),
    "161-tube": (4,),
    "088-snake": (1,),
}
PRIOR_PATH_OPEN_OBJECTS = ("072-cotton-clohesline",)

IMPLEMENTATION_SHA256 = {
    "src/bayesian_phystwin/bias_aware_belief.py": (
        "fc6fd6b9b8bb8fbb84b515532553f6de3a9d8f43e7e45f0d3b7c4b9627e981cd"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_belief_development.py": (
        "35acbca215e0d0e98189f1e2ca81dff573817242fec807a5b22485ab2ba0a7b3"
    ),
    "src/bayesian_phystwin/deform360_raw_camera_observation.py": (
        "2c24e587e9acd1dda589363240a81b268863fbe808ce05c58f8a5b70f12f76c3"
    ),
    "src/bayesian_phystwin/deform360_bias_aware_prospective_physical.py": (
        "9f87a8aec1bd8bd8ae35865b819bae75d2a1d8d97d4950ca90b96ca90fa21226"
    ),
    "scripts/remote/run_deform360_bias_aware_frame_zero.py": (
        "d51410521cd8a894a653d930c2e80257b27099f033fe45951a2d091ec142fdec"
    ),
    "scripts/remote/run_deform360_bias_aware_physical_prior.py": (
        "e5d81207e89ccaa170a3711708d8ee5ba6b4b181fb08ab17819a35e8c0a9a4ff"
    ),
    "scripts/remote/run_deform360_bias_aware_prediction.py": (
        "93efacdd665bcc80eb87248fb580108e8f0e78bc9fcec289149865dcf21ccb1a"
    ),
}

CANONICAL_CONFIG_SHA256 = (
    "67e1157fa04283f1376855a7ac60f85a4de02434612592ffe8b4ef1e4607ebe4"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_ranked_fresh_filament_objects() -> tuple[str, ...]:
    return tuple(
        sorted(
            FRESH_FILAMENT_POOL,
            key=lambda object_id: hashlib.sha256(
                f"{SELECTION_SEED}:object:filament:{object_id}".encode()
            ).hexdigest(),
        )
    )


def metadata_ranked_episode_ids(object_id: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(10),
            key=lambda episode: hashlib.sha256(
                (
                    f"{SELECTION_SEED}:episode:calibration:"
                    f"{object_id}:{episode}"
                ).encode()
            ).hexdigest(),
        )
    )


def _cohort_records(
    cohort: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> dict[str, list[dict[str, object]]]:
    return {
        stratum: [
            {"object_id": object_id, "episode_ids": list(episode_ids)}
            for object_id, episode_ids in cohort[stratum].items()
        ]
        for stratum in EXPECTED_STRATA
    }


def _combined_calibration_cohort() -> dict[str, dict[str, tuple[int, ...]]]:
    combined = {
        stratum: dict(EXPECTED_CALIBRATION_COHORT[stratum])
        for stratum in EXPECTED_STRATA
    }
    combined["filament"].update(EXPECTED_FRESH_CALIBRATION)
    return combined


def build_bias_aware_prospective_v2_protocol() -> dict[str, Any]:
    fresh_rank = metadata_ranked_fresh_filament_objects()
    fresh = fresh_rank[:FRESH_CALIBRATION_OBJECT_COUNT]
    expected_fresh = tuple(EXPECTED_FRESH_CALIBRATION)
    if fresh != expected_fresh:
        raise AssertionError("fresh calibration rank changed")
    for object_id, episode_ids in EXPECTED_FRESH_CALIBRATION.items():
        if metadata_ranked_episode_ids(object_id)[:1] != episode_ids:
            raise AssertionError("fresh calibration episode rank changed")

    config: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "locked-before-fresh-calibration-download-or-media-access",
        "locked_at": "2026-07-21",
        "claim_boundary": (
            "Prospective calibration-support repair for the unchanged source-v4 "
            "bias-aware update. The original twelve target objects remain sealed. "
            "This is not official Deform360 Table-4 parity or a direct state-of-"
            "the-art claim."
        ),
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
        },
        "base_protocol": {
            "protocol_id": BASE_PROTOCOL_ID,
            "file": (
                "configs/sota/"
                "deform360_bias_aware_guarded_belief_prospective_v1.json"
            ),
            "file_sha256": BASE_PROTOCOL_FILE_SHA256,
            "config_sha256": BASE_PROTOCOL_CONFIG_SHA256,
            "calibration_prediction_cohort_file_sha256": (
                BASE_CALIBRATION_COHORT_FILE_SHA256
            ),
            "calibration_prediction_cohort_result_sha256": (
                BASE_CALIBRATION_COHORT_RESULT_SHA256
            ),
            "support_rejection_file_sha256": (
                BASE_SUPPORT_REJECTION_FILE_SHA256
            ),
            "support_rejection_result_sha256": (
                BASE_SUPPORT_REJECTION_RESULT_SHA256
            ),
            "calibration_future_read": False,
            "target_prefix_staged": False,
            "target_media_read": False,
            "target_future_read": False,
        },
        "repair": {
            "reason": (
                "v1 stopped outcome-blind with five automatic twins; add three "
                "fresh filament calibration objects so the unchanged seven-object "
                "and two-per-stratum support gate can be met"
            ),
            "method_family_changed": False,
            "candidate_threshold_changed": False,
            "calibration_gate_changed": False,
            "target_cohort_changed": False,
            "fresh_filament_pool": list(FRESH_FILAMENT_POOL),
            "prior_path_open_objects_excluded": list(PRIOR_PATH_OPEN_OBJECTS),
            "selection_seed": SELECTION_SEED,
            "selection_rule": "SHA256 rank over frozen name-only pool",
            "fresh_calibration": [
                {
                    "object_id": object_id,
                    "episode_ids": list(EXPECTED_FRESH_CALIBRATION[object_id]),
                }
                for object_id in fresh
            ],
            "prelock_server_path_audit": [
                {
                    "host": "gpuserver6000",
                    "roots": [
                        "/mnt/corsair/florianpfaff",
                        "/home/florianpfaff",
                    ],
                    "selected_object_matching_path_count": 0,
                    "audit_scope": "directory names only",
                },
                {
                    "host": "gpuserver4090",
                    "roots": [
                        "/mnt/corsair/florianpfaff",
                        "/home/florianpfaff",
                    ],
                    "selected_object_matching_path_count": 0,
                    "audit_scope": "directory names only",
                },
            ],
        },
        "calibration_cohort": _cohort_records(_combined_calibration_cohort()),
        "target_cohort": _cohort_records(EXPECTED_TARGET_COHORT),
        "method": {
            "source_lock_sha256": SOURCE_LOCK_SHA256,
            "source_group_count": SOURCE_LOCK_GROUP_COUNT,
            "candidate": "unchanged bias-aware response-constrained v4 update",
            "baseline": "unchanged selected raw physical/persistence backbone",
            "fallback": "bit-exact selected raw baseline",
            "implementation_sha256": dict(IMPLEMENTATION_SHA256),
            "lock_commit": LOCK_COMMIT,
        },
        "reconstruction_fallback": {
            "protocol": (
                "configs/sota/"
                "deform360_reconstruction_failure_persistence_fallback_v1.json"
            ),
            "protocol_file_sha256": FALLBACK_PROTOCOL_FILE_SHA256,
            "integration_audit": (
                "results/sota/"
                "deform360_reconstruction_failure_persistence_fallback_v1/"
                "integration_audit.json"
            ),
            "integration_audit_file_sha256": FALLBACK_INTEGRATION_FILE_SHA256,
            "integration_audit_result_sha256": (
                FALLBACK_INTEGRATION_RESULT_SHA256
            ),
            "physical_policy": "persistence_only",
            "candidate_and_baseline_bit_exact": True,
            "eligible_for_absolute_accuracy_or_calibration": False,
            "counts_as_paired_non_regression_tie": True,
        },
        "calibration_support_gate": {
            "minimum_automatic_twin_objects": 7,
            "minimum_automatic_twin_objects_per_stratum": 2,
            "minimum_new_eligible_object_groups": 5,
            "minimum_combined_eligible_object_groups": 9,
            "required_finite_sample_coverage": 0.9,
            "fresh_filament_automatic_twins_required": 2,
            "fallback_objects_do_not_count": True,
            "replacement_allowed": False,
            "failed_gate_action": (
                "publish support failure and keep every calibration future and "
                "all target data sealed"
            ),
        },
        "calibration_accuracy_gate": {
            "permitted_fit": "direct source-group regret bound only",
            "required_upper_regret_m": -0.000005,
            "co_primary_object_balanced_mean_regret_must_be_negative": True,
            "accepted_harmful_object_count_allowed": 0,
            "unit_of_replication": "physical object",
            "fallback_ties_excluded_from_certificate_fit": True,
            "target_access_if_gate_fails": "forbidden",
        },
        "target_evaluation": {
            "unit_of_replication": "physical object",
            "episodes_nested_within_object": 2,
            "co_primary_metrics": [
                "post_update_hidden_identity_rmse_m",
                "post_update_hidden_symmetric_chamfer_m",
            ],
            "comparison": "frozen v4 selector minus exact selected raw baseline",
            "minimum_automatic_twin_objects": 9,
            "minimum_automatic_twin_objects_per_stratum": 3,
            "replacement_allowed": False,
            "success_gates": {
                "both_object_balanced_mean_differences_negative": True,
                "both_object_cluster_upper_95_bounds_negative": True,
                "no_stratum_mean_regression": True,
                "accepted_harmful_object_rate_at_most": 0.1,
                "every_rejection_bit_exact_fallback": True,
                "all_quality_failures_reported": True,
            },
            "fallback_ties_reported_but_excluded_from_absolute_metrics": True,
            "direct_official_sota_claim_allowed": False,
        },
        "information_order": [
            "commit and push the v2 protocol before fresh-object download",
            "download only the three fresh calibration objects",
            "stage causal prefixes and seal automatic-twin or fallback disposition",
            "verify at least seven automatic twins and two per stratum",
            "open calibration futures only if the support gate passes",
            "fit only the declared direct group-regret certificate",
            "authorize target access only if every calibration accuracy gate passes",
            "stage and seal the unchanged twelve-object target cohort",
            "open target futures only after the complete prediction cohort seal",
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "config_sha256": protocol_config_sha256(config),
    }


def validate_bias_aware_prospective_v2_protocol(
    payload: Mapping[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prospective v2 schema version changed")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("prospective v2 config is missing")
    digest = protocol_config_sha256(config)
    if payload.get("config_sha256") != digest:
        raise ValueError("prospective v2 config checksum changed")
    if CANONICAL_CONFIG_SHA256 and digest != CANONICAL_CONFIG_SHA256:
        raise ValueError("prospective v2 canonical checksum changed")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("prospective v2 protocol ID changed")
    if config.get("target_cohort") != _cohort_records(EXPECTED_TARGET_COHORT):
        raise ValueError("reserved target cohort changed")
    if config.get("calibration_cohort") != _cohort_records(
        _combined_calibration_cohort()
    ):
        raise ValueError("v2 calibration cohort changed")
    fresh = config["repair"]["fresh_calibration"]
    expected_fresh = [
        {
            "object_id": object_id,
            "episode_ids": list(EXPECTED_FRESH_CALIBRATION[object_id]),
        }
        for object_id in metadata_ranked_fresh_filament_objects()[
            :FRESH_CALIBRATION_OBJECT_COUNT
        ]
    ]
    if fresh != expected_fresh:
        raise ValueError("fresh name-only selection changed")
    if config["calibration_support_gate"] != {
        "minimum_automatic_twin_objects": 7,
        "minimum_automatic_twin_objects_per_stratum": 2,
        "minimum_new_eligible_object_groups": 5,
        "minimum_combined_eligible_object_groups": 9,
        "required_finite_sample_coverage": 0.9,
        "fresh_filament_automatic_twins_required": 2,
        "fallback_objects_do_not_count": True,
        "replacement_allowed": False,
        "failed_gate_action": (
            "publish support failure and keep every calibration future and all "
            "target data sealed"
        ),
    }:
        raise ValueError("v2 support gate changed")
    if root is not None:
        root_path = Path(root)
        for relative, expected in IMPLEMENTATION_SHA256.items():
            if _file_sha256(root_path / relative) != expected:
                raise ValueError(f"implementation checksum changed: {relative}")
        bound_files = {
            config["base_protocol"]["file"]: BASE_PROTOCOL_FILE_SHA256,
            config["reconstruction_fallback"]["protocol"]: (
                FALLBACK_PROTOCOL_FILE_SHA256
            ),
            config["reconstruction_fallback"]["integration_audit"]: (
                FALLBACK_INTEGRATION_FILE_SHA256
            ),
        }
        for relative, expected in bound_files.items():
            if _file_sha256(root_path / relative) != expected:
                raise ValueError(f"bound artifact checksum changed: {relative}")
    return dict(payload)


def load_bias_aware_prospective_v2_protocol(
    path: str | Path, *, root: str | Path | None = None
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("prospective v2 protocol must be a JSON object")
    return validate_bias_aware_prospective_v2_protocol(payload, root=root)


__all__ = [
    "CANONICAL_CONFIG_SHA256",
    "EXPECTED_FRESH_CALIBRATION",
    "PROTOCOL_ID",
    "build_bias_aware_prospective_v2_protocol",
    "load_bias_aware_prospective_v2_protocol",
    "metadata_ranked_episode_ids",
    "metadata_ranked_fresh_filament_objects",
    "protocol_config_sha256",
    "validate_bias_aware_prospective_v2_protocol",
]
