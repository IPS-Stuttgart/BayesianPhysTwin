#!/usr/bin/env python3
"""Run the opened-source residual-history adapter without target access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    CLAIM_BOUNDARY,
    TARGET_QUARANTINE_ROOT,
    ResidualHistoryDryRunPolicyV1,
    assert_outside_target_quarantine,
    run_source_only_residual_history_dry_run,
)

LOCK_SCHEMA = "bayesian-phystwin/deform360-covariance-residual-history-dry-run-lock-v1"
SOURCE_SCHEMA = "bayesian-phystwin/deform360-covariance-residual-history-source-v1"
RESULT_SCHEMA = (
    "bayesian-phystwin/deform360-covariance-residual-history-dry-run-result-v1"
)

SOURCE_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "source_unit_id",
        "archive",
        "camera_ids",
        "provider_camera_ids",
        "scoring_camera_ids",
        "provider_reconstruction_artifact_id",
        "scoring_reconstruction_artifact_id",
        "information_boundary",
    }
)
SOURCE_ARCHIVE_FIELDS = frozenset({"path", "sha256"})
SOURCE_BOUNDARY_FIELDS = frozenset(
    {
        "opened_source_only",
        "fresh_target_payload_opened",
        "fresh_target_prediction_opened",
        "fresh_target_outcome_opened",
    }
)

EXPECTED_ARRAYS = frozenset(
    {
        "physical_prefix_m",
        "provider_observation_prefix_m",
        "provider_validity",
        "physical_future_m",
        "physical_fallback_covariance_m2",
        "donor_covariance_m2",
        "frame_indices",
        "material_ids",
        "future_horizon_bins",
    }
)


class _StrictJsonError(ValueError):
    pass


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJsonError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise _StrictJsonError(f"{name} contains non-finite number {token!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _canonical_sha256(value: Mapping[str, Any], *, digest_key: str) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _literal_lower_hex(
    value: object,
    *,
    name: str,
    lengths: frozenset[int],
) -> str:
    if (
        type(value) is not str
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        expected = ", ".join(str(length) for length in sorted(lengths))
        raise ValueError(
            f"{name} must be lowercase hexadecimal with length in {{{expected}}}"
        )
    return value


def _literal_sha256(value: object, *, name: str) -> str:
    return _literal_lower_hex(value, name=name, lengths=frozenset({64}))


def load_locked_policy(
    path: Path,
) -> tuple[dict[str, Any], ResidualHistoryDryRunPolicyV1]:
    """Load the pre-target dry-run lock and construct its exact policy."""

    protocol = _load_json(path.resolve(), name="dry-run protocol")
    _require(protocol.get("schema") == LOCK_SCHEMA, "dry-run protocol schema changed")
    _require(protocol.get("schema_version") == 1, "dry-run protocol version changed")
    _require(
        protocol.get("status") == "locked-before-source-dry-run-and-target-decode",
        "dry-run protocol is not prospective",
    )
    supplied = _literal_sha256(protocol.get("lock_sha256"), name="lock_sha256")
    _require(
        supplied == _canonical_sha256(protocol, digest_key="lock_sha256"),
        "dry-run protocol digest changed",
    )
    target = _mapping(protocol.get("target_binding"), name="target_binding")
    _require(
        target.get("target_roster_sha256")
        == "f9106b3dd6e0cec089623e07fed3506755fb334952c7761846d0854dfba45783",
        "target roster binding changed",
    )
    _require(
        target.get("exact_file_plan_sha256")
        == "d5bab5a05cf49ba6cc7bd31ffe57d2abc15040dd3f2de163d5f5034800b3ee51",
        "target file-plan binding changed",
    )
    _require(
        target.get("download_manifest_sha256")
        == "41bfb0feb246ac235e6364cfb46304dd8b2679801d73532a1e78281f243d59af",
        "target download binding changed",
    )
    _require(
        target.get("independent_verification_sha256")
        == "3f58caf2e5cff977b34ddce6f42c86438696ce9f36f637238597f4ca86c15997",
        "target verification binding changed",
    )
    _require(
        Path(str(target.get("quarantine_root"))).resolve()
        == TARGET_QUARANTINE_ROOT.resolve(),
        "target quarantine root changed",
    )
    boundary = _mapping(
        protocol.get("information_boundary"),
        name="information_boundary",
    )
    for key in (
        "fresh_target_media_decoded",
        "fresh_target_arrays_opened",
        "fresh_target_predictions_opened",
        "fresh_target_outcomes_opened",
    ):
        _require(boundary.get(key) is False, f"information boundary changed: {key}")
    policy = _mapping(protocol.get("policy"), name="policy")
    scales = policy.get("covariance_scales")
    if not isinstance(scales, list) or type(scales) is not list or len(scales) != 3:
        raise ValueError(
            "covariance scales must contain early, middle, and late values"
        )
    return protocol, ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=policy["minimum_prefix_frames"],
        minimum_final_observed_count=policy["minimum_final_observed_count"],
        minimum_final_observed_fraction=policy["minimum_final_observed_fraction"],
        minimum_cameras_per_role=policy["minimum_cameras_per_role"],
        minimum_camera_families_per_role=policy["minimum_camera_families_per_role"],
        covariance_scales=tuple(scales),
    )


def load_source_manifest(path: Path) -> dict[str, Any]:
    """Load an opened-source input manifest while keeping the target closed."""

    manifest_path = assert_outside_target_quarantine(path, name="source manifest")
    manifest = _load_json(manifest_path, name="source manifest")
    _exact_fields(
        manifest,
        expected=SOURCE_MANIFEST_FIELDS,
        name="source manifest",
    )
    _require(manifest.get("schema") == SOURCE_SCHEMA, "source manifest schema changed")
    _require(manifest.get("schema_version") == 1, "source manifest version changed")
    boundary = _mapping(
        manifest.get("information_boundary"),
        name="source information boundary",
    )
    _exact_fields(
        boundary,
        expected=SOURCE_BOUNDARY_FIELDS,
        name="source information boundary",
    )
    _require(
        boundary.get("opened_source_only") is True,
        "source-only declaration missing",
    )
    for key in (
        "fresh_target_payload_opened",
        "fresh_target_prediction_opened",
        "fresh_target_outcome_opened",
    ):
        _require(boundary.get(key) is False, f"source boundary changed: {key}")
    provider_id = _literal_sha256(
        manifest.get("provider_reconstruction_artifact_id"),
        name="provider_reconstruction_artifact_id",
    )
    scoring_id = _literal_sha256(
        manifest.get("scoring_reconstruction_artifact_id"),
        name="scoring_reconstruction_artifact_id",
    )
    _require(
        provider_id != scoring_id,
        "provider/scoring reconstruction artifact reused",
    )
    for field_name in (
        "camera_ids",
        "provider_camera_ids",
        "scoring_camera_ids",
    ):
        cameras = manifest.get(field_name)
        _require(
            isinstance(cameras, list) and all(type(value) is str for value in cameras),
            f"source {field_name} roster is missing",
        )
    archive = _mapping(manifest.get("archive"), name="source archive")
    _exact_fields(
        archive,
        expected=SOURCE_ARCHIVE_FIELDS,
        name="source archive",
    )
    archive_value = archive.get("path")
    if (
        not isinstance(archive_value, str)
        or type(archive_value) is not str
        or not archive_value
    ):
        raise ValueError("archive path missing")
    candidate = Path(archive_value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    archive_path = assert_outside_target_quarantine(
        candidate,
        name="source archive",
    )
    _require(archive_path.is_file(), "source archive does not exist")
    _require(
        _file_sha256(archive_path)
        == _literal_sha256(archive.get("sha256"), name="source archive SHA-256"),
        "source archive bytes changed",
    )
    result = dict(manifest)
    result["resolved_archive_path"] = str(archive_path)
    return result


def _atomic_output_directory(path: Path) -> tuple[Path, Path]:
    target = assert_outside_target_quarantine(path, name="output directory")
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    return target, temporary


def run_dry_run(
    *,
    protocol_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Execute one source-only adapter dry run and publish a no-clobber receipt."""

    revision = _literal_lower_hex(
        implementation_revision,
        name="implementation_revision",
        lengths=frozenset({40}),
    )
    protocol, policy = load_locked_policy(protocol_path)
    manifest = load_source_manifest(source_manifest_path)
    archive_path = Path(manifest["resolved_archive_path"])
    with np.load(archive_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == EXPECTED_ARRAYS,
            "source dry-run archive field roster changed",
        )
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    result = run_source_only_residual_history_dry_run(
        arrays["physical_prefix_m"],
        arrays["provider_observation_prefix_m"],
        arrays["provider_validity"],
        arrays["physical_future_m"],
        arrays["physical_fallback_covariance_m2"],
        arrays["donor_covariance_m2"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_horizon_bins=arrays["future_horizon_bins"],
        camera_ids=manifest["camera_ids"],
        provider_camera_ids=manifest["provider_camera_ids"],
        scoring_camera_ids=manifest["scoring_camera_ids"],
        provider_reconstruction_artifact_id=manifest[
            "provider_reconstruction_artifact_id"
        ],
        scoring_reconstruction_artifact_id=manifest[
            "scoring_reconstruction_artifact_id"
        ],
        source_unit_id=manifest["source_unit_id"],
        reference_predictor_id=protocol["candidate"]["reference_predictor_id"],
        covariance_donor_id=protocol["candidate"]["covariance_donor_id"],
        policy=policy,
        metadata={
            "implementation_revision": revision,
            "source_manifest_sha256": _file_sha256(source_manifest_path.resolve()),
            "source_archive_sha256": _file_sha256(archive_path),
            "target_lock_sha256": protocol["lock_sha256"],
        },
    )
    target, temporary = _atomic_output_directory(output_dir)
    try:
        archive_output = temporary / "dry_run_arrays.npz"
        np.savez_compressed(
            archive_output,
            mean_m=result.mean_m,
            covariance_m2=result.covariance_m2,
            residual_history_m=result.adapter.residual_history_m,
            observed_validity=result.adapter.observed_validity,
            frame_indices=result.adapter.frame_indices,
            material_ids=result.adapter.material_ids,
            provider_camera_ids=np.asarray(
                result.adapter.partition.provider_camera_ids,
                dtype=np.str_,
            ),
            scoring_camera_ids=np.asarray(
                result.adapter.partition.scoring_camera_ids,
                dtype=np.str_,
            ),
        )
        payload: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "schema_version": 1,
            "protocol_lock_sha256": protocol["lock_sha256"],
            "implementation_revision": revision,
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "sha256": _file_sha256(source_manifest_path.resolve()),
            },
            "source_archive": {
                "path": str(archive_path),
                "sha256": _file_sha256(archive_path),
            },
            "policy": {
                **policy.descriptor(),
                "policy_id": policy.policy_id,
            },
            "camera_partition": {
                **result.adapter.partition.descriptor(),
                "partition_id": result.adapter.partition.partition_id,
            },
            "adapter": {
                **result.adapter.descriptor(),
                "adapter_id": result.adapter.adapter_id,
            },
            "decision": {
                **result.decision.descriptor(),
                "decision_id": result.decision.decision_id,
            },
            "output": {
                "archive": archive_output.name,
                "archive_sha256": _file_sha256(archive_output),
            },
            "information_boundary": {
                "opened_source_archive_read": True,
                "fresh_target_quarantine_read": False,
                "fresh_target_media_decoded": False,
                "fresh_target_arrays_opened": False,
                "fresh_target_predictions_opened": False,
                "fresh_target_outcomes_opened": False,
            },
            "claim_boundary": CLAIM_BOUNDARY,
        }
        payload["result_id"] = content_id(payload)
        (temporary / "dry_run_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "protocols/locks/deform360_covariance_residual_history_dry_run_v1.json"
        ),
    )
    parser.add_argument("--implementation-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_dry_run(
        protocol_path=arguments.protocol,
        source_manifest_path=arguments.source_manifest,
        output_dir=arguments.output_dir,
        implementation_revision=arguments.implementation_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
