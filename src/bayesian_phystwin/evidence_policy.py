"""Fail-closed admission for evidence-bound Bayesian-PhysTwin runs."""

from __future__ import annotations

from pathlib import Path

from .run_manifest_v2 import RunManifestV2, verify_run_manifest_artifacts

_REQUIRED_IDENTIFIERS = (
    "method_freeze_id",
    "protocol_id",
    "split_id",
    "baseline_id",
)


def require_promotable_run_manifest(
    manifest: RunManifestV2,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Validate that a V2 run is complete enough to support a promoted claim.

    Loading a :class:`RunManifestV2` proves schema and content-address integrity.
    Promotion is intentionally stricter: every participating repository must be
    clean, scientific identifiers and claims must be explicit, and both input and
    output artifacts must be present. When ``root`` is supplied, artifact bytes are
    verified before the manifest is admitted.
    """

    if not isinstance(manifest, RunManifestV2):
        raise TypeError("promotable evidence requires RunManifestV2")

    dirty_repositories = []
    if manifest.dirty:
        dirty_repositories.append(manifest.repository)
    dirty_repositories.extend(
        state.repository for state in manifest.related_repositories if state.dirty
    )
    if dirty_repositories:
        raise ValueError(
            "promotable evidence requires clean repositories: "
            + ", ".join(sorted(dirty_repositories))
        )

    missing_identifiers = [
        name
        for name in _REQUIRED_IDENTIFIERS
        if not str(getattr(manifest, name)).strip()
    ]
    if missing_identifiers:
        raise ValueError(
            "promotable evidence is missing identifiers: "
            + ", ".join(missing_identifiers)
        )
    if not manifest.claim_ids:
        raise ValueError("promotable evidence must identify at least one claim")
    if not manifest.inputs or not manifest.outputs:
        raise ValueError("promotable evidence requires input and output artifacts")
    if not manifest.package_versions:
        raise ValueError("promotable evidence requires package versions")
    if not manifest.runtime_environment:
        raise ValueError("promotable evidence requires a runtime environment")
    if not manifest.information_boundary:
        raise ValueError("promotable evidence requires an information boundary")
    if not manifest.configuration:
        raise ValueError("promotable evidence requires an explicit configuration")

    if root is not None:
        verify_run_manifest_artifacts(manifest, root=root)

    return {
        "status": "promotable",
        "manifest_id": manifest.manifest_id,
        "evidence_fingerprint": manifest.evidence_fingerprint,
        "claim_ids": list(manifest.claim_ids),
        "repositories": [
            {
                "repository": manifest.repository,
                "revision": manifest.revision,
                "role": "primary",
            },
            *[
                {
                    "repository": state.repository,
                    "revision": state.revision,
                    "role": state.role,
                }
                for state in manifest.related_repositories
            ],
        ],
        "input_artifact_count": len(manifest.inputs),
        "output_artifact_count": len(manifest.outputs),
    }


__all__ = ["require_promotable_run_manifest"]
