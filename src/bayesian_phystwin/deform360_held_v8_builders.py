"""Explicit v8 adapters for the frozen Deform360 numerical builders.

The expensive frame-zero, physical-prior, and online-prefix implementations
were written against the held-v7 artifact API.  Reusing their numerical code
must not make the v7 validators accept v8 artifacts (or vice versa).  This
module therefore provides one narrow, opt-in adapter for the three dedicated
v8 CLI entry points.  It replaces only protocol hooks and constants inside a
single short-lived subprocess and restores the original objects on exit.

No v7 module or CLI imports this adapter.  The guard checks below intentionally
fail if the frozen v7 hook set changes, rather than silently adapting a future
implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
import runpy
import sys
from types import ModuleType
from typing import Any, Iterator, Literal, Mapping

from . import deform360_frame_zero_assets as frame_zero
from . import deform360_frame_zero_semantic_gate as semantic_gate
from . import deform360_held_online_prefix as online
from . import deform360_held_physical_prior as physical
from . import deform360_held_protocol as v7_protocol
from . import deform360_held_v8_protocol as v8_protocol


V7_PROTOCOL_ID = "deform360-held-online-belief-v7"
V8_PROTOCOL_ID = "deform360-held-online-belief-v8"
V8_PYCACHE_PREFIX = "/nonexistent/bpt-held-v8-pycache"

_V8_SEMANTIC_LABEL_BY_PREFIX = {
    "002": "rope",
    "072": "rope",
    "081": "rope",
    "083": "blanket",
    "085": "scarf",
    "092": "squirrel toy",
    "170": "spider toy",
}

_V7_FRAME_ZERO_LOCK_LOADER = frame_zero.load_generic_held_lock
_V7_FRAME_ZERO_SEMANTIC_LABEL = frame_zero.semantic_label_for_object_id
_V7_SEMANTIC_GATE_LABEL = semantic_gate.semantic_label_for_object_id
_V7_ONLINE_FRAME_ZERO_ASSET_VALIDATOR = online.validate_frame_zero_asset_manifest


def semantic_label_for_v8_object_id(object_id: str) -> str:
    """Return the v8-only frozen semantic label, including object 072."""

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("object id is missing")
    prefix = object_id.split("-", 1)[0]
    try:
        return _V8_SEMANTIC_LABEL_BY_PREFIX[prefix]
    except KeyError as error:
        raise ValueError("object id is outside the frozen v8 semantic map") from error


def _require_same(observed: Any, expected: Any, label: str) -> None:
    if observed is not expected:
        raise RuntimeError(f"frozen v7 builder hook changed: {label}")


def _require_v7_baseline() -> None:
    if frame_zero.HELD_PROTOCOL_ID != V7_PROTOCOL_ID:
        raise RuntimeError("frame-zero builder is not at the frozen v7 baseline")
    if physical.PROTOCOL_ID != V7_PROTOCOL_ID:
        raise RuntimeError("physical builder is not at the frozen v7 baseline")
    if online.PROTOCOL_ID != V7_PROTOCOL_ID:
        raise RuntimeError("online builder is not at the frozen v7 baseline")
    if physical.HELD_PYCACHE_PREFIX != "/nonexistent/bpt-held-v7-pycache":
        raise RuntimeError("physical builder pycache contract changed")

    _require_same(
        frame_zero.load_generic_held_lock,
        _V7_FRAME_ZERO_LOCK_LOADER,
        "frame_zero.load_generic_held_lock",
    )
    _require_same(
        frame_zero.semantic_label_for_object_id,
        _V7_FRAME_ZERO_SEMANTIC_LABEL,
        "frame_zero.semantic_label_for_object_id",
    )
    _require_same(
        semantic_gate.semantic_label_for_object_id,
        _V7_SEMANTIC_GATE_LABEL,
        "semantic_gate.semantic_label_for_object_id",
    )
    _require_same(
        physical.load_held_protocol_lock,
        v7_protocol.load_held_protocol_lock,
        "physical.load_held_protocol_lock",
    )
    _require_same(
        physical.create_physical_prior_seal,
        v7_protocol.create_physical_prior_seal,
        "physical.create_physical_prior_seal",
    )
    _require_same(
        physical.validate_frame_zero_bundle_manifest,
        v7_protocol.validate_frame_zero_bundle_manifest,
        "physical.validate_frame_zero_bundle_manifest",
    )
    _require_same(
        online.create_online_prediction_seal,
        v7_protocol.create_online_prediction_seal,
        "online.create_online_prediction_seal",
    )
    _require_same(
        online.validate_frame_zero_bundle_manifest,
        v7_protocol.validate_frame_zero_bundle_manifest,
        "online.validate_frame_zero_bundle_manifest",
    )
    _require_same(
        online.validate_physical_prior_seal,
        v7_protocol.validate_physical_prior_seal,
        "online.validate_physical_prior_seal",
    )
    _require_same(
        online.validate_prefix_stage_authorization,
        v7_protocol.validate_prefix_stage_authorization,
        "online.validate_prefix_stage_authorization",
    )
    _require_same(
        online.validate_frame_zero_asset_manifest,
        _V7_ONLINE_FRAME_ZERO_ASSET_VALIDATOR,
        "online.validate_frame_zero_asset_manifest",
    )


def _patch(
    module: ModuleType, name: str, value: Any, saved: list[tuple[Any, str, Any]]
) -> None:
    saved.append((module, name, getattr(module, name)))
    setattr(module, name, value)


def _validate_already_protocol_validated_v8_frame_zero(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the identity of the manifest already deeply validated by v8.

    ``online._load_frame_zero_arrays`` first invokes the v8 protocol validator,
    which replays the full legacy source audit on a transient v7 view.  Its
    immediately following legacy-only identity check cannot accept the original
    v8 protocol ID, so this hook performs only that redundant identity check.
    """

    if manifest.get("protocol_id") != V8_PROTOCOL_ID:
        raise ValueError("online frame-zero manifest is not held-v8")
    if manifest.get("artifact_sha256") != v8_protocol.held_artifact_sha256(manifest):
        raise ValueError("online frame-zero manifest checksum changed")
    return dict(manifest)


@contextmanager
def explicit_v8_builder_context(
    stage: Literal["frame-zero", "physical", "online"],
) -> Iterator[None]:
    """Install stage-specific v8 hooks and restore the exact v7 hooks."""

    _require_v7_baseline()
    if stage not in {"frame-zero", "physical", "online"}:
        raise ValueError("unsupported v8 numerical builder stage")
    saved: list[tuple[Any, str, Any]] = []
    try:
        if stage == "frame-zero":
            _patch(frame_zero, "HELD_PROTOCOL_ID", V8_PROTOCOL_ID, saved)
            _patch(
                frame_zero,
                "load_generic_held_lock",
                v8_protocol.validate_protocol_lock,
                saved,
            )
            _patch(
                frame_zero,
                "semantic_label_for_object_id",
                semantic_label_for_v8_object_id,
                saved,
            )
            _patch(
                semantic_gate,
                "semantic_label_for_object_id",
                semantic_label_for_v8_object_id,
                saved,
            )

        if stage in {"physical", "online"}:
            _patch(physical, "PROTOCOL_ID", V8_PROTOCOL_ID, saved)
            _patch(physical, "HELD_PYCACHE_PREFIX", V8_PYCACHE_PREFIX, saved)
            _patch(
                physical,
                "load_held_protocol_lock",
                v8_protocol.validate_protocol_lock,
                saved,
            )
            _patch(
                physical,
                "validate_frame_zero_bundle_manifest",
                v8_protocol.validate_frame_zero_bundle_manifest,
                saved,
            )
            _patch(
                physical,
                "create_physical_prior_seal",
                v8_protocol.create_physical_prior_seal,
                saved,
            )
            _patch(
                physical,
                "held_artifact_sha256",
                v8_protocol.held_artifact_sha256,
                saved,
            )
            _patch(
                physical,
                "held_contract_sha256",
                v8_protocol.held_contract_sha256,
                saved,
            )

        if stage == "online":
            _patch(online, "PROTOCOL_ID", V8_PROTOCOL_ID, saved)
            _patch(online, "PRIMARY_METHOD", v8_protocol.PRIMARY_METHOD, saved)
            _patch(online, "UPDATE_FRAMES", v8_protocol.UPDATE_FRAMES, saved)
            _patch(
                online,
                "ONLINE_ARTIFACT_ROLES",
                v8_protocol.ONLINE_ARTIFACT_ROLES,
                saved,
            )
            _patch(
                online,
                "validate_frame_zero_bundle_manifest",
                v8_protocol.validate_frame_zero_bundle_manifest,
                saved,
            )
            _patch(
                online,
                "validate_frame_zero_asset_manifest",
                _validate_already_protocol_validated_v8_frame_zero,
                saved,
            )
            _patch(
                online,
                "validate_physical_prior_seal",
                v8_protocol.validate_physical_prior_seal,
                saved,
            )
            _patch(
                online,
                "validate_prefix_stage_authorization",
                v8_protocol.validate_prefix_stage_authorization,
                saved,
            )
            _patch(
                online,
                "create_online_prediction_seal",
                v8_protocol.create_online_prediction_seal,
                saved,
            )
            _patch(
                online,
                "held_artifact_sha256",
                v8_protocol.held_artifact_sha256,
                saved,
            )
            _patch(
                online,
                "semantic_label_for_object_id",
                semantic_label_for_v8_object_id,
                saved,
            )
        yield
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


def run_v8_adapted_cli(module_name: str) -> None:
    """Run one allowlisted legacy numerical CLI under the explicit v8 hooks."""

    allowed = {
        "bayesian_phystwin.cli.deform360_frame_zero_assets",
        "bayesian_phystwin.cli.deform360_held_physical_prior",
        "bayesian_phystwin.cli.deform360_held_online_prefix",
    }
    if module_name not in allowed:
        raise ValueError("module is outside the v8 numerical CLI allowlist")
    stage_by_module = {
        "bayesian_phystwin.cli.deform360_frame_zero_assets": "frame-zero",
        "bayesian_phystwin.cli.deform360_held_physical_prior": "physical",
        "bayesian_phystwin.cli.deform360_held_online_prefix": "online",
    }
    with explicit_v8_builder_context(stage_by_module[module_name]):
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)


def main_for(module_name: str) -> None:
    """Small entry-point helper that preserves the caller's argument vector."""

    if not sys.argv:
        raise RuntimeError("missing process argument vector")
    run_v8_adapted_cli(module_name)


__all__ = [
    "V7_PROTOCOL_ID",
    "V8_PROTOCOL_ID",
    "V8_PYCACHE_PREFIX",
    "explicit_v8_builder_context",
    "main_for",
    "run_v8_adapted_cli",
    "semantic_label_for_v8_object_id",
]
