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
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType
from typing import Any, Iterator, Literal, Mapping, Sequence

from . import deform360_frame_zero_assets as frame_zero
from . import deform360_frame_zero_semantic_gate as semantic_gate
from . import deform360_held_online_prefix as online
from . import deform360_held_physical_prior as physical
from . import deform360_held_protocol as v7_protocol
from . import deform360_held_v8_protocol as v8_protocol


V7_PROTOCOL_ID = "deform360-held-online-belief-v7"
V8_PROTOCOL_ID = "deform360-held-online-belief-v8.3"
V8_PYCACHE_PREFIX = "/nonexistent/bpt-held-v83-pycache"
V8_EXTERNAL_CALIBRATION_CASE_NAME = "072-cotton-clohesline-ep0003"
V8_EXTERNAL_CALIBRATION_OBJECT_ID = "072-cotton-clohesline"
V8_EXTERNAL_CALIBRATION_EPISODE_ID = 3
V8_EXTERNAL_ADMISSION_PROTOCOL_ID = str(
    v8_protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT["protocol_id"]
)
V8_EXTERNAL_ADMISSION_CONTRACT_SHA256 = (
    v8_protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256
)

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
_V7_PHYSICAL_ISOLATED_RUNPY_COMMAND = physical._isolated_runpy_command
_V7_PHYSICAL_INADMISSIBLE_TWIN_VALIDATOR = (
    physical._validate_inadmissible_automatic_twin
)
_V7_PHYSICAL_PREDICTION_ARCHIVE_BUILDER = physical.build_physical_prediction_archive
_V7_AUTOMATIC_TWIN_PROTOCOL_ID = physical.AUTOMATIC_TWIN_PROTOCOL_ID
_V7_AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 = (
    physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
)


def _external_admission_bootstrap() -> str:
    """Build the exact-case child bootstrap around the frozen v7 bootstrap."""

    canonical_contract = json.dumps(
        v8_protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    insertion = f"""\
import hashlib
import json

_v8_contract_json = {canonical_contract!r}
_v8_contract_sha256 = {V8_EXTERNAL_ADMISSION_CONTRACT_SHA256!r}
if hashlib.sha256(_v8_contract_json.encode("utf-8")).hexdigest() != _v8_contract_sha256:
    raise RuntimeError("v8 external admission contract checksum changed")
_v8_contract = json.loads(_v8_contract_json)

from causal4d_public import deform360_dense_reusable_panel as _v8_panel

_v8_expected_panel = os.path.join(
    roots[0], "causal4d_public", "deform360_dense_reusable_panel.py"
)
if os.path.realpath(getattr(_v8_panel, "__file__", "")) != os.path.realpath(
    _v8_expected_panel
):
    raise RuntimeError("v8 external admission panel provenance changed")
_v8_original_authorize = _v8_panel.authorize_dense_panel_episode
_v8_original_run_path = runpy.run_path

def _v8_authorize_external_calibration(
    payload,
    *,
    object_id,
    episode_id,
    phase,
    source_admission_passed=False,
):
    if (
        object_id != {V8_EXTERNAL_CALIBRATION_OBJECT_ID!r}
        or int(episode_id) != {V8_EXTERNAL_CALIBRATION_EPISODE_ID}
        or phase != "calibration"
        or source_admission_passed is not True
    ):
        raise ValueError("case is outside the exact v8 external calibration admission")
    validated = _v8_panel.validate_dense_reusable_panel_config(payload)
    if (
        validated.get("protocol_id")
        != _v8_contract["inherited_numerical_protocol_id"]
        or validated.get("config_sha256")
        != _v8_contract["inherited_numerical_config_sha256"]
    ):
        raise ValueError("inherited dense-panel numerical contract changed")
    return {{
        **validated,
        "protocol_id": _v8_contract["protocol_id"],
        "config_sha256": _v8_contract_sha256,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "phase": phase,
        "target_access": False,
        "admission_contract_sha256": _v8_contract_sha256,
        "inherited_numerical_protocol_id": validated["protocol_id"],
        "inherited_numerical_config_sha256": validated["config_sha256"],
    }}

def _v8_run_path(*args, **kwargs):
    try:
        return _v8_original_run_path(*args, **kwargs)
    finally:
        _v8_observed_authorize = _v8_panel.authorize_dense_panel_episode
        _v8_observed_run_path = runpy.run_path
        _v8_panel.authorize_dense_panel_episode = _v8_original_authorize
        runpy.run_path = _v8_original_run_path
        if (
            _v8_observed_authorize is not _v8_authorize_external_calibration
            or _v8_observed_run_path is not _v8_run_path
            or _v8_panel.authorize_dense_panel_episode is not _v8_original_authorize
            or runpy.run_path is not _v8_original_run_path
        ):
            raise RuntimeError("v8 external admission hooks changed during execution")

_v8_panel.authorize_dense_panel_episode = _v8_authorize_external_calibration
runpy.run_path = _v8_run_path
"""
    marker = "sys.path[:0] = roots\n"
    baseline = physical._ISOLATED_RUNPY_BOOTSTRAP
    if baseline.count(marker) != 1:
        raise RuntimeError("frozen isolated bootstrap insertion point changed")
    return baseline.replace(marker, marker + insertion, 1)


_V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP = _external_admission_bootstrap()


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
    _require_same(
        physical._isolated_runpy_command,
        _V7_PHYSICAL_ISOLATED_RUNPY_COMMAND,
        "physical._isolated_runpy_command",
    )
    _require_same(
        physical._validate_inadmissible_automatic_twin,
        _V7_PHYSICAL_INADMISSIBLE_TWIN_VALIDATOR,
        "physical._validate_inadmissible_automatic_twin",
    )
    _require_same(
        physical.build_physical_prediction_archive,
        _V7_PHYSICAL_PREDICTION_ARCHIVE_BUILDER,
        "physical.build_physical_prediction_archive",
    )
    if (
        physical.AUTOMATIC_TWIN_PROTOCOL_ID != _V7_AUTOMATIC_TWIN_PROTOCOL_ID
        or physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        != _V7_AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
    ):
        raise RuntimeError("frozen v7 automatic-twin admission identity changed")


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


def _argument_value(arguments: Sequence[str], name: str) -> str:
    if arguments.count(name) != 1:
        raise RuntimeError(f"v8 automatic-twin arguments contain {name} incorrectly")
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise RuntimeError(f"v8 automatic-twin argument {name} has no value")
    return arguments[index + 1]


def _require_flag_once(arguments: Sequence[str], name: str) -> None:
    if arguments.count(name) != 1:
        raise RuntimeError(f"v8 automatic-twin arguments contain {name} incorrectly")


def _authorize_external_admission_command(
    script: str | Path,
    *,
    import_roots: Sequence[str | Path],
    arguments: Sequence[str],
    provenance_root: str | Path | None,
) -> None:
    """Revalidate the exact v8 lock, case, and replacement frame zero."""

    object_id = _argument_value(arguments, "--object-id")
    try:
        episode_id = int(_argument_value(arguments, "--episode-id"))
    except ValueError as error:
        raise RuntimeError("v8 automatic-twin episode is not an integer") from error
    case_name = f"{object_id}-ep{episode_id:04d}"
    if case_name != V8_EXTERNAL_CALIBRATION_CASE_NAME:
        raise RuntimeError("case is outside the exact v8 external calibration case")
    if (
        object_id != V8_EXTERNAL_CALIBRATION_OBJECT_ID
        or episode_id != V8_EXTERNAL_CALIBRATION_EPISODE_ID
        or _argument_value(arguments, "--phase") != "calibration"
    ):
        raise RuntimeError("v8 external calibration identity changed")
    _require_flag_once(arguments, "--source-admission-passed")
    _require_flag_once(arguments, "--prediction-only-input")
    for forbidden in (
        "--fresh-parent-lock",
        "--sota-protocol",
        "--held-calibration-lock",
    ):
        if forbidden in arguments:
            raise RuntimeError(
                "v8 external calibration cannot combine admission protocols"
            )
    if int(_argument_value(arguments, "--canonical-node-count")) != int(
        physical.CANONICAL_NODE_COUNT
    ):
        raise RuntimeError("v8 external calibration graph capacity changed")

    parent_arguments = list(sys.argv)
    if (
        _argument_value(parent_arguments, "--case-name") != case_name
        or _argument_value(parent_arguments, "--role") != "calibration"
    ):
        raise RuntimeError("v8 physical parent authorized another case or role")
    lock_path = _argument_value(parent_arguments, "--lock")
    frame_zero_path = _argument_value(parent_arguments, "--frame-zero-manifest")
    lock = v8_protocol.validate_protocol_lock(lock_path)
    if (
        lock.get("stage") != "calibration"
        or lock.get("confirmation_access_authorized") is not False
        or lock.get("calibration_case_whitelist", []).count(case_name) != 1
        or lock.get("immutable_bindings", {}).get(
            "replacement_automatic_twin_admission_contract"
        )
        != V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
    ):
        raise RuntimeError("v8 lock does not bind the exact external admission")
    frame_zero_manifest = v8_protocol.validate_frame_zero_bundle_manifest(
        frame_zero_path,
        lock_path,
        expected_case_name=case_name,
        expected_role="calibration",
    )
    if (
        frame_zero_manifest.get("object_id") != object_id
        or int(frame_zero_manifest.get("episode_id", -1)) != episode_id
    ):
        raise RuntimeError("v8 replacement frame zero changed identity")

    script_path = Path(os.path.abspath(os.fspath(script)))
    roots = tuple(Path(os.path.abspath(os.fspath(root))) for root in import_roots)
    if (
        len(roots) != 2
        or script_path
        != roots[0].parent
        / "scripts"
        / "remote"
        / "build_deform360_automatic_episode_twin.py"
        or Path(_argument_value(arguments, "--repo")) != roots[0].parent
        or Path(_argument_value(parent_arguments, "--upstream-repo")) != roots[0].parent
        or Path(_argument_value(parent_arguments, "--deform360-repo")) != roots[1]
        or provenance_root is not None
    ):
        raise RuntimeError("v8 external admission runtime provenance changed")


def _v8_isolated_runpy_command(
    python: str | Path,
    script: str | Path,
    *,
    import_roots: list[str | Path] | tuple[str | Path, ...],
    arguments: list[str] | tuple[str, ...],
    provenance_root: str | Path | None = None,
) -> list[str]:
    """Use the exact-case admission bootstrap and preserve every legacy command."""

    adapted = list(arguments)
    if Path(script).name == "build_deform360_automatic_episode_twin.py":
        object_id = _argument_value(adapted, "--object-id")
        try:
            episode_id = int(_argument_value(adapted, "--episode-id"))
        except ValueError as error:
            raise RuntimeError("automatic-twin episode is not an integer") from error
        case_name = f"{object_id}-ep{episode_id:04d}"
        if case_name == V8_EXTERNAL_CALIBRATION_CASE_NAME:
            _authorize_external_admission_command(
                script,
                import_roots=import_roots,
                arguments=adapted,
                provenance_root=provenance_root,
            )
            command = _V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
                python,
                script,
                import_roots=import_roots,
                arguments=adapted,
                provenance_root=provenance_root,
            )
            if command.count(physical._ISOLATED_RUNPY_BOOTSTRAP) != 1:
                raise RuntimeError("frozen automatic-twin bootstrap command changed")
            return [
                (
                    _V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP
                    if value == physical._ISOLATED_RUNPY_BOOTSTRAP
                    else value
                )
                for value in command
            ]
    return _V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        python,
        script,
        import_roots=import_roots,
        arguments=adapted,
        provenance_root=provenance_root,
    )


def _v8_validate_inadmissible_automatic_twin(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    state_path: str | Path,
    twin_summary_path: str | Path,
    *,
    case_name: str,
    object_id: str,
    episode_id: int,
    role: str,
) -> dict[str, Any]:
    """Expect the distinct v8 admission identity only for exact-case fallback."""

    if case_name != V8_EXTERNAL_CALIBRATION_CASE_NAME:
        return _V7_PHYSICAL_INADMISSIBLE_TWIN_VALIDATOR(
            prediction_data_path,
            simulator_data_path,
            graph_path,
            state_path,
            twin_summary_path,
            case_name=case_name,
            object_id=object_id,
            episode_id=episode_id,
            role=role,
        )
    if (
        object_id != V8_EXTERNAL_CALIBRATION_OBJECT_ID
        or int(episode_id) != V8_EXTERNAL_CALIBRATION_EPISODE_ID
        or role != "calibration"
    ):
        raise ValueError("external admission fallback changed case or role")
    if (
        physical.AUTOMATIC_TWIN_PROTOCOL_ID != _V7_AUTOMATIC_TWIN_PROTOCOL_ID
        or physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        != _V7_AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
    ):
        raise RuntimeError("automatic-twin validation identity was already patched")
    physical.AUTOMATIC_TWIN_PROTOCOL_ID = V8_EXTERNAL_ADMISSION_PROTOCOL_ID
    physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 = (
        V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
    )
    try:
        return _V7_PHYSICAL_INADMISSIBLE_TWIN_VALIDATOR(
            prediction_data_path,
            simulator_data_path,
            graph_path,
            state_path,
            twin_summary_path,
            case_name=case_name,
            object_id=object_id,
            episode_id=episode_id,
            role=role,
        )
    finally:
        observed_protocol = physical.AUTOMATIC_TWIN_PROTOCOL_ID
        observed_config = physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        physical.AUTOMATIC_TWIN_PROTOCOL_ID = _V7_AUTOMATIC_TWIN_PROTOCOL_ID
        physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 = (
            _V7_AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        )
        if (
            observed_protocol != V8_EXTERNAL_ADMISSION_PROTOCOL_ID
            or observed_config != V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
        ):
            raise RuntimeError(
                "v8 external admission validation hooks changed during execution"
            )


def _v8_build_physical_prediction_archive(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    readout_path: str | Path,
    twin_summary_path: str | Path,
    driven_result_path: str | Path,
    zero_result_path: str | Path,
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    case_name: str,
    role: str,
    runtime_provenance: Mapping[str, Any],
    stage_runtime_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Require truthful v8 admission identity on exact-case successful twins."""

    if case_name == V8_EXTERNAL_CALIBRATION_CASE_NAME:
        if role != "calibration":
            raise ValueError("external admission archive changed role")
        twin = physical._load_json(twin_summary_path)
        if not (
            twin.get("protocol_id") == V8_EXTERNAL_ADMISSION_PROTOCOL_ID
            and twin.get("protocol_config_sha256")
            == V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
            and twin.get("object_id") == V8_EXTERNAL_CALIBRATION_OBJECT_ID
            and int(twin.get("episode_id", -1)) == V8_EXTERNAL_CALIBRATION_EPISODE_ID
            and twin.get("phase") == "calibration"
            and twin.get("passed") is True
            and twin.get("result_sha256") == physical._upstream_result_sha256(twin)
        ):
            raise ValueError(
                "successful external automatic twin lacks exact v8 admission"
            )
    return _V7_PHYSICAL_PREDICTION_ARCHIVE_BUILDER(
        prediction_data_path,
        simulator_data_path,
        graph_path,
        readout_path,
        twin_summary_path,
        driven_result_path,
        zero_result_path,
        archive_path,
        manifest_path,
        frame_zero_manifest_path=frame_zero_manifest_path,
        lock_path=lock_path,
        case_name=case_name,
        role=role,
        runtime_provenance=runtime_provenance,
        stage_runtime_seconds=stage_runtime_seconds,
    )


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
            _patch(
                physical,
                "_isolated_runpy_command",
                _v8_isolated_runpy_command,
                saved,
            )
            _patch(
                physical,
                "_validate_inadmissible_automatic_twin",
                _v8_validate_inadmissible_automatic_twin,
                saved,
            )
            _patch(
                physical,
                "build_physical_prediction_archive",
                _v8_build_physical_prediction_archive,
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
    "V8_EXTERNAL_ADMISSION_CONTRACT_SHA256",
    "V8_EXTERNAL_ADMISSION_PROTOCOL_ID",
    "V8_EXTERNAL_CALIBRATION_CASE_NAME",
    "explicit_v8_builder_context",
    "main_for",
    "run_v8_adapted_cli",
    "semantic_label_for_v8_object_id",
]
