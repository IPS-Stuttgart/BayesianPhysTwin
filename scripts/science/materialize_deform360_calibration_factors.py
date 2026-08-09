#!/usr/bin/env python3
"""Build grouped Deform360 contact anchors and calibration factor bundles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_calibration_factor_materializer import (
    build_deform360_kinematic_contact_anchor,
    materialize_deform360_calibration_factors,
    publish_deform360_calibration_factor_materialization,
)
from bayesian_phystwin.deform360_contact_anchor import (
    load_deform360_contact_anchor,
    save_deform360_contact_anchor,
)
from bayesian_phystwin.observation_belief import load_observation_belief
from bayesian_phystwin.physical_linearization import load_physical_linearization


def _ordinary_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _json(path: Path, *, name: str) -> Any:
    source = _ordinary_file(path, name=name)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {name}: {path}") from error


def _string_list(path: Path, *, name: str) -> tuple[str, ...]:
    value = _json(path, name=name)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty JSON string list")
    return tuple(value)


def _mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    value = _json(path, name=name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON mapping")
    return value


def _npy(path: Path, *, name: str) -> np.ndarray:
    source = _ordinary_file(path, name=name)
    if source.suffix.lower() != ".npy":
        raise ValueError(f"{name} must be an ordinary .npy file")
    try:
        value = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {name}: {path}") from error
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain one real numeric array")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    anchor = subparsers.add_parser(
        "contact-anchor",
        help="reduce public tactile/robot prefix evidence to grouped metric rows",
    )
    anchor.add_argument("--object-id", required=True)
    anchor.add_argument("--observation-case-id", required=True)
    anchor.add_argument("--episode-id", type=int, required=True)
    anchor.add_argument("--causal-frame-stop", type=int, required=True)
    anchor.add_argument("--frame-ids", type=Path, required=True)
    anchor.add_argument("--sensor-names", type=Path, required=True)
    anchor.add_argument("--contact-episode-ids", type=Path, required=True)
    anchor.add_argument("--tactile-response", type=Path, required=True)
    anchor.add_argument("--taxel-world-positions", type=Path, required=True)
    anchor.add_argument("--physical-patch-prediction", type=Path, required=True)
    anchor.add_argument("--state-jacobian", type=Path, required=True)
    anchor.add_argument("--source-reliability", type=Path, required=True)
    anchor.add_argument("--source-revision", required=True)
    anchor.add_argument("--source-artifacts", type=Path, required=True)
    anchor.add_argument("--metadata", type=Path)
    anchor.add_argument("--output", type=Path, required=True)

    posterior = subparsers.add_parser(
        "posterior",
        help="materialize claim-bearing visual and visual-plus-contact precisions",
    )
    posterior.add_argument("--observation-belief", type=Path, required=True)
    posterior.add_argument("--physical-linearization", type=Path, required=True)
    posterior.add_argument("--physical-prediction", type=Path, required=True)
    posterior.add_argument("--contact-anchor", type=Path, required=True)
    posterior.add_argument("--physical-query-jacobian", type=Path, required=True)
    posterior.add_argument("--state-prior-covariance", type=Path)
    posterior.add_argument("--metadata", type=Path)
    posterior.add_argument("--output-dir", type=Path, required=True)
    return parser


def _contact_anchor(arguments: argparse.Namespace) -> int:
    if os.path.lexists(arguments.output):
        raise ValueError(f"output already exists: {arguments.output}")
    metadata = (
        None
        if arguments.metadata is None
        else _mapping(arguments.metadata, name="metadata")
    )
    anchor = build_deform360_kinematic_contact_anchor(
        object_id=arguments.object_id,
        observation_case_id=arguments.observation_case_id,
        episode_id=arguments.episode_id,
        causal_frame_stop=arguments.causal_frame_stop,
        frame_ids=_npy(arguments.frame_ids, name="frame IDs"),
        sensor_names=_string_list(arguments.sensor_names, name="sensor names"),
        contact_episode_ids=_string_list(
            arguments.contact_episode_ids,
            name="contact episode IDs",
        ),
        tactile_response=_npy(
            arguments.tactile_response,
            name="tactile response",
        ),
        taxel_world_positions_m=_npy(
            arguments.taxel_world_positions,
            name="taxel world positions",
        ),
        physical_patch_prediction_m=_npy(
            arguments.physical_patch_prediction,
            name="physical patch prediction",
        ),
        state_jacobian=_npy(arguments.state_jacobian, name="state Jacobian"),
        source_reliability=_npy(
            arguments.source_reliability,
            name="source reliability",
        ),
        source_revision=arguments.source_revision,
        source_artifacts=_mapping(
            arguments.source_artifacts,
            name="source artifacts",
        ),
        metadata=metadata,
    )
    save_deform360_contact_anchor(arguments.output, anchor)
    loaded = load_deform360_contact_anchor(arguments.output)
    if loaded.artifact_id != anchor.artifact_id:
        raise ValueError("published contact anchor failed self-verification")
    print(json.dumps(anchor.summary(), sort_keys=True))
    return 0


def _posterior(arguments: argparse.Namespace) -> int:
    metadata = (
        None
        if arguments.metadata is None
        else _mapping(arguments.metadata, name="metadata")
    )
    observation_belief = _ordinary_file(
        arguments.observation_belief,
        name="observation belief",
    )
    physical_linearization = _ordinary_file(
        arguments.physical_linearization,
        name="physical linearization",
    )
    contact_anchor = _ordinary_file(
        arguments.contact_anchor,
        name="contact anchor",
    )
    materialization = materialize_deform360_calibration_factors(
        load_observation_belief(observation_belief),
        load_physical_linearization(physical_linearization),
        load_deform360_contact_anchor(contact_anchor),
        physical_prediction_xyz_m=_npy(
            arguments.physical_prediction,
            name="physical prediction",
        ),
        physical_query_jacobian=_npy(
            arguments.physical_query_jacobian,
            name="physical query Jacobian",
        ),
        state_prior_covariance_m2=(
            None
            if arguments.state_prior_covariance is None
            else _npy(
                arguments.state_prior_covariance,
                name="state prior covariance",
            )
        ),
        metadata=metadata,
    )
    publish_deform360_calibration_factor_materialization(
        arguments.output_dir,
        materialization,
        load_deform360_contact_anchor(contact_anchor),
    )
    print(json.dumps(materialization.to_record(), sort_keys=True))
    return 0 if materialization.observability_evaluable else 3


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "contact-anchor":
            return _contact_anchor(arguments)
        if arguments.command == "posterior":
            return _posterior(arguments)
        raise ValueError(f"unsupported command: {arguments.command}")
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
