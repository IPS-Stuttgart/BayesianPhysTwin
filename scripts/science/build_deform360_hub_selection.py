#!/usr/bin/env python3
"""Build a target-blind fresh-object Deform360 selection from the official Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-official-hub-visuotactile-protocol"
SELECTION_SCHEMA = "bayesian-phystwin/deform360-official-hub-selection-v1"
_REQUIRED_STRATA = ("sheet", "volumetric")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object: {path}") from error
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _require_revision(value: object, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) in {40, 64}
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be an exact lowercase revision",
    )
    return result


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the locked official-Hub information boundary."""

    protocol = _load_json(path.resolve())
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(protocol.get("schema_version") == 1, "unsupported protocol version")
    _require(
        protocol.get("status") == "locked-before-official-raw-payload-access",
        "protocol must remain locked before raw payload access",
    )
    boundary = protocol.get("information_boundary")
    _require(isinstance(boundary, dict), "information_boundary must be an object")
    _require(
        boundary.get("object_directory_names_allowed") is True
        and boundary.get("object_metadata_json_allowed") is True
        and boundary.get("camera_media_opened") is False
        and boundary.get("tactile_arrays_opened") is False
        and boundary.get("target_outcomes_opened") is False,
        "official-Hub information boundary changed",
    )
    selection = protocol.get("selection")
    _require(isinstance(selection, dict), "selection must be an object")
    _require(
        tuple(selection.get("strata", ())) == _REQUIRED_STRATA,
        "only sheet and volumetric strata are registered",
    )
    for name in ("calibration_objects_per_stratum", "confirmation_objects_per_stratum"):
        value = selection.get(name)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"{name} must be a positive integer",
        )
    _require(
        isinstance(selection.get("seed"), str) and selection["seed"],
        "selection seed is missing",
    )
    cache = protocol.get("cache_preflight")
    _require(isinstance(cache, dict), "cache_preflight must be an object")
    _require_sha256(cache.get("inventory_sha256"), name="cache inventory SHA-256")
    _require_sha256(
        cache.get("content_inventory_sha256"),
        name="cache content inventory SHA-256",
    )
    excluded = cache.get("excluded_candidate_objects")
    _require(
        isinstance(excluded, list)
        and len(excluded) == len(set(excluded))
        and all(isinstance(value, str) and value for value in excluded),
        "cache excluded_candidate_objects must be unique nonempty strings",
    )
    return protocol


def _cohort_objects(value: object) -> set[str]:
    _require(isinstance(value, Mapping), "cohort must be an object")
    result: set[str] = set()
    for records in value.values():
        _require(isinstance(records, list), "cohort strata must contain lists")
        for record in records:
            _require(isinstance(record, Mapping), "cohort records must be objects")
            object_id = record.get("object_id")
            _require(
                isinstance(object_id, str) and object_id,
                "cohort object_id must be a nonempty string",
            )
            result.add(object_id)
    return result


def load_prior_context(
    repository: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], set[str], dict[str, dict[str, str]]]:
    """Load the frozen object vocabulary and every previously opened identity."""

    repo = repository.resolve()
    prior = protocol.get("prior_protocols")
    _require(isinstance(prior, Mapping), "prior_protocols must be an object")
    _require(set(prior) == {"v1", "v2"}, "expected v1 and v2 prior protocols")
    loaded: dict[str, Mapping[str, Any]] = {}
    source_records: dict[str, dict[str, str]] = {}
    for name, record in prior.items():
        _require(
            isinstance(record, Mapping),
            f"prior protocol {name} must be an object",
        )
        relative_path = record.get("path")
        _require(
            isinstance(relative_path, str) and relative_path,
            f"prior protocol {name} path is missing",
        )
        expected = _require_sha256(
            record.get("expected_config_sha256"),
            name=f"prior protocol {name} config SHA-256",
        )
        payload = _load_json(repo / relative_path)
        _require(
            payload.get("config_sha256") == expected,
            f"prior protocol {name} config checksum changed",
        )
        config = payload.get("config")
        _require(isinstance(config, Mapping), f"prior protocol {name} lacks config")
        loaded[str(name)] = config
        source_records[str(name)] = {
            "path": relative_path,
            "config_sha256": expected,
        }

    v1 = loaded["v1"]
    pools = v1.get("candidate_pools")
    _require(isinstance(pools, Mapping), "v1 candidate pools are missing")
    candidate_pools: dict[str, tuple[str, ...]] = {}
    for stratum in _REQUIRED_STRATA:
        values = pools.get(stratum)
        _require(isinstance(values, list) and values, f"missing {stratum} pool")
        _require(
            len(values) == len(set(values))
            and all(isinstance(value, str) and value for value in values),
            f"{stratum} candidate pool must contain unique nonempty strings",
        )
        candidate_pools[stratum] = tuple(values)

    excluded = set(map(str, v1.get("open_or_reserved_objects", ())))
    for config in loaded.values():
        excluded.update(_cohort_objects(config.get("calibration_cohort", {})))
        excluded.update(_cohort_objects(config.get("target_cohort", {})))
    excluded.update(map(str, protocol["cache_preflight"]["excluded_candidate_objects"]))
    return candidate_pools, excluded, source_records


def _rank(seed: str, *parts: object) -> str:
    return hashlib.sha256(
        ":".join((seed, *(str(part) for part in parts))).encode("utf-8")
    ).hexdigest()


def select_objects(
    available_objects: Sequence[str],
    *,
    candidate_pools: Mapping[str, Sequence[str]],
    excluded_objects: set[str],
    selection: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Select calibration and confirmation objects from names only."""

    available = set(available_objects)
    _require(
        len(available) == len(tuple(available_objects)),
        "official raw object names contain duplicates",
    )
    seed = str(selection["seed"])
    calibration_count = int(selection["calibration_objects_per_stratum"])
    confirmation_count = int(selection["confirmation_objects_per_stratum"])
    result: dict[str, list[dict[str, str]]] = {
        "calibration": [],
        "confirmation": [],
    }
    for stratum in _REQUIRED_STRATA:
        eligible = sorted(
            set(candidate_pools[stratum]) & available - excluded_objects,
            key=lambda object_id: (
                _rank(seed, "object", stratum, object_id),
                object_id,
            ),
        )
        required = calibration_count + confirmation_count
        _require(
            len(eligible) >= required,
            f"{stratum} has {len(eligible)} eligible objects but requires {required}",
        )
        for role, selected in (
            ("calibration", eligible[:calibration_count]),
            ("confirmation", eligible[calibration_count:required]),
        ):
            result[role].extend(
                {"object_id": object_id, "stratum": stratum} for object_id in selected
            )
    return result


def _episode_ids(metadata: Mapping[str, Any], *, object_id: str) -> tuple[int, ...]:
    sequences = metadata.get("sequences")
    _require(
        isinstance(sequences, (Mapping, list)),
        f"{object_id} metadata has no sequences",
    )
    raw_ids: Sequence[object]
    if isinstance(sequences, Mapping):
        raw_ids = tuple(sequences.keys())
    else:
        raw_ids = tuple(range(len(sequences)))
    result: list[int] = []
    for value in raw_ids:
        _require(not isinstance(value, bool), f"{object_id} episode ID is Boolean")
        if isinstance(value, int):
            episode_id = value
        elif isinstance(value, str) and value.isdigit():
            episode_id = int(value)
        else:
            raise ValueError(f"{object_id} episode ID is not an integer: {value!r}")
        _require(episode_id >= 0, f"{object_id} episode ID must be nonnegative")
        result.append(episode_id)
    _require(result, f"{object_id} metadata contains no episodes")
    _require(len(result) == len(set(result)), f"{object_id} episode IDs are duplicated")
    return tuple(sorted(result))


def bind_episodes(
    object_selection: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    metadata_by_object: Mapping[str, Mapping[str, Any]],
    metadata_sha256_by_object: Mapping[str, str],
    seed: str,
) -> dict[str, list[dict[str, object]]]:
    """Choose one episode per object after metadata-only object selection."""

    result: dict[str, list[dict[str, object]]] = {}
    for role in ("calibration", "confirmation"):
        records: list[dict[str, object]] = []
        for item in object_selection[role]:
            object_id = item["object_id"]
            _require(
                object_id in metadata_by_object,
                f"metadata is missing for selected object {object_id}",
            )
            digest = _require_sha256(
                metadata_sha256_by_object.get(object_id),
                name=f"{object_id} metadata SHA-256",
            )
            episode_ids = _episode_ids(
                metadata_by_object[object_id], object_id=object_id
            )
            episode_id = min(
                episode_ids,
                key=lambda value: (
                    _rank(seed, "episode", role, object_id, value),
                    value,
                ),
            )
            records.append(
                {
                    "object_id": object_id,
                    "stratum": item["stratum"],
                    "episode_id": episode_id,
                    "metadata_path": f"raw/{object_id}/metadata.json",
                    "metadata_sha256": digest,
                }
            )
        result[role] = records
    return result


def build_selection(
    snapshot: Mapping[str, Any],
    *,
    repository: Path,
    protocol_path: Path,
    implementation_revision: str | None = None,
) -> dict[str, Any]:
    """Validate a names/metadata snapshot and build the sealed cohort proposal."""

    protocol = load_protocol(protocol_path)
    candidate_pools, excluded, prior_sources = load_prior_context(repository, protocol)
    resolved_revision = _require_revision(
        snapshot.get("resolved_revision"),
        name="official dataset resolved revision",
    )
    available = snapshot.get("raw_objects")
    _require(
        isinstance(available, list)
        and all(isinstance(value, str) and value for value in available),
        "snapshot raw_objects must be a list of nonempty strings",
    )
    metadata = snapshot.get("metadata_by_object")
    metadata_sha = snapshot.get("metadata_sha256_by_object")
    opened_paths = snapshot.get("opened_paths")
    _require(isinstance(metadata, Mapping), "snapshot metadata_by_object is missing")
    _require(
        isinstance(metadata_sha, Mapping),
        "snapshot metadata_sha256_by_object is missing",
    )
    _require(
        isinstance(opened_paths, list)
        and all(
            isinstance(path, str)
            and path.startswith("raw/")
            and path.endswith("/metadata.json")
            for path in opened_paths
        ),
        "snapshot opened_paths may contain only raw object metadata.json files",
    )
    object_selection = select_objects(
        available,
        candidate_pools=candidate_pools,
        excluded_objects=excluded,
        selection=protocol["selection"],
    )
    selected_ids = {
        item["object_id"] for records in object_selection.values() for item in records
    }
    _require(
        set(opened_paths)
        == {f"raw/{object_id}/metadata.json" for object_id in selected_ids},
        "metadata access does not match the selected object set",
    )
    bound = bind_episodes(
        object_selection,
        metadata_by_object=metadata,
        metadata_sha256_by_object=metadata_sha,
        seed=str(protocol["selection"]["seed"]),
    )
    _require(
        not ({item["object_id"] for item in bound["calibration"]} & excluded),
        "an excluded object entered calibration",
    )
    _require(
        not ({item["object_id"] for item in bound["confirmation"]} & excluded),
        "an excluded object entered confirmation",
    )
    _require(
        not (
            {item["object_id"] for item in bound["calibration"]}
            & {item["object_id"] for item in bound["confirmation"]}
        ),
        "calibration and confirmation objects overlap",
    )

    protocol_sha256 = _sha256_json(protocol)
    content: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "dataset": {
            "repo_id": protocol["dataset"]["repo_id"],
            "requested_revision": protocol["dataset"]["requested_revision"],
            "resolved_revision": resolved_revision,
            "raw_prefix": protocol["dataset"]["raw_prefix"],
        },
        "official_processing": protocol["official_processing"],
        "prior_protocols": prior_sources,
        "cache_preflight": protocol["cache_preflight"],
        "information_boundary": {
            "object_directory_names_opened": True,
            "object_metadata_json_opened": True,
            "opened_metadata_paths": sorted(opened_paths),
            "camera_media_opened": False,
            "tactile_arrays_opened": False,
            "robot_arrays_opened": False,
            "geometry_annotations_opened": False,
            "target_outcomes_opened": False,
        },
        "available_raw_object_count": len(available),
        "excluded_object_count": len(excluded),
        "selection": bound,
        "replacement_allowed_after_payload_access": False,
        "next_gate": (
            "commit this exact selection and its content SHA-256 before downloading "
            "or processing any selected raw camera, tactile, robot, or geometry payload"
        ),
    }
    content["selection_sha256"] = _sha256_json(bound)
    content["content_selection_sha256"] = _sha256_json(content)
    result = dict(content)
    result["implementation_revision"] = (
        None
        if implementation_revision is None
        else _require_revision(
            implementation_revision,
            name="BayesianPhysTwin implementation revision",
        )
    )
    result["selection_artifact_sha256"] = _sha256_json(result)
    return result


def _load_snapshot(path: Path) -> dict[str, Any]:
    return _load_json(path.resolve())


def _download_official_snapshot(
    protocol: Mapping[str, Any],
    *,
    candidate_pools: Mapping[str, Sequence[str]],
    excluded: set[str],
) -> dict[str, Any]:
    """Read only official object names and selected metadata JSON files."""

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "live official-Hub inventory requires the huggingface_hub package"
        ) from error

    dataset = protocol["dataset"]
    repo_id = str(dataset["repo_id"])
    requested_revision = str(dataset["requested_revision"])
    raw_prefix = str(dataset["raw_prefix"]).rstrip("/")
    api = HfApi()
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=requested_revision,
        files_metadata=False,
    )
    resolved_revision = _require_revision(info.sha, name="Hub dataset revision")
    entries = api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=raw_prefix,
        recursive=False,
        expand=False,
        repo_type="dataset",
        revision=resolved_revision,
    )
    available: list[str] = []
    for entry in entries:
        path = str(getattr(entry, "path", ""))
        if path.startswith(f"{raw_prefix}/") and path.count("/") == 1:
            available.append(path.split("/", 1)[1])
    available = sorted(set(available))
    object_selection = select_objects(
        available,
        candidate_pools=candidate_pools,
        excluded_objects=excluded,
        selection=protocol["selection"],
    )
    selected_ids = sorted(
        {item["object_id"] for records in object_selection.values() for item in records}
    )
    metadata_by_object: dict[str, Any] = {}
    metadata_sha256_by_object: dict[str, str] = {}
    opened_paths: list[str] = []
    for object_id in selected_ids:
        metadata_path = f"{raw_prefix}/{object_id}/metadata.json"
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=metadata_path,
                repo_type="dataset",
                revision=resolved_revision,
            )
        )
        raw = local_path.read_bytes()
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid official metadata JSON: {metadata_path}"
            ) from error
        _require(
            isinstance(metadata, dict),
            f"official metadata must be an object: {metadata_path}",
        )
        metadata_by_object[object_id] = metadata
        metadata_sha256_by_object[object_id] = hashlib.sha256(raw).hexdigest()
        opened_paths.append(metadata_path)
    return {
        "resolved_revision": resolved_revision,
        "raw_objects": available,
        "metadata_by_object": metadata_by_object,
        "metadata_sha256_by_object": metadata_sha256_by_object,
        "opened_paths": opened_paths,
    }


def write_selection(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot-json",
        type=Path,
        help="offline names/metadata fixture; omit for the official Hugging Face Hub",
    )
    parser.add_argument(
        "--implementation-revision",
        default=(
            os.environ.get("BPT_IMPLEMENTATION_REVISION")
            or os.environ.get("GITHUB_SHA")
        ),
        help=(
            "exact BayesianPhysTwin head revision; live PR workflows must pass "
            "github.event.pull_request.head.sha explicitly"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_protocol(args.protocol)
    candidate_pools, excluded, _ = load_prior_context(args.repository, protocol)
    snapshot = (
        _load_snapshot(args.snapshot_json)
        if args.snapshot_json is not None
        else _download_official_snapshot(
            protocol,
            candidate_pools=candidate_pools,
            excluded=excluded,
        )
    )
    result = build_selection(
        snapshot,
        repository=args.repository,
        protocol_path=args.protocol,
        implementation_revision=args.implementation_revision,
    )
    write_selection(args.output, result)
    summary = {
        "resolved_revision": result["dataset"]["resolved_revision"],
        "available_raw_object_count": result["available_raw_object_count"],
        "calibration_object_count": len(result["selection"]["calibration"]),
        "confirmation_object_count": len(result["selection"]["confirmation"]),
        "content_selection_sha256": result["content_selection_sha256"],
        "selection_artifact_sha256": result["selection_artifact_sha256"],
        "camera_media_opened": False,
        "tactile_arrays_opened": False,
        "target_outcomes_opened": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
