"""Fail-closed source admission and cohort locking for fresh Deform360 objects.

Admission reads metadata, provenance, split metadata, and the frame-zero PLY
header. It hashes but never deserializes ``final_data.pkl`` so future object
positions cannot influence cohort selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .deform360_official_parity import audit_parity_contract


SCHEMA_VERSION = 1
ADMISSION_KIND = "Deform360FreshSourceAdmission"
EXCLUSION_KIND = "Deform360FreshObjectExclusionManifest"
COHORT_LOCK_KIND = "Deform360FreshObjectCohortLock"
UPSTREAM_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"
UPSTREAM_BINDING = {
    "repository": "https://github.com/lhy0807/deform360",
    "revision": UPSTREAM_REVISION,
    "bound_files": {
        "README.md": (
            "52f2ebed1800eb8c1e6dde05fefaca15ebba4456f0756b1ca05cfc4380fc8f7a"
        ),
        "deform360/processing/control_points_stage.py": (
            "9ff82c86c22e38c56dd2ce5d872850afb6ffeb502da7338baf0b55108afb7373"
        ),
        "deform360/processing/pcd_stage.py": (
            "87553e1ea3dac5a90e46114c76aaf65901b43a064025626ae6871523065c864d"
        ),
    },
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")
_CATEGORY = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_OBJECT_HASH_NAMESPACE = b"deform360-fresh-object-exclusion-v1\0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload: Mapping[str, Any], *, digest_key: str) -> dict[str, Any]:
    sealed = json.loads(json.dumps(payload, allow_nan=False))
    sealed[digest_key] = _canonical_sha256(sealed, digest_key=digest_key)
    return sealed


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _integer_pair(value: Any, *, name: str) -> tuple[int, int]:
    _require(
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"{name} must be an integer pair",
    )
    return int(value[0]), int(value[1])


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(1024 * 1024)
    marker = b"end_header"
    _require(marker in header, "frame-zero PLY header is incomplete")
    text = header[: header.index(marker)].decode("ascii", errors="strict")
    _require(text.startswith("ply\n") or text.startswith("ply\r\n"), "invalid PLY")
    counts = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) == 3 and tokens[:2] == ["element", "vertex"]:
            try:
                counts.append(int(tokens[2]))
            except ValueError as exc:
                raise ValueError("invalid PLY vertex count") from exc
    _require(len(counts) == 1 and counts[0] >= 0, "PLY vertex count is ambiguous")
    return counts[0]


def _metadata_identity(
    metadata: Mapping[str, Any], episode_id: int
) -> tuple[str, bool]:
    metadata_object = metadata.get("object")
    _require(
        isinstance(metadata_object, str) and bool(metadata_object),
        "raw metadata object label is missing",
    )
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), "raw metadata sequences are missing")
    record = sequences.get(str(episode_id))
    _require(isinstance(record, Mapping), "raw metadata episode is missing")
    value = record.get("bimanual")
    _require(value in {"yes", "no"}, "bimanual must be exactly 'yes' or 'no'")
    return metadata_object, value == "yes"


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


def _valid_stage_inputs(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not _valid_digest(value.get("robot_sha256")):
        return False
    if not _valid_digest(value.get("pcd_sha256")):
        return False
    tactile = value.get("tactile_sha256")
    return isinstance(tactile, Mapping) and all(
        isinstance(name, str) and bool(name) and _valid_digest(digest)
        for name, digest in tactile.items()
    )


@dataclass(frozen=True)
class FreshSourceAdmissionConfig:
    minimum_camera_count: int = 3
    minimum_point_count: int = 128
    maximum_point_count: int = 10_000
    required_frame_count: int = 76
    update_frames: tuple[int, ...] = (19, 38, 57)
    minimum_test_frame_count: int = 8

    def __post_init__(self) -> None:
        _require(self.minimum_camera_count >= 2, "minimum camera count is too small")
        _require(self.minimum_point_count >= 3, "minimum point count is too small")
        _require(
            self.maximum_point_count >= self.minimum_point_count,
            "maximum point count precedes minimum",
        )
        _require(self.required_frame_count >= 2, "required frame count is too small")
        _require(bool(self.update_frames), "update frames are empty")
        _require(
            tuple(sorted(set(self.update_frames))) == self.update_frames,
            "update frames must be strictly increasing",
        )
        _require(self.update_frames[0] >= 0, "update frame is negative")
        _require(self.minimum_test_frame_count >= 1, "test frame count is too small")


def build_fresh_source_admission(
    episode_dir: str | Path,
    raw_metadata_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
    category: str,
    config: FreshSourceAdmissionConfig | None = None,
) -> dict[str, Any]:
    """Build one admission decision without deserializing future geometry."""

    cfg = config or FreshSourceAdmissionConfig()
    episode = Path(episode_dir).resolve()
    metadata_path = Path(raw_metadata_path).resolve()
    paths = {
        "metadata": metadata_path,
        "control_meta": episode / "control_points.meta.json",
        "split": episode / "split.json",
        "calibrate": episode / "calibrate.pkl",
        "frame_zero": episode / "start_obj_pcd.ply",
        "future_payload": episode / "final_data.pkl",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"required {name} file is missing: {path}")
    _require(bool(_OBJECT_ID.fullmatch(object_id)), "object ID is malformed")
    _require(episode_id >= 0, "episode ID is negative")
    _require(bool(_CATEGORY.fullmatch(category)), "category is malformed")

    metadata = _load_json(metadata_path)
    control_meta = _load_json(paths["control_meta"])
    split = _load_json(paths["split"])
    reasons: list[str] = []
    metadata_parent = metadata_path.parent.name
    if metadata_parent != object_id:
        reasons.append("raw metadata directory differs from requested object ID")

    try:
        metadata_object, bimanual = _metadata_identity(metadata, episode_id)
    except ValueError as exc:
        metadata_object = None
        bimanual = None
        reasons.append(str(exc))

    if control_meta.get("schema") != "deform360.processing/control-points/v1":
        reasons.append("control-point manifest schema differs from upstream v1")
    inputs = control_meta.get("inputs")
    if not _valid_stage_inputs(inputs):
        reasons.append("control-point input provenance is missing or malformed")
    outputs = control_meta.get("outputs")
    parameters = control_meta.get("parameters")
    if not isinstance(outputs, Mapping):
        outputs = {}
        reasons.append("control-point output provenance is missing")
    if not isinstance(parameters, Mapping):
        parameters = {}
        reasons.append("control-point parameters are missing")

    file_hashes = {name: _sha256(path) for name, path in paths.items()}
    expected_hashes = {
        "calibrate": outputs.get("calibrate_sha256"),
        "frame_zero": outputs.get("start_ply_sha256"),
        "split": outputs.get("split_sha256"),
        "future_payload": outputs.get("final_data_sha256"),
    }
    for name, expected in expected_hashes.items():
        if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
            reasons.append(f"{name} provenance digest is missing or malformed")
        elif file_hashes[name] != expected:
            reasons.append(f"{name} checksum differs from control-point provenance")

    cameras = parameters.get("cameras")
    if (
        not isinstance(cameras, list)
        or not all(isinstance(camera, str) and camera for camera in cameras)
        or len(cameras) != len(set(cameras))
    ):
        cameras = []
        reasons.append("camera panel is missing, malformed, or duplicated")
    if len(cameras) < cfg.minimum_camera_count:
        reasons.append("camera panel is below the preregistered minimum")
    train_fraction = parameters.get("train_fraction")
    if not isinstance(train_fraction, (int, float)) or isinstance(train_fraction, bool):
        train_fraction = None
        reasons.append("control-point train fraction is missing")
    elif float(train_fraction) != 0.8:
        reasons.append("control-point train fraction differs from released rule")

    try:
        vertex_count = _ply_vertex_count(paths["frame_zero"])
    except ValueError as exc:
        vertex_count = None
        reasons.append(str(exc))
    if vertex_count is not None and not (
        cfg.minimum_point_count <= vertex_count <= cfg.maximum_point_count
    ):
        reasons.append("frame-zero point count is outside backend admission")

    frame_len = split.get("frame_len")
    if not isinstance(frame_len, int) or isinstance(frame_len, bool):
        frame_len = None
        reasons.append("split frame_len is not an integer")
    try:
        train = _integer_pair(split.get("train"), name="train")
        test = _integer_pair(split.get("test"), name="test")
    except ValueError as exc:
        train = None
        test = None
        reasons.append(str(exc))
    active_frame_count = outputs.get("num_active_frames")
    if not isinstance(active_frame_count, int) or isinstance(active_frame_count, bool):
        active_frame_count = None
        reasons.append("num_active_frames is missing from provenance")
    contact_start = outputs.get("contact_start_frame")
    contact_end = outputs.get("contact_end_frame")
    if not (
        isinstance(contact_start, int)
        and not isinstance(contact_start, bool)
        and isinstance(contact_end, int)
        and not isinstance(contact_end, bool)
        and 0 <= contact_start <= contact_end
    ):
        contact_start = None
        contact_end = None
        reasons.append("contact-window provenance is missing or malformed")

    if frame_len is not None:
        if frame_len != cfg.required_frame_count:
            reasons.append("split frame count differs from the frozen method")
        if active_frame_count is not None and active_frame_count != frame_len:
            reasons.append(
                "split indexes the undropped contact window rather than final_data"
            )
        if (
            contact_start is not None
            and contact_end is not None
            and contact_end - contact_start + 1 != frame_len
        ):
            reasons.append("split frame count differs from the contact window")
    if frame_len is not None and train is not None and test is not None:
        expected_train_end = int(0.8 * frame_len)
        if not (
            train == (0, expected_train_end) and test == (expected_train_end, frame_len)
        ):
            reasons.append("split does not match the released contiguous 80/20 rule")
        if test[1] - test[0] < cfg.minimum_test_frame_count:
            reasons.append("test partition is too short")
        if cfg.update_frames[-1] >= train[1]:
            reasons.append("a frozen online update falls outside the train prefix")

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ADMISSION_KIND,
        "upstream_binding": UPSTREAM_BINDING,
        "case": f"{object_id}-ep{episode_id:04d}",
        "object_id": object_id,
        "episode_id": episode_id,
        "category": category,
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "config": asdict(cfg),
        "observed_source_contract": {
            "metadata_parent": metadata_parent,
            "metadata_object": metadata_object,
            "bimanual": bimanual,
            "camera_count": len(cameras),
            "cameras": cameras,
            "frame_zero_point_count": vertex_count,
            "split_frame_count": frame_len,
            "active_frame_count": active_frame_count,
            "contact_start_frame": contact_start,
            "contact_end_frame": contact_end,
            "train_fraction": train_fraction,
            "stage_inputs_valid": _valid_stage_inputs(inputs),
            "train": None if train is None else list(train),
            "test": None if test is None else list(test),
        },
        "source_files": {
            name: {
                "basename": path.name,
                "sha256": file_hashes[name],
            }
            for name, path in paths.items()
        },
        "information_boundary": {
            "future_object_positions_deserialized": False,
            "future_payload_bytes_hashed": True,
            "future_metrics_read": False,
            "selection_inputs": (
                "raw object/episode identity and enums, stage input/output "
                "provenance hashes/counts, contact-window and split indices, "
                "camera names, and frame-zero PLY vertex count only"
            ),
        },
    }
    return _seal(artifact, digest_key="admission_sha256")


def validate_fresh_source_admission(artifact: Mapping[str, Any]) -> None:
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "wrong schema")
    _require(artifact.get("artifact_kind") == ADMISSION_KIND, "wrong artifact kind")
    _require(
        artifact.get("admission_sha256")
        == _canonical_sha256(artifact, digest_key="admission_sha256"),
        "admission checksum changed",
    )
    _require(
        artifact.get("upstream_binding") == UPSTREAM_BINDING,
        "admission upstream binding changed",
    )
    object_id = artifact.get("object_id")
    episode_id = artifact.get("episode_id")
    category = artifact.get("category")
    _require(
        isinstance(object_id, str) and bool(_OBJECT_ID.fullmatch(object_id)),
        "admission object ID is malformed",
    )
    _require(
        isinstance(episode_id, int)
        and not isinstance(episode_id, bool)
        and episode_id >= 0,
        "admission episode ID is malformed",
    )
    _require(
        isinstance(category, str) and bool(_CATEGORY.fullmatch(category)),
        "admission category is malformed",
    )
    _require(
        artifact.get("case") == f"{object_id}-ep{episode_id:04d}",
        "admission case identity is inconsistent",
    )
    boundary = artifact.get("information_boundary")
    _require(isinstance(boundary, Mapping), "information boundary is missing")
    _require(
        boundary.get("future_object_positions_deserialized") is False,
        "admission read future positions",
    )
    _require(
        boundary.get("future_metrics_read") is False,
        "admission read future metrics",
    )
    _require(
        boundary.get("future_payload_bytes_hashed") is True,
        "admission did not bind the future payload bytes",
    )
    accepted = artifact.get("accepted")
    reasons = artifact.get("rejection_reasons")
    _require(isinstance(accepted, bool), "admission decision is malformed")
    _require(
        isinstance(reasons, list)
        and all(isinstance(reason, str) and bool(reason) for reason in reasons)
        and len(reasons) == len(set(reasons)),
        "admission reasons are malformed",
    )
    _require(accepted == (len(reasons) == 0), "admission decision is inconsistent")
    source_files = artifact.get("source_files")
    _require(
        isinstance(source_files, Mapping)
        and set(source_files)
        == {
            "metadata",
            "control_meta",
            "split",
            "calibrate",
            "frame_zero",
            "future_payload",
        }
        and all(
            isinstance(record, Mapping)
            and isinstance(record.get("basename"), str)
            and bool(record["basename"])
            and _valid_digest(record.get("sha256"))
            for record in source_files.values()
        ),
        "admission source-file binding is malformed",
    )
    if accepted:
        config = artifact.get("config")
        frozen_config = json.loads(
            json.dumps(asdict(FreshSourceAdmissionConfig()), allow_nan=False)
        )
        _require(
            config == frozen_config, "accepted admission changed the frozen config"
        )
        observed = artifact.get("observed_source_contract")
        _require(
            isinstance(observed, Mapping),
            "accepted admission source contract is missing",
        )
        cameras = observed.get("cameras")
        _require(
            observed.get("metadata_parent") == object_id,
            "accepted admission metadata directory is inconsistent",
        )
        _require(
            isinstance(observed.get("metadata_object"), str)
            and bool(observed["metadata_object"]),
            "accepted admission metadata label is malformed",
        )
        _require(
            isinstance(observed.get("bimanual"), bool),
            "accepted admission bimanual value is malformed",
        )
        _require(
            isinstance(cameras, list)
            and all(isinstance(camera, str) and bool(camera) for camera in cameras)
            and len(cameras) == len(set(cameras))
            and observed.get("camera_count") == len(cameras)
            and len(cameras) >= FreshSourceAdmissionConfig().minimum_camera_count,
            "accepted admission camera panel violates the frozen contract",
        )
        point_count = observed.get("frame_zero_point_count")
        _require(
            isinstance(point_count, int)
            and not isinstance(point_count, bool)
            and frozen_config["minimum_point_count"]
            <= point_count
            <= frozen_config["maximum_point_count"],
            "accepted admission point count violates the frozen contract",
        )
        frame_count = frozen_config["required_frame_count"]
        _require(
            observed.get("split_frame_count")
            == observed.get("active_frame_count")
            == frame_count,
            "accepted admission row count violates the frozen contract",
        )
        contact_start = observed.get("contact_start_frame")
        contact_end = observed.get("contact_end_frame")
        _require(
            isinstance(contact_start, int)
            and not isinstance(contact_start, bool)
            and isinstance(contact_end, int)
            and not isinstance(contact_end, bool)
            and contact_end - contact_start + 1 == frame_count,
            "accepted admission contact window violates the frozen contract",
        )
        expected_train_end = int(0.8 * frame_count)
        _require(
            observed.get("train") == [0, expected_train_end]
            and observed.get("test") == [expected_train_end, frame_count]
            and observed.get("train_fraction") == 0.8,
            "accepted admission split violates the frozen contract",
        )
        _require(
            observed.get("stage_inputs_valid") is True,
            "accepted admission lacks valid source-stream provenance",
        )


def object_exclusion_hash(object_id: str) -> str:
    _require(bool(object_id), "object ID is empty")
    return hashlib.sha256(
        _OBJECT_HASH_NAMESPACE + object_id.encode("utf-8")
    ).hexdigest()


def build_object_exclusion_manifest(
    object_ids: Sequence[str],
    *,
    owner: str,
    source_artifact_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Build a target-free object exclusion set without exposing object IDs."""

    _require(bool(owner), "exclusion owner is empty")
    _require(bool(object_ids), "exclusion object set is empty")
    _require(
        len(object_ids) == len(set(object_ids)),
        "duplicate exclusion object ID",
    )
    hashes = sorted({object_exclusion_hash(object_id) for object_id in object_ids})
    sources = sorted(set(source_artifact_sha256s))
    _require(
        bool(sources)
        and all(
            isinstance(value, str) and _HEX64.fullmatch(value) for value in sources
        ),
        "exclusion source digest is malformed",
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": EXCLUSION_KIND,
        "hash_namespace": _OBJECT_HASH_NAMESPACE.decode("ascii").rstrip("\0"),
        "owner": owner,
        "object_hashes": hashes,
        "source_artifact_sha256s": sources,
        "information_boundary": {
            "target_artifact_read": False,
            "object_ids_emitted": False,
        },
    }
    return _seal(artifact, digest_key="exclusion_sha256")


def validate_object_exclusion_manifest(artifact: Mapping[str, Any]) -> None:
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "wrong schema")
    _require(artifact.get("artifact_kind") == EXCLUSION_KIND, "wrong artifact kind")
    _require(
        artifact.get("exclusion_sha256")
        == _canonical_sha256(artifact, digest_key="exclusion_sha256"),
        "exclusion checksum changed",
    )
    _require(
        artifact.get("hash_namespace")
        == _OBJECT_HASH_NAMESPACE.decode("ascii").rstrip("\0"),
        "exclusion hash namespace changed",
    )
    _require(
        isinstance(artifact.get("owner"), str) and bool(artifact["owner"]),
        "exclusion owner is malformed",
    )
    hashes = artifact.get("object_hashes")
    _require(
        isinstance(hashes, list)
        and hashes == sorted(set(hashes))
        and all(isinstance(value, str) and _HEX64.fullmatch(value) for value in hashes),
        "exclusion hashes are malformed",
    )
    sources = artifact.get("source_artifact_sha256s")
    _require(
        isinstance(sources, list)
        and bool(sources)
        and sources == sorted(set(sources))
        and all(_valid_digest(value) for value in sources),
        "exclusion source digests are malformed",
    )
    _require(
        artifact.get("information_boundary", {}).get("target_artifact_read") is False,
        "exclusion manifest crossed the target boundary",
    )
    _require(
        artifact.get("information_boundary", {}).get("object_ids_emitted") is False,
        "exclusion manifest emitted protected object IDs",
    )


def _select_round_robin(
    admissions: Sequence[Mapping[str, Any]], cohort_size: int
) -> list[Mapping[str, Any]]:
    per_object: dict[str, Mapping[str, Any]] = {}
    for admission in admissions:
        object_id = str(admission["object_id"])
        current = per_object.get(object_id)
        if current is None or int(admission["episode_id"]) < int(current["episode_id"]):
            per_object[object_id] = admission
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for admission in per_object.values():
        buckets.setdefault(str(admission["category"]), []).append(admission)
    for values in buckets.values():
        values.sort(key=lambda item: (str(item["object_id"]), int(item["episode_id"])))
    categories = sorted(buckets)
    selected: list[Mapping[str, Any]] = []
    while len(selected) < cohort_size:
        progressed = False
        for category in categories:
            if buckets[category] and len(selected) < cohort_size:
                selected.append(buckets[category].pop(0))
                progressed = True
        _require(progressed, "insufficient fresh admitted physical objects")
    return selected


def build_fresh_cohort_lock(
    admissions: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    *,
    cohort_size: int,
    method_commit: str,
    method_config_sha256: str,
    parity_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock a deterministic category-round-robin cohort from source evidence."""

    _require(cohort_size >= 2, "cohort size is too small")
    _require(bool(_HEX40.fullmatch(method_commit)), "method commit is malformed")
    _require(
        bool(_HEX64.fullmatch(method_config_sha256)),
        "method config digest is malformed",
    )
    _require(
        bool(exclusions), "at least one independent exclusion manifest is required"
    )
    parity = audit_parity_contract(parity_contract)
    parity_contract_sha256 = str(parity_contract["contract_sha256"])
    parity_ready = bool(parity["parity_ready"])
    for admission in admissions:
        validate_fresh_source_admission(admission)
    for exclusion in exclusions:
        validate_object_exclusion_manifest(exclusion)
    excluded = {
        value for exclusion in exclusions for value in exclusion["object_hashes"]
    }
    case_digests: dict[tuple[str, int], str] = {}
    object_categories: dict[str, str] = {}
    for admission in admissions:
        identity = (str(admission["object_id"]), int(admission["episode_id"]))
        digest = str(admission["admission_sha256"])
        previous = case_digests.setdefault(identity, digest)
        _require(previous == digest, "case has conflicting admission artifacts")
        object_id = identity[0]
        category = str(admission["category"])
        previous_category = object_categories.setdefault(object_id, category)
        _require(
            previous_category == category,
            "physical object has conflicting category labels",
        )
    eligible = [
        admission
        for admission in admissions
        if admission["accepted"]
        and object_exclusion_hash(str(admission["object_id"])) not in excluded
    ]
    selected = _select_round_robin(eligible, cohort_size)
    cases = [
        {
            "case": admission["case"],
            "object_id": admission["object_id"],
            "episode_id": admission["episode_id"],
            "category": admission["category"],
            "admission_sha256": admission["admission_sha256"],
        }
        for admission in selected
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": COHORT_LOCK_KIND,
        "upstream_binding": UPSTREAM_BINDING,
        "cohort_size": cohort_size,
        "cases": cases,
        "selection_rule": (
            "lowest admitted episode per non-excluded physical object, then "
            "deterministic category-sorted round robin and object-ID ordering"
        ),
        "method": {
            "commit": method_commit,
            "config_sha256": method_config_sha256,
        },
        "evaluation": {
            "parity_contract_sha256": parity_contract_sha256,
            "parity_ready": parity_ready,
            "allowed_claim_label": (
                "official_deform360_3d_parity"
                if parity_ready
                else "fresh_object_candidate_conventions_only"
            ),
        },
        "exclusion_manifests": [
            exclusion["exclusion_sha256"]
            for exclusion in sorted(
                exclusions, key=lambda item: str(item["exclusion_sha256"])
            )
        ],
        "information_boundary": {
            "future_object_positions_deserialized": False,
            "future_metrics_read": False,
            "cohort_membership_uses_outcomes": False,
        },
    }
    return _seal(artifact, digest_key="cohort_lock_sha256")


def validate_fresh_cohort_lock(artifact: Mapping[str, Any]) -> None:
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "wrong schema")
    _require(artifact.get("artifact_kind") == COHORT_LOCK_KIND, "wrong artifact kind")
    _require(
        artifact.get("cohort_lock_sha256")
        == _canonical_sha256(artifact, digest_key="cohort_lock_sha256"),
        "cohort lock checksum changed",
    )
    _require(
        artifact.get("upstream_binding") == UPSTREAM_BINDING,
        "cohort upstream binding changed",
    )
    cohort_size = artifact.get("cohort_size")
    _require(
        isinstance(cohort_size, int)
        and not isinstance(cohort_size, bool)
        and cohort_size >= 2,
        "cohort size is malformed",
    )
    cases = artifact.get("cases")
    _require(
        isinstance(cases, list)
        and len(cases) == cohort_size
        and all(isinstance(case, Mapping) for case in cases),
        "cohort cases are malformed",
    )
    _require(
        len({case.get("object_id") for case in cases}) == len(cases),
        "cohort does not contain unique physical objects",
    )
    for case in cases:
        object_id = case.get("object_id")
        episode_id = case.get("episode_id")
        _require(
            isinstance(object_id, str)
            and bool(_OBJECT_ID.fullmatch(object_id))
            and isinstance(episode_id, int)
            and not isinstance(episode_id, bool)
            and episode_id >= 0
            and case.get("case") == f"{object_id}-ep{episode_id:04d}"
            and isinstance(case.get("category"), str)
            and bool(_CATEGORY.fullmatch(case["category"]))
            and _valid_digest(case.get("admission_sha256")),
            "cohort case binding is malformed",
        )
    method = artifact.get("method")
    _require(
        isinstance(method, Mapping)
        and isinstance(method.get("commit"), str)
        and bool(_HEX40.fullmatch(method["commit"]))
        and _valid_digest(method.get("config_sha256")),
        "cohort method binding is malformed",
    )
    exclusions = artifact.get("exclusion_manifests")
    _require(
        isinstance(exclusions, list)
        and bool(exclusions)
        and exclusions == sorted(set(exclusions))
        and all(_valid_digest(value) for value in exclusions),
        "cohort exclusion bindings are malformed",
    )
    evaluation = artifact.get("evaluation")
    _require(isinstance(evaluation, Mapping), "cohort evaluation binding is missing")
    parity_ready = evaluation.get("parity_ready")
    _require(isinstance(parity_ready, bool), "cohort parity decision is malformed")
    _require(
        _valid_digest(evaluation.get("parity_contract_sha256")),
        "cohort parity contract digest is malformed",
    )
    _require(
        evaluation.get("allowed_claim_label")
        == (
            "official_deform360_3d_parity"
            if parity_ready
            else "fresh_object_candidate_conventions_only"
        ),
        "cohort claim label is inconsistent",
    )
    boundary = artifact.get("information_boundary", {})
    _require(
        boundary.get("cohort_membership_uses_outcomes") is False,
        "cohort membership used outcomes",
    )
    _require(
        boundary.get("future_object_positions_deserialized") is False
        and boundary.get("future_metrics_read") is False,
        "cohort lock crossed the future-outcome boundary",
    )


def write_fresh_source_artifact(
    artifact: Mapping[str, Any], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ADMISSION_KIND",
    "COHORT_LOCK_KIND",
    "EXCLUSION_KIND",
    "FreshSourceAdmissionConfig",
    "UPSTREAM_BINDING",
    "build_fresh_cohort_lock",
    "build_fresh_source_admission",
    "build_object_exclusion_manifest",
    "object_exclusion_hash",
    "validate_fresh_cohort_lock",
    "validate_fresh_source_admission",
    "validate_object_exclusion_manifest",
    "write_fresh_source_artifact",
]
