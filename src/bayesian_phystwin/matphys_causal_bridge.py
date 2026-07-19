"""Causal input and provenance helpers for the public MatPhys release."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MATPHYS_CAUSAL_AUDIT_SCHEMA_VERSION = 2
MATPHYS_SOURCE_SUPERVISED_AUDIT_SCHEMA_VERSION = 1
MATPHYS_SOURCE_SUPERVISED_AUDIT_CONTRACT = (
    "matphys-source-supervised-meta-audit-v1"
)
_EXTERNAL_BACKBONE_SHARED_FIELDS = (
    "name",
    "source_repository",
    "source_commit",
    "future_observations_used",
    "coordinate_frame",
    "vertex_contract",
    "proxy_contract",
    "claim_boundary",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identities(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    identities = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        identities.append({"path": str(path), "sha256": sha256_file(path)})
    return identities


def _validate_artifact_identities(
    identities: object,
    *,
    label: str,
) -> None:
    if identities is None:
        return
    if not isinstance(identities, list):
        raise ValueError(f"{label} identities must be a list")
    for identity in identities:
        if not isinstance(identity, dict):
            raise ValueError(f"{label} identity must be an object")
        if sha256_file(identity.get("path", "")) != identity.get("sha256"):
            raise ValueError(f"{label} bytes changed")


def causal_uniform_frame_indices(
    frame_count: int,
    evidence_end_frame_exclusive: int,
    sample_count: int,
) -> np.ndarray:
    """Select uniform video inputs without touching the held-out interval."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 1 <= evidence_end_frame_exclusive <= frame_count:
        raise ValueError("evidence end must lie inside the available video")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.linspace(
        0,
        evidence_end_frame_exclusive - 1,
        sample_count,
        dtype=int,
    )


def numeric_frame_paths(
    directory: str | Path,
    *,
    suffixes: Sequence[str] = (".png", ".jpg", ".jpeg"),
) -> dict[int, Path]:
    """Index unpadded frame files by their numeric stem.

    Lexicographic sorting places ``10.png`` before ``2.png`` and can therefore
    cross a causal frame boundary even when list positions appear valid. This
    helper makes the frame identifier, rather than directory order, the unit
    audited by the causal MatPhys path.
    """

    root = Path(directory)
    allowed = {suffix.lower() for suffix in suffixes}
    result: dict[int, Path] = {}
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            frame_id = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"nonnumeric frame filename: {path}") from exc
        if frame_id < 0:
            raise ValueError(f"negative frame identifier: {path}")
        if frame_id in result:
            raise ValueError(f"duplicate numeric frame identifier {frame_id}")
        result[frame_id] = path.resolve()
    if not result:
        raise ValueError(f"no numeric image frames found under {root}")
    return dict(sorted(result.items()))


def causal_uniform_frame_ids(
    available_frame_ids: Sequence[int],
    evidence_end_frame_exclusive: int,
    sample_count: int,
) -> np.ndarray:
    """Select causal frames from explicit numeric identifiers."""

    if evidence_end_frame_exclusive < 1:
        raise ValueError("evidence end must be positive")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    eligible = np.asarray(
        sorted(
            {
                int(frame_id)
                for frame_id in available_frame_ids
                if 0 <= int(frame_id) < evidence_end_frame_exclusive
            }
        ),
        dtype=int,
    )
    if len(eligible) == 0:
        raise ValueError("no available frame lies before the evidence boundary")
    positions = np.linspace(0, len(eligible) - 1, sample_count, dtype=int)
    selected = eligible[positions]
    if np.any(selected >= evidence_end_frame_exclusive):
        raise AssertionError("causal frame selection crossed the evidence boundary")
    return selected


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepare_global_material_proxy(
    data_root: str | Path,
    case_names: Sequence[str],
    source_mapping_path: str | Path,
    output_root: str | Path,
    *,
    num_materials: int = 10,
    semantic_dimension: int = 1024,
) -> dict[str, object]:
    """Create deterministic one-part inputs for missing MatPhys release assets.

    MatPhys does not publish its per-case ``node_sem.npz`` and
    ``train_ready.pt`` artifacts. The released simplified decoder does not use
    ``z_sem`` directly, so zeros are sufficient for that unused field. The
    material proxy preserves the repository's declared object-class label as a
    one-hot, single-part distribution. This is an explicit global-material
    ablation, not a reproduction of MatPhys's part decomposition.
    """

    if num_materials < 1 or semantic_dimension < 1:
        raise ValueError("proxy dimensions must be positive")
    root = Path(data_root).resolve()
    mapping_path = Path(source_mapping_path).resolve()
    destination = Path(output_root).resolve()
    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    case_to_label = raw_mapping.get("case_to_material", raw_mapping)
    class_to_id = raw_mapping.get("class_to_id", {})
    if not class_to_id:
        labels = sorted(set(case_to_label.values()))
        class_to_id = {label: index for index, label in enumerate(labels)}
    selected_mapping: dict[str, str | int] = {}
    case_records: list[dict[str, object]] = []

    import torch

    for case in case_names:
        if case not in case_to_label:
            raise ValueError(f"MatPhys material mapping omits {case}")
        label = case_to_label[case]
        material_id = int(class_to_id[label]) if isinstance(label, str) else int(label)
        if not 0 <= material_id < num_materials:
            raise ValueError(f"{case}: material id exceeds decoder dimensions")
        final_data_path = root / case / "final_data.pkl"
        data = _load_pickle(final_data_path)
        structure_points = np.concatenate(
            (
                np.asarray(data["object_points"])[0],
                np.asarray(data["surface_points"]),
                np.asarray(data["interior_points"]),
            ),
            axis=0,
        ).astype(np.float32)
        node_sem_path = destination / "semantic_cache" / f"{case}_node_sem.npz"
        node_sem_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            node_sem_path,
            node_sem=np.zeros(
                (len(structure_points), semantic_dimension), dtype=np.float32
            ),
        )
        material_distribution = torch.zeros(
            (1, num_materials), dtype=torch.float32
        )
        material_distribution[0, material_id] = 1.0
        train_ready_path = destination / "results" / case / "train" / "train_ready.pt"
        train_ready_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "part_assignments": torch.zeros(
                    len(structure_points), dtype=torch.long
                ),
                "material_distributions": material_distribution,
                "part_features": torch.zeros(
                    (1, semantic_dimension), dtype=torch.float32
                ),
                "xyz": torch.from_numpy(structure_points),
                "proxy_contract": "global-onehot-single-part-v1",
            },
            train_ready_path,
        )
        selected_mapping[case] = label
        case_records.append(
            {
                "name": case,
                "material_label": label,
                "material_id": material_id,
                "structure_point_count": len(structure_points),
                "node_sem": {
                    "path": str(node_sem_path),
                    "sha256": sha256_file(node_sem_path),
                },
                "train_ready": {
                    "path": str(train_ready_path),
                    "sha256": sha256_file(train_ready_path),
                },
            }
        )

    proxy_mapping_path = destination / "case_to_material.json"
    proxy_mapping_path.write_text(
        json.dumps(
            {
                "class_to_id": class_to_id,
                "case_to_material": selected_mapping,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "contract": "global-onehot-single-part-v1",
        "claim_boundary": (
            "Proxy for unpublished MatPhys preprocessing artifacts; not a "
            "part-level MatPhys reproduction."
        ),
        "source_mapping": {
            "path": str(mapping_path),
            "sha256": sha256_file(mapping_path),
        },
        "mapping": {
            "path": str(proxy_mapping_path),
            "sha256": sha256_file(proxy_mapping_path),
        },
        "semantic_cache_dir": str(destination / "semantic_cache"),
        "results_dir": str(destination / "results"),
        "cases": case_records,
    }
    summary_path = destination / "proxy_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def write_causal_training_audit(
    checkpoint_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    source_repository: str,
    source_commit: str,
    data_root: str | Path,
    accessed_frame_indices: Mapping[str, Sequence[int]],
    accessed_frame_paths: Mapping[str, Mapping[int, str | Path]],
    objective_end_frames_exclusive: Mapping[str, int],
    evidence_end_frames_exclusive: Mapping[str, int],
    split_by_case: Mapping[str, Mapping[str, object]],
    proxy_summary_path: str | Path,
    parameterization: Mapping[str, object] | None = None,
    runtime_access_log_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Bind checkpoint bytes to an observed-frame access log."""

    if not source_repository or not source_commit:
        raise ValueError("source repository and commit are required")
    cases: list[dict[str, object]] = []
    for case, raw_indices in sorted(accessed_frame_indices.items()):
        if case not in split_by_case:
            raise ValueError(f"missing split for accessed case {case}")
        train_end = int(split_by_case[case]["train"][1])
        if case not in evidence_end_frames_exclusive:
            raise ValueError(f"missing evidence boundary for accessed case {case}")
        evidence_end = int(evidence_end_frames_exclusive[case])
        if not 1 <= evidence_end <= train_end:
            raise ValueError(f"{case}: evidence boundary crosses the released prefix")
        if case not in objective_end_frames_exclusive:
            raise ValueError(f"missing objective boundary for accessed case {case}")
        objective_end = int(objective_end_frames_exclusive[case])
        if not 1 <= objective_end <= evidence_end:
            raise ValueError(f"{case}: objective accessed a future frame")
        indices = sorted(set(int(index) for index in raw_indices))
        if not indices:
            raise ValueError(f"no video frames were recorded for {case}")
        if indices[0] < 0 or indices[-1] >= evidence_end:
            raise ValueError(f"{case}: checkpoint training accessed a future frame")
        case_paths = accessed_frame_paths.get(case)
        if not isinstance(case_paths, Mapping):
            raise ValueError(f"{case}: exact accessed frame paths were not recorded")
        frame_files: list[dict[str, object]] = []
        for frame_id in indices:
            if frame_id not in case_paths:
                raise ValueError(f"{case}: frame {frame_id} has no bound source file")
            frame_path = Path(case_paths[frame_id]).resolve()
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            try:
                path_frame_id = int(frame_path.stem)
            except ValueError as exc:
                raise ValueError(f"{case}: frame source has a nonnumeric stem") from exc
            if path_frame_id != frame_id:
                raise ValueError(f"{case}: audited frame id disagrees with its filename")
            frame_files.append(
                {
                    "frame_id": frame_id,
                    "path": str(frame_path),
                    "sha256": sha256_file(frame_path),
                }
            )
        cases.append(
            {
                "name": case,
                "train_end_frame_exclusive": train_end,
                "evidence_end_frame_exclusive": evidence_end,
                "accessed_frame_indices": indices,
                "maximum_accessed_frame": indices[-1],
                "accessed_frame_files": frame_files,
                "objective_frame_interval": [1, objective_end],
                "maximum_objective_frame": objective_end - 1,
                "validation_frame_interval": [evidence_end, train_end],
            }
        )
    checkpoints = []
    for raw_path in checkpoint_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append({"path": str(path), "sha256": sha256_file(path)})
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    proxy_path = Path(proxy_summary_path).resolve()
    audit = {
        "schema_version": MATPHYS_CAUSAL_AUDIT_SCHEMA_VERSION,
        "future_observations_used": False,
        "frame_id_contract": "numeric-filename-stem-v2",
        "video_sampling": "uniform-numeric-ids-before-evidence-boundary-v2",
        "optimization_and_checkpoint_selection": "fit-prefix-only-v2",
        "checkpoint_policy": "fixed-terminal-epoch-v1",
        "fit_all_frames": False,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "data_root": str(Path(data_root).resolve()),
        "proxy": {"path": str(proxy_path), "sha256": sha256_file(proxy_path)},
        "checkpoints": checkpoints,
        "cases": cases,
        "runtime_access_logs": _artifact_identities(runtime_access_log_paths),
    }
    if parameterization is not None:
        audit["parameterization"] = dict(parameterization)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit["audit_path"] = str(destination.resolve())
    audit["audit_sha256"] = sha256_file(destination)
    return audit


def validate_causal_training_audit(
    audit_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Require a future-blind audit that binds the exact exported checkpoint."""

    source = Path(audit_path).resolve()
    audit = json.loads(source.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MATPHYS_CAUSAL_AUDIT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported or legacy MatPhys causal-audit schema; schema 1 used "
            "unsafe lexicographic frame positions"
        )
    if audit.get("future_observations_used") is not False:
        raise ValueError("MatPhys checkpoint audit does not forbid future observations")
    if audit.get("fit_all_frames") is not False:
        raise ValueError("MatPhys checkpoint audit permits fit_all_frames")
    if audit.get("frame_id_contract") != "numeric-filename-stem-v2":
        raise ValueError("MatPhys checkpoint audit does not bind numeric frame ids")
    if (
        audit.get("video_sampling")
        != "uniform-numeric-ids-before-evidence-boundary-v2"
    ):
        raise ValueError("MatPhys checkpoint audit uses an unsafe frame order")
    if audit.get("optimization_and_checkpoint_selection") != "fit-prefix-only-v2":
        raise ValueError("MatPhys checkpoint audit does not bind objective access")
    if audit.get("checkpoint_policy") != "fixed-terminal-epoch-v1":
        raise ValueError("MatPhys checkpoint audit does not use the terminal epoch")
    _validate_artifact_identities(
        audit.get("runtime_access_logs"), label="MatPhys runtime access log"
    )
    checkpoint = Path(checkpoint_path).resolve()
    identity = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    if identity not in audit.get("checkpoints", []):
        raise ValueError("checkpoint bytes are not bound by the causal training audit")
    proxy_identity = audit.get("proxy")
    if not isinstance(proxy_identity, dict):
        raise ValueError("MatPhys checkpoint audit omits its input proxy")
    proxy_path = Path(proxy_identity.get("path", ""))
    if sha256_file(proxy_path) != proxy_identity.get("sha256"):
        raise ValueError("MatPhys proxy summary bytes changed")
    proxy_summary = json.loads(proxy_path.read_text(encoding="utf-8"))
    for proxy_case in proxy_summary.get("cases", []):
        for key in ("node_sem", "train_ready"):
            proxy_source = proxy_case.get(key)
            if not isinstance(proxy_source, dict):
                raise ValueError(f"MatPhys proxy omits {key}")
            if sha256_file(proxy_source["path"]) != proxy_source.get("sha256"):
                raise ValueError(f"MatPhys proxy {key} bytes changed")
    for case in audit.get("cases", []):
        train_end = int(case["train_end_frame_exclusive"])
        evidence_end = int(case["evidence_end_frame_exclusive"])
        if not 1 <= evidence_end <= train_end:
            raise ValueError(f"{case.get('name')}: invalid evidence boundary")
        if case.get("validation_frame_interval") != [evidence_end, train_end]:
            raise ValueError(
                f"{case.get('name')}: validation interval disagrees with evidence"
            )
        indices = [int(index) for index in case["accessed_frame_indices"]]
        if not indices or min(indices) < 0 or max(indices) >= evidence_end:
            raise ValueError(f"{case.get('name')}: audit contains future video access")
        frame_files = case.get("accessed_frame_files")
        if not isinstance(frame_files, list) or len(frame_files) != len(indices):
            raise ValueError(f"{case.get('name')}: exact frame sources are unbound")
        bound_ids: list[int] = []
        for record in frame_files:
            frame_id = int(record["frame_id"])
            frame_path = Path(record["path"])
            if int(frame_path.stem) != frame_id:
                raise ValueError(f"{case.get('name')}: numeric frame binding changed")
            if sha256_file(frame_path) != record["sha256"]:
                raise ValueError(f"{case.get('name')}: accessed frame bytes changed")
            bound_ids.append(frame_id)
        if sorted(bound_ids) != sorted(indices):
            raise ValueError(f"{case.get('name')}: accessed frame ids are inconsistent")
        objective_start, objective_end = (
            int(value) for value in case["objective_frame_interval"]
        )
        if objective_start != 1 or not objective_start <= objective_end <= evidence_end:
            raise ValueError(
                f"{case.get('name')}: audit contains future objective access"
            )
        if int(case["maximum_objective_frame"]) != objective_end - 1:
            raise ValueError(f"{case.get('name')}: objective audit is inconsistent")
    parameterization = audit.get("parameterization")
    if parameterization is not None:
        if not isinstance(parameterization, dict):
            raise ValueError("MatPhys parameterization audit must be an object")
        from .matphys_teacher_residual import (
            TEACHER_PARAMETERIZATION,
            validate_matphys_teacher_manifest,
        )

        if parameterization.get("name") != TEACHER_PARAMETERIZATION:
            raise ValueError("unsupported MatPhys parameterization audit")
        scale = float(parameterization.get("residual_log_scale", -1.0))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("invalid teacher residual scale")
        teacher = parameterization.get("teacher")
        if not isinstance(teacher, dict):
            raise ValueError("teacher parameterization omits its source manifest")
        validate_matphys_teacher_manifest(teacher)
    return {
        **audit,
        "audit_path": str(source),
        "audit_sha256": sha256_file(source),
    }


def _case_split_identity(
    data_root: Path,
    case: str,
    split: Mapping[str, object],
) -> dict[str, object]:
    split_path = data_root / case / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    train = [int(value) for value in split["train"]]
    test = [int(value) for value in split["test"]]
    frame_len = int(split.get("frame_len", test[1]))
    if train[0] < 0 or train[0] >= train[1] or train[1] > frame_len:
        raise ValueError(f"{case}: malformed training interval")
    if test[0] < train[1] or test[0] >= test[1] or test[1] > frame_len:
        raise ValueError(f"{case}: malformed test interval")
    return {
        "path": str(split_path.resolve()),
        "sha256": sha256_file(split_path),
        "train": train,
        "test": test,
        "frame_len": frame_len,
    }


def write_source_supervised_training_audit(
    checkpoint_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    source_repository: str,
    source_commit: str,
    data_root: str | Path,
    source_cases: Sequence[str],
    target_cases: Sequence[str],
    accessed_frame_indices: Mapping[str, Sequence[int]],
    accessed_frame_paths: Mapping[str, Mapping[int, str | Path]],
    objective_end_frames_exclusive: Mapping[str, int],
    evidence_end_frames_exclusive: Mapping[str, int],
    target_evidence_end_frames_exclusive: Mapping[str, int],
    split_by_case: Mapping[str, Mapping[str, object]],
    proxy_summary_path: str | Path,
    split_registration_path: str | Path,
    parameterization: Mapping[str, object] | None = None,
    runtime_access_log_paths: Sequence[str | Path] = (),
    implementation_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Bind a causal-input/full-source-outcome checkpoint to a disjoint split.

    Source outcomes may extend beyond the source input prefix. Target videos,
    outcomes, and metrics are forbidden during training and checkpoint
    selection. This is a separate contract from per-case causal adaptation.
    """

    sources = tuple(str(case) for case in source_cases)
    targets = tuple(str(case) for case in target_cases)
    if not sources or not targets:
        raise ValueError("source and target case lists must both be nonempty")
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise ValueError("source and target case lists must be unique")
    if set(sources) & set(targets):
        raise ValueError("source and target cases must be disjoint")
    if set(accessed_frame_indices) != set(sources):
        raise ValueError("video access log must contain exactly the source cases")
    if set(objective_end_frames_exclusive) != set(sources):
        raise ValueError("objective log must contain exactly the source cases")
    if set(evidence_end_frames_exclusive) != set(sources):
        raise ValueError("evidence boundaries must contain exactly the source cases")
    if set(target_evidence_end_frames_exclusive) != set(targets):
        raise ValueError("target evidence boundaries must match the target cases")
    if set(split_by_case) != set(sources) | set(targets):
        raise ValueError("split metadata must contain every registered case exactly")
    if not source_repository or not source_commit:
        raise ValueError("source repository and commit are required")

    root = Path(data_root).resolve()
    split_registration = Path(split_registration_path).resolve()
    registration = json.loads(split_registration.read_text(encoding="utf-8"))
    if set(registration.get("source_cases", [])) != set(sources):
        raise ValueError("registered source cases differ from the training request")
    if set(registration.get("target_cases", [])) != set(targets):
        raise ValueError("registered target cases differ from the training request")

    proxy_path = Path(proxy_summary_path).resolve()
    proxy_summary = json.loads(proxy_path.read_text(encoding="utf-8"))
    proxy_cases = {str(record.get("name")) for record in proxy_summary.get("cases", [])}
    if proxy_cases != set(sources):
        raise ValueError("training proxy must contain exactly the source cases")

    source_records: list[dict[str, object]] = []
    for case in sources:
        split_identity = _case_split_identity(root, case, split_by_case[case])
        train_end = int(split_identity["train"][1])
        frame_len = int(split_identity["frame_len"])
        evidence_end = int(evidence_end_frames_exclusive[case])
        objective_end = int(objective_end_frames_exclusive[case])
        if not 1 <= evidence_end <= train_end:
            raise ValueError(f"{case}: source video evidence crosses its prefix")
        if not evidence_end <= objective_end <= frame_len:
            raise ValueError(f"{case}: source objective boundary is invalid")
        indices = sorted(set(int(index) for index in accessed_frame_indices[case]))
        if not indices or indices[0] < 0 or indices[-1] >= evidence_end:
            raise ValueError(f"{case}: source video access crossed its evidence prefix")
        case_paths = accessed_frame_paths.get(case)
        if not isinstance(case_paths, Mapping):
            raise ValueError(f"{case}: exact source frame paths were not recorded")
        frame_files = []
        for frame_id in indices:
            if frame_id not in case_paths:
                raise ValueError(f"{case}: frame {frame_id} has no bound source file")
            frame_path = Path(case_paths[frame_id]).resolve()
            if int(frame_path.stem) != frame_id:
                raise ValueError(f"{case}: source frame id disagrees with its filename")
            frame_files.append(
                {
                    "frame_id": frame_id,
                    "path": str(frame_path),
                    "sha256": sha256_file(frame_path),
                }
            )
        source_records.append(
            {
                "name": case,
                "split": split_identity,
                "evidence_end_frame_exclusive": evidence_end,
                "accessed_frame_indices": indices,
                "maximum_accessed_frame": indices[-1],
                "accessed_frame_files": frame_files,
                "objective_frame_interval": [1, objective_end],
                "maximum_objective_frame": objective_end - 1,
                "source_future_outcomes_used": objective_end > evidence_end,
            }
        )

    target_records = []
    for case in targets:
        split_identity = _case_split_identity(root, case, split_by_case[case])
        evidence_end = int(target_evidence_end_frames_exclusive[case])
        if not 1 <= evidence_end <= int(split_identity["train"][1]):
            raise ValueError(f"{case}: target evidence boundary crosses its prefix")
        target_records.append(
            {
                "name": case,
                "split": split_identity,
                "evidence_end_frame_exclusive": evidence_end,
                "video_accessed_during_training": False,
                "outcome_accessed_during_training": False,
                "metric_accessed_during_training": False,
            }
        )

    checkpoints = []
    for raw_path in checkpoint_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append({"path": str(path), "sha256": sha256_file(path)})
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    audit = {
        "schema_version": MATPHYS_SOURCE_SUPERVISED_AUDIT_SCHEMA_VERSION,
        "contract": MATPHYS_SOURCE_SUPERVISED_AUDIT_CONTRACT,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "data_root": str(root),
        "frame_id_contract": "numeric-filename-stem-v2",
        "video_sampling": "uniform-numeric-ids-before-evidence-boundary-v2",
        "training_semantics": "causal-input-full-registered-source-outcome-v1",
        "checkpoint_policy": "fixed-terminal-epoch-v1",
        "source_future_video_used": False,
        "source_future_outcomes_used": any(
            bool(record["source_future_outcomes_used"])
            for record in source_records
        ),
        "target_future_observations_used": False,
        "target_metrics_used_for_checkpoint_selection": False,
        "split_registration": {
            "path": str(split_registration),
            "sha256": sha256_file(split_registration),
        },
        "proxy": {"path": str(proxy_path), "sha256": sha256_file(proxy_path)},
        "checkpoints": checkpoints,
        "source_cases": source_records,
        "target_cases": target_records,
        "runtime_access_logs": _artifact_identities(runtime_access_log_paths),
        "implementation_files": _artifact_identities(implementation_paths),
    }
    if parameterization is not None:
        audit["parameterization"] = dict(parameterization)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **audit,
        "audit_path": str(destination),
        "audit_sha256": sha256_file(destination),
    }


def validate_source_supervised_training_audit(
    audit_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Validate source-only supervision and exact target non-access."""

    source = Path(audit_path).resolve()
    audit = json.loads(source.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version")
        != MATPHYS_SOURCE_SUPERVISED_AUDIT_SCHEMA_VERSION
        or audit.get("contract") != MATPHYS_SOURCE_SUPERVISED_AUDIT_CONTRACT
    ):
        raise ValueError("unsupported MatPhys source-supervised audit")
    expected_fields = {
        "source_future_video_used": False,
        "target_future_observations_used": False,
        "target_metrics_used_for_checkpoint_selection": False,
        "frame_id_contract": "numeric-filename-stem-v2",
        "video_sampling": "uniform-numeric-ids-before-evidence-boundary-v2",
        "training_semantics": "causal-input-full-registered-source-outcome-v1",
        "checkpoint_policy": "fixed-terminal-epoch-v1",
    }
    for field, expected in expected_fields.items():
        if audit.get(field) != expected:
            raise ValueError(f"source-supervised audit violates {field}")
    _validate_artifact_identities(
        audit.get("runtime_access_logs"),
        label="MatPhys source-supervised runtime access log",
    )
    _validate_artifact_identities(
        audit.get("implementation_files"),
        label="MatPhys source-supervised implementation file",
    )
    checkpoint = Path(checkpoint_path).resolve()
    identity = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    if identity not in audit.get("checkpoints", []):
        raise ValueError("checkpoint bytes are not bound by the source audit")
    for identity_field in ("split_registration", "proxy"):
        artifact = audit.get(identity_field)
        if not isinstance(artifact, dict):
            raise ValueError(f"source-supervised audit omits {identity_field}")
        if sha256_file(artifact["path"]) != artifact.get("sha256"):
            raise ValueError(f"source-supervised {identity_field} bytes changed")

    registration = json.loads(
        Path(audit["split_registration"]["path"]).read_text(encoding="utf-8")
    )
    sources = audit.get("source_cases", [])
    targets = audit.get("target_cases", [])
    source_names = {str(record.get("name")) for record in sources}
    target_names = {str(record.get("name")) for record in targets}
    if not source_names or not target_names or source_names & target_names:
        raise ValueError("source-supervised audit has an invalid case partition")
    if source_names != set(registration.get("source_cases", [])):
        raise ValueError("audited sources differ from the registered split")
    if target_names != set(registration.get("target_cases", [])):
        raise ValueError("audited targets differ from the registered split")

    proxy_summary = json.loads(Path(audit["proxy"]["path"]).read_text(encoding="utf-8"))
    proxy_names = {
        str(record.get("name")) for record in proxy_summary.get("cases", [])
    }
    if proxy_names != source_names:
        raise ValueError("source-supervised proxy includes non-source cases")
    for proxy_case in proxy_summary.get("cases", []):
        for key in ("node_sem", "train_ready"):
            artifact = proxy_case.get(key)
            if not isinstance(artifact, dict):
                raise ValueError(f"MatPhys source proxy omits {key}")
            if sha256_file(artifact["path"]) != artifact.get("sha256"):
                raise ValueError(f"MatPhys source proxy {key} bytes changed")

    for record in sources:
        split = record["split"]
        if sha256_file(split["path"]) != split["sha256"]:
            raise ValueError(f"{record.get('name')}: source split bytes changed")
        evidence_end = int(record["evidence_end_frame_exclusive"])
        objective_start, objective_end = (
            int(value) for value in record["objective_frame_interval"]
        )
        if not 1 <= evidence_end <= int(split["train"][1]):
            raise ValueError(f"{record.get('name')}: invalid source evidence boundary")
        if objective_start != 1 or not evidence_end <= objective_end <= int(
            split["frame_len"]
        ):
            raise ValueError(f"{record.get('name')}: invalid source objective boundary")
        indices = [int(index) for index in record["accessed_frame_indices"]]
        if not indices or min(indices) < 0 or max(indices) >= evidence_end:
            raise ValueError(f"{record.get('name')}: source video crossed its prefix")
        frame_files = record.get("accessed_frame_files")
        if not isinstance(frame_files, list) or len(frame_files) != len(indices):
            raise ValueError(f"{record.get('name')}: source frame files are unbound")
        bound_ids = []
        for frame in frame_files:
            frame_id = int(frame["frame_id"])
            path = Path(frame["path"])
            if int(path.stem) != frame_id or sha256_file(path) != frame["sha256"]:
                raise ValueError(f"{record.get('name')}: source frame bytes changed")
            bound_ids.append(frame_id)
        if sorted(bound_ids) != sorted(indices):
            raise ValueError(f"{record.get('name')}: source frame ids are inconsistent")
        if int(record["maximum_objective_frame"]) != objective_end - 1:
            raise ValueError(f"{record.get('name')}: source objective audit changed")

    for record in targets:
        split = record["split"]
        if sha256_file(split["path"]) != split["sha256"]:
            raise ValueError(f"{record.get('name')}: target split bytes changed")
        if any(
            record.get(field) is not False
            for field in (
                "video_accessed_during_training",
                "outcome_accessed_during_training",
                "metric_accessed_during_training",
            )
        ):
            raise ValueError(f"{record.get('name')}: target data were accessed")
        evidence_end = int(record["evidence_end_frame_exclusive"])
        if not 1 <= evidence_end <= int(split["train"][1]):
            raise ValueError(f"{record.get('name')}: invalid target evidence boundary")
    parameterization = audit.get("parameterization")
    if parameterization is not None:
        if not isinstance(parameterization, dict):
            raise ValueError("source-supervised parameterization must be an object")
        from .matphys_teacher_residual import (
            TEACHER_PARAMETERIZATION,
            validate_matphys_teacher_manifest,
        )

        if parameterization.get("name") != TEACHER_PARAMETERIZATION:
            raise ValueError("unsupported source-supervised parameterization")
        scale = float(parameterization.get("residual_log_scale", -1.0))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("invalid source-supervised teacher residual scale")
        validate_matphys_teacher_manifest(parameterization.get("teacher", {}))
    return {
        **audit,
        "audit_path": str(source),
        "audit_sha256": sha256_file(source),
    }


def merge_matphys_external_manifests(
    manifest_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    case_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """Merge per-case MatPhys exports without erasing their provenance."""

    if not manifest_paths:
        raise ValueError("at least one external-backbone manifest is required")
    components: list[dict[str, object]] = []
    cases_by_name: dict[str, dict[str, object]] = {}
    shared_backbone: dict[str, object] | None = None
    for raw_path in manifest_paths:
        source = Path(raw_path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"{source}: unsupported external manifest schema")
        backbone = payload.get("backbone")
        raw_cases = payload.get("cases")
        if not isinstance(backbone, dict) or not isinstance(raw_cases, list):
            raise ValueError(f"{source}: malformed external manifest")
        identity = {field: backbone.get(field) for field in _EXTERNAL_BACKBONE_SHARED_FIELDS}
        if shared_backbone is None:
            shared_backbone = identity
        elif identity != shared_backbone:
            raise ValueError("component manifests describe incompatible backbones")
        component_cases: list[str] = []
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError(f"{source}: malformed case entry")
            name = str(raw_case.get("name", ""))
            if not name or name in cases_by_name:
                raise ValueError("component case names must be nonempty and unique")
            trajectory = Path(str(raw_case.get("trajectory", "")))
            if not trajectory.is_absolute():
                trajectory = source.parent / trajectory
            normalized = {**raw_case, "trajectory": str(trajectory.resolve())}
            cases_by_name[name] = normalized
            component_cases.append(name)
        components.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "cases": component_cases,
                "checkpoint": backbone.get("checkpoint"),
                "causal_training_audit": backbone.get("causal_training_audit"),
            }
        )

    requested_order = (
        tuple(str(case) for case in case_order)
        if case_order is not None
        else tuple(cases_by_name)
    )
    if len(requested_order) != len(set(requested_order)):
        raise ValueError("case order must not contain duplicates")
    if set(requested_order) != set(cases_by_name):
        raise ValueError("case order must contain every component case exactly once")
    assert shared_backbone is not None
    merged = {
        "schema_version": 1,
        "backbone": {
            **shared_backbone,
            "training_scope": "independent-per-case-fixed-terminal-v1",
            "component_manifests": components,
        },
        "cases": [cases_by_name[name] for name in requested_order],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **merged,
        "manifest_path": str(destination),
        "manifest_sha256": sha256_file(destination),
    }
