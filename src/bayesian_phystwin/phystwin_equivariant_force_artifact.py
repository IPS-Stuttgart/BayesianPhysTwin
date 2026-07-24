"""Typed artifacts for source-trained equivariant PhysTwin force models."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_equivariant_force import (
    EQUIVARIANT_FORCE_CONTRACT,
    EquivariantForceConfig,
    build_equivariant_force_model,
)


EQUIVARIANT_FORCE_ARTIFACT_SCHEMA = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(array.shape, separators=(",", ":")).encode("ascii")
    )
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _json_mapping(values: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(dict(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _paths(prefix: str | Path) -> tuple[Path, Path]:
    base = Path(prefix)
    if base.suffix in {".json", ".npz"}:
        base = base.with_suffix("")
    return base.with_suffix(".json"), base.with_suffix(".npz")


@dataclass(frozen=True)
class EquivariantForceArtifact:
    """A model plus the source-only boundary under which it was selected."""

    config: EquivariantForceConfig
    weights: Mapping[str, np.ndarray]
    source_checksums: Mapping[str, str]
    information_boundary: Mapping[str, Any]
    training_summary: Mapping[str, Any]
    admission_policy: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, EquivariantForceConfig):
            raise TypeError("config must be EquivariantForceConfig")
        arrays: dict[str, np.ndarray] = {}
        for name, values in sorted(dict(self.weights).items()):
            if not name:
                raise ValueError("weight names must be nonempty")
            array = np.asarray(values).copy()
            if array.dtype.kind not in "fc" or not np.all(np.isfinite(array)):
                raise ValueError("model weights must be finite floating arrays")
            array.setflags(write=False)
            arrays[name] = array
        if not arrays:
            raise ValueError("artifact must contain model weights")
        checksums = dict(sorted(dict(self.source_checksums).items()))
        if not checksums or any(
            not name or not _valid_sha256(digest)
            for name, digest in checksums.items()
        ):
            raise ValueError("source_checksums must contain SHA-256 values")
        boundary = _json_mapping(
            self.information_boundary, name="information_boundary"
        )
        required = {
            "target_future_used_for_fit_or_selection": False,
            "exact_zero_force_fallback": True,
            "force_location": "inside_official_warp",
            "force_unit_contract": (
                "warp_simulator_generalized_force_not_newtons"
            ),
        }
        if any(boundary.get(key) != value for key, value in required.items()):
            raise ValueError("artifact violates its causal or fallback boundary")
        object.__setattr__(self, "weights", arrays)
        object.__setattr__(self, "source_checksums", checksums)
        object.__setattr__(self, "information_boundary", boundary)
        object.__setattr__(
            self,
            "training_summary",
            _json_mapping(self.training_summary, name="training_summary"),
        )
        object.__setattr__(
            self,
            "admission_policy",
            _json_mapping(self.admission_policy, name="admission_policy"),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, name="metadata"),
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        config: EquivariantForceConfig,
        source_checksums: Mapping[str, str],
        information_boundary: Mapping[str, Any],
        training_summary: Mapping[str, Any],
        admission_policy: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> "EquivariantForceArtifact":
        weights = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in model.state_dict().items()
        }
        return cls(
            config=config,
            weights=weights,
            source_checksums=source_checksums,
            information_boundary=information_boundary,
            training_summary=training_summary,
            admission_policy=admission_policy,
            metadata={} if metadata is None else metadata,
        )

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EQUIVARIANT_FORCE_ARTIFACT_SCHEMA,
            "contract": EQUIVARIANT_FORCE_CONTRACT,
            "config": self.config.to_dict(),
            "source_checksums": self.source_checksums,
            "information_boundary": self.information_boundary,
            "training_summary": self.training_summary,
            "admission_policy": self.admission_policy,
            "metadata": self.metadata,
            "weight_names": sorted(self.weights),
            "weight_digests": {
                name: _array_digest(values)
                for name, values in sorted(self.weights.items())
            },
        }

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self._scalar_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def instantiate(self, torch: Any):
        model = build_equivariant_force_model(torch, self.config)
        expected = model.state_dict()
        if set(expected) != set(self.weights):
            raise ValueError("artifact weights do not match the model architecture")
        state = {}
        for name, reference in expected.items():
            values = self.weights[name]
            if tuple(values.shape) != tuple(reference.shape):
                raise ValueError(f"weight shape changed for {name}")
            state[name] = torch.as_tensor(
                np.array(values, copy=True), dtype=reference.dtype
            )
        model.load_state_dict(state, strict=True)
        return model


def write_equivariant_force_artifact(
    prefix: str | Path,
    artifact: EquivariantForceArtifact,
) -> dict[str, str]:
    """Write atomic JSON/NPZ files and return their immutable identifiers."""

    manifest_path, weights_path = _paths(prefix)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_weights = weights_path.with_name(weights_path.name + ".tmp")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary_weights.open("wb") as handle:
        np.savez_compressed(handle, **dict(artifact.weights))
    weights_sha256 = _sha256(temporary_weights)
    payload = artifact._scalar_payload()
    payload.update(
        {
            "artifact_id": artifact.artifact_id,
            "weights_file": weights_path.name,
            "weights_file_sha256": weights_sha256,
        }
    )
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_weights.replace(weights_path)
    temporary_manifest.replace(manifest_path)
    return {
        "artifact_id": artifact.artifact_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "weights_path": str(weights_path),
        "weights_sha256": weights_sha256,
    }


def load_equivariant_force_artifact(
    prefix: str | Path,
) -> EquivariantForceArtifact:
    """Load an artifact after verifying both file and array-level identities."""

    manifest_path, default_weights_path = _paths(prefix)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EQUIVARIANT_FORCE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported equivariant-force artifact schema")
    if payload.get("contract") != EQUIVARIANT_FORCE_CONTRACT:
        raise ValueError("equivariant-force artifact contract changed")
    weights_path = manifest_path.parent / payload.get(
        "weights_file", default_weights_path.name
    )
    if _sha256(weights_path) != payload.get("weights_file_sha256"):
        raise ValueError("equivariant-force weight file hash mismatch")
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    artifact = EquivariantForceArtifact(
        config=EquivariantForceConfig(**payload["config"]),
        weights=weights,
        source_checksums=payload["source_checksums"],
        information_boundary=payload["information_boundary"],
        training_summary=payload["training_summary"],
        admission_policy=payload["admission_policy"],
        metadata=payload.get("metadata", {}),
    )
    if artifact.artifact_id != payload.get("artifact_id"):
        raise ValueError("equivariant-force artifact identity mismatch")
    if artifact._scalar_payload()["weight_digests"] != payload.get(
        "weight_digests"
    ):
        raise ValueError("equivariant-force array identity mismatch")
    return artifact
