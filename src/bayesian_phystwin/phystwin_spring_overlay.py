"""Apply an externally predicted spring field to a PhysTwin checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SPRING_OVERLAY_CONTRACT = "phystwin-spring-field-overlay-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("PhysTwin checkpoint must contain a dictionary")
    if "spring_Y" not in payload:
        raise ValueError("PhysTwin checkpoint does not contain spring_Y")
    return payload


def build_spring_overlay_checkpoint(
    source_checkpoint: str | Path,
    spring_y_path: str | Path,
    output_checkpoint: str | Path,
    *,
    summary_path: str | Path | None = None,
    strength: float = 1.0,
) -> dict[str, object]:
    """Apply a log-space spring proposal and bind the operation to hashed inputs."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("spring-field overlays require PyTorch") from exc

    source = Path(source_checkpoint).resolve()
    field_path = Path(spring_y_path).resolve()
    output = Path(output_checkpoint).resolve()
    if output == source:
        raise ValueError("output checkpoint must not overwrite its source")
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("spring proposal strength must lie in [0, 1]")
    checkpoint = _load_checkpoint(torch, source)
    source_tensor = torch.as_tensor(checkpoint["spring_Y"]).detach().cpu()
    source_values = source_tensor.numpy().astype(np.float64, copy=False).reshape(-1)
    candidate = np.load(field_path, allow_pickle=False)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if candidate.shape != source_values.shape:
        raise ValueError(
            "candidate and checkpoint spring fields disagree: "
            f"{candidate.shape} versus {source_values.shape}"
        )
    if not np.isfinite(candidate).all() or np.any(candidate <= 0.0):
        raise ValueError("candidate spring field must be finite and positive")
    if not np.isfinite(source_values).all() or np.any(source_values <= 0.0):
        raise ValueError("source spring field must be finite and positive")

    if strength == 0.0:
        applied = source_values.copy()
    elif strength == 1.0:
        applied = candidate
    else:
        applied = np.exp(
            np.log(source_values) + strength * (np.log(candidate) - np.log(source_values))
        )

    overlaid = dict(checkpoint)
    overlaid["spring_Y"] = torch.as_tensor(
        applied.reshape(tuple(source_tensor.shape)), dtype=source_tensor.dtype
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(overlaid, output)

    log_ratio = np.log(applied) - np.log(source_values)
    summary = {
        "schema_version": 1,
        "contract": SPRING_OVERLAY_CONTRACT,
        "source_checkpoint": {
            "path": str(source),
            "sha256": _sha256(source),
        },
        "candidate_spring_y": {
            "path": str(field_path),
            "sha256": _sha256(field_path),
            "count": int(len(candidate)),
        },
        "output_checkpoint": {
            "path": str(output),
            "sha256": _sha256(output),
        },
        "replacement_scope": "spring_Y only",
        "proposal_strength": strength,
        "interpolation": "log-space geodesic from released field to candidate",
        "identity_field": bool(np.array_equal(applied, source_values)),
        "spring_ratio": {
            "minimum": float(np.min(applied / source_values)),
            "median": float(np.median(applied / source_values)),
            "maximum": float(np.max(applied / source_values)),
            "log_rms": float(np.sqrt(np.mean(np.square(log_ratio)))),
        },
    }
    summary_output = (
        Path(summary_path).resolve()
        if summary_path is not None
        else output.with_suffix(output.suffix + ".overlay.json")
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_digest = _sha256(summary_output)
    summary_output.with_suffix(summary_output.suffix + ".sha256").write_text(
        f"{summary_digest}  {summary_output.name}\n", encoding="ascii"
    )
    return {
        **summary,
        "summary_artifact": {
            "path": str(summary_output),
            "sha256": summary_digest,
        },
    }
