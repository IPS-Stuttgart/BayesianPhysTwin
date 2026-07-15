"""Object-persistent rest metrics for reusable Deform360 physical twins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_replication_graph import Deform360SparseGraph


REUSABLE_TWIN_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class Deform360ReusableTwin:
    """A shared graph topology and intrinsic rest metric for one object."""

    object_id: str
    stratum: str
    spring_edges: np.ndarray
    spring_families: np.ndarray
    object_rest_lengths_m: np.ndarray
    source_episode_ids: tuple[str, ...]
    source_geometry_sha256: tuple[str, ...]
    fit_policy: dict[str, Any]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        edges = np.asarray(self.spring_edges, dtype=np.int32)
        families = np.asarray(self.spring_families, dtype=np.int8)
        rest = np.asarray(self.object_rest_lengths_m, dtype=np.float64)
        _require(bool(self.object_id), "reusable twin needs an object identity")
        _require(self.stratum == "filament", "only filament twins are implemented")
        _require(
            edges.ndim == 2 and edges.shape[1] == 2 and len(edges) >= 3,
            "reusable-twin spring edges are invalid",
        )
        _require(families.shape == (len(edges),), "spring family count differs")
        _require(
            rest.shape == (len(edges),)
            and np.all(np.isfinite(rest))
            and np.all(rest > 1e-6),
            "reusable-twin rest lengths are invalid",
        )
        _require(
            len(self.source_episode_ids) == len(self.source_geometry_sha256) >= 3,
            "reusable twin needs at least three source prefixes",
        )
        _require(
            len(set(self.source_episode_ids)) == len(self.source_episode_ids),
            "reusable-twin source episode is repeated",
        )
        _require(
            all(_valid_sha256(value) for value in self.source_geometry_sha256),
            "reusable-twin source checksum is invalid",
        )
        _require(
            self.fit_policy.get("future_outcomes_read") is False
            and self.fit_policy.get("information_scope")
            == "source-prefix-geometry-only",
            "reusable-twin fit crossed its information boundary",
        )
        for name, values in (
            ("spring_edges", edges),
            ("spring_families", families),
            ("object_rest_lengths_m", rest),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    def rest_lengths_for_graph(self, graph: Deform360SparseGraph) -> np.ndarray:
        """Return the shared metric after exact automatic topology matching."""

        _require(graph.stratum == self.stratum, "episode graph stratum differs")
        _require(
            np.array_equal(graph.spring_edges, self.spring_edges)
            and np.array_equal(graph.spring_families, self.spring_families),
            "episode graph topology differs from the reusable twin",
        )
        return self.object_rest_lengths_m.copy()

    def as_artifact(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": REUSABLE_TWIN_SCHEMA_VERSION,
            "artifact_kind": "Deform360ReusableTwin",
            "object_id": self.object_id,
            "stratum": self.stratum,
            "spring_edges": self.spring_edges.astype(int).tolist(),
            "spring_families": self.spring_families.astype(int).tolist(),
            "object_rest_lengths_m": self.object_rest_lengths_m.tolist(),
            "source_episode_ids": list(self.source_episode_ids),
            "source_geometry_sha256": list(self.source_geometry_sha256),
            "fit_policy": self.fit_policy,
            "diagnostics": self.diagnostics,
        }
        payload["result_sha256"] = _result_sha256(payload)
        return payload


def _filament_total_length(graph: Deform360SparseGraph) -> float:
    _require(graph.stratum == "filament", "reusable fit requires filament graphs")
    stretch = graph.spring_edges[graph.spring_families == 0]
    expected = np.column_stack(
        (np.arange(len(graph.positions_m) - 1), np.arange(1, len(graph.positions_m)))
    )
    _require(
        np.array_equal(stretch, expected),
        "filament graph does not have the canonical open-chain topology",
    )
    return float(
        np.sum(
            np.linalg.norm(
                graph.positions_m[stretch[:, 1]] - graph.positions_m[stretch[:, 0]],
                axis=1,
            )
        )
    )


def fit_reusable_filament_twin(
    object_id: str,
    source_graphs: Sequence[Deform360SparseGraph],
    source_episode_ids: Sequence[str],
    source_geometry_sha256: Sequence[str],
    *,
    rest_length_quantile: float = 0.10,
) -> Deform360ReusableTwin:
    """Fit an intrinsic rest-length quantile from source prefixes only."""

    graphs = tuple(source_graphs)
    episode_ids = tuple(map(str, source_episode_ids))
    geometry_hashes = tuple(map(str, source_geometry_sha256))
    _require(
        len(graphs) == len(episode_ids) == len(geometry_hashes) >= 3,
        "source inputs differ",
    )
    _require(
        0.0 <= rest_length_quantile <= 1.0,
        "rest-length quantile must be in the unit interval",
    )
    reference = graphs[0]
    for graph in graphs[1:]:
        _require(
            graph.stratum == reference.stratum
            and np.array_equal(graph.spring_edges, reference.spring_edges)
            and np.array_equal(graph.spring_families, reference.spring_families),
            "source graph topology is not reusable",
        )
    observed = np.asarray([_filament_total_length(graph) for graph in graphs])
    total_rest = float(np.quantile(observed, rest_length_quantile, method="linear"))
    node_count = len(reference.positions_m)
    segment_rest = total_rest / float(node_count - 1)
    edge_hops = np.abs(reference.spring_edges[:, 1] - reference.spring_edges[:, 0])
    _require(np.all(np.isin(edge_hops, (1, 2))), "filament graph has a nonlocal spring")
    rest = segment_rest * edge_hops
    initial_relative_strain = []
    for graph in graphs:
        current = np.linalg.norm(
            graph.positions_m[graph.spring_edges[:, 1]]
            - graph.positions_m[graph.spring_edges[:, 0]],
            axis=1,
        )
        relative = np.abs(current / rest - 1.0)
        initial_relative_strain.append(
            {
                "median": float(np.median(relative)),
                "p99": float(np.quantile(relative, 0.99)),
                "maximum": float(np.max(relative)),
            }
        )
    quantile_levels = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    diagnostics = {
        "observed_source_centerline_lengths_m": observed.tolist(),
        "observed_length_quantiles_m": {
            f"q{round(level * 100):02d}": float(
                np.quantile(observed, level, method="linear")
            )
            for level in quantile_levels
        },
        "observed_length_coefficient_of_variation": float(
            np.std(observed) / np.mean(observed)
        ),
        "selected_total_rest_length_m": total_rest,
        "selected_segment_rest_length_m": segment_rest,
        "initial_relative_strain_by_source": initial_relative_strain,
    }
    return Deform360ReusableTwin(
        object_id=object_id,
        stratum="filament",
        spring_edges=reference.spring_edges,
        spring_families=reference.spring_families,
        object_rest_lengths_m=rest,
        source_episode_ids=episode_ids,
        source_geometry_sha256=geometry_hashes,
        fit_policy={
            "information_scope": "source-prefix-geometry-only",
            "future_outcomes_read": False,
            "estimator": "empirical-quantile-of-observed-centerline-length",
            "rest_length_quantile": rest_length_quantile,
            "episode_state_policy": "retain-observed-prefix-graph",
            "association_policy": "normalized-open-chain-arc-coordinate",
        },
        diagnostics=diagnostics,
    )


def load_reusable_twin_artifact(path: str | Path) -> Deform360ReusableTwin:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == REUSABLE_TWIN_SCHEMA_VERSION,
        "reusable-twin schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360ReusableTwin",
        "reusable-twin artifact kind changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "reusable-twin checksum mismatch",
    )
    return Deform360ReusableTwin(
        object_id=str(payload["object_id"]),
        stratum=str(payload["stratum"]),
        spring_edges=np.asarray(payload["spring_edges"], dtype=np.int32),
        spring_families=np.asarray(payload["spring_families"], dtype=np.int8),
        object_rest_lengths_m=np.asarray(
            payload["object_rest_lengths_m"], dtype=np.float64
        ),
        source_episode_ids=tuple(map(str, payload["source_episode_ids"])),
        source_geometry_sha256=tuple(map(str, payload["source_geometry_sha256"])),
        fit_policy=dict(payload["fit_policy"]),
        diagnostics=dict(payload["diagnostics"]),
    )


def write_reusable_twin_artifact(path: str | Path, twin: Deform360ReusableTwin) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(twin.as_artifact(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "Deform360ReusableTwin",
    "fit_reusable_filament_twin",
    "load_reusable_twin_artifact",
    "write_reusable_twin_artifact",
]
