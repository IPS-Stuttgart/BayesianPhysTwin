from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import bayesian_phystwin.jax_fem_hyperelastic_source_qualification_v2 as module
from bayesian_phystwin.jax_fem_hyperelastic_source_qualification_v2 import (
    load_jax_fem_hyperelastic_source_protocol_v2,
    run_jax_fem_hyperelastic_source_qualification_v2,
    runtime_descriptor_v2,
)
from bayesian_phystwin.jax_fem_hyperelastic_v2 import (
    HyperelasticReplayV2,
    interpolate_rotation_v2,
)
from bayesian_phystwin.jax_fem_source_qualification_v1 import (
    file_sha256,
    rigid_transform_v1,
)
from bayesian_phystwin.physical_rollout_v1 import write_deterministic_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/jax_fem_zebra_source_physics_v2.json"
BASE_PROTOCOL = ROOT / "configs/sota/jax_fem_zebra_source_physics_v1.json"
RUNNER = ROOT / "scripts/remote/run_jax_fem_hyperelastic_source_qualification_v2.py"


def test_frozen_v2_protocol_binds_the_v1_roster_and_runtime_identity() -> None:
    protocol = load_jax_fem_hyperelastic_source_protocol_v2(PROTOCOL)

    assert protocol.base_protocol.protocol_sha256 == file_sha256(BASE_PROTOCOL)
    assert [group.group_id for group in protocol.base_protocol.source_groups] == [
        "double_lift_zebra",
        "double_stretch_zebra",
    ]
    assert (
        protocol.runtime_id
        == "0c1a24a70c805eb6ade62d176fafd574b3fc1c07fef8ca80592db6bb9ad23d15"
    )
    assert protocol.runtime_id == module.content_id(
        runtime_descriptor_v2(protocol.backend, protocol.simulation)
    )
    assert protocol.simulation["base_interval_substeps"] == 1
    assert protocol.simulation["refined_interval_substeps"] == 2
    assert (
        protocol.value["information_boundary"]["source_object_outcomes_allowed"]
        is False
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["information_boundary"].update(
                source_object_outcomes_allowed=True
            ),
            "information boundary",
        ),
        (
            lambda value: value["simulation"].update(
                constitutive_model="small-strain-isotropic-linear-elasticity"
            ),
            "constitutive model",
        ),
        (
            lambda value: value["backend"].update(runtime_id="0" * 64),
            "runtime_id",
        ),
    ],
)
def test_v2_protocol_rejects_custody_or_method_mutation(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    value: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["base_protocol"]["relative_path"] = BASE_PROTOCOL.name
    (tmp_path / BASE_PROTOCOL.name).write_bytes(BASE_PROTOCOL.read_bytes())
    mutation(value)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jax_fem_hyperelastic_source_protocol_v2(path)


def test_rotation_interpolation_stays_in_so3_and_preserves_endpoints() -> None:
    left = rigid_transform_v1([1.0, 2.0, 3.0], 0.17)
    right = rigid_transform_v1([3.0, 1.0, 2.0], 0.83)

    np.testing.assert_array_equal(interpolate_rotation_v2(left, right, 0.0), left)
    np.testing.assert_array_equal(interpolate_rotation_v2(left, right, 1.0), right)
    middle = interpolate_rotation_v2(left, right, 0.5)
    np.testing.assert_allclose(middle.T @ middle, np.eye(3), atol=1e-12, rtol=0.0)
    assert np.linalg.det(middle) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: module._mapping([], name="value"), "must be a mapping"),
        (
            lambda: module._exact_fields(
                {"extra": 1}, frozenset({"required"}), "value"
            ),
            "fields changed",
        ),
        (lambda: module._canonical_string(" padded ", name="value"), "canonical"),
        (lambda: module._sha256("0" * 63, name="value"), "SHA-256"),
        (lambda: module._git_revision("g" * 40, name="value"), "Git revision"),
        (lambda: module._positive_int(0, name="value"), "positive integer"),
        (lambda: module._nonnegative_int(-1, name="value"), "nonnegative integer"),
        (lambda: module._finite(True, name="value"), "must be finite"),
        (
            lambda: module._finite(float("nan"), name="value"),
            "must be finite",
        ),
        (
            lambda: module._canonical_relative_path("../value", name="value"),
            "not canonical",
        ),
        (
            lambda: module._canonical_relative_path("dir\\value", name="value"),
            "POSIX separators",
        ),
        (lambda: module._integer_tuple([], name="value"), "nonempty integer list"),
        (
            lambda: module._integer_tuple([1, 1], name="value"),
            "sorted and unique",
        ),
    ],
)
def test_v2_protocol_helpers_reject_noncanonical_values(
    operation: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def _points() -> npt.NDArray[np.float32]:
    tetrahedron = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.002],
        ],
        dtype=np.float32,
    )
    return np.asarray(
        np.concatenate((tetrahedron, tetrahedron + [0.03, 0.0, 0.0])),
        dtype=np.float32,
    )


def _source_archive(path: Path, points: npt.NDArray[np.float32]) -> None:
    controller = np.zeros((10, 2, 3), dtype=np.float32)
    controller[:, 0, 0] = np.linspace(0.0, 0.002, 10)
    controller[:, 1, 1] = np.linspace(0.0, 0.001, 10)
    weights = np.zeros((len(points), 2), dtype=np.float32)
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0
    write_deterministic_npz(
        path,
        {
            "frame_zero_points_m": points,
            "controller_points_m": controller,
            "attachment_indices": np.arange(len(points), dtype=np.int32),
            "attachment_weights": weights,
            "action_support": np.ones(len(points), dtype=np.float32),
        },
    )


def _incumbent_archive(path: Path, points: npt.NDArray[np.float32]) -> Path:
    prediction = np.repeat(points[None], 10, axis=0).astype(np.float32)
    return cast(
        Path,
        write_deterministic_npz(
            path,
            {
                "action_support": np.ones(len(points), dtype=np.float32),
                "driven_readout_m": prediction,
                "frame_zero_points_m": prediction[0],
                "persistence_m": prediction.copy(),
                "prediction_m": prediction,
                "zero_action_readout_m": prediction.copy(),
            },
        ),
    )


def _synthetic_protocol_and_roots(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    base: dict[str, Any] = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    points = _points()
    roots: dict[str, Path] = {}
    for group in base["source_groups"]:
        group_id = str(group["group_id"])
        root = tmp_path / "source" / group_id
        root.mkdir(parents=True)
        roots[group_id] = root
        source = root / "source-inputs.npz"
        _source_archive(source, points)
        incumbent = _incumbent_archive(root / "incumbent.npz", points)
        group.update(
            {
                "source_inputs_relative_path": source.name,
                "source_inputs_sha256": file_sha256(source),
                "incumbent_relative_path": incumbent.name,
                "incumbent_sha256": file_sha256(incumbent),
                "frame_count": 10,
                "material_node_count": len(points),
                "controller_point_count": 2,
                "attached_node_count": len(points),
                "expected_contact_patch_sizes": [4, 4],
                "expected_base_cell_count": 3,
                "expected_coarse_cell_count": 3,
            }
        )
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")

    overlay: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    overlay["base_protocol"] = {
        "relative_path": base_path.name,
        "sha256": file_sha256(base_path),
    }
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay), encoding="utf-8")
    return overlay_path, roots


def _fake_replay(**kwargs: Any) -> HyperelasticReplayV2:
    points = np.asarray(kwargs["points_m"], dtype=np.float64)
    contact = kwargs["contact"]
    poisson = float(kwargs["poisson_ratio"])
    driven = bool(kwargs["driven"])
    attached = np.concatenate(contact.patch_local_indices)
    positions: list[npt.NDArray[np.float64]] = []
    for frame in range(len(contact.rotations)):
        displacement = np.zeros(3, dtype=np.float64)
        if driven:
            target = contact.projected_targets_m[frame, attached]
            reference = points[np.asarray(kwargs["attachment_indices"])[attached]]
            displacement = np.mean(target - reference, axis=0)
            displacement *= 1.0 + 0.3 * (poisson - 0.35)
        positions.append(points + displacement)
    trajectory = np.ascontiguousarray(np.stack(positions))
    return HyperelasticReplayV2(
        positions_m=trajectory,
        deformation_determinants=np.ones(
            (len(trajectory), len(kwargs["cells"])), dtype=np.float64
        ),
        minimum_continuation_deformation_determinant=1.0,
        native_solve_count=len(trajectory) * int(kwargs["interval_substeps"]),
    )


def test_v2_qualification_copies_exact_fallback_without_outcome_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, roots = _synthetic_protocol_and_roots(tmp_path)
    cells = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5]], dtype=np.int32)
    monkeypatch.setattr(
        module,
        "_git_provenance",
        lambda _: {
            "git_head": "1" * 40,
            "git_worktree_clean": True,
            "source_files": {"synthetic.py": "2" * 64},
        },
    )
    monkeypatch.setattr(
        module,
        "load_native_jax_fem_modules_v2",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "normalized_objectivity_errors_v2",
        lambda *_, **__: (0.0, 0.0),
    )
    monkeypatch.setattr(
        module,
        "build_tetrahedral_cells_v1",
        lambda *_, **__: cells.copy(),
    )
    monkeypatch.setattr(module, "_run_native_replay_v2", _fake_replay)

    output = tmp_path / "result"
    result = run_jax_fem_hyperelastic_source_qualification_v2(
        protocol_path=protocol,
        group_roots=roots,
        output_dir=output,
        repo_root=ROOT,
    )

    assert result["qualified"] is True
    assert result["source_value_scoring_authorized"] is True
    assert result["information_boundary"] == {
        "source_inputs_read": True,
        "incumbent_predictions_read": True,
        "source_object_outcomes_read": False,
        "target_or_held_out_artifact_read": False,
    }
    base = json.loads((protocol.parent / "base.json").read_text(encoding="utf-8"))
    for group in base["source_groups"]:
        incumbent = roots[group["group_id"]] / group["incumbent_relative_path"]
        fallback = output / group["group_id"] / "exact-incumbent-fallback.npz"
        assert fallback.read_bytes() == incumbent.read_bytes()


def test_v2_qualification_rejects_incomplete_roster_or_existing_output(
    tmp_path: Path,
) -> None:
    protocol, roots = _synthetic_protocol_and_roots(tmp_path)

    with pytest.raises(ValueError, match="complete frozen source roster"):
        run_jax_fem_hyperelastic_source_qualification_v2(
            protocol_path=protocol,
            group_roots={},
            output_dir=tmp_path / "unused",
            repo_root=ROOT,
        )

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_jax_fem_hyperelastic_source_qualification_v2(
            protocol_path=protocol,
            group_roots=roots,
            output_dir=output,
            repo_root=ROOT,
        )


def test_v2_runner_help_does_not_import_native_jax_fem() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--group-root" in completed.stdout
