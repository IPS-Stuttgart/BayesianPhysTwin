from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_deform360_group_tempered_experiment.py"
PROTOCOL = ROOT / "protocols" / "deform360_group_tempered_experiment_v1.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("_deform360_group_tempered", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol(module):
    return module._load_protocol(PROTOCOL)[0]


def test_protocol_is_locked_and_well_formed() -> None:
    module = _load_script()
    payload, digest = module._load_protocol(PROTOCOL)

    assert payload["status"] == "external-group-held-out-nonofficial"
    assert len(digest) == 64
    assert payload["temperature"]["log2_exponents"][0] < 0
    assert payload["temperature"]["log2_exponents"][-1] > 0
    assert payload["split"]["group_identity"] == "canonical-object-directory"
    assert payload["split"]["minimum_calibration_groups"] == 1
    assert len(payload["cohort"]["expected_object_ids"]) == 6
    assert payload["cohort"]["expected_archive_count"] == 36


def test_group_identity_uses_locked_archive_layout(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "001-rope" / "episode_0004" / "sampled_hulls.npz"
    path.parent.mkdir(parents=True)
    path.touch()

    identity = module._group_identity(path, tmp_path)

    assert identity == ("001-rope", "001-rope")
    assert module._group_identity(tmp_path / "unscoped.npz", tmp_path) is None
    wrong = tmp_path / "001-rope" / "episode_0004" / "tracks.npz"
    wrong.touch()
    assert module._group_identity(wrong, tmp_path) is None


def test_discovery_accepts_only_locked_sampled_hull_layout(tmp_path: Path) -> None:
    module = _load_script()
    accepted = tmp_path / "001-rope" / "episode_0004" / "sampled_hulls.npz"
    accepted.parent.mkdir(parents=True)
    accepted.touch()
    contaminated = tmp_path / "results" / "001-rope" / "episode_0004" / "x.npz"
    contaminated.parent.mkdir(parents=True)
    contaminated.touch()

    specs, excluded = module._discover_specs(tmp_path, maximum_paths=10)

    assert [spec.relative_path for spec in specs] == [
        "001-rope/episode_0004/sampled_hulls.npz"
    ]
    assert [spec.group_id for spec in specs] == ["001-rope"]
    assert excluded == ("results/001-rope/episode_0004/x.npz",)


def test_group_split_is_disjoint_and_deterministic() -> None:
    module = _load_script()
    protocol = _protocol(module)
    groups = [f"{index:03d}-rope/session" for index in range(12)]

    first = module._split_groups(groups, protocol)
    second = module._split_groups(list(reversed(groups)), protocol)

    assert first == second
    assert set(first["source"]).isdisjoint(first["calibration"])
    assert set(first["source"]).isdisjoint(first["target"])
    assert set(first["calibration"]).isdisjoint(first["target"])
    assert set(first["source"] + first["calibration"] + first["target"]) == set(
        groups
    )
    assert len(first["source"]) >= 2
    assert len(first["calibration"]) >= 1
    assert len(first["target"]) >= 3


def test_archive_cap_preserves_group_diversity(tmp_path: Path) -> None:
    module = _load_script()
    specs = []
    for group_index in range(6):
        for archive_index in range(4):
            relative = f"{group_index:03d}-rope/session/archive_{archive_index}.npz"
            specs.append(
                module.ArchiveSpec(
                    path=tmp_path / relative,
                    relative_path=relative,
                    object_id=f"{group_index:03d}-rope",
                    group_id=f"{group_index:03d}-rope/session",
                )
            )

    capped = module._cap_specs_by_group(
        specs,
        maximum_archives=6,
        hash_salt="test",
    )

    assert len(capped) == 6
    assert len({spec.group_id for spec in capped}) == 6


def _posterior(*, evidence_scale: float, update_count: int):
    return SimpleNamespace(
        component_log_evidence=np.asarray(
            [[-2.0, -4.0], [-3.0, -1.0]], dtype=np.float64
        )
        * evidence_scale,
        update_count=np.asarray([update_count, update_count], dtype=np.int64),
        config=SimpleNamespace(component_prior_probability=(0.5, 0.5)),
        component_mean_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        component_variance_m2=np.asarray(
            [[1.0, 1.0], [1.0, 1.0]], dtype=np.float64
        ),
        component_process_variance_m2=np.asarray([0.0, 0.0], dtype=np.float64),
        component_weights=np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
    )


def test_per_observation_evidence_removes_prefix_count_scaling() -> None:
    module = _load_script()
    short = module._normalized_tempered_moments(
        _posterior(evidence_scale=1.0, update_count=2),
        temperature=1.0,
        horizon_steps=1,
    )
    repeated = module._normalized_tempered_moments(
        _posterior(evidence_scale=2.0, update_count=4),
        temperature=1.0,
        horizon_steps=1,
    )

    np.testing.assert_allclose(short.component_weights, repeated.component_weights)
    np.testing.assert_allclose(short.mean_m, repeated.mean_m)
    np.testing.assert_allclose(short.covariance_m2, repeated.covariance_m2)


def test_temperature_selection_records_an_interior_optimum() -> None:
    module = _load_script()
    temperatures = (0.5, 1.0, 2.0)
    source = {
        group: {
            "one_step_nll_by_temperature": {
                "0.5": 2.0 + offset,
                "1": 1.0 + offset,
                "2": 3.0 + offset,
            }
        }
        for group, offset in (("g0", 0.0), ("g1", 0.1))
    }

    selected = module._select_temperature(source, temperatures)

    assert selected["selected_temperature"] == 1.0
    assert selected["interior_bracketed"] is True


def test_guard_accepts_only_source_safe_scores() -> None:
    module = _load_script()
    protocol = _protocol(module)
    source = {
        "g0": {
            "archives": [
                {
                    "guard_records": [
                        {"score": 1.0, "candidate_regret_m": -0.002},
                        {"score": 3.0, "candidate_regret_m": 0.004},
                    ]
                }
            ]
        },
        "g1": {
            "archives": [
                {
                    "guard_records": [
                        {"score": 1.5, "candidate_regret_m": -0.001},
                        {"score": 2.5, "candidate_regret_m": 0.003},
                    ]
                }
            ]
        },
    }

    selection = module._select_guard(source, protocol)

    assert selection["selected_threshold"] == pytest.approx(1.5)
    assert selection["selected_record"]["maximum_group_regret_m"] <= 0.0
    assert selection["selected_record"]["accepted_step_count"] == 2


def test_smooth_horizon_scale_and_independent_multiplier() -> None:
    module = _load_script()
    protocol = _protocol(module)
    protocol["uncertainty"]["maximum_horizon"] = 3
    source = {
        "g0": {
            "horizon_nees": {
                "1": [5.0, 6.0, 7.0],
                "2": [8.0, 9.0, 10.0],
                "3": [11.0, 12.0, 13.0],
            }
        },
        "g1": {
            "horizon_nees": {
                "1": [4.0, 5.0, 6.0],
                "2": [7.0, 8.0, 9.0],
                "3": [10.0, 11.0, 12.0],
            }
        },
    }

    shape = module._source_horizon_scales(source, protocol)
    scales = [shape["scales_by_horizon"][str(index)] for index in (1, 2, 3)]
    assert scales == sorted(scales)
    assert all(scale >= 1.0 for scale in scales)

    calibration = {
        "c0": {
            "horizon_nees": {
                "1": [6.0, 7.0],
                "2": [9.0, 10.0],
                "3": [12.0, 13.0],
            }
        },
        "c1": {
            "horizon_nees": {
                "1": [7.0, 8.0],
                "2": [10.0, 11.0],
                "3": [13.0, 14.0],
            }
        },
    }
    conformal = module._calibration_multiplier(calibration, shape, protocol)
    assert conformal["multiplier"] >= 1.0
    assert set(conformal["group_scores"]) == {"c0", "c1"}


def test_paired_bootstrap_is_group_level_and_deterministic() -> None:
    module = _load_script()
    candidate = np.asarray([1.0, 2.0, 3.0])
    reference = np.asarray([2.0, 3.0, 4.0])

    first = module._paired_bootstrap(candidate, reference, samples=200, seed=7)
    second = module._paired_bootstrap(candidate, reference, samples=200, seed=7)

    assert first == second
    assert first["mean_delta_m"] == pytest.approx(-1.0)
    assert first["bootstrap_probability_mean_improvement"] == pytest.approx(1.0)
