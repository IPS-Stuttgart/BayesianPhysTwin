import importlib.util
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_action_robust_all18 import SOURCE_FIELD
from bayesian_phystwin.pokeflex_missing5_scale import (
    BASE_EFFECTIVE_SCALE,
    CANDIDATE_MULTIPLIERS,
    OFFICIAL_TARGET_TAKES,
    SOURCE_TAKES,
    UPSTREAM_COMMIT,
    build_source_protocol,
    build_source_result,
    protocol_sha256,
    select_cross_validated_multiplier,
    source_take_ids,
    synthetic_control_summary,
    take_row_from_smoke,
    validate_source_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
REMOTE_RUNNER = (
    ROOT / "scripts" / "remote" / "run_pokeflex_missing5_scale_source_take.py"
)


def _protocol() -> dict[str, object]:
    inventory = {
        take_id: {
            "relative_path": f"{take_id.rpartition('_T')[0]}/{take_id}.zip",
            "sha256": f"{index:064x}",
            "bytes": index + 100,
        }
        for index, take_id in enumerate(source_take_ids(), start=1)
    }
    return build_source_protocol(
        inventory,
        locked_at_utc="2026-08-06T00:00:00Z",
        implementation_revision="a" * 40,
        source_projection_runner_file_sha256="b" * 64,
        source_runner_file_sha256="c" * 64,
        legacy_runner_file_sha256="d" * 64,
        registration_protocol_file_sha256="e" * 64,
    )


def _scores(gains: dict[float, float], *, baseline: float = 10.0) -> dict[str, float]:
    return {
        f"{multiplier:g}": baseline * (1.0 - gains.get(multiplier, -0.05))
        for multiplier in CANDIDATE_MULTIPLIERS
    }


def _row(take_id: str, gains: dict[float, float]) -> dict[str, object]:
    return {"take_id": take_id, "scores_CD_UL1_mm": _scores(gains)}


def _smoke(
    take_id: str,
    protocol: dict[str, object],
    *,
    preferred_multiplier: float = 2.0,
) -> dict[str, object]:
    aggregates = {}
    for multiplier in CANDIDATE_MULTIPLIERS:
        scale = BASE_EFFECTIVE_SCALE * multiplier
        score = 10.0
        if multiplier == preferred_multiplier:
            score = 9.7
        elif multiplier != 1.0:
            score = 10.2
        aggregates[
            f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"
        ] = {"mean_CD_UL1_mm": score}
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "missing5_source_protocol_sha256": protocol["protocol_sha256"],
        "source_runner_file_sha256": protocol["implementation"][
            "source_runner_file_sha256"
        ],
        "legacy_runner_file_sha256": protocol["implementation"][
            "legacy_runner_file_sha256"
        ],
        "future_observation_used": False,
        "correction_fields": [SOURCE_FIELD],
        "take": {"id": take_id},
        "source_archive_sha256": protocol["archive_inventory"]["takes"][take_id][
            "sha256"
        ],
        "projection_manifest_sha256": "f" * 64,
        "official_target_outcome_used": False,
        "held_v8_accessed": False,
        "upstream": {"git_commit": UPSTREAM_COMMIT},
        "aggregates": aggregates,
        "updates": [
            {"accepted": True, "action_supported": True},
            {"accepted": False, "action_supported": True},
        ],
    }


def test_source_inventory_is_exact_and_official_targets_are_disjoint() -> None:
    assert len(source_take_ids()) == 30
    assert sum(len(takes) for takes in SOURCE_TAKES.values()) == 30
    assert set(SOURCE_TAKES) == set(OFFICIAL_TARGET_TAKES)
    assert not (set(source_take_ids()) & set(OFFICIAL_TARGET_TAKES.values()))


def test_protocol_is_canonical_and_rejects_resigned_target_change() -> None:
    protocol = _protocol()
    validation = validate_source_protocol(protocol)
    assert validation["passed"] is True
    assert protocol["protocol_sha256"] == protocol_sha256(protocol)

    changed = deepcopy(protocol)
    changed["source_cohort"]["official_target_takes"]["Sponge"] = "Sponge_T9"
    changed["protocol_sha256"] = protocol_sha256(changed)
    with pytest.raises(ValueError, match="official target"):
        validate_source_protocol(changed)


def test_selector_promotes_cross_action_safe_scale() -> None:
    rows = [
        _row(
            f"Pillow_T{index}",
            {0.5: 0.005, 1.0: 0.0, 1.5: 0.015, 2.0: 0.03},
        )
        for index in range(1, 7)
    ]

    selected = select_cross_validated_multiplier(rows)

    assert selected["promoted"] is True
    assert selected["multiplier"] == 2.0
    assert selected["strict_loo_win_count"] == 6
    assert selected["deployed_loo_regression_count"] == 0


def test_selector_falls_back_when_held_action_reveals_conflict() -> None:
    rows = [
        _row(
            "Sponge_T1",
            {1.0: 0.0, 1.5: 0.02, 2.0: 0.03},
        ),
        _row(
            "Sponge_T2",
            {1.0: 0.0, 1.5: 0.02, 2.0: 0.03},
        ),
        _row(
            "Sponge_T3",
            {1.0: 0.0, 1.5: 0.00, 2.0: -0.10},
        ),
    ]

    selected = select_cross_validated_multiplier(rows)

    assert selected["unpromoted_full_multiplier"] == 1.5
    assert selected["promoted"] is False
    assert selected["multiplier"] == 1.0
    assert selected["loo_regression_count"] == 1
    assert selected["deployed_loo_regression_count"] == 0
    assert selected["source_relative_improvements"] == [0.0, 0.0, 0.0]


def test_synthetic_controls_calibrate_selector_false_admission() -> None:
    controls = synthetic_control_summary()

    assert controls == {
        "positive_control_count": 12,
        "positive_detection_count": 12,
        "placebo_control_count": 12,
        "placebo_admission_count": 0,
        "passed": True,
    }


def test_smoke_row_requires_causal_and_runner_bound_artifact() -> None:
    protocol = _protocol()
    take_id = source_take_ids()[0]
    smoke = _smoke(take_id, protocol)

    row = take_row_from_smoke(smoke, protocol)

    assert row["take_id"] == take_id
    assert row["supported_frame_count"] == 1
    assert row["scores_CD_UL1_mm"]["2"] == 9.7

    leaked = deepcopy(smoke)
    leaked["future_observation_used"] = True
    with pytest.raises(ValueError, match="future observation"):
        take_row_from_smoke(leaked, protocol)

    wrong_runner = deepcopy(smoke)
    wrong_runner["source_runner_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source runner"):
        take_row_from_smoke(wrong_runner, protocol)


def test_complete_source_result_passes_without_target_evidence() -> None:
    protocol = _protocol()
    takes = source_take_ids()
    artifacts = [_smoke(take_id, protocol) for take_id in takes]
    digests = {take_id: f"{index:064x}" for index, take_id in enumerate(takes, 1)}

    result = build_source_result(
        artifacts,
        protocol,
        artifact_file_sha256s=digests,
        implementation_revision="a" * 40,
    )

    assert result["source_gate"]["passed"] is True
    assert result["source_gate"]["adjusted_object_count"] == 5
    assert result["source_gate"]["deployed_loo_held_action_regression_count"] == 0
    assert result["official_target_outcomes_used"] is False
    assert result["held_v8_accessed"] is False
    json.dumps(result, allow_nan=False)


def test_remote_wrapper_runs_exact_bank_and_restores_legacy_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_loader(_path):
        return {"payload": {"cohort": {"development_objects": ["Other"]}}}

    def fake_summary(values):
        return {"mean_CD_UL1_mm": sum(values) / len(values)}

    def fake_run_smoke(*args, **kwargs):
        captured["kwargs"] = kwargs
        captured["authorized_objects"] = tuple(
            fake_module.load_pokeflex_registration_protocol(None)["payload"][
                "cohort"
            ]["development_objects"]
        )
        targets = []
        aggregates = {}
        for frame in (6, 7):
            target = {
                "target_frame": frame,
                "released_checkpoint_CD_UL1_mm": float(frame),
            }
            for scale in kwargs["correction_scales"]:
                key = f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"
                target[key] = float(frame)
                aggregates[key] = {"mean_CD_UL1_mm": 6.5}
            targets.append(target)
        return {
            "targets": targets,
            "updates": [
                {"target_frame": frame, "accepted": True, "action_supported": True}
                for frame in (6, 7)
            ],
            "aggregates": aggregates,
        }

    fake_module = types.ModuleType("run_pokeflex_checkpoint_registration_smoke")
    fake_module.json = json
    fake_module.load_pokeflex_registration_protocol = fake_loader
    fake_module.run_smoke = fake_run_smoke
    fake_module._summary = fake_summary
    monkeypatch.setitem(
        sys.modules,
        "run_pokeflex_checkpoint_registration_smoke",
        fake_module,
    )
    spec = importlib.util.spec_from_file_location("missing5_source_runner", REMOTE_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    take_root = tmp_path / "Pillow_T1"
    take_root.mkdir()
    robot = [
        {
            "frame": 1,
            "forces": [0.0],
            "T_WE": [[1.0, 0.0, 0.0, 0.0]] * 4,
        }
    ]
    (take_root / "robot_data.json").write_text(json.dumps(robot), encoding="utf-8")

    module._run_smoke(
        fake_module,
        take_root=take_root,
        registration_protocol=tmp_path / "registration.json",
        upstream_checkout=tmp_path / "upstream",
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert captured["kwargs"]["correction_scales"] == (
        0.0,
        0.0625,
        0.125,
        0.1875,
        0.25,
        0.375,
        0.5,
    )
    assert "Pillow" in captured["authorized_objects"]
    assert captured["kwargs"]["record_online_observation_regret"] is False
    assert fake_module.load_pokeflex_registration_protocol is fake_loader
    assert fake_module.json is json
