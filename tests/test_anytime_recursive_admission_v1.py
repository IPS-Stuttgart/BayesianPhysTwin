import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin_experiments.anytime_recursive_admission_v1 as experiment
from bayesian_phystwin_experiments.anytime_recursive_admission_v1 import (
    AnytimeRecursiveAdmissionV1Config,
    canonical_result_digest,
    load_anytime_recursive_protocol,
    run_anytime_recursive_admission_v1,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols/anytime_recursive_admission_v1.json"
RUNNER = REPOSITORY_ROOT / "scripts/science/run_anytime_recursive_admission_v1.py"


def test_protocol_binds_fresh_roster_and_statistical_contract(
    tmp_path: Path,
) -> None:
    config = load_anytime_recursive_protocol(PROTOCOL)

    assert config.seed_start == 200_000
    assert config.seed_count == 400
    assert config.seeds[0] == 200_000
    assert config.seeds[-1] == 200_399
    assert config.minimum_mean_gain_m == pytest.approx(0.00025)
    assert config.maximum_harm_rate == pytest.approx(0.10)
    assert config.total_alpha_gain == pytest.approx(0.025)
    assert config.total_alpha_harm == pytest.approx(0.025)

    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["information_boundary"]["fresh_seed_outcomes_opened_before_protocol"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary changed"):
        load_anytime_recursive_protocol(changed)


def test_stream_registers_before_outcome_and_preserves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_seeds: list[int] = []

    def fake_evaluate_seed(
        seed: int,
        *,
        conditions: object,
        benchmark_config: object,
    ) -> dict[str, object]:
        del conditions, benchmark_config
        opened_seeds.append(seed)
        fallback = np.asarray((seed, seed + 1), dtype=np.float64)
        candidate = fallback - 0.001
        return {
            "candidate_loss_m": 0.002,
            "fallback_loss_m": 0.014,
            "candidate_forecast": candidate,
            "fallback_forecast": fallback,
            "candidate_harmful_update_count": 0,
            "candidate_accepted_update_count": 10,
            "candidate_exact_fallback_violation_count": 0,
        }

    monkeypatch.setattr(experiment, "_evaluate_seed", fake_evaluate_seed)
    config = AnytimeRecursiveAdmissionV1Config(
        seed_start=900,
        seed_count=80,
        delay_min_episodes=1,
        delay_max_episodes=3,
        delay_seed=7,
        minimum_resolved_trials=5,
        null_world_count=20,
        null_epoch_count=2,
        null_trials_per_epoch=20,
        null_seed=8,
    )

    result = run_anytime_recursive_admission_v1(config)
    stream = result["fresh_stream"]
    records = result["records"]

    assert opened_seeds == list(config.seeds)
    assert stream["first_authorized_issue_index"] is not None
    assert stream["authorized_deployment_count"] > 0
    assert stream["fallback_deployment_count"] > 0
    assert stream["exact_fallback_violation_count"] == 0
    assert all(record["exact_fallback_valid"] for record in records)
    assert result["decision"]["fresh_stream_authorized_at_least_once"] is True
    assert result["decision"]["selected_stream_improves_fallback"] is True


def test_negative_stream_remains_on_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_harmful_seed(
        seed: int,
        *,
        conditions: object,
        benchmark_config: object,
    ) -> dict[str, object]:
        del seed, conditions, benchmark_config
        fallback = np.asarray((0.0, 1.0), dtype=np.float64)
        candidate = np.asarray((2.0, 3.0), dtype=np.float64)
        return {
            "candidate_loss_m": 0.014,
            "fallback_loss_m": 0.002,
            "candidate_forecast": candidate,
            "fallback_forecast": fallback,
            "candidate_harmful_update_count": 5,
            "candidate_accepted_update_count": 10,
            "candidate_exact_fallback_violation_count": 0,
        }

    monkeypatch.setattr(experiment, "_evaluate_seed", fake_harmful_seed)
    config = AnytimeRecursiveAdmissionV1Config(
        seed_start=1000,
        seed_count=40,
        delay_min_episodes=1,
        delay_max_episodes=1,
        minimum_resolved_trials=5,
        null_world_count=10,
        null_epoch_count=1,
        null_trials_per_epoch=10,
    )

    result = run_anytime_recursive_admission_v1(config)
    stream = result["fresh_stream"]

    assert stream["authorized_deployment_count"] == 0
    assert stream["fallback_deployment_count"] == 40
    assert stream["selected_mean_loss_m"] == pytest.approx(
        stream["fallback_mean_loss_m"]
    )
    assert stream["selected_harmful_episode_count"] == 0
    assert stream["exact_fallback_violation_count"] == 0


def test_canonical_digest_is_key_order_invariant() -> None:
    first = {"b": [2, 3], "a": {"x": 1}}
    second = {"a": {"x": 1}, "b": [2, 3]}

    assert canonical_result_digest(first) == canonical_result_digest(second)


def test_runner_writes_method_seal_before_fresh_stream() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    seal_write = source.index("_write_json(method_seal_path, method_seal)")
    experiment_run = source.index("result = run_anytime_recursive_admission_v1(config)")
    assert seal_write < experiment_run
    assert '"fresh_seed_outcomes_opened": False' in source
    assert '"target_dependent_retuning": False' in source
