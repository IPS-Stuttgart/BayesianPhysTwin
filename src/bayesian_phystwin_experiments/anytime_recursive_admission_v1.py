"""Fresh episode-stream benchmark for anytime-valid simulator admission.

Each independent recursive-corruption seed-domain is one statistical trial.
Before generating its target trajectory, the benchmark registers a shadow
comparison between the guarded Gaussian correction and the exact physical
baseline.  Its outcome is revealed only after a frozen random episode delay.
The deployed arm is selected from evidence available before that seed-domain is
opened.

The benchmark also runs direct null simulations for the gain and harm
e-processes under continuous monitoring and geometrically alpha-spent restarts.
It is controlled synthetic mechanism evidence, not a deployment-safety or
real-object result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin.anytime_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    BernoulliHarmMixtureEProcess,
    BoundedGainMixtureEProcess,
    GeometricAlphaSpending,
)
from bayesian_phystwin_experiments.recursive_corruption_benchmark_v2 import (
    STRESS_CONDITIONS,
    RecursiveCorruptionV2Config,
    draw_seed_domain,
    generate_corrupted_sequence_v2,
    run_methods_v2,
)

SCHEMA: Final = "bayesian-phystwin.anytime-recursive-admission-v1"
SCHEMA_VERSION: Final = 1
RESULT_SCHEMA: Final = "bayesian-phystwin.anytime-recursive-admission-result-v1"
_Z975: Final = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True, slots=True)
class AnytimeRecursiveAdmissionV1Config:
    """Frozen configuration for one fresh claim-bearing synthetic stream."""

    seed_start: int = 200_000
    seed_count: int = 400
    conditions: tuple[str, ...] = STRESS_CONDITIONS
    delay_min_episodes: int = 1
    delay_max_episodes: int = 12
    delay_seed: int = 2_026_090_2
    loss_cap_m: float = 0.015
    minimum_mean_gain_m: float = 0.00025
    harmful_margin_m: float = 0.0
    maximum_harm_rate: float = 0.10
    total_alpha_gain: float = 0.025
    total_alpha_harm: float = 0.025
    epoch_alpha_continuation: float = 0.5
    minimum_resolved_trials: int = 25
    gain_bet_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.40,
        0.60,
        0.80,
    )
    harm_alternative_fractions: tuple[float, ...] = (
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
    )
    null_world_count: int = 5_000
    null_epoch_count: int = 4
    null_trials_per_epoch: int = 100
    null_seed: int = 2_026_090_3

    def __post_init__(self) -> None:
        if isinstance(self.seed_start, bool) or self.seed_start < 0:
            raise ValueError("seed_start must be a nonnegative literal integer")
        for name, value in (
            ("seed_count", self.seed_count),
            ("delay_min_episodes", self.delay_min_episodes),
            ("delay_max_episodes", self.delay_max_episodes),
            ("minimum_resolved_trials", self.minimum_resolved_trials),
            ("null_world_count", self.null_world_count),
            ("null_epoch_count", self.null_epoch_count),
            ("null_trials_per_epoch", self.null_trials_per_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive literal integer")
        if self.delay_max_episodes < self.delay_min_episodes:
            raise ValueError("delay bounds are reversed")
        for name, value in (
            ("delay_seed", self.delay_seed),
            ("null_seed", self.null_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative literal integer")
        if tuple(self.conditions) != STRESS_CONDITIONS:
            raise ValueError("conditions must equal the registered stress roster")
        if self.loss_cap_m <= 0.0 or not math.isfinite(self.loss_cap_m):
            raise ValueError("loss_cap_m must be positive and finite")
        if self.minimum_mean_gain_m < 0.0 or not math.isfinite(
            self.minimum_mean_gain_m
        ):
            raise ValueError("minimum_mean_gain_m must be nonnegative and finite")
        if self.harmful_margin_m < 0.0 or not math.isfinite(self.harmful_margin_m):
            raise ValueError("harmful_margin_m must be nonnegative and finite")
        for name, value in (
            ("maximum_harm_rate", self.maximum_harm_rate),
            ("total_alpha_gain", self.total_alpha_gain),
            ("total_alpha_harm", self.total_alpha_harm),
            ("epoch_alpha_continuation", self.epoch_alpha_continuation),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))

    @property
    def admission_config(self) -> AnytimeAdmissionConfig:
        return AnytimeAdmissionConfig(
            loss_cap=self.loss_cap_m,
            minimum_mean_gain=self.minimum_mean_gain_m,
            harmful_margin=self.harmful_margin_m,
            maximum_harm_rate=self.maximum_harm_rate,
            total_alpha_gain=self.total_alpha_gain,
            total_alpha_harm=self.total_alpha_harm,
            epoch_alpha_continuation=self.epoch_alpha_continuation,
            minimum_resolved_trials=self.minimum_resolved_trials,
            gain_bet_fractions=self.gain_bet_fractions,
            harm_alternative_fractions=self.harm_alternative_fractions,
        )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_anytime_recursive_protocol(
    path: str | Path,
) -> AnytimeRecursiveAdmissionV1Config:
    """Load the exact frozen protocol and reject silent contract changes."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    root = _mapping(payload, label="protocol")
    if root.get("schema") != SCHEMA or root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported anytime recursive protocol")
    stream = _mapping(root.get("stream"), label="stream")
    admission = _mapping(root.get("admission"), label="admission")
    calibration = _mapping(root.get("null_calibration"), label="null calibration")
    boundary = _mapping(root.get("information_boundary"), label="boundary")
    conditions = stream.get("conditions")
    if not isinstance(conditions, list) or not all(
        isinstance(value, str) for value in conditions
    ):
        raise ValueError("stream conditions must be a string list")
    raw_gain_bets = admission.get("gain_bet_fractions")
    raw_harm_alternatives = admission.get("harm_alternative_fractions")
    if not isinstance(raw_gain_bets, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in raw_gain_bets
    ):
        raise ValueError("gain_bet_fractions must be a numeric list")
    if not isinstance(raw_harm_alternatives, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in raw_harm_alternatives
    ):
        raise ValueError("harm_alternative_fractions must be a numeric list")
    config = AnytimeRecursiveAdmissionV1Config(
        seed_start=int(cast(Any, stream.get("seed_start"))),
        seed_count=int(cast(Any, stream.get("seed_count"))),
        conditions=tuple(conditions),
        delay_min_episodes=int(cast(Any, stream.get("delay_min_episodes"))),
        delay_max_episodes=int(cast(Any, stream.get("delay_max_episodes"))),
        delay_seed=int(cast(Any, stream.get("delay_seed"))),
        loss_cap_m=float(cast(Any, admission.get("loss_cap_m"))),
        minimum_mean_gain_m=float(cast(Any, admission.get("minimum_mean_gain_m"))),
        harmful_margin_m=float(cast(Any, admission.get("harmful_margin_m"))),
        maximum_harm_rate=float(cast(Any, admission.get("maximum_harm_rate"))),
        total_alpha_gain=float(cast(Any, admission.get("total_alpha_gain"))),
        total_alpha_harm=float(cast(Any, admission.get("total_alpha_harm"))),
        epoch_alpha_continuation=float(
            cast(Any, admission.get("epoch_alpha_continuation"))
        ),
        minimum_resolved_trials=int(
            cast(Any, admission.get("minimum_resolved_trials"))
        ),
        gain_bet_fractions=tuple(float(value) for value in raw_gain_bets),
        harm_alternative_fractions=tuple(
            float(value) for value in raw_harm_alternatives
        ),
        null_world_count=int(cast(Any, calibration.get("world_count"))),
        null_epoch_count=int(cast(Any, calibration.get("epoch_count"))),
        null_trials_per_epoch=int(cast(Any, calibration.get("trials_per_epoch"))),
        null_seed=int(cast(Any, calibration.get("seed"))),
    )
    if (
        stream.get("candidate") != "guarded_recursive"
        or stream.get("fallback") != "physical_baseline"
        or stream.get("statistical_unit") != "independent-seed-domain"
        or stream.get("loss") != "equal-condition-mean-rmse-m"
        or admission.get("authorize_when") != "both-current-e-processes-cross"
        or float(cast(Any, admission.get("combined_bad_regime_alpha")))
        != config.total_alpha_gain + config.total_alpha_harm
        or calibration.get("gain_null") != "iid-rademacher-bounded-score"
        or calibration.get("harm_null") != "iid-bernoulli-at-rate-ceiling"
        or boundary.get("fresh_seed_outcomes_opened_before_protocol") is not False
        or boundary.get("real_object_claim") is not False
        or boundary.get("deployment_safety_claim") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("anytime recursive protocol boundary changed")
    return config


def _mean_condition_rmse(
    forecasts: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> float:
    if len(forecasts) != len(targets) or not forecasts:
        raise ValueError("forecast and target condition panels must align")
    values = []
    for forecast, target in zip(forecasts, targets, strict=True):
        if forecast.shape != target.shape or forecast.ndim != 1:
            raise ValueError("condition forecast and target must align")
        values.append(float(np.sqrt(np.mean(np.square(forecast - target)))))
    return float(np.mean(values))


def _evaluate_seed(
    seed: int,
    *,
    conditions: Sequence[str],
    benchmark_config: RecursiveCorruptionV2Config,
) -> dict[str, object]:
    domain = draw_seed_domain(seed, benchmark_config)
    candidate_forecasts: list[np.ndarray] = []
    fallback_forecasts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    candidate_harmful_update_count = 0
    candidate_accepted_update_count = 0
    candidate_exact_fallback_violations = 0
    for condition in conditions:
        sequence = generate_corrupted_sequence_v2(
            condition,
            domain=domain,
            config=benchmark_config,
        )
        methods = run_methods_v2(
            sequence,
            domain=domain,
            config=benchmark_config,
        )
        candidate = methods["guarded_recursive"]
        fallback = methods["physical_baseline"]
        candidate_forecasts.append(candidate.forecast_mean_m)
        fallback_forecasts.append(fallback.forecast_mean_m)
        targets.append(sequence.true_position_m[1:])
        candidate_harmful_update_count += int(
            np.sum(candidate.materially_harmful_update)
        )
        candidate_accepted_update_count += int(np.sum(candidate.accepted_update))
        candidate_exact_fallback_violations += int(
            np.sum(candidate.exact_fallback & ~candidate.exact_fallback_valid)
        )
    candidate_loss = _mean_condition_rmse(candidate_forecasts, targets)
    fallback_loss = _mean_condition_rmse(fallback_forecasts, targets)
    candidate_vector = np.ascontiguousarray(np.concatenate(candidate_forecasts))
    fallback_vector = np.ascontiguousarray(np.concatenate(fallback_forecasts))
    return {
        "candidate_loss_m": candidate_loss,
        "fallback_loss_m": fallback_loss,
        "candidate_forecast": candidate_vector,
        "fallback_forecast": fallback_vector,
        "candidate_harmful_update_count": candidate_harmful_update_count,
        "candidate_accepted_update_count": candidate_accepted_update_count,
        "candidate_exact_fallback_violation_count": (
            candidate_exact_fallback_violations
        ),
    }


def _wilson_interval(successes: int, trials: int) -> list[float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid Wilson count")
    proportion = successes / trials
    z2 = _Z975**2
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        _Z975
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _gain_null_calibration(
    config: AnytimeRecursiveAdmissionV1Config,
) -> dict[str, object]:
    rng = np.random.default_rng(config.null_seed)
    schedule = GeometricAlphaSpending(
        total_alpha=config.total_alpha_gain,
        continuation=config.epoch_alpha_continuation,
    )
    false_promotions = 0
    crossing_epoch_counts = [0] * config.null_epoch_count
    for _world in range(config.null_world_count):
        crossed = False
        for epoch in range(config.null_epoch_count):
            process = BoundedGainMixtureEProcess(
                config.admission_config.gain_bet_fractions
            )
            threshold = -math.log(schedule.alpha_for_epoch(epoch))
            scores = rng.choice(
                np.asarray((-1.0, 1.0), dtype=np.float64),
                size=config.null_trials_per_epoch,
            )
            for score in scores:
                if process.update(float(score)) >= threshold:
                    false_promotions += 1
                    crossing_epoch_counts[epoch] += 1
                    crossed = True
                    break
            if crossed:
                break
    fraction = false_promotions / config.null_world_count
    return {
        "null": "conditional-mean-bounded-score-at-zero",
        "world_count": config.null_world_count,
        "false_promotion_count": false_promotions,
        "false_promotion_fraction": fraction,
        "wilson_95_interval": _wilson_interval(
            false_promotions,
            config.null_world_count,
        ),
        "total_alpha": config.total_alpha_gain,
        "crossing_epoch_counts": crossing_epoch_counts,
        "empirical_below_total_alpha": fraction <= config.total_alpha_gain,
    }


def _harm_null_calibration(
    config: AnytimeRecursiveAdmissionV1Config,
) -> dict[str, object]:
    rng = np.random.default_rng(config.null_seed + 1)
    schedule = GeometricAlphaSpending(
        total_alpha=config.total_alpha_harm,
        continuation=config.epoch_alpha_continuation,
    )
    false_promotions = 0
    crossing_epoch_counts = [0] * config.null_epoch_count
    for _world in range(config.null_world_count):
        crossed = False
        for epoch in range(config.null_epoch_count):
            process = BernoulliHarmMixtureEProcess(
                maximum_harm_rate=config.maximum_harm_rate,
                alternative_fractions=(config.harm_alternative_fractions),
            )
            threshold = -math.log(schedule.alpha_for_epoch(epoch))
            harms = rng.random(config.null_trials_per_epoch) < config.maximum_harm_rate
            for harmful in harms:
                if process.update(bool(harmful)) >= threshold:
                    false_promotions += 1
                    crossing_epoch_counts[epoch] += 1
                    crossed = True
                    break
            if crossed:
                break
    fraction = false_promotions / config.null_world_count
    return {
        "null": "conditional-harm-probability-at-registered-ceiling",
        "world_count": config.null_world_count,
        "false_promotion_count": false_promotions,
        "false_promotion_fraction": fraction,
        "wilson_95_interval": _wilson_interval(
            false_promotions,
            config.null_world_count,
        ),
        "total_alpha": config.total_alpha_harm,
        "crossing_epoch_counts": crossing_epoch_counts,
        "empirical_below_total_alpha": fraction <= config.total_alpha_harm,
    }


def run_anytime_recursive_admission_v1(
    config: AnytimeRecursiveAdmissionV1Config,
) -> dict[str, object]:
    """Run the fresh delayed-outcome stream and direct null calibrations."""

    if not isinstance(config, AnytimeRecursiveAdmissionV1Config):
        raise TypeError("config must be AnytimeRecursiveAdmissionV1Config")
    delays = np.random.default_rng(config.delay_seed).integers(
        config.delay_min_episodes,
        config.delay_max_episodes + 1,
        size=config.seed_count,
    )
    controller = AnytimeAdmissionController(config.admission_config)
    benchmark_config = RecursiveCorruptionV2Config()
    pending_outcomes: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    resolution_records: list[dict[str, object]] = []
    first_authorized_issue_index: int | None = None

    def resolve_matured(current_step: int) -> None:
        matured = sorted(
            (
                (trial_id, outcome)
                for trial_id, outcome in pending_outcomes.items()
                if int(cast(Any, outcome["maturity_step"])) <= current_step
            ),
            key=lambda item: (
                int(cast(Any, item[1]["maturity_step"])),
                int(cast(Any, item[1]["issue_index"])),
            ),
        )
        for trial_id, outcome in matured:
            resolved = controller.resolve_trial(
                trial_id=trial_id,
                resolved_step=current_step,
                candidate_loss=float(cast(Any, outcome["candidate_loss_m"])),
                fallback_loss=float(cast(Any, outcome["fallback_loss_m"])),
            )
            resolution_records.append(
                {
                    **asdict(resolved),
                    "gain_log_e_value_after": controller.snapshot().gain_log_e_value,
                    "harm_log_e_value_after": controller.snapshot().harm_log_e_value,
                    "authorized_after": controller.authorized,
                }
            )
            del pending_outcomes[trial_id]

    for issue_index, (seed, delay) in enumerate(zip(config.seeds, delays, strict=True)):
        resolve_matured(issue_index)
        authorized_before = controller.authorized
        if authorized_before and first_authorized_issue_index is None:
            first_authorized_issue_index = issue_index
        trial_id = f"seed-{seed}"
        maturity_step = issue_index + int(delay)
        controller.issue_trial(
            trial_id=trial_id,
            issued_step=issue_index,
            maturity_step=maturity_step,
        )

        # Outcome generation occurs only after the shadow trial and its reveal
        # time have been registered above.
        evaluation = _evaluate_seed(
            seed,
            conditions=config.conditions,
            benchmark_config=benchmark_config,
        )
        candidate_loss = float(cast(Any, evaluation["candidate_loss_m"]))
        fallback_loss = float(cast(Any, evaluation["fallback_loss_m"]))
        candidate_forecast = cast(np.ndarray, evaluation["candidate_forecast"])
        fallback_forecast = cast(np.ndarray, evaluation["fallback_forecast"])
        selected_forecast = (
            candidate_forecast if authorized_before else fallback_forecast
        )
        exact_fallback_valid = True
        if not authorized_before:
            exact_fallback_valid = (
                selected_forecast.dtype == fallback_forecast.dtype
                and selected_forecast.shape == fallback_forecast.shape
                and selected_forecast.tobytes(order="C")
                == fallback_forecast.tobytes(order="C")
            )
        pending_outcomes[trial_id] = {
            "issue_index": issue_index,
            "maturity_step": maturity_step,
            "candidate_loss_m": candidate_loss,
            "fallback_loss_m": fallback_loss,
        }
        records.append(
            {
                "issue_index": issue_index,
                "seed": seed,
                "delay_episodes": int(delay),
                "maturity_step": maturity_step,
                "authorized_before_issue": authorized_before,
                "selected_method": (
                    "guarded_recursive" if authorized_before else "physical_baseline"
                ),
                "candidate_loss_m": candidate_loss,
                "fallback_loss_m": fallback_loss,
                "selected_loss_m": (
                    candidate_loss if authorized_before else fallback_loss
                ),
                "candidate_gain_m": fallback_loss - candidate_loss,
                "candidate_harmful_episode": candidate_loss > fallback_loss,
                "selected_harmful_episode": (
                    authorized_before and candidate_loss > fallback_loss
                ),
                "exact_fallback_valid": exact_fallback_valid,
                "candidate_harmful_update_count": evaluation[
                    "candidate_harmful_update_count"
                ],
                "candidate_accepted_update_count": evaluation[
                    "candidate_accepted_update_count"
                ],
                "candidate_exact_fallback_violation_count": evaluation[
                    "candidate_exact_fallback_violation_count"
                ],
                "resolved_evidence_count_before_issue": controller.snapshot().resolved_current_epoch_count,
                "gain_log_e_value_before_issue": controller.snapshot().gain_log_e_value,
                "harm_log_e_value_before_issue": controller.snapshot().harm_log_e_value,
            }
        )

    terminal_step = config.seed_count
    while pending_outcomes:
        resolve_matured(terminal_step)
        terminal_step += 1
        if terminal_step > config.seed_count + config.delay_max_episodes + 1:
            raise RuntimeError("pending outcomes did not mature within delay bound")

    candidate_losses = np.asarray(
        [float(record["candidate_loss_m"]) for record in records],
        dtype=np.float64,
    )
    fallback_losses = np.asarray(
        [float(record["fallback_loss_m"]) for record in records],
        dtype=np.float64,
    )
    selected_losses = np.asarray(
        [float(record["selected_loss_m"]) for record in records],
        dtype=np.float64,
    )
    candidate_harmful = candidate_losses > fallback_losses
    selected_harmful = np.asarray(
        [bool(record["selected_harmful_episode"]) for record in records],
        dtype=bool,
    )
    authorized_mask = np.asarray(
        [bool(record["authorized_before_issue"]) for record in records],
        dtype=bool,
    )
    exact_fallback_violations = sum(
        not bool(record["exact_fallback_valid"])
        for record in records
        if not bool(record["authorized_before_issue"])
    )
    terminal_snapshot = controller.snapshot()
    gain_calibration = _gain_null_calibration(config)
    harm_calibration = _harm_null_calibration(config)
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": asdict(config),
        "statistical_contract": {
            "gain_null": (
                "conditional expected registered capped loss improvement is at "
                "most minimum_mean_gain_m"
            ),
            "harm_null": (
                "conditional probability of a materially harmful episode is at "
                "least maximum_harm_rate"
            ),
            "authorization": (
                "both current epoch e-processes cross after the minimum evidence count"
            ),
            "epoch_familywise_control": (
                "geometric alpha spending; theorem conditional on predictable "
                "registration and the stated null assumptions"
            ),
        },
        "fresh_stream": {
            "seed_count": config.seed_count,
            "seed_start": config.seed_start,
            "seed_stop_exclusive": config.seed_start + config.seed_count,
            "candidate_mean_loss_m": float(np.mean(candidate_losses)),
            "fallback_mean_loss_m": float(np.mean(fallback_losses)),
            "selected_mean_loss_m": float(np.mean(selected_losses)),
            "candidate_relative_improvement_over_fallback": float(
                1.0 - np.mean(candidate_losses) / np.mean(fallback_losses)
            ),
            "selected_relative_improvement_over_fallback": float(
                1.0 - np.mean(selected_losses) / np.mean(fallback_losses)
            ),
            "selected_regret_to_always_candidate_m": float(
                np.mean(selected_losses) - np.mean(candidate_losses)
            ),
            "candidate_wins": int(np.sum(candidate_losses < fallback_losses)),
            "candidate_ties": int(np.sum(candidate_losses == fallback_losses)),
            "candidate_loss_count": int(np.sum(candidate_losses > fallback_losses)),
            "candidate_harmful_episode_count": int(np.sum(candidate_harmful)),
            "selected_harmful_episode_count": int(np.sum(selected_harmful)),
            "authorized_deployment_count": int(np.sum(authorized_mask)),
            "fallback_deployment_count": int(np.sum(~authorized_mask)),
            "first_authorized_issue_index": first_authorized_issue_index,
            "exact_fallback_violation_count": exact_fallback_violations,
            "candidate_internal_exact_fallback_violation_count": int(
                sum(
                    int(cast(Any, record["candidate_exact_fallback_violation_count"]))
                    for record in records
                )
            ),
            "terminal_evidence": terminal_snapshot.as_dict(),
        },
        "gain_null_calibration": gain_calibration,
        "harm_null_calibration": harm_calibration,
        "records": records,
        "resolution_records": resolution_records,
        "decision": {
            "fresh_stream_authorized_at_least_once": any(
                bool(record["authorized_before_issue"]) for record in records
            ),
            "selected_stream_improves_fallback": float(np.mean(selected_losses))
            < float(np.mean(fallback_losses)),
            "candidate_improves_fallback": float(np.mean(candidate_losses))
            < float(np.mean(fallback_losses)),
            "exact_fallback_preserved": exact_fallback_violations == 0,
            "gain_null_empirical_check_passed": bool(
                cast(
                    Mapping[str, object],
                    gain_calibration,
                )["empirical_below_total_alpha"]
            ),
            "harm_null_empirical_check_passed": bool(
                cast(
                    Mapping[str, object],
                    harm_calibration,
                )["empirical_below_total_alpha"]
            ),
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "Fresh controlled synthetic evidence for an anytime-valid admission "
            "mechanism under the registered capped loss, harm event, candidate, "
            "fallback, delay process, and conditional null assumptions. It is not "
            "real-object validation, a universal model-improvement guarantee, "
            "probabilistic calibration of the physical state, deployment safety, "
            "or state of the art."
        ),
    }


def canonical_result_digest(result: Mapping[str, object]) -> str:
    """Return the SHA-256 of the canonical JSON result payload."""

    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
