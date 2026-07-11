import numpy as np

from causal4d.baselines import fit_baselines
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    build_protocol,
    generate_episodes,
    make_parameter_grid,
)
from causal4d.contact_inference import (
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
    true_contact_state,
    true_parameter_predictive_for_state,
)


def _configs() -> tuple[CounterfactualBenchmarkConfig, LatentContactConfig]:
    benchmark = CounterfactualBenchmarkConfig(
        frame_count=20,
        training_repeats=1,
        parameter_grid_count=3,
        fit_frame_stride=3,
    )
    contact = LatentContactConfig(
        parameter_particle_count=3,
        observation_fraction=0.20,
    )
    return benchmark, contact


def test_action_conditioned_prior_excludes_target_and_spans_shifted_graph_node() -> (
    None
):
    benchmark, contact = _configs()
    protocols = build_protocol(benchmark)
    target = protocols[0]
    sources = protocols[1:]
    prior = fit_contact_prior(sources, contact, action_split="test")
    model = GraphContactHypothesisModel(prior=prior, config=contact)
    states, weights = model.hypotheses(target.graph_object, target.test_action)
    shifted_truth = true_contact_state(
        target.graph_object,
        target.test_action,
        target.test_conditions[1],
    )

    assert target.graph_object.name not in prior.source_objects
    assert prior.source_action_split == "test"
    assert np.isclose(prior.shift_probability, 0.5)
    assert shifted_truth.contact_nodes in {state.contact_nodes for state in states}
    assert shifted_truth.delay_steps in {state.delay_steps for state in states}
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0.0)


def test_online_update_uses_prefix_only_and_returns_joint_mixture_intervals() -> None:
    benchmark, contact = _configs()
    protocols = build_protocol(benchmark)
    target = protocols[0]
    training, validation, held_out = generate_episodes(target, benchmark, seed=4)
    baselines = fit_baselines(
        training,
        validation,
        make_parameter_grid(target.graph_object, benchmark),
        benchmark,
    )
    prior = fit_contact_prior(protocols[1:], contact, action_split="test")
    model = GraphContactHypothesisModel(prior=prior, config=contact)
    bank = build_rollout_bank(
        target.graph_object,
        target.test_action,
        baselines.physics.posterior,
        model,
        simulator_config=benchmark.simulator,
        parameter_particle_count=contact.parameter_particle_count,
        variance_floor_m2=benchmark.predictive_variance_floor_m2,
        confidence_level=contact.confidence_level,
    )
    prefix = contact.prefix_frame_count(benchmark.frame_count)
    observations = held_out[1].truth.copy()
    changed_future = observations.copy()
    changed_future[prefix:] += 10.0

    first = bank.update_weights(
        observations,
        prefix_frame_count=prefix,
        likelihood_scale_m=0.002,
        likelihood_power=1.0,
        dynamic_likelihood_weight=1.0,
    )
    second = bank.update_weights(
        changed_future,
        prefix_frame_count=prefix,
        likelihood_scale_m=0.002,
        likelihood_power=1.0,
        dynamic_likelihood_weight=1.0,
    )
    prediction = bank.predictive_distribution(
        first,
        method="latent_contact",
    )

    assert np.array_equal(first, second)
    assert np.isclose(np.sum(first), 1.0)
    assert bank.contact_marginal(first).shape == (len(bank.contact_states),)
    assert bank.parameter_marginal(first).shape == bank.parameter_weights.shape
    assert prediction.interval_lower is not None
    assert prediction.interval_upper is not None
    assert prediction.interval_lower.shape == held_out[1].truth.shape
    assert np.all(prediction.interval_lower <= prediction.interval_upper)


def test_true_contact_and_parameter_control_is_a_strict_simulation_ceiling() -> None:
    benchmark, contact = _configs()
    protocol = build_protocol(benchmark)[2]
    training, validation, held_out = generate_episodes(protocol, benchmark, seed=2)
    baselines = fit_baselines(
        training,
        validation,
        make_parameter_grid(protocol.graph_object, benchmark),
        benchmark,
    )
    episode = held_out[1]
    nominal = baselines.physics.predict(episode)
    oracle = true_parameter_predictive_for_state(
        protocol.graph_object,
        episode.action,
        true_contact_state(protocol.graph_object, episode.action, episode.condition),
        simulator_config=benchmark.simulator,
        variance_floor_m2=benchmark.predictive_variance_floor_m2,
    )

    nominal_rmse = np.sqrt(np.mean(np.square(nominal.mean - episode.truth)))
    oracle_rmse = np.sqrt(np.mean(np.square(oracle.mean - episode.truth)))
    assert oracle.method == "oracle_contact_theta"
    assert oracle_rmse < nominal_rmse
