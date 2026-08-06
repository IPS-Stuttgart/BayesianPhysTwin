"""Temporary exact patch for PR #183; removed by its one-shot workflow."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

SOURCE_PATH = Path("src/bayesian_phystwin/persistent_prob4d_visual_bias.py")
TEST_PATH = Path("tests/test_persistent_prob4d_visual_bias.py")


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {name} target, found {count}")
    return text.replace(old, new, 1)


def patch_source(source: str) -> str:
    joint_mean = dedent(
        """\
        def _joint_mean(belief: PersistentVisualBiasBeliefV1) -> np.ndarray:
            return np.concatenate((belief.physical_mean, belief.bias_latent_mean))


        """
    )
    helpers = joint_mean + dedent(
        """\
        def _require_compatible_posterior(
            prior: PersistentVisualBiasBeliefV1,
            posterior: PersistentVisualBiasBeliefV1,
        ) -> None:
            if posterior.stream_binding_id != prior.stream_binding_id:
                raise ValueError("candidate posterior uses a different stream binding")
            if posterior.visual_bias_model_id != prior.visual_bias_model_id:
                raise ValueError("candidate posterior uses a different visual-bias model")
            if posterior.physical_state_domain_id != prior.physical_state_domain_id:
                raise ValueError(
                    "candidate posterior uses a different physical state domain"
                )
            if posterior.physical_dimension != prior.physical_dimension:
                raise ValueError("candidate posterior physical dimension differs")
            if posterior.bias_dimension != prior.bias_dimension:
                raise ValueError("candidate posterior bias dimension differs")
            if not np.array_equal(
                posterior.bias_covariance_root,
                prior.bias_covariance_root,
            ):
                raise ValueError(
                    "candidate posterior uses a different visual-bias covariance root"
                )


        def _require_measurement_covariance_contraction(
            prior_covariance: np.ndarray,
            posterior_covariance: np.ndarray,
        ) -> None:
            reduction = prior_covariance - posterior_covariance
            reduction = 0.5 * (reduction + reduction.T)
            eigenvalues = np.linalg.eigvalsh(reduction)
            scale = max(
                1.0,
                float(np.linalg.norm(prior_covariance, ord=2)),
                float(np.linalg.norm(posterior_covariance, ord=2)),
            )
            if float(np.min(eigenvalues)) < -1e-10 * scale:
                raise ValueError(
                    "candidate posterior covariance is not a measurement contraction"
                )


        """
    )
    source = replace_once(
        source,
        joint_mean,
        helpers,
        name="posterior helper insertion",
    )

    candidate_target = (
        "        if self.posterior_belief.stream_binding_id != self.stream_binding_id:\n"
        "            raise ValueError(\"candidate posterior uses a different stream binding\")\n"
        "        quadratic = _finite_real(\n"
    )
    candidate_replacement = (
        "        if self.posterior_belief.stream_binding_id != self.stream_binding_id:\n"
        "            raise ValueError(\"candidate posterior uses a different stream binding\")\n"
        "        posterior_lineage = self.posterior_belief.metadata\n"
        "        if posterior_lineage.get(\"source_update_index\") != index:\n"
        "            raise ValueError(\"candidate posterior source update index differs\")\n"
        "        if (\n"
        "            posterior_lineage.get(\"physical_linearization_id\")\n"
        "            != self.physical_linearization_id\n"
        "        ):\n"
        "            raise ValueError(\"candidate posterior physical linearization differs\")\n"
        "        quadratic = _finite_real(\n"
    )
    source = replace_once(
        source,
        candidate_target,
        candidate_replacement,
        name="candidate lineage validation",
    )

    selection_target = (
        "    accept = genuine_boolean(accepted, name=\"accepted\")\n"
        "    if candidate.stream_binding_id != run.stream_binding.binding_id:\n"
        "        raise ValueError(\"candidate uses a different stream binding\")\n"
        "    if candidate.update_index != run.next_update_index:\n"
        "        raise ValueError(\"candidate update index is stale or out of order\")\n"
        "    if candidate.prior_belief_id != run.belief.belief_id:\n"
        "        raise ValueError(\"candidate prior belief is stale\")\n"
        "    update = run.stream_binding.visual_bias_stream.updates[run.next_update_index]\n"
        "    if candidate.visual_bias_stream_update_id != update.update_id:\n"
        "        raise ValueError(\"candidate identifies a different visual-bias update\")\n"
        "    selected = candidate.posterior_belief if accept else run.belief\n"
    )
    selection_replacement = (
        "    accept = genuine_boolean(accepted, name=\"accepted\")\n"
        "    binding = run.stream_binding\n"
        "    if candidate.stream_binding_id != binding.binding_id:\n"
        "        raise ValueError(\"candidate uses a different stream binding\")\n"
        "    index = run.next_update_index\n"
        "    if candidate.update_index != index:\n"
        "        raise ValueError(\"candidate update index is stale or out of order\")\n"
        "    if candidate.prior_belief_id != run.belief.belief_id:\n"
        "        raise ValueError(\"candidate prior belief is stale\")\n"
        "    update = binding.visual_bias_stream.updates[index]\n"
        "    if candidate.visual_bias_stream_update_id != update.update_id:\n"
        "        raise ValueError(\"candidate identifies a different visual-bias update\")\n"
        "    if candidate.factor_stream_update_id != binding.factor_stream_update_ids[index]:\n"
        "        raise ValueError(\n"
        "            \"candidate factor-stream update differs from the active stream member\"\n"
        "        )\n"
        "    if candidate.observation_binding_id != binding.observation_binding_ids[index]:\n"
        "        raise ValueError(\n"
        "            \"candidate observation binding differs from the active stream member\"\n"
        "        )\n"
        "    posterior = candidate.posterior_belief\n"
        "    _require_compatible_posterior(run.belief, posterior)\n"
        "    _require_measurement_covariance_contraction(\n"
        "        run.belief.joint_covariance,\n"
        "        posterior.joint_covariance,\n"
        "    )\n"
        "    expected_information_gain = _information_gain_nats(\n"
        "        run.belief.joint_covariance,\n"
        "        posterior.joint_covariance,\n"
        "    )\n"
        "    gain_scale = max(1.0, abs(expected_information_gain))\n"
        "    if not np.isclose(\n"
        "        candidate.information_gain_nats,\n"
        "        expected_information_gain,\n"
        "        rtol=1e-10,\n"
        "        atol=1e-12 * gain_scale,\n"
        "    ):\n"
        "        raise ValueError(\n"
        "            \"candidate information gain does not match posterior covariance\"\n"
        "        )\n"
        "    selected = posterior if accept else run.belief\n"
    )
    return replace_once(
        source,
        selection_target,
        selection_replacement,
        name="selection-boundary hardening",
    )


def patch_tests(tests: str) -> str:
    marker = "def test_selection_rejects_forged_candidate_bindings("
    if marker in tests:
        raise RuntimeError("persistent hardening tests already exist")
    return tests + dedent(
        """\


        def _single_candidate():
            binding = _binding((1,))
            run = start_persistent_visual_bias_run(
                binding,
                physical_state_domain_id="physical-state-v1",
                physical_mean=np.zeros(1, dtype=np.float64),
                physical_covariance=np.eye(1, dtype=np.float64),
            )
            measurement, jacobian, covariance = _measurement_arrays(1)
            candidate = propose_persistent_visual_bias_update(
                run,
                innovation_xyz=measurement,
                physical_jacobian=jacobian,
                conditional_covariance=covariance,
                physical_linearization_id=_sha("1"),
            )
            return run, candidate


        @pytest.mark.parametrize(
            ("field", "message"),
            [
                ("factor_stream_update_id", "factor-stream update differs"),
                ("observation_binding_id", "observation binding differs"),
            ],
        )
        def test_selection_rejects_forged_candidate_bindings(
            field: str,
            message: str,
        ) -> None:
            run, candidate = _single_candidate()
            forged = replace(candidate, **{field: _sha("f")}, candidate_id=None)
            with pytest.raises(ValueError, match=message):
                select_persistent_visual_bias_candidate(
                    run,
                    forged,
                    accepted=False,
                    reason="reject-forged-binding",
                )


        def test_candidate_rejects_forged_posterior_lineage() -> None:
            _, candidate = _single_candidate()
            lineage = dict(candidate.posterior_belief.metadata)
            lineage["source_update_index"] = 7
            posterior = replace(
                candidate.posterior_belief,
                metadata=lineage,
                belief_id=None,
            )
            with pytest.raises(ValueError, match="source update index"):
                replace(candidate, posterior_belief=posterior, candidate_id=None)

            lineage = dict(candidate.posterior_belief.metadata)
            lineage["physical_linearization_id"] = _sha("f")
            posterior = replace(
                candidate.posterior_belief,
                metadata=lineage,
                belief_id=None,
            )
            with pytest.raises(ValueError, match="physical linearization"):
                replace(candidate, posterior_belief=posterior, candidate_id=None)


        def test_selection_rejects_incompatible_posterior_contract() -> None:
            run, candidate = _single_candidate()
            wrong_domain = replace(
                candidate.posterior_belief,
                physical_state_domain_id="different-state-domain",
                belief_id=None,
            )
            with pytest.raises(ValueError, match="physical state domain"):
                select_persistent_visual_bias_candidate(
                    run,
                    replace(
                        candidate,
                        posterior_belief=wrong_domain,
                        candidate_id=None,
                    ),
                    accepted=False,
                    reason="reject-domain-mismatch",
                )

            wrong_root = replace(
                candidate.posterior_belief,
                bias_covariance_root=np.zeros_like(
                    candidate.posterior_belief.bias_covariance_root
                ),
                belief_id=None,
            )
            with pytest.raises(ValueError, match="covariance root"):
                select_persistent_visual_bias_candidate(
                    run,
                    replace(
                        candidate,
                        posterior_belief=wrong_root,
                        candidate_id=None,
                    ),
                    accepted=False,
                    reason="reject-root-mismatch",
                )


        def test_selection_rejects_noncontracting_or_misreported_candidate() -> None:
            run, candidate = _single_candidate()
            expanded = replace(
                candidate.posterior_belief,
                joint_covariance=2.0 * np.asarray(run.belief.joint_covariance),
                belief_id=None,
            )
            with pytest.raises(ValueError, match="measurement contraction"):
                select_persistent_visual_bias_candidate(
                    run,
                    replace(
                        candidate,
                        posterior_belief=expanded,
                        information_gain_nats=0.0,
                        candidate_id=None,
                    ),
                    accepted=False,
                    reason="reject-expanded-covariance",
                )

            with pytest.raises(ValueError, match="information gain"):
                select_persistent_visual_bias_candidate(
                    run,
                    replace(
                        candidate,
                        information_gain_nats=candidate.information_gain_nats + 1.0,
                        candidate_id=None,
                    ),
                    accepted=False,
                    reason="reject-misreported-gain",
                )


        def test_persistent_belief_views_are_irreversibly_immutable() -> None:
            run, _ = _single_candidate()
            for array in (
                run.belief.physical_mean,
                run.belief.bias_latent_mean,
                run.belief.joint_covariance,
                run.belief.bias_covariance_root,
                run.belief.physical_covariance,
                run.belief.physical_bias_cross_covariance,
                run.belief.bias_latent_covariance,
                run.belief.provider_bias_mean,
                run.belief.provider_bias_covariance,
            ):
                assert not array.flags.writeable
                with pytest.raises(ValueError):
                    array.setflags(write=True)
        """
    )


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tests = TEST_PATH.read_text(encoding="utf-8")
    SOURCE_PATH.write_text(patch_source(source), encoding="utf-8")
    TEST_PATH.write_text(patch_tests(tests), encoding="utf-8")


if __name__ == "__main__":
    main()
