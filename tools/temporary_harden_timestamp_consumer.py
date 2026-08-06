"""One-shot exact patch for PR #187; deleted by its publisher workflow."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one replacement target in {path}, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(
    path: str,
    *,
    start: str,
    end: str,
    replacement: str,
) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise SystemExit(
            f"replacement markers in {path} are not unique: "
            f"start={start_count}, end={end_count}"
        )
    first = text.index(start)
    last = text.index(end, first + len(start))
    target.write_text(text[:first] + replacement + text[last:], encoding="utf-8")


replace_once(
    "src/bayesian_phystwin/prob4d_observation_timestamps.py",
    "from .observation_belief import ObservationBeliefV1\n"
    "from .observation_timing_interchange import "
    "observation_timing_prior_from_payload\n"
    "from .observation_timing_nuisance import (\n",
    "from .causal4d_observation_clock_prior import (\n"
    "    causal4d_observation_timing_prior_from_record,\n"
    ")\n"
    "from .observation_belief import ObservationBeliefV1\n"
    "from .observation_timing_nuisance import (\n",
)

replace_between(
    "src/bayesian_phystwin/prob4d_observation_timestamps.py",
    start="    def shared_clock_prior_from_payload(\n",
    end="\n\n\ndef load_prob4d_observation_timestamp_binding(\n",
    replacement=(
        "    def shared_clock_prior_from_causal4d_record(\n"
        "        self,\n"
        "        value: Mapping[str, Any],\n"
        "    ) -> ObservationTimingPrior:\n"
        "        \"\"\"Independently reconstruct and bind a complete Causal4D prior.\"\"\"\n"
        "\n"
        "        expected_id = self.shared_clock_offset_prior_artifact_id\n"
        "        if expected_id is None:\n"
        "            raise ValueError(\"timestamp lineage declares no shared clock prior\")\n"
        "        return causal4d_observation_timing_prior_from_record(\n"
        "            value,\n"
        "            expected_artifact_id=expected_id,\n"
        "            expected_clock_domain=self.clock_domain,\n"
        "            expected_time_scale=self.time_scale,\n"
        "        )\n"
        "\n"
        "    def shared_clock_prior_from_payload(\n"
        "        self,\n"
        "        value: Mapping[str, object],\n"
        "    ) -> ObservationTimingPrior:\n"
        "        \"\"\"Reject compact Gaussian payloads at the claim-bearing boundary.\"\"\"\n"
        "\n"
        "        if self.shared_clock_offset_prior_artifact_id is None:\n"
        "            raise ValueError(\"timestamp lineage declares no shared clock prior\")\n"
        "        del value\n"
        "        raise ValueError(\n"
        "            \"compact shared clock payload is not independently verifiable; \"\n"
        "            \"provide the full Causal4D clock-prior record\"\n"
        "        )\n"
    ),
)

replace_between(
    "tests/test_prob4d_observation_timestamps.py",
    start="def test_shared_clock_prior_binds_artifact_domain_and_sign(\n",
    end="\n\n\ndef test_source_order_frame_and_checksum_mismatches_fail_closed(\n",
    replacement=(
        "def test_compact_shared_clock_payload_fails_closed(tmp_path: Path) -> None:\n"
        "    binding = _binding(tmp_path)\n"
        "    payload: dict[str, object] = {\n"
        "        \"clock_domain\": \"camera-hardware-clock\",\n"
        "        \"mean_offset_s\": 0.001,\n"
        "        \"standard_deviation_s\": 0.0005,\n"
        "        \"source_artifact_id\": SHARED_PRIOR,\n"
        "        \"offset_convention\": (\n"
        "            \"aligned_observation_time_s = observation_time_s + offset_s\"\n"
        "        ),\n"
        "    }\n"
        "\n"
        "    with pytest.raises(ValueError, match=\"not independently verifiable\"):\n"
        "        binding.shared_clock_prior_from_payload(payload)\n"
    ),
)

replace_between(
    "docs/prob4d_observation_timestamps.md",
    start="## Shared clock design and prior\n",
    end="\n## Information-order boundary\n",
    replacement=(
        "## Shared clock design and prior\n"
        "\n"
        "```python\n"
        "clock_design = binding.shared_clock_design(\n"
        "    observation_derivative_xyz_per_s\n"
        ")\n"
        "prior = binding.shared_clock_prior_from_causal4d_record(\n"
        "    source_only_prior_record\n"
        ")\n"
        "```\n"
        "\n"
        "The design has shape `(3N, 1)` and uses the same coordinate flattening as the\n"
        "physical and nuisance Jacobians. Claim-bearing consumption requires the complete\n"
        "content-addressed Causal4D prior record, not only its compact Gaussian payload.\n"
        "BayesianPhysTwin independently checks the closed schema, source-only information\n"
        "boundary, source execution count and ordering, source offsets, predictive-width\n"
        "formula and floor, exact content ID, clock domain, time scale, and correction\n"
        "convention:\n"
        "\n"
        "```text\n"
        "aligned_observation_time_s = observation_time_s + offset_s\n"
        "```\n"
        "\n"
        "A compact payload containing an artifact ID, mean, and standard deviation is\n"
        "deliberately rejected: those numeric values could otherwise be altered while\n"
        "retaining the expected ID. The resulting `ObservationTimingPrior` can be passed\n"
        "to the explicit timing nuisance machinery only after the full record has been\n"
        "reconstructed successfully.\n"
        "\n"
        "Timing identifiability must still be assessed against physical-state, gauge,\n"
        "visual-bias, and material-lag modes. A source timestamp sidecar and a valid\n"
        "source-only prior do not by themselves distinguish hardware clock error from\n"
        "physical relaxation.\n"
    ),
)
