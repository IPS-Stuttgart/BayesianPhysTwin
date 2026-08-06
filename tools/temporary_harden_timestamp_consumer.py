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
    '        """Bind a source-only prior without merging it into local jitter."""\n',
    '        """Construct an exploratory compact prior; not claim-bearing."""\n',
)

replace_between(
    "docs/prob4d_observation_timestamps.md",
    start="## Shared clock design and prior\n",
    end="\n## Information-order boundary\n",
    replacement=(
        "## Shared clock design and prior\n"
        "\n"
        "```python\n"
        "from bayesian_phystwin.causal4d_observation_clock_binding import (\n"
        "    bind_causal4d_observation_clock_prior,\n"
        ")\n"
        "\n"
        "clock_design = binding.shared_clock_design(\n"
        "    observation_derivative_xyz_per_s\n"
        ")\n"
        "prior = bind_causal4d_observation_clock_prior(\n"
        "    binding,\n"
        "    source_only_prior_record,\n"
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
        "A compact payload containing an artifact ID, mean, and standard deviation cannot\n"
        "tie those numeric values to that ID. The legacy\n"
        "`binding.shared_clock_prior_from_payload(...)` helper is therefore exploratory\n"
        "only and must not authorize a claim-bearing run. The dedicated binder rejects\n"
        "compact payloads and returns an `ObservationTimingPrior` only after the full\n"
        "record has been reconstructed successfully.\n"
        "\n"
        "Timing identifiability must still be assessed against physical-state, gauge,\n"
        "visual-bias, and material-lag modes. A source timestamp sidecar and a valid\n"
        "source-only prior do not by themselves distinguish hardware clock error from\n"
        "physical relaxation.\n"
    ),
)
