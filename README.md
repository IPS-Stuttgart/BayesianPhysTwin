# Bayesian PhysTwin

Reliability-aware Bayesian belief estimation for PhysTwin-style deformable
digital twins.

Bayesian-PhysTwin treats learned tracks, depth points, masks, flow, and related
4-D perception outputs as uncertain pseudo-measurements. It combines them with
a PhysTwin physical prior while keeping reliability, structured covariance,
physical-parameter uncertainty, and simulator discrepancy explicit. When an
update is not identifiable or fails a prospective guard, the library returns
the exact caller-owned physical baseline instead of silently applying an unsafe
correction.

## Five-minute start

From a checked-out source tree:

```bash
python3 -m pip install -e ".[dev,graph]"
python examples/ecosystem_minimal_v1.py \
  --output-dir outputs/ecosystem-minimal-v1
bpt commands list
```

The example verifies the portable Prob4D-compatible observation contract,
exercises accepted and exact-fallback routes, and records a Causal4D provider
manifest. It is a deterministic software demonstration, not physical evidence.

New integrations should keep candidate construction, guard choice, and
complete-belief routing separate:

```python
from bayesian_phystwin.inference.v2 import InferenceSession

session = InferenceSession(
    session_id=frozen_protocol_id,
    candidate_factory=build_candidate,
    guard_policy=choose_guard,
    metadata={"protocol_id": protocol_id},
)
result = session.assimilate(
    prior=baseline_belief,
    observation=observation,
    context={"case_id": case_id},
)
assert result.selected_belief is (
    baseline_belief if result.exact_fallback else result.candidate_belief
)
```

The session delegates candidate construction and guard choice to caller-owned
policies. It does not choose a provider or establish covariance calibration.
Strict claim-bearing Prob4D candidate construction remains available through
`bayesian_phystwin.inference.v1`. See the
[provider-neutral inference guide](docs/inference_v2.md), the
[strict Prob4D inference guide](docs/inference_v1.md), and the
[minimal ecosystem guide](docs/ecosystem_minimal_v1.md) for the exact contracts
and scientific boundaries.

## Evidence at a glance

<!-- public-claim-status:begin -->
| Question | Current status | Boundary |
| --- | --- | --- |
| Better than released PhysTwin under the frozen full-22 point contract? | **Confirmed, bounded** | The Bayesian anchor improves both primary point metrics, but this does not establish overall state of the art. |
| Unique deterministic winner over the matched last-residual comparator? | **Not confirmed** | The methods are essentially tied in Chamfer distance, and last-residual is marginally better in track error. |
| Raw posterior covariance calibrated? | **No** | Operational 3-D NEES is extremely high and nominal-90% ellipsoid coverage is far below nominal. |
| Retrospective covariance-only proper-score value established? | **Yes, with width cost** | The exact last-residual mean is preserved; Gaussian NLL and marginal coverage improve, while mean full interval width grows by 3.10x. |
| Fresh independent covariance-only confirmation established? | **Not opened** | The separate route still requires 100 sealed source prediction records and source-positive authorization; twelve disjoint confirmation object-sessions remain closed. |
| Fresh-object-session v6/v6.1 transfer established? | **Terminal, no claim** | A retained endpoint-processing technical failure occurred before the source gate was evaluated; replacement, retry, and source continuation are forbidden, and no fresh-target payload was opened. |
| Real Prob4D feeder transfer established? | **Not established** | Portable contracts and synthetic integration tests do not substitute for a real provider-value experiment. |
| Downstream Causal4D counterfactual benefit established? | **Not established** | Provider compatibility is implemented, but a registered downstream causal-value result is still required. |
<!-- public-claim-status:end -->

This table is generated from
[`evidence/public_claim_snapshot_v1.json`](evidence/public_claim_snapshot_v1.json),
which pins the release-facing claim contract by Git blob identity. Regenerate it
with `python scripts/render_public_claim_status.py --write`; CI checks that the
snapshot, source document, and README stay synchronized.

## Architecture

```text
Prob4D or another 4-D perception feeder
                │
                ▼
       ObservationBeliefV1 ───────────────┐
                                          │
       PhysTwin physical prior ───────────┼──► robust likelihood
                                          │    + guarded Bayesian update
                                          │
                                          └──► predictive belief
                                               or exact fallback
                                                        │
                                                        ▼
                                           Causal4D provider artifacts
```

[Prob4D](https://github.com/IPS-Stuttgart/Prob4D) can export the portable
`ObservationBeliefV1` contract. Bayesian-PhysTwin owns the reliability-aware
belief update and PhysTwin provider boundary.
[Causal4D](https://github.com/IPS-Stuttgart/Causal4D) separately owns abduction,
intervention, and counterfactual prediction.

Compatibility tests and synthetic examples do not authorize real-provider,
calibration, fresh-transfer, or downstream-causal claims. The complete wording
is maintained in the [release-facing claim contract](docs/phystwin_release_claim_v1.md).

## Development installation

```bash
python3 -m pip install -e ".[dev,data,graph]"
bash scripts/local_smoke_test.sh
bpt --help
```

The base package requires only NumPy. Optional groups add development tools,
data retrieval, sparse graph routines, vision utilities, or PyRecEst.

## Stable command surface

The package installs exactly one executable: `bpt`.

<!-- bpt-stable-commands:begin -->
| Command | Purpose | Documentation |
| --- | --- | --- |
| `bpt provider manifest` | Print the Causal4D provider capability manifest. | [Guide](docs/causal4d_provider_v1.md) |
| `bpt observation validate` | Validate or summarize an ObservationBeliefV1 artifact. | [Guide](docs/observation_belief_contract.md) |
| `bpt residual replay` | Replay exported residuals through the robust likelihood. | [Guide](docs/residual_replay.md) |
| `bpt benchmark synthetic` | Run the controlled synthetic benchmark. | [Guide](docs/synthetic_benchmark.md) |
| `bpt benchmark recursive-corruption` | Run the controlled recursive corruption benchmark. | [Guide](docs/recursive_corruption_benchmark.md) |
| `bpt evidence summarize` | Summarize matched guarded prospective evidence. | [Guide](docs/decisive_evidence_protocol.md) |
| `bpt evidence bundle` | Build or validate a content-addressed claim bundle. | [Guide](docs/claim_bundle_v1.md) |
| `bpt run manifest` | Create or validate a content-addressed run manifest. | [Guide](docs/reproducible_runs.md) |
<!-- bpt-stable-commands:end -->

Research functionality is organized under `bpt experiment`, `bpt diagnostic`,
and `bpt archive`. Use `bpt commands list --json` for the complete
machine-readable registry and see the [CLI guide](docs/command_line.md) for
lifecycle definitions.

## Python API and scientific boundaries

`bayesian_phystwin.v1` owns portable observations, physical queries, run
manifests, evidence decisions, and claim bundles. Loading an observation belief
revalidates its schema, content address, covariance, identities, and exclusive
causal cutoff:

```python
from bayesian_phystwin.v1 import load_observation_belief

belief = load_observation_belief("observation_belief.npz")
print(belief.summary())
```

For strict Prob4D candidate construction, use the frozen v1 inference namespace:

```python
from bayesian_phystwin.inference.v1 import (
    finalize_guarded_update,
    infer_prob4d_candidate,
)
```

`bayesian_phystwin.inference.v2` owns provider-neutral candidate construction,
caller-owned guard choice, and exact complete-belief routing.
`bayesian_phystwin.inference.v1` retains the strict Prob4D-specific candidate
surface and its frozen 0.4 compatibility contract. The historical package-root
namespace is a compatibility surface rather than the destination for new
integrations.

A predictive readout-discrepancy belief is not automatically a corrected latent
physical state. Released trajectories do not identify a unique physical cause.
Experiments and papers must preserve that distinction.

Backend registration likewise demonstrates interface compatibility, not native
physical evidence. The [Evidence-first backend admission
policy](docs/backend_admission_policy_v1.md) freezes new backend-family admission
until a selected backend passes source-physics and source-value qualification.
Existing labels such as `preferred`, `supported`, and `experimental` are not
evidence stages.

The installed [five-backend support matrix](docs/backend_support_matrix_v1.md)
tracks integration and predictive evidence independently:

| Backend | Integration | Retained scientific decision |
| --- | --- | --- |
| DEFORM DLO v7 | fully supported | benchmark-qualified on released DLO2 only |
| MatPhys/Warp | fully supported | source covariance value rejected |
| JAX-FEM v2 | fully supported | source physics passed; source value physically rejected |
| MuJoCo Flex | fully supported | native smoke passed; source physics rejected |
| SOFA FEM v3 | fully supported | source physics passed; source value physically rejected |

Only DEFORM is currently recommendation-authorized, and only within its exact
DLO2 benchmark contract. `fully supported` means executable, tested,
documented, evidence-retaining, and protected by exact fallback; it does not
mean that a backend won its predictive gate.

Specialized predictive backends are tracked separately from that generic
material-solver registry. In particular, the public DEFORM DLO pipeline is a
fully exercised integration rather than an unsupported paper reference. On the
frozen released DLO2 evaluation, the source-confirmed physical model plus causal
local residual reached `7.8606 mm` mean coordinate L1 over all 14 unique
trajectories, improving its identically trained physical checkpoint by `10.13%`
with `14/14` paired wins. The canonical released-loader compatibility draw was
`8.5037 mm`, below DEFORM's published `9.7 mm` reference. This supports a
best-publicly-reported-result claim for that exact DLO2 contract only, not a
general deformable-object or cross-benchmark state-of-the-art claim. See the
[executable protocol](docs/deform_dlo2_local_residual_official_v7.md) and the
[canonical result record](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/deform_dlo2_local_residual_official_v7_result.md).

## Experimental cross-action evaluation

The following research instruments are intentionally outside the stable wheel
and do not constitute physical evidence by themselves:

- [Cross-action physical transport](docs/cross_action_transport_v1.md) compares
  physical transport with discrepancy persistence, `last_residual`, and exact
  fallback on complete held-out action matrices.
- [Multi-action query identifiability](docs/multi_action_query_identifiability_v1.md)
  tests whether complementary actions make registered queries identifiable
  modulo one jointly declared nuisance model.
- [Whole-trajectory and decision value](docs/trajectory_value_v1.md) defines
  energy and variogram scores plus a target-blind finite-action regret record.

A positive certificate cannot rescue a failed empirical transport result, and a
favorable action regret cannot rescue failed provider, support, calibration, or
identifiability gates.

## Documentation map

- [Public claim snapshot](evidence/public_claim_snapshot_v1.json): generated
  release-status source with a pinned claim-contract identity.
- [Release-facing claim contract](docs/phystwin_release_claim_v1.md): authorized
  point, comparator, calibration, retrospective covariance, width, and
  independent-validation wording.
- [Minimal ecosystem smoke](docs/ecosystem_minimal_v1.md): executable
  Prob4D-compatible observation, guarded routing, exact fallback, and Causal4D
  provider manifest.
- [Provider-neutral guarded inference API v2](docs/inference_v2.md): caller-owned
  provider adapters, caller-owned guards, complete-belief routing, and exact
  fallback.
- [Strict Prob4D guarded inference API v1](docs/inference_v1.md): claim-bearing
  Prob4D candidate inference and frozen 0.4 compatibility.
- [Cross-action physical transport](docs/cross_action_transport_v1.md): sealed
  off-diagonal action evaluation and complete-session decision rules.
- [Multi-action query identifiability](docs/multi_action_query_identifiability_v1.md):
  stacked action-conditioned designs, joint nuisance structure, and bounded
  certificate semantics.
- [Whole-trajectory and decision value](docs/trajectory_value_v1.md): energy,
  variogram, and finite-action regret evaluation contracts.
- [Evidence-first backend admission](docs/backend_admission_policy_v1.md):
  implementation-versus-evidence maturity and the qualification freeze.
- [Five-backend support matrix](docs/backend_support_matrix_v1.md): installed
  support contract, retained decisions, and recommendation boundaries.
- [DEFORM DLO2 official protocol](docs/deform_dlo2_local_residual_official_v7.md):
  specialized DLO integration, frozen evaluation, and benchmark-specific claim
  boundary.
- [Experiment and evidence index](docs/experiment_index.md): frozen reports,
  negative results, commands, and placement policy.
- [Causal4D provider v1](docs/causal4d_provider_v1.md): supported provider
  surface and provenance boundary.
- [Canonical paper notes](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper):
  scope, figures, result artifacts, and paper claims.

## Repository layout

```text
src/bayesian_phystwin/   reusable Python package and versioned contracts
tests/                   unit, conformance, and integration tests
examples/                small synthetic inputs and demos
configs/                 frozen experiment and compute configurations
scripts/                 local and remote execution helpers
docs/                    contracts, protocols, public boundaries, and experiment index
evidence/                public machine-readable claim status
results/                 implementation fixtures and protocol/source-gate receipts
```

Finalized paper-facing result evidence, interpretation, and claim tests belong
in the private
[BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper)
repository. This public repository retains the executable method, frozen
protocols, source tags, and compact receipts required to validate behavior.

Large datasets, checkpoints, rendered videos, and raw runs remain outside git in
ignored paths such as `data/`, `checkpoints/`, `runs/`, and `outputs/`.
