# Query-conditional simulator competence certificates v1

## Motivation

A backend name is not a competence statement. The same inference and guard
strategy can retain useful decision value on one deformable-manipulation query
and fail on another query in the same public simulator family. Pooling those
outcomes into one backend-wide label either hides the failure or discards the
valid positive result.

`bayesian_phystwin.query_competence_certificate_v1` makes the narrower claim
executable. It binds competence to the exact tuple

```text
(simulator, task, observation policy, action bank,
 metric, world distribution, statistical unit).
```

Every semantic component is content-addressed. The certificate also binds the
candidate and baseline policies, frozen gate, complete denominator, source
result, verifier, and verified result tree. A passing certificate for one tuple
does not transfer to another tuple merely because both use the same backend.

## Contract

A `SimulatorQueryScopeV1` defines an exact query identity. A
`QueryCompetenceGateV1` freezes:

- the complete independent-group denominator;
- minimum mean decision gain and a positive paired confidence lower bound;
- the maximum exact one-sided harm-risk upper bound;
- minimum downside reduction;
- minimum retained candidate value and oracle headroom; and
- technical-failure, retry, and replacement budgets.

A `QueryCompetenceCertificateV1` then binds the observed metrics and custody
record. It recomputes the one-sided Clopper-Pearson harm bound from the harmful
and total group counts. Its pass/fail value and failed-check roster are derived,
not trusted as free booleans. A source study rejected by its own registered
gate cannot be promoted through this adapter even if a subset of the generic
numeric checks happens to pass.

A `QueryCompetenceRegistryV1` stores independent certificates by exact query
ID. It deliberately does not pool evidence across tasks and does not infer a
backend-wide label.

## Exact fallback

`select_query_competent_belief` admits the candidate complete belief only when:

1. the application query exactly matches a registered query;
2. the candidate and baseline policy identities exactly match that query's
   certificate;
3. the query certificate passes; and
4. the current inference is independently admissible.

Every other path, including a failed query, unknown query, changed observation
policy, changed action bank, policy mismatch, or inference failure, routes
through `select_complete_belief`. Rejection returns the original baseline
belief object by identity; it does not reconstruct a nominal state from zero
corrections.

This differs intentionally from
`DomainGuardHarmRiskCertificateV1`. That earlier contract evaluates one shared
policy over a declared domain roster and rejects deployment everywhere when any
supported domain fails. The query registry instead holds separately versioned
policies and evidence. Failure of the Slingshot policy therefore cannot erase a
separately certified wrapping policy, but neither can wrapping authorize
Slingshot.

## Prospective cross-task evidence

The committed registry is generated only from the frozen, read-only verified
source records:

```text
results/source/dlolab_query_competence_registry_v1/registry.json
```

- registry artifact ID:
  `017fe497894142cb5b4cffac933d8e1ff2ee6bd9e18463f43e1868b0ad731a4b`
- registry file SHA-256:
  `8f8b3dc7ab750420cbe8732d0a24679be772b21aff45abc69be0633b638e0159`

Both studies use public DLO-Lab simulation, 288 fresh worlds, 4096 sensor draws
per world, a world-level statistical unit, no replacements, and no task
retries. Their task, observation, action, metric, distribution, and policy
identities remain separate.

| Exact query | Gain over baseline | Paired 95% gain CI | Harmed worlds | One-sided 95% harm upper | Downside reduction | Retained candidate gain | Oracle headroom | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DLO-Lab wrapping v9 | +0.004721 | [0.003894, 0.005597] | 1/288 | 0.016365 | 97.72% | 19.43% | 8.70% | **Certified** |
| DLO-Lab Slingshot v2 | +0.000220 | [-0.000111, 0.000530] | 14/288 | 0.074952 | 92.65% | 1.38% | 0.86% | **Rejected** |

The Slingshot guard removes 48 of the unguarded posterior controller's 62 harm
events and reduces mean downside substantially, but it misses the frozen gain,
paired-interval, risk, retained-value, and oracle-headroom gates. It remains an
informative negative result, not a promoted controller and not a retuning set.

A later prospective policy-level follow-up tested the most direct rescue: use a
five-neighbor predictor for only the posterior-selected action, calibrate its
one-sided gain bound on 96 new worlds, and compare it with a simultaneous-action
guard under the identical calibration budget. All 96 calibration futures and
all 288 fresh evaluation prefixes passed native QA, but the frozen pre-future
gate admitted only 12 worlds versus the required 24. The study therefore
stopped with 276 exact fallbacks and zero evaluation futures generated. Its
content-bound result is `f0ac1753c92630bcc738db30f466f0745ec726d7aff74b99a0198e5aca6fb25b`.
This does not change the table above because no prospective value outcome was
opened. It closes the exact local-five-neighbor plus global-conformal-offset
rescue and preserves the Slingshot query as rejected.

The retained negative made its 96 calibration worlds and 288 prefix-only worlds
available for explicitly labeled development. A posterior-aware successor now
combines causal relative geometry with joint/iid posterior weights and
incumbent-relative action diagnostics, then predicts selected-policy gain with
seven inverse-distance neighbors. Across 30 deterministic
train/calibrate/evaluate rotations over 147 opened source worlds, median guarded
gain is `+0.004774`, every rotation remains positive, median marginal coverage
is `90.48%`, and median harmful admissions are zero. An outcome-free capacity
check admits 44/288 prefixes while leaving every associated future unread. The
content-bound development artifact is
`5b8e50986f1f7dc7785389fa840a2e0993cc8bcaa5a5c3d8095567ff4c81e682`.
This passes only the source-development advancement gate; a Slingshot
certificate still requires entirely fresh calibration and evaluation worlds.

The cross-task result is stronger than either study alone: an exact-fallback
Bayesian guard has useful, prospectively verified decision value for one public
deformable task, while an independently frozen task in the same simulator
family demonstrates that this value must not be generalized by backend name.

## Reproduction

The registry builder first rehashes every bound protocol, result, summary, and
verifier source artifact, then checks the registered values before constructing
the certificates:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_competence_registry_v1.py \
  --output /tmp/dlolab-query-competence-registry.json
```

The committed tests rebuild the registry, compare its full content-addressed
record, recompute the exact harm bounds, and exercise positive, failed,
unknown-query, changed-policy, and inference-rejected routing.

## Paper contribution

The paper-level contribution is not a claim that BayesianPhysTwin is globally
safe or that DLO-Lab predicts physical robots. It is a falsification-first
evaluation and deployment contract:

1. define simulator competence at the query-policy level;
2. require prospective value and finite-group harm evidence;
3. preserve negative tasks as rejected certificates rather than averaging them
   away;
4. retain separately valid positive certificates; and
5. make unsupported scope transfer fail closed through exact complete-belief
   fallback.

This is public-simulator evidence only. It is not an official benchmark or SOTA
claim, a distribution-free guarantee, independent human review, a real-robot
safety certificate, or evidence for arbitrary unseen world distributions. The
two 288-world panels are closed to policy, threshold, and gate selection.
