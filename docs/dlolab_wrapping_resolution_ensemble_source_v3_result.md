# DLO-Lab wrapping model-resolution ensemble source result v3

## Status

**Complete source result; prospective gate failed.**

The one registered attempt ran under frozen revision
`78cc16b61e80f7745b3c7783739f965c3dcdf0e2`. Its CPU runtime preflight
passed, all six prefix batches sealed, and the pre-future gate passed before
any task future was generated. All 48 fresh continuous-material worlds then
completed ordinary native QA with zero technical failures or replacements.

The equal finite/continuous posterior-value ensemble improved mean native
reward over the continuous-prior best fixed action by `0.0183675` (paired 95%
world-bootstrap interval `[0.0029963, 0.0297600]`) and captured `57.84%` of
available oracle headroom. It harmed two worlds beyond the frozen numerical
margin, compared with three for finite-particle Bayes.

The ensemble did not improve on continuous Bayes and narrowly missed the
registered value-retention threshold:

| Comparison for equal-resolution ensemble | Mean reward difference | Paired 95% CI |
| --- | ---: | ---: |
| best fixed action | `+0.0183675` | `[+0.0029963, +0.0297600]` |
| finite-particle Bayes | `-0.0009745` | `[-0.0019398, +0.0001001]` |
| continuous Bayes | `-0.0004201` | `[-0.0050521, +0.0029594]` |
| continuous MAP | `-0.0035138` | `[-0.0105561, +0.0020279]` |
| resolution maximin | `+0.0014884` | `[-0.0029641, +0.0046038]` |

It retained `94.9617%` of finite Bayes's gain, just below the frozen `95%`
threshold. Continuous Bayes also harmed only one world, so the ensemble failed
the nonnegative mean-gain and no-more-harms comparisons against continuous
Bayes. The source gate therefore failed three declared checks. No successor is
automatically authorized.

## Interpretation

This result independently extends the positive decision-value signal to a
second fresh off-grid roster: Bayesian action selection again beats fixed
control with a positive paired interval. Across v2 and v3, the central useful
finding is now robust: a compact physical belief over material uncertainty has
prospective control value on public native simulation.

Equal averaging across finite and interpolated belief resolutions is not the
mechanism that improves it. The v2 post-open lead did not transfer: averaging
reduced one finite-Bayes harm but was slightly worse than both finite and
continuous Bayes, while continuous Bayes itself was safer. The evidence favors
reporting finite and continuous resolutions as sensitivity controls rather than
fusing them into a new controller.

Continuous MAP achieved the highest mean reward (`0.912796`) on this roster,
but this is a post-result observation with two harmed worlds and no prospective
MAP-promotion gate. It is hypothesis-generating only and does not authorize a
new study by itself.

The result strengthens a scoped paper contribution around prospective Bayesian
decision-making under material uncertainty, exact information barriers, and
honest negative mechanism tests. It does not establish official benchmark SOTA,
real-world transfer, physical-parameter identification, or safety.

## Evidence identities

| Artifact | Identity |
| --- | --- |
| Frozen source commit | `78cc16b61e80f7745b3c7783739f965c3dcdf0e2` |
| Development diagnostic | `b2c2365c3d8f8f5702d250d7d1de62cfd2ac283ba297203e8e920a51b5c6b594` |
| Runtime preflight | `7537a19b82da8d478073466783d30e7d5a626ce5e1f149a39705c6aaef7fc43a` |
| Study lock | `a783f8148dbdf1a570b2a060b526468c1924bd86d89b1574aa72845344a4026a` |
| Decision barrier | `5d73917f1dc12bfec474433bf9b7c273481e9da8147cef93e705caa483565a42` |
| Generation seal | `a03062d2ffab0daa2ff656743800bdea65ac17fce4c00170541b93878bd96b58` |
| Result | `c187d1002f9c0244cea0356a5daac7cf987d3bb754ebb1021d9329f15ac47b19` |

The second arithmetic implementation reconstructs all decisions, 48 native
reward vectors, equal-world aggregation, paired intervals, and gate predicates.
It is not independent human review. The complete simulator tree remains under
`/home/fpfaff/source-only/dlolab-wrapping-resolution-ensemble-source-v3`.
