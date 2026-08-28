# Contact-Path Screen: Retained-Control Replay Failure

The source implementation was frozen at
`c20e8866d086513c05e00907858a4f20c72c19eb`. All three registered CPU batches
completed. The screen stopped at its unchanged reference replay gate;
**no source information-value calculation or method evaluation ran**.

| Accounting | Count |
|---|---:|
| Registered and completed native worlds | 3/3 |
| Base native QA passed | 3/3 |
| Complete retained-reference QA passed | 2/3 |
| Retained reference-qualification failure | 1/3 |
| Unsealable, unrun, replaced, or retried worlds | 0 |
| Source value calculations | 0 |

All 93 actual force commands and three release commands matched the lock.
All common-prefix, fixed-endpoint, within-batch duplicate, and original
fallback checks passed. The maximum original-fallback position error was
1.09305e-7 m, and the maximum entire-prefix difference was 1.36315e-13 m.

The failure occurred in the middle coupling world (0.6), for slot 6, which
retains the previous force screen's same slot-6 motion and -24 N force.
The Cartesian input bytes match, and the force schedule is verified, but
its target-cube position differs by 7.47641e-5 m (0.074764 mm) from the
earlier source run. This exceeds the locked 1 micrometre tolerance, and
the cumulative native reward is not bit-identical.

For that retained control, the maximum differences are:

| Readout | Maximum absolute difference (m) |
|---|---:|
| Rod positions | 9.67543e-9 |
| Gripper positions | 7.33191e-13 |
| Projectile positions | 1.13404e-6 |
| Target-cube positions | 7.47641e-5 |

This pattern is consistent with amplification through subsequent contact;
it does not establish the numerical cause or estimate a noise distribution.
It is neither evidence that the new recovery paths help nor a scientific
rejection of those paths. No action-value ranking is reported, no tolerance
is relaxed, and the failed bank is not rerun. A numerical-repeatability
study would need its own source-only lock and cannot retrospectively
authorize this stopped screen.

## Verification and Preservation

The implementation passed 223 relevant tests, Ruff, focused MyPy, and the
exact source/runtime preflight before native execution. An initial preflight
command omitted the required CPU/software-rendering environment and was
rejected before initialization; correcting that command did not run a
scientific attempt. The native screen itself ran once per registered world.

The second arithmetic implementation `verify_path.py` rehashed source
records, arrays and committed code; checked 24 native rewards, all force and
release commands, and all reference errors; reproduced the failed decision;
and verified that the source-value bank and metrics are absent. It executed
no simulator and is not independent human review.

- Lock ID: `af8acc9e7de8bc4b25551a089be06e2af2a61fc7ff474c42e4ff5b49342d6672`.
- Lock file SHA-256: `5e807ab2c552513e0531b905a3e200851dfeceabd73bd87adfbd4a102d6a3cea`.
- Result ID: `baa9b3d5baca3853255b191d06f38f1797479409c192556c318dd49f54d6eaa9`.
- Result file SHA-256: `a2124c85954e6e455a5f0d9b7826c20c99bca330cc00af1df0f0fe70d30fabf8`.

The complete write-once native root remains
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/contact-path-source-v1`.
Compact source records accompany this note under
`results/source/dlolab_slingshot_contact_path_v1/`.
Nothing is pushed or merged to main. Previous DEFORM, contact, grip-force,
and failed belief-study evidence remains unchanged. No new recordings,
GPU, robot, protected targets, held-v8, or DLO4/DLO5 were accessed.
