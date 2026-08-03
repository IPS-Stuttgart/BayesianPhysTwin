# Third-party notices and artifact licensing

## Scope of the project license

Unless an individual file states otherwise, source code, configuration files,
schemas, tests, and original documentation authored for this repository are
licensed under the MIT License in [`LICENSE`](LICENSE).

That license does **not** relicense third-party source trees, datasets, videos,
images, model weights, checkpoints, papers, or other externally supplied
artifacts. It also does not override restrictions attached to derived outputs
whose provenance includes such material. Obtain external assets from their
original publishers and comply with their current license, access, citation,
and use conditions.

The Python wheel and source distribution are intended to contain only this
project's package code and documentation. Large data, upstream repositories,
model weights, checkpoints, and generated runs are not intended to be bundled
with a release.

This inventory records the terms visible at the pinned revisions used by the
repository. It is not legal advice and does not replace the complete upstream
license text. Re-check upstream terms before downloading, using, or
redistributing an external asset.

## External projects and assets

The repository contains adapters, download helpers, provenance records, or
experimental instructions for the following external resources. This inventory
is attribution and scope documentation, not a substitute for the upstream
terms.

| Resource | Role in this project | Distribution and license boundary |
| --- | --- | --- |
| [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) | Official spring-mass simulator, released trajectories, data layout, renderer, and metric definitions used by integration and evaluation paths. | The integration is pinned to commit `2b6630528141b9cba5a7677c8b88b2129b4a8390`; the source license at that revision is MIT. Released data, checkpoints, videos, and generated artifacts are separate assets. The current PhysTwin dataset card also identifies MIT terms, but users must verify the card and file-specific notices at download time. Bayesian PhysTwin does not bundle or relicense those assets. |
| [CoTracker / CoTracker3](https://github.com/facebookresearch/co-tracker) | Optional point-tracking code and pretrained checkpoints used to recover continuous camera-track cues. | Frozen extraction uses revision `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` and checkpoint SHA-256 `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218`. The repository license at that revision is Creative Commons Attribution-NonCommercial 4.0. Code and checkpoints are obtained separately, are not MIT-licensed by this repository, and must not be treated as commercially unrestricted. |
| [MotionCrafter](https://github.com/TencentARC/MotionCrafter) | Optional external geometry and scene-flow predictions used by diagnostic association and assimilation experiments. | The workflow is pinned to revision `1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257`. Its upstream terms limit use to academic purposes, prohibit commercial or production use, and state that MotionCrafter is not intended for use within the European Union. Source, weights, caches, and generated predictions remain external. Review the pinned `LICENSE.txt` and `NOTICE` before enabling this optional path. |
| [Deform360](https://github.com/lhy0807/deform360) and its [dataset card](https://huggingface.co/datasets/brownu/deform360) | Optional public-data evaluation and prospective belief experiments. | Dataset contents and helper-library code remain under their upstream terms. Downloaded episodes, camera streams, tactile data, annotations, and derived media must not be committed here unless redistribution is explicitly permitted. |
| [MatPhys](https://arxiv.org/abs/2605.19386) | Optional learned material-prior and spring-field experiments. | Upstream source, checkpoints, and unpublished/generated semantic artifacts are not part of this project's license. Protocols must record the exact external revision and must not imply that public proxy experiments reproduce unavailable paper artifacts. |
| [Prob4D](https://github.com/IPS-Stuttgart/Prob4D) and [Causal4D](https://github.com/IPS-Stuttgart/Causal4D) | Companion repositories that produce observation beliefs and consume provider/artifact contracts. | Each repository is independently versioned and governed by its own current repository terms. The Bayesian PhysTwin MIT license does not apply automatically or grant rights to either repository. Cross-repository manifests must record exact revisions for frozen evidence. |
| NumPy, SciPy, OpenCV, RemoteZip, PyRecEst, and other declared Python dependencies | Runtime or optional package dependencies. | They are installed separately by the package manager and retain their own licenses. Consult the installed distribution metadata and upstream project notices for the exact versions used. |

## Generated results and bundled small artifacts

Small project-authored result summaries, fixtures, schemas, and configuration
files committed to this repository are covered by the project license unless a
nearby notice or provenance manifest says otherwise. Raw or processed data,
checkpoints, rendered media, and outputs derived from third-party inputs may
remain subject to upstream terms even when the transformation code is MIT
licensed. When redistributing an output, retain its manifest and verify every
recorded source license independently.

## Contribution requirements

A contribution that adds or downloads an external component must also record:

1. the canonical source or project page;
2. the exact version, revision, checkpoint digest, or dataset release;
3. the applicable license or access terms;
4. whether any bytes are redistributed by the repository, wheel, source
   distribution, workflow artifact, or release asset; and
5. the required citation and any non-commercial, research-only, privacy,
   geographic, or redistribution restrictions.

Do not copy third-party license text into `LICENSE`. Add a component-specific
notice next to vendored material, when vendoring is permitted, and update this
inventory instead.
