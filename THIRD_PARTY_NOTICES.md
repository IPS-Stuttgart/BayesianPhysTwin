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

## External projects and assets

The repository contains adapters, download helpers, provenance records, or
experimental instructions for the following external resources. This inventory
is attribution and scope documentation, not a substitute for the upstream
terms.

| Resource | Role in this project | Distribution and license boundary |
| --- | --- | --- |
| [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) | Official spring-mass simulator, released trajectories, data layout, renderer, and metric definitions used by integration and evaluation paths. | Bayesian PhysTwin does not relicense PhysTwin source, released datasets, checkpoints, videos, or generated artifacts. Use the pinned revision recorded by each protocol and follow the upstream repository and dataset terms. |
| [CoTracker / CoTracker3](https://github.com/facebookresearch/co-tracker) | Optional point-tracking code and pretrained checkpoints used to recover continuous camera-track cues. | Code and checkpoints are obtained separately and are not covered by this repository's MIT license. Follow the upstream component-specific terms; frozen protocols additionally record the source revision and checkpoint digest. |
| [MotionCrafter](https://ruijiezhu94.github.io/MotionCrafter_Page/) | Optional external geometry and scene-flow predictions used by diagnostic association and assimilation experiments. | Source, pretrained models, caches, and generated predictions are external and are not redistributed by the Bayesian PhysTwin package. Follow the terms linked by the upstream project and preserve the pinned revision in experiment manifests. |
| [Deform360](https://github.com/lhy0807/deform360) and its [dataset card](https://huggingface.co/datasets/brownu/deform360) | Optional public-data evaluation and prospective belief experiments. | Dataset contents and helper-library code remain under their upstream terms. Downloaded episodes, camera streams, tactile data, annotations, and derived media must not be committed here unless redistribution is explicitly permitted. |
| [MatPhys](https://arxiv.org/abs/2605.19386) | Optional learned material-prior and spring-field experiments. | Upstream source, checkpoints, and unpublished/generated semantic artifacts are not part of this project's license. Protocols must record the exact external revision and must not imply that public proxy experiments reproduce unavailable paper artifacts. |
| [Prob4D](https://github.com/FlorianPfaff/Prob4D) and [Causal4D](https://github.com/FlorianPfaff/Causal4D) | Companion repositories that produce observation beliefs and consume provider/artifact contracts. | Each repository is independently versioned. Consult its own license and citation files; the Bayesian PhysTwin MIT license does not apply automatically. Cross-repository manifests must record exact revisions for frozen evidence. |
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
5. the required citation and any non-commercial, research-only, privacy, or
   redistribution restrictions.

Do not copy third-party license text into `LICENSE`. Add a component-specific
notice next to vendored material, when vendoring is permitted, and update this
inventory instead.
