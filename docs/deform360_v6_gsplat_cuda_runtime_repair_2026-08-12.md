# Deform360 v6 gsplat CUDA runtime repair

## Failure evidence

Protected-main workflow run `31532027045` completed the isolated Python/GPU
bootstrap and the first prefix stage, then failed at
`frame-zero:026-sock-cloth-ep0007`. Nerfstudio reached the first gsplat
rasterization call, but gsplat had disabled its CUDA backend because no
`nvcc` toolkit was visible. The resulting backend object was `None`, and the
call failed while resolving `CameraModelType`.

The retained artifact is `9117335082` with digest
`sha256:bb14ce5365ba3dcd46c17d52ea0c2ba36d71ec678176ef761bfe65a526e49917`.
It records zero physical manifests and zero sealed source predictions. No
development suffix, v5 confirmation payload, v6 target payload, or target
outcome was opened.

## Repair

The runtime now materializes a run-local CUDA 12.1.1 compiler root from three
official NVIDIA redistributable archives:

- CUDA CCCL 12.1.109;
- CUDA runtime and headers 12.1.105; and
- CUDA NVCC 12.1.105.

Every archive URL and SHA-256 is frozen in
`protocols/amendments/deform360_official_hub_fresh_object_session_v6_gsplat_cuda_runtime.json`.
The archive bytes are verified before extraction. The normalized root is
exported as `CUDA_HOME`/`CUDA_PATH`, and the JIT extension cache is unique to
the workflow run and attempt.

Before any source-science command runs, the bootstrap now requires:

1. `nvcc` to report CUDA 12.1;
2. a trivial CUDA translation unit to compile; and
3. gsplat's compiled backend to load and expose `CameraModelType`.

Failure remains terminal and target-closed.

## Scientific boundary

This repair changes no source or target object, point predictor, covariance
candidate, model family, model size, observation model, loss, gate, endpoint,
or reporting rule. It authorizes only technical completion of the already
registered source-prediction execution. Claim authorization, fresh-target
selection, and target access remain false.

- Repair ID: `44da91d95947d07d9d930bd0c707d16da9555bc7b9ea3042fcf0a88444ec3bb4`
- Repair file SHA-256: `fb532bf9626c0ba48cb9c7e4aca80488e12f255d18765a31bf4f4324deb385c7`
