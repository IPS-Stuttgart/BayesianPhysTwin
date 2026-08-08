#!/usr/bin/env python3
"""Execute the admitted Deform360 calibration-only visual production."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import stat
import subprocess
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from bayesian_phystwin._canonical_contracts import canonical_relative_posix_path
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    load_deform360_calibration_visual_execution_admission,
)
from bayesian_phystwin.deform360_calibration_visual_production import (
    build_deform360_calibration_visual_prediction_seal,
    build_deform360_calibration_visual_production_result,
    build_deform360_calibration_visual_technical_failure,
    deform360_calibration_visual_command_descriptor,
    validate_deform360_calibration_visual_prediction_seal,
    validate_deform360_calibration_visual_production_result,
    validate_deform360_calibration_visual_technical_failure,
    validate_deform360_motioncrafter_model_set_binding,
    validate_deform360_motioncrafter_prediction_manifest,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)

_CHUNK = 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ProcessOutcome:
    return_code: int
    stdout: bytes
    stderr: bytes


_VARIABLE_CONFIG_FIELDS = frozenset(
    {"video_path", "output_directory", "seed", "frame_start", "frame_stop"}
)


class _SharedAdapterFactory:
    """Load the pinned model set once and rebind only per-job run fields."""

    def __init__(self, factory: Callable[[Any], Any]) -> None:
        self._factory = factory
        self._adapter: Any | None = None
        self._fixed_config: dict[str, object] | None = None
        self._creation_attempted = False
        self.creation_attempt_count = 0
        self.creation_count = 0

    @staticmethod
    def _fixed_record(config: Any) -> dict[str, object]:
        record = asdict(config)
        for field in _VARIABLE_CONFIG_FIELDS:
            record.pop(field, None)
        return {
            key: str(value) if isinstance(value, Path) else cast(object, value)
            for key, value in record.items()
        }

    @property
    def adapter(self) -> Any | None:
        return self._adapter

    def __call__(self, config: Any) -> Any:
        fixed = self._fixed_record(config)
        if self._adapter is None:
            if self._creation_attempted:
                raise RuntimeError(
                    "initial MotionCrafter adapter creation already failed"
                )
            self._creation_attempted = True
            self.creation_attempt_count += 1
            self._adapter = self._factory(config)
            self._fixed_config = fixed
            self.creation_count += 1
        else:
            if fixed != self._fixed_config:
                raise ValueError("per-job MotionCrafter config changed fixed fields")
            self._adapter.config = config
        return self._adapter


class _SharedMotionCrafterProducer:
    """Run integrity-bound per-view journals through one loaded adapter."""

    def __init__(
        self,
        *,
        model_binding: Mapping[str, Any],
        motioncrafter_root: Path,
        cache_directory: Path,
        provider_lock: Deform360VisualProviderLockV1,
    ) -> None:
        from prob4d.motioncrafter_integrity import (
            verify_motioncrafter_prediction_manifest,
        )
        from prob4d.motioncrafter_models import PinnedMotionCrafterModelSet
        from prob4d.motioncrafter_runner import SafeMotionCrafterRunner

        sources = cast(Mapping[str, Mapping[str, str]], model_binding["sources"])
        model_set = PinnedMotionCrafterModelSet.inspect(
            model_type="determ",
            unet_reference=sources["unet"]["repository"],
            unet_revision=sources["unet"]["revision"],
            vae_reference=sources["vae"]["repository"],
            vae_revision=sources["vae"]["revision"],
            image_vae_reference=sources["image_vae"]["repository"],
            image_vae_revision=sources["image_vae"]["revision"],
            base_pipeline_reference=sources["base_pipeline"]["repository"],
            base_pipeline_revision=sources["base_pipeline"]["revision"],
        )
        if model_set.set_sha256 != model_binding["model_set_id"]:
            raise ValueError("installed Prob4D model-set identity changed")
        try:
            installed_manifest = json.loads(model_set.manifest_json)
        except (AttributeError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                "installed Prob4D model-set manifest is invalid"
            ) from error
        if installed_manifest != model_binding["manifest"]:
            raise ValueError("installed Prob4D model-set manifest changed")
        self._model_set = model_set
        self._motioncrafter_root = motioncrafter_root
        self._cache_directory = cache_directory
        self._provider_lock = provider_lock
        self._runner_class = SafeMotionCrafterRunner
        self._verifier = verify_motioncrafter_prediction_manifest
        self._shared_factory = _SharedAdapterFactory(model_set.adapter_factory())

    @property
    def model_load_count(self) -> int:
        return self._shared_factory.creation_count

    @property
    def model_load_attempt_count(self) -> int:
        return self._shared_factory.creation_attempt_count

    def produce(
        self,
        *,
        job: Mapping[str, Any],
        source_video_path: Path,
        output_directory: Path,
        resume: bool,
    ) -> Path:
        prefix = cast(Sequence[int], job["prefix_source_frame_range_half_open"])
        config = self._model_set.build_config(
            upstream_root=self._motioncrafter_root,
            video_path=source_video_path,
            output_directory=output_directory,
            cache_directory=str(self._cache_directory),
            height=self._provider_lock.height,
            width=self._provider_lock.width,
            window_size=self._provider_lock.window_size,
            overlap=self._provider_lock.overlap,
            num_inference_steps=5,
            guidance_scale=1.0,
            decode_chunk_size=25,
            seed=cast(int, job["view_root_seed"]),
            seed_policy="derived-per-call",
            low_memory_usage=False,
            frame_start=prefix[0],
            frame_stop=prefix[1],
            frame_stride=1,
        )
        # The outer resume flag also covers receipt revalidation. Prob4D's
        # narrower flag is valid only after a per-job bundle has begun.
        resume_bundle = (
            resume and output_directory.is_dir() and any(output_directory.iterdir())
        )
        manifest = self._runner_class(
            config,
            adapter_factory=self._shared_factory,
        ).run(resume=resume_bundle)
        if self.model_load_attempt_count > 1:
            raise RuntimeError(
                "MotionCrafter adapter creation was attempted more than once"
            )
        return cast(Path, manifest)

    def verify(self, manifest_path: Path) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._verifier(manifest_path, verify_hashes=True),
        )

    def release_temporaries(self) -> None:
        gc.collect()
        adapter = self._shared_factory.adapter
        torch = getattr(adapter, "torch", None)
        cuda = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlinks(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(f"path contains a symbolic link: {candidate}")


def _read_stable(
    path: Path,
    *,
    label: str,
    capture: bool,
    maximum_bytes: int | None = None,
) -> tuple[bytes, str, int]:
    _reject_symlinks(path)
    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
        digest = hashlib.sha256()
        payload: list[bytes] = []
        count = 0
        while True:
            block = os.read(descriptor, _CHUNK)
            if not block:
                break
            digest.update(block)
            count += len(block)
            if capture:
                payload.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or count != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return b"".join(payload), digest.hexdigest(), count


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload, _digest, _count = _read_stable(
        path,
        label=label,
        capture=True,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return cast(dict[str, Any], value)


def _safe_member(root: Path, value: object, *, label: str) -> Path:
    safe = canonical_relative_posix_path(value, name=f"{label} path")
    _reject_symlinks(root)
    if not root.is_dir():
        raise ValueError(f"{label} root is not a directory")
    path = root
    for part in PurePosixPath(safe).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{label} path contains a symbolic link")
    return path


def _verify_source(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    path = _safe_member(root, record["path"], label=label)
    _payload, digest, count = _read_stable(path, label=label, capture=False)
    if digest != record["sha256"] or count != record["byte_count"]:
        raise ValueError(f"{label} differs from the admitted bytes")
    return path


def _portable_return_code(value: int) -> int:
    return value if value >= 0 else 128 + abs(value)


def _write_log(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = _safe_member(root, relative, label="log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def _descriptor(root: Path, path: Path) -> dict[str, object]:
    _payload, digest, count = _read_stable(
        path,
        label=path.name,
        capture=False,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "byte_count": count,
    }


def _verify_descriptor(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _safe_member(root, record["path"], label=label)
    _payload, digest, count = _read_stable(
        path,
        label=label,
        capture=False,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if digest != record["sha256"] or count != record["byte_count"]:
        raise ValueError(f"{label} differs from its receipt descriptor")
    return path


def _provider_lock(
    path: Path, admission: Mapping[str, Any]
) -> Deform360VisualProviderLockV1:
    lock = Deform360VisualProviderLockV1.from_mapping(
        _load_json(path, label="visual provider lock")
    )
    expected = (
        admission["visual_provider_lock_id"],
        admission["provider_revision"],
        admission["motioncrafter_revision"],
        admission["model_set_id"],
        admission["protocol_id"],
    )
    observed = (
        lock.artifact_id,
        lock.provider_revision,
        lock.motioncrafter_revision,
        lock.model_set_id,
        lock.protocol_id,
    )
    if observed != expected:
        raise ValueError("visual provider lock differs from the admission")
    if lock.seed_policy != "per-object-derived-seed-v1":
        raise ValueError("provider lock changed the object seed policy")
    if lock.additional_metric_anchor_policy != "none":
        raise ValueError("primary visual production cannot add metric anchors")
    return lock


def _checkout_revision(root: Path, *, expected: str, label: str) -> None:
    _reject_symlinks(root)
    if not root.is_dir():
        raise ValueError(f"{label} checkout is missing")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect {label} checkout") from error
    if revision != expected or status.strip():
        raise ValueError(f"{label} checkout differs from the frozen clean revision")


def _existing_receipt(
    *,
    run_root: Path,
    admission: Mapping[str, Any],
    lock: Deform360VisualProviderLockV1,
    implementation_revision: str,
    job: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    verify_prediction: Callable[[Path], Mapping[str, Any]],
    resume: bool,
) -> tuple[str, Path] | None:
    output = _safe_member(
        run_root,
        job["output_relative_directory"],
        label="job output",
    )
    seal_path = output / "prediction-seal.json"
    failure_path = run_root / "failures" / f"{job['job_id']}.json"
    present = [path for path in (seal_path, failure_path) if path.exists()]
    if len(present) > 1:
        raise ValueError("job has both success and failure receipts")
    if not present:
        return None
    if not resume:
        raise FileExistsError(f"job receipt already exists: {job['job_id']}")
    path = present[0]
    if path == seal_path:
        receipt = validate_deform360_calibration_visual_prediction_seal(
            _load_json(path, label="prediction seal")
        )
        status = "succeeded"
    else:
        receipt = validate_deform360_calibration_visual_technical_failure(
            _load_json(path, label="technical failure")
        )
        _verify_descriptor(
            run_root,
            cast(Mapping[str, Any], receipt["stdout"]),
            label="retained failure stdout",
        )
        _verify_descriptor(
            run_root,
            cast(Mapping[str, Any], receipt["stderr"]),
            label="retained failure stderr",
        )
        status = "technical-failure"
    descriptor = deform360_calibration_visual_command_descriptor(
        admission=admission,
        job=job,
        provider_lock=lock,
        model_binding=model_binding,
    )
    expected = {
        "implementation_revision": implementation_revision,
        "admission_id": admission["admission_id"],
        "job_id": job["job_id"],
        "object_id": job["object_id"],
        "episode_id": job["episode_id"],
        "stratum": job["stratum"],
        "camera_id": job["camera_id"],
        "provider_revision": lock.provider_revision,
        "motioncrafter_revision": lock.motioncrafter_revision,
        "visual_provider_lock_id": lock.artifact_id,
        "model_set_id": lock.model_set_id,
        "command_id": content_id(descriptor),
        "output_relative_directory": job["output_relative_directory"],
    }
    if {key: receipt.get(key) for key in expected} != expected:
        raise ValueError("existing receipt differs from the admitted execution")
    if status == "succeeded":
        manifest_record = cast(Mapping[str, Any], receipt["prediction_manifest"])
        manifest_path = _verify_descriptor(
            output,
            manifest_record,
            label="sealed prediction manifest",
        )
        verification = verify_prediction(manifest_path)
        contract = validate_deform360_motioncrafter_prediction_manifest(
            _load_json(manifest_path, label="sealed prediction manifest"),
            verification=verification,
            job=job,
            provider_lock=lock,
            model_binding=model_binding,
        )
        if (
            receipt["run_spec_sha256"] != contract["run_spec_sha256"]
            or receipt["verified_member_count"] != contract["member_count"]
        ):
            raise ValueError("sealed prediction verification identity changed")
    return status, path


def _result_row(
    *,
    run_root: Path,
    job: Mapping[str, Any],
    status: str,
    receipt: Path,
) -> dict[str, object]:
    return {
        "job_id": job["job_id"],
        "object_id": job["object_id"],
        "camera_id": job["camera_id"],
        "status": status,
        "receipt": _descriptor(run_root, receipt),
    }


def _failure(
    *,
    run_root: Path,
    attempt_id: str,
    implementation_revision: str,
    admission: Mapping[str, Any],
    lock: Deform360VisualProviderLockV1,
    job: Mapping[str, Any],
    command_id: str,
    stage: str,
    outcome: ProcessOutcome,
    detail: bytes,
) -> Path:
    job_id = cast(str, job["job_id"])
    base = f"logs/{attempt_id}/{job_id}.{stage}"
    stdout = _write_log(run_root, f"{base}.stdout.bin", outcome.stdout)
    stderr = _write_log(run_root, f"{base}.stderr.bin", outcome.stderr)
    receipt = build_deform360_calibration_visual_technical_failure(
        implementation_revision=implementation_revision,
        admission=admission,
        job=job,
        provider_lock=lock,
        command_id=command_id,
        stage=stage,
        return_code=_portable_return_code(outcome.return_code),
        detail=detail,
        stdout=stdout,
        stderr=stderr,
    )
    path = run_root / "failures" / f"{job_id}.json"
    write_atomic_json(receipt, path, overwrite=False)
    return path


def execute(
    *,
    admission_path: Path,
    provider_lock_path: Path,
    model_binding_path: Path,
    retained_root: Path,
    output_root: Path,
    prob4d_root: Path,
    motioncrafter_root: Path,
    cache_directory: Path,
    implementation_revision: str,
    attempt_id: str,
    resume: bool,
) -> dict[str, Any]:
    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    if not attempt_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in attempt_id
    ):
        raise ValueError("attempt_id must contain only letters, digits, '-' and '_'")
    admission = load_deform360_calibration_visual_execution_admission(admission_path)
    lock = _provider_lock(provider_lock_path, admission)
    binding = validate_deform360_motioncrafter_model_set_binding(
        _load_json(model_binding_path, label="model-set binding"),
        expected_model_set_id=lock.model_set_id,
    )
    _checkout_revision(prob4d_root, expected=lock.provider_revision, label="Prob4D")
    _checkout_revision(
        motioncrafter_root,
        expected=lock.motioncrafter_revision,
        label="MotionCrafter",
    )
    _reject_symlinks(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / cast(str, admission["admission_id"]) / implementation
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "visual-production-result.json"
    existing_result: dict[str, Any] | None = None
    if result_path.exists():
        if not resume:
            raise FileExistsError("visual production result already exists")
        existing_result = validate_deform360_calibration_visual_production_result(
            _load_json(result_path, label="visual production result")
        )
        expected = {
            "implementation_revision": implementation,
            "admission_id": admission["admission_id"],
            "visual_provider_lock_id": lock.artifact_id,
            "provider_revision": lock.provider_revision,
            "motioncrafter_revision": lock.motioncrafter_revision,
            "model_set_id": lock.model_set_id,
        }
        if {key: existing_result.get(key) for key in expected} != expected:
            raise ValueError("existing result differs from this execution")

    jobs = [cast(Mapping[str, Any], row) for row in admission["jobs"]]
    sources: dict[str, Path] = {}
    for job in jobs:
        job_id = cast(str, job["job_id"])
        sources[job_id] = _verify_source(
            retained_root,
            cast(Mapping[str, Any], job["source_video"]),
            label=f"source video {job_id}",
        )
        _verify_source(
            retained_root,
            cast(Mapping[str, Any], job["source_timestamps"]),
            label=f"source timestamps {job_id}",
        )

    producer = _SharedMotionCrafterProducer(
        model_binding=binding,
        motioncrafter_root=motioncrafter_root,
        cache_directory=cache_directory,
        provider_lock=lock,
    )

    lock_path = run_root / ".production.lock"
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another visual production process holds the run lock"
            ) from error
        rows: list[dict[str, object]] = []
        for job in jobs:
            existing = _existing_receipt(
                run_root=run_root,
                admission=admission,
                lock=lock,
                implementation_revision=implementation,
                job=job,
                model_binding=binding,
                verify_prediction=producer.verify,
                resume=resume,
            )
            if existing is not None:
                status, receipt = existing
                rows.append(
                    _result_row(
                        run_root=run_root,
                        job=job,
                        status=status,
                        receipt=receipt,
                    )
                )
                continue
            output = _safe_member(
                run_root,
                job["output_relative_directory"],
                label="job output",
            )
            output.mkdir(parents=True, exist_ok=True)
            descriptor = deform360_calibration_visual_command_descriptor(
                admission=admission,
                job=job,
                provider_lock=lock,
                model_binding=binding,
            )
            command_id = content_id(descriptor)
            try:
                manifest_path = producer.produce(
                    job=job,
                    source_video_path=sources[cast(str, job["job_id"])],
                    output_directory=output,
                    resume=resume,
                )
            except Exception:
                outcome = ProcessOutcome(
                    1,
                    b"",
                    traceback.format_exc().encode("utf-8", errors="replace"),
                )
                receipt = _failure(
                    run_root=run_root,
                    attempt_id=attempt_id,
                    implementation_revision=implementation,
                    admission=admission,
                    lock=lock,
                    job=job,
                    command_id=command_id,
                    stage="motioncrafter-production",
                    outcome=outcome,
                    detail=b"single-session producer raised an exception",
                )
                rows.append(
                    _result_row(
                        run_root=run_root,
                        job=job,
                        status="technical-failure",
                        receipt=receipt,
                    )
                )
                producer.release_temporaries()
                continue

            verification_outcome = ProcessOutcome(1, b"", b"")
            try:
                verification = producer.verify(manifest_path)
                manifest = _load_json(manifest_path, label="prediction manifest")
                verified_contract = (
                    validate_deform360_motioncrafter_prediction_manifest(
                        manifest,
                        verification=verification,
                        job=job,
                        provider_lock=lock,
                        model_binding=binding,
                    )
                )
                manifest_descriptor = _descriptor(output, manifest_path)
                seal = build_deform360_calibration_visual_prediction_seal(
                    implementation_revision=implementation,
                    admission=admission,
                    job=job,
                    provider_lock=lock,
                    command_id=command_id,
                    prediction_manifest=manifest_descriptor,
                    run_spec_sha256=cast(
                        str,
                        verified_contract["run_spec_sha256"],
                    ),
                    verified_member_count=cast(
                        int,
                        verified_contract["member_count"],
                    ),
                )
            except Exception:
                detail = traceback.format_exc().encode("utf-8", errors="replace")
                verification_outcome = ProcessOutcome(1, b"", detail)
                receipt = _failure(
                    run_root=run_root,
                    attempt_id=attempt_id,
                    implementation_revision=implementation,
                    admission=admission,
                    lock=lock,
                    job=job,
                    command_id=command_id,
                    stage="prediction-verification",
                    outcome=verification_outcome,
                    detail=detail,
                )
                rows.append(
                    _result_row(
                        run_root=run_root,
                        job=job,
                        status="technical-failure",
                        receipt=receipt,
                    )
                )
                producer.release_temporaries()
                continue
            seal_path = output / "prediction-seal.json"
            write_atomic_json(seal, seal_path, overwrite=False)
            rows.append(
                _result_row(
                    run_root=run_root,
                    job=job,
                    status="succeeded",
                    receipt=seal_path,
                )
            )
            producer.release_temporaries()
        if producer.model_load_attempt_count > 1:
            raise RuntimeError(
                "MotionCrafter adapter creation was attempted more than once"
            )
        result = build_deform360_calibration_visual_production_result(
            implementation_revision=implementation,
            admission=admission,
            provider_lock=lock,
            jobs=rows,
        )
        if existing_result is not None:
            if result != existing_result:
                raise ValueError("existing result differs from revalidated receipts")
            return existing_result
        write_atomic_json(result, result_path, overwrite=False)
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute or resume all admitted jobs")
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--visual-provider-lock", type=Path, required=True)
    run.add_argument("--model-set-binding", type=Path, required=True)
    run.add_argument("--retained-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--prob4d-root", type=Path, required=True)
    run.add_argument("--motioncrafter-root", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--implementation-revision", required=True)
    run.add_argument("--attempt-id", required=True)
    run.add_argument("--resume", action="store_true")

    validate = subparsers.add_parser(
        "validate-result",
        help="validate one compact visual-production result",
    )
    validate.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate-result":
            result = validate_deform360_calibration_visual_production_result(
                _load_json(arguments.path, label="visual production result")
            )
        else:
            result = execute(
                admission_path=arguments.admission,
                provider_lock_path=arguments.visual_provider_lock,
                model_binding_path=arguments.model_set_binding,
                retained_root=arguments.retained_root,
                output_root=arguments.output_root,
                prob4d_root=arguments.prob4d_root,
                motioncrafter_root=arguments.motioncrafter_root,
                cache_directory=arguments.cache_dir,
                implementation_revision=arguments.implementation_revision,
                attempt_id=arguments.attempt_id,
                resume=arguments.resume,
            )
    except (FileExistsError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result["technical_failure_job_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
