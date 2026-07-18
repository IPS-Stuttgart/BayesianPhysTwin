#!/usr/bin/env python3
"""Render one trajectory method on every official PhysTwin evaluation case."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import struct
import subprocess
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXPERIMENT = (
    "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"asset directory contains no files: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _files_sha256(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _renderer_code_sha256(upstream_root: Path) -> str:
    suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
    files = [
        path
        for path in upstream_root.glob("*.py")
        if path.is_file() and path.suffix in suffixes
    ]
    package = upstream_root / "gaussian_splatting"
    if package.is_dir():
        files.extend(
            path
            for path in package.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    if not files:
        raise ValueError(f"renderer source tree contains no code: {upstream_root}")
    return _files_sha256(upstream_root, files)


def _runtime_identity(python: Path) -> dict[str, object]:
    probe = """
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

names = ("torch", "torchvision", "gsplat", "kornia", "pytorch3d")
installed = {
    dist.metadata["Name"].lower()
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
versions = {
    name: metadata.version(name) if name in installed else None for name in names
}
native_files = []
for name in ("torch", "torchvision", "gsplat", "diff_gaussian_rasterization"):
    try:
        module = importlib.import_module(name)
    except (ImportError, OSError):
        continue
    module_path = Path(module.__file__).resolve()
    module_root = module_path.parent if module_path.name.startswith("__init__") else module_path
    if module_root.is_dir():
        native_files.extend(str(path.resolve()) for path in module_root.rglob("*.so"))
    elif module_root.suffix == ".so":
        native_files.append(str(module_root))

cuda = {"available": False}
try:
    import torch
    cuda.update(
        {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    )
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        cuda.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
except (ImportError, OSError, RuntimeError):
    pass

print(
    json.dumps(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "versions": versions,
            "cuda": cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_home": os.environ.get("CUDA_HOME"),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "native_files": sorted(set(native_files)),
        },
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [str(python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    native_files = [Path(value) for value in payload.pop("native_files")]
    native_digest = hashlib.sha256()
    for path in native_files:
        native_digest.update(str(path).encode("utf-8"))
        native_digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                native_digest.update(block)
        native_digest.update(b"\0")
    identity: dict[str, object] = {
        "python": str(python.resolve()),
        "python_sha256": _sha256(python.resolve()),
        "probe": payload,
        "native_extensions": {
            "files": [str(path) for path in native_files],
            "sha256": native_digest.hexdigest(),
        },
    }
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvidia_smi = None
    identity["nvidia_smi"] = nvidia_smi
    extension_root_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    if extension_root_value:
        extension_root = Path(extension_root_value).resolve()
        extensions = (
            sorted(extension_root.rglob("*.so")) if extension_root.is_dir() else []
        )
        identity["torch_extensions"] = {
            "root": str(extension_root),
            "sha256": (
                _files_sha256(extension_root, extensions) if extensions else None
            ),
            "files": [
                path.relative_to(extension_root).as_posix() for path in extensions
            ],
        }
    return identity


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_directory(staged: Path, target: Path) -> None:
    """Replace one case tree without exposing a mixed old/new directory."""

    backup = target.with_name(f".{target.name}.{os.getpid()}.backup")
    shutil.rmtree(backup, ignore_errors=True)
    had_target = target.exists()
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except BaseException:
        if had_target and backup.exists():
            os.replace(backup, target)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


@contextmanager
def _output_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".render_sweep.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another render sweep owns {output_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _expected_frames(split_path: Path) -> tuple[int, set[str]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    count = int(split["frame_len"])
    return count, {f"{index:05d}.png" for index in range(count)}


def _verified_png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG signature")
    offset = 8
    dimensions: tuple[int, int] | None = None
    saw_end = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ValueError("truncated PNG chunk data")
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + payload[data_start:data_end])
        if expected_crc != actual_crc:
            raise ValueError("invalid PNG chunk checksum")
        if chunk_type == b"IHDR":
            if length != 13 or dimensions is not None:
                raise ValueError("invalid PNG header")
            dimensions = struct.unpack(">II", payload[data_start : data_start + 8])
            if min(dimensions) < 1:
                raise ValueError("invalid PNG dimensions")
        if chunk_type == b"IEND":
            saw_end = True
            if crc_end != len(payload):
                raise ValueError("unexpected data after PNG end")
            break
        offset = crc_end
    if dimensions is None or not saw_end:
        raise ValueError("incomplete PNG")
    return dimensions


def _complete_output(output_dir: Path, case: str, expected: set[str]) -> bool:
    frame_dir = output_dir / case / "0"
    if not frame_dir.is_dir():
        return False
    actual = {path.name for path in frame_dir.glob("*.png")}
    if actual != expected:
        return False
    dimensions = set()
    try:
        for name in expected:
            dimensions.add(_verified_png_dimensions(frame_dir / name))
    except (OSError, ValueError):
        return False
    return len(dimensions) == 1


def _run_sweep_unlocked(
    upstream_root: Path,
    python: Path,
    trajectory_template: str,
    output_dir: Path,
    log_dir: Path,
    *,
    cases: tuple[str, ...] | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    resume: bool = False,
) -> dict[str, object]:
    """Run the released LBS renderer and verify every expected view-0 frame."""

    if "{case}" not in trajectory_template:
        raise ValueError("trajectory template must contain '{case}'")
    reference_root = upstream_root / "data" / "render_eval_data"
    selected = (
        tuple(sorted(path.name for path in reference_root.iterdir() if path.is_dir()))
        if cases is None
        else cases
    )
    if not selected:
        raise ValueError("no rendering cases were selected")
    if len(selected) != len(set(selected)):
        raise ValueError("rendering cases must be unique")

    renderer = upstream_root / "gs_render_dynamics.py"
    if not renderer.is_file():
        raise FileNotFoundError(renderer)
    if not python.is_file():
        raise FileNotFoundError(python)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    renderer_sha256 = _sha256(renderer)
    runner = Path(__file__).resolve()
    runner_sha256 = _sha256(runner)
    renderer_code_sha256 = _renderer_code_sha256(upstream_root)
    runtime_identity = _runtime_identity(python)
    prior_manifest_path = output_dir / "render_sweep_manifest.json"
    prior_manifest: dict[str, object] = {}
    if resume and prior_manifest_path.is_file():
        prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_renderer_sha256 = prior_manifest.get("renderer", {}).get("sha256")
    prior_runner_sha256 = prior_manifest.get("runner", {}).get("sha256")
    prior_renderer_code_sha256 = prior_manifest.get("renderer_code_sha256")
    prior_runtime_identity = prior_manifest.get("runtime_identity")
    prior_experiment = prior_manifest.get("experiment")
    prior_upstream_root = prior_manifest.get("upstream_root")
    prior_cases = prior_manifest.get("cases", {})
    records: dict[str, object] = {}
    started = datetime.now(timezone.utc).isoformat()
    result: dict[str, object] = {
        "schema_version": 2,
        "status": "in_progress",
        "started_at_utc": started,
        "completed_at_utc": None,
        "upstream_root": str(upstream_root.resolve()),
        "renderer": {
            "path": str(renderer.resolve()),
            "sha256": renderer_sha256,
        },
        "runner": {"path": str(runner), "sha256": runner_sha256},
        "renderer_code_sha256": renderer_code_sha256,
        "runtime_identity": runtime_identity,
        "experiment": experiment,
        "python": str(python.resolve()),
        "trajectory_template": trajectory_template,
        "output_dir": str(output_dir.resolve()),
        "case_count": len(selected),
        "cases": records,
    }
    _write_json_atomic(prior_manifest_path, result)

    for case in selected:
        split_path = reference_root / case / "split.json"
        frame_count, expected = _expected_frames(split_path)
        trajectory = Path(trajectory_template.format(case=case)).resolve()
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        source = upstream_root / "data" / "gaussian_data" / case
        model = upstream_root / "gaussian_output" / case / experiment
        for required in (source, model):
            if not required.is_dir():
                raise FileNotFoundError(required)

        trajectory_sha256 = _sha256(trajectory)
        source_sha256 = _tree_sha256(source)
        model_sha256 = _tree_sha256(model)
        prior_case = prior_cases.get(case, {})
        output_complete = _complete_output(output_dir, case, expected)
        output_sha256 = (
            _tree_sha256(output_dir / case / "0") if output_complete else None
        )
        reusable = (
            resume
            and prior_renderer_sha256 == renderer_sha256
            and prior_runner_sha256 == runner_sha256
            and prior_renderer_code_sha256 == renderer_code_sha256
            and prior_runtime_identity == runtime_identity
            and prior_experiment == experiment
            and prior_upstream_root == str(upstream_root.resolve())
            and prior_case.get("trajectory_sha256") == trajectory_sha256
            and prior_case.get("source_sha256") == source_sha256
            and prior_case.get("model_sha256") == model_sha256
            and prior_case.get("output_sha256") == output_sha256
            and output_complete
        )
        if reusable:
            status = "reused"
        else:
            staging_root = output_dir / ".staging" / f"{case}-{os.getpid()}"
            shutil.rmtree(staging_root, ignore_errors=True)
            staging_root.mkdir(parents=True)
            command = [
                str(python),
                str(renderer),
                "-s",
                str(source),
                "-m",
                str(model),
                "--name",
                case,
                "--output_dir",
                str(staging_root),
                "--quiet",
            ]
            environment = os.environ.copy()
            environment["PHYSTWIN_TRAJECTORY_PATH"] = str(trajectory)
            try:
                with (log_dir / f"{case}.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    subprocess.run(
                        command,
                        cwd=upstream_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                if not _complete_output(staging_root, case, expected):
                    raise RuntimeError(f"renderer output is incomplete for {case}")
                output_sha256 = _tree_sha256(staging_root / case / "0")
                _replace_directory(staging_root / case, output_dir / case)
            finally:
                shutil.rmtree(staging_root, ignore_errors=True)
            status = "rendered"

        records[case] = {
            "status": status,
            "frame_count": frame_count,
            "trajectory": str(trajectory),
            "trajectory_sha256": trajectory_sha256,
            "source": str(source.resolve()),
            "source_sha256": source_sha256,
            "model": str(model.resolve()),
            "model_sha256": model_sha256,
            "output": str((output_dir / case / "0").resolve()),
            "output_sha256": output_sha256,
        }
        if (
            _sha256(trajectory) != trajectory_sha256
            or _tree_sha256(source) != source_sha256
            or _tree_sha256(model) != model_sha256
            or not _complete_output(output_dir, case, expected)
            or _tree_sha256(output_dir / case / "0") != output_sha256
        ):
            raise RuntimeError(f"render inputs or outputs changed during {case}")
        _write_json_atomic(prior_manifest_path, result)

    final_runtime_identity = _runtime_identity(python)
    if (
        final_runtime_identity != runtime_identity
        or _sha256(renderer) != renderer_sha256
        or _sha256(runner) != runner_sha256
        or _renderer_code_sha256(upstream_root) != renderer_code_sha256
    ):
        raise RuntimeError("render runtime identity changed during the sweep")
    for case, record in records.items():
        _, expected = _expected_frames(reference_root / case / "split.json")
        if (
            _sha256(Path(record["trajectory"])) != record["trajectory_sha256"]
            or _tree_sha256(Path(record["source"])) != record["source_sha256"]
            or _tree_sha256(Path(record["model"])) != record["model_sha256"]
            or not _complete_output(output_dir, case, expected)
            or _tree_sha256(output_dir / case / "0") != record["output_sha256"]
        ):
            raise RuntimeError(
                f"render inputs or outputs changed before completion: {case}"
            )
    result["status"] = "complete"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest = output_dir / "render_sweep_manifest.json"
    _write_json_atomic(manifest, result)
    result["manifest"] = str(manifest.resolve())
    return result


def run_sweep(
    upstream_root: Path,
    python: Path,
    trajectory_template: str,
    output_dir: Path,
    log_dir: Path,
    *,
    cases: tuple[str, ...] | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    resume: bool = False,
) -> dict[str, object]:
    """Run one identity-locked sweep with exclusive ownership of its output."""

    with _output_lock(output_dir):
        return _run_sweep_unlocked(
            upstream_root,
            python,
            trajectory_template,
            output_dir,
            log_dir,
            cases=cases,
            experiment=experiment,
            resume=resume,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_root", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("trajectory_template")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_sweep(
        args.upstream_root,
        args.python,
        args.trajectory_template,
        args.output_dir,
        args.log_dir,
        cases=None if args.cases is None else tuple(args.cases),
        experiment=args.experiment,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
