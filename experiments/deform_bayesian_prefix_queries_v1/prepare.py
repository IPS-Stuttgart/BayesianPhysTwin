"""Export only fitting/validation replays from the frozen native checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("an exact execution commit is required")
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked(identity: dict) -> Path:
    path = Path(identity["path"]).resolve(strict=True)
    if digest(path) != identity["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")
    if "size_bytes" in identity and path.stat().st_size != identity["size_bytes"]:
        raise ValueError(f"size mismatch: {path}")
    return path


def permitted_names(manifest: dict, protocol: dict) -> dict[str, list[str]]:
    if (
        manifest["dlo_type"],
        manifest["partition"],
        manifest["official_eval_read"],
    ) != ("DLO3", "train", False):
        raise ValueError("not the authorized DLO3 training manifest")
    split = manifest["split"]
    if {key: len(value) for key, value in split.items()} != protocol["split_counts"]:
        raise ValueError("partition counts differ")
    flattened = sum((list(value) for value in split.values()), [])
    if len(flattened) != len(set(flattened)):
        raise ValueError("overlapping split membership")
    return {key: list(split[key]) for key in ("fit", "calibration")}


def causal_replay_input(full: np.ndarray) -> np.ndarray:
    """Native rollout receives no future free-node coordinates, even as baggage."""
    if full.shape != (500, 12, 3) or not np.isfinite(full).all():
        raise ValueError("invalid source trajectory")
    safe = full.copy()
    safe[2:, 2:-2] = full[1, 2:-2]
    return safe


def install_data_guard(allowed: set[Path], forbidden: set[Path]) -> None:
    def audit(event: str, args: tuple) -> None:
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        text = path.as_posix()
        if path in forbidden or "held-v8" in path.parts:
            raise PermissionError("protected source-test or held-v8 payload")
        if "/data_set/" in text and path.suffix == ".pkl" and path not in allowed:
            raise PermissionError("dataset payload not in fit/validation whitelist")

    sys.addaudithook(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", type=revision, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest_path = checked(protocol["source_manifest"])
    manifest = json.loads(manifest_path.read_text())
    names = permitted_names(manifest, protocol)
    allowed = {
        Path(manifest["trajectories"][n]["path"]).resolve()
        for group in names.values()
        for n in group
    }
    forbidden = {
        Path(manifest["trajectories"][n]["path"]).resolve()
        for n in manifest["split"]["source_test"]
    }
    install_data_guard(allowed, forbidden)
    checkpoint = checked(protocol["checkpoint"])
    for group in names.values():
        for name in group:
            checked(manifest["trajectories"][name])
    provenance = {
        "implementation_commit": args.revision,
        "contract": protocol["contract"],
        "protocol_sha256": digest(protocol_path),
        "source_manifest_sha256": digest(manifest_path),
        "checkpoint_sha256": digest(checkpoint),
        "device": args.device,
        "partitions": names,
        "source_test_payload_read": False,
        "official_evaluation_read": False,
        "held_v8_access": False,
        "future_free_nodes_passed_to_simulator": False,
        "source_files": {
            str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.glob("*.py"))
        },
    }
    write_json(output / "preflight.json", provenance)
    if args.preflight_only:
        print(json.dumps(provenance, indent=2), flush=True)
        return
    try:
        import run_deform_dlo_longrun_posterior as runtime
        import run_deform_dlo_source as source
        import torch

        torch.set_num_threads(args.threads)
        upstream = Path(protocol["upstream_root"])
        source._assert_upstream(upstream, protocol["upstream_commit"])
        modules = source._load_upstream(upstream)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "model_state_dict"
        ]
        receipts = {}
        for partition, members in names.items():
            started = time.perf_counter()
            trajectories = source._load_named_trajectories(
                manifest, members, frame_count=500, node_count=12
            )
            safe = {n: causal_replay_input(trajectories[n]) for n in members}
            print(f"replaying {partition}: {len(members)} recordings", flush=True)
            replay = runtime._evaluate_state(
                state,
                safe,
                modules=modules,
                torch=torch,
                device=args.device,
                dlo_type="DLO3",
                node_count=12,
            )
            if replay["names"] != members:
                raise ValueError("native replay changed recording order")
            full = np.stack([trajectories[n] for n in members])
            path = output / f"{partition}.npz"
            np.savez_compressed(
                path,
                names=np.asarray(members),
                initial=full[:, :2],
                action=full[:, 2:, (0, 1, 10, 11)],
                baseline=np.asarray(replay["predictions"]),
                target=full[:, 2:],
            )
            receipts[partition] = {
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
                "count": len(members),
                "seconds": time.perf_counter() - started,
            }
            print(json.dumps({partition: receipts[partition]}), flush=True)
        write_json(output / "manifest.json", {**provenance, "files": receipts})
    except Exception as exc:
        write_json(
            output / "technical_failure.json",
            {
                **provenance,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "retry_authorized": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
