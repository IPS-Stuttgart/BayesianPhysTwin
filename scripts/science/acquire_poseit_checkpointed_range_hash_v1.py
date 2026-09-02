#!/usr/bin/env python3
"""Bind the checkpoint transport to an explicit amendment before any request."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

from bayesian_phystwin._portable_contracts import (
    exact_revision,
    require_exact_fields,
    sha256_digest,
)
from bayesian_phystwin_experiments import poseit_checkpoint_acquisition as acquisition
from bayesian_phystwin_experiments.poseit_hash_checkpoint import RHashCheckpointEngine
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    load_poseit_range_transport_lock,
)
from bayesian_phystwin_experiments.poseit_remote_archive import RemoteArchiveExpectation

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/home/florianpfaff/source-only/poseit-real-decision-v1")
PROTOCOL_PREFIX = "protocols/poseit_real_decision_probe_v1"
PARENT_FILES = {
    f"{PROTOCOL_PREFIX}.json": "221803b109a82d3a2d923d5e0c18284b965a8848bcd69e25addd97409d31c5d4",
    f"{PROTOCOL_PREFIX}_preaccess_mapping_constraints.json": "8bf66c087437d77589d5fcd35d74a47b2a4d8ba69b311041123d719da8445210",
    f"{PROTOCOL_PREFIX}_method_lock.json": "4fa1ef3c96df28a67e13461b79c44690f53f5abb4c90e06200c4e90bcf8e1a1c",
    f"{PROTOCOL_PREFIX}_range_transport_lock.json": "8b3843bd4255aae980e3c8474f60fb38431bdb61e043a9a2e062d1c2acf8b67a",
    "src/bayesian_phystwin_experiments/poseit_remote_archive.py": "2db9da81a84e3d1b7cc0ad4e0270aa18e19b64e42a1c72aade98b35228db0881",
    "evidence/poseit-real-decision-v1/range-hash-v1-failure/terminal_observation.json": "1186ced3e319a8186a804d954868537ef293280bccf1586ba2c8bbc43bbc0c16",
    "evidence/poseit-real-decision-v1/range-hash-v2-failure/terminal_observation.json": "11f7adb855fd5dc4af9f368eb875b7715b383a9deff7443e61e46146ec329364",
    "evidence/poseit-real-decision-v1/delivery-recovery-feasibility-v1.json": "a5a9a0bd0bf55a7a26c2220728318da8cd0f65135ec0e2ea079551e4463f3fed",
}
IMPLEMENTATION_FILES = frozenset(
    {
        "scripts/science/acquire_poseit_checkpointed_range_hash_v1.py",
        "src/bayesian_phystwin/__init__.py",
        "src/bayesian_phystwin/_canonical_contracts.py",
        "src/bayesian_phystwin/_portable_contracts.py",
        "src/bayesian_phystwin_experiments/__init__.py",
        "src/bayesian_phystwin_experiments/poseit_checkpoint_acquisition.py",
        "src/bayesian_phystwin_experiments/poseit_hash_checkpoint.py",
        "src/bayesian_phystwin_experiments/poseit_real_decision_protocol.py",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_hash(path: Path) -> str:
    _require(path.is_file() and path.resolve() == path, "file is missing or linked")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context(
    amendment_path: Path,
    *,
    expected_amendment_sha256: str,
) -> tuple[acquisition.AcquisitionSpec, RHashCheckpointEngine]:
    """Validate only code, fixed protocol files, and administrative metadata."""
    sha256_digest(expected_amendment_sha256, name="amendment digest")
    _require(
        _file_hash(amendment_path) == expected_amendment_sha256,
        "transport amendment file changed",
    )
    amendment = acquisition._read(amendment_path, "checkpoint-transport-amendment")
    acquisition._fields(
        amendment,
        "status",
        "implementation_revision",
        "implementation_files",
        "parent_files",
        "native_library",
        "execution",
        "scientific_method_changed",
        "prior_partial_hashes_reused",
        "legacy_receipt_compatible",
        "resume_policy",
        "boundaries",
    )
    _require(
        amendment["status"] == "frozen-acquisition-only", "amendment is not frozen"
    )
    exact_revision(amendment["implementation_revision"], name="implementation revision")
    _require(
        amendment["parent_files"] == PARENT_FILES, "frozen parent bindings changed"
    )
    _require(
        isinstance(amendment["implementation_files"], dict)
        and set(amendment["implementation_files"]) == IMPLEMENTATION_FILES,
        "implementation file roster changed",
    )
    for relative, expected in {
        **PARENT_FILES,
        **amendment["implementation_files"],
    }.items():
        sha256_digest(expected, name=relative)
        path = ROOT / relative
        _require(_file_hash(path) == expected, f"bound file changed: {relative}")
        if relative.startswith("src/"):
            name = relative[4:-3].replace("/", ".").removesuffix(".__init__")
            module = sys.modules.get(name)
            _require(
                module is not None and Path(str(module.__file__)).resolve() == path,
                f"import did not use the bound source tree: {name}",
            )
    for name in (
        "scientific_method_changed",
        "prior_partial_hashes_reused",
        "legacy_receipt_compatible",
    ):
        _require(amendment[name] is False, f"amendment boundary changed: {name}")
    _require(
        amendment["resume_policy"]
        == "new-authorization-after-preserved-terminal-and-cooldown",
        "automatic resumption is not admitted",
    )
    acquisition._boundaries(amendment["boundaries"])
    execution = amendment["execution"]
    require_exact_fields(
        execution,
        name="checkpoint execution",
        expected=frozenset(
            {
                "host_alias",
                "hostname",
                "root",
                "lock_path",
                "first_request_not_before_utc",
                "resume_delay_seconds",
            }
        ),
    )
    _require(
        execution["host_alias"] == "gpuserver4090"
        and execution["hostname"] == socket.gethostname(),
        "registered execution host changed",
    )
    _require(
        execution["root"] == str(REMOTE_ROOT / "checkpoint-range-hash-v1")
        and execution["lock_path"] == str(REMOTE_ROOT / "range-hash.lock"),
        "registered acquisition or shared lock path changed",
    )
    _require(
        acquisition._utc(execution["first_request_not_before_utc"])
        >= acquisition._utc("2026-09-03T17:08:20.674819+00:00"),
        "provider cooldown was shortened",
    )
    native = amendment["native_library"]
    require_exact_fields(
        native, name="native library", expected=frozenset({"path", "sha256"})
    )
    library_path = Path(native["path"])
    _require(
        library_path.is_absolute() and library_path.resolve() == library_path,
        "native library path is not exact",
    )
    transport = load_poseit_range_transport_lock(
        ROOT / f"{PROTOCOL_PREFIX}_range_transport_lock.json",
        parent_protocol_path=ROOT / f"{PROTOCOL_PREFIX}.json",
        mapping_constraints_path=ROOT
        / f"{PROTOCOL_PREFIX}_preaccess_mapping_constraints.json",
        method_lock_path=ROOT / f"{PROTOCOL_PREFIX}_method_lock.json",
        range_transport_core_path=ROOT
        / "src/bayesian_phystwin_experiments/poseit_remote_archive.py",
    )
    source, settings = transport["source_identity"], transport["execution"]
    spec = acquisition.AcquisitionSpec(
        root=Path(execution["root"]),
        lock_path=Path(execution["lock_path"]),
        expectation=RemoteArchiveExpectation(
            source_url=source["source_url"],
            file_name=source["content_disposition_file_name"],
            size_bytes=source["archive_size_bytes"],
            last_modified=source["last_modified"],
            chunk_size_bytes=settings["chunk_size_bytes"],
            max_workers=settings["max_workers"],
            max_attempts_per_range=settings["max_attempts_per_range"],
            timeout_seconds=settings["timeout_seconds_per_range"],
        ),
        amendment_sha256=expected_amendment_sha256,
        library_sha256=native["sha256"],
        parent_sha256=PARENT_FILES,
        first_request_not_before_utc=execution["first_request_not_before_utc"],
        resume_delay_seconds=execution["resume_delay_seconds"],
    )
    engine = RHashCheckpointEngine(
        library_path, expected_library_sha256=spec.library_sha256
    )
    return spec, engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "verify", "publish"))
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--expected-authorization-sha256")
    args = parser.parse_args(argv)
    if args.mode == "run":
        if args.authorization is None or args.expected_authorization_sha256 is None:
            parser.error("run requires an exact authorization file and SHA-256")
    elif (
        args.authorization is not None or args.expected_authorization_sha256 is not None
    ):
        parser.error("only run accepts attempt authorization arguments")
    spec, engine = load_context(
        args.amendment.absolute(),
        expected_amendment_sha256=args.expected_amendment_sha256,
    )
    if args.mode == "run":
        assert args.authorization is not None
        assert args.expected_authorization_sha256 is not None
        result = acquisition.run_checkpointed_attempt(
            spec,
            engine,
            args.authorization.absolute(),
            expected_authorization_sha256=args.expected_authorization_sha256,
        )
    elif args.mode == "verify":
        result = acquisition.verify_completed_receipt(spec, engine)
    elif args.mode == "publish":
        result = acquisition.publish_completed_receipt(spec, engine)
    else:
        result = acquisition._seal(
            "checkpoint-transport-preflight",
            spec_id=spec.spec_id,
            amendment_sha256=spec.amendment_sha256,
            library_sha256=engine.library_sha256,
            archive_size_bytes=spec.expectation.size_bytes,
            chunk_count=spec.chunk_count,
            provider_contacted=False,
            attempt_consumed=False,
            execution_authorized=False,
            boundaries=dict(acquisition._BOUNDARIES),
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
