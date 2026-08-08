#!/usr/bin/env python3
"""Apply the reviewed irreversible-array ownership patch exactly once."""

from __future__ import annotations

import gzip
import hashlib
import subprocess
from pathlib import Path

BASE_REVISION = "22983ff291066f0c2c93a68a7eda6abba5bf5156"
PATCH_PATH = Path("scripts/_one_shot_immutable_arrays.patch.gz")
PATCH_SHA256 = "7998bab2593e2747fa72e61c660a6eca59efab220a3a2e42b4fc0f4bf4607fba"
WORKFLOW_PATH = Path(".github/workflows/_one_shot_apply_immutable_arrays.yml")
SCRIPT_PATH = Path("scripts/_one_shot_apply_immutable_arrays.py")
EXPECTED_FILES = {
    "CHANGELOG.md": "c685ecb4f6264a0a5fdc64ec2a87bbc015829ef7d69c7f11fc77331186d7d050",
    "src/bayesian_phystwin/_canonical_contracts.py": "cad21625ab398a30c570488554d233bb0132c0988fb12ef0a8ec12e24e72ca90",
    "src/bayesian_phystwin/_gauge_aware_contracts.py": "b9a818118300cf967b096f7790d61a97674b95ac8b27ecb20c9f8c13c9737816",
    "src/bayesian_phystwin/observation_belief.py": "2bf80db83500f02c98c6bd962a6c0a7f91e39d2fd0cd0460c42c63a6ccf388eb",
    "src/bayesian_phystwin/physical_linearization.py": "18953f5f9684a3abc1c9c5f9734b5f99e3bcf505e9cf4f6ee177154e20d0e416",
    "tests/test_gauge_aware_belief.py": "16a495d6ccb4337385562433ed096a2f5fc046efafc4b2c5e7ad5afcbe5e2cb5",
    "tests/test_observation_belief.py": "77b1eee956d7324ef5a57edd55435aeba4d9902e9943ca1c43d76f6397ca706c",
    "tests/test_prior_aware_and_linearization.py": "bb7141ef2b4e6b46dc5b2fb1686d162b5c857182af524bb48761208e6ce8162b",
    "tests/test_prospective_prob4d_update.py": "bbe3a96a35c956fef770566def89d7d1d6130a8a5d2ab8bb49eee8a647777127",
}


def _run(*args: str, input_bytes: bytes | None = None) -> str:
    completed = subprocess.run(
        args,
        check=True,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.decode("utf-8").strip()


def main() -> int:
    parent = _run("git", "rev-parse", "HEAD^")
    if parent != BASE_REVISION:
        raise SystemExit(f"unexpected helper parent: {parent}")

    patch = gzip.decompress(PATCH_PATH.read_bytes())
    if hashlib.sha256(patch).hexdigest() != PATCH_SHA256:
        raise SystemExit("embedded patch digest changed")

    _run("git", "apply", "--check", "--whitespace=error-all", "-", input_bytes=patch)
    _run("git", "apply", "--whitespace=error-all", "-", input_bytes=patch)

    for path, expected in EXPECTED_FILES.items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"post-apply digest mismatch for {path}: {actual}")

    WORKFLOW_PATH.unlink()
    SCRIPT_PATH.unlink()
    PATCH_PATH.unlink()

    changed = set(_run("git", "diff", "--name-only", "HEAD").splitlines())
    expected_changed = set(EXPECTED_FILES) | {
        str(WORKFLOW_PATH),
        str(SCRIPT_PATH),
        str(PATCH_PATH),
    }
    if changed != expected_changed:
        raise SystemExit(
            f"unexpected final patch paths: missing={sorted(expected_changed - changed)}, "
            f"extra={sorted(changed - expected_changed)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
