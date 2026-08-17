from __future__ import annotations

from typing import Any, cast

import pytest

from bayesian_phystwin.repository_provenance import (
    RepositoryState,
    normalize_github_repository,
    validate_revision,
)


class _StringableRevision:
    def __str__(self) -> str:
        return "a" * 40


@pytest.mark.parametrize(
    "revision",
    (
        "A" * 40,
        " " + "a" * 40,
        "a" * 40 + " ",
        cast(Any, _StringableRevision()),
    ),
)
def test_revision_requires_an_exact_lowercase_string(revision: str) -> None:
    with pytest.raises(ValueError, match="exact lowercase"):
        validate_revision(revision)


def test_revision_accepts_exact_lowercase_commit() -> None:
    revision = "0123456789abcdef" * 2 + "01234567"

    assert validate_revision(revision) == revision


@pytest.mark.parametrize(
    ("remote", "expected"),
    (
        (
            "https://github.com/IPS-Stuttgart/BayesianPhysTwin.git",
            "IPS-Stuttgart/BayesianPhysTwin",
        ),
        (
            "git@github.com:FlorianPfaff/Bayesian-PhysTwin.git",
            "FlorianPfaff/Bayesian-PhysTwin",
        ),
        (
            "ssh://git@github.com:22/owner/repository_name.git",
            "owner/repository_name",
        ),
    ),
)
def test_remote_normalization_accepts_exact_github_transports(
    remote: str,
    expected: str,
) -> None:
    assert normalize_github_repository(remote) == expected


@pytest.mark.parametrize(
    "remote",
    (
        "https://evil.example/github.com/owner/repository.git",
        "https://notgithub.com/owner/repository.git",
        "https://github.com.evil.example/owner/repository.git",
        "https://user@github.com/owner/repository.git",
        "https://github.com:443/owner/repository.git",
        "https://github.com/owner/repository.git?ref=main",
        "https://github.com/owner/repository.git#fragment",
        "http://github.com/owner/repository.git",
        "ssh://other@github.com/owner/repository.git",
        "ssh://git@github.com:2222/owner/repository.git",
        "git@github.com:owner/repository.git#fragment",
        " https://github.com/owner/repository.git",
        "https://github.com/owner/repository.git ",
        cast(Any, _StringableRevision()),
    ),
)
def test_remote_normalization_rejects_spoofed_or_noncanonical_urls(
    remote: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_github_repository(remote)


@pytest.mark.parametrize(
    "repository",
    (
        " owner/repository",
        "owner/repository ",
        "owner name/repository",
        "-owner/repository",
        "owner-/repository",
        "owner/repository name",
        "owner/..",
        "owner/repository/extra",
        cast(Any, _StringableRevision()),
    ),
)
def test_repository_state_rejects_noncanonical_owner_name(
    repository: str,
) -> None:
    with pytest.raises(ValueError):
        RepositoryState(
            repository=repository,
            revision="a" * 40,
            dirty=False,
            role="primary",
        )


def test_repository_state_retains_exact_canonical_identity() -> None:
    state = RepositoryState(
        repository="IPS-Stuttgart/BayesianPhysTwin",
        revision="a" * 40,
        dirty=False,
        role="primary",
    )

    assert state.as_dict() == {
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "revision": "a" * 40,
        "dirty": False,
        "role": "primary",
    }
