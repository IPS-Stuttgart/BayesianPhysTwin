from __future__ import annotations

import json
import subprocess
from pathlib import Path


TEST_BLOCK = r'''


def _converged_config() -> PriorAwareGaugeConfigV1:
    return replace(
        _exhausted_config(),
        maximum_iterations=100,
        convergence_tolerance=1.0e-12,
    )


def test_tree_sparse_structured_v2_admits_converged_result() -> None:
    batch, tree = _tree_fixture()
    result = update_sparse_prior_aware_gauge_belief_structured_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    assert result.inference_admissible
    assert result.diagnostics["strict_admission_passed"] is True
    assert result.diagnostics["strict_admission_reason"] == "strict-admission-passed"


def test_tree_sparse_structured_v2_preserves_underlying_rejection() -> None:
    batch, tree = _tree_fixture()
    config = replace(_converged_config(), maximum_state_update_m=1.0e-12)
    result = update_sparse_prior_aware_gauge_belief_structured_v2(
        batch,
        tree,
        config=config,
    )
    assert not result.inference_admissible
    assert result.diagnostics["strict_admission_reason"] == (
        "underlying-inference-rejected"
    )
    assert result.diagnostics["underlying_inference_admissible"] is False


def test_tree_sparse_structured_v2_rejects_invalid_argument_types() -> None:
    batch, tree = _tree_fixture()
    with pytest.raises(TypeError, match="batch must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(object(), tree)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gauge must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(batch, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(
            batch,
            tree,
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="admission_config must"):
        update_sparse_prior_aware_gauge_belief_structured_v2(
            batch,
            tree,
            admission_config=object(),  # type: ignore[arg-type]
        )


def test_dense_sparse_v2_rejects_structured_fallback_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, tree = _tree_fixture()
    config = _exhausted_config()
    historical = update_sparse_prior_aware_gauge_belief(batch, tree, config=config)
    structured = update_sparse_prior_aware_gauge_belief_structured(
        batch,
        tree,
        config=config,
    )
    assert historical.inference_admissible
    monkeypatch.setattr(
        strict_v2,
        "update_sparse_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: historical,
    )
    monkeypatch.setattr(
        strict_v2,
        "_sparse_fallback_result",
        lambda *_args, **_kwargs: structured,
    )
    with pytest.raises(RuntimeError, match="returned a structured result"):
        update_sparse_prior_aware_gauge_belief_v2(batch, tree, config=config)
'''


def update_source() -> None:
    path = Path("src/bayesian_phystwin/prior_aware_gauge_belief_v2.py")
    text = path.read_text(encoding="utf-8")
    if "from typing import Final, TypeAlias\n" not in text:
        text = text.replace(
            "from typing import Final\n",
            "from typing import Final, TypeAlias\n",
            1,
        )
    text = text.replace(
        "GaugeDesignV1 = SparseGaugeDesignV1 | TreeSparseGaugeDesignV1\n"
        "AdmissionInputResult = GaugeAwareBeliefResult | StructuredGaugeAwareBeliefResultV1\n",
        "GaugeDesignV1: TypeAlias = SparseGaugeDesignV1 | TreeSparseGaugeDesignV1\n"
        "AdmissionInputResult: TypeAlias = (\n"
        "    GaugeAwareBeliefResult | StructuredGaugeAwareBeliefResultV1\n"
        ")\n",
        1,
    )
    path.write_text(text, encoding="utf-8")


def update_tests() -> None:
    path = Path("tests/test_claim_bearing_strict_admission.py")
    text = path.read_text(encoding="utf-8")
    if "import pytest\n" not in text:
        text = text.replace(
            "import numpy as np\n\n"
            "import bayesian_phystwin.prospective_prob4d_update as prospective_update\n",
            "import numpy as np\n"
            "import pytest\n\n"
            "import bayesian_phystwin.prior_aware_gauge_belief_v2 as strict_v2\n"
            "import bayesian_phystwin.prospective_prob4d_update as prospective_update\n",
            1,
        )
    if "update_sparse_prior_aware_gauge_belief_structured," not in text:
        text = text.replace(
            "    update_sparse_prior_aware_gauge_belief,\n",
            "    update_sparse_prior_aware_gauge_belief,\n"
            "    update_sparse_prior_aware_gauge_belief_structured,\n",
            1,
        )
    if "def _converged_config()" not in text:
        text = text.rstrip() + TEST_BLOCK + "\n"
    path.write_text(text, encoding="utf-8")


def update_manifest() -> None:
    path = Path(".github/quality/test-suites.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    additions = [
        "tests/test_prior_aware_gauge_belief_v2.py",
        "tests/test_claim_bearing_strict_admission.py",
    ]
    for suite_name in ("stable-core-coverage", "core-contracts"):
        suite = manifest["suites"][suite_name]
        for item in additions:
            while item in suite:
                suite.remove(item)
        anchor = suite.index("tests/test_prior_aware_likelihood_conformance.py") + 1
        suite[anchor:anchor] = additions
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def commit_generated_source() -> None:
    subprocess.run(
        ["git", "config", "user.name", "github-actions[bot]"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "add",
            ".github/quality/test-suites.json",
            "src/bayesian_phystwin/prior_aware_gauge_belief_v2.py",
            "tests/test_claim_bearing_strict_admission.py",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Complete strict-admission coverage [finalize-pr349]",
        ],
        check=True,
    )


def main() -> None:
    update_source()
    update_tests()
    update_manifest()
    commit_generated_source()


if __name__ == "__main__":
    main()
