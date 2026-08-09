from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
SOURCE = ROOT / "_contract_source"
PROB4D_REVISION = "b2953319e9b7afea04013c214c502b38c5a83489"
BUNDLE_SHA256 = "a62c693a14c227daa1f4c8db850e691a1d0081df0c853cf0174c33d0b8504ce9"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected one exact replacement target")
    write(path, text.replace(old, new))


def insert_test_after(suite: list[str], anchor: str, value: str) -> None:
    if value in suite:
        return
    try:
        position = suite.index(anchor)
    except ValueError as error:
        raise RuntimeError(f"test-suite anchor {anchor!r} is missing") from error
    suite.insert(position + 1, value)


if not SOURCE.is_dir() or not (SOURCE / ".git").exists():
    raise RuntimeError("exact Prob4D contract source checkout is missing")

# Copy the neutral helper and byte-identical data-only corpus.
shutil.copy2(
    SOURCE / "src/prob4d/observation_contract_bundle.py",
    ROOT / "src/bayesian_phystwin/observation_contract_bundle.py",
)
destination_bundle = ROOT / "src/bayesian_phystwin/contract_data/observation_belief_v1"
if destination_bundle.exists():
    shutil.rmtree(destination_bundle)
shutil.copytree(
    SOURCE / "src/prob4d/contract_data/observation_belief_v1",
    destination_bundle,
)
shutil.copy2(
    SOURCE / "docs/observation-contract-conformance.md",
    ROOT / "docs/observation-contract-conformance.md",
)

# Keep the estimator implementation independent, but make the serialized loader
# enforce the normative closed descriptor/member and exact-dtype rules.
observation_path = "src/bayesian_phystwin/observation_belief.py"
observation_text = read(observation_path)
constant_anchor = (
    'OBSERVATION_BELIEF_SCHEMA = "phys4d.observation_belief"\n'
    "OBSERVATION_BELIEF_VERSION = 1\n"
)
constant_replacement = constant_anchor + '''

_OBSERVATION_BELIEF_SERIALIZED_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "case_id",
        "stream_id",
        "causal_frame_stop",
        "view_names",
        "window_names",
        "factor_names",
        "source_repository",
        "source_revision",
        "source_artifact_sha256",
        "metadata",
        "artifact_id",
    }
)
_OBSERVATION_BELIEF_ARRAY_DTYPES = {
    "declared_frame_ids": np.dtype(np.int64),
    "mean_xyz_m": np.dtype(np.float64),
    "frame_ids": np.dtype(np.int64),
    "entity_ids": np.dtype(np.int64),
    "view_indices": np.dtype(np.int64),
    "window_indices": np.dtype(np.int64),
    "correlation_group_ids": np.dtype(np.int64),
    "factor_group_ids": np.dtype(np.int64),
    "prior_reliability": np.dtype(np.float64),
    "association_probability": np.dtype(np.float64),
    "local_covariance_m2": np.dtype(np.float64),
    "low_rank_factor_m": np.dtype(np.float64),
    "group_ids": np.dtype(np.int64),
    "group_prior_nominal_probability": np.dtype(np.float64),
    "group_composite_weight": np.dtype(np.float64),
}
'''
if observation_text.count(constant_anchor) != 1:
    raise RuntimeError(f"{observation_path}: contract constant anchor changed")
observation_text = observation_text.replace(constant_anchor, constant_replacement)

new_loader = '''def load_observation_belief(path: str | Path) -> ObservationBeliefV1:
    """Load and fully revalidate an ``ObservationBeliefV1`` artifact."""

    with np.load(path, allow_pickle=False) as archive:
        required_members = {
            "descriptor_json",
            *_OBSERVATION_BELIEF_ARRAY_DTYPES,
        }
        members = set(archive.files)
        missing = required_members - members
        extra = members - required_members
        if missing or extra:
            raise ValueError(
                "observation artifact members changed; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        descriptor_member = np.asarray(archive["descriptor_json"])
        if descriptor_member.shape != ():
            raise ValueError("descriptor_json must be a scalar array")
        raw_descriptor = descriptor_member.item()
        if isinstance(raw_descriptor, bytes):
            try:
                raw_descriptor = raw_descriptor.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("descriptor_json is not valid UTF-8") from error
        if type(raw_descriptor) is not str:
            raise ValueError("descriptor_json must contain a string")
        try:
            descriptor = json.loads(raw_descriptor)
        except (TypeError, ValueError) as error:
            raise ValueError("descriptor_json is not valid JSON") from error
        if not isinstance(descriptor, dict):
            raise ValueError("observation descriptor must be a JSON object")
        if set(descriptor) != _OBSERVATION_BELIEF_SERIALIZED_DESCRIPTOR_FIELDS:
            raise ValueError("observation descriptor fields changed")
        if descriptor.get("schema_name") != OBSERVATION_BELIEF_SCHEMA:
            raise ValueError("unsupported observation-belief schema")
        version = genuine_integer(
            descriptor.get("schema_version"),
            name="observation-belief schema_version",
            minimum=0,
        )
        if version != OBSERVATION_BELIEF_VERSION:
            raise ValueError("unsupported observation-belief version")

        arrays: dict[str, np.ndarray] = {}
        for name, expected_dtype in _OBSERVATION_BELIEF_ARRAY_DTYPES.items():
            values = np.asarray(archive[name])
            if values.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must have exact dtype {expected_dtype.name}"
                )
            arrays[name] = values

    belief = ObservationBeliefV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **arrays,
    )
    expected = descriptor["artifact_id"]
    _validate_sha256(expected, name="artifact_id")
    if belief.artifact_id != expected:
        raise ValueError("observation artifact digest does not match its payload")
    return belief'''
loader_pattern = re.compile(
    r"def load_observation_belief\(path: str \| Path\) -> ObservationBeliefV1:\n"
    r".*?\n\n\n__all__ = \[",
    flags=re.DOTALL,
)
observation_text, replacement_count = loader_pattern.subn(
    lambda _: new_loader + "\n\n\n__all__ = [",
    observation_text,
    count=1,
)
if replacement_count != 1:
    raise RuntimeError(f"{observation_path}: current loader target changed")
write(observation_path, observation_text)

# Package the corpus in wheels and source distributions.
replace_once(
    "pyproject.toml",
    'bayesian_phystwin = ["py.typed"]',
    '''bayesian_phystwin = [
    "py.typed",
    "contract_data/observation_belief_v1/*.json",
    "contract_data/observation_belief_v1/vectors/*.json",
]''',
)
manifest_path = "MANIFEST.in"
manifest = read(manifest_path)
for addition in (
    "include docs/observation-contract-conformance.md",
    "recursive-include src/bayesian_phystwin/contract_data/observation_belief_v1 *.json",
):
    if addition not in manifest.splitlines():
        manifest += addition + "\n"
write(manifest_path, manifest)

# Register the focused conformance test in every relevant stable suite.
suite_path = ".github/quality/test-suites.json"
suite_payload = json.loads(read(suite_path))
for suite_name in ("stable-core-coverage", "core-contracts", "provider-contract"):
    insert_test_after(
        suite_payload["suites"][suite_name],
        "tests/test_observation_belief.py",
        "tests/test_observation_contract_bundle.py",
    )
write(suite_path, json.dumps(suite_payload, indent=2) + "\n")

# Verify the same installed bundle identity across all three wheels.
golden_path = "scripts/run_three_repository_golden_path.sh"
golden_text = read(golden_path)
golden_anchor = '"${TEST_VENV}/bin/python" -m pip check\n'
python_check = (
    'from importlib import import_module; '
    f'expected="{BUNDLE_SHA256}"; '
    'names=("prob4d.observation_contract_bundle",'
    '"bayesian_phystwin.observation_contract_bundle",'
    '"causal4d.observation_contract_bundle"); '
    'observed={name:import_module(name).observation_contract_bundle_manifest()'
    '["bundle_sha256"] for name in names}; '
    'assert set(observed.values())=={expected}, observed; '
    'print(f"verified shared observation-contract bundle {expected}")'
)
golden_addition = (
    golden_anchor
    + '\nenv -u PYTHONPATH PYTHONNOUSERSITE=1 \\\n'
    + '  "${TEST_VENV}/bin/python" -I -c '
    + repr(python_check)
    + "\n"
)
if golden_text.count(golden_anchor) != 1:
    raise RuntimeError(f"{golden_path}: pip-check anchor changed")
write(golden_path, golden_text.replace(golden_anchor, golden_addition))

# Adapt the canonical producer test to this independent consumer implementation.
source_test = (SOURCE / "tests/test_observation_contract_bundle.py").read_text(
    encoding="utf-8"
)
source_test = re.sub(
    r"from prob4d\.observation_contract import \(.*?\)\n"
    r"from prob4d\.observation_validation import .*?\n",
    '''from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    array_sha256,
    load_observation_belief,
    save_observation_belief,
)
''',
    source_test,
    count=1,
    flags=re.DOTALL,
)
source_test = source_test.replace(
    "from prob4d.observation_contract_bundle import (",
    "from bayesian_phystwin.observation_contract_bundle import (",
)
source_test = source_test.replace(
    "ObservationBeliefExportV1", "ObservationBeliefV1"
)
source_test = source_test.replace(
    "save_observation_belief_export", "save_observation_belief"
)
source_test = source_test.replace(
    "load_observation_belief_export", "load_observation_belief"
)
source_test = source_test.replace("belief.arrays().items()", "belief._arrays().items()")
source_test += '''


def test_bundle_report_and_unknown_names_fail_closed(capsys) -> None:
    from bayesian_phystwin.observation_contract_bundle import main

    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle_sha256"] == OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256
    with pytest.raises(KeyError):
        observation_contract_vector("unknown")
    with pytest.raises(KeyError):
        invalid_observation_contract_vector("unknown")


def test_loader_rejects_non_scalar_descriptor(tmp_path: Path) -> None:
    vector = observation_contract_vector("minimal")
    payload = dict(vector.descriptor)
    payload["artifact_id"] = vector.expected_artifact_id
    path = tmp_path / "descriptor-array.npz"
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            [json.dumps(payload, sort_keys=True, separators=(",", ":"))]
        ),
        **vector.arrays,
    )
    with pytest.raises(ValueError, match="scalar"):
        load_observation_belief(path)
'''
write("tests/test_observation_contract_bundle.py", source_test)

# Remove the temporary nested checkout from the product tree.
shutil.rmtree(SOURCE)
print(
    json.dumps(
        {
            "prob4d_revision": PROB4D_REVISION,
            "bundle_sha256": BUNDLE_SHA256,
            "status": "prepared",
        },
        sort_keys=True,
    )
)
