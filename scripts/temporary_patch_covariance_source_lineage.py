from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected one match, found {count}")
    return text.replace(old, new, 1)


runner_path = Path("scripts/science/run_full22_covariance_only_hybrid_v1.py")
runner = runner_path.read_text(encoding="utf-8")

runner = replace_once(
    runner,
    'LOWER_HEX: Final = frozenset("0123456789abcdef")\n',
    '''LOWER_HEX: Final = frozenset("0123456789abcdef")
PREFIX_MANIFEST_CONTRACT: Final = (
    "bayesian-phystwin-full22-discrepancy-prefix-manifest"
)
PREDICTION_MANIFEST_CONTRACT: Final = (
    "bayesian-phystwin-full22-discrepancy-prediction-manifest"
)
REQUIRED_SOURCE_FILENAMES: Final = (
    "final_data.pkl",
    "inference.pkl",
    "gt_track_3d.pkl",
    "split.json",
)
PREFIX_CASE_FIELDS: Final = frozenset(
    {
        "residual_m",
        "valid",
        "geometry_m",
        "baseline_prefix_m",
        "observed_prefix_m",
        "visible_prefix",
        "gt_track_prefix_m",
        "lift_indices",
        "lift_weights",
        "fit_end",
        "train_end",
        "frame_count",
        "original_count",
        "num_surface_points",
    }
)
''',
    name="source-lineage constants",
)

runner = replace_once(
    runner,
    '''@dataclass(frozen=True, slots=True)
class PredictionCaseRecord:
    case_id: str
    path: str
    sha256: str
''',
    '''@dataclass(frozen=True, slots=True)
class PredictionCaseRecord:
    case_id: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PrefixCaseRecord:
    case_id: str
    path: str
    sha256: str
    fit_end: int
    train_end: int
    frame_count: int
    source_files_sha256: tuple[tuple[str, str], ...]
''',
    name="prefix case record",
)

marker = "\n\ndef _prediction_records(\n"
helpers = '''


def _integer_scalar_array(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar array")
    return int(array.item())


def _load_prefix_case_split(
    path: Path,
    expected_sha256: str,
    *,
    case_id: str,
) -> tuple[int, int, int]:
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"prefix case digest changed: {case_id}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PREFIX_CASE_FIELDS:
            raise ValueError(f"prefix case contract changed: {case_id}")
        fit_end = _integer_scalar_array(
            archive["fit_end"], name=f"{case_id}.fit_end"
        )
        train_end = _integer_scalar_array(
            archive["train_end"], name=f"{case_id}.train_end"
        )
        frame_count = _integer_scalar_array(
            archive["frame_count"], name=f"{case_id}.frame_count"
        )
    if not 0 < fit_end < train_end < frame_count:
        raise ValueError(f"invalid prefix split for {case_id}")
    return fit_end, train_end, frame_count


def _prefix_records(
    source_root: Path,
    *,
    expected_protocol_id: str,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[Mapping[str, object], dict[str, PrefixCaseRecord]]:
    manifest = _load_json(source_root / "prefix" / "prefix_manifest.json")
    declared_id = _literal_sha(
        manifest.get("prefix_manifest_id"),
        name="prefix_manifest_id",
        length=64,
    )
    descriptor = {
        key: value for key, value in manifest.items() if key != "prefix_manifest_id"
    }
    if declared_id != _canonical_sha256(descriptor):
        raise ValueError("prefix manifest identity changed")
    if (
        manifest.get("contract") != PREFIX_MANIFEST_CONTRACT
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != expected_protocol_id
    ):
        raise ValueError("prefix manifest lineage changed")
    boundary = _mapping(
        manifest.get("information_boundary"), name="prefix information boundary"
    )
    required_boundary = {
        "contains_fit_prefix": True,
        "contains_guard_validation_prefix": True,
        "contains_scored_future": False,
        "candidate_prediction_receives_future": False,
        "confirmation_payload_opened": False,
        "target_outcome_opened": False,
    }
    if any(boundary.get(name) is not value for name, value in required_boundary.items()):
        raise ValueError("prefix information boundary changed")
    records: dict[str, PrefixCaseRecord] = {}
    for index, raw in enumerate(
        _sequence(manifest.get("cases"), name="prefix cases")
    ):
        row = _mapping(raw, name=f"prefix cases[{index}]")
        case_id = _text(row.get("case_id"), name="case_id")
        fit_end = _integer(row.get("fit_end"), name="fit_end", minimum=1)
        train_end = _integer(row.get("train_end"), name="train_end", minimum=1)
        frame_count = _integer(
            row.get("frame_count"), name="frame_count", minimum=1
        )
        if not fit_end < train_end < frame_count:
            raise ValueError(f"invalid prefix split for {case_id}")
        if case_id in records or row.get("future_arrays_serialized") is not False:
            raise ValueError(f"invalid prefix record for {case_id}")
        source_files = _mapping(
            row.get("source_files_sha256"),
            name=f"{case_id}.source_files_sha256",
        )
        if set(source_files) != set(REQUIRED_SOURCE_FILENAMES):
            raise ValueError(f"source file roster changed for {case_id}")
        record = PrefixCaseRecord(
            case_id=case_id,
            path=_text(row.get("path"), name="path"),
            sha256=_literal_sha(row.get("sha256"), name="sha256", length=64),
            fit_end=fit_end,
            train_end=train_end,
            frame_count=frame_count,
            source_files_sha256=tuple(
                (
                    filename,
                    _literal_sha(
                        source_files.get(filename),
                        name=f"{case_id}.{filename}.sha256",
                        length=64,
                    ),
                )
                for filename in REQUIRED_SOURCE_FILENAMES
            ),
        )
        sealed_split = _load_prefix_case_split(
            source_root / "prefix" / record.path,
            record.sha256,
            case_id=case_id,
        )
        if sealed_split != (fit_end, train_end, frame_count):
            raise ValueError(f"prefix manifest and case split differ for {case_id}")
        records[case_id] = record
    if (
        len(records) != expected_case_count
        or manifest.get("case_count") != expected_case_count
    ):
        raise ValueError(f"expected {expected_case_count} prefix cases")
    return manifest, records


def _verify_public_source_files(data_root: Path, record: PrefixCaseRecord) -> None:
    case_root = data_root / record.case_id
    for filename, expected_sha256 in record.source_files_sha256:
        path = case_root / filename
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            raise ValueError(
                f"public source file differs from the seal: {record.case_id}/{filename}"
            )
'''
runner = replace_once(
    runner,
    marker,
    helpers + marker,
    name="source-lineage helper insertion",
)

old_prediction_records = '''def _prediction_records(
    source_root: Path,
    candidate_id: str,
    *,
    expected_protocol_id: str,
) -> tuple[Mapping[str, object], dict[str, PredictionCaseRecord]]:
    manifest = _load_json(
        source_root / "predictions" / candidate_id / "prediction_manifest.json"
    )
    if manifest.get("contract") != (
        "bayesian-phystwin-full22-discrepancy-prediction-manifest"
    ):
        raise ValueError(f"unexpected prediction manifest for {candidate_id}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("candidate_id") != candidate_id
        or manifest.get("protocol_id") != expected_protocol_id
    ):
        raise ValueError(f"prediction lineage changed for {candidate_id}")
    records: dict[str, PredictionCaseRecord] = {}
    for index, raw in enumerate(
        _sequence(manifest.get("case_records"), name=f"{candidate_id}.case_records")
    ):
        row = _mapping(raw, name=f"{candidate_id}.case_records[{index}]")
        case_id = _text(row.get("case_id"), name="case_id")
        if case_id in records or row.get("prediction_success") is not True:
            raise ValueError(f"invalid prediction record for {candidate_id}/{case_id}")
        records[case_id] = PredictionCaseRecord(
            case_id=case_id,
            path=_text(row.get("path"), name="path"),
            sha256=_literal_sha(row.get("sha256"), name="sha256", length=64),
        )
    if (
        len(records) != EXPECTED_CASE_COUNT
        or manifest.get("case_count") != EXPECTED_CASE_COUNT
    ):
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} cases for {candidate_id}")
    return manifest, records
'''
new_prediction_records = '''def _prediction_records(
    source_root: Path,
    candidate_id: str,
    *,
    expected_protocol_id: str,
    expected_prefix_manifest_id: str,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[Mapping[str, object], dict[str, PredictionCaseRecord]]:
    manifest = _load_json(
        source_root / "predictions" / candidate_id / "prediction_manifest.json"
    )
    declared_id = _literal_sha(
        manifest.get("prediction_artifact_sha256"),
        name=f"{candidate_id}.prediction_artifact_sha256",
        length=64,
    )
    descriptor = {
        key: value
        for key, value in manifest.items()
        if key != "prediction_artifact_sha256"
    }
    if declared_id != _canonical_sha256(descriptor):
        raise ValueError(f"prediction manifest identity changed for {candidate_id}")
    if (
        manifest.get("contract") != PREDICTION_MANIFEST_CONTRACT
        or manifest.get("schema_version") != 1
        or manifest.get("candidate_id") != candidate_id
        or manifest.get("protocol_id") != expected_protocol_id
        or manifest.get("prefix_manifest_id") != expected_prefix_manifest_id
    ):
        raise ValueError(f"prediction lineage changed for {candidate_id}")
    records: dict[str, PredictionCaseRecord] = {}
    for index, raw in enumerate(
        _sequence(manifest.get("case_records"), name=f"{candidate_id}.case_records")
    ):
        row = _mapping(raw, name=f"{candidate_id}.case_records[{index}]")
        case_id = _text(row.get("case_id"), name="case_id")
        if case_id in records or row.get("prediction_success") is not True:
            raise ValueError(f"invalid prediction record for {candidate_id}/{case_id}")
        records[case_id] = PredictionCaseRecord(
            case_id=case_id,
            path=_text(row.get("path"), name="path"),
            sha256=_literal_sha(row.get("sha256"), name="sha256", length=64),
        )
    if (
        len(records) != expected_case_count
        or manifest.get("case_count") != expected_case_count
    ):
        raise ValueError(f"expected {expected_case_count} cases for {candidate_id}")
    return manifest, records
'''
runner = replace_once(
    runner,
    old_prediction_records,
    new_prediction_records,
    name="prediction manifest validation",
)

marker = "\n\ndef _exact_reference_mean(value: object, *, case_id: str) -> np.ndarray:\n"
sealed_split = '''


def _sealed_prediction_split(
    prediction: Mapping[str, np.ndarray],
    *,
    case_id: str,
) -> tuple[int, int, int]:
    fit_end = _integer_scalar_array(
        prediction["fit_end"], name=f"{case_id}.fit_end"
    )
    train_end = _integer_scalar_array(
        prediction["train_end"], name=f"{case_id}.train_end"
    )
    frame_count = _integer_scalar_array(
        prediction["frame_count"], name=f"{case_id}.frame_count"
    )
    if not 0 < fit_end < train_end < frame_count:
        raise ValueError(f"invalid sealed prediction split for {case_id}")
    expected_counts = {
        "validation_mean_m": train_end - fit_end,
        "validation_covariance_m2": train_end - fit_end,
        "future_mean_m": frame_count - train_end,
        "future_covariance_m2": frame_count - train_end,
    }
    for name, expected_count in expected_counts.items():
        array = np.asarray(prediction[name])
        if array.ndim < 1 or array.shape[0] != expected_count:
            raise ValueError(
                f"sealed {name} length differs from the split for {case_id}"
            )
    return fit_end, train_end, frame_count
'''
runner = replace_once(
    runner,
    marker,
    sealed_split + marker,
    name="sealed prediction split helper",
)

runner = replace_once(
    runner,
    '''    discovered = _discover_source_root(source_root)
    manifests: dict[str, Mapping[str, object]] = {}
    records: dict[str, dict[str, PredictionCaseRecord]] = {}
    for candidate_id in (REFERENCE, *DONORS):
        manifests[candidate_id], records[candidate_id] = _prediction_records(
            discovered,
            candidate_id,
            expected_protocol_id=source_protocol_id,
        )
    case_ids = tuple(sorted(records[REFERENCE]))
    if any(tuple(sorted(records[donor])) != case_ids for donor in DONORS):
        raise ValueError("prediction candidate case rosters differ")
''',
    '''    discovered = _discover_source_root(source_root)
    prefix_manifest, prefix_records = _prefix_records(
        discovered,
        expected_protocol_id=source_protocol_id,
    )
    prefix_manifest_id = _literal_sha(
        prefix_manifest.get("prefix_manifest_id"),
        name="prefix_manifest_id",
        length=64,
    )
    manifests: dict[str, Mapping[str, object]] = {}
    records: dict[str, dict[str, PredictionCaseRecord]] = {}
    for candidate_id in (REFERENCE, *DONORS):
        manifests[candidate_id], records[candidate_id] = _prediction_records(
            discovered,
            candidate_id,
            expected_protocol_id=source_protocol_id,
            expected_prefix_manifest_id=prefix_manifest_id,
        )
    case_ids = tuple(sorted(prefix_records))
    if tuple(sorted(records[REFERENCE])) != case_ids or any(
        tuple(sorted(records[donor])) != case_ids for donor in DONORS
    ):
        raise ValueError("prefix and prediction case rosters differ")
''',
    name="prefix and prediction lineage load",
)

runner = replace_once(
    runner,
    "    from bayesian_phystwin.phystwin_confirmatory import _split_for_case\n",
    "",
    name="current split helper import",
)

runner = replace_once(
    runner,
    '''        fit_end, train_end, frame_count = _split_for_case(
            data_root / case_id,
            _finite(
                _mapping(protocol["cohort"], name="cohort")["fit_fraction"],
                name="fit_fraction",
            ),
        )
        if (
            int(reference_prediction["fit_end"]) != fit_end
            or int(reference_prediction["train_end"]) != train_end
            or int(reference_prediction["frame_count"]) != frame_count
        ):
            raise ValueError(f"reference split changed for {case_id}")
        data = _load_pickle(data_root / case_id / "final_data.pkl")
        baseline = np.asarray(
            _load_pickle(data_root / case_id / "inference.pkl"),
            dtype=np.float64,
        )[:frame_count]
        observed = np.asarray(data["object_points"], dtype=np.float64)[:frame_count]
        valid = _target_validity(
            np.asarray(data["object_visibilities"], dtype=bool),
            np.asarray(data["object_motions_valid"], dtype=bool),
        )[:frame_count]
''',
    '''        prefix_record = prefix_records[case_id]
        _verify_public_source_files(data_root, prefix_record)
        fit_end = prefix_record.fit_end
        train_end = prefix_record.train_end
        frame_count = prefix_record.frame_count
        if _sealed_prediction_split(
            reference_prediction,
            case_id=f"{REFERENCE}/{case_id}",
        ) != (fit_end, train_end, frame_count):
            raise ValueError(f"reference and prefix splits differ for {case_id}")
        data = _load_pickle(data_root / case_id / "final_data.pkl")
        baseline_all = np.asarray(
            _load_pickle(data_root / case_id / "inference.pkl"),
            dtype=np.float64,
        )
        observed_all = np.asarray(data["object_points"], dtype=np.float64)
        visible_all = np.asarray(data["object_visibilities"], dtype=bool)
        motion_valid_all = np.asarray(data["object_motions_valid"], dtype=bool)
        if (
            len(baseline_all) < frame_count
            or len(observed_all) < frame_count
            or len(visible_all) < frame_count
        ):
            raise ValueError(f"public scoring arrays are shorter than the seal for {case_id}")
        baseline = baseline_all[:frame_count]
        observed = observed_all[:frame_count]
        if baseline.shape[1] < observed.shape[1]:
            raise ValueError(f"baseline track roster is shorter for {case_id}")
        valid_all = _target_validity(visible_all, motion_valid_all)
        if len(valid_all) < frame_count:
            raise ValueError(f"public validity is shorter than the seal for {case_id}")
        valid = valid_all[:frame_count]
''',
    name="sealed prefix split and source-file use",
)

runner = replace_once(
    runner,
    '''            if (
                int(donor_prediction["fit_end"]) != fit_end
                or int(donor_prediction["train_end"]) != train_end
                or int(donor_prediction["frame_count"]) != frame_count
            ):
                raise ValueError(f"donor split changed for {donor}/{case_id}")
''',
    '''            if _sealed_prediction_split(
                donor_prediction,
                case_id=f"{donor}/{case_id}",
            ) != (fit_end, train_end, frame_count):
                raise ValueError(f"donor and prefix splits differ for {donor}/{case_id}")
''',
    name="donor sealed split agreement",
)

runner = replace_once(
    runner,
    '''        "source_root_identity": {
            "prediction_manifest_ids": {
''',
    '''        "source_root_identity": {
            "prefix_manifest_id": prefix_manifest_id,
            "public_source_files_verified": True,
            "prediction_manifest_ids": {
''',
    name="reported source lineage",
)
runner_path.write_text(runner, encoding="utf-8")


test_path = Path("tests/test_full22_covariance_only_hybrid_v1.py")
tests = test_path.read_text(encoding="utf-8")
addition = r'''


def _write_prefix_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source_root = tmp_path / "source"
    data_root = tmp_path / "data"
    case_id = "case-0"
    protocol_id = "a" * 64
    prefix_case = source_root / "prefix" / "cases" / f"{case_id}.npz"
    prefix_case.parent.mkdir(parents=True)
    np.savez_compressed(
        prefix_case,
        residual_m=np.zeros((5, 2, 3), dtype=np.float64),
        valid=np.ones((5, 2), dtype=bool),
        geometry_m=np.zeros((2, 3), dtype=np.float64),
        baseline_prefix_m=np.zeros((5, 2, 3), dtype=np.float64),
        observed_prefix_m=np.zeros((5, 2, 3), dtype=np.float64),
        visible_prefix=np.ones((5, 2), dtype=bool),
        gt_track_prefix_m=np.zeros((5, 2, 3), dtype=np.float64),
        lift_indices=np.zeros((0, 4), dtype=np.int64),
        lift_weights=np.zeros((0, 4), dtype=np.float64),
        fit_end=np.asarray(2, dtype=np.int64),
        train_end=np.asarray(5, dtype=np.int64),
        frame_count=np.asarray(9, dtype=np.int64),
        original_count=np.asarray(2, dtype=np.int64),
        num_surface_points=np.asarray(2, dtype=np.int64),
    )
    case_root = data_root / case_id
    case_root.mkdir(parents=True)
    source_files: dict[str, str] = {}
    for index, filename in enumerate(MODULE.REQUIRED_SOURCE_FILENAMES):
        path = case_root / filename
        path.write_bytes(f"sealed-{index}".encode("ascii"))
        source_files[filename] = MODULE._file_sha256(path)
    descriptor: dict[str, object] = {
        "contract": MODULE.PREFIX_MANIFEST_CONTRACT,
        "schema_version": 1,
        "protocol_id": protocol_id,
        "source_archives": {},
        "case_count": 1,
        "cases": [
            {
                "case_id": case_id,
                "path": f"cases/{case_id}.npz",
                "sha256": MODULE._file_sha256(prefix_case),
                "fit_end": 2,
                "train_end": 5,
                "frame_count": 9,
                "track_count": 2,
                "future_arrays_serialized": False,
                "source_files_sha256": source_files,
            }
        ],
        "information_boundary": {
            "contains_fit_prefix": True,
            "contains_guard_validation_prefix": True,
            "contains_scored_future": False,
            "candidate_prediction_receives_future": False,
            "confirmation_payload_opened": False,
            "target_outcome_opened": False,
        },
    }
    descriptor["prefix_manifest_id"] = MODULE._canonical_sha256(descriptor)
    MODULE._write_json(source_root / "prefix" / "prefix_manifest.json", descriptor)
    return source_root, data_root, case_id, protocol_id


def test_prefix_manifest_binds_split_case_and_public_source_files(
    tmp_path: Path,
) -> None:
    source_root, data_root, case_id, protocol_id = _write_prefix_fixture(tmp_path)

    manifest, records = MODULE._prefix_records(
        source_root,
        expected_protocol_id=protocol_id,
        expected_case_count=1,
    )
    MODULE._verify_public_source_files(data_root, records[case_id])

    assert manifest["prefix_manifest_id"]
    assert (records[case_id].fit_end, records[case_id].train_end) == (2, 5)


def test_prefix_manifest_rejects_identity_tampering(tmp_path: Path) -> None:
    source_root, _, _, protocol_id = _write_prefix_fixture(tmp_path)
    path = source_root / "prefix" / "prefix_manifest.json"
    payload = dict(MODULE._load_json(path))
    payload["prefix_manifest_id"] = "0" * 64
    MODULE._write_json(path, payload)

    with pytest.raises(ValueError, match="prefix manifest identity"):
        MODULE._prefix_records(
            source_root,
            expected_protocol_id=protocol_id,
            expected_case_count=1,
        )


def test_public_source_file_drift_fails_closed(tmp_path: Path) -> None:
    source_root, data_root, case_id, protocol_id = _write_prefix_fixture(tmp_path)
    _, records = MODULE._prefix_records(
        source_root,
        expected_protocol_id=protocol_id,
        expected_case_count=1,
    )
    (data_root / case_id / "inference.pkl").write_bytes(b"changed")

    with pytest.raises(ValueError, match="public source file differs"):
        MODULE._verify_public_source_files(data_root, records[case_id])


def _sealed_prediction(
    *,
    fit_end: int = 2,
    train_end: int = 5,
    frame_count: int = 9,
) -> dict[str, np.ndarray]:
    validation_count = max(0, train_end - fit_end)
    future_count = max(0, frame_count - train_end)
    return {
        "fit_end": np.asarray(fit_end, dtype=np.int64),
        "train_end": np.asarray(train_end, dtype=np.int64),
        "frame_count": np.asarray(frame_count, dtype=np.int64),
        "validation_mean_m": np.zeros((validation_count, 2, 3)),
        "validation_covariance_m2": np.zeros((validation_count, 2, 3, 3)),
        "future_mean_m": np.zeros((future_count, 2, 3)),
        "future_covariance_m2": np.zeros((future_count, 2, 3, 3)),
    }


def test_sealed_prediction_split_binds_array_lengths() -> None:
    prediction = _sealed_prediction()

    assert MODULE._sealed_prediction_split(prediction, case_id="case") == (2, 5, 9)
    prediction["future_mean_m"] = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match="length differs"):
        MODULE._sealed_prediction_split(prediction, case_id="case")
'''
if "def test_prefix_manifest_binds_split_case" in tests:
    raise SystemExit("source-lineage tests already exist")
test_path.write_text(tests + addition, encoding="utf-8")


doc_path = Path("docs/full22_covariance_only_hybrid_v1.md")
documentation = doc_path.read_text(encoding="utf-8")
paragraph = (
    "\nThe evaluator consumes the exact split sealed by the historical prefix "
    "manifest. It independently verifies that manifest's content identity, "
    "each prefix-case archive, every prediction-manifest binding, and the "
    "SHA-256 of all four public source files before opening scoring arrays. "
    "It never recomputes the historical split with current helper code.\n"
)
if "never recomputes the historical split" not in documentation:
    doc_path.write_text(documentation + paragraph, encoding="utf-8")
