"""Verify compact exported scores without rerunning source fitting or physics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root: Path) -> dict:
    result_path = root / 'results' / 'result.json'
    if not result_path.exists():
        result_path = root / 'result.json'
    results = result_path.parent
    value = json.loads(result_path.read_text())
    unsigned = dict(value)
    claimed = unsigned.pop('result_sha256')
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if actual != claimed:
        raise ValueError('Result content hash mismatch')
    with (results / 'event_probabilities.csv').open() as f:
        rows = list(csv.DictReader(f))
    methods = ('joint_student', 'diagonal_student', 'trajectory_bootstrap', 'symmetric_bootstrap', 'joint_gaussian')
    families = ('relative', 'bending', 'temporal')
    if len(rows) != 5 * 16 * 99 * 3:
        raise ValueError(f'Incomplete event table: {len(rows)}')
    cells, event_labels = {}, {}
    for row in rows:
        key = (row['dlo'], row['trajectory'], row['query'], row['threshold_m'])
        p, z = float(row['probability']), int(row['event'])
        if not np.isfinite(p) or not 0 <= p <= 1 or z not in (0, 1):
            raise ValueError('Invalid probability or label')
        if key in event_labels and event_labels[key] != z:
            raise ValueError('Methods were scored against different events')
        event_labels[key] = z
        k = (row['method'], row['dlo'], row['trajectory'], row['family'])
        cells.setdefault(k, []).append((p-z)**2)
    cases = sorted({(r['dlo'],r['trajectory']) for r in rows})
    assert len(cases) == 16
    vectors = {}
    for method in methods:
        vectors[method] = np.array([np.mean([np.mean(cells[(method,dlo,name,family)]) for family in families]) for dlo,name in cases])
        recorded = value['summary']['heldout_macro'][method]['brier']
        if abs(vectors[method].mean() - recorded) > 1e-12:
            raise ValueError('Brier aggregate mismatch: ' + method)
    labels = np.array([dlo for dlo, _ in cases])
    comparisons = {}
    for method in methods[1:]:
        d = vectors['joint_student'] - vectors[method]
        rng = np.random.default_rng(20260906)
        samples = np.zeros(10000)
        by_dlo = {}
        for dlo in ('DLO4', 'DLO5'):
            x = d[labels == dlo]
            if len(x) != 8:
                raise ValueError('Wrong source-test group count')
            by_dlo[dlo] = float(x.mean())
            samples += x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)/2
        interval = np.quantile(samples,(0.0125,0.9875)).tolist()
        retained = value['paired_comparisons'][method]['heldout_macro']['brier']
        np.testing.assert_allclose(interval, retained['ci97_5_two_primary_bonferroni'],atol=1e-12,rtol=0)
        comparisons[method] = {'difference': float(d.mean()), 'ci97_5':interval,'by_dlo':by_dlo}
    supported = all(comparisons[method]['ci97_5'][1] < 0 and all(x < 0 for x in comparisons[method]['by_dlo'].values()) for method in ('diagonal_student','trajectory_bootstrap'))
    if supported != value['hypothesis_supported']:
        raise ValueError('Primary decision differs')
    return {'status':'verified', 'result_sha256':actual, 'event_rows':len(rows),
            'unique_recorded_events':len(event_labels), 'trajectories':len(cases),
            'brier': {method:float(vectors[method].mean()) for method in methods},
            'comparisons':comparisons,'hypothesis_supported':supported,
            'scope':'independent compact-score and bootstrap recomputation, not raw source fitting or simulator replay'}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    report=verify(a.root)
    encoded=json.dumps(report,indent=2)
    if a.output:
        a.output.write_text(encoded+'\n')
    print(encoded)


if __name__ == '__main__':
    main()
