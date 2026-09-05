"""Independent verifier: does not import BayesianPhysTwin or its audit code."""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
a = parser.parse_args()
root = a.root
result = json.loads((root/'result.json').read_text())
nd = NormalDist()
max_probability_error = 0.0
max_variance_error = 0.0
objects = []
for file in sorted((root/'predictions').glob('*.npz')):
    with np.load(file, allow_pickle=False) as f:
        source = f['source_query_errors']
        truth = f['source_query_truth']
        mean = f['target_query_mean']
        labels = f['labels']
        names = list(f['query_names'])
        empirical = np.mean(source**2, axis=0)
        probabilities = []
        for j, name in enumerate(names):
            absolute = name in ('sensor_imbalance','horizontal_balance','vertical_balance')
            threshold = float(np.quantile(np.abs(truth[:,j]) if absolute else truth[:,j], 0.9))
            assert np.isclose(threshold, f['thresholds'][j], atol=1e-12)
            sd = math.sqrt(max(float(empirical[j]), 1e-12))
            p = []
            for m in mean[:,j]:
                value = 1-nd.cdf((threshold-m)/sd)
                if absolute:
                    value += nd.cdf((-threshold-m)/sd)
                p.append(value)
            probabilities.append(np.clip(p, 1e-9, 1-1e-9))
        p = np.column_stack(probabilities)
        max_probability_error = max(max_probability_error, float(np.max(abs(p-f['prob_full_v6']))))
        max_variance_error = max(max_variance_error, float(np.max(abs(empirical-f['variances'][:,0]))))
        brier = float(np.mean((p-labels)**2))
        full_brier = float(np.mean((f['prob_full_v6']-labels)**2))
        n = labels.size
        count = max(1, math.floor(.4*n))
        def risk(pr):
            order = np.lexsort((np.arange(n), pr.ravel()))[:count]
            return float(labels.ravel()[order].mean())
        objects.append({'object_id':file.stem, 'empirical_brier':brier, 'full_brier':full_brier,
                        'empirical_risk40':risk(p), 'full_risk40':risk(f['prob_full_v6'])})
assert len(objects) == result['object_count'] == 92
assert max_probability_error < 1e-10
assert max_variance_error < 1e-10
assert np.isclose(np.mean([o['empirical_brier'] for o in objects]), result['metrics']['full_v6']['brier'])
verification = {'status':'passed', 'objects':len(objects), 'registered_queries':len(objects)*5,
                'max_probability_difference':max_probability_error,
                'max_variance_difference':max_variance_error,
                'equal_object_empirical_brier':float(np.mean([o['empirical_brier'] for o in objects])),
                'max_matched40_risk_difference':max(abs(o['empirical_risk40']-o['full_risk40']) for o in objects),
                'does_not_import_study_code':True}
(root/'independent-verification.json').write_text(json.dumps(verification, indent=2)+'\n')
print(json.dumps(verification, indent=2))
