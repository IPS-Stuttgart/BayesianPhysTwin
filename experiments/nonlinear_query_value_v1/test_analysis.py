"""Software checks only: synthetic tests are not real-data evidence."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import analysis as a
from evaluate import ArrayUnpickler, Inputs, evaluate


class Tests(unittest.TestCase):
    def test_query_roster(self):
        w,info=a.queries()
        self.assertEqual(w.shape,(27,8))
        self.assertEqual(sum(x['heldout'] for x in info),13)
        np.testing.assert_array_equal(w.sum(1),0)
    def test_geometry_translation_rotation(self):
        w,_=a.queries(); rng=np.random.default_rng(7)
        x=rng.normal(size=(2,4,8,3)); q,_=np.linalg.qr(rng.normal(size=(3,3)))
        np.testing.assert_allclose(a.geometric(x,w),a.geometric(x@q+np.array([2,-5,4]),w),atol=1e-12)
    def test_gaussian_quadratic_moment(self):
        rng=np.random.default_rng(9); w,_=a.queries()
        mu=rng.normal(size=(8,3)); n=120000
        noise=rng.normal(size=(n,8,3))*.1
        c=np.broadcast_to(np.eye(3)*.01,(8,3,3)).copy()
        expected=a.geometric(mu,w)+a.corrections(c,w)
        np.testing.assert_allclose(a.geometric(mu+noise,w).mean(0),expected,rtol=.002,atol=.003)
    def test_common_translation_cancels(self):
        w,_=a.queries(); c=np.broadcast_to(np.eye(3)*.02,(2,4,8,3,3)).copy()
        r=np.tile(np.eye(3),(8,8))
        np.testing.assert_allclose(a.corrections(c,w,r),0,atol=1e-12)
    def test_diagonal_trace_parity(self):
        rng=np.random.default_rng(1); b=rng.normal(size=(3,5,8,3,3)); c=b@b.swapaxes(-1,-2)
        d=np.zeros_like(c); ids=np.arange(3);d[...,ids,ids]=np.diagonal(c,axis1=-2,axis2=-1)
        w,_=a.queries()
        np.testing.assert_allclose(a.corrections(c,w),a.corrections(d,w),atol=1e-12)
    def test_coupling_psd_marginal(self):
        rng=np.random.default_rng(3); e=rng.normal(size=(8,20,8,3)); c=np.broadcast_to(np.eye(3),(8,20,8,3,3)).copy()
        r=a.fit_coupling(e,c)
        self.assertGreater(np.linalg.eigvalsh(r).min(),0)
        for i in range(8):np.testing.assert_allclose(r[3*i:3*i+3,3*i:3*i+3],np.eye(3),atol=1e-12)
    def test_square_loss_identity(self):
        rng=np.random.default_rng(3);r=rng.normal(size=100);v=rng.uniform(size=100)
        np.testing.assert_allclose(r*r-(r-v)**2,2*r*v-v*v,atol=1e-12)
    def test_reject_indefinite(self):
        with self.assertRaises(ValueError):a.matrix_power(np.diag([1,-1,1]),.5)
    def test_end_to_end_shapes_means_immutable(self):
        rng=np.random.default_rng(11); sm=rng.normal(size=(8,10,8,3))*.02; sy=sm+rng.normal(size=sm.shape)*.002
        tm=rng.normal(size=(14,10,8,3))*.02; old=tm.copy(); sc=np.broadcast_to(np.eye(3)*1e-5,(8,10,8,3,3)).copy();tc=np.broadcast_to(np.eye(3)*1e-5,(14,10,8,3,3)).copy()
        for mode in (False,True):
            arms,scale,meta=a.build_readouts(sm,sy,sc,sm[:,0],tm,tc,tm[:,0],heldout_mode=mode)
            self.assertEqual(len(arms),9)
            self.assertTrue(all(x.shape==(14,10,27) for x in arms.values()))
            self.assertEqual(scale.shape,(27,))
        np.testing.assert_array_equal(tm,old)
    def test_bootstrap_paired_unit(self):
        r=a.paired_bootstrap(np.ones((2,14)),np.ones((2,14))*2,reps=100)
        self.assertEqual(r['bootstrap95'],[-1.,-1.])
        self.assertEqual(r['wins'],28)


class DriverTests(unittest.TestCase):
    def test_restricted_loader_rejects_non_array_global(self):
        with self.assertRaises(pickle.UnpicklingError):
            ArrayUnpickler(io.BytesIO(pickle.dumps(Path('/tmp/no-read')))).load()

    def test_synthetic_two_object_driver_and_identity_tamper(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            root = work / 'runs' / '33361441865-1'
            records = []
            expected = {}
            def record(path, metadata=None):
                blob = path.read_bytes()
                value = dict(path=str(path), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
                if metadata is not None:
                    value['metadata'] = metadata
                records.append(value)
            for dlo in ('DLO4', 'DLO5'):
                for stage, count in (('source', 8), ('target', 14)):
                    folder = root / f'{dlo.lower()}-{stage}'
                    folder.mkdir(parents=True)
                    names = [f'{i}.pkl' for i in range(count)]
                    truth = rng.normal(0.1, 0.02, (count, 500, 12, 3)).astype(np.float32)
                    manifests = {}
                    for i, name in enumerate(names):
                        path = folder / name
                        path.write_bytes(pickle.dumps(truth[i].transpose(0, 2, 1), protocol=4))
                        blob = path.read_bytes()
                        manifests[name] = dict(path=str(path), size_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
                    mean = truth[:, 2:].astype(np.float64) + rng.normal(0, 0.002, (count, 498, 12, 3))
                    cov = np.broadcast_to(np.eye(3) * 4e-6, (count, 498, 12, 3, 3)).copy()
                    pred = folder / ('source_predictions.npz' if stage == 'source' else 'target_predictions.npz')
                    np.savez_compressed(pred, names=np.array(names), candidate=mean,
                        bayesian_covariance_m2__calibrated_full_coordinate_covariance_v1=cov)
                    record(pred)
                    manifest = folder / ('source_manifest.json' if stage == 'source' else 'eval_manifest.json')
                    manifest.write_text(json.dumps(dict(ordered_names=names, trajectories=manifests)))
                    record(manifest)
                    if stage == 'target':
                        expected[dlo] = float(np.mean(np.abs(mean-truth[:, 2:])) * 1000)
            parent = root / 'result.json'
            metadata = dict(contract='deform-dlo45-frozen-transfer-result-v1', target_outcomes_scored=True, target_case_count=28)
            parent.write_text(json.dumps(metadata)); record(parent, metadata)
            invpath = work / 'inventory.json'
            invpath.write_text(json.dumps(dict(status='complete', run_id='33984475829', parent_roots=[str(root)], records=records)))
            req = dict(inventory_path=str(invpath), parent_cache_root=str(root.parent), data_root=str(work/'not-used'),
                datasets=['DLO4', 'DLO5'], expected_candidate_coordinate_l1_mm=expected, bootstrap_repetitions=100)
            reqpath = work / 'experiments/nonlinear_query_value_v1/request.json'
            reqpath.parent.mkdir(parents=True); reqpath.write_text(json.dumps(req))
            out = work / 'output'; out.mkdir()
            before = Path.cwd()
            try:
                os.chdir(work)
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluate(req, out)
            finally:
                os.chdir(before)
            result = json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['panels']), 4)
            self.assertTrue(all(x['passed'] for x in result['parent_mean_parity'].values()))
            self.assertEqual(len((out/'units.csv').read_text().splitlines()), 1009)
            self.assertTrue((out/'prediction_seal.json').is_file())
            parent.write_text(parent.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                Inputs(req)


if __name__ == '__main__':
    unittest.main()
