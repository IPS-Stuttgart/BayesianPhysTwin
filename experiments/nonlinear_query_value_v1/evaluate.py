"""Read-only replay of hash-bound, completed DEFORM DLO4/DLO5 predictions."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import platform
import time
import numpy as np
from analysis import build_readouts, geometric, paired_bootstrap, queries


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar'),('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer')}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected dataset global: {module}.{name}')
        return super().find_class(module,name)


class Inputs:
    def __init__(self,request):
        self.request=request
        self.inventory_path=Path(request['inventory_path'])
        inv=json.loads(self.inventory_path.read_text())
        if inv['status']!='complete' or inv['run_id']!='33984475829':
            raise ValueError('Inventory identity differs')
        self.root=Path(request['parent_cache_root'])/'33361441865-1'
        if inv['parent_roots']!=[str(self.root)]:raise ValueError('Parent root differs')
        self.records={r['path']:r for r in inv['records']}
        self.used={}
        # Only the completed parent result authorizes this retrospective replay.
        final=[r for r in inv['records'] if r.get('metadata',{}).get('contract')=='deform-dlo45-frozen-transfer-result-v1']
        if len(final)!=1:raise ValueError('Expected exactly one completed parent result')
        parent=self.read_json(Path(final[0]['path']))
        if parent.get('target_outcomes_scored') is not True or parent.get('target_case_count')!=28:
            raise ValueError('Parent target panel was not already scored')
        self.parent=parent

    def verified(self,path):
        path=Path(path)
        r=self.records[str(path)]
        if path.stat().st_size!=r['bytes'] or digest(path)!=r['sha256']:
            raise ValueError(f'Frozen cache identity mismatch: {path}')
        self.used[str(path)]={'sha256':r['sha256'],'bytes':r['bytes']}
        return path

    def read_json(self,path):return json.loads(self.verified(path).read_text())

    def bundle(self,dlo,stage):
        folder=self.root/f'{dlo.lower()}-{stage}'
        path=self.verified(folder/('source_predictions.npz' if stage=='source' else 'target_predictions.npz'))
        with np.load(path,allow_pickle=False) as z:
            names=z['names'].tolist()
            mean=np.asarray(z['candidate'],dtype=np.float64)
            key='bayesian_covariance_m2__calibrated_full_coordinate_covariance_v1'
            cov=np.asarray(z[key],dtype=np.float64)
        count=8 if stage=='source' else 14
        if len(names)!=count or len(set(names))!=count or mean.shape!=(count,498,12,3) or cov.shape!=(count,498,12,3,3):
            raise ValueError(f'Unexpected prediction roster/shape for {dlo}/{stage}')
        if not np.isfinite(mean).all() or not np.isfinite(cov).all():raise ValueError('Nonfinite predictions')
        return names,mean,cov

    def truth(self,dlo,stage,names):
        folder=self.root/f'{dlo.lower()}-{stage}'
        manifest=self.read_json(folder/('source_manifest.json' if stage=='source' else 'eval_manifest.json'))
        if stage=='target' and manifest['ordered_names']!=names:raise ValueError('Target name order differs')
        trajectories=[]
        for name in names:
            r=manifest['trajectories'][name]
            path=Path(r['path'])
            if not path.is_file():
                path=Path(self.request['data_root'])/dlo/('train' if stage=='source' else 'eval')/name
            blob=path.read_bytes()
            if len(blob)!=r['size_bytes'] or hashlib.sha256(blob).hexdigest()!=r['sha256']:
                raise ValueError(f'Dataset identity mismatch: {path}')
            # Independent replay of the exact parent float32 transpose/z-floor loader.
            x=np.asarray(ArrayUnpickler(io.BytesIO(blob)).load(),dtype=np.float32)
            if x.shape!=(500,3,12) or not np.isfinite(x).all():raise ValueError('Bad trajectory')
            nodes=x.transpose(0,2,1).copy();nodes[:,:,2]=np.clip(nodes[:,:,2],.002001,10000)
            trajectories.append(nodes[2:])
            self.used[str(path)]={'sha256':r['sha256'],'bytes':r['size_bytes']}
        return np.asarray(trajectories,dtype=np.float64)


def evaluate(request,output):
    output=Path(output); started=time.monotonic();inputs=Inputs(request)
    w,info=queries();cache={};parameters={};payload={}
    for dlo in request['datasets']:
        sn,sm,sc=inputs.bundle(dlo,'source');sy=inputs.truth(dlo,'source',sn)
        tn,tm,tc=inputs.bundle(dlo,'target')
        # No target outcome is loaded until all readouts for both objects are sealed.
        # The first forecast geometry is a predictor feature, not an observed state.
        models={}
        for panel,heldout in [('all_queries',False),('heldout_queries',True)]:
            arms,scales,meta=build_readouts(sm[:,:,2:10],sy[:,:,2:10],sc[:,:,2:10],sm[:,0,2:10],tm[:,:,2:10],tc[:,:,2:10],tm[:,0,2:10],heldout_mode=heldout)
            models[panel]=(arms,scales)
            parameters[f'{dlo}__{panel}']=meta
            for arm,x in arms.items():payload[f'{dlo}__{panel}__{arm}']=x
            payload[f'{dlo}__{panel}__scale']=scales
        cache[dlo]=(tn,tm,models)
    np.savez_compressed(output/'sealed_query_predictions.npz',**payload)
    save_json(output/'source_fitted_parameters.json',parameters)
    seal={'status':'predictions-sealed-before-target-truth-read','predictions_sha256':digest(output/'sealed_query_predictions.npz'),'parameters_sha256':digest(output/'source_fitted_parameters.json'),'target_outcome_use_for_prediction':False,'request_sha256':digest('experiments/nonlinear_query_value_v1/request.json')}
    save_json(output/'prediction_seal.json',seal)
    bypanel={};units=[];parity={};payload_truth={}
    for dlo,(names,mean,models) in cache.items():
        truth=inputs.truth(dlo,'target',names)
        l1=float(np.mean(np.abs(mean-truth))*1000)
        expected=request['expected_candidate_coordinate_l1_mm'][dlo]
        if abs(l1-expected)>1e-5:raise ValueError(f'Parent mean/loader parity failed: {dlo}: {l1} != {expected}')
        parity[dlo]={'candidate_coordinate_l1_mm':l1,'expected':expected,'passed':True}
        gt=geometric(truth[:,:,2:10],w);payload_truth[dlo]=gt
        for panel,(arms,scales) in models.items():
            for family in ['squared_distance','bending_statistic']:
                ids=[i for i,q in enumerate(info) if q['family']==family and (panel!='heldout_queries' or q['heldout'])]
                key=f'{panel}/{family}';entry=bypanel.setdefault(key,{})
                for arm,pred in arms.items():
                    err=pred[:,:,ids]-gt[:,:,ids]
                    nmse=np.mean((err/scales[ids])**2,axis=(1,2))
                    raw=np.sqrt(np.mean(err**2,axis=(1,2)))*1e6
                    entry.setdefault(arm,{})[dlo]=nmse
                    for i,name in enumerate(names):
                        units.append(dict(panel=panel,family=family,dlo=dlo,trajectory=name,arm=arm,normalized_mse=float(nmse[i]),rmse_mm2=float(raw[i])))
    np.savez_compressed(output/'query_truth.npz',**payload_truth)
    panels={}
    for key,arms in bypanel.items():
        stacked={name:np.stack([values[d] for d in request['datasets']]) for name,values in arms.items()}
        base=stacked['point_plugin'];base_rmse=np.sqrt(base.mean())
        summary={}
        for name,scores in stacked.items():
            summary[name]={'normalized_rmse':float(np.sqrt(scores.mean())),
                'improvement_vs_plugin_pct':float(100*(1-np.sqrt(scores.mean())/base_rmse)),
                'per_dlo':{d:{'normalized_rmse':float(np.sqrt(scores[i].mean())),'improvement_vs_plugin_pct':float(100*(1-np.sqrt(scores[i].mean()/base[i].mean()))),'wins_vs_plugin':int(np.sum(scores[i]<base[i]))} for i,d in enumerate(request['datasets'])}}
        comparisons={}
        for primary in ['native_block_posterior','source_coupled_posterior_extension']:
            comparisons[primary]={name:paired_bootstrap(stacked[primary],scores,reps=request['bootstrap_repetitions']) for name,scores in stacked.items() if name!=primary}
        gates={}
        for primary in comparisons:
            baseline_pass=comparisons[primary]['point_plugin']['bootstrap95'][1]<0 and all(v['improvement_vs_plugin_pct']>=1 for v in summary[primary]['per_dlo'].values())
            strong=all(comparisons[primary][name]['bootstrap95'][1]<0 for name in ['empirical_global_centered','empirical_horizon_centered','empirical_symmetric_residual','source_query_bias','source_query_horizon_bias','source_query_ridge'])
            gates[primary]={'beats_plugin_consistently':bool(baseline_pass),'beats_all_strong_controls':bool(strong)}
        panels[key]={'arms':summary,'paired_comparisons':comparisons,'gates':gates}
    decision={}
    for arm in ['native_block_posterior','source_coupled_posterior_extension']:
        decision[arm]={p:all(panels[f'{p}/{f}']['gates'][arm]['beats_plugin_consistently'] for f in ['squared_distance','bending_statistic']) for p in ['all_queries','heldout_queries']}
        decision[arm]['strong_control_superiority']=all(panels[f'all_queries/{f}']['gates'][arm]['beats_all_strong_controls'] for f in ['squared_distance','bending_statistic'])
    result={'schema_version':1,'experiment':'nonlinear-query-value-v1','status':'complete','classification':'retrospective-fixed-public-panel','datasets':request['datasets'],'source_trajectories_per_dlo':8,'test_trajectories_per_dlo':14,'forecast_frames':498,'queries':info,'same_mean':True,'native_covariance':'3x3 within-marker blocks; no native cross-marker covariance','coupled_arm':'NEW source-estimated block-normalized correlation with fixed 0.5 identity shrinkage; not an existing joint Bayesian posterior','controls':'Empirical centered covariance, symmetric empirical residuals, signed query bias, horizon bias and ridge; all source fitted','mean_model_shift':'Source controls use eight out-of-fit predictions from the 39-fit parent; evaluation uses the parent all-56 refit. No test outcome enters a fit. This source-to-target predictor change is retained, not concealed.','statistical_unit':'complete trajectory resampled within each of two FIXED DLO objects; no arbitrary-object population inference','heldout_query_scope':'No query-specific readout calibration for far-marker pairs or odd-center bends; generic field covariance may use complete source geometry','target_loader':'Exact parent float32 transpose and z>=0.002001 preprocessing; not pristine raw geometry','causal_timing':seal,'parent_mean_parity':parity,'decisions':decision,'panels':panels,'input_identities':inputs.used,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'seconds':time.monotonic()-started,'github_run_id':os.environ.get('GITHUB_RUN_ID'),'github_sha':os.environ.get('GITHUB_SHA')},'limitations':['Only two real objects; retrospective analysis, not independent confirmation.','The source-coupled extension is an empirical covariance extension, not evidence for uniquely Bayesian inference.','Quadratic-query gains cannot establish higher-order posterior value or latent physical-state accuracy.','Recorded boundary inputs are shared; no robot or new physical action was used.']}
    save_json(output/'result.json',result)
    with (output/'units.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(units[0]));writer.writeheader();writer.writerows(units)
    lines=['# Nonlinear deformation-query posterior test','','Completed retrospective DEFORM DLO4/DLO5 replay: 28 evaluation trajectories, 498 forecast frames each, 27 quadratic queries.','', 'All readouts use exactly the same coordinate mean. RMSE below is dimensionless, normalized using source-only family scales. Positive improvement means lower RMSE.','', '| Panel / query | Readout | Normalized RMSE | Improvement vs point |','|---|---|---:|---:|']
    for key,record in panels.items():
        for arm,values in record['arms'].items():
            lines.append(f"| {key} | {arm} | {values['normalized_rmse']:.6f} | {values['improvement_vs_plugin_pct']:+.2f}% |")
    lines+=['','## Frozen decision rules','',json.dumps(decision,indent=2),'','Both query families must improve by at least 1% in each DLO and have negative upper paired-bootstrap MSE-difference bounds against the point readout. Strong-control superiority additionally requires beating every declared empirical and deterministic control.','', 'The native posterior contains no cross-marker covariance. The separately named coupled extension is source-estimated, not a retained full Bayesian posterior. Source controls and target means inherit the parent 39-fit/all-56-refit difference. Trajectories, not frames or marker pairs, are the resampling units. Results describe two fixed objects, not unseen-object generalization.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','decisions':decision,'parent_mean_parity':parity},indent=2))
