"""Test data-boundary and correction helpers without accessing actual DEFORM."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_development import correction_to_world, query_keys, split_names, trajectory_records


def manifest(tmp_path):
    fit=[f'fit_{i}' for i in range(40)]
    validation=[f'validation_{i}' for i in range(8)]
    source=[f'source_{i}' for i in range(8)]
    return {'dlo_type':'DLO2','partition':'train','official_eval_read':False,
            'split':{'fit':fit,'validation':validation,'source_test':source},
            'trajectories':{n:{'path':str(tmp_path/(n+'.pkl'))} for n in fit+validation+source}}


def test_historical_split(tmp_path):
    fit,val=split_names(manifest(tmp_path))
    assert len(fit)==40 and len(val)==8


def test_split_overlap_rejected(tmp_path):
    m=manifest(tmp_path);m['split']['validation'][0]='fit_0'
    with pytest.raises(ValueError):split_names(m)


def test_source_alias_rejected(tmp_path):
    m=manifest(tmp_path);trajectory_records(m)['source_0']['path']=trajectory_records(m)['fit_0']['path']
    with pytest.raises(ValueError):split_names(m)


def test_eval_input_rejected(tmp_path):
    m=manifest(tmp_path);trajectory_records(m)['fit_0']['path']=str(tmp_path/'eval'/'f.pkl')
    with pytest.raises(ValueError):split_names(m)


def test_wrong_cohort_rejected(tmp_path):
    m=manifest(tmp_path);m['dlo_type']='DLO4'
    with pytest.raises(ValueError):split_names(m)


def test_exact_clamped_baseline_preserved():
    rng=np.random.default_rng(33)
    base=rng.normal(size=(3,15,12,3));copy=base.copy()
    local=rng.normal(size=(3,15,8,3));frames=np.repeat(np.eye(3)[None],3,axis=0)
    prediction=correction_to_world(base,local,frames,.25)
    assert np.array_equal(prediction[:,:,[0,1,-2,-1]],base[:,:,[0,1,-2,-1]])
    np.testing.assert_allclose(prediction[:,:,2:-2],base[:,:,2:-2]+.25*local)
    assert np.array_equal(base,copy)


def test_zero_shrinkage_exact_fallback():
    base=np.ones((2,3,6,3));local=np.ones((2,3,2,3));frames=np.repeat(np.eye(3)[None],2,axis=0)
    assert np.array_equal(correction_to_world(base,local,frames,0),base)


def test_frame_rotation():
    frame=np.array([[0.,-1,0],[1,0,0],[0,0,1]])[None]
    base=np.zeros((1,3,6,3));local=np.zeros((1,3,2,3));local[:,:,:,0]=1
    result=correction_to_world(base,local,frame,1)
    np.testing.assert_allclose(result[:,:,2:-2,1],1)
    np.testing.assert_allclose(result[:,:,2:-2,0],0)


def test_query_key_ignores_future_truth():
    a=np.zeros((3,2,6,3));b=np.zeros((3,4,4,3));c=np.zeros((3,4,6,3))
    before=query_keys(a,b,c);a[0,0,0,0]=1
    after=query_keys(a,b,c)
    assert before[0]!=after[0] and before[1:]==after[1:]


def test_read_guard_blocks_source_eval_and_dataset_writes(tmp_path):
    import json
    import subprocess
    m=manifest(tmp_path/'train')
    data=tmp_path/'train';data.mkdir()
    allowed=Path(trajectory_records(m)['fit_0']['path']);allowed.write_text('allowed')
    source=Path(trajectory_records(m)['source_0']['path']);source.write_text('forbidden')
    upstream=tmp_path/'upstream'
    official=upstream/'data_set'/'DLO2'/'eval'/'sealed.pkl'
    official.parent.mkdir(parents=True);official.write_text('forbidden')
    code='''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from run_development import install_read_guard, trajectory_records
import json
m=json.loads(sys.argv[2]);up=Path(sys.argv[3]);official=Path(sys.argv[4])
allowed=Path(trajectory_records(m)['fit_0']['path'])
source=Path(trajectory_records(m)['source_0']['path'])
opened=install_read_guard(m,m['split']['fit']+m['split']['validation'],up)
assert allowed.read_text()=='allowed'
for path,mode in [(source,'r'),(official,'r'),(allowed,'w')]:
    try:
        with path.open(mode): pass
    except PermissionError: pass
    else: raise AssertionError('access guard failed')
assert allowed.read_text()=='allowed'
assert len(opened)==2
'''
    completed=subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve().parents[1]),
                              json.dumps(m),str(upstream),str(official)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_native_mapping_does_not_need_source_opened_flag(tmp_path):
    m=manifest(tmp_path)
    assert 'source_test_opened' not in m
    assert len(trajectory_records(m)) == 56
    assert len(split_names(m)[0]) == 40


def test_malformed_file_identity_rejected(tmp_path):
    m=manifest(tmp_path)
    m['trajectories']['fit_0'] = []
    with pytest.raises(ValueError):
        split_names(m)


def test_non_native_manifest_rejected(tmp_path):
    m=manifest(tmp_path)
    m['trajectories']=list(m['trajectories'].values())
    with pytest.raises(ValueError):
        split_names(m)
