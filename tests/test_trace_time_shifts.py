import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from vespainv.trace_time_shifts import load_trace_time_shifts, trace_time_shifts
from vespainv.model import Bookkeeping, VespaModel
from vespainv import runner
from vespainv.waveformBuilder import create_U_from_model_freqdomain, create_U_from_model_3c_freqdomain
from test_trace_stf import inputs, write_dataset


def test_csv_matches_by_id_not_row_order(tmp_path):
    (tmp_path/'trace_ids.txt').write_text('a\nb\nc\n')
    p=tmp_path/'shifts.csv';p.write_text('trace_id,time_shift_s\nc,-0.125\na,0\nb,1.75\n')
    ids, shifts=load_trace_time_shifts(p,tmp_path,3)
    assert ids==['a','b','c']
    np.testing.assert_array_equal(shifts,[0,1.75,-.125])


@pytest.mark.parametrize('contents,message',[
 ('trace_id,time_shift_s\na,0\nb,1\n','missing'),
 ('trace_id,time_shift_s\na,0\nb,1\nc,2\nd,3\n','extra'),
 ('trace_id,time_shift_s\na,0\na,1\nc,2\n','duplicate'),
 ('trace_id,time_shift_s\na,nan\nb,1\nc,2\n','finite'),
 ('trace_id,time_shift_s\na,inf\nb,1\nc,2\n','finite'),
 ('trace_id,time_shift_s\na,bad\nb,1\nc,2\n','Invalid'),
 ('trace_id,time_shift_s\na\nb,1\nc,2\n','Invalid'),
 ('trace_id,time_shift_s\na,0,extra\nb,1\nc,2\n','Extra'),
 ('time_shift_s,trace_id\n0,a\n','header'),
])
def test_bad_shift_files_fail(tmp_path,contents,message):
    (tmp_path/'trace_ids.txt').write_text('a\nb\nc\n')
    p=tmp_path/'shifts.csv';p.write_text(contents)
    with pytest.raises(ValueError,match=message):load_trace_time_shifts(p,tmp_path,3)


@pytest.mark.parametrize('ids',['a\na\nc\n','a\nb\n','a\n\nc\n'])
def test_invalid_dataset_ids_fail(tmp_path,ids):
    (tmp_path/'trace_ids.txt').write_text(ids)
    p=tmp_path/'shifts.csv';p.write_text('trace_id,time_shift_s\na,0\nb,0\nc,0\n')
    with pytest.raises(ValueError,match='trace_ids'):load_trace_time_shifts(p,tmp_path,3)


@pytest.mark.parametrize('is3c',[False,True])
@pytest.mark.parametrize('kind,locdiff',[('receiver',False),('source',False),('mars',False),('source',True),('mars',True)])
@pytest.mark.parametrize('n',[31,32])
@pytest.mark.parametrize('per_trace',[False,True])
def test_fixed_delays_equal_individual_arrival_changes(is3c,kind,locdiff,n,per_trace):
    model,metadata,time,st,amplitudes,bk=inputs(is3c,kind,locdiff,True,n)
    stf=amplitudes if per_trace else amplitudes[:,0]
    fn=create_U_from_model_3c_freqdomain if is3c else create_U_from_model_freqdomain
    expected=[]
    # Independent oracle: shift every phase's arrival in a one-trace model.
    for j,delay in enumerate([0.,1.,-.375]):
        one=copy.deepcopy(model);one.Ntrace=1;one.arr=one.arr+delay
        one.distDiff=model.distDiff[j:j+1];one.bazDiff=model.bazDiff[j:j+1]
        expected.append(fn(one,metadata[j:j+1],time,st,amplitudes[:,j] if per_trace else stf,bk))
    bk.trace_time_shifts=np.array([0.,1.,-.375])
    np.testing.assert_allclose(fn(model,metadata,time,st,stf,bk),np.concatenate(expected,axis=1),rtol=1e-11,atol=1e-11)


def test_positive_delay_moves_pulse_later_and_old_pickle_defaults_to_zero():
    bk=Bookkeeping(refLat=0,refLon=0,refBaz=0,srcLat=0,srcLon=0)
    del bk.__dict__['trace_time_shifts']  # old saved Bookkeeping object
    bk=pickle.loads(pickle.dumps(bk))
    time=np.arange(129)*.1;st=np.arange(-10,11)*.1;stf=np.exp(-(st/.2)**2)
    model=VespaModel(Nphase=1,Ntrace=1,arr=np.array([4.]),slw=np.array([0.]),amp=np.array([1.]),atts=np.array([0.]),loge=0.)
    before=create_U_from_model_freqdomain(model,np.array([[0.,0.]]),time,st,stf,bk)
    bk.trace_time_shifts=np.array([.5])
    after=create_U_from_model_freqdomain(model,np.array([[0.,0.]]),time,st,stf,bk)
    np.testing.assert_allclose(after,np.roll(before,5,axis=0),atol=1e-14)


@pytest.mark.parametrize('is3c',[False,True])
def test_yaml_reaches_chain_and_saved_reconstruction(tmp_path,monkeypatch,is3c):
    data=tmp_path/'case';write_dataset(data)
    (data/'trace_ids.txt').write_text('a\nb\nc\n')
    (tmp_path/'delays.csv').write_text('trace_id,time_shift_s\nc,-.25\na,0\nb,1.5\n')
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/example_parameter_setup.yaml').read_text())
    config['defaults'].update(data_root=str(tmp_path),runs_root=str(tmp_path/'runs'),is3c=is3c,max_procs=1,trace_time_shifts_file='delays.csv')
    config['experiments']=[dict(dataset='case',runname='shifted')]
    path=tmp_path/'run.yaml';path.write_text(yaml.safe_dump(config))
    seen=[];monkeypatch.setattr(runner,'_run_chain',lambda cid,exp:seen.append(exp['bookkeeping']))
    runner.run_config(path)
    out=tmp_path/'runs/case/shifted'
    ids,shifts=load_trace_time_shifts(out/'trace_time_shifts_used.csv',data,3)
    np.testing.assert_array_equal(shifts,[0,1.5,-.25])
    with (out/'Bookkeeping.pkl').open('rb') as f: saved=pickle.load(f)
    assert saved.trace_ids==tuple(ids)
    np.testing.assert_array_equal(saved.trace_time_shifts,seen[0].trace_time_shifts)
    # A changed correction must not silently reuse an incompatible Bookkeeping.pkl.
    (tmp_path/'delays.csv').write_text('trace_id,time_shift_s\na,0\nb,2\nc,-.25\n')
    with pytest.raises(ValueError,match='new runname'):runner.run_config(path)
    np.testing.assert_array_equal(load_trace_time_shifts(out/'trace_time_shifts_used.csv',data,3)[1],shifts)
