import copy
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pytest
import yaml

from vespainv.model import Bookkeeping, VespaModel, VespaModel3c
from vespainv.runner import _filter_stf, _prepare_stf, _resolve_stf_bandpass, run_config
from vespainv.stf import load_stf, stf_header, validate_stf_table
from vespainv.utils import est_stf_wid
from vespainv.validation import validate_dataset
from vespainv.waveformBuilder import (
    create_U_from_model_freqdomain, create_U_from_model_3c_freqdomain,
    _create_U_from_model_freqdomain_loop, _create_U_from_model_3c_freqdomain_loop,
)


def inputs(is3c, kind, locdiff=False, fit=False, n=32):
    time = np.arange(n) * 0.5
    st = np.arange(-2, 2.1, 0.5)
    rng = np.random.default_rng(52)
    amplitudes = rng.normal(size=(len(st), 3)) * np.array([1., -.4, 2.])
    source = kind != "receiver"
    metadata = np.array([[50., 100.], [51., 102.], [52., 104.]]) if source else np.array([[35., -118.], [36., -117.], [37., -116.]])
    bk = Bookkeeping(refLat=35., refLon=-117., refBaz=250., srcLat=-18., srcLon=178.,
                     srcArray=source, isMars=kind == "mars", locDiff=locdiff,
                     fitAtts=fit, fitPhase=fit, pref=4.)
    kw = dict(Nphase=3, Ntrace=3, arr=np.array([1.13, 5.77, 9.42]),
              slw=np.array([.1, .2, .4]), amp=np.array([1., -.4, .3]),
              atts=np.array([0., .5, 2.]), loge=0., distDiff=np.array([.2, -.1, .15]),
              bazDiff=np.array([.3, -.1, .2]))
    model = VespaModel3c(**kw, azi=np.array([10., 30., 50.]), ph_vh=np.array([20., 40., 60.]),
                        ph_hh=np.array([30., 50., 70.]), wvtype=np.array([1, 0, 0])) if is3c else VespaModel(**kw)
    return model, metadata, time, st, amplitudes, bk


@pytest.mark.parametrize(("is3c", "fit_att", "fit_phase"), [
    (False, False, False), (False, True, False), (True, False, False),
    (True, True, False), (True, False, True), (True, True, True),
])
@pytest.mark.parametrize(("kind", "locdiff"), [("receiver", False), ("source", False), ("mars", False), ("source", True), ("mars", True)])
@pytest.mark.parametrize("n", [31, 32])
def test_trace_stfs_equal_individual_shared_stf_predictions(is3c, kind, locdiff, fit_att, fit_phase, n):
    model, metadata, time, st, amplitudes, bk = inputs(is3c, kind, locdiff, fit_att, n)
    bk.fitPhase = fit_phase
    fast = create_U_from_model_3c_freqdomain if is3c else create_U_from_model_freqdomain
    loop = _create_U_from_model_3c_freqdomain_loop if is3c else _create_U_from_model_freqdomain_loop
    expected = []
    for j in range(3):
        one = copy.deepcopy(model)
        one.Ntrace = 1
        one.distDiff, one.bazDiff = model.distDiff[j:j+1], model.bazDiff[j:j+1]
        expected.append(loop(one, metadata[j:j+1], time, st, amplitudes[:, j], bk))
    expected = np.concatenate(expected, axis=1)
    np.testing.assert_allclose(fast(model, metadata, time, st, amplitudes, bk), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(loop(model, metadata, time, st, amplitudes, bk), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("is3c", [False, True])
def test_repeated_stf_columns_preserve_shared_behavior(is3c):
    model, metadata, time, st, amplitudes, bk = inputs(is3c, "receiver", fit=True)
    f = create_U_from_model_3c_freqdomain if is3c else create_U_from_model_freqdomain
    shared = amplitudes[:, 0]
    a = f(model, metadata, time, st, shared, bk)
    np.testing.assert_allclose(f(model, metadata, time, st, shared[:, None], bk), a, atol=1e-14)
    np.testing.assert_allclose(f(model, metadata, time, st, np.tile(shared[:, None], (1, 3)), bk), a, atol=1e-14)


def test_trace_column_mapping_has_analytical_zero_shift_solution():
    model, _, time, st, amplitudes, bk = inputs(False, "receiver")
    model.Nphase = 1
    model.arr, model.slw, model.amp = np.array([st[-1]]), np.array([0.]), np.array([1.])
    metadata = np.zeros((3, 2))
    actual = create_U_from_model_freqdomain(model, metadata, time, st, amplitudes, bk)
    expected = np.pad(amplitudes, ((0, len(time)-len(st)), (0, 0)))
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_filter_preserves_between_trace_amplitude_and_polarity_ratios():
    time = np.arange(-10., 10., .05)
    wave = np.exp(-(time/.8)**2)
    table = np.column_stack([time, wave, -3*wave])
    original = table.copy()
    filtered = _filter_stf(table, (.2, .8))
    np.testing.assert_array_equal(table, original)
    np.testing.assert_array_equal(filtered[:, 0], time)
    np.testing.assert_allclose(filtered[:, 2], -3*filtered[:, 1], atol=1e-13)
    assert np.max(np.abs(filtered[:, 1:])) == pytest.approx(1.)


@pytest.mark.parametrize(("change", "message"), [
    ("count", "expected 1 shared column or 3"), ("interval", "sample interval"),
    ("irregular", "uniformly"), ("decreasing", "strictly increasing"),
    ("nan", "finite"), ("zero", "nonzero"), ("length", "data have only"),
])
def test_bad_trace_stfs_fail_before_forward_model(change, message):
    st = np.arange(-2., 2.1, .5)
    table = np.column_stack([st, np.ones((len(st), 3))])
    n_samples = 32
    if change == "count": table = table[:, :-1]
    if change == "interval": table[:, 0] *= 2
    if change == "irregular": table[3, 0] += .1
    if change == "decreasing": table[:, 0] = table[::-1, 0]
    if change == "nan": table[3, 2] = np.nan
    if change == "zero": table[:, 2] = 0
    if change == "length": n_samples = 5
    with pytest.raises(ValueError, match=message):
        validate_stf_table(table, n_traces=3, dt=.5, n_samples=n_samples)


def test_auto_spacing_uses_widest_trace_even_when_weak():
    time = np.arange(-2., 2.1, .5)
    narrow = np.zeros(len(time)); narrow[4] = 1.
    broad = np.full(len(time), .001)
    assert est_stf_wid(np.column_stack([time, narrow, broad])) == 4.


def write_dataset(root, ntraces=3):
    root.mkdir()
    time = np.arange(101)*.2
    waves = np.sin(2*np.pi*.5*time[:, None]) * np.linspace(.5, 1., ntraces)
    np.savetxt(root/'time.csv', time, delimiter=',')
    for c in 'ZRT': np.savetxt(root/f'U{c}.csv', waves, delimiter=',')
    np.savetxt(root/'station_metadata.csv', np.column_stack([np.linspace(35., 36., ntraces), np.linspace(-118., -117., ntraces)]), delimiter=',', header='lat,lon', comments='')
    np.savetxt(root/'eventinfo.csv', [[-18., 178.]], delimiter=',', header='lat,lon', comments='')
    st = np.arange(-.8, .81, .2)
    amplitudes = np.exp(-(st[:, None]/np.linspace(.2, .5, ntraces))**2) * np.linspace(.3, 1., ntraces)
    table = np.column_stack([st, amplitudes])
    np.savetxt(root/'stf.csv', table, delimiter=',', header=stf_header(table), comments='')
    return table


@pytest.mark.parametrize("is3c", [False, True])
@pytest.mark.parametrize(("manual", "shape"), [(True, "dGaussian"), (False, "dGaussian"), (False, "Gaussian")])
def test_real_short_runner_keeps_all_stf_columns(tmp_path, is3c, manual, shape, monkeypatch):
    from vespainv.config import load_yaml_mapping
    from vespainv import waveformBuilder as wb
    data = tmp_path/'case'
    supplied = write_dataset(data)
    config = load_yaml_mapping(Path(__file__).parents[1]/'configs/example_parameter_setup.yaml')
    config['defaults'].update(data_root=str(tmp_path), runs_root=str(tmp_path/'runs'),
                              is3c=is3c, man_stf=manual, stfshape=shape, runname='short', totalSteps=8,
                              burnInSteps=4, nSaveModels=2, maxN=2, fitAtts=True,
                              attsRange=[0., 4.], fitPhase=is3c, num_chains=1,
                              minSpace=None, bandpass=None)
    config['experiments']=[dict(dataset='case', runname='short')]
    cfg=tmp_path/'run.yaml';cfg.write_text(yaml.safe_dump(config))
    fn='create_U_from_model_3c_freqdomain' if is3c else 'create_U_from_model_freqdomain'
    original=getattr(wb,fn);seen=[]
    def checked(model, metadata, time, stf_time, stf, bookkeeping):
        seen.append(stf.shape)
        if manual: np.testing.assert_allclose(stf, supplied[:, 1:])
        return original(model, metadata, time, stf_time, stf, bookkeeping)
    monkeypatch.setattr(wb,fn,checked)
    np.random.seed(19)
    run_config(cfg)
    run=tmp_path/'runs/case/short'
    used=load_stf(run/'stf_used.csv', n_traces=3, dt=.2)
    assert seen
    assert used.shape[1] == (4 if manual else 2)
    if manual: np.testing.assert_array_equal(used, supplied)
    np.testing.assert_array_equal(load_stf(data/'stf.csv'), supplied)
    assert len(pickle.loads((run/'ensemble.pkl').read_bytes()))==2
    assert np.isfinite(np.loadtxt(run/'log_likelihood.txt')).all()


def test_validate_data_and_prepare_reject_column_mismatch(tmp_path):
    data=tmp_path/'case'; supplied=write_dataset(data)
    np.savetxt(data/'stf.csv', supplied[:, :3], delimiter=',', header='time,stf_1,stf_2', comments='')
    with pytest.raises(ValueError, match='2 STF amplitude columns'):
        validate_dataset(data, manual_stf=True)
    with pytest.raises(ValueError, match='2 STF amplitude columns'):
        _prepare_stf(str(data), True, 'Gaussian', np.ones((101,3,3)), True, .2)


def test_already_filtered_trace_stfs_are_verified_not_filtered_twice(tmp_path):
    data=tmp_path/'case'; supplied=write_dataset(data)
    checksum=hashlib.sha256((data/'stf.csv').read_bytes()).hexdigest()
    (data/'dataset.yaml').write_text(yaml.safe_dump(dict(processing=dict(bandpass=[.1, 1.]), reference_wavelet=dict(already_processed=True,bandpass=[.1,1.],sha256=checksum))))
    bandpass=_resolve_stf_bandpass(dict(man_stf=True),str(data))
    assert bandpass is None
    used=_prepare_stf(str(data),True,'Gaussian',np.ones((101,3,3)),True,.2,bandpass)
    np.testing.assert_array_equal(used,supplied)
    (data/'stf.csv').write_text((data/'stf.csv').read_text()+'\n')
    with pytest.raises(ValueError,match='differs'):
        _resolve_stf_bandpass(dict(man_stf=True),str(data))


def test_header_is_not_silently_treated_as_numeric_data(tmp_path):
    path=tmp_path/'stf.csv'
    np.savetxt(path, [[0.,1.],[.2,1.]], delimiter=',')
    with pytest.raises(ValueError, match='header'):
        load_stf(path)


def test_one_trace_1c_runner_loads_a_column_as_one_trace(tmp_path):
    from vespainv.utils import prep_data
    data = tmp_path/'single'
    supplied = write_dataset(data, ntraces=1)
    observed, _, _, _, metadata, _ = prep_data(str(tmp_path), 'single', False, 'Z', 0)
    assert observed.shape == (101, 1)
    assert metadata.shape == (1, 2)
    used = _prepare_stf(str(data), True, 'Gaussian', observed, False, .2)
    np.testing.assert_array_equal(used, supplied)


@pytest.mark.parametrize("is3c", [False, True])
def test_waveform_plot_reconstructs_with_all_trace_stfs(monkeypatch, is3c):
    import matplotlib.pyplot as plt
    from vespainv import waveformBuilder as wb
    from vespainv.visualization import plot_seismogram_compare
    model, metadata, time, st, amplitudes, bk = inputs(is3c, "receiver", fit=True)
    fn = 'create_U_from_model_3c_freqdomain' if is3c else 'create_U_from_model_freqdomain'
    original = getattr(wb, fn)
    observed = original(model, metadata, time, st, amplitudes, bk)
    seen = []
    def checked(*args):
        np.testing.assert_array_equal(args[4], amplitudes)
        seen.append(True)
        return original(*args)
    monkeypatch.setattr(wb, fn, checked)
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        plot_seismogram_compare(observed, time, ensemble=[model], metadata=metadata,
                                stf=np.column_stack([st, amplitudes]), bookkeeping=bk)
        assert seen
    finally:
        plt.close('all')
