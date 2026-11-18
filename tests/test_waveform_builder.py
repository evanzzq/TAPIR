import numpy as np
import pytest
from scipy.fft import fft, fftfreq, ifft

from vespainv.model import Bookkeeping, VespaModel, VespaModel3c
from vespainv.waveformBuilder import (
    _create_U_from_model_3c_freqdomain_loop,
    _create_U_from_model_freqdomain_loop,
    create_U_from_model_3c_freqdomain,
    create_U_from_model_freqdomain,
    tstar_conv,
    tstar_conv_freqdomain,
)


def _geometry(kind, n_traces, rng, **options):
    """Station lat/lon (receiver array) or source dist/baz (source array, Mars) metadata and bookkeeping."""
    if kind == "receiver":
        metadata = np.column_stack([rng.uniform(30, 40, n_traces), rng.uniform(-125, -110, n_traces)])
        geometry = dict(refLat=35.0, refLon=-117.0, refBaz=250.0, srcLat=-18.0, srcLon=178.0)
    else:
        metadata = np.column_stack([rng.uniform(40, 70, n_traces), rng.uniform(80, 110, n_traces)])
        geometry = dict(refLat=5.0, refLon=-150.0, refBaz=95.0, srcLat=4.5, srcLon=135.6,
                        srcArray=True, isMars=kind == "mars")
    return metadata, Bookkeeping(**geometry, **options)


def _inputs(n_samples):
    time = np.arange(n_samples) * 0.5
    stf_time = np.arange(-10.0, 10.01, 0.5)
    stf = -2 * stf_time / 4.0 * np.exp(-(stf_time / 2.0) ** 2)
    return time, stf_time, stf


@pytest.mark.parametrize("n_samples", [119, 128])
@pytest.mark.parametrize("fit_atts", [False, True])
@pytest.mark.parametrize("kind", ["receiver", "source", "mars"])
def test_vectorized_1c_forward_matches_trace_loop(kind, fit_atts, n_samples):
    rng = np.random.default_rng(1)
    metadata, bookkeeping = _geometry(kind, 17, rng, fitAtts=fit_atts)
    time, stf_time, stf = _inputs(n_samples)
    n = 9
    model = VespaModel(Nphase=n, Ntrace=17, arr=rng.uniform(0, time[-1], n), slw=rng.uniform(-2, 2, n),
                       amp=rng.uniform(-1, 1, n), atts=rng.uniform(0, 4, n), loge=0.0)
    expected = _create_U_from_model_freqdomain_loop(model, metadata, time, stf_time, stf, bookkeeping)
    actual = create_U_from_model_freqdomain(model, metadata, time, stf_time, stf, bookkeeping)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10 * np.abs(expected).max())


@pytest.mark.parametrize("n_samples", [119, 128])
@pytest.mark.parametrize("fit_phase", [False, True])
@pytest.mark.parametrize("fit_atts", [False, True])
@pytest.mark.parametrize("kind", ["receiver", "source", "mars"])
def test_vectorized_3c_forward_matches_trace_loop(kind, fit_atts, fit_phase, n_samples):
    rng = np.random.default_rng(2)
    metadata, bookkeeping = _geometry(kind, 17, rng, fitAtts=fit_atts, fitPhase=fit_phase, pref=6.0)
    time, stf_time, stf = _inputs(n_samples)
    n = 12
    model = VespaModel3c(Nphase=n, Ntrace=17, arr=rng.uniform(0, time[-1], n), slw=rng.uniform(-2, 2, n),
                         amp=rng.uniform(-1, 1, n), azi=rng.uniform(-90, 90, n), ph_hh=rng.uniform(0, 180, n),
                         ph_vh=rng.uniform(-90, 90, n), atts=rng.uniform(0, 4, n), loge=0.0,
                         wvtype=rng.integers(0, 2, n))
    expected = _create_U_from_model_3c_freqdomain_loop(model, metadata, time, stf_time, stf, bookkeeping)
    actual = create_U_from_model_3c_freqdomain(model, metadata, time, stf_time, stf, bookkeeping)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10 * np.abs(expected).max())


def test_forward_models_without_phases_are_zero():
    rng = np.random.default_rng(3)
    metadata, bookkeeping = _geometry("receiver", 5, rng)
    time, stf_time, stf = _inputs(64)
    empty = VespaModel(Nphase=0, Ntrace=5, arr=np.array([]), slw=np.array([]), amp=np.array([]),
                       atts=np.array([]), loge=0.0)
    assert not create_U_from_model_freqdomain(empty, metadata, time, stf_time, stf, bookkeeping).any()
    assert create_U_from_model_3c_freqdomain(VespaModel3c.create_empty(5), metadata, time, stf_time, stf,
                                             bookkeeping).shape == (64, 5, 3)


def _pulse(n=256, dt=0.1):
    t = np.arange(n) * dt
    return t, np.exp(-((t - 8.0) / 0.8) ** 2) * np.cos(2 * np.pi * 0.6 * (t - 8.0))


def test_tstar_zero_is_identity():
    t, x = _pulse()
    W, f = fft(x), fftfreq(len(x), t[1] - t[0])
    np.testing.assert_allclose(np.real(ifft(tstar_conv_freqdomain(W, f, 0.0))), x, atol=1e-12)
    np.testing.assert_allclose(tstar_conv(x, t, 0.0), x, atol=1e-12)


@pytest.mark.parametrize("t_star", [1e-9, 0.5, 2.0])
def test_tstar_keeps_waveform_real_and_low_frequency_level(t_star):
    t, x = _pulse()
    W, f = fft(x), fftfreq(len(x), t[1] - t[0])
    y = ifft(tstar_conv_freqdomain(W, f, t_star))
    assert np.max(np.abs(y.imag)) < 1e-12 * np.max(np.abs(y.real))
    # the operator is 1 at f = 0, so the waveform's area is unchanged
    assert np.isclose(np.sum(y.real), np.sum(x))
    if t_star < 1e-6:
        np.testing.assert_allclose(y.real, x, atol=1e-8)


def test_tstar_attenuates_delays_and_broadens():
    t, x = _pulse()
    y = tstar_conv(x, t, 1.0)
    assert y.max() < x.max()
    assert t[np.argmax(y)] > t[np.argmax(x)]


@pytest.mark.parametrize("n", [15, 16])
def test_tstar_is_real_for_broadband_inputs(n):
    rng = np.random.default_rng(45)
    x = rng.normal(size=n)
    time = np.arange(n) * 0.2
    y = ifft(tstar_conv_freqdomain(fft(x), fftfreq(n, 0.2), 0.1))
    np.testing.assert_allclose(y.imag, 0.0, atol=1e-14)
    np.testing.assert_allclose(y.real, tstar_conv(x, time, 0.1), atol=1e-14)
    np.testing.assert_allclose(tstar_conv(x, time, 0.0), x, atol=1e-14)
    assert y.real.sum() == pytest.approx(x.sum())


def test_tstar_nyquist_uses_real_projection():
    n, dt, t_star = 16, 0.2, 0.1
    x = (-1.0) ** np.arange(n)
    fnyq = 0.5 / dt
    expected = np.exp(-np.pi * fnyq * t_star) * np.cos(2 * fnyq * t_star * np.log(fnyq))
    y = ifft(tstar_conv_freqdomain(fft(x), fftfreq(n, dt), t_star))
    np.testing.assert_allclose(y, x * expected, atol=1e-14)


@pytest.mark.parametrize("is3c", [False, True])
@pytest.mark.parametrize("n", [15, 16])
def test_broadband_forward_matches_loop(is3c, n):
    rng = np.random.default_rng(87)
    metadata, bk = _geometry("receiver", 3, rng, fitAtts=True, fitPhase=True)
    time = np.arange(n) * 0.2
    stf_time = np.arange(-3, 4) * 0.2
    stf = rng.normal(size=7)
    common = dict(Nphase=2, Ntrace=3, arr=np.array([0.13, 0.77]), slw=np.array([0.1, 0.3]),
                  amp=np.array([1., -0.5]), atts=np.array([0.1, 0.3]), loge=0.)
    if is3c:
        model = VespaModel3c(**common, azi=np.array([30., 50.]), ph_hh=np.array([20., 40.]),
                            ph_vh=np.array([10., 30.]), wvtype=np.array([0, 1]))
        fast, loop = create_U_from_model_3c_freqdomain, _create_U_from_model_3c_freqdomain_loop
    else:
        model = VespaModel(**common)
        fast, loop = create_U_from_model_freqdomain, _create_U_from_model_freqdomain_loop
    expected = loop(model, metadata, time, stf_time, stf, bk)
    actual = fast(model, metadata, time, stf_time, stf, bk)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
