import numpy as np
import pytest
from scipy.fft import fft, fftfreq, ifft

from vespainv.waveformBuilder import tstar_conv, tstar_conv_freqdomain


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
