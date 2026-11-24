import copy

import numpy as np
import pytest

from vespainv.stf_sweep import construct_windows


def inputs():
    time = np.arange(351) * 0.2

    # Different offsets and amplitudes, identical main and secondary pulses.
    def signal(t):
        return np.exp(-(((t + 1) / 0.5) ** 2)) + 0.3 * np.exp(-(((t - 3) / 0.5) ** 2))

    data = np.column_stack((3 + 2 * signal(time - 20), -2 + 5 * signal(time - 20.4)))
    selection = dict(
        settings=dict(window=[-5, 5], contributors=["A", "B"], mode="stack", polarities={}),
        arrivals={key: dict(seconds_from_input_start=20) for key in ["A", "B"]},
        alignment_shifts_s={"A": 0, "B": 0.4},
    )
    return time, data, ["A", "B"], selection


def test_sweep_freezes_weights_alignment_and_common_interior():
    time, data, ids, selection = inputs()
    original = data.copy()
    settings = copy.deepcopy(selection)
    variants, meta = construct_windows(time, data, ids, selection, -4, [1.5, 2, 4])
    assert variants[0][1]["sampled_window_s"] == pytest.approx([-4, 1.4])
    assert meta["fixed_contributors"]["B"]["alignment_shift_s"] == 0.4
    # All exports use the same symmetric padding and time origin.
    a, b = variants[0][0], variants[-1][0]
    assert a.shape == b.shape == (41, 2)
    interior = (a[:, 0] >= -3.4) & (a[:, 0] <= 0.8)
    np.testing.assert_allclose(a[interior], b[interior], atol=1e-14)
    assert a[np.argmax(a[:, 1]), 0] == pytest.approx(-1)
    assert np.all(a[a[:, 0] >= 1.4 + 1e-9, 1] == 0)
    assert b[np.argmin(abs(b[:, 0] - 3)), 1] > 0.2
    # Right-edge taper spans exactly 0.6 seconds, not a fraction of the window.
    assert b[-1, 1] == 0
    idx = np.argmin(abs(b[:, 0] - 3.8))
    assert 0 < b[idx, 1] < b[idx - 1, 1]
    np.testing.assert_array_equal(data, original)
    assert settings == selection


def test_reordering_variants_does_not_change_references():
    args = inputs()
    forward, _ = construct_windows(*args, -4, [1.5, 4])
    reverse, _ = construct_windows(*args, -4, [4, 1.5])
    np.testing.assert_array_equal(forward[0][0], reverse[1][0])
    np.testing.assert_array_equal(forward[1][0], reverse[0][0])


def test_rejects_extrapolation_and_overlapping_tapers():
    args = inputs()
    with pytest.raises(ValueError, match="anchor"):
        construct_windows(*args, -4, [6])
    with pytest.raises(ValueError, match="short"):
        construct_windows(*args, 0, [1])
    args[-1]["arrivals"]["A"]["seconds_from_input_start"] = 0
    with pytest.raises(ValueError, match="available data"):
        construct_windows(*args, -4, [2])
