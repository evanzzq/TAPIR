import numpy as np
import pytest

from vespainv.model import Prior, Prior3c, VespaModel, VespaModel3c
from vespainv.rjmcmc import birth, birth3c


@pytest.mark.parametrize("is3c", [False, True])
def test_crowded_birth_rejected_without_mutating_model(is3c):
    prior = (Prior3c if is3c else Prior)(timeRange=(0., 1.), minSpace=2., maxN=3)
    common = dict(Nphase=1, Ntrace=1, arr=np.array([0.5]), slw=np.array([0.]),
                  amp=np.array([1.]), atts=np.array([1.]), loge=0.)
    if is3c:
        model = VespaModel3c(**common, azi=np.array([0.]), ph_hh=np.array([0.]),
                            ph_vh=np.array([0.]), wvtype=np.array([0]))
    else:
        model = VespaModel(**common)
    proposed, success = (birth3c if is3c else birth)(model, prior)
    assert not success
    assert model.Nphase == proposed.Nphase == 1
    for name, value in vars(model).items():
        np.testing.assert_equal(vars(proposed)[name], value)


@pytest.mark.parametrize("is3c", [False, True])
def test_birth_still_succeeds_when_space_available(is3c):
    prior = (Prior3c if is3c else Prior)(timeRange=(0., 10.), maxN=3)
    model = VespaModel3c.create_empty(1) if is3c else VespaModel.create_empty(1, prior)
    proposed, success = (birth3c if is3c else birth)(model, prior)
    assert success
    assert model.Nphase == 0
    assert proposed.Nphase == 1
    assert len(proposed.arr) == len(proposed.slw) == len(proposed.amp) == len(proposed.atts) == 1
