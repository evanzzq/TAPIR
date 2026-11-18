import numpy as np
import pytest

from vespainv.runner import _prepare_prior, _validate_run_parameters


def _valid_parameters():
    return {
        "totalSteps": 20,
        "burnInSteps": 10,
        "nSaveModels": 5,
        "num_chains": 1,
        "actionsPerStep": 1,
        "maxN": 2,
        "normOpt": 1,
    }


def test_validate_run_parameters_accepts_valid_values():
    _validate_run_parameters(_valid_parameters())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("totalSteps", 0, "totalSteps"),
        ("burnInSteps", 20, "burnInSteps"),
        ("nSaveModels", 11, "nSaveModels"),
        ("num_chains", 0, "num_chains"),
        ("actionsPerStep", 0, "actionsPerStep"),
        ("maxN", 0, "maxN"),
        ("normOpt", 3, "normOpt"),
    ],
)
def test_validate_run_parameters_rejects_invalid_values(name, value, message):
    parameters = _valid_parameters()
    parameters[name] = value

    with pytest.raises(ValueError, match=message):
        _validate_run_parameters(parameters)


@pytest.mark.parametrize("is3c", [False, True])
def test_prepare_prior_passes_tstar_prior(is3c):
    args = (is3c, 1.0, 10, 0.05, np.arange(0.0, 60.0, 0.5), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    prior = _prepare_prior(*args, attsRange=[0.0, 8.0], attsStd=0.5)
    assert prior.attsRange == (0.0, 8.0)
    assert prior.attsStd == 0.5
    default = _prepare_prior(*args)
    assert default.attsRange == ((1.0, 1.0) if is3c else (0, 4))
