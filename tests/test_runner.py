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


@pytest.mark.parametrize("is3c", [False, True])
def test_yaml_prior_options_reach_chain(tmp_path, monkeypatch, is3c):
    from pathlib import Path
    import yaml
    from vespainv import runner
    from vespainv.config import load_yaml_mapping

    config = load_yaml_mapping(Path(__file__).parents[1] / "configs/example_parameter_setup.yaml")
    config["experiments"] = [config["experiments"][0]]
    defaults = config["defaults"]
    defaults.update(data_root=str(Path(__file__).parents[1] / "examples/data"),
                    runs_root=str(tmp_path / "runs"), is3c=is3c, max_procs=1,
                    attsRange=[0., 8.], attsStd=0.5, arrStd=0.25, slwStd=0.3,
                    ampStd=0.15, aziStd=7., logeStd=0.2, logeRange=[0., 6.], fitAtts=True)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))
    captured = []
    monkeypatch.setattr(runner, "_run_chain", lambda cid, exp: captured.append(exp["prior"]))
    runner.run_config(config_path)
    prior = captured[0]
    for name in ("attsStd", "arrStd", "slwStd", "ampStd", "logeStd"):
        assert getattr(prior, name) == defaults[name]
    assert prior.attsRange == (0., 8.)
    assert prior.logeRange == (0., 6.)
    if is3c:
        assert prior.aziStd == 7.


@pytest.mark.parametrize(("name", "value"), [
    ("attsRange", [-1, 4]), ("attsRange", [2, 1]), ("attsRange", [0]),
    ("attsStd", -1), ("slwStd", float("nan")), ("ampStd", float("inf")),
    ("logeRange", [6, 0]), ("max_procs", 0),
])
def test_invalid_optional_settings_fail_before_running(name, value):
    params = _valid_parameters()
    params[name] = value
    with pytest.raises(ValueError, match=name):
        _validate_run_parameters(params)
