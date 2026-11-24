"""Verify empirical extraction, ID ordering, replay and the inversion handoff."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import yaml

from vespainv.cli import main, build_parser, _resolve_prep_source_earth_args
from vespainv.config import load_yaml_mapping
from vespainv.empirical_stf import (
    ReferenceEditor, build_reference, reference_settings, save_receiver_reference,
    select_receiver_reference,
)
from vespainv.runner import run_config, _resolve_stf_bandpass
from vespainv.stf import load_stf
from vespainv.stf_sweep import construct_windows
from test_empirical_stf import records, pulse_trace


def test_individual_windows_order_polarity_and_time_zero():
    recs = records()
    settings = reference_settings(dict(
        mode="per_trace", included_ids=["TA.B", "TA.A"], window=[-0.5, 1],
        trace_windows={"TA.B": [-1.2, 0.8]}, polarities={"TA.B": -1},
    ))
    stf, info = build_reference(recs, {"TA.A": 5, "TA.B": 5.3}, .05, settings)
    assert info["stf_trace_ids"] == ["TA.A", "TA.B"]
    assert stf[0, 0] == -stf[-1, 0]
    np.testing.assert_allclose(np.max(np.abs(stf[:, 1:]), axis=0), 1)
    assert stf[np.argmax(stf[:, 1]), 0] == pytest.approx(0)
    assert stf[np.argmin(stf[:, 2]), 0] == pytest.approx(0)
    assert np.all(stf[stf[:, 0] < -.5, 1] == 0)
    assert info["alignment_shifts_s"] == {"TA.A": 0., "TA.B": 0.}


@pytest.mark.parametrize("options, message", [
    ({"trace_windows": {"TA.A": [1, -1]}}, "trace_windows"),
    ({"trace_windows": {"missing": [-1, 1]}}, "missing"),
    ({"trace_windows": {"TA.B": [-8, 1]}}, "TA.B.*outside"),
    ({"trace_windows": {"TA.A": [0, .05]}}, "five samples"),
    ({"refine_alignment": True}, "Refine alignment"),
])
def test_per_trace_selection_errors_identify_problem(options, message):
    with pytest.raises(ValueError, match=message):
        settings = reference_settings(dict(mode="per_trace", **options))
        build_reference(records(), {"TA.A": 5, "TA.B": 5.3}, .05, settings)


def test_replay_ids_and_padded_length(tmp_path):
    settings = dict(mode="per_trace", interactive=False, included_ids=["TA.B", "TA.A"],
                    trace_windows={"TA.B": [-1, 1]}, window=[-.5, 1])
    ids, stf, selection = select_receiver_reference(records(), .05, 400, settings)
    assert ids == selection["stf_trace_ids"] == ["TA.A", "TA.B"]
    save_receiver_reference(tmp_path, stf, selection, None)
    replay = dict(selection_file=str(tmp_path / "stf_selection.yaml"))
    ids2, stf2, _ = select_receiver_reference(records(), .05, 400, replay)
    assert ids2 == ids
    np.testing.assert_array_equal(stf2, stf)
    assert (tmp_path / "stf.csv").read_text().splitlines()[0] == "time,stf_1,stf_2"
    with pytest.raises(ValueError, match="longer"):
        select_receiver_reference(records(), .05, 5, settings)
    with pytest.raises(ValueError, match="IDs must match"):
        save_receiver_reference(tmp_path, stf[:, :2], selection, None)
    with pytest.raises(ValueError, match="shared STFs"):
        construct_windows(np.arange(400)*.05, np.ones((400, 2)), ids, selection, -1, [1])


def test_per_trace_editor_keeps_windows_and_drops_in_sync():
    import matplotlib.pyplot as plt
    settings = reference_settings(dict(mode="per_trace", window=[-1, 1], view_window=[-2, 2]))
    editor = ReferenceEditor(records(), {"TA.A": 5, "TA.B": 5.3}, .05, settings)
    try:
        editor.active = "TA.A"
        editor.select_window(-.5, .8)
        editor.active = "TA.B"
        editor.select_window(-1.2, 1.)
        assert editor.settings["trace_windows"] == {"TA.A": [-.5, .8], "TA.B": [-1.2, 1.]}
        editor.refine.set_active(0)
        assert not editor.settings["refine_alignment"]
        editor.toggle_keep(None)
        stf, diag = build_reference(editor.records, editor.arrivals, editor.dt, editor.settings)
        assert stf.shape[1] == 2
        assert diag["stf_trace_ids"] == ["TA.A"]
        editor.fig.canvas.draw()
    finally:
        plt.close(editor.fig)


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("is3c", [False, True])
def test_cli_prepare_replay_and_run_trace_stfs(tmp_path, source, is3c):
    data, noise = tmp_path / "sac", tmp_path / "noise"
    data.mkdir(); noise.mkdir()
    rng = np.random.default_rng(12)
    for j, (station, distance, width) in enumerate([("A", 48, .18), ("B", 45, .35), ("C", 50, .22)]):
        for ch in ("Z", "R", "T"):
            tr = pulse_trace(station=station, component=ch)
            tr.data = np.exp(-((tr.times()-5)/width)**2)
            tr.stats.sac.gcarc = distance
            tr.stats.sac.baz = 30
            tr.stats.sac.evla, tr.stats.sac.evlo = -20+j, 170+j
            name = f"event{station}_TA.{station}.BH{ch}.sac"
            tr.write(str(data / name), format="SAC")
            tr.data = rng.normal(0, .005 if station != "C" else 1., len(tr.data))
            tr.write(str(noise / name.replace(".sac", ".noise.sac")), format="SAC")
    ids = ["eventA", "eventB"] if source else ["TA.A", "TA.B"]
    expected_ids = ids[::-1]  # Distance sorting must also sort the STF columns.
    prep = dict(dataset=dict(input_dir=str(data), noise_dir=str(noise), output_dir=str(tmp_path / "prepared")),
                processing=dict(bandpass=[.2, 2], downsample_hz=10, time_window=[3, 13]),
                qc=dict(snr_threshold=2),
                empirical_stf=dict(mode="per_trace", interactive=False, window=[-1, 1],
                                   included_ids=ids, trace_windows={ids[1]: [-1.2, 1.2]}))
    cfg = tmp_path / "prep.yaml"
    cfg.write_text(yaml.safe_dump(prep))
    command = "prep-source-earth" if source else "prep-earth"
    main([command, "--config", str(cfg)])
    output = tmp_path / "prepared"
    stf = load_stf(output / "stf.csv", n_traces=2, dt=.1)
    assert (output / "trace_ids.txt").read_text().splitlines() == expected_ids
    if source:
        assert (output / "source_event_ids.txt").read_text().splitlines() == expected_ids
    saved = yaml.safe_load((output / "stf_selection.yaml").read_text())
    assert saved["stf_trace_ids"] == expected_ids
    assert saved["array_type"] == ("source" if source else "receiver")
    assert [item["id"] for item in saved["traces"]] == expected_ids
    np.testing.assert_allclose(np.max(np.abs(stf[:, 1:]), axis=0), 1)
    assert not np.allclose(stf[:, 1], stf[:, 2])
    assert np.sum(stf[:, 1] ** 2) > np.sum(stf[:, 2] ** 2)  # B has the broader pulse.
    assert np.loadtxt(output / "UZ.csv", delimiter=",").shape[1] == 2
    # Replay from the raw records must reproduce every STF column exactly.
    prep["dataset"]["output_dir"] = str(tmp_path / "replayed")
    prep["empirical_stf"] = dict(selection_file=str(output / "stf_selection.yaml"))
    cfg.write_text(yaml.safe_dump(prep))
    main([command, "--config", str(cfg)])
    np.testing.assert_array_equal(load_stf(tmp_path / "replayed/stf.csv"), stf)
    assert _resolve_stf_bandpass(dict(man_stf=True), str(output)) is None
    config = load_yaml_mapping(Path(__file__).parents[1] / "configs/example_parameter_setup.yaml")
    config["defaults"].update(data_root=str(tmp_path), runs_root=str(tmp_path / "runs"),
                              is3c=is3c, srcArray=source, totalSteps=8, burnInSteps=4,
                              nSaveModels=2, maxN=2, num_chains=1, fitAtts=True,
                              attsRange=[0., 4.], fitPhase=is3c)
    config["experiments"] = [dict(dataset="prepared", runname="short")]
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump(config))
    run_config(run_yaml)
    run = tmp_path / "runs/prepared/short"
    np.testing.assert_array_equal(load_stf(run / "stf_used.csv"), stf)
    assert np.isfinite(np.loadtxt(run / "log_likelihood.txt")).all()


def test_source_cli_boolean_preserves_mapping(tmp_path):
    cfg = tmp_path / "prep.yaml"
    cfg.write_text(yaml.safe_dump(dict(data_dir="/tmp/sac", output_dir="/tmp/output",
                                      empirical_stf=dict(mode="per_trace", window=[-1, 1]))))
    parser = build_parser()
    args = parser.parse_args(["prep-source-earth", "--config", str(cfg), "--empirical-stf"])
    assert _resolve_prep_source_earth_args(args)["empirical_stf"]["mode"] == "per_trace"
    args = parser.parse_args(["prep-source-earth", "--config", str(cfg), "--no-empirical-stf"])
    assert _resolve_prep_source_earth_args(args)["empirical_stf"] is False
