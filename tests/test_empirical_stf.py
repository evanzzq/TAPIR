from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from vespainv.empirical_stf import (
    ReferenceEditor,
    ReferenceTrace,
    build_reference,
    reference_settings,
    resolve_arrival,
    save_receiver_reference,
)
from vespainv.config import write_dataset_manifest
from vespainv.model import Bookkeeping, VespaModel
from vespainv.runner import _prepare_stf, _resolve_stf_bandpass
from vespainv.utils import prepare_inputs_from_sac
from vespainv.validation import validate_dataset
from vespainv.waveformBuilder import create_U_from_model_freqdomain


def pulse_trace(delay=0.0, station="A", component="Z"):
    dt = 0.05
    time = np.arange(400) * dt
    trace = Trace(np.exp(-(((time - 5.0 - delay) / 0.18) ** 2)))
    trace.stats.update(
        dict(
            network="TA",
            station=station,
            channel="BH" + component,
            delta=dt,
            starttime=UTCDateTime(2020, 1, 1),
        )
    )
    trace.stats.sac = dict(
        b=0.0, o=0.0, a=5.0, ka="P", evdp=10.0, evla=0.0, evlo=0.0, stla=45.0, stlo=20.0, gcarc=48.0
    )
    return trace


def records():
    return [
        ReferenceTrace(
            "TA." + station, {ch: pulse_trace(delay, station, ch) for ch in ("Z", "R", "T")}, 48.0
        )
        for station, delay in [("A", 0.0), ("B", 0.3)]
    ]


def test_sac_pick_is_relative_to_reference_time_not_trace_start():
    trace = pulse_trace()
    trace.stats.sac.b = 20.0
    trace.stats.sac.a = 27.5
    time, info = resolve_arrival(trace, "TA.A", reference_settings(True))
    assert time == 7.5
    assert info == {"source": "header", "header": "a"}


def test_prediction_requires_explicit_depth_units_and_uses_origin_offset():
    trace = pulse_trace()
    del trace.stats.sac.a
    trace.stats.sac.evdp = 560300.0
    trace.stats.sac.b = 10.0
    trace.stats.sac.o = -570.0

    class Model:
        def get_travel_times(self, **kwargs):
            assert kwargs["source_depth_in_km"] == 560.3
            return [SimpleNamespace(time=610.0, name="P")]

    with pytest.raises(ValueError, match="depth_unit"):
        resolve_arrival(trace, "TA.A", reference_settings(True), Model())
    time, info = resolve_arrival(trace, "TA.A", reference_settings({"depth_unit": "m"}), Model())
    assert time == 30.0
    assert info["source"] == "prediction"


def test_alignment_sharpens_stack_and_does_not_mutate_observations():
    traces = records()
    before = traces[1].components["Z"].data.copy()
    settings = reference_settings(dict(window=[-1, 1], refine_alignment=True))
    stf, diagnostics = build_reference(traces, {"TA.A": 5, "TA.B": 5}, 0.05, settings)
    assert diagnostics["alignment_shifts_s"]["TA.B"] == pytest.approx(0.3)
    assert stf[np.argmax(stf[:, 1]), 0] == pytest.approx(0.0)
    np.testing.assert_allclose(
        diagnostics["aligned_pulses"][0], diagnostics["aligned_pulses"][1], atol=1e-12
    )
    np.testing.assert_array_equal(before, traces[1].components["Z"].data)


def test_asymmetric_window_preserves_fft_arrival_time():
    stf, _ = build_reference(
        records()[:1], {"TA.A": 5}, 0.05, reference_settings(dict(window=[-0.5, 1.0]))
    )
    assert stf[0, 0] == -stf[-1, 0]
    model = VespaModel(
        1, 1, np.array([6.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), 0.0
    )
    bookkeeping = Bookkeeping(0, 0, 0, 0, 0)
    time = np.arange(400) * 0.05
    prediction = create_U_from_model_freqdomain(
        model,
        np.array([[0, 0]]),
        time,
        stf[:, 0],
        stf[:, 1],
        bookkeeping,
    )
    assert time[np.argmax(prediction[:, 0])] == pytest.approx(6.0)


def test_reference_out_of_bounds_and_unknown_ids_are_rejected():
    with pytest.raises(ValueError, match="outside available"):
        build_reference(records(), {"TA.A": 0, "TA.B": 0}, 0.05, reference_settings(True))
    with pytest.raises(ValueError, match="included_ids"):
        build_reference(records(), {}, 0.05, reference_settings(dict(included_ids=["missing"])))


def test_processed_reference_skips_only_its_own_filter(tmp_path):
    stf, _ = build_reference(records()[:1], {"TA.A": 5}, 0.05, reference_settings(True))
    reference = save_receiver_reference(tmp_path, stf, {}, (0.2, 1.0))
    write_dataset_manifest(
        tmp_path,
        body="earth",
        array_type="receiver",
        bandpass=(0.2, 1.0),
        downsample_hz=None,
        time_window=None,
        reference_wavelet=reference,
    )
    params = dict(man_stf=True)
    assert _resolve_stf_bandpass(params, str(tmp_path)) is None
    loaded = _prepare_stf(
        str(tmp_path),
        True,
        "dGaussian",
        np.zeros((400, 1)),
        False,
        0.05,
        _resolve_stf_bandpass(params, str(tmp_path)),
    )
    np.testing.assert_array_equal(loaded, stf)
    assert _resolve_stf_bandpass(dict(man_stf=False), str(tmp_path)) == (0.2, 1.0)
    with pytest.raises(ValueError, match="does not match"):
        _resolve_stf_bandpass(dict(man_stf=True, bandpass=[0.1, 0.5]), str(tmp_path))
    (tmp_path / "stf.csv").write_text("time,stf\n0,1\n1,0\n")
    with pytest.raises(ValueError, match="differs"):
        _resolve_stf_bandpass(params, str(tmp_path))


def test_receiver_qc_and_manual_selection_keep_all_columns_in_sync(tmp_path):
    import yaml

    data, noise, output = [tmp_path / name for name in ("sac", "noise", "out")]
    data.mkdir()
    noise.mkdir()
    rng = np.random.default_rng(22)
    for station in ("A", "B", "C"):
        for component in ("Z", "R", "T"):
            trace = pulse_trace(station=station, component=component)
            trace.stats.sac.gcarc += ord(station) - ord("A")
            filename = f"event_TA.{station}.BH{component}.sac"
            trace.write(str(data / filename), format="SAC")
            trace.data = rng.normal(0, 0.005 if station != "C" else 0.5, len(trace.data))
            trace.write(str(noise / filename.replace(".sac", ".noise.sac")), format="SAC")
    reference = prepare_inputs_from_sac(
        str(data),
        noise_dir=str(noise),
        output_dir=str(output),
        snr_threshold=2,
        empirical_stf=dict(interactive=False, included_ids=["TA.A"], contributors=["TA.A"]),
    )
    write_dataset_manifest(
        output,
        body="earth",
        array_type="receiver",
        bandpass=None,
        downsample_hz=None,
        time_window=None,
        reference_wavelet=reference,
    )
    validation = validate_dataset(output, manual_stf=True)
    assert validation.n_traces == 1
    assert (output / "trace_ids.txt").read_text().strip() == "TA.A"
    selection = yaml.safe_load((output / "stf_selection.yaml").read_text())
    assert any(item["id"] == "TA.C" and item["status"] == "rejected" for item in selection["qc"])
    replay = reference_settings(dict(selection_file=str(output / "stf_selection.yaml")))
    assert replay["interactive"] is False
    assert replay["included_ids"] == ["TA.A"]

    # Exercise the actual handoff from empirical SAC preparation to inversion.
    from pathlib import Path
    from vespainv.config import load_yaml_mapping
    from vespainv.runner import run_config
    from vespainv.stf import load_stf

    config = load_yaml_mapping(Path(__file__).parents[1] / "configs/example_parameter_setup.yaml")
    config["defaults"].update(
        data_root=str(tmp_path), runs_root=str(tmp_path / "runs"),
        totalSteps=8, burnInSteps=4, nSaveModels=2, num_chains=1,
        maxN=2, minSpace=0.2,
    )
    config["experiments"] = [dict(dataset="out", runname="empirical_smoke")]
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_config(config_path)
    run = tmp_path / "runs/out/empirical_smoke"
    np.testing.assert_array_equal(load_stf(run / "stf_used.csv"), load_stf(output / "stf.csv"))
    assert np.isfinite(np.loadtxt(run / "log_likelihood.txt")).all()


def test_editor_controls_share_the_export_selection():
    import matplotlib.pyplot as plt

    editor = ReferenceEditor(
        records(), {"TA.A": 5, "TA.B": 5}, 0.05, reference_settings(dict(window=[-1, 1]))
    )
    editor.active = "TA.B"
    editor.toggle_keep(None)
    assert editor.settings["included_ids"] == ["TA.A"]
    assert editor.settings["contributors"] == ["TA.A"]
    editor.select_window(-0.5, 0.7)
    assert editor.settings["window"] == [-0.5, 0.7]
    editor.flip(None)
    assert editor.settings["polarities"]["TA.B"] == -1
    editor.fig.canvas.draw()
    plt.close(editor.fig)


def test_editor_mouse_drag_updates_window_after_redraw():
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent

    editor = ReferenceEditor(
        records(),
        {"TA.A": 5, "TA.B": 5},
        0.05,
        reference_settings(dict(window=[-1, 1], view_window=[-2, 2])),
    )
    editor.fig.canvas.draw()
    for lo, hi in [(-0.7, 0.8), (-0.4, 0.6)]:
        for name, x in [
            ("button_press_event", lo),
            ("motion_notify_event", hi),
            ("button_release_event", hi),
        ]:
            px, py = editor.preview.transData.transform((x, 0.2))
            event = MouseEvent(name, editor.fig.canvas, px, py, button=1)
            editor.fig.canvas.callbacks.process(name, event)
        np.testing.assert_allclose(editor.settings["window"], [lo, hi], atol=0.05)
    plt.close(editor.fig)
