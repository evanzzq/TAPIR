"""Receiver-array reference selection; waveform construction also works without a GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import yaml


@dataclass
class ReferenceTrace:
    trace_id: str
    components: dict
    distance: float
    snr: float | None = None


def reference_settings(settings):
    if settings is None or settings is False:
        return None
    if settings is True:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("empirical_stf must be a mapping, true, or false.")
    settings = dict(settings)
    if settings.get("selection_file"):
        with Path(settings.pop("selection_file")).expanduser().open() as handle:
            saved = yaml.safe_load(handle)
        settings = {**saved["settings"], "interactive": False, **settings}
    defaults = dict(
        interactive=True,
        phase="P",
        model="ak135",
        component="Z",
        mode="stack",
        window=[-2.0, 5.0],
        view_window=[-10.0, 15.0],
        depth_unit="km",
        taper_fraction=0.1,
        refine_alignment=False,
        max_shift=1.0,
        arrival_times={},
        polarities={},
    )
    defaults.update(settings)
    defaults["component"] = str(defaults["component"]).upper().removeprefix("U")
    if defaults["component"] not in ("Z", "R", "T"):
        raise ValueError("empirical_stf.component must be Z, R, or T.")
    if defaults["mode"] not in ("single", "stack"):
        raise ValueError("empirical_stf.mode must be single or stack.")
    for key in ("window", "view_window"):
        value = np.asarray(defaults[key], dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)) or value[0] >= value[1]:
            raise ValueError(f"empirical_stf.{key} must be an increasing finite pair.")
        defaults[key] = value.tolist()
    if defaults["depth_unit"] not in ("km", "m"):
        raise ValueError("empirical_stf.depth_unit must be km or m.")
    if not 0 <= float(defaults["taper_fraction"]) <= 0.5:
        raise ValueError("taper_fraction must be between 0 and 0.5 (per end).")
    if not np.isfinite(defaults["max_shift"]) or defaults["max_shift"] < 0:
        raise ValueError("max_shift must be finite and non-negative.")
    return defaults


def resolve_arrival(trace, trace_id, settings, model=None):
    """Return seconds from this input trace's start, and provenance. SAC times use reftime."""
    override = settings["arrival_times"].get(trace_id)
    if override is not None:
        if not np.isfinite(override):
            raise ValueError(f"Nonfinite arrival override for {trace_id}.")
        return float(override), {"source": "override"}
    sac = trace.stats.sac
    header = settings.get("pick_header")
    if header is not None and header not in ["a", *[f"t{i}" for i in range(10)]]:
        raise ValueError("pick_header must be a or t0 through t9.")
    for name in [header] if header else ["a", *[f"t{i}" for i in range(10)]]:
        label = "ka" if name == "a" else "k" + name
        if not header and str(sac.get(label, "")).strip() != settings["phase"]:
            continue
        value = sac.get(name, -12345)
        if np.isfinite(value) and value != -12345:
            return float(value - sac.b), {"source": "header", "header": name}
    if model is None:
        from obspy.taup import TauPyModel

        model = TauPyModel(model=settings["model"])
    depth = float(sac.get("evdp", -12345))
    if settings["depth_unit"] == "m":
        depth /= 1000.0
    if not 0 <= depth < 6371:
        raise ValueError(
            f"Invalid depth {depth:g} km for {trace_id}; check empirical_stf.depth_unit "
            "(these SAC files may store metres)."
        )
    origin = float(sac.get("o", -12345))
    if origin == -12345 or not np.isfinite(origin):
        raise ValueError(f"No SAC origin offset for {trace_id}; supply arrival_times override.")
    distance = float(sac.get("gcarc", -12345))
    if not 0 <= distance <= 180:
        from obspy.geodetics import locations2degrees

        distance = locations2degrees(sac.evla, sac.evlo, sac.stla, sac.stlo)
    arrivals = model.get_travel_times(
        source_depth_in_km=depth,
        distance_in_degree=distance,
        phase_list=[settings["phase"]],
    )
    if not arrivals:
        raise ValueError(f"No predicted {settings['phase']} for {trace_id}.")
    arrival = min(arrivals, key=lambda item: item.time)
    return float(origin - sac.b + arrival.time), dict(
        source="prediction",
        model=settings["model"],
        phase=arrival.name,
        depth_km=depth,
        travel_time_s=float(arrival.time),
    )


def extract_segment(trace, arrival, grid):
    """Sample without extrapolation or changing the observation's time reference."""
    sample_times = arrival + grid
    end = (trace.stats.npts - 1) * trace.stats.delta
    if sample_times[0] < -1e-6 or sample_times[-1] > end + 1e-6:
        raise ValueError("Reference window is outside available SAC data; shorten the window.")
    return np.interp(sample_times, trace.times(), np.asarray(trace.data, dtype=float))


def _normalized_pulse(values, polarity):
    if polarity not in (-1, 1):
        raise ValueError("Reference polarities must be +1 or -1.")
    # Estimate the baseline only from the selected window edges, not the pulse mean.
    edge = max(1, min(5, len(values) // 10))
    values = values - np.mean(np.r_[values[:edge], values[-edge:]])
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Reference window contains no finite nonzero pulse.")
    return polarity * values / scale


def build_reference(records, arrivals, dt, settings):
    """Build a peak-normalized, symmetric-time-axis STF; return alignment diagnostics."""
    from scipy.signal import correlate, correlation_lags
    from scipy.signal.windows import tukey

    by_id = {record.trace_id: record for record in records}
    included = settings.get("included_ids", list(by_id))
    contributors = settings.get("contributors", included)
    if not included or len(set(included)) != len(included) or not set(included) <= by_id.keys():
        raise ValueError("included_ids must be nonempty, unique IDs of QC-retained traces.")
    if (
        not contributors
        or len(set(contributors)) != len(contributors)
        or not set(contributors) <= set(included)
    ):
        raise ValueError("contributors must be nonempty, unique IDs of included traces.")
    if settings["mode"] == "single":
        contributors = contributors[:1]
    lo, hi = settings["window"]
    grid = np.arange(int(np.ceil(lo / dt)), int(np.floor(hi / dt)) + 1) * dt
    if len(grid) < 5:
        raise ValueError("Select at least five samples for the reference wavelet.")
    pulses, shifts = [], {}
    for trace_id in contributors:
        trace = by_id[trace_id].components[settings["component"]]
        polarity = settings["polarities"].get(trace_id, 1)
        pulse = _normalized_pulse(extract_segment(trace, arrivals[trace_id], grid), polarity)
        shift = 0.0
        if settings["refine_alignment"] and pulses:
            corr = correlate(pulse, pulses[0], mode="full")
            lags = correlation_lags(len(pulse), len(pulses[0]), mode="full")
            allowed = np.abs(lags * dt) <= settings["max_shift"] + 1e-9
            shift = float(lags[allowed][np.argmax(corr[allowed])] * dt)
            pulse = _normalized_pulse(
                extract_segment(trace, arrivals[trace_id] + shift, grid),
                polarity,
            )
        pulses.append(pulse)
        shifts[trace_id] = shift
    pulse = np.mean(pulses, axis=0)
    pulse *= tukey(len(grid), alpha=2 * float(settings["taper_fraction"]))
    scale = np.max(np.abs(pulse))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Reference stack cancels to zero; inspect reference polarities.")
    pulse /= scale
    # TAPIR's FFT builder shifts by stf_time[-1]; symmetry makes that equal -time[0].
    half = int(np.ceil(max(abs(grid[0]), abs(grid[-1])) / dt))
    time = np.arange(-half, half + 1) * dt
    stf = np.column_stack((time, np.zeros(len(time))))
    stf[np.rint(grid / dt).astype(int) + half, 1] = pulse
    return stf, dict(
        contributors=contributors,
        alignment_shifts_s=shifts,
        window_samples_s=grid.tolist(),
        aligned_pulses=np.array(pulses),
    )


def select_receiver_reference(records, dt, n_samples, settings):
    """Resolve reference arrivals, optionally review, and return an extraction result."""
    settings = reference_settings(settings)
    from obspy.taup import TauPyModel

    model = TauPyModel(model=settings["model"])
    arrivals, provenance = {}, {}
    for record in records:
        arrival, info = resolve_arrival(record.components["Z"], record.trace_id, settings, model)
        arrivals[record.trace_id] = arrival
        provenance[record.trace_id] = {**info, "seconds_from_input_start": arrival}
    settings.setdefault("included_ids", [record.trace_id for record in records])
    settings.setdefault("contributors", list(settings["included_ids"]))
    if settings["mode"] == "single":
        settings["contributors"] = settings["contributors"][:1]
    if settings["interactive"]:
        editor = ReferenceEditor(records, arrivals, dt, settings)
        editor.show()
        settings = editor.settings
    stf, diagnostics = build_reference(records, arrivals, dt, settings)
    if len(stf) > n_samples:
        raise ValueError("Padded reference wavelet is longer than the inversion data window.")
    replay = {**settings, "interactive": False}
    metadata = dict(
        schema_version=1,
        array_type="receiver",
        settings=replay,
        arrivals=provenance,
        alignment_shifts_s=diagnostics["alignment_shifts_s"],
        baseline="mean of up to five samples at each selected window edge",
        time_zero="reference pick/prediction; stack aligned to first contributor",
        traces=[dict(id=r.trace_id, distance_deg=r.distance, snr=r.snr) for r in records],
    )
    return settings["included_ids"], stf, metadata


def save_receiver_reference(output_dir, stf, metadata, bandpass):
    """Write the waveform plus reproducible selection; manifest stores processing + identity."""
    root = Path(output_dir)
    path = root / "stf.csv"
    np.savetxt(path, stf, delimiter=",", header="time,stf", comments="")
    metadata["processing"] = dict(bandpass=list(bandpass) if bandpass else None)
    with (root / "stf_selection.yaml").open("w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return dict(
        kind="empirical",
        file="stf.csv",
        already_processed=True,
        bandpass=list(bandpass) if bandpass else None,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selection_file="stf_selection.yaml",
    )


class ReferenceEditor:
    """Matplotlib review: click a row, toggle inclusion/contribution, drag the window."""

    def __init__(self, records, arrivals, dt, settings):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, CheckButtons

        self.records, self.arrivals, self.dt = records, arrivals, dt
        self.settings = {**settings, "polarities": dict(settings["polarities"])}
        self.settings["included_ids"] = list(
            settings.get("included_ids", [r.trace_id for r in records])
        )
        self.settings["contributors"] = list(
            settings.get("contributors", self.settings["included_ids"])
        )
        self.active = records[0].trace_id
        self.accepted = False
        self.fig = plt.figure(figsize=(14, 9))
        self.overview = self.fig.add_axes([0.10, 0.29, 0.48, 0.62])
        self.preview = self.fig.add_axes([0.65, 0.60, 0.32, 0.30])
        self.output = self.fig.add_axes([0.65, 0.29, 0.32, 0.22])
        self.status = self.fig.text(0.10, 0.205, "", fontsize=10)
        self.fig.suptitle(
            "TAPIR empirical reference — click a trace; drag the window in the preview"
        )
        self.buttons = []
        for j, (label, callback) in enumerate(
            [
                ("Keep/drop", self.toggle_keep),
                ("Use/unuse", self.toggle_use),
                ("Use all kept", self.use_all),
                ("Flip polarity", self.flip),
                ("Save & finish", self.finish),
            ]
        ):
            button = Button(self.fig.add_axes([0.10 + j * 0.177, 0.07, 0.16, 0.045]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.mode = RadioButtons(
            self.fig.add_axes([0.10, 0.125, 0.17, 0.065]),
            ("single", "stack"),
            active=(settings["mode"] == "stack"),
        )
        self.mode.on_clicked(self.change_mode)
        self.refine = CheckButtons(
            self.fig.add_axes([0.30, 0.13, 0.25, 0.045]),
            ["Refine alignment"],
            [settings["refine_alignment"]],
        )
        self.refine.on_clicked(self.toggle_refine)
        self.span = None
        self.fig.canvas.mpl_connect("pick_event", self.pick)
        self.refresh()

    def refresh(self):
        from matplotlib.widgets import SpanSelector

        if self.span is not None:
            self.span.disconnect_events()
        self.overview.clear()
        self.preview.clear()
        self.output.clear()
        included = self.settings["included_ids"]
        contributors = self.settings["contributors"]
        view = self.settings["view_window"]
        x = np.arange(view[0], view[1] + self.dt / 2, self.dt)
        labels = []
        for j, record in enumerate(self.records):
            trace_id = record.trace_id
            trace = record.components[self.settings["component"]]
            raw = np.interp(
                self.arrivals[trace_id] + x, trace.times(), trace.data, left=np.nan, right=np.nan
            )
            finite = np.isfinite(raw)
            scale = np.max(np.abs(raw[finite])) if np.any(finite) else 1
            y = self.settings["polarities"].get(trace_id, 1) * raw / (scale or 1)
            color = (
                "tab:orange"
                if trace_id == self.active
                else ("tab:blue" if trace_id in included else "0.8")
            )
            (line,) = self.overview.plot(x, j + 0.35 * y, color=color, picker=5)
            line.set_gid(trace_id)
            labels.append(f"{'*' if trace_id in contributors else ' '} {trace_id}")
            if trace_id == self.active:
                self.preview.plot(x, y, color="0.3", label=trace_id)
        self.overview.set(
            yticks=np.arange(len(labels)),
            yticklabels=labels,
            xlabel=f"Seconds relative to {self.settings['phase']} pick/prediction",
            title="Blue: kept | grey: dropped | *: reference contributor",
            xlim=view,
        )
        self.overview.tick_params(axis="y", labelsize=8)
        self.overview.axvline(0, color="0.5", ls="--", lw=0.7)
        self.preview.set(
            xlim=view, title=f"Active: {self.active}", xlabel="Seconds relative to arrival"
        )
        self.preview.axvspan(*self.settings["window"], color="tab:blue", alpha=0.15)
        self.span = SpanSelector(
            self.preview,
            self.select_window,
            "horizontal",
            useblit=False,
            props=dict(alpha=0.2, facecolor="tab:blue"),
            interactive=True,
        )
        self.span.extents = tuple(self.settings["window"])
        record = next(r for r in self.records if r.trace_id == self.active)
        snr = "unavailable" if record.snr is None else f"{record.snr:.2f}"
        message = f"Active {self.active} | SNR {snr} | {len(included)} kept | window {self.settings['window']} s"
        try:
            stf, diag = build_reference(self.records, self.arrivals, self.dt, self.settings)
            grid = diag["window_samples_s"]
            for pulse in diag["aligned_pulses"]:
                self.output.plot(grid, pulse, color="0.75", lw=0.5)
            self.output.plot(stf[:, 0], stf[:, 1], color="black", lw=1.5)
            self.output.set(
                title=f"Export preview ({len(diag['contributors'])} contributors)",
                xlabel="Reference-wavelet time (s)",
            )
        except ValueError as exc:
            message = str(exc)
        self.status.set_text(message)
        self.fig.canvas.draw_idle()

    def pick(self, event):
        if event.artist.get_gid():
            self.active = event.artist.get_gid()
            if self.settings["mode"] == "single" and self.active in self.settings["included_ids"]:
                self.settings["contributors"] = [self.active]
            self.refresh()

    def toggle_keep(self, _):
        ids = list(self.settings["included_ids"])
        if self.active in ids:
            ids.remove(self.active)
            self.settings["contributors"] = [
                i for i in self.settings["contributors"] if i != self.active
            ]
        else:
            ids.append(self.active)
        self.settings["included_ids"] = [r.trace_id for r in self.records if r.trace_id in ids]
        self.refresh()

    def toggle_use(self, _):
        if self.active in self.settings["included_ids"]:
            ids = list(self.settings["contributors"])
            if self.settings["mode"] == "single":
                ids = [self.active]
            elif self.active in ids:
                ids.remove(self.active)
            else:
                ids.append(self.active)
            self.settings["contributors"] = ids
        self.refresh()

    def use_all(self, _):
        self.settings["contributors"] = list(self.settings["included_ids"])
        if self.settings["mode"] == "single":
            self.settings["contributors"] = self.settings["contributors"][:1]
        self.refresh()

    def flip(self, _):
        polarities = self.settings["polarities"]
        polarities[self.active] = -polarities.get(self.active, 1)
        self.refresh()

    def change_mode(self, mode):
        self.settings["mode"] = mode
        if mode == "single":
            self.settings["contributors"] = (
                [self.active]
                if self.active in self.settings["included_ids"]
                else self.settings["included_ids"][:1]
            )
        self.refresh()

    def toggle_refine(self, _):
        self.settings["refine_alignment"] = not self.settings["refine_alignment"]
        self.refresh()

    def select_window(self, lo, hi):
        self.settings["window"] = [float(lo), float(hi)]
        self.refresh()

    def finish(self, _):
        import matplotlib.pyplot as plt

        try:
            build_reference(self.records, self.arrivals, self.dt, self.settings)
        except ValueError as exc:
            self.status.set_text(str(exc))
            self.fig.canvas.draw_idle()
            return
        self.accepted = True
        plt.close(self.fig)

    def show(self):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import rcsetup

        if matplotlib.get_backend().lower() not in [b.lower() for b in rcsetup.interactive_bk]:
            plt.close(self.fig)
            raise ValueError(
                "Interactive STF selection requires a GUI backend (e.g. QtAgg or MacOSX); "
                "use interactive: false to replay a saved selection."
            )
        plt.show(block=True)
        if not self.accepted:
            raise ValueError("Reference selection cancelled; dataset export was not completed.")
