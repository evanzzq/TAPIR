"""Construct receiver STF window variants without changing prepared observations.

Run with ``python -m vespainv.stf_sweep --help``. Alignment, baseline and
normalization are frozen on the previously selected anchor window. The legacy
long reference is a separate historical control, not another controlled crop.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
import yaml


def construct_windows(time, observations, trace_ids, selection, start, ends, taper_seconds=0.6):
    """Return symmetric, zero-padded STFs and JSON/YAML-compatible provenance.

    Times are seconds from the untrimmed input record start. Baseline and peak
    weights are estimated once per contributor on the saved anchor window,
    using its saved alignment shifts. All crops share one stack normalization.
    """
    time = np.asarray(time, dtype=float)
    observations = np.asarray(observations, dtype=float)
    dt = float(np.median(np.diff(time)))
    if dt <= 0 or not np.allclose(np.diff(time), dt):
        raise ValueError("Time must increase uniformly.")
    if observations.shape != (len(time), len(trace_ids)) or len(set(trace_ids)) != len(trace_ids):
        raise ValueError("Observation columns and unique trace IDs must agree.")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Observations must be finite.")
    if not np.isfinite(taper_seconds) or taper_seconds <= 0:
        raise ValueError("taper_seconds must be positive and finite.")
    settings = selection["settings"]
    contributors = settings["contributors"]
    if settings["mode"] == "single":
        contributors = contributors[:1]
    if not contributors or len(set(contributors)) != len(contributors):
        raise ValueError("Contributors must be nonempty and unique.")
    if not set(contributors) <= set(trace_ids):
        raise ValueError("Saved contributors are missing from the prepared dataset.")
    lo, hi = settings["window"]
    # Small tolerance avoids dropping mathematically exact grid endpoints.
    first = int(np.ceil(lo / dt - 1e-9))
    last = int(np.floor(hi / dt + 1e-9))
    anchor_indices = np.arange(first, last + 1)
    grid = anchor_indices * dt
    if len(grid) < 5:
        raise ValueError("Anchor window must contain at least five samples.")
    pulses, fixed = [], {}
    shifts = selection["alignment_shifts_s"]
    for trace_id in contributors:
        arrival = float(selection["arrivals"][trace_id]["seconds_from_input_start"])
        shift = float(shifts[trace_id])
        sample_times = arrival + shift + grid
        if sample_times[0] < time[0] or sample_times[-1] > time[-1]:
            raise ValueError(f"Anchor window exceeds available data for {trace_id}.")
        pulse = np.interp(sample_times, time, observations[:, trace_ids.index(trace_id)])
        edge = max(1, min(5, len(pulse) // 10))
        baseline = float(np.mean(np.r_[pulse[:edge], pulse[-edge:]]))
        pulse -= baseline
        scale = float(np.max(np.abs(pulse)))
        polarity = settings.get("polarities", {}).get(trace_id, 1)
        if polarity not in (-1, 1) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid polarity or reference amplitude for {trace_id}.")
        pulses.append(polarity * pulse / scale)
        fixed[trace_id] = dict(
            arrival_s=arrival,
            alignment_shift_s=shift,
            baseline=baseline,
            peak_scale=scale,
            polarity=polarity,
        )
    stack = np.mean(pulses, axis=0)
    stack_scale = float(np.max(np.abs(stack)))
    if not np.isfinite(stack_scale) or stack_scale <= 0:
        raise ValueError("Anchor stack cancels to zero.")
    stack /= stack_scale
    variants = []
    for end in ends:
        if not np.isfinite([start, end]).all() or start >= end:
            raise ValueError("Each requested window must have finite start < end.")
        begin = int(np.ceil(start / dt - 1e-9))
        finish = int(np.floor(end / dt + 1e-9))
        if begin < first or finish > last:
            raise ValueError(
                "Requested window exceeds the fixed anchor; choose a wider anchor first."
            )
        indices = np.arange(begin, finish + 1)
        x = indices * dt
        if len(x) < 5 or x[-1] - x[0] < 2 * taper_seconds:
            raise ValueError("Window is too short for the fixed-duration edge tapers.")
        edge_distance = np.minimum(x - x[0], x[-1] - x)
        taper = 0.5 * (1 - np.cos(np.pi * np.clip(edge_distance / taper_seconds, 0, 1)))
        pulse = stack[indices - first] * taper
        half = max(abs(begin), abs(finish))
        stf = np.column_stack((np.arange(-half, half + 1) * dt, np.zeros(2 * half + 1)))
        stf[indices + half, 1] = pulse
        variants.append(
            (
                stf,
                dict(
                    requested_window_s=[float(start), float(end)],
                    sampled_window_s=[float(x[0]), float(x[-1])],
                    exported_window_s=[float(stf[0, 0]), float(stf[-1, 0])],
                    sample_count=len(stf),
                ),
            )
        )
    return variants, dict(
        anchor_window_s=[float(grid[0]), float(grid[-1])],
        fixed_contributors=fixed,
        stack_peak_scale=stack_scale,
        taper_seconds_per_edge=float(taper_seconds),
        normalization="one shared untapered anchor-stack peak; no per-window rescaling",
        alignment="saved shifts, applied to reference extraction only",
        time_zero="same as saved reference; padding does not recenter the pulse",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=-4)
    parser.add_argument("--ends", nargs="+", type=float, default=[1.5, 2, 2.5, 3, 3.5, 4])
    parser.add_argument("--taper-seconds", type=float, default=0.6)
    args = parser.parse_args()
    source = args.dataset.resolve()
    manifest = yaml.safe_load((source / "dataset.yaml").read_text())
    if (
        manifest.get("array_type") != "receiver"
        or manifest["processing"].get("time_window") is not None
    ):
        raise ValueError("This tool currently requires an untrimmed receiver dataset.")
    selection = yaml.safe_load((source / "stf_selection.yaml").read_text())
    ids = (source / "trace_ids.txt").read_text().splitlines()
    time = np.loadtxt(source / "time.csv", delimiter=",")
    component = selection["settings"]["component"]
    obs = np.loadtxt(source / f"U{component}.csv", delimiter=",")
    variants, shared = construct_windows(
        time, obs, ids, selection, args.start, args.ends, args.taper_seconds
    )
    labels = [f"end_{end:g}".replace(".", "p") for end in args.ends]
    if len(set(labels)) != len(labels):
        raise ValueError("Window endpoints must be unique.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shared.update(
        component=component,
        bandpass=manifest["processing"].get("bandpass"),
        source_selection_sha256=hashlib.sha256(
            (source / "stf_selection.yaml").read_bytes()
        ).hexdigest(),
    )
    cases = []
    for label, (stf, detail) in zip(labels, variants):
        target = args.output_dir / label
        shutil.copytree(source, target)
        # These describe the old selected STF and must not masquerade as the new one.
        for filename in ["stf_selection.yaml", "prep_summary.pdf"]:
            (target / filename).unlink(missing_ok=True)
        np.savetxt(target / "stf.csv", stf, delimiter=",", header="time,stf", comments="")
        provenance = dict(
            schema_version=1, construction="fixed_anchor_window_sweep", **shared, **detail
        )
        (target / "stf_sweep.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
        current = dict(manifest)
        current["reference_wavelet"] = dict(
            kind="empirical",
            file="stf.csv",
            already_processed=True,
            bandpass=shared["bandpass"],
            sha256=hashlib.sha256((target / "stf.csv").read_bytes()).hexdigest(),
            selection_file="stf_sweep.yaml",
        )
        (target / "dataset.yaml").write_text(yaml.safe_dump(current, sort_keys=False))
        cases.append(dict(case=label, **detail))
    (args.output_dir / "sweep.yaml").write_text(
        yaml.safe_dump(dict(shared=shared, cases=cases), sort_keys=False)
    )
    print(f"Created {len(cases)} fixed-alignment references in {args.output_dir}")


if __name__ == "__main__":
    main()
