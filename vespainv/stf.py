"""Shared and trace-specific STF validation and CSV conventions.

Tables have a time column followed by one shared amplitude column, or exactly
one amplitude column per observed trace. All columns share one time grid.
"""
from pathlib import Path

import numpy as np


def validate_stf_samples(time, amplitudes, *, n_traces=None, dt=None, n_samples=None):
    """Validate a real STF array and return amplitudes as (samples, columns)."""
    if np.iscomplexobj(time) or np.iscomplexobj(amplitudes):
        raise ValueError("STF time and amplitudes must be real.")
    time = np.asarray(time, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, None]
    if (time.ndim != 1 or len(time) < 2 or amplitudes.ndim != 2
            or amplitudes.shape[0] != len(time) or amplitudes.shape[1] < 1):
        raise ValueError("STF must have at least two time samples and one amplitude column.")
    if not np.isfinite(time).all() or not np.isfinite(amplitudes).all():
        raise ValueError("STF time and amplitudes must contain finite numeric values.")
    intervals = np.diff(time)
    stf_dt = float(intervals[0])
    if (np.any(intervals <= 0)
            or np.any(np.abs(intervals - stf_dt) > 1e-12 + 1e-6 * abs(stf_dt))):
        raise ValueError("STF time samples must be strictly increasing and uniformly sampled.")
    if dt is not None and abs(stf_dt - dt) > 1e-12 + 1e-6 * abs(dt):
        raise ValueError(f"STF sample interval {stf_dt:g} must match data sample interval {dt:g}.")
    if n_samples is not None and len(time) > n_samples:
        raise ValueError(f"STF has {len(time)} samples but data have only {n_samples}; shorten its support.")
    columns = amplitudes.shape[1]
    if n_traces is not None and columns not in (1, n_traces):
        raise ValueError(
            f"stf.csv has {columns} STF amplitude columns; expected 1 shared column or "
            f"{n_traces} columns matching the waveform traces (excluding the time column)."
        )
    if np.any(np.max(np.abs(amplitudes), axis=0) == 0):
        raise ValueError("Each STF amplitude column must contain a nonzero waveform.")
    return amplitudes


def validate_stf_table(table, **kwargs):
    if np.iscomplexobj(table):
        raise ValueError("STF time and amplitudes must be real.")
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("stf.csv must contain a time column followed by STF amplitude column(s).")
    validate_stf_samples(table[:, 0], table[:, 1:], **kwargs)
    return table


def load_stf(path, **kwargs):
    """Read the existing headered time/amplitude format, including multiple STFs."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig") as handle:
            first = handle.readline().lstrip("# ").strip().split(",")[0]
        if first.lower() != "time":
            raise ValueError("The first header column must be 'time', followed by STF amplitude columns.")
        table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    return validate_stf_table(table, **kwargs)


def stf_header(table):
    n = table.shape[1] - 1
    return "time,stf" if n == 1 else "time," + ",".join(f"stf_{i+1}" for i in range(n))
