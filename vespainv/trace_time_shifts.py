"""Externally supplied trace delays, independent of their physical origin."""
import csv
from pathlib import Path

import numpy as np


def load_trace_time_shifts(path, dataset_dir, n_traces):
    """Return IDs and delays in waveform-column order; positive means later."""
    ids_path = Path(dataset_dir) / "trace_ids.txt"
    ids = ids_path.read_text(encoding="utf-8-sig").splitlines()
    ids = [value.strip() for value in ids]
    if len(ids) != n_traces or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("trace_ids.txt must contain one unique, nonempty ID per waveform column.")
    values = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["trace_id", "time_shift_s"]:
            raise ValueError("Trace time shifts require the CSV header trace_id,time_shift_s.")
        for row in reader:
            trace_id = (row.get("trace_id") or "").strip()
            if not trace_id or trace_id in values:
                raise ValueError(f"Empty or duplicate trace ID in time shifts: {trace_id!r}.")
            if None in row:
                raise ValueError(f"Extra columns in time shifts for {trace_id!r}.")
            try:
                value = float(row["time_shift_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid time_shift_s for {trace_id!r}.") from exc
            if not np.isfinite(value):
                raise ValueError(f"time_shift_s must be finite for {trace_id!r}.")
            values[trace_id] = value
    missing = sorted(set(ids) - values.keys())
    extra = sorted(values.keys() - set(ids))
    if missing or extra:
        raise ValueError(f"Trace time shift IDs do not match dataset: missing={missing}, extra={extra}.")
    return ids, np.array([values[key] for key in ids], dtype=float)


def trace_time_shifts(bookkeeping, n_traces):
    """Also accept historical Bookkeeping pickles without this optional field."""
    shifts = getattr(bookkeeping, "trace_time_shifts", None)
    if shifts is None:
        return np.zeros(n_traces)
    shifts = np.asarray(shifts, dtype=float)
    if shifts.shape != (n_traces,) or not np.isfinite(shifts).all():
        raise ValueError("trace_time_shifts must contain one finite delay per trace.")
    return shifts


def save_trace_time_shifts(path, ids, shifts):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trace_id", "time_shift_s"])
        writer.writerows(zip(ids, shifts))
