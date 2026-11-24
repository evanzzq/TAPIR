# Empirical receiver-array references

`prep-earth` can build a common empirical reference after the existing noise/SNR
QC, using one station or a selected stack. Source-array wavelet extraction and
inversion with per-event references are not implemented yet. Future references
should be linked to stable event IDs rather than relying on column order alone.

Enable the optional preparation stage with `--empirical-stf` or a top-level
`empirical_stf` mapping in the preparation YAML. See
`configs/example_prep_empirical_receiver.yaml`. The original noninteractive
preparation remains available when this setting is absent or false.

## Timing and processing

The reference arrival is resolved in this order:

1. `arrival_times: {NET.STA: seconds_from_input_start}` overrides.
2. SAC `a` or `t0`–`t9` with a phase label matching `phase`. `pick_header: a`
   (or `t0`, etc.) explicitly selects a header even if it has no label.
3. The earliest predicted arrival of `phase` from the chosen TauP `model`.

SAC picks and origin offset `o` are interpreted relative to SAC reference time,
not blindly as offsets from the trace start. Prediction needs event depth,
distance (or coordinates), and the SAC origin offset. Depth defaults to SAC's
kilometre convention; explicitly use `depth_unit: m` for files storing metres.
Unphysical depths fail with an explanatory error, without guessing units.

When empirical extraction is enabled, preparation filters the full available
signal/noise records before optional trimming, retaining the wider signal record
for reference selection. The reference uses the prepared sample interval. The
existing integer-factor downsampling behaviour is unchanged; choose a suitable
input bandwidth for the output sample rate. Preparation does not remove an
instrument response or rotate N/E into R/T.

The interface aligns only the extraction previews. Exported observations keep
their original relative timing, and `time.csv` still starts at the prepared
window's beginning. Zero on the reference axis denotes the selected reference
pick/prediction, with optional stack refinement relative to the first contributor.

## Interactive selection

Run preparation from a terminal with an interactive Matplotlib backend, such as
MacOSX or QtAgg. A notebook's inline backend and Agg cannot accept this review.

- Click an overview trace to make it active (orange).
- **Keep/drop** changes whether that station is exported. Automatic QC rejections
  have already been removed before this view.
- **Use/unuse** changes whether a kept station contributes to the reference.
- Choose **single** to use the active kept station, or **stack** to average the
  selected contributors. **Use all kept** restores all contributors in stack mode.
- Drag horizontally in the active-trace preview to choose the extraction window.
  Its bounds apply relative to each station's reference arrival.
- **Refine alignment** enables bounded, signed cross-correlation alignment against
  the first contributor. `max_shift` sets the bound in seconds. No automatic
  polarity reversal is performed; **Flip polarity** changes the reference copy
  of the active trace only. Inspect alignment and polarity before accepting.
- The export preview shows the aligned contributor pulses in grey and the final
  tapered, normalized reference in black.
- **Save & finish** accepts the selection and exports. Closing the window cancels
  before dataset export. Existing output directories are not cleaned automatically.

Component selection (`Z`, `R`, or `T`) is configured with `component`.
`view_window` controls the initial displayed interval. `window` sets the initial
extraction interval. Put window edges in a quiet baseline and include the entire
pulse: baseline is estimated from up to five samples at each edge. Each contributor
is baseline-corrected and peak-normalized before averaging. A Tukey taper uses
`taper_fraction` at each end, then the result is peak-normalized. Symmetric zero
padding preserves time zero for TAPIR's FFT forward model; it does not add signal.

The extraction window must be inside every contributor's available data and contain
at least five samples. The padded reference must fit in the final inversion window.

## Outputs and replay

- `stf.csv`: `time,stf`, ready for `man_stf: true`.
- `stf_selection.yaml`: settings, contributing/retained IDs, predicted or picked
  arrivals, alignment shifts, processing, and QC exclusions.
- `trace_ids.txt`: exported station IDs in exact waveform/metadata column order.
- `dataset.yaml`: processing information and an empirical-reference identity.

The runner recognizes the saved empirical reference and does not filter it again,
even when the observations were bandpassed. It checks the file digest and passband
to prevent silently treating a replacement file as already processed. Original
manual/generated STF behaviour is preserved for other datasets.

For reproducible, noninteractive extraction, provide explicit `included_ids`,
`contributors`, and `window` with `interactive: false`, or replay the saved choice:

```yaml
empirical_stf:
  selection_file: /absolute/path/to/stf_selection.yaml
```

Replay uses the same preparation input and processing settings as the original
run. The saved choices are ID-based; IDs missing after QC produce an error instead
of selecting the wrong columns. Put output in a separate dataset folder when
comparing against a Gaussian/dGaussian run. Keep an explicit, unchanged `minSpace`
for the first comparison so that changing the waveform does not also change the
minimum permitted phase spacing.

### Controlled receiver-array window sweep

After selecting a common receiver reference, construct several window endpoints
without rerunning QC or changing observations:

```bash
python -m vespainv.stf_sweep \
  --dataset RealData/your_receiver_dataset \
  --output-dir local/your_window_sweep \
  --start -4 --ends 1.5 2 2.5 3 3.5 4 --taper-seconds 0.6
```

The output directory must be new. This command currently requires an untrimmed
receiver dataset with `trace_ids.txt`, `stf_selection.yaml`, and its prepared
component CSV. It interpolates the already-filtered observations at the saved
arrival predictions plus saved alignment shifts. It does not filter observations
or references again, rerun alignment, or change the data used by the inversion.

Per-trace baselines (mean of the saved anchor window's edge samples), peak weights,
polarities, and the stack normalization are calculated once on the anchor window.
Each variant crops that same untapered stack, applies a raised-cosine taper of
fixed duration at both ends, and zero-pads to a symmetric time axis. There is no
per-window peak rescaling. Thus shared interior samples are identical across
variants; only the right-hand taper and retained tail differ when start is fixed.
Endpoints are snapped inward to the sampling grid, with requested, sampled, and
exported ranges recorded separately in `stf_sweep.yaml` and the sweep index.

Requested windows must fit inside the anchor window. For more distant endpoints,
first select a wider anchor; the tool never extrapolates. Each variant includes
unchanged copies of observations, noise, covariance and station files. Its manifest
marks the empirical reference already processed and includes its hash. The old
interactive selection and summary plot are omitted from variant folders because
they describe a different STF. Sweep provenance is not an interactive-selection
replay file. Keep an original long reference as a separately labelled historical
control: it uses the original taper convention and is not a controlled crop.
