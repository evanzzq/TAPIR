# Mac preparation and Terrawulf inversion

Use the combined `codex/mac-terra-workflow` branch on both machines. It contains
empirical receiver/source STF selection and reproducible shared receiver window
sweeps, the validated forward-model speedup, and shared/per-trace manual STF
support. Select `mode: per_trace` to export one STF per retained seismogram.

## Mac

The normal checkout is `~/Documents/Research/TAPIR`. In the `tapir` environment,
install it with `python -m pip install -e . --no-deps`. This points the command
line tools and Python imports at this checkout.

Keep machine paths in ignored `configs/workspace.local.yaml`. Personal prep
configs can use ignored names such as `configs/prep_receiver.local.yaml` and
`configs/prep_source.local.yaml`, with `workspace: workspace.yaml`. The tracked
example configs remain portable. Download and preprocess locally, inspect QC
and STF selection, and retain selection metadata for replay.

Copy the entire prepared dataset directory to Terra, including waveform,
noise/covariance, geometry, time, STF, manifest, selection, and trace-ID files.
Copy a campaign YAML separately and adjust its `data_root` and `runs_root` for
Terra. A Mac absolute path in a YAML will not work on Terra. For example:

```bash
scp -r /path/to/prepared/EVENT ziqi@terrawulf.anu.edu.au:~/projects/TAPIR/RealData/
scp /path/to/campaign.yaml ziqi@terrawulf.anu.edu.au:~/projects/TAPIR/configs/
```

## Terra

Do not switch or update this checkout while jobs using it are queued or running.
Historical frozen sweep bundles keep their own code and are unaffected.

For the initial transition from `codex/trace-specific-stf`, first inspect
`git status`. Preserve any tracked edits before switching; untracked personal
configs and data can remain. Then:

```bash
cd ~/projects/TAPIR
git fetch origin
git switch --track origin/codex/mac-terra-workflow
conda activate tapir
python -m pip install -e . --no-deps
git rev-parse --short HEAD
python -c 'import vespainv; print(vespainv.__file__)'
```

Later code updates need only `git pull --ff-only` on this branch. Reinstall if
packaging or dependency requirements change. Data and configs still travel
separately. Validate a prepared 3C receiver dataset before submission:

```bash
tapir validate-data /home/ziqi/projects/TAPIR/RealData/EVENT --cdopt 3 --manual-stf
```

Choose `--cdopt` to match the campaign, and omit `--manual-stf` for generated
wavelets. Start with a short campaign before a long run. Submit with absolute
paths (replace `campaign.yaml`):

```bash
mkdir -p ~/projects/TAPIR/logs
qsub -o /home/ziqi/projects/TAPIR/logs/ \
     -e /home/ziqi/projects/TAPIR/logs/ \
     -v TAPIR_CONFIG=/home/ziqi/projects/TAPIR/configs/campaign.yaml \
     scripts/hpc/terrawulf.pbs
```

The PBS launcher logs the Git revision and uses the allocated CPUs with one
numerical-library thread per chain. Use distinct output run names to preserve
previous results. Keep the launch log, config, prepared data/metadata, and
`stf_used.csv` with results so later comparisons identify the exact inputs.

## Optional fixed time shifts per trace

Supply externally calculated corrections with the run setting
`trace_time_shifts_file`. TAPIR does not calculate depth corrections. The CSV
must contain exactly these two columns, with one row for every ID in the
prepared dataset's `trace_ids.txt`:

```csv
trace_id,time_shift_s
event_reference,0.0
event_other,1.25
```

Set the file in `defaults` or an individual experiment:

```yaml
trace_time_shifts_file: ../trace_time_shifts.csv
```

The path is relative to the campaign YAML, not to the dataset directory. Rows
are matched by ID, so file row order need not match waveform columns. Missing,
extra, duplicate or empty IDs and nonfinite shifts fail before chains start.
Omit the option or set it to `null` to use zero corrections. An explicitly
supplied file requires unique dataset trace IDs even with a shared/generated STF.

**Positive seconds delay the synthetic waveform.** For trace i and phase j,
TAPIR constructs arrivals as `arrival_j + slowness_j * offset_i + time_shift_i`.
The same fixed delay applies to every phase and Z/R/T component for that trace,
using Fourier phase factors for fractional samples. Observed data and STFs are
not shifted. This works with shared, trace-specific and generated STFs, and
receiver, source and Mars arrays, including the source-array location-perturbation
fallback. For a depth-correction file, the approximation must be appropriate for
the phases being fitted; TAPIR treats these as generic delays.

Choose a reference event with a zero delay when calculating relative corrections
outside TAPIR. Changing the reference adds a common timing offset, so reference
arrival priors must still cover the resulting arrivals. Do not also apply the
same correction to the observations or STF.

Each configured run saves `trace_time_shifts_used.csv` in waveform-column order
and stores the delays and IDs in `Bookkeeping.pkl`. Reconstruct waveforms using
that saved bookkeeping to retain corrections in posterior plots. Historical
bookkeeping files without delays use zero. Reusing a run name with different
corrections is rejected; choose a new run name instead.
