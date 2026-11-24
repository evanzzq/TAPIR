# TAPIR-compatible data

TAPIR waveform CSVs contain samples in rows and traces in columns. Files have
no header unless noted below, and all values must be finite.

## Waveforms and time

- `time.csv`: one strictly increasing, uniformly sampled time vector in seconds.
- `UZ.csv`, `UR.csv`, `UT.csv`: three-component data.
- `U.csv`: optional generic one-component data. Otherwise a one-component run
  reads `UZ.csv`, `UR.csv`, or `UT.csv` according to the selected component.

All selected waveform files must have the same number of rows as `time.csv`
and the same number of trace columns.

## Geometry

`eventinfo.csv` has the header `lat,lon` and one coordinate row.

Earth receiver arrays use `station_metadata.csv`, with header `lat,lon` and one
receiver row per waveform column.

Earth source arrays and Mars use `station_metadata_db.csv`, with header
`dist_deg,baz` and one source row per waveform column. Distance is in degrees.
`baz` is the azimuth from the single receiver toward the source, in degrees
clockwise from north. In these workflows `eventinfo.csv` identifies the single
receiver location.

## Covariance and source-time function

With `CDopt: 3`, TAPIR reads `CD_UZ_fit.csv`, `CD_UR_fit.csv`, and
`CD_UT_fit.csv` for 3C data. Each is a symmetric square matrix with one row and
column per time sample. One-component runs read only their selected component.

With `man_stf: false`, TAPIR generates one shared Gaussian/dGaussian wavelet,
using the existing `stfshape` setting. An existing manual STF file is ignored
and preserved, so switching this flag does not overwrite a supplied matrix.
If no `stf.csv` exists, the generated reference is also written there for
compatibility. The run's `stf_used.csv` is always the authoritative reference
for reconstruction, regardless of whether `stf.csv` already existed.

With `man_stf: true`, `stf.csv` has a header and a **time column first**:

```csv
time,stf_1,stf_2,stf_3
-0.4,0.0,0.0,0.0
-0.2,0.2,-0.1,0.3
0.0,1.0,-0.5,0.8
0.2,0.2,-0.1,0.3
0.4,0.0,0.0,0.0
```

One amplitude column (`time,stf`) is shared across every trace. Otherwise there
must be exactly as many amplitude columns as seismogram columns. STF amplitude
column i belongs to waveform column i and metadata row i, for either receiver
or source arrays. The time column is excluded from this count. For 3C data, a
trace's STF is used for all its Z/R/T components and all modeled phases; this
format does not specify a separate STF per component or phase.

Column ordering is positional; header labels are not used to reorder traces.
Apply the same trace selections/reordering to the observations, STFs and
metadata. Matching column counts cannot detect swapped source/station identities.

Every STF must be real, finite and nonzero. All STFs share the time coordinate,
which must increase uniformly with the same sample interval as `time.csv`.
The STF need not have as many rows as the observations, but cannot be longer.
Pad shorter references with zeros onto this common grid. Preserve their common
zero-time reference when padding; no independent peak alignment is performed.
The historical centering convention (offset by the last STF time) is retained:
use the existing symmetric window about zero, rather than assuming an arbitrary
STF start time will be corrected automatically. Gaussian/dGaussian generation
and existing shared-wavelet centering have not been changed.

Column-count, timing, non-finite, zero-wavelet and excessive-length problems
raise errors before sampling. `tapir validate-data --manual-stf` checks these
rules. `stf_used.csv` in each run records the exact time and every STF column
passed to the sampler, with a matching header. Waveform reconstruction uses
all columns as well.

Without filtering, supplied STF amplitudes are preserved. When the runner
bandpasses the STF, it filters each column along time and applies one common
peak-normalization scale to the whole matrix. This preserves relative column
amplitudes and polarities, and gives the existing normalization for a shared
STF. If only shape differences are intended, normalize each reference before
saving it. Supplied amplitude ratios affect the inferred shared phase amplitudes.
When `minSpace` is omitted/null, the multi-STF default uses the widest wavelet
at 1% of each column's own peak; set `minSpace` explicitly to choose another
separation rule. The existing shared-STF width rule is retained.

Earth preparation writes `dataset.yaml`, including `processing.bandpass`. The
runner reads this value and applies it to the STF. For externally prepared Mars
data, declare the same `bandpass: [fmin, fmax]` in the run YAML. If both the run
config and an Earth dataset manifest supply passbands, they must match.
An empirical reference marked `reference_wavelet.already_processed` in the
manifest is verified against its recorded passband and file SHA-256, then used
without a second filter. This protection applies to the complete STF file,
including all trace-specific columns.

Receiver preparation can optionally construct an empirical reference after QC;
see [empirical-reference.md](empirical-reference.md). It writes `stf.csv`,
`stf_selection.yaml`, and `trace_ids.txt` (one stable station ID per waveform
column). The manifest's `reference_wavelet` entry records that this particular
STF is already processed, its passband, and its SHA-256 digest. With `man_stf:
true`, the runner verifies these values and skips a second bandpass. Generated
wavelets and manual files without this entry retain the usual filtering behaviour.
Replacing a recorded empirical `stf.csv` requires rebuilding it or removing the
`reference_wavelet` entry if the replacement should be treated as unprocessed.

## Validation examples

```bash
tapir validate-data DATASET
tapir validate-data DATASET --source-array --cdopt 3
tapir validate-data DATASET --mars --cdopt 3 --manual-stf
tapir validate-data DATASET --no-is-3c --component Z
```
