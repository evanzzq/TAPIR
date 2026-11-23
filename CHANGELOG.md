# Changelog

This project follows semantic versioning beginning with the first public
release. Changes currently target version 0.1.0.

## 0.1.0 - Unreleased

- Package and command-line interfaces for Earth and Mars inversions.
- Earth receiver-array and source-array SAC preparation.
- TAPIR-compatible dataset validation.
- Dataset manifests and consistent STF bandpass filtering.
- Incremental likelihood and phase-count diagnostics.
- Run summary command and self-contained example datasets.
- Stable covariance whitening and configurable free-surface velocities.
- Optional vector rendering for publication-quality vespagram PDF and SVG output.
- t* operator is now Hermitian: attenuated phases keep their full amplitude
  (previously halved) and t* = 0 is the identity. Optional `attsRange` and
  `attsStd` run settings set the t* prior.
- Vectorized 1C and 3C forward models: all traces and phases are evaluated
  at once, about 10-30 times faster per call. The per-trace loop is kept for
  source-array `locDiff`.
- Real Nyquist t* factors for even-length signals, with regression coverage.
- Crowded-window birth proposals are rejected rather than terminating a chain.
- Optional YAML proposal widths, `logeRange`, and `max_procs`; the worker cap
  composes with the existing HPC `TAPIR_MAX_WORKERS` limit.
- Manual STFs can now contain one waveform per observed trace, in positional
  column order, while shared and generated STFs remain supported. Validation,
  filtering, sampling, snapshots and waveform reconstruction retain every column.
- Generated wavelets no longer overwrite an existing manual stf.csv; each run
  still saves its actual reference in stf_used.csv.
