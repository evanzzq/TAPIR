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
