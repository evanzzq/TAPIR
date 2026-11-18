# Forward model and sampler update

This update is based on `feature/hpc-worker-control` at `e62ffd1`. It includes
Ved's two patches, plus the follow-up changes described below. The existing
empirical STF work in the original Mac checkout is separate from this branch.

## Attenuation configuration

For a 3C run that should sample attenuation, explicitly set:

```yaml
fitAtts: true
attsRange: [0.0, 4.0]
attsStd: 0.4
```

Both settings reach the prior through the runner. `attsStd` is the proposal
width in seconds; it can be omitted to derive a width from the range. To retain
compatibility, omitting `attsRange` still means fixed t* = 1 s in 3C, and a
[0, 4] s range in 1C. Setting a positive width alone does not widen a fixed prior.
The amplitude correction changes the interpretation of previous fitAtts runs;
rerun those inversions for consistent posterior results.

## Other optional YAML settings

`arrStd`, `slwStd`, `ampStd`, `aziStd` (3C only), `logeStd`, and `logeRange`
now pass through to the prior. Omitted settings keep their class defaults.
Widths must be finite and nonnegative. Attenuation bounds must additionally
be nonnegative, finite, and ordered. A zero-width attenuation range is allowed
for deliberately fixed attenuation.

`max_procs` is a positive integer cap on concurrent chains. It is combined
with the existing HPC worker count using the smaller value. For example,
`TAPIR_MAX_WORKERS=8` and `max_procs: 4` run at most four chains concurrently.
`num_chains` still controls the total number of independent chains. Retain
scheduler allocation checks and set numerical library thread counts in the PBS
launcher before Python starts; this setting does not request cluster resources.

## Even-length FFT convention

For a real signal, negative-frequency bins are the conjugates of their positive
partners. DC and, for an even-length DFT, the Nyquist bin are self-conjugate and
must be real. The t* operator uses its real projection at Nyquist, namely
`exp(-pi*f_N*t*) * cos(2*f_N*t*log(f_N / 1 Hz))`. All other bins use the supplied
complex operator. This retains the real part of the proposed Nyquist response,
keeps t* = 0 as identity, and is used by both the full FFT and rFFT paths.
Fractional time shifts retain the existing real-output convention. Broadband
and alternating-sign regression tests cover the bin that smooth pulses can miss.

## Birth proposals

If all 500 arrival-placement attempts fail, `birth` and `birth3c` now return an
unsuccessful proposal without increasing the phase count or changing the input
model. The sampler can continue instead of terminating the chain.

## Git and Terrawulf

The original Mac checkout, its uncommitted empirical STF work, the HPC branch,
and existing cluster bundles are not modified by this update. Before adopting
it on Terrawulf, inspect the checkout and the actual imported code path. Existing
bundles contain their own `code/vespainv` copy; pulling the enclosing repository
does not update that copy. Create a new bundle and output directory for a new
run rather than replacing the code of an existing run.
