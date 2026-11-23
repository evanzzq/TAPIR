# Forward-model speedup validation — 23 November 2026

The implementation at `617741b32c6fda862cc710155c9185fc51aca364` is validated for
numerical correctness and the observed Terrawulf performance improvement.
The historical and corrected sweeps retain the dominant posterior structure;
this is not a claim that the complete posteriors are identical or converged.

## Matched comparison

The two frozen STF bundles contain the same eight Earth receiver-array cases,
25 stations, 350 samples, 5 Hz sampling and eight chains per case. Both runs use
1,000,000 steps, 800,000 burn-in steps, 200 saved models per chain and maxN=20.
All 128 chains (64 per version) completed with finite matching-length scalar
histories. All 25,600 saved models were included in the phase/parameter summaries.

Data files, YAML configs, actual used STFs, saved prior fields, Bookkeeping
fields and effective scientific settings are identical. Each case pair ran on
the same named node with eight allocated slots. The corrected launch manifests
match the locally frozen bundle manifest:
`a4bb12825c72a2a26b9e8e19f7091790fa4898a2eb8f10ddcc4c2cfd2051d19f`.
The historical runner's empirical-STF filtering guard is retained in that bundle;
its full contents are defined by its manifest. The branch waveformBuilder.py and
rjmcmc.py are byte-identical to the corrected bundle copies.

## Numerical checks

On 1,024 actual saved models, sampled equally from both sweeps and all cases:

- Vectorized versus corrected-loop maximum relative waveform L2 difference:
  4.93e-15, with both attenuation and phase fitting enabled.
- Maximum corresponding covariance-weighted L1 log-likelihood difference:
  7.28e-12 in absolute units.
- With attenuation disabled, the vectorized model versus the untouched
  historical loop differs by at most 5.05e-15 relative L2.
- All 117 repository tests pass in the Python 3.11 TAPIR environment. These
  include receiver/source/Mars geometries, odd/even lengths, 1C/3C, attenuation
  and phase options, Nyquist/DC handling, failed births and YAML controls.

The same corrected likelihood is thus evaluated to round-off by the two paths.
This is stronger evidence for the vectorization than agreement between finite,
partially mixed MCMC chains.

Reproduce the real-model checks with:

```bash
python scripts/validation/check_saved_forward_models.py \
  --historical-bundle /path/to/terrawulf_receiver_stf_sweep_20111122_20261111 \
  --corrected-bundle /path/to/terrawulf_receiver_stf_sweep_speedup_20261118 \
  --output /path/to/forward_likelihood_validation.json
```

## Observed full-chain timing

Times are medians of eight chains per case, from their final progress checkpoint.
They exclude process startup and final summary plotting. The case pairs used
the same named nodes, but trajectories and machine load were not controlled.

| STF case | Historical hours | Corrected hours | Observed speedup | New/old median absolute P amplitude |
|---|---:|---:|---:|---:|
| end_1p5 | 57.95 | 6.12 | 9.47x | 0.539 |
| end_2 | 58.22 | 6.11 | 9.52x | 0.464 |
| end_2p5 | 58.03 | 6.11 | 9.50x | 0.514 |
| end_3 | 58.43 | 6.09 | 9.60x | 0.477 |
| end_3p5 | 58.64 | 6.09 | 9.64x | 0.572 |
| end_4 | 58.47 | 6.11 | 9.56x | 0.459 |
| long_original | 58.76 | 6.14 | 9.57x | 0.507 |
| dgaussian | 61.14 | 6.55 | 9.33x | 0.513 |

The full-chain speedup is 9.3–9.6x (approximately 58–61 hours down to 6.1–6.6
hours). The earlier same-model forward-call benchmark on Terrawulf measured 22x.

## Posterior comparison

The main P slowness concentration remains near 6.15 s/degree in all cases. The
largest change in the pooled median P slowness is 0.027 s/degree. This is a
summary over saved P-phase samples, not a matched physical-arrival estimate.
Mean old/new predicted Z/R waveforms have correlations of 0.961–0.988 across
all stations/times, using 20 evenly spaced saved draws per chain (160 per case).
Relative L2 residuals remain similar. Mean predictions still leave substantial
signal unexplained; these comparisons do not establish adequate waveform fits.

Corrected median absolute P amplitudes are 0.46–0.57 of the historical values,
consistent with removing the old attenuation operator's half-amplitude bug.
The stored attenuation parameter distributions and dominant P slowness band
remain broadly similar. For P phases the applied operator uses 0.25 times the
stored atts parameter, as before; plots label the stored parameter explicitly.
DC and Nyquist corrections are also included. With unchanged amplitude-prior
bounds, the old and corrected targets are not exactly related by rescaling.

A descriptive Jensen–Shannon distance between smoothed, normalized pooled P
phase maps is smaller than the median four-chain-versus-four-chain distance
within each sweep for every case. This supports broad compatibility at the
scale of existing chain variability. It is not a formal posterior-equivalence
test: the two summaries use different pool sizes, phase intensities are not
physical-arrival probabilities, and chains are not independent posterior draws.
The S features are weaker and less stable across chains and versions.

Both versions retain median N=20 for all cases. The corrected chains spend
93.0–100.0% of retained steps at that bound, versus 97.0–99.3% historically.
Corrected noise-scale rank-normalized split R-hat remains above one, reaching
1.326. Phase-count R-hat is not estimable from the saved dGaussian models because
all have N=20; that constancy is not evidence of mixing.

## Decision and scope

Retain this as the validated forward-speedup/t* implementation. General
agreement of the dominant posterior features, expected amplitude changes,
round-off forward/likelihood agreement and successful full runs support adoption.
Do not describe it as reproducing the uncorrected posterior exactly, or as
resolving the sampler's phase-count ceiling and mixing limitations. Full-chain
empirical validation here covers this Earth receiver sweep; other geometries
have numerical unit coverage, not equivalent long-chain field validation.

Machine-readable metrics are in validation/forward-speedup-20261123.json.
The full plots and input-dependent scripts remain under the local
stf_speedup_comparison_20261123 analysis directory, separate from source runs.
