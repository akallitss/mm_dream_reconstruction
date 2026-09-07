# mm_strip_reconstruction
Reconstruction algorithms build specifically for Micromegas 2D strip detectors built at CEA Saclay.

## Build

Use the **Release** build for processing — it is ~12× faster than the debug
build (which was unknowingly used for all processing before 2026-07-24):

```
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

`orchestrator/process_run.py` points at `cmake-build-release/`. Reference
timing: one 13k-event non-ZS FEU file (32 samples × 512 ch) analyzes in ~12 s
(was ~160 s debug); remaining time is ~⅓ ROOT decompression, ~⅓ CNS medians.

## Pipeline

`decode` (raw fdf → `nt` sample tree) → `analyze_waveforms` (→ per-FEU `hits`
tree) → `combine_feus_hits` (→ combined hits with `feu` branch) →
`clusterize1d`. `orchestrator/process_run.py` drives the chain for a run.

## Waveform analysis (`waveform_analysis/`)

```
analyze_waveforms <input.root> [output.root] [pedestal.root] [--tps <ns>] [--cns <0|1>]
                  [--thr <sigma>] [--mf <samples>] [--zs-baseline <0|1>]
```

Per event: pedestal subtraction → densify zero-suppressed samples → optional
common-noise subtraction (median per 64-channel block per sample; ON by
default) → pulse finding → one `hits` entry per pulse.

Pedestals: mean = raw baseline per channel; RMS = noise measured **after** CNS
(the data path is CNS-subtracted, so thresholds must be calibrated on post-CNS
noise). Without a pedestal file the data is assumed zero-suppressed
(baseline 256, RMS 1). CNS on zero-suppressed (sparse) data is signal-biased
(the block median *is* signal → ~8× hit loss), and ZS data is already
common-mode corrected on the FEU, so the analyzer **auto-detects ZS input**
(median distinct channels/event < 256) and **forces CNS off regardless of
`--cns`** — a global CNS setting is therefore safe across mixed ZS/RAW
reprocessing.

`--zs-baseline 1` (ported from the banco P2 fork, 2026-07-24): for
zero-suppressed data whose pedestals were already subtracted **on the FEU**
(waveforms re-centred at 256), subtract the uniform 256 baseline instead of the
pedestal file's per-channel raw means — those differ from 256 by tens of ADC
per channel and would shift every threshold and amplitude by that much. The
pedestal file is still used for the per-channel noise RMS, so thresholds stay
per-channel 5 σ instead of the no-pedestal fallback's fixed 5 ADC. ZS data
*without* on-FEU pedestal subtraction sits on the raw per-channel baselines,
where the pedestal means ARE the right baseline — so the flag must follow the
DAQ configuration: `process_run.py` (and the DAQ watcher) set it automatically
when `run_config.json` has `dream_daq_info.zero_suppress` **and**
`dream_daq_info.pedestal_subtraction` both true.

### Pulse finding (rewritten 2026-07-24)

Single unified algorithm (replaces the previous legacy-threshold /
derivative-trigger modes; the derivative trigger silently lost 22–46 % of
genuine hits — slow risers never passed its slope threshold, and smoothing lag
placed seeds before the amplitude-threshold crossing, discarding even large
pulses):

1. **Gate** — contiguous runs of samples above `thresholdSigma` (5) × noise RMS.
2. **Bridge** — sub-threshold gaps ≤ `gapMergeSamples` (2) don't end a pulse.
3. **Pile-up split** — recursively split a run at the most prominent interior
   valley (depth ≥ `splitProminenceSigma` (4) × RMS below *both* flanking
   maxima), only if both halves stay ≥ `minSamplesForPeak` wide. A pulse on
   the tail of another makes peak–valley–peak, so pile-up (e.g. nTOF
   gamma-flash regions) is still separated; noise wiggles never reach the
   prominence cut.

Per pulse: local baseline = **median** of up to `baselineLeftWindow` (4)
pre-pulse samples, skipping the `baselineGapSamples` (2) immediately before the
pulse start — those sit on the sub-threshold rise of slow pulses (at a pile-up
split boundary: the valley level, i.e. the previous pulse's tail); amplitude
from a 3-point parabola (or, when ≥
`minSatSamples` (2) consecutive samples sit near ADC max, from least-squares
edge-slope extrapolation); timing at `timingPercentMax` (30 %) of the peak on
the leading edge; threshold crossings interpolated (float); **integral over the
full pulse span** — from where the waveform leaves the baseline to where it
returns, clamped by neighbouring pulses — so the sub-threshold rise and tail
are included (the old between-crossings sum kept only ~half the charge of a
5–10 σ pulse).

### `hits` tree branches

> **`time` is not a drift-time measurement (2026-07-28).** On resistive-strip
> detectors each strip's waveform is its own charge plus delayed, dispersed
> copies of its neighbours' (~29 % at tau ~ 47 ns to +-1 strip on MX17), so the
> per-strip `time` is an *aggregate*. Fitting strip position vs `time` gives a
> ladder compressed 20-30 % — ~4 deg too steep, with the cluster fanning away
> from the true track with depth — and this is independent of how the time is
> extracted (rising edge, CFD, matched filter all show it). Use the hits tree
> for cluster finding, amplitudes, efficiency and QA; reconstruct geometry from
> the waveforms. Evidence and the replacement (forward-model fit):
> `nTof_x17/RECONSTRUCTION_BASIS.md`.

`eventId, trigger_timestamp_ns, channel, amplitude, time, time_of_max, sample,
max_sample, local_baseline, local_max, left_sample, right_sample,
time_over_threshold, integral, saturated, trunc_left, trunc_right,
significance` (+ `feu` after combining). Times are ftst-corrected ns; `left/right_sample` and
`time_over_threshold` are interpolated (not integer samples). `trunc_left` =
pulse already above threshold at the first sample (rise unrecorded — treat
`time` as unreliable); `trunc_right` = still above threshold at the last sample
(tail clipped — `integral`/`amplitude` are floor estimates). The combiner
tolerates older hits files without the trunc branches (defaults false).

### Matched-filter gate (DEFAULT) and the low-gain nTOF mode

Pulse finding (gate, gap-merge, valley split, crossings/TOT) runs **by
default** on a boxcar-smoothed copy of the waveform against the
*smoothed*-noise sigma, measured per channel in the same pedestal pass
(`rms_gate` branch of the pedestals tree). Amplitude, baseline, timing and
integral are still measured on the raw waveform. The gate width is **auto**:
~300 ns (the DREAM shaper pulse width) divided by `--tps` → 5 samples at
60 ns, 15 at 20 ns; override with `--mf <samples>`, or `--mf 0` for the
pre-2026-07-24 raw-waveform gate. Auto-disabled on zero-suppressed data and
when no pedestal file is given (gate sigma unmeasurable; boxcar would dilute
the sparse ZS islands).

**The gate only triggers/segments — every physics quantity except the
crossings (amplitude, baseline, timing, integral, saturation) is measured on
the raw, unsmoothed waveform.**

`process_run.py` refines the auto width from the run's actual **Dream shaping
time**: it locates the DAQ config (the exact `*.cfg_cpy` the DAQ writes next
to the raw data on the nTOF layout, else the template named in
`run_config.json` resolved against a local `Cosmic_Bench_DAQ_Control`
checkout), decodes the per-FEU peaking time from Dream register 1 (code =
bits [7:4] of the second word; table in `DREAM_PEAKING_NS`), and passes
`--mf round(1.7 × Tp / tps)` per FEU (1.7 = measured pulse-FWHM/peaking).
MX17 FEUs at 180 ns peaking give width 5 at 60 ns / 15 at 20 ns — identical
to the auto default — while e.g. the M3 FEU (283 ns peaking) correctly gets 8.

Rationale: real pulses are many samples wide while the noise tail is dominated
by narrow excursions, so the smoothed gate beats the raw gate on BOTH axes.
Measured on June det3 FEU08 at the same nominal 5-sigma threshold: fakes
0.67 % vs 1.05 % per waveform (pedestal data), recovery of real >5-sigma-raw
candidates up in every bin (5–7σ 40 % vs 11 %, 20–50σ 97 % vs 92 %, >50σ 99 %
vs 96 %), CPU +9 %. Injection truth (measured pulse shape into real pedestal
noise): at ~1 % fakes, 4-sigma pulses 42 % vs 10 %. The noise is
time-correlated, so the smoothing gain is NOT 1/sqrt(k) (boxcar-5 sigma ratio
~0.8); zero-mean/derivative kernels were tested and lose (the pulse and the
correlated noise overlap spectrally).

**Mode split is by threshold only**: default `--thr 5`; **low-gain nTOF
micro-TPC: `--thr 3`** (~2 % fake waveforms/channel ≈ 10 isolated fake
strips/event on a 512-channel FEU — downstream clustering/time-coincidence
must absorb that; on normal-gain data the added sub-5-sigma hits are
noise-dominated, so only run low at low gain). Every hit carries a
`significance` branch (gate peak / gate noise sigma), so a single
low-threshold processing can serve strict analyses too: cutting
`significance >= 5` offline recovers the high-rejection hit set.

Note: with the gate smoothed, `left_sample`/`right_sample`/
`time_over_threshold` describe the smoothed waveform's threshold crossings
(median TOT is ~2 samples wider than the raw-gate convention). Hit sets
produced before/after this default change are not directly comparable —
reprocess consistently within a campaign.

Investigated and deliberately left unchanged (June det3 data, 2026-07-24, so
this analysis isn't redone): the CNS `nth_element` upper-middle median
convention (offset −0.1 ADC, negligible); MAD/robust-noise thresholds (the
pedestal noise tails are genuine — P(>4·MAD) is ~275× Gaussian — so a
lower robust threshold would roughly double the fake-waveform rate; the
std-based RMS stays); pre-timing smoothing (only ~5 % jitter improvement in a
shape+real-noise MC); the 3-point parabola amplitude (bias ≤2.5 %).

TODO: expose the analyzer configuration (thresholds, split prominence, timing
fraction, …) through the yaml `ConfigManager` in `common/` instead of
compile-time constants in `WaveformAnalyzer.h`.
