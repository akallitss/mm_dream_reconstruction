//
// Created by dn277127 on 2025-12-05.
//

#ifndef WAVEFORM_ANALYZER_H
#define WAVEFORM_ANALYZER_H
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include "TFile.h"
#include "TTree.h"
#include "TGraph.h"
#include "TCanvas.h"

struct PedestalData {
    double sum = 0;
    double sumsq = 0;
    uint64_t count = 0;
};

struct ChannelPedestal {
    float mean = 0;
    float rms  = 0;      // post-CNS noise of the raw waveform
    float rmsGate = 0;   // noise of the gate waveform (== rms unless matched filter on)
};

struct PeakInfo {
    int peakIndex;           // integer sample index of the maximum sample (or plateau center)
    float peakAmplitude;     // baseline-subtracted amplitude from parabola fit
    float peakMax;           // maximum sample amplitude (baseline-subtracted)
    float peakSample;        // sub-sample peak position (sample units) from parabola
    float timingSample;      // x% timing sample (sample units)
    float leftCross;         // interpolated threshold-crossing sample on the left (float)
    float rightCross;        // interpolated threshold-crossing sample on the right (float)
    float timeOverThreshold; // rightCross - leftCross, sample units (float, interpolated)
    float integral;          // baseline-subtracted sum over the full pulse span (to baseline return)
    bool saturated;          // whether the peak shows saturation (flat top near ADC max)
    bool truncLeft;          // pulse already above threshold at the first sample (rise not recorded)
    bool truncRight;         // pulse still above threshold at the last sample (tail clipped)
    float localBaseline;     // baseline used for this peak (same units as samples)
    float significance;      // gate-waveform peak / gate noise sigma — cut on this offline to tighten
};

class WaveformAnalyzer {
public:
    WaveformAnalyzer(const std::string& inputFileName,
                     const std::string& outputFileName,
                     const std::string& pedestalFileName = "");

    void computePedestals();
    void analyzeWaveforms();
    int run();

    void setAllowMultiplePeaks(bool v) { allowMultiplePeaks = v; }
    void setThresholdSigma(float v)    { thresholdSigma = v; }
    void setTimePerSample(float v)     { timePerSample = v; }
    void setCommonNoiseSubtraction(bool v) { commonNoiseSubtraction = v; }  // toggles CNS on DATA (pedestal RMS is always post-CNS)
    // For zero-suppressed data whose pedestals were already subtracted ON THE
    // FEU (waveforms re-centred at 256): subtract the uniform 256 baseline
    // instead of the pedestal file's per-channel RAW means (which differ from
    // 256 by tens of ADC and would shift every threshold/amplitude by that
    // much). The pedestal file is still used for the per-channel noise RMS.
    // (Ported from the banco P2 fork, 2026-07-24.)
    void setZsBaseline(bool v)         { zsBaseline = v; }
    void setMatchedFilterWidth(int v)  { matchedFilterWidth = v; }  // -1 = auto (default), 0 = off (raw gate), >0 = samples

private:
    std::string inputFileName;
    std::string outputFileName;
    std::string pedestalFileName;
    bool hasPedestal = false;

    std::unordered_map<int, ChannelPedestal> pedestalMap;

    // configuration
    // TODO: expose these through the yaml ConfigManager (common/) instead of
    // compile-time constants — needed once one binary serves both the bench
    // micro-TPC detectors and P2. Documented, deferred.
    bool commonNoiseSubtraction = true;  // if true, subtract common noise per event per channel (Cosmic Bench default ON)
    bool zsBaseline = false;  // if true, data baseline = zeroSupressedBaseline (256), not the pedestal-file means
    bool allowMultiplePeaks = true;  // if false, only the LARGEST peak per channel per event is kept
    bool local_baseline = true;  // if true, use local baseline per peak; if false, use global pedestal mean
    float thresholdSigma = 5.0;  // Number of pedestal RMS above which a hit is registered
    // TEST/DEBUG override: if >0 (set via env WFA_FLAT_SIGMA), every channel's pedestal RMS
    // is replaced by this flat value AFTER the per-channel mean is computed. Keeps correct
    // baseline subtraction but makes the 5σ threshold uniform/low, recovering strips that a
    // spark-contaminated pedestal would otherwise suppress. -1 = disabled (normal behaviour).
    float flatSigma = -1.0f;

    int minSamplesForPeak = 3;  // minimum number of samples above threshold to consider a peak
    int minWidthSamples = 2;  // minimum width in samples above threshold to consider a pulse
    int baselineLeftWindow = 4;  // number of samples used for the median local baseline
    int baselineGapSamples = 2;  // guard gap between the baseline window and the pulse start (skips the sub-threshold rise)
    float satFrac = 0.94;  // fraction of max_adc above which samples are considered saturated
    int minSatSamples = 2;  // consecutive samples above satFrac*adcMax required to flag saturation
    int satSlopeFitSamples = 3;  // samples per side used to fit the edge slopes of a saturated peak

    // --- Pulse-region finding (single unified algorithm) ---
    // Contiguous above-threshold runs are the pulse gate; short sub-threshold
    // gaps are bridged; pile-up inside one run is separated by splitting at
    // valleys whose depth below both flanking maxima is significant vs noise.
    int gapMergeSamples = 2;            // sub-threshold gaps <= this many samples do not end a pulse
    float splitProminenceSigma = 4.0f;  // valley depth (units of noiseRMS) needed to split pile-up

    // --- Matched-filter gate (DEFAULT since 2026-07-24) ---
    // The pulse GATE runs on a boxcar-smoothed copy of the waveform against
    // the smoothed-noise sigma (measured in the same pedestal pass). Real
    // pulses are many samples wide (DREAM shaper ~300 ns) while the noise tail
    // is dominated by narrow excursions, so at a fixed fake rate the smoothed
    // gate beats the raw-threshold gate at every amplitude (June det3
    // injection study: at ~1% fakes/waveform, 4-sigma pulses 42% vs 10%;
    // and at the same nominal 5-sigma threshold it has FEWER fakes, 0.7% vs
    // 1.05%). Amplitude/baseline/timing/integral are still measured on the
    // RAW waveform; regions/crossings/TOT come from the gate waveform.
    // matchedFilterWidth: -1 = AUTO (width = matchedFilterNs / timePerSample:
    // 5 samples at 60 ns, 15 at 20 ns), 0 = off (raw gate, pre-2026-07-24
    // behaviour), >0 = explicit width in samples. Auto-disabled on
    // zero-suppressed data / missing pedestal file (gate sigma unmeasurable,
    // boxcar would dilute the sparse islands).
    // Mode split is by THRESHOLD only: default 5.0; nTOF low-gain ~3.0.
    int matchedFilterWidth = -1;
    float matchedFilterNs = 300.0f;  // auto gate width in ns (~DREAM shaper pulse width)

    float zeroSupressedBaseline = 256.0f; // baseline level for zero-suppressed pedestal-subtracted waveforms

    // float timePerSample = 20.0;  // ns per sample. Sampling period
    float timePerSample = 60.0;  // ns per sample. Sampling period -- set from commandline
    float timePerFtst = 10.0;  // ns per fine timestamp unit. Fixed by DREAM clock of 100MHz --> 10 ns. Shift the timestamp by this amount.
    float timePerTimestamp = 10.0;  // ns Timestamp is in clock cycles of 10 ns
    float timingPercentMax = 0.3;  // fraction of peak amplitude at which timing is calculated

    int max_adc = 4095;  // maximum ADC value (saturation level)

    // helpers
    // Resolved matched-filter width for this run: auto/off/explicit, in samples.
    // 0 when the gate runs on the raw waveform (ZS data / no pedestal / --mf 0).
    int resolvedMfWidth() const {
        if (!hasPedestal) return 0;                      // gate sigma unmeasurable
        if (matchedFilterWidth >= 0) return matchedFilterWidth;
        return std::max(3, (int)std::lround(matchedFilterNs / timePerSample));
    }

    // gate: the waveform pulse-finding runs on (aliases wf unless the matched
    // filter is enabled); noiseGate: its per-channel noise sigma.
    // gateIsFiltered: acceptance uses the gate criterion instead of the raw
    // amplitude cut when true.
    std::vector<PeakInfo> analyzeWaveform(
    const std::vector<float>& wf,
    const std::vector<float>& gate,
    float noiseRMS,
    float noiseGate,
    float adcMax,          // ADC saturation value for detection
    bool gateIsFiltered
    ) const;

    void saturatedLinearExtrapolation(
    int satStartIdx,
    int satEndIdx,
    const float* wf,
    int N,
    float baseline,
    float& peakSample,
    float& peakAmpFit
    ) const;

    // Pulse-region finder. Returns [start, end] threshold cores plus the
    // analysis bounds [loBound, hiBound] a pulse may extend into (limited by
    // its neighbours), one region per detected pulse.
    struct PulseRegion { int start; int end; int loBound; int hiBound; };
    std::vector<PulseRegion> findPulseRegions(
        const std::vector<float>& gate,
        float noiseGate
    ) const;

    static void splitRegionByValleys(
        const std::vector<float>& wf,
        int start, int end,
        float prominence,
        int minHalfWidth,
        std::vector<std::pair<int,int>>& out);
};

#endif
