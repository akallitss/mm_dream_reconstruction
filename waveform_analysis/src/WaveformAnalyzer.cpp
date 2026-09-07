//
// Created by dn277127 on 2025-12-05.
//

#include "WaveformAnalyzer.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <algorithm>

namespace {

// Reusable per-event dense waveform buffer. Vectors keep their capacity across
// events (clear() does not free), so after the first event the assembly loop is
// allocation-free — the previous unordered_map-of-vectors rebuild dominated the
// runtime (~50% of the total).
struct DenseEvent {
    static constexpr int kMaxCh = 512;
    static constexpr int kMaxSample = 2048;   // sanity cap on malformed sample indices

    std::vector<std::vector<float>> wf;
    std::vector<int>  present;     // channel ids seen this event, sorted before analysis
    std::vector<char> has;

    DenseEvent() : wf(kMaxCh), has(kMaxCh, 0) { present.reserve(kMaxCh); }

    void clear() {
        for (int ch : present) { wf[ch].clear(); has[ch] = 0; }
        present.clear();
    }

    // Record one sample; missing (zero-suppressed) samples stay 0 = baseline
    // after pedestal subtraction, densifying inline.
    void set(int ch, int s, float a) {
        if (ch < 0 || ch >= kMaxCh || s < 0 || s >= kMaxSample) return;
        if (!has[ch]) { has[ch] = 1; present.push_back(ch); }
        auto& w = wf[ch];
        if ((int)w.size() <= s) w.resize(s + 1, 0.0f);
        w[s] = a;
    }

    void sortPresent() { std::sort(present.begin(), present.end()); }
};

// Common-noise subtraction across 64-channel blocks: per sample, subtract the
// block median (same nth_element upper-middle convention as always). Used by
// both the data path and computePedestals so the pedestal RMS is measured on
// the same common-noise-subtracted waveforms the data path produces.
void applyCommonNoiseSubtraction(DenseEvent& ev)
{
    static constexpr int blockSize = 64;
    float vals[blockSize];

    // walk the sorted present list block by block
    size_t lo = 0;
    while (lo < ev.present.size()) {
        int blockStart = (ev.present[lo] / blockSize) * blockSize;
        int blockEnd   = blockStart + blockSize;
        size_t hi = lo;
        int maxLen = 0;
        while (hi < ev.present.size() && ev.present[hi] < blockEnd) {
            maxLen = std::max(maxLen, (int)ev.wf[ev.present[hi]].size());
            ++hi;
        }

        for (int s = 0; s < maxLen; s++) {
            int n = 0;
            for (size_t k = lo; k < hi; ++k) {
                auto& w = ev.wf[ev.present[k]];
                if (s < (int)w.size()) vals[n++] = w[s];
            }
            if (n == 0) continue;
            std::nth_element(vals, vals + n / 2, vals + n);
            float median = vals[n / 2];
            for (size_t k = lo; k < hi; ++k) {
                auto& w = ev.wf[ev.present[k]];
                if (s < (int)w.size()) w[s] -= median;
            }
        }
        lo = hi;
    }
}

// Boxcar running mean of full width `width` (effective window is the odd
// 2*(width/2)+1, clamped at the waveform edges). O(N) via a running sum.
// Used as the matched-filter gate for low-gain mode; the SAME implementation
// measures the gate-noise sigma in the pedestal pass, so thresholds are
// consistent by construction.
void boxcarSmooth(const std::vector<float>& in, int width, std::vector<float>& out)
{
    const int N = (int)in.size();
    const int h = width / 2;
    out.resize(N);
    if (N == 0) return;
    // prefix sums: cum[i] = sum of in[0..i-1]
    static thread_local std::vector<double> cum;
    cum.resize(N + 1);
    cum[0] = 0.0;
    for (int i = 0; i < N; ++i) cum[i + 1] = cum[i] + in[i];
    for (int i = 0; i < N; ++i) {
        int lo = std::max(0, i - h);
        int hi = std::min(N - 1, i + h);
        out[i] = (float)((cum[hi + 1] - cum[lo]) / double(hi - lo + 1));
    }
}

} // namespace

WaveformAnalyzer::WaveformAnalyzer(const std::string& inputFileName,
                                   const std::string& outputFileName,
                                   const std::string& pedestalFileName)
        : inputFileName(inputFileName),
          outputFileName(outputFileName),
          pedestalFileName(pedestalFileName)
{
    hasPedestal = !pedestalFileName.empty();
    // TEST override: flat pedestal sigma from environment (WFA_FLAT_SIGMA).
    if (const char* fs = std::getenv("WFA_FLAT_SIGMA")) {
        flatSigma = std::atof(fs);
        std::cout << "[WFA_FLAT_SIGMA] Overriding all pedestal RMS with flat sigma = "
                  << flatSigma << " ADC (means kept).\n";
    }
    // TEST override: hit threshold in sigma from environment (WFA_THRESHOLD_SIGMA).
    // Combined with WFA_FLAT_SIGMA this gives a uniform threshold = thresholdSigma*flatSigma,
    // e.g. 3 sigma of the median pedestal RMS. Default behaviour unchanged when unset.
    if (const char* ts = std::getenv("WFA_THRESHOLD_SIGMA")) {
        thresholdSigma = std::atof(ts);
        std::cout << "[WFA_THRESHOLD_SIGMA] Overriding hit threshold sigma = "
                  << thresholdSigma << ".\n";
    }
}

void WaveformAnalyzer::computePedestals() {
    if (pedestalFileName.empty()) {
        std::cout << "No pedestal file provided — assuming zero suppressed and using 256.\n";
        // For zero-suppressed and pedestal subtracted data, the pedestals for each strip are set to 256.
        // If channels is not suppressed, it should be a hit, so set RMS to 1 ADC count
        for (int ch = 0; ch < 512; ch++) {
            pedestalMap[ch] = {zeroSupressedBaseline, 1.0f, 1.0f};
        }
    }
    else  // compute from pedestal file
    {
        std::cout << "Computing pedestals from " << pedestalFileName << "\n";
        std::cout << "  RMS is measured AFTER common-noise subtraction; mean is the raw baseline.\n";

        TFile f(pedestalFileName.c_str(), "READ");
        if (!f.IsOpen()) {
            std::cerr << "Cannot open pedestal file.\n";
            return;
        }

        TTree* nt = (TTree*)f.Get("nt");
        if (!nt) {
            std::cerr << "Pedestal file has no nt tree.\n";
            return;
        }

        std::vector<UInt_t>* channel = nullptr;
        std::vector<UShort_t>* sample = nullptr;
        std::vector<UShort_t>* amplitude = nullptr;

        nt->SetBranchAddress("channel", &channel);
        nt->SetBranchAddress("sample", &sample);
        nt->SetBranchAddress("amplitude", &amplitude);

        // raw accumulator  -> per-channel MEAN (the DC baseline subtracted from data)
        std::unordered_map<int, PedestalData> rawAccum;
        // CNS-subtracted accumulator -> per-channel RMS (the true noise floor). The data
        // path also applies common-noise subtraction, so the threshold (thresholdSigma*rms)
        // must be calibrated on the CNS-subtracted noise, not the raw (common-mode-inflated) one.
        std::unordered_map<int, PedestalData> cnsAccum;
        // gate-waveform accumulator (matched-filter mode): sigma of the SMOOTHED
        // noise, needed because the noise is time-correlated (sigma_smooth is
        // ~0.8*sigma here, nowhere near the white-noise 1/sqrt(k)).
        std::unordered_map<int, PedestalData> gateAccum;
        std::vector<float> gateBuf;

        Long64_t nentries = nt->GetEntries();
        DenseEvent ev;
        const int mfW = resolvedMfWidth();
        for (Long64_t i = 0; i < nentries; i++) {
            nt->GetEntry(i);

            // assemble this event's raw samples (dense, reused buffer)
            ev.clear();
            for (size_t j = 0; j < channel->size(); j++) {
                int ch = (*channel)[j];
                float a = (float)(*amplitude)[j];
                ev.set(ch, (int)(*sample)[j], a);
                // raw accumulation -> mean (recorded samples only)
                auto& pd = rawAccum[ch];
                pd.sum += a; pd.count++;
            }
            ev.sortPresent();

            // common-noise subtraction (median across 64-channel blocks per
            // sample) — identical order to the data path — then accumulate -> rms
            applyCommonNoiseSubtraction(ev);
            for (int ch : ev.present) {
                auto& pd = cnsAccum[ch];
                for (float v : ev.wf[ch]) { pd.sum += v; pd.sumsq += v * v; pd.count++; }
                if (mfW > 0) {
                    boxcarSmooth(ev.wf[ch], mfW, gateBuf);
                    auto& pg = gateAccum[ch];
                    for (float v : gateBuf) { pg.sum += v; pg.sumsq += v * v; pg.count++; }
                }
            }
        }

        // mean from raw, RMS from CNS-subtracted
        pedestalMap.clear();
        for (auto& kv : rawAccum) {
            int ch = kv.first;
            const PedestalData& raw = kv.second;
            float mean = raw.count ? (float)(raw.sum / raw.count) : 0.0f;

            float rms = 0.0f;
            auto it = cnsAccum.find(ch);
            if (it != cnsAccum.end() && it->second.count) {
                const PedestalData& c = it->second;
                double cmean = c.sum / c.count;
                rms = (float)std::sqrt(std::max(0.0, c.sumsq / c.count - cmean * cmean));
            }
            float rmsGate = rms;
            if (mfW > 0) {
                rmsGate = 0.0f;
                auto ig = gateAccum.find(ch);
                if (ig != gateAccum.end() && ig->second.count) {
                    const PedestalData& g = ig->second;
                    double gmean = g.sum / g.count;
                    rmsGate = (float)std::sqrt(std::max(0.0, g.sumsq / g.count - gmean * gmean));
                }
            }
            pedestalMap[ch] = {mean, rms, rmsGate};
        }
        f.Close();
    }

    // TEST override: replace every channel's RMS with the flat sigma (means untouched).
    if (flatSigma > 0.0f) {
        // Both noise fields: the pulse-finding rewrite (8ec6769) split the gate
        // sigma out of rms, and the matched-filter gate is now the default, so
        // overriding rms alone would leave the gate -- the thing that suppresses
        // spark-contaminated strips -- on the original per-channel noise.
        for (auto& kv : pedestalMap) { kv.second.rms = flatSigma; kv.second.rmsGate = flatSigma; }
        std::cout << "[WFA_FLAT_SIGMA] Applied flat sigma " << flatSigma
                  << " to " << pedestalMap.size() << " channels.\n";
    }

    // Write pedestal tree to output
    TFile fout(outputFileName.c_str(), "UPDATE");
    TTree ped("pedestals", "pedestal values");

    UShort_t ch;
    Float_t mean, rms, rms_gate;

    ped.Branch("channel", &ch, "channel/s");
    ped.Branch("mean",    &mean, "mean/F");
    ped.Branch("rms",     &rms,  "rms/F");
    ped.Branch("rms_gate", &rms_gate, "rms_gate/F");  // sigma of the gate waveform (== rms when matched filter off)

    for (auto& kv : pedestalMap) {
        ch = kv.first;
        mean = kv.second.mean;
        rms  = kv.second.rms;
        rms_gate = kv.second.rmsGate;
        ped.Fill();
    }

    ped.Write();

    // Prepare data arrays for TGraph
    int nChannels = pedestalMap.size();
    if (nChannels == 0) {
        std::cout << "No channels found to graph.\n";
        fout.Close();
        return;
    }

    // We must use arrays or vectors of doubles for TGraph
    std::vector<double> chVec, meanVec, rmsVec;
    for (auto const& [channel, data] : pedestalMap) {
        chVec.push_back((double)channel);
        meanVec.push_back(data.mean);
        rmsVec.push_back(data.rms);
    }

    // Create Graphs
    TGraph* gMean = new TGraph(nChannels, chVec.data(), meanVec.data());
    gMean->SetName("g_mean_vs_channel");
    gMean->SetTitle("Pedestal Mean vs. Channel;Channel;Mean Amplitude (ADC)");
    gMean->SetMarkerStyle(20); // Circle marker
    gMean->SetMarkerSize(0.8);
    gMean->SetMarkerColor(kBlue);
    gMean->SetLineColor(kBlue);

    TGraph* gRMS = new TGraph(nChannels, chVec.data(), rmsVec.data());
    gRMS->SetName("g_rms_vs_channel");
    gRMS->SetTitle("Pedestal RMS vs. Channel;Channel;RMS (ADC)");
    gRMS->SetMarkerStyle(20);
    gRMS->SetMarkerSize(0.8);
    gRMS->SetMarkerColor(kRed);
    gRMS->SetLineColor(kRed);

    // Create Canvas and Draw
    TCanvas* c1 = new TCanvas("c_pedestals", "Pedestal Analysis Graphs", 1000, 500);
    c1->Divide(2, 1);

    // Draw Mean Graph
    c1->cd(1);
    gMean->Draw("AP");

    // Draw RMS Graph
    c1->cd(2);
    gRMS->Draw("AP");

    // Write the canvas (which contains both graphs) to the file
    c1->Write("", TObject::kOverwrite);

    // Clean up TObjects (ROOT manages memory, but we delete the canvas)
    delete c1;

    fout.Close();

    std::cout << "Pedestals TTree and analysis graphs written to output file.\n";
}



void WaveformAnalyzer::analyzeWaveforms() {
    std::cout << "Analyzing waveforms from " << inputFileName << "\n";

    TFile f(inputFileName.c_str(), "READ");
    TTree* nt = (TTree*)f.Get("nt");

    if (!nt) {
        std::cerr << "Error: input file missing nt tree.\n";
        return;
    }

    // input branches
    ULong64_t eventID, timestamp;
    UShort_t ftst;

    std::vector<UShort_t>* sample = nullptr;
    std::vector<UInt_t>*   channel = nullptr;
    std::vector<UShort_t>* amplitude = nullptr;

    nt->SetBranchAddress("eventId", &eventID);
    nt->SetBranchAddress("timestamp", &timestamp);
    nt->SetBranchAddress("ftst", &ftst);
    nt->SetBranchAddress("sample", &sample);
    nt->SetBranchAddress("channel", &channel);
    nt->SetBranchAddress("amplitude", &amplitude);

    // output file
    TFile fout(outputFileName.c_str(), "UPDATE");

    TTree hitTree("hits", "reconstructed hits");

    ULong64_t out_eventID;
    ULong64_t out_trigger_timestamp_ns;
    UShort_t  out_channel;
    Float_t   out_amp;
    Float_t   out_time_ns;
    Float_t   out_time_of_max_ns;
    Float_t   out_sample;
    Float_t   out_max_sample;
    Float_t   out_local_baseline;
    Float_t   out_local_max;
    Float_t   out_left_sample;
    Float_t   out_right_sample;
    Float_t   out_time_over_threshold;
    Float_t   out_integral;
    Bool_t    out_saturated;
    Bool_t    out_trunc_left;
    Bool_t    out_trunc_right;
    Float_t   out_significance;

    hitTree.Branch("eventId", &out_eventID, "eventId/l");
    hitTree.Branch("trigger_timestamp_ns", &out_trigger_timestamp_ns, "trigger_timestamp_ns/l");
    hitTree.Branch("channel", &out_channel, "channel/s");
    hitTree.Branch("amplitude", &out_amp, "amplitude/F");
    hitTree.Branch("time", &out_time_ns, "time/F");
    hitTree.Branch("time_of_max", &out_time_of_max_ns, "time_of_max/F");
    hitTree.Branch("sample", &out_sample, "sample/F");
    hitTree.Branch("max_sample", &out_max_sample, "max_sample/F");
    hitTree.Branch("local_baseline", &out_local_baseline, "local_baseline/F");
    hitTree.Branch("local_max", &out_local_max, "local_max/F");
    hitTree.Branch("left_sample", &out_left_sample, "left_sample/F");
    hitTree.Branch("right_sample", &out_right_sample, "right_sample/F");
    hitTree.Branch("time_over_threshold", &out_time_over_threshold, "time_over_threshold/F");
    hitTree.Branch("integral", &out_integral, "integral/F");
    hitTree.Branch("saturated", &out_saturated, "saturated/O");
    hitTree.Branch("trunc_left", &out_trunc_left, "trunc_left/O");
    hitTree.Branch("trunc_right", &out_trunc_right, "trunc_right/O");
    // gate-peak / gate-noise-sigma: analyses that want stronger noise rejection
    // than this processing's threshold cut on significance offline
    hitTree.Branch("significance", &out_significance, "significance/F");

    // Flat per-channel pedestal lookup (subtractPedestal's per-sample hash
    // find was ~200M lookups per file). Missing channels: mean 0, RMS 3.0
    // fallback, matching the previous behaviour.
    std::vector<float> pedMean(DenseEvent::kMaxCh, 0.0f);
    std::vector<float> pedRms(DenseEvent::kMaxCh, 3.0f);
    std::vector<float> pedRmsGate(DenseEvent::kMaxCh, 3.0f);
    for (auto& kv : pedestalMap) {
        if (kv.first >= 0 && kv.first < DenseEvent::kMaxCh) {
            pedMean[kv.first] = kv.second.mean;
            pedRms[kv.first]  = kv.second.rms;
            pedRmsGate[kv.first] = (resolvedMfWidth() > 0) ? kv.second.rmsGate : kv.second.rms;
        }
    }
    // ZS + on-FEU pedestal subtraction: the FEU already removed the per-channel
    // pedestal and re-centred every waveform at 256, so the per-channel raw
    // means from the pedestal file are the WRONG baseline here (off by tens of
    // ADC per channel). Keep the pedestal-file RMS for thresholds; override
    // only the means.
    if (zsBaseline) {
        std::fill(pedMean.begin(), pedMean.end(), zeroSupressedBaseline);
        std::cout << "ZS-baseline mode: data baseline forced to "
                  << zeroSupressedBaseline
                  << " (on-FEU pedestal subtraction); pedestal file used for "
                     "noise RMS only.\n";
    }

    Long64_t nentries = nt->GetEntries();
    DenseEvent ev;

    // Zero-suppression detection. CNS's per-64-channel-block median is only a
    // common-mode estimate on a full detector frame. On ZS data every surviving
    // channel is above threshold, so the median IS signal and CNS subtracts real
    // pulses (~8x hit loss); ZS data is also already common-mode corrected
    // upstream on the FEU. Decide once per file from the median distinct-channel
    // count (RAW frame ~512, ZS tens) and force CNS off regardless of the flag,
    // so a global common_noise_subtraction setting is safe across mixed ZS/RAW
    // reprocessing.
    bool effectiveCNS = commonNoiseSubtraction;
    bool dataLooksZS = false;
    if (commonNoiseSubtraction) {
        Long64_t nscan = std::min<Long64_t>(nentries, 200);
        std::vector<int> presentCounts;
        presentCounts.reserve((size_t)nscan);
        for (Long64_t i = 0; i < nscan; i++) {
            nt->GetEntry(i);
            ev.clear();
            for (size_t j = 0; j < channel->size(); j++) {
                unsigned int ch = (*channel)[j];
                if (ch >= DenseEvent::kMaxCh) continue;
                if (!ev.has[ch]) { ev.has[ch] = 1; ev.present.push_back(ch); }
            }
            presentCounts.push_back((int)ev.present.size());
        }
        ev.clear();
        int medianPresent = 0;
        if (!presentCounts.empty()) {
            std::nth_element(presentCounts.begin(),
                             presentCounts.begin() + presentCounts.size() / 2,
                             presentCounts.end());
            medianPresent = presentCounts[presentCounts.size() / 2];
        }
        if (medianPresent < 256) {
            effectiveCNS = false;
            std::cout << "Common-noise subtraction requested but data looks "
                         "zero-suppressed (median " << medianPresent
                      << " channels/event) — forcing CNS OFF (the block median "
                         "would be signal-biased on sparse data).\n";
            dataLooksZS = true;
        }
    }

    // Matched-filter gate width for this file. Auto-off on ZS data: the boxcar
    // would dilute the sparse sample islands and the ZS stream is already
    // firmware-thresholded.
    int mfW = resolvedMfWidth();
    if (dataLooksZS && mfW > 0) {
        std::cout << "Matched-filter gate disabled: data looks zero-suppressed.\n";
        mfW = 0;
    }
    if (mfW > 0)
        std::cout << "Matched-filter gate: boxcar width " << mfW
                  << " samples (" << mfW * timePerSample << " ns), threshold "
                  << thresholdSigma << " x gate-noise sigma.\n";
    else
        std::cout << "Gate on raw waveform, threshold " << thresholdSigma
                  << " x noise sigma.\n";

    for (Long64_t i = 0; i < nentries; i++) {
        nt->GetEntry(i);

        // Assemble the event into the reused dense buffer: pedestal-subtract
        // recorded samples; missing zero-suppressed samples stay 0 = baseline
        // after pedestal subtraction.
        ev.clear();
        for (size_t j = 0; j < channel->size(); j++) {
            unsigned int ch = (*channel)[j];
            if (ch >= DenseEvent::kMaxCh) continue;
            ev.set((int)ch, (int)(*sample)[j], (*amplitude)[j] - pedMean[ch]);
        }
        ev.sortPresent();

        if (effectiveCNS) {
            applyCommonNoiseSubtraction(ev);
        }

        // analyze per channel (ascending channel order)
        std::vector<float> gateBuf;
        for (int ch : ev.present) {
            std::vector<float>& amps = ev.wf[ch];

            float noiseRMS = pedRms[ch];
            float max_adc_ped_sub = max_adc - pedMean[ch];
            const std::vector<float>* gate = &amps;
            float noiseGate = noiseRMS;
            if (mfW > 0) {
                boxcarSmooth(amps, mfW, gateBuf);
                gate = &gateBuf;
                noiseGate = pedRmsGate[ch];
            }
            auto peaks = analyzeWaveform(amps, *gate, noiseRMS, noiseGate, max_adc_ped_sub, mfW > 0);
            for (auto& peak : peaks) {

                // Correct samples for ftst. ftst in units of clock cycles
                float ftstShift = static_cast<float>(ftst) * timePerFtst / timePerSample;
                peak.peakSample   += ftstShift;
                peak.timingSample += ftstShift;

                out_eventID        = eventID;
                out_trigger_timestamp_ns = timestamp * static_cast<int>(timePerTimestamp);
                out_channel        = ch;
                out_amp            = peak.peakAmplitude;
                out_local_max      = peak.peakMax;
                out_time_ns        = peak.timingSample * timePerSample;
                out_time_of_max_ns = peak.peakSample * timePerSample;
                out_sample         = peak.timingSample;
                out_max_sample     = peak.peakSample;
                out_local_baseline = peak.localBaseline;
                out_left_sample    = peak.leftCross + ftstShift;
                out_right_sample   = peak.rightCross + ftstShift;
                out_time_over_threshold = peak.timeOverThreshold * timePerSample;
                out_integral       = peak.integral;
                out_saturated      = peak.saturated;
                out_trunc_left     = peak.truncLeft;
                out_trunc_right    = peak.truncRight;
                out_significance   = peak.significance;

                hitTree.Fill();
            }
        }
    }

    hitTree.Write();
    fout.Close();
}


/// ---------------------------------------------------------------------------
/// splitRegionByValleys — recursive pile-up separation by valley prominence
/// ---------------------------------------------------------------------------
/// Within [start, end], find the interior valley whose depth below BOTH the
/// left-side and right-side maxima is largest. If that depth exceeds
/// `prominence`, split there (the valley sample starts the right-hand pulse,
/// matching the previous convention) and recurse on both halves. Noise wiggles
/// on a single pulse never reach the prominence cut; genuinely piled-up pulses
/// (a pulse on the tail of another makes peak-valley-peak) do.
/// A split is only taken if BOTH halves stay at least minHalfWidth wide —
/// splitting off an unusably narrow fragment would just delete charge.
/// ---------------------------------------------------------------------------
void WaveformAnalyzer::splitRegionByValleys(
    const std::vector<float>& wf,
    int start, int end,
    float prominence,
    int minHalfWidth,
    std::vector<std::pair<int,int>>& out)
{
    const int n = end - start + 1;
    int bestV = -1;
    float bestDepth = 0.0f;

    if (n >= 2 * minHalfWidth) {
        // prefix/suffix running maxima
        std::vector<float> pre(n), suf(n);
        pre[0] = wf[start];
        for (int i = 1; i < n; ++i) pre[i] = std::max(pre[i - 1], wf[start + i]);
        suf[n - 1] = wf[end];
        for (int i = n - 2; i >= 0; --i) suf[i] = std::max(suf[i + 1], wf[start + i]);

        // valley at start+i splits into [start, start+i-1] (width i) and
        // [start+i, end] (width n-i); require both >= minHalfWidth
        for (int i = minHalfWidth; i <= n - minHalfWidth; ++i) {
            if (i > n - 2) break;
            float depth = std::min(pre[i], suf[i]) - wf[start + i];
            if (depth >= prominence && depth > bestDepth) {
                bestDepth = depth;
                bestV = start + i;
            }
        }
    }

    if (bestV < 0) {
        out.push_back({start, end});
        return;
    }
    splitRegionByValleys(wf, start, bestV - 1, prominence, minHalfWidth, out);
    splitRegionByValleys(wf, bestV, end, prominence, minHalfWidth, out);
}


/// ---------------------------------------------------------------------------
/// findPulseRegions — unified threshold-run + valley-prominence pulse finder
/// ---------------------------------------------------------------------------
/// 1. Collect contiguous runs of wf > thresholdSigma*noiseRMS (the gate — a
///    strict superset of both the old legacy scan and the accepted output of
///    the old derivative trigger; no slope requirement, so slow risers and
///    window-truncated pulses are kept).
/// 2. Bridge sub-threshold gaps <= gapMergeSamples (noise dips inside a pulse).
/// 3. Split each run at significant valleys (splitRegionByValleys) so pile-up
///    on tails is still separated, which is what the derivative trigger was
///    originally for.
/// 4. Record per-region analysis bounds [loBound, hiBound]: how far a pulse's
///    baseline walk / integral may extend without crossing a neighbour.
/// ---------------------------------------------------------------------------
std::vector<WaveformAnalyzer::PulseRegion>
WaveformAnalyzer::findPulseRegions(
    const std::vector<float>& wf,
    float noiseRMS) const
{
    std::vector<PulseRegion> regions;
    const int N = (int)wf.size();
    if (N < 2) return regions;

    const float ampThr = thresholdSigma * noiseRMS;

    // Step 1 — contiguous above-threshold runs
    std::vector<std::pair<int,int>> runs;
    int i = 0;
    while (i < N) {
        if (wf[i] <= ampThr) { ++i; continue; }
        int j = i;
        while (j + 1 < N && wf[j + 1] > ampThr) ++j;
        runs.push_back({i, j});
        i = j + 1;
    }
    if (runs.empty()) return regions;

    // Step 2 — bridge short sub-threshold gaps
    std::vector<std::pair<int,int>> mergedRuns;
    for (auto& r : runs) {
        if (!mergedRuns.empty() && r.first - mergedRuns.back().second - 1 <= gapMergeSamples)
            mergedRuns.back().second = r.second;
        else
            mergedRuns.push_back(r);
    }

    // Step 3 — split pile-up at prominent valleys
    std::vector<std::pair<int,int>> pulses;
    for (auto& r : mergedRuns)
        splitRegionByValleys(wf, r.first, r.second, splitProminenceSigma * noiseRMS,
                             minSamplesForPeak, pulses);

    // Step 4 — analysis bounds from neighbours
    for (size_t k = 0; k < pulses.size(); ++k) {
        PulseRegion reg;
        reg.start   = pulses[k].first;
        reg.end     = pulses[k].second;
        reg.loBound = (k > 0) ? pulses[k - 1].second + 1 : 0;
        reg.hiBound = (k + 1 < pulses.size()) ? pulses[k + 1].first - 1 : N - 1;
        regions.push_back(reg);
    }

    return regions;
}



std::vector<PeakInfo> WaveformAnalyzer::analyzeWaveform(
    const std::vector<float>& wf,
    const std::vector<float>& gate,
    float noiseRMS,
    float noiseGate,
    float adcMax,          // ADC saturation value for detection
    bool gateIsFiltered
) const
{
    std::vector<PeakInfo> results;
    if (wf.size() < 3) return results;  // Waveform is too short

    const int N = (int)wf.size();
    const bool mf = gateIsFiltered;              // gate is the smoothed waveform
    const float gateThr = thresholdSigma * noiseGate;

    for (auto& reg : findPulseRegions(gate, noiseGate)) {
        int startIdx = reg.start;
        int endIdx   = reg.end;

        // Width guard
        if (endIdx - startIdx + 1 < minWidthSamples) continue;

        bool truncLeft  = (startIdx == 0);          // rise not recorded
        bool truncRight = (endIdx == N - 1);        // tail clipped by window end

        // Within [startIdx..endIdx] find plateau-aware maximum sample (middle of plateau)
        int maxIdx = startIdx;
        float maxVal = wf[startIdx];
        int k = startIdx;
        while (k <= endIdx) {
            // handle plateau runs: detect equal neighbor values
            int runStart = k;
            int runEnd = k;
            while (runEnd + 1 <= endIdx && wf[runEnd + 1] == wf[k]) ++runEnd;
            // choose plateau middle sample as candidate
            int candidate = (runStart + runEnd) / 2;
            if (wf[candidate] > maxVal) {
                maxVal = wf[candidate];
                maxIdx = candidate;
            }
            k = runEnd + 1;
        }

        // Local baseline: median of up to baselineLeftWindow samples left of the
        // pulse start, skipping the baselineGapSamples immediately before it —
        // those sit on the sub-threshold rise of slow pulses (measured: gap 2
        // halves the +1.3 sigma rise contamination on slow micro-TPC risers).
        // Median is unbiased on noise; the previous minimum was biased low by
        // ~0.5 sigma and latched onto undershoots.
        float baseline = 0.0f;
        if (local_baseline)
        {
            int bHi = std::max(reg.loBound, startIdx - baselineGapSamples);
            int bLo = std::max(reg.loBound, bHi - baselineLeftWindow);
            if (bHi == bLo) {
                // pulse starts too early for the guard gap — use the ungapped window
                bHi = startIdx;
                bLo = std::max(reg.loBound, bHi - baselineLeftWindow);
            }
            int nB  = bHi - bLo;
            if (nB > 0) {
                std::vector<float> win(wf.begin() + bLo, wf.begin() + bHi);
                std::nth_element(win.begin(), win.begin() + win.size() / 2, win.end());
                baseline = win[win.size() / 2];
                if (win.size() % 2 == 0) {
                    // average the two middle elements for an even-sized window
                    float upper = baseline;
                    float lower = *std::max_element(win.begin(), win.begin() + win.size() / 2);
                    baseline = 0.5f * (lower + upper);
                }
            } else if (startIdx > 0) {
                // pile-up split boundary: the valley level is the tail of the
                // previous pulse — measure this pulse above it
                baseline = wf[startIdx];
            } else {
                // truncated at window start: no pre-samples; global pedestal (0)
                baseline = 0.0f;
            }
        }

        // Detect saturation: require >= minSatSamples CONSECUTIVE samples near
        // ADC max (a single large sample is a real pulse, not saturation).
        bool saturated = false;
        int satStart = -1, satEnd = -1;
        float satThreshold = satFrac * adcMax;
        {
            int runStart = -1;
            for (int t = startIdx; t <= endIdx + 1; ++t) {
                bool high = (t <= endIdx) && (wf[t] >= satThreshold);
                if (high && runStart < 0) runStart = t;
                if (!high && runStart >= 0) {
                    int len = t - runStart;
                    if (len >= minSatSamples && len > (satEnd - satStart + 1)) {
                        satStart = runStart;
                        satEnd   = t - 1;
                        saturated = true;
                    }
                    runStart = -1;
                }
            }
        }

        // Find peak amplitude and time of max
        float peakAmpFit = wf[maxIdx] - baseline;
        float peakSample = maxIdx;
        if (saturated)  // For saturated peaks, do linear extrapolation from edges of saturated region
        {
            saturatedLinearExtrapolation(
                satStart,
                satEnd,
                wf.data(),
                N,
                baseline,
                peakSample,
                peakAmpFit
            );
        }
        else  // Fit parabola around the integer maxIdx if possible to get sub-sample peak
        {
            float peakOffset = 0.0f;
            if (maxIdx > 0 && maxIdx < N-1) {
                float y1 = wf[maxIdx-1] - baseline;
                float y2 = wf[maxIdx]   - baseline;
                float y3 = wf[maxIdx+1] - baseline;
                float denom = (y1 - 2*y2 + y3);
                if (std::abs(denom) > 1e-9f) {
                    peakOffset = 0.5f * (y1 - y3) / denom; // in samples
                    peakAmpFit = y2 - 0.25f * (y1 - y3) * peakOffset;
                } else {
                    peakOffset = 0.0f;
                    peakAmpFit = y2;
                }
            }
            peakSample = maxIdx + peakOffset;
        }

        // find x% timing on leading edge relative to baseline-subtracted peak amplitude
        float fraction = timingPercentMax; // e.g. 0.5
        float timing_amp = std::min(peakAmpFit, adcMax);
        float target = fraction * timing_amp + baseline;
        int leadIdx = maxIdx;
        while (leadIdx > startIdx && wf[leadIdx] > target) --leadIdx;
        // linear interp between leadIdx and leadIdx+1
        float timingSample = (float)leadIdx;
        if (leadIdx + 1 < N) {
            float yL = wf[leadIdx];
            float yH = wf[leadIdx+1];
            if (std::abs(yH - yL) > 1e-9f) {
                timingSample = leadIdx + (target - yL) / (yH - yL);
            } else {
                timingSample = leadIdx + 0.5f;
            }
        } else {
            timingSample = (float)maxIdx;
        }

        // Interpolated threshold crossings at the region boundaries, measured on
        // the GATE waveform (aliases the raw one unless the matched filter is on).
        // Left: crossing of gateThr between startIdx-1 (below) and startIdx (above),
        // unless the pulse is truncated / abuts a neighbour (then the boundary itself).
        float leftCross = (float)startIdx;
        if (startIdx > reg.loBound && startIdx > 0 && gate[startIdx - 1] <= gateThr) {
            float yL = gate[startIdx - 1];
            float yH = gate[startIdx];
            if (yH - yL > 1e-9f)
                leftCross = (startIdx - 1) + (gateThr - yL) / (yH - yL);
        }
        float rightCross = (float)endIdx;
        if (endIdx < reg.hiBound && endIdx + 1 < N && gate[endIdx + 1] <= gateThr) {
            float yH = gate[endIdx];
            float yL = gate[endIdx + 1];
            if (yH - yL > 1e-9f)
                rightCross = endIdx + (yH - gateThr) / (yH - yL);
        }
        float tot = rightCross - leftCross;

        // Integral over the FULL pulse span: extend from the threshold core out
        // to where the waveform returns to baseline (or the neighbour/window
        // limit), so the sub-threshold rise and tail are included. The old
        // between-crossings sum kept only ~half the charge of a 5-10 sigma
        // pulse — a strong amplitude-dependent nonlinearity.
        int pl = startIdx;
        while (pl - 1 >= reg.loBound && wf[pl - 1] - baseline > 0.0f) --pl;
        int pr = endIdx;
        while (pr + 1 <= reg.hiBound && wf[pr + 1] - baseline > 0.0f) ++pr;
        float integral = 0.0f;
        for (int tt = pl; tt <= pr; ++tt) integral += (wf[tt] - baseline);

        // Build PeakInfo
        PeakInfo pi;
        pi.peakIndex = maxIdx;
        pi.peakAmplitude = peakAmpFit;
        pi.peakMax = wf[maxIdx] - baseline;
        pi.peakSample = peakSample;
        pi.timingSample = timingSample;
        pi.leftCross = leftCross;
        pi.rightCross = rightCross;
        pi.timeOverThreshold = tot;
        pi.integral = integral;
        pi.saturated = saturated;
        pi.truncLeft = truncLeft;
        pi.truncRight = truncRight;
        pi.localBaseline = baseline;

        // significance = gate peak within the region, in gate-noise sigmas
        float gmax = gate[startIdx];
        for (int t = startIdx + 1; t <= endIdx; ++t) gmax = std::max(gmax, gate[t]);
        pi.significance = (noiseGate > 0.0f) ? gmax / noiseGate : 0.0f;

        // Acceptance. Normal mode: baseline-subtracted amplitude above threshold
        // (unchanged). Matched-filter mode: the gate condition IS the detection
        // criterion — requiring the raw amplitude to also clear thresholdSigma
        // would undo the filter gain — so only the width guard applies.
        bool widthOK = (endIdx - startIdx + 1) >= minSamplesForPeak;
        bool ampOK = mf || (pi.peakAmplitude >= thresholdSigma * noiseRMS);
        if (ampOK && widthOK) {
            results.push_back(pi);
        }
    }

    // If only one peak per waveform is requested, keep the LARGEST (the old
    // code kept the first, contradicting its own comment).
    if (!allowMultiplePeaks && results.size() > 1) {
        auto best = std::max_element(results.begin(), results.end(),
            [](const PeakInfo& a, const PeakInfo& b) { return a.peakAmplitude < b.peakAmplitude; });
        PeakInfo keep = *best;
        results.clear();
        results.push_back(keep);
    }

    return results;
}


// Function to perform linear extrapolation to find peak for saturated pulses.
// Edge slopes are least-squares fits over up to satSlopeFitSamples samples per
// side (single-sample differences were noise-limited).
void WaveformAnalyzer::saturatedLinearExtrapolation(
    int satStartIdx,
    int satEndIdx,
    const float* wf,
    int N,
    float baseline,
    float& peakSample,
    float& peakAmpFit
) const {
    // least-squares slope/intercept of wf over [i0, i1]
    auto fitLine = [&](int i0, int i1, float& slope, float& intercept) -> bool {
        int n = i1 - i0 + 1;
        if (n < 2) return false;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = i0; i <= i1; ++i) {
            sx += i; sy += wf[i]; sxx += (double)i * i; sxy += (double)i * wf[i];
        }
        double denom = n * sxx - sx * sx;
        if (std::abs(denom) < 1e-12) return false;
        slope     = (float)((n * sxy - sx * sy) / denom);
        intercept = (float)((sy - slope * sx) / n);
        return true;
    };

    float leftSlope = 0.0f, leftIntercept = 0.0f;
    float rightSlope = 0.0f, rightIntercept = 0.0f;
    bool okL = fitLine(std::max(0, satStartIdx - satSlopeFitSamples), satStartIdx,
                       leftSlope, leftIntercept);
    bool okR = fitLine(satEndIdx, std::min(N - 1, satEndIdx + satSlopeFitSamples),
                       rightSlope, rightIntercept);

    // Ensure we have valid slopes. Left should be positive, right should be negative
    if (!okL || !okR || leftSlope <= 0 || rightSlope >= 0)
    {
        peakSample = static_cast<float>(satStartIdx);
        peakAmpFit = wf[satStartIdx] - baseline; // fallback to first saturated sample
        return;
    }

    // Estimate peak position by finding intersection of the two lines
    peakSample = (rightIntercept - leftIntercept) / (leftSlope - rightSlope);
    peakAmpFit = leftSlope * peakSample + leftIntercept - baseline;
}


int WaveformAnalyzer::run() {
    TFile fout(outputFileName.c_str(), "RECREATE");
    fout.Close();
    computePedestals();  // If no pedestals, this will assume zero-suppressed data and subtract 256
    analyzeWaveforms();
    return 0;
}
