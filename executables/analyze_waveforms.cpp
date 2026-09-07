//
// Created by dn277127 on 2025-12-05.
//

#include <iostream>
// #include "../waveform_analysis/include/WaveformAnalyzer.h"
#include "WaveformAnalyzer.h"

int main(int argc, char **argv) {
//    WaveformAnalyzer wf("decoded.root", "hits.root", "pedestal.root");
    // WaveformAnalyzer wf("/local/home/dn277127/x17/decoder_test/ftest.root", "hits.root", "/local/home/dn277127/x17/dream_run/ped_thresh_1_12_25_18_30/Mx17_ped_pedthr_251201_18H27_000_05.root");
    // WaveformAnalyzer wf("Mx17_run_datrun_251204_18H17_000_05.root", "hits.root", "/home/dylan/CLionProjects/mm_strip_reconstruction/test/ped_thresh_1_12_25_18_30/Mx17_ped_pedthr_251201_18H27_000_05.root");
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input.root> [output.root] [pedestal.root]"
                     " [--tps <ns>] [--cns <0|1>] [--thr <sigma>] [--mf <samples>]"
                     " [--zs-baseline <0|1>]\n"
                     "  --thr  gate threshold in noise sigmas (default 5.0; nTOF low-gain ~3)\n"
                     "  --zs-baseline  1 = data is zero-suppressed with ON-FEU pedestal\n"
                     "         subtraction (re-centred at 256): subtract 256 instead of the\n"
                     "         pedestal file's per-channel means; keep its RMS for thresholds.\n"
                     "  --mf   matched-filter (boxcar) gate width in samples.\n"
                     "         Default: AUTO (~300 ns / tps: 5 at 60 ns, 15 at 20 ns);\n"
                     "         0 = raw-waveform gate (pre-2026-07-24 behaviour).\n"
                     "         Auto-disabled on zero-suppressed data / missing pedestal.\n"
                     "         Cut the hits 'significance' branch offline to tighten." << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = "hits.root";
    std::string pedestalFile = "";
    float timePerSample = -1.0f;  // negative means use default
    int commonNoiseSub = -1;      // -1 means leave the compiled default (ON)
    float thresholdSigma = -1.0f; // negative means use default (5.0)
    int mfWidth = -999;           // sentinel: leave compiled default (auto)
    int zsBaseline = -1;          // -1 means leave the compiled default (OFF)
    if (argc >= 3) outputFile = argv[2];
    if (argc >= 4) pedestalFile = argv[3];
    for (int i = 4; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--tps") {
            timePerSample = std::stof(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--cns") {
            commonNoiseSub = std::stoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--thr") {
            thresholdSigma = std::stof(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--mf") {
            mfWidth = std::stoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--zs-baseline") {
            zsBaseline = std::stoi(argv[i + 1]);
            ++i;
        }
    }

    WaveformAnalyzer wf(inputFile, outputFile, pedestalFile);
    if (timePerSample > 0.0f) wf.setTimePerSample(timePerSample);
    if (commonNoiseSub >= 0) wf.setCommonNoiseSubtraction(commonNoiseSub != 0);
    if (thresholdSigma > 0.0f) wf.setThresholdSigma(thresholdSigma);
    if (mfWidth != -999) wf.setMatchedFilterWidth(mfWidth);
    if (zsBaseline >= 0) wf.setZsBaseline(zsBaseline != 0);
    wf.run();
    return 0;
}
