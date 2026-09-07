//
// Created by dylan on 2/1/26.
//

#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

struct InputFile {
    std::string path;
    int feu;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: merge_hits inputs.txt output.root\n";
        return 1;
    }

    std::string listFile = argv[1];
    std::string outFilePath = argv[2];

    std::string treeName = "hits";

    // -------------------------
    // Read input list
    // -------------------------
    std::vector<InputFile> inputs;
    {
        std::ifstream fin(listFile);
        if (!fin.is_open()) {
            std::cerr << "Failed to open " << listFile << "\n";
            return 1;
        }

        InputFile in;
        while (fin >> in.path >> in.feu) {
            inputs.push_back(in);
        }
    }

    if (inputs.empty()) {
        std::cerr << "No input files provided\n";
        return 1;
    }

    // -------------------------
    // Output file / tree
    // -------------------------
    TFile outFile(outFilePath.c_str(), "RECREATE");
    TTree outTree("hits", "reconstructed hits");

    // --- Branch variables ---
    ULong64_t eventId;
    ULong64_t trigger_timestamp_ns;
    UShort_t channel;
    Float_t amplitude;
    Float_t time;
    Float_t time_of_max;
    Float_t sample;
    Float_t max_sample;
    Float_t local_baseline;
    Float_t local_max;
    Float_t left_sample;
    Float_t right_sample;
    Float_t time_over_threshold;
    Float_t integral;
    Bool_t saturated;
    Bool_t trunc_left;
    Bool_t trunc_right;
    Float_t significance;
    Int_t feu;

    // --- Branches ---
    outTree.Branch("eventId", &eventId, "eventId/l");
    outTree.Branch("trigger_timestamp_ns", &trigger_timestamp_ns, "trigger_timestamp_ns/l");
    outTree.Branch("channel", &channel, "channel/s");
    outTree.Branch("amplitude", &amplitude, "amplitude/F");
    outTree.Branch("time", &time, "time/F");
    outTree.Branch("time_of_max", &time_of_max, "time_of_max/F");
    outTree.Branch("sample", &sample, "sample/F");
    outTree.Branch("max_sample", &max_sample, "max_sample/F");
    outTree.Branch("local_baseline", &local_baseline, "local_baseline/F");
    outTree.Branch("local_max", &local_max, "local_max/F");
    outTree.Branch("left_sample", &left_sample, "left_sample/F");
    outTree.Branch("right_sample", &right_sample, "right_sample/F");
    outTree.Branch("time_over_threshold", &time_over_threshold, "time_over_threshold/F");
    outTree.Branch("integral", &integral, "integral/F");
    outTree.Branch("saturated", &saturated, "saturated/O");
    outTree.Branch("trunc_left", &trunc_left, "trunc_left/O");
    outTree.Branch("trunc_right", &trunc_right, "trunc_right/O");
    outTree.Branch("significance", &significance, "significance/F");
    outTree.Branch("feu", &feu, "feu/I");

    // -------------------------
    // Loop over input files
    // -------------------------
    for (const auto& input : inputs) {
        TFile inFile(input.path.c_str(), "READ");
        if (inFile.IsZombie()) {
            std::cerr << "Failed to open " << input.path << "\n";
            continue;
        }

        TTree* inTree = nullptr;
        inFile.GetObject(treeName.c_str(), inTree);
        if (!inTree) {
            std::cerr << "Tree " << treeName << " not found in "
                      << input.path << "\n";
            continue;
        }

        // Set branch addresses
        inTree->SetBranchAddress("eventId", &eventId);
        inTree->SetBranchAddress("trigger_timestamp_ns", &trigger_timestamp_ns);
        inTree->SetBranchAddress("channel", &channel);
        inTree->SetBranchAddress("amplitude", &amplitude);
        inTree->SetBranchAddress("time", &time);
        inTree->SetBranchAddress("time_of_max", &time_of_max);
        inTree->SetBranchAddress("sample", &sample);
        inTree->SetBranchAddress("max_sample", &max_sample);
        inTree->SetBranchAddress("local_baseline", &local_baseline);
        inTree->SetBranchAddress("local_max", &local_max);
        inTree->SetBranchAddress("left_sample", &left_sample);
        inTree->SetBranchAddress("right_sample", &right_sample);
        inTree->SetBranchAddress("time_over_threshold", &time_over_threshold);
        inTree->SetBranchAddress("integral", &integral);
        inTree->SetBranchAddress("saturated", &saturated);
        // Older hits files predate the truncation flags — default them to false
        trunc_left = trunc_right = false;
        if (inTree->GetBranch("trunc_left"))
            inTree->SetBranchAddress("trunc_left", &trunc_left);
        if (inTree->GetBranch("trunc_right"))
            inTree->SetBranchAddress("trunc_right", &trunc_right);
        significance = -1.0f;   // marker for pre-significance hits files
        if (inTree->GetBranch("significance"))
            inTree->SetBranchAddress("significance", &significance);

        feu = input.feu;

        const Long64_t nEntries = inTree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            inTree->GetEntry(i);
            outTree.Fill();
        }
    }

    outFile.Write();
    outFile.Close();

    return 0;
}
