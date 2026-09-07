//
// Created by dn277127 on 2025-12-02.
//

#include "DreamDecoder.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <bitset>
#include <vector>
#include <map>
#include <array>

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "RtypesCore.h"

#include <arpa/inet.h>
#include "dreamdataline.h"

#include <stdexcept>
#include <sstream>
#include <algorithm>

using namespace std;

DreamDecoder::DreamDecoder(const std::string &input_filename, const std::string &output_root)
        : input_filename_(input_filename), output_root_(output_root)
{
    // Validate file existence & not empty here (to match original code behavior)
    std::ifstream is(input_filename_, std::ifstream::binary);
    if (!is) {
        throw std::runtime_error("Cannot open " + input_filename_);
    }
    if (is.peek() == ifstream::traits_type::eof()) {
        throw std::runtime_error("File is empty " + input_filename_);
    }
    // close; actual file opened in run()
    is.close();
}

bool DreamDecoder::read16( ifstream &is, uint16_t &data ){
    is.read( (char*) &data, sizeof(data) );

    data = ntohs( data );

    return is.eof();
}

void DreamDecoder::print_data( uint16_t data ){

    bitset<16> x(data);

    cout << setw(20)<< x << setw(5)
         << is_final_trailer(data) << setw(5)
         << is_Feu_header(data) << setw(5)
         << is_data(data) << setw(5)
         << is_data_header(data) << setw(5)
         << is_data_trailer( data ) ;
    cout << endl;
}

int DreamDecoder::run() {
    // open input stream
    ifstream is( input_filename_, std::ifstream::binary );
    if (!is) {
        std::cerr << "Cannot open " << input_filename_ << " Exiting." << std::endl;
        throw std::runtime_error("Cannot open input file");
    }

    // output ROOT name
    string outputFileName = output_root_;
    if (outputFileName.find(".root") == string::npos) {
        outputFileName += ".root";
    }

    // variables, copied from original
    uint16_t data = 0;

    Short_t   iFeuH = 0;
    ULong64_t timestamp = 0;
    UShort_t  fine_timestamp = 0;
    ULong64_t eventID   = 0;
    UInt_t    FeuID = 0;
    UShort_t  sampleID = 0;
    UShort_t  channelID = 0;
    UShort_t  dreamID = 0;
    UShort_t  ampl = 0;

    // Header values of the event whose data is currently accumulated, so a
    // flush triggered by a NEW header still stamps the OLD event's header.
    ULong64_t accEventID = 0;
    ULong64_t accTimestamp = 0;
    UShort_t  accFineTs = 0;

    // ---- data-loss accounting ------------------------------------------
    // A DREAM FEU under RAW bandwidth pressure silently drops sample-group
    // packets (run_71 lost ~24%).  Nothing downstream can tell that from
    // genuinely quiet channels, so the decoder counts it and both SHOUTS and
    // writes the numbers into the output file, where they cannot be separated
    // from the data.
    ULong64_t nFlushEoE = 0;       // events closed by the FEU end-of-event marker
    ULong64_t nFlushEvtId = 0;     // events closed only because eventID changed
                                   //   -> their EoE packet was lost
    ULong64_t nFlushEof = 0;       // last event, closed at end of file
    ULong64_t nFramesTotal = 0;    // FEU header frames seen
    ULong64_t sumDistinct = 0;     // sum over events of distinct sampleIDs
    ULong64_t firstEventID = 0, lastEventID = 0;
    bool haveFirstEvent = false;
    UShort_t  maxSampleID = 0;
    bool sawZS = false, sawRaw = false;
    std::vector<uint8_t>   sampSeen(1024, 0);
    std::vector<ULong64_t> sampHist(1024, 0);

    bool isEvent = false;  // check the end of the event
    bool isFT    = false;  // on if FT (Final Trailer) is reached and set off by the header
    bool isZS    = true;   // true if zero suppressed data. false if not
    int i = 0;             // just a counter
    bool debug = false;     // printing stuff
    char prev = cout.fill(); // for debug formatting

    // store data
    vector<uint16_t> sample;
    vector<uint16_t> channel;
    vector<uint16_t> amplitude;

    TFile fout(outputFileName.data(), "recreate");
    TTree nt("nt","nt");
    nt.SetDirectory(&fout);

    nt.Branch("eventId",         &eventID,         "eventId/l");
    nt.Branch("timestamp",       &timestamp,       "timestamp/l");
    nt.Branch("ftst",            &fine_timestamp,  "ftst/s");
    nt.Branch("sample",          &sample);
    nt.Branch("channel",         &channel);
    nt.Branch("amplitude",       &amplitude);

    // Close out the accumulated event: stamp the right header, fill, reset,
    // and fold this event's sample coverage into the loss statistics.
    //   reason 0 = FEU end-of-event marker (normal)
    //   reason 1 = eventID changed (the EoE packet was lost)
    //   reason 2 = end of file
    auto flushEvent = [&](int reason) {
        ULong64_t distinct = 0;
        for (int sIdx = 0; sIdx <= (int)maxSampleID; ++sIdx) {
            if (sampSeen[sIdx]) { sampHist[sIdx]++; distinct++; }
        }
        std::fill(sampSeen.begin(), sampSeen.end(), 0);
        sumDistinct += distinct;
        if (!haveFirstEvent) { firstEventID = accEventID; haveFirstEvent = true; }
        lastEventID = accEventID;
        if      (reason == 0) nFlushEoE++;
        else if (reason == 1) nFlushEvtId++;
        else                  nFlushEof++;
        nt.Fill();
        channel.clear();
        sample.clear();
        amplitude.clear();
        i++;
    };

    // Prime first word (original code had data initialized to 0 and logic relied on read16 at various points)
    // To mimic behavior exactly, read first word here:
    if (read16(is, data)) {
        // if file is only 2 bytes and EOF after read, still continue with data value
    }

    // loop over the file (copied nearly verbatim)
    while( true ){

        // FEU header
        // ---------
        if ( is_Feu_header( data ) ){
            // we enter here the first time the data is a FEU header
            // then we loop over all the FEU header data

            isEvent = true; // we are in an event
            isFT = false;
            isZS = get_zs_mode(data);  // true if zero supressed data, false otherwise

            timestamp = 0;
            eventID   = 0;
            FeuID     = 0;
            sampleID  = 0;
            iFeuH = 0;

            if (debug) cout << "Start FEU header..." << endl;
            // loop over the FEU header data
            while ( is_Feu_header( data ) ){
                if (debug) {
                    print_data(data);
                }
                if( iFeuH == 0 ){
                    FeuID = get_Feu_ID( data );
//          sampleID = data & 0x800;
                    sampleID = (data & 0x800) >> 3;  // Shift 11th bit down to bit 8
//          sampleID = 0;
                }
                else if( iFeuH == 1 ){
                    eventID =  get_Event_ID( data );
                }
                else if( iFeuH == 2 ){
                    timestamp =  get_timestamp( data );
                }
                else if( iFeuH == 3 ) {
                    sampleID += get_sample_ID( data );
                    fine_timestamp = get_fine_timestamp( data );
                }
                else if( iFeuH == 4 ){
                    eventID  += (uint64_t)get_Event_ID( data )<<12;
                }
                else if( iFeuH == 5 ){
                    timestamp +=  (uint64_t)get_timestamp( data )<<12;
                }
                else if( iFeuH == 6 ){
                    timestamp +=  (uint64_t)get_timestamp( data )<<24;
                }
                else if( iFeuH == 7 ){
//          timestamp +=  (uint64_t)get_timestamp( data )<<36;
                    timestamp += (uint64_t)(get_timestamp(data) & 0xFF) << 36;
                }

                iFeuH++;
                if (read16(is, data)) break;
            }

            if ( debug ){
                cout
                        << " * FEU H"
                        << setw(6) << FeuID
                        << setw(10) << eventID
                        << setw(6) << sampleID
                        << setw(20) << timestamp
                        << setw(5) << fine_timestamp
                        << " === " << iFeuH<<endl;
            }

            // ---- flush on eventID change -------------------------------
            // The FEU end-of-event marker is not reliable on RAW (non-ZS)
            // runs: when the packet carrying it is dropped, the next event
            // accumulates into the same entry and two events merge.  run_71
            // lost ~24% of its sample-groups to FEU bandwidth and ~33% of its
            // decoded entries were merged pairs.
            //
            // Every FEU header carries the eventID, so a change of eventID is
            // an unambiguous boundary.  Verified on run_71 group 023 at the
            // word level (analysis/fdf_scan.py): eventID present on every
            // surviving frame, stepping by exactly +1 with no gaps, and
            // sampleID strictly increasing within an event, never repeated.
            //
            // This is additive: on ZS runs the EoE fires normally and clears
            // the buffers, so this branch sees an empty buffer and does
            // nothing.  Behaviour on existing ZS data is unchanged.
            if ( !channel.empty() && eventID != accEventID ) {
                ULong64_t nEvt = eventID, nTs = timestamp;
                UShort_t  nFts = fine_timestamp;
                eventID = accEventID; timestamp = accTimestamp;
                fine_timestamp = accFineTs;
                flushEvent(1);
                eventID = nEvt; timestamp = nTs; fine_timestamp = nFts;
            }
            accEventID   = eventID;
            accTimestamp = timestamp;
            accFineTs    = fine_timestamp;

            // sample coverage of the event now being accumulated
            if (isZS) sawZS = true; else sawRaw = true;
            nFramesTotal++;
            if (sampleID < 1024) {
                sampSeen[sampleID] = 1;
                if (sampleID > maxSampleID) maxSampleID = sampleID;
            }

        }
        else if( isZS && is_data( data ) && isEvent && ! isFT ) {
            // read zero-suppressed data
            // =======================================
            // first line dreamId and channel Id
            // second line channel data
            // isEvent is to avoid reading a empty line after the EoE

            channelID = get_channel_ID( data );
            dreamID   = get_dream_ID_ZS( data );

            if( debug ){
                print_data(data);
                prev = cout.fill('0');
                cout << isEvent << "  "  << setw(4)  << hex << data << "   ";
            }
            // read next line
            read16(is,data);

            ampl = get_data( data );

            if( debug ){
                cout << setw(4)  << hex << data << endl;
                cout << dec ;
                cout.fill( prev);
                print_data(data);
            }


            channel.push_back( dreamID*64 + channelID );
            sample.push_back( sampleID );
            amplitude.push_back( ampl );
            if (read16(is, data)) break;

        }
        else if (!isZS && is_data_header(data) && isEvent && !isFT) {
            // read non-zero suppressed data
            // =======================================
            // first 4 words raw header, first 3 seem to be trigger id, skip
            // word 4 contains dream id
            // 5th to 68th words are channel data for channels 0-63
            // 69th to 74th words raw trailer, skip
            // isEvent is to avoid reading a empty line after the EoE

            int data_header_num = 0;
            bool got_dream_id = false;
            bool eof = false;
            // Raw header word 1 is Trigger Id MSB, 2 is Trigger Id ISB, 3 is Trigger Id LSB, 4 contains Dream Id
            while (is_data_header(data)) {
                data_header_num++;
                if (debug) { cout << "Data header #" << data_header_num << " "; print_data(data); }
                if (data_header_num == 4) {
                    dreamID = get_dream_ID(data);  // Contains Dream Id
                    got_dream_id = true;
                }
                eof = read16(is, data);
                if (eof)  break;
            }
            if (eof) {
                cout << "End of file reached while reading data header " << data_header_num << ". Exiting early!" << endl;
                break;
            }

            if (!got_dream_id) {
                cout << "Bad read, didn't get dream id from data header." << endl;
                dreamID = 0;
            }

            channelID = 0;
            while (is_data(data)) {
                ampl = get_data(data);

                if (debug) {
                    cout << setw(4) << hex << data;
                    cout << dec;
                    cout.fill(prev);
                    cout << "  channel #" << channelID << endl;
                    print_data(data);
                }

                channel.push_back(dreamID * 64 + channelID);
                sample.push_back(sampleID);
                amplitude.push_back(ampl);
                channelID++;
                eof = read16(is, data);
                if (eof)  break;
            }
            if (eof) {
                cout << "End of file reached while reading data channel " << channelID << ". Exiting early!" << endl;
                break;
            }

            if (channelID != 64) {
                cout << "Bad read, last channel ID " << channelID << " != 64" << endl;
                if (debug) {
                    print_data(data);
                    cout << "Press Enter to continue..." << endl;
                    cin.get(); // Wait for user to press Enter
                    while (!read16(is, data)) {
                        print_data(data);
                    }
                }
            }

            int data_trailer_num = 0;
            while (is_data_trailer(data)) {
                data_trailer_num++;
                if (debug) {
                    cout << "Data trailer #" << data_trailer_num << " ";
                    print_data(data);
                }
                eof = read16(is, data);
                if (eof) break;
            }
            if (eof) break;  // End of file

            // Shouldn't be true if successful read. If bad read, read till data header to reset for next event.
            while (!is_final_trailer(data) && !is_data_header(data)) {
                cout << "Bad read, expected data header but got the following:" << endl;
                print_data(data);
                eof = read16(is, data);
                if (eof) break;
            }
            if (eof) break;  // End of file
        }

        else{
            if (debug) {
                cout << "other ... " ;
                print_data(data);
            }
            if (read16(is, data)) break;
        }

        if (is_final_trailer(data)) {
            // read the final trailer
            // =====================

            isFT = true;
            // check if this is the end of the event (EoE)
            if (get_EoE(data) == 1) {

                if (i == 0) {
                    cout << " reading FEU " << FeuID << endl;
                }
                flushEvent(0);   // normal close, on the FEU end-of-event marker

                // reset all
                isEvent = false;
            }

            if (debug) {
                prev = cout.fill('0');
                cout << "FT " << setw(4) << hex << data << "   ";
            }

            read16(is, data);  // read one additional line for the VEP

            if (debug) {
                cout << setw(4) << hex << data << "  " << isEvent << endl;
                cout << dec;
                cout.fill(prev);
            }
            if (read16(is, data)) break;
        }
    }

    // Final flush: the last event of a RAW file often has no end-of-event
    // marker (its packet was among the dropped ones), so without this its
    // data would be silently discarded.
    if ( !channel.empty() ) {
        eventID = accEventID; timestamp = accTimestamp;
        fine_timestamp = accFineTs;
        flushEvent(2);
    }

    // ================= data-loss report =================================
    const ULong64_t nEvents      = (ULong64_t)i;
    const int       nSampExpect  = (int)maxSampleID + 1;
    const ULong64_t evtSpan      = (haveFirstEvent && lastEventID >= firstEventID)
                                 ? (lastEventID - firstEventID + 1) : nEvents;
    const ULong64_t missingEvts  = (evtSpan > nEvents) ? (evtSpan - nEvents) : 0;
    const bool      rawMode      = sawRaw && !sawZS;
    const double    acceptance   = (nEvents && nSampExpect)
        ? (double)sumDistinct / ((double)nEvents * (double)nSampExpect) : 1.0;
    const double    lossFrac     = 1.0 - acceptance;

    cout << "\n ---------------- decode summary ----------------\n"
         << "   mode                 : " << (rawMode ? "RAW (not zero-suppressed)"
                                          : (sawZS && sawRaw ? "MIXED ZS/RAW (!)" : "zero-suppressed"))
         << "\n   events written       : " << nEvents
         << "\n   eventID range        : " << firstEventID << " .. " << lastEventID
         << "  (span " << evtSpan << ")"
         << "\n   events MISSING       : " << missingEvts
         << "\n   closed by EoE marker : " << nFlushEoE
         << "\n   closed by eventID    : " << nFlushEvtId
         << (nFlushEvtId ? "   <-- their end-of-event packet was LOST" : "")
         << "\n   closed at EOF        : " << nFlushEof << endl;
    if (rawMode) {
        cout << "   samples per event    : " << nSampExpect << " expected\n"
             << "   sample completeness  : " << fixed << setprecision(1)
             << 100.0 * acceptance << " %" << endl;
    }

    const bool lossy = missingEvts > 0 || (rawMode && lossFrac > 0.001);
    if (lossy) {
        ostringstream banner;
        banner << "\n"
               << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
               << " !!  DATA LOSS in " << input_filename_ << "\n";
        if (rawMode && lossFrac > 0.001)
            banner << " !!  " << fixed << setprecision(1) << 100.0 * lossFrac
                   << " % of sample-groups never arrived (FEU bandwidth).\n"
                   << " !!  Mean waveforms MUST be divided by the per-sample\n"
                   << " !!  acceptance, saved here as TH1D 'sample_acceptance'.\n";
        if (missingEvts > 0)
            banner << " !!  " << missingEvts << " whole events missing from the eventID sequence.\n";
        if (nFlushEvtId)
            banner << " !!  " << nFlushEvtId << " events lost their end-of-event packet\n"
                   << " !!  (closed on eventID change instead -- data is intact).\n";
        banner << " !!  Numbers are also stored in the TTree 'decode_stats'.\n"
               << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        cout << banner.str() << flush;
        cerr << banner.str() << flush;   // so it survives a piped stdout
    } else {
        cout << "   >> no data loss detected" << endl;
    }
    cout << " ------------------------------------------------\n" << endl;

    // per-sample acceptance, written next to the data so the correction can
    // never be separated from the file it applies to
    TH1D hAcc("sample_acceptance",
              "fraction of events containing each sample index;sample;acceptance",
              nSampExpect, -0.5, nSampExpect - 0.5);
    hAcc.SetDirectory(&fout);
    for (int sIdx = 0; sIdx < nSampExpect; ++sIdx) {
        hAcc.SetBinContent(sIdx + 1, nEvents ? (double)sampHist[sIdx] / (double)nEvents : 0.0);
    }

    TTree stats("decode_stats", "decode_stats");
    stats.SetDirectory(&fout);
    ULong64_t s_events = nEvents, s_missing = missingEvts, s_eoe = nFlushEoE,
              s_evtid = nFlushEvtId, s_eof = nFlushEof, s_frames = nFramesTotal,
              s_first = firstEventID, s_last = lastEventID;
    Int_t     s_nsamp = nSampExpect, s_raw = rawMode ? 1 : 0;
    Double_t  s_acc = acceptance;
    stats.Branch("events",          &s_events,  "events/l");
    stats.Branch("events_missing",  &s_missing, "events_missing/l");
    stats.Branch("closed_eoe",      &s_eoe,     "closed_eoe/l");
    stats.Branch("closed_eventid",  &s_evtid,   "closed_eventid/l");
    stats.Branch("closed_eof",      &s_eof,     "closed_eof/l");
    stats.Branch("feu_frames",      &s_frames,  "feu_frames/l");
    stats.Branch("first_eventid",   &s_first,   "first_eventid/l");
    stats.Branch("last_eventid",    &s_last,    "last_eventid/l");
    stats.Branch("samples_expected",&s_nsamp,   "samples_expected/I");
    stats.Branch("raw_mode",        &s_raw,     "raw_mode/I");
    stats.Branch("sample_acceptance_mean", &s_acc, "sample_acceptance_mean/D");
    stats.Fill();

    cout << " Events analysed : " << i << endl;

    is.close();
    fout.Write();
    // hAcc and stats are stack objects registered with fout; detach them
    // before Close() so the file does not try to delete them.
    hAcc.SetDirectory(nullptr);
    stats.SetDirectory(nullptr);
    //fout.SetName( Form("FeuID%d.root", FeuID) );
    fout.Close();
    return 0;
}
