#!/usr/bin/env python3
"""
QA script: decode (optional) → analyze_waveforms (optional) → plot waveforms + hit overlays per FEU.

Typical workflow:
  1. Set DECODE=False, ANALYZE=False → just plot existing decoded + hits files.
  2. Edit waveform_analysis C++ code, recompile, set ANALYZE=True → rerun analysis, re-plot.
  3. Compare PDFs across iterations.

Requires: uproot, numpy, matplotlib
  Suggested interpreter: /home/dylan/PycharmProjects/nTof_x17/venv/bin/python3
  Or: pip install uproot numpy matplotlib
"""

import os
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless PDF; change to 'TkAgg' or 'Qt5Agg' for INTERACTIVE=True
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ════════════════════════════════════════════════════════════════════════════
#   CONFIG  ── edit this block
# ════════════════════════════════════════════════════════════════════════════

BASE_SOFT  = '/home/dylan/CLionProjects/mm_strip_reconstruction/cmake-build-debug'
# SUBRUN_DIR = '/media/dylan/data/x17/may_beam/runs/run_67/hv_scan_drift_600_resist_530'
SUBRUN_DIR = '/media/dylan/data/x17/may_beam/runs/run_74/run6'
OUTPUT_DIR = '/media/dylan/data/x17/may_beam/qa_test'

# ── Pipeline steps ────────────────────────────────────────────────────────
DECODE  = True   # decode fdf → decoded_root  (no fdf files exist for this run)
ANALYZE = True   # re-run analyze_waveforms; flip to True after recompiling
COMBINE = True   # combine per-FEU hits → combined_hits_root
FORCE   = True   # overwrite existing decoded/hits/combined files

TIME_PER_SAMPLE = 20.0  # ns per sample (passed to analyze_waveforms via --tps)

# Pedestal discovery for the analyze step (ignored when ANALYZE=False)
PEDESTAL_LOC  = 'find'   # 'same' | 'abs' | 'find'
PEDESTAL_BASE = '/media/dylan/data/x17/may_beam/pedestals'

# ── Plotting ──────────────────────────────────────────────────────────────
FEU_NUMS  = None   # None → all; or e.g. [1, 2, 3]
FILE_NUMS = None   # None → all; or e.g. [0]

# Events with no hit ≥ MIN_AMP are skipped; only the top MAX_EVENTS (by total
# hit amplitude) are plotted per (FEU, file_num) pair.
MIN_AMP    = 50.0
MAX_EVENTS = 10

# Specific event IDs to plot — overrides MAX_EVENTS / MIN_AMP when set.
# Example: EVENTS = [5, 17, 42]
# EVENTS = None
EVENTS = [5]

MAX_CH_PER_PAGE  = 6    # max channel subplots per figure page
MAX_CH_PER_EVENT = 24   # max channels to plot per event (top-N by amplitude)

INTERACTIVE = False   # True → plt.show() instead of writing PDF

# ════════════════════════════════════════════════════════════════════════════

RAW_DIR_NAME          = 'raw_daq_data'
DECODED_DIR_NAME      = 'decoded_root'
HITS_DIR_NAME         = 'hits_root'
COMBINED_DIR_NAME     = 'combined_hits_root'

DECODE_EXE  = os.path.join(BASE_SOFT, 'decoder', 'decode')
ANALYZE_EXE = os.path.join(BASE_SOFT, 'waveform_analysis', 'analyze_waveforms')
COMBINE_EXE = os.path.join(BASE_SOFT, 'feu_hit_combiner', 'combine_feus_hits')


# ── Filename helpers ──────────────────────────────────────────────────────

def extract_nums(filename):
    """Return (file_num, feu_num) from  ..._XXX_YY.root  or None."""
    m = re.search(r'_(\d{3})_(\d{2})[._]', filename)
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_pedestal(feu_num, ped_dir):
    """Return path to pedestal root for feu_num in ped_dir, or ''."""
    if not ped_dir or not os.path.isdir(ped_dir):
        return ''
    for f in os.listdir(ped_dir):
        m = re.search(r'_(\d{3})_(\d{2})', f)
        if m and int(m.group(2)) == feu_num and '_pedthr_' in f and f.endswith('.root'):
            return os.path.join(ped_dir, f)
    return ''


def resolve_pedestal_dir(raw_dir):
    """Resolve ped directory from PEDESTAL_LOC setting."""
    if PEDESTAL_LOC == 'same':
        return raw_dir
    if PEDESTAL_LOC == 'abs':
        return PEDESTAL_BASE
    if PEDESTAL_LOC == 'find':
        txt = os.path.join(raw_dir, 'pedestal_run.txt')
        if os.path.exists(txt):
            ped_run = open(txt).read().strip()
            return os.path.join(PEDESTAL_BASE, ped_run, 'pedestals')
        print(f'[analyze] pedestal_run.txt not found in {raw_dir}')
    return ''


# ── Pipeline steps ────────────────────────────────────────────────────────

def run_decode(raw_dir, decoded_dir):
    """Decode all fdf data files found in raw_dir → decoded_dir."""
    if not os.path.isdir(raw_dir):
        print(f'[decode] raw_dir not found: {raw_dir}')
        return
    all_fdfs = [f for f in os.listdir(raw_dir) if f.endswith('.fdf')]
    if not all_fdfs:
        print(f'[decode] No .fdf files found in {raw_dir} — skipping decode')
        return
    os.makedirs(decoded_dir, exist_ok=True)
    # Pedestals first (in-place, same dir)
    for f in os.listdir(raw_dir):
        if '_pedthr_' in f and f.endswith('.fdf'):
            fdf = os.path.join(raw_dir, f)
            root_out = os.path.join(raw_dir, f.replace('.fdf', '.root'))
            if FORCE or not os.path.exists(root_out):
                print(f'[decode] ped {f}')
                os.system(f'"{DECODE_EXE}" "{fdf}" "{root_out}"')
    # Data files
    for f in os.listdir(raw_dir):
        if '_datrun_' in f and f.endswith('.fdf'):
            fdf = os.path.join(raw_dir, f)
            root_out = os.path.join(decoded_dir, f.replace('.fdf', '.root'))
            if FORCE or not os.path.exists(root_out):
                print(f'[decode] data {f}')
                os.system(f'"{DECODE_EXE}" "{fdf}" "{root_out}"')


def run_analyze(decoded_dir, raw_dir, hits_dir):
    """Run analyze_waveforms on all decoded root files → hits_dir."""
    if not os.path.isdir(decoded_dir):
        print(f'[analyze] decoded_dir not found: {decoded_dir}')
        return
    os.makedirs(hits_dir, exist_ok=True)
    ped_dir = resolve_pedestal_dir(raw_dir)
    for f in sorted(os.listdir(decoded_dir)):
        if '_datrun_' not in f or not f.endswith('.root'):
            continue
        nums = extract_nums(f)
        if nums is None:
            continue
        fnum, feunum = nums
        root_path = os.path.join(decoded_dir, f)
        hits_path = os.path.join(hits_dir, f.replace('.root', '_hits.root'))
        ped_path = find_pedestal(feunum, ped_dir)
        if ped_path:
            print(f'[analyze] FEU {feunum:02d} file {fnum:03d}  ped={os.path.basename(ped_path)}')
        else:
            print(f'[analyze] FEU {feunum:02d} file {fnum:03d}  (no pedestal)')
        cmd = f'"{ANALYZE_EXE}" "{root_path}" "{hits_path}" "{ped_path}" --tps {TIME_PER_SAMPLE}'
        os.system(cmd)


def run_combine(hits_dir, combined_dir):
    """Combine per-FEU hits files for each file_num → combined_hits_root."""
    if not os.path.isdir(hits_dir):
        print(f'[combine] hits_dir not found: {hits_dir}')
        return
    os.makedirs(combined_dir, exist_ok=True)

    # Group hits files by file_num
    by_fnum: dict = {}
    for f in os.listdir(hits_dir):
        if not f.endswith('_hits.root'):
            continue
        nums = extract_nums(f)
        if nums:
            fnum, feunum = nums
            by_fnum.setdefault(fnum, {})[feunum] = os.path.join(hits_dir, f)

    for fnum, feu_map in sorted(by_fnum.items()):
        # Derive combined output name from the first hits file
        first_name = os.path.basename(next(iter(feu_map.values())))
        combined_name = re.sub(r'(_\d{3}_)\d{2}(_hits\.root)$', r'\1feu-combined\2', first_name)
        combined_path = os.path.join(combined_dir, combined_name)
        if not FORCE and os.path.exists(combined_path):
            print(f'[combine] file {fnum:03d} already combined, skipping')
            continue
        print(f'[combine] file {fnum:03d}  ({len(feu_map)} FEU(s)) → {combined_name}')
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=True) as tmp:
            for feunum, path in sorted(feu_map.items()):
                tmp.write(f'{path} {feunum}\n')
            tmp.flush()
            subprocess.run([COMBINE_EXE, tmp.name, combined_path], check=True)


def find_file_pairs(decoded_dir, hits_dir):
    """
    Return list of (file_num, feu_num, decoded_path, hits_path) for matching pairs.
    """
    decoded_map = {}
    if os.path.isdir(decoded_dir):
        for f in os.listdir(decoded_dir):
            if '_datrun_' in f and f.endswith('.root'):
                nums = extract_nums(f)
                if nums:
                    decoded_map[nums] = os.path.join(decoded_dir, f)

    pairs = []
    if os.path.isdir(hits_dir):
        for f in os.listdir(hits_dir):
            if not f.endswith('_hits.root'):
                continue
            nums = extract_nums(f)
            if nums and nums in decoded_map:
                pairs.append((*nums, decoded_map[nums], os.path.join(hits_dir, f)))
    return sorted(pairs)


# ── Plotting ──────────────────────────────────────────────────────────────

def load_hits(hits_path):
    """Load hits tree and pedestals from hits root file. Returns dicts."""
    import uproot
    with uproot.open(hits_path) as hf:
        hits = hf['hits']
        data = {k: hits[k].array(library='np') for k in hits.keys()}

        ped_by_ch = {}
        if 'pedestals' in hf:
            peds = hf['pedestals']
            for ch, mean, rms in zip(
                peds['channel'].array(library='np'),
                peds['mean'].array(library='np'),
                peds['rms'].array(library='np'),
            ):
                ped_by_ch[int(ch)] = (float(mean), float(rms))
    return data, ped_by_ch


def load_waveforms(decoded_path):
    """Load waveform nt tree. Returns arrays."""
    import uproot
    with uproot.open(decoded_path) as wf:
        nt = wf['nt']
        evt_ids  = nt['eventId'].array(library='np')
        ftsts    = nt['ftst'].array(library='np')
        wf_samps = nt['sample'].array(library='np')
        wf_chs   = nt['channel'].array(library='np')
        wf_amps  = nt['amplitude'].array(library='np')
    return evt_ids, ftsts, wf_samps, wf_chs, wf_amps


def make_summary_page(feu_num, file_num, hits_data, ped_by_ch):
    """One figure with amplitude histogram + channel occupancy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'FEU {feu_num:02d}  file {file_num:03d}  —  summary', fontsize=11)

    amps = hits_data['amplitude']
    chs  = hits_data['channel']
    evts = hits_data['eventId']

    # Hit amplitude distribution
    ax = axes[0]
    ax.hist(amps, bins=100, log=True, color='steelblue', edgecolor='none')
    ax.axvline(MIN_AMP, color='red', linestyle='--', label=f'MIN_AMP={MIN_AMP}')
    ax.set_xlabel('Amplitude (ADC)')
    ax.set_ylabel('Hits')
    ax.set_title('Hit amplitude distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Channel occupancy
    ax = axes[1]
    ax.hist(chs, bins=64, color='steelblue', edgecolor='none')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Hit count')
    ax.set_title('Channel occupancy')
    ax.grid(alpha=0.3)

    # Hits per event distribution
    ax = axes[2]
    hits_per_evt = np.bincount(evts.astype(int))
    hits_per_evt = hits_per_evt[hits_per_evt > 0]
    ax.hist(hits_per_evt, bins=40, color='steelblue', edgecolor='none')
    ax.set_xlabel('Hits per event')
    ax.set_ylabel('Events')
    ax.set_title('Hits per event')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def make_event_overview(evt_id, chs, samps, amps):
    """Small scatter overview: channel vs timing, size ~ amplitude."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    sizes = np.clip(amps / amps.max() * 150, 10, 300) if amps.max() > 0 else np.full_like(amps, 20)
    ax.scatter(chs, samps, s=sizes, c=amps, cmap='hot', edgecolors='none', alpha=0.85)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Timing sample')
    ax.set_title(f'Event {evt_id} — hit overview (size/colour ∝ amplitude)')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def make_channel_subplot(ax, ch, samp_idx, wf_amp, ped_mean, ped_rms,
                         hit_samps, hit_amps, hit_imaxes,
                         hit_lefts, hit_rights, hit_bases):
    """Fill one axes with waveform + hit overlays for a single channel."""
    ax.plot(samp_idx, wf_amp, 'b.-', lw=1.0, ms=3, label='Waveform', zorder=2)
    y_min = wf_amp.min() if len(wf_amp) else 0

    # Threshold line (5σ above zero after ped subtraction)
    thr = 5.0 * ped_rms
    ax.axhline(thr, color='purple', linestyle=':', lw=0.8, alpha=0.7, label=f'5σ thr ({thr:.1f})')

    for hs, ha, hi, hl, hr, hb in zip(hit_samps, hit_amps, hit_imaxes,
                                        hit_lefts, hit_rights, hit_bases):
        # baseline_disp = hb - ped_mean
        baseline_disp = hb
        peak_disp     = baseline_disp + ha
        # Integration window shading
        ax.axvspan(hl, hr, color='gray', alpha=0.12, zorder=1)
        # Baseline estimate
        ax.hlines(baseline_disp, hl - 5, hr, colors='green', lw=1.5, alpha=0.8, zorder=3)
        # Peak height marker
        ax.hlines(peak_disp, hi - 0.5, hi + 0.5, colors='red', lw=2.5, zorder=4)
        ax.scatter([hi], [peak_disp], color='red', marker='x', s=60, zorder=5)
        # Timing vertical line
        ax.axvline(hs, color='red', linestyle='--', lw=1.2, alpha=0.8, zorder=3)
        ax.annotate(f'A={ha:.0f}\nt={hs:.2f}',
                    xy=(hs, y_min), xytext=(2, 2), textcoords='offset points',
                    fontsize=7, color='darkred', va='bottom')

    ax.set_title(f'Ch {ch}  |  ped={ped_mean:.1f} ± {ped_rms:.1f}', fontsize=8)
    ax.set_xlabel('Sample index', fontsize=7)
    ax.set_ylabel('Amp − ped', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(alpha=0.3)


def plot_feu_file(feu_num, file_num, decoded_path, hits_path, pdf=None):
    """
    Plot waveforms + hit overlays for one (feu_num, file_num) file pair.

    If pdf (a PdfPages object) is given, figures are saved and closed immediately
    to avoid accumulating them in memory.  Otherwise a list of figures is returned
    for the caller to handle (e.g. plt.show()).
    """
    figs = []

    def emit(fig):
        """Save + close fig if streaming to PDF, else collect it."""
        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        else:
            figs.append(fig)

    print(f'  Loading {os.path.basename(hits_path)} ...')
    hits_data, ped_by_ch = load_hits(hits_path)
    evt_ids, ftsts, wf_samps, wf_chs, wf_amps = load_waveforms(decoded_path)
    max_ftst = int(np.max(ftsts)) if len(ftsts) > 0 else 0

    # ── Summary page ────────────────────────────────────────────────────
    emit(make_summary_page(feu_num, file_num, hits_data, ped_by_ch))

    # ── Select events to plot ────────────────────────────────────────────
    hit_evt = hits_data['eventId']
    hit_amp = hits_data['amplitude']
    hit_ch  = hits_data['channel']

    strong_mask = hit_amp >= MIN_AMP

    if EVENTS is not None:
        # User-specified event list — plot exactly these (in order given)
        top_evts = [e for e in EVENTS if e in hit_evt]
        missing  = [e for e in EVENTS if e not in hit_evt]
        if missing:
            print(f'  Warning: event IDs not found in hits tree: {missing}')
        print(f'  Plotting {len(top_evts)} user-specified event(s): {top_evts}')
    else:
        qualifying_evts = np.unique(hit_evt[strong_mask])

        def total_amp(evt):
            return float(np.sum(hit_amp[hit_evt == evt]))

        top_evts = sorted(qualifying_evts, key=total_amp, reverse=True)[:MAX_EVENTS]
        print(f'  {len(qualifying_evts)} events with hits ≥ {MIN_AMP}, plotting top {len(top_evts)}')

    # ── Per-event pages ──────────────────────────────────────────────────
    for evt in top_evts:
        emask = hit_evt == evt

        # All channels with a strong hit; cap to MAX_CH_PER_EVENT by amplitude
        strong_ch_mask = emask & strong_mask
        if not np.any(strong_ch_mask):
            continue

        # Pick top-N channels by max amplitude in this event
        ch_amp = {}
        for ch, amp in zip(hit_ch[strong_ch_mask], hit_amp[strong_ch_mask]):
            ch_amp[int(ch)] = max(ch_amp.get(int(ch), 0.0), float(amp))
        top_chs = sorted(ch_amp, key=ch_amp.get, reverse=True)[:MAX_CH_PER_EVENT]
        chs_to_plot = sorted(top_chs)

        # Waveform entry for this event
        wf_match = np.where(evt_ids == evt)[0]
        if len(wf_match) == 0:
            print(f'    Event {evt}: waveform entry not found, skipping')
            continue
        wi = wf_match[0]
        ftst         = int(ftsts[wi])
        evt_wf_chs   = wf_chs[wi]
        evt_wf_samps = wf_samps[wi]
        evt_wf_amps  = wf_amps[wi]

        # Overview scatter (all hits, not just strong ones)
        emit(make_event_overview(evt, hit_ch[emask],
                                 hits_data['sample'][emask], hit_amp[emask]))

        # Channel subplots, split into pages
        n_shown = len(chs_to_plot)
        total_chs = len(ch_amp)
        for page_start in range(0, n_shown, MAX_CH_PER_PAGE):
            page_chs = chs_to_plot[page_start:page_start + MAX_CH_PER_PAGE]
            n = len(page_chs)
            fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)
            caption = (f'FEU {feu_num:02d}  file {file_num:03d}  event {evt}'
                       f'  — top {n_shown}/{total_chs} channels by amplitude'
                       f'  (ch {page_chs[0]}–{page_chs[-1]})')
            fig.suptitle(caption, fontsize=9)

            for row, ch in enumerate(page_chs):
                ax = axes[row, 0]
                ch_mask = evt_wf_chs == ch
                ped_mean, ped_rms = ped_by_ch.get(ch, (0.0, 1.0))

                if np.any(ch_mask):
                    samp_idx    = evt_wf_samps[ch_mask].astype(float)
                    if max_ftst > 0:
                        samp_idx = samp_idx + ftst / (max_ftst + 1)
                    wf_amp_disp = evt_wf_amps[ch_mask].astype(float) - ped_mean
                else:
                    samp_idx    = np.array([])
                    wf_amp_disp = np.array([])

                ch_hit_mask = emask & (hit_ch == ch)
                make_channel_subplot(
                    ax, ch, samp_idx, wf_amp_disp, ped_mean, ped_rms,
                    hits_data['sample'][ch_hit_mask],
                    hits_data['amplitude'][ch_hit_mask],
                    hits_data['max_sample'][ch_hit_mask],
                    hits_data['left_sample'][ch_hit_mask],
                    hits_data['right_sample'][ch_hit_mask],
                    hits_data['local_baseline'][ch_hit_mask],
                )

            plt.tight_layout()
            emit(fig)

    return figs


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_dir      = os.path.join(SUBRUN_DIR, RAW_DIR_NAME)
    decoded_dir  = os.path.join(SUBRUN_DIR, DECODED_DIR_NAME)
    hits_dir     = os.path.join(SUBRUN_DIR, HITS_DIR_NAME)
    combined_dir = os.path.join(SUBRUN_DIR, COMBINED_DIR_NAME)

    if DECODE:
        run_decode(raw_dir, decoded_dir)

    if ANALYZE:
        run_analyze(decoded_dir, raw_dir, hits_dir)

    if COMBINE:
        run_combine(hits_dir, combined_dir)

    pairs = find_file_pairs(decoded_dir, hits_dir)
    if not pairs:
        print('No decoded+hits file pairs found. Check SUBRUN_DIR and run DECODE/ANALYZE first.')
        return

    # Group by FEU
    by_feu: dict = {}
    for fnum, feunum, dec_path, hits_path in pairs:
        if FEU_NUMS is not None and feunum not in FEU_NUMS:
            continue
        if FILE_NUMS is not None and fnum not in FILE_NUMS:
            continue
        by_feu.setdefault(feunum, []).append((fnum, dec_path, hits_path))

    if not by_feu:
        print('No FEUs/files selected. Check FEU_NUMS / FILE_NUMS filters.')
        return

    for feunum, items in sorted(by_feu.items()):
        print(f'\n=== FEU {feunum:02d} ({len(items)} file(s)) ===')

        if INTERACTIVE:
            for fnum, dec_path, hits_path in sorted(items):
                plot_feu_file(feunum, fnum, dec_path, hits_path)
            plt.show()
        else:
            pdf_path = os.path.join(OUTPUT_DIR, f'qa_feu{feunum:02d}.pdf')
            with PdfPages(pdf_path) as pdf:
                for fnum, dec_path, hits_path in sorted(items):
                    print(f'  File {fnum:03d}')
                    plot_feu_file(feunum, fnum, dec_path, hits_path, pdf=pdf)
            print(f'  → {pdf_path}')

    print('\ndonzo')


if __name__ == '__main__':
    main()
