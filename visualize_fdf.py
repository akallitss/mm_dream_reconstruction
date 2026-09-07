#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick end-to-end visualization: FDF → decode → analyze → plot waveforms + hits.

Usage:
    python visualize_fdf.py <fdf_file> [options]

    fdf_file can also be an already-decoded .root file (skips decode step).

Options:
    --ped <dir>          Directory containing pedestal .root files (default: same dir as fdf)
    --event <id>         Event ID to plot (default: first event in file)
    --channels <range>   Channels to plot, e.g. "0-63" or "10,12,15" (default: all)
    --min-amp <val>      Skip channels whose max amplitude is below this (default: none)
    --feu <num>          FEU number override (auto-detected from filename if not given)
    --no-hits            Only decode/plot waveforms, skip hit analysis
    --derivative         Show derivative-trigger overlay on waveform plots
    --out <dir>          Where to write decoded/hits files (default: sibling dirs decoded_root/hits_root)
    --soft <dir>         Path to cmake build dir (default: auto-detect from script location)
    --redecode           Re-run decode even if output already exists
    --reanalyze          Re-run analysis even if hits file already exists
"""

import os
import re
import sys
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config — edit these defaults if your paths differ
# ---------------------------------------------------------------------------
DEFAULT_BASE_SOFT = os.path.join(os.path.dirname(__file__), 'cmake-build-debug')
DECODE_EXECUTABLE_REL = 'decoder/decode'
ANALYZE_EXECUTABLE_REL = 'waveform_analysis/analyze_waveforms'

DECODED_DIR_NAME = 'decoded_root'
HITS_DIR_NAME = 'hits_root'


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_channels(spec):
    """Parse '0-63' or '10,12,15' into a sorted numpy array."""
    if spec is None:
        return None
    if '-' in spec and ',' not in spec:
        lo, hi = spec.split('-')
        return np.arange(int(lo), int(hi) + 1)
    return np.array([int(c) for c in spec.split(',')])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('fdf_file', help='Path to .fdf or decoded .root file')
    p.add_argument('--ped', default=None, metavar='DIR', help='Pedestal directory')
    p.add_argument('--event', type=int, default=None, metavar='ID', help='Event ID to plot')
    p.add_argument('--channels', default=None, metavar='RANGE', help='Channel range, e.g. "0-63"')
    p.add_argument('--min-amp', type=float, default=None, metavar='VAL')
    p.add_argument('--feu', type=int, default=None, metavar='NUM')
    p.add_argument('--no-hits', action='store_true', help='Skip hit analysis')
    p.add_argument('--derivative', action='store_true', help='Show derivative overlay')
    p.add_argument('--out', default=None, metavar='DIR', help='Output directory for decoded/hits files')
    p.add_argument('--soft', default=DEFAULT_BASE_SOFT, metavar='DIR', help='cmake build directory')
    p.add_argument('--redecode', action='store_true')
    p.add_argument('--reanalyze', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Filename helpers (mirrors orchestrator conventions)
# ---------------------------------------------------------------------------

def extract_file_feu(filename):
    """Return (file_num, feu_num) from a filename like ..._NNN_FF.* or None."""
    m = re.match(r'.*_(\d{3})_(\d{2})[\._]', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def find_ped_file(ped_dir, feu_num):
    """Find the decoded pedestal root for a given FEU number."""
    if not ped_dir or not os.path.isdir(ped_dir):
        return ''
    matches = [
        f for f in os.listdir(ped_dir)
        if '_pedthr_' in f and f.endswith('.root')
        and extract_file_feu(f)[1] == feu_num
    ]
    if len(matches) == 1:
        return os.path.join(ped_dir, matches[0])
    if len(matches) > 1:
        print(f'[warn] Multiple pedestal files found for FEU {feu_num}: {matches}')
    return ''


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def resolve_paths(args):
    """Return (decoded_root, hits_root, feu_num, ped_dir) from CLI args."""
    input_path = os.path.abspath(args.fdf_file)
    basename = os.path.basename(input_path)

    is_fdf = input_path.endswith('.fdf')
    is_root = input_path.endswith('.root') and '_hits' not in basename

    if not is_fdf and not is_root:
        sys.exit(f'[error] Input must be a .fdf or decoded .root file: {input_path}')

    # Determine base name (without extension) for deriving sibling paths
    stem = basename.replace('.fdf', '').replace('.root', '')
    file_num, feu_num = extract_file_feu(basename)
    if args.feu is not None:
        feu_num = args.feu

    parent = os.path.dirname(input_path)

    # Output directory: args.out, or sibling of raw_daq_data, or same dir
    if args.out:
        out_base = os.path.abspath(args.out)
    elif 'raw_daq_data' in parent:
        out_base = os.path.dirname(parent)  # subrun dir
    else:
        out_base = parent

    decoded_dir = os.path.join(out_base, DECODED_DIR_NAME)
    hits_dir = os.path.join(out_base, HITS_DIR_NAME)

    if is_fdf:
        decoded_root = os.path.join(decoded_dir, stem + '.root')
    else:
        decoded_root = input_path
        decoded_dir = os.path.dirname(input_path)

    hits_root = os.path.join(hits_dir, stem + '_hits.root')

    # Pedestal location
    ped_dir = args.ped
    if ped_dir is None:
        # Default: same dir as decoded root (mirrors orchestrator PEDESTAL_LOC='same')
        ped_dir = decoded_dir if is_fdf else os.path.dirname(decoded_root)

    return decoded_root, hits_root, feu_num, ped_dir, is_fdf


def decode(fdf_path, decoded_root, base_soft, redecode=False):
    exe = os.path.join(base_soft, DECODE_EXECUTABLE_REL)
    if not os.path.isfile(exe):
        sys.exit(f'[error] Decode executable not found: {exe}')

    if os.path.exists(decoded_root) and not redecode:
        print(f'[decode] Already exists, skipping: {os.path.basename(decoded_root)}')
        return

    os.makedirs(os.path.dirname(decoded_root), exist_ok=True)
    print(f'[decode] {os.path.basename(fdf_path)} → {decoded_root}')
    ret = subprocess.run([exe, fdf_path, decoded_root])
    if ret.returncode != 0:
        sys.exit(f'[error] Decode failed (exit {ret.returncode})')


def analyze(decoded_root, hits_root, ped_path, base_soft, reanalyze=False):
    exe = os.path.join(base_soft, ANALYZE_EXECUTABLE_REL)
    if not os.path.isfile(exe):
        sys.exit(f'[error] Analyzer executable not found: {exe}')

    if os.path.exists(hits_root) and not reanalyze:
        print(f'[analyze] Already exists, skipping: {os.path.basename(hits_root)}')
        return

    os.makedirs(os.path.dirname(hits_root), exist_ok=True)
    cmd = [exe, decoded_root, hits_root]
    if ped_path:
        cmd.append(ped_path)
        print(f'[analyze] Using pedestal: {os.path.basename(ped_path)}')
    else:
        print('[analyze] No pedestal file found, continuing without')

    print(f'[analyze] {os.path.basename(decoded_root)} → {hits_root}')
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        sys.exit(f'[error] Analyze failed (exit {ret.returncode})')


# ---------------------------------------------------------------------------
# Plotting (adapted from nTof_x17/plot_waveform_hits.py)
# ---------------------------------------------------------------------------

def load_waveforms(decoded_root):
    try:
        import uproot
    except ImportError:
        sys.exit('[error] uproot not installed. Run: pip install uproot')
    with uproot.open(decoded_root) as f:
        nt = f['nt']
        evt_ids = nt['eventId'].array(library='np')
        ftsts = nt['ftst'].array(library='np')
        samples = nt['sample'].array(library='np')
        channels = nt['channel'].array(library='np')
        amplitudes = nt['amplitude'].array(library='np')
    return evt_ids, ftsts, samples, channels, amplitudes


def load_hits(hits_root):
    try:
        import uproot
    except ImportError:
        sys.exit('[error] uproot not installed. Run: pip install uproot')
    with uproot.open(hits_root) as f:
        hits = f['hits']
        data = {k: hits[k].array(library='np') for k in [
            'eventId', 'channel', 'sample', 'amplitude',
            'max_sample', 'left_sample', 'right_sample',
            'local_baseline', 'time_over_threshold'
        ]}
        peds = None
        if 'pedestals' in f:
            peds = {k: f['pedestals'][k].array(library='np') for k in ['channel', 'mean']}
    return data, peds


def plot_overview(evt_ids, amplitudes):
    max_amps = [np.max(a) if len(a) > 0 else 0 for a in amplitudes]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(evt_ids, max_amps, marker='o', linestyle='none', markersize=3)
    ax.set_title('Max Amplitude per Event')
    ax.set_xlabel('Event ID')
    ax.set_ylabel('Max Amplitude (ADC)')
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_waveforms_and_hits(evt_ids, ftsts, samples, channels_arr, amplitudes,
                             hits_data, peds, event_id, channel_ids,
                             min_amp=None, plot_derivative=False):
    match = np.where(evt_ids == event_id)[0]
    if len(match) == 0:
        print(f'[warn] Event {event_id} not found in waveform tree.')
        return

    idx = match[0]
    ftst = ftsts[idx]
    max_ftst = int(np.max(ftsts))
    sample_period_ns = 10 * (max_ftst + 1)

    evt_samples = samples[idx]
    evt_channels = channels_arr[idx]
    evt_amps = amplitudes[idx]

    has_hits = hits_data is not None

    for ch in channel_ids:
        mask = (evt_channels == ch)
        if not np.any(mask):
            continue

        wf_x = evt_samples[mask] + (ftst / (max_ftst + 1))
        wf_y = evt_amps[mask].astype(float)

        if min_amp is not None and np.max(wf_y) < min_amp:
            print(f'  skipping ch {ch}: max amp {np.max(wf_y):.0f} < {min_amp}')
            continue

        # Pedestal subtraction
        pedestal = 0.0
        if peds is not None:
            pm = (peds['channel'] == ch)
            if np.any(pm):
                pedestal = float(peds['mean'][pm][0])
        wf_y = wf_y - pedestal

        # Gather hits for this channel/event
        hit_info = []
        if has_hits:
            hm = (hits_data['eventId'] == event_id) & (hits_data['channel'] == ch)
            for hs, ha, hi, hl, hr, hb, ht in zip(
                hits_data['sample'][hm],
                hits_data['amplitude'][hm],
                hits_data['max_sample'][hm],
                hits_data['left_sample'][hm],
                hits_data['right_sample'][hm],
                hits_data['local_baseline'][hm],
                hits_data['time_over_threshold'][hm],
            ):
                hit_info.append((float(hs), float(ha), float(hi), float(hl), float(hr),
                                  float(hb), float(ht)))

        # --- Main waveform + hits plot ---
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(wf_x, wf_y, marker='.', linewidth=1, label='Waveform', zorder=2)
        y_min = float(np.min(wf_y))

        for hs, ha, hi, hl, hr, hb, ht in hit_info:
            print(f'  ch {ch:3d}  hit @ sample {hs:.2f}  amp {ha:.1f}  '
                  f'left {hl:.1f}  right {hr:.1f}  base {hb:.1f}  ToT {ht:.1f} ns')
            ax.axvline(hs, color='red', linestyle='--', alpha=0.7, zorder=3)
            ax.axvspan(hl, hr, color='gray', alpha=0.1)
            ax.hlines(hb - pedestal, hl - 8, hr, color='green', linestyle='-', alpha=0.7, zorder=3)
            ax.hlines(ha + hb - pedestal, hi - 1, hi + 1, color='red', lw=2, alpha=0.7, zorder=3)
            ax.scatter(hi, ha + hb - pedestal, color='red', marker='x', s=40, zorder=4)
            ax.annotate(f'{hs:.1f}', xy=(hs, y_min), xytext=(-4, 0),
                        textcoords='offset pixels', color='red', fontsize=10,
                        rotation=90, va='center', ha='right')

        n_hits = len(hit_info)
        title_suffix = f'  |  {n_hits} hit{"s" if n_hits != 1 else ""}' if has_hits else ''
        ax.set_title(f'Event {event_id}, Channel {ch}{title_suffix}')
        ax.set_xlabel(f'Sample index  (sample period ≈ {sample_period_ns} ns)')
        ax.set_ylabel('Amplitude (ADC, ped-subtracted)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        # --- Optional derivative overlay ---
        if plot_derivative:
            _plot_derivative(event_id, ch, wf_x, wf_y, hit_info, y_min)


def _plot_derivative(event_id, ch, wf_x, wf_y, hit_info, y_min):
    DERIV_SMOOTH_HW = 3
    DERIV_THR_SIGMA = 3.0
    DERIV_MERGE_DIST = 4
    AMP_THR_SIGMA = 5.0
    MIN_WIDTH_SAMPLES = 2
    noise_rms = 1.0

    N = len(wf_y)
    if N < 3:
        return
    amp = np.asarray(wf_y, dtype=float)
    idx = np.asarray(wf_x, dtype=float)

    hw = DERIV_SMOOTH_HW
    smooth = np.convolve(amp, np.ones(2 * hw + 1) / (2 * hw + 1), mode='same')
    for i in range(hw):
        lo, hi = max(0, i - hw), min(N - 1, i + hw)
        smooth[i] = amp[lo:hi + 1].mean()
    for i in range(N - hw, N):
        lo, hi = max(0, i - hw), min(N - 1, i + hw)
        smooth[i] = amp[lo:hi + 1].mean()

    deriv = np.empty(N)
    deriv[1:-1] = 0.5 * (smooth[2:] - smooth[:-2])
    deriv[0] = smooth[1] - smooth[0]
    deriv[-1] = smooth[-1] - smooth[-2]

    deriv_thr = DERIV_THR_SIGMA * noise_rms
    amp_thr = AMP_THR_SIGMA * noise_rms

    is_local_max = (deriv[1:-1] > deriv_thr) & \
                   (deriv[1:-1] >= deriv[:-2]) & \
                   (deriv[1:-1] >= deriv[2:])
    seeds = np.where(is_local_max)[0] + 1

    merged = []
    for s in seeds:
        if merged and (s - merged[-1]) <= DERIV_MERGE_DIST:
            merged[-1] = s if deriv[s] > deriv[merged[-1]] else merged[-1]
        else:
            merged.append(s)

    pulse_regions = []
    for k, seed in enumerate(merged):
        start = seed
        while start > 0 and amp[start - 1] > amp_thr:
            start -= 1
        end = seed
        while end + 1 < N and amp[end + 1] > amp_thr:
            end += 1
        if k + 1 < len(merged):
            ns = merged[k + 1]
            ns_start = ns
            while ns_start > 0 and amp[ns_start - 1] > amp_thr:
                ns_start -= 1
            if ns_start <= end:
                end = seed + int(np.argmin(amp[seed:ns]))
        if end - start + 1 >= MIN_WIDTH_SAMPLES:
            pulse_regions.append((start, end))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx, amp, marker='.', linewidth=1, label='Waveform')

    deriv_scale = np.ptp(amp) / (np.ptp(deriv) + 1e-9)
    ax.plot(idx, deriv * deriv_scale + y_min, color='orange', linewidth=1,
            linestyle='--', alpha=0.75, label=f'Deriv ×{deriv_scale:.1f}')
    ax.axhline(deriv_thr * deriv_scale + y_min, color='orange', linestyle=':',
               linewidth=0.8, label=f'Deriv thr ({DERIV_THR_SIGMA}σ)')

    if merged:
        ax.scatter(idx[merged], amp[merged], marker='^', s=80, color='red',
                   zorder=5, label='Rising-edge seeds')

    for i, (start, end) in enumerate(pulse_regions):
        ax.axvline(idx[start], color='green', linestyle='--', linewidth=1.0, alpha=0.8,
                   label='Pulse region start' if i == 0 else None)
        ax.axvline(idx[end], color='purple', linestyle='--', linewidth=1.0, alpha=0.8,
                   label='Pulse region end' if i == 0 else None)

    for hs, ha, hi, hl, hr, hb, ht in hit_info:
        ax.axvline(hs, color='red', linestyle='--', alpha=0.5, zorder=3)

    ax.set_title(f'Event {event_id}, Channel {ch} — derivative view')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Amplitude (ADC, ped-subtracted)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    decoded_root, hits_root, feu_num, ped_dir, is_fdf = resolve_paths(args)

    print(f'[info] Input:        {args.fdf_file}')
    print(f'[info] Decoded root: {decoded_root}')
    if not args.no_hits:
        print(f'[info] Hits root:    {hits_root}')
    print(f'[info] Pedestal dir: {ped_dir}')
    if feu_num is not None:
        print(f'[info] FEU:          {feu_num}')

    # --- Decode ---
    if is_fdf:
        decode(args.fdf_file, decoded_root, args.soft, redecode=args.redecode)

    if not os.path.exists(decoded_root):
        sys.exit(f'[error] Decoded root not found: {decoded_root}')

    # --- Analyze ---
    hits_data, peds = None, None
    if not args.no_hits:
        ped_path = find_ped_file(ped_dir, feu_num) if feu_num is not None else ''
        analyze(decoded_root, hits_root, ped_path, args.soft, reanalyze=args.reanalyze)
        if os.path.exists(hits_root):
            hits_data, peds = load_hits(hits_root)
        else:
            print('[warn] Hits file not produced — plotting waveforms only.')

    # --- Load waveforms ---
    print('[info] Loading waveforms...')
    evt_ids, ftsts, samples, channels_arr, amplitudes = load_waveforms(decoded_root)
    print(f'[info] Found {len(evt_ids)} events, '
          f'event IDs [{evt_ids.min()}–{evt_ids.max()}]')

    # --- Resolve event ---
    event_id = args.event if args.event is not None else int(evt_ids[0])
    print(f'[info] Plotting event {event_id}')

    # --- Resolve channels ---
    channel_ids = parse_channels(args.channels)
    if channel_ids is None:
        # Default: all channels present in this event
        ev_match = np.where(evt_ids == event_id)[0]
        if len(ev_match) == 0:
            sys.exit(f'[error] Event {event_id} not in waveform file.')
        channel_ids = np.unique(channels_arr[ev_match[0]])
        print(f'[info] Plotting {len(channel_ids)} channels present in event {event_id}')

    # --- Plot ---
    plot_overview(evt_ids, amplitudes)
    plot_waveforms_and_hits(
        evt_ids, ftsts, samples, channels_arr, amplitudes,
        hits_data, peds,
        event_id=event_id,
        channel_ids=channel_ids,
        min_amp=args.min_amp,
        plot_derivative=args.derivative,
    )

    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
