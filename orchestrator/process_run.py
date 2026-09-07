#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on December 11 5:08 PM 2025
Created in CLion
Created as mm_strip_reconstruction/process_run

@author: Dylan Neff, dylan
"""

import os
import sys
import re
import math
import json
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import subprocess
from time import sleep

# =========================
# CONFIG
# =========================

BASE_SOFT = '/home/dylan/CLionProjects/mm_strip_reconstruction/cmake-build-release/'  # Release build: ~12x faster than debug
BASE_DATA = '/media/dylan/data/x17/cosmic_bench/'
# BASE_DATA = '/media/dylan/data/x17/may_beam/'
# BASE_SOFT = '/home/mx17/CLionProjects/mm_strip_reconstruction/build/'
# BASE_DATA = '/mnt/data/x17/beam_may/'
# BASE_SOFT = '/afs/cern.ch/work/d/dneff/git/mm_strip_reconstruction/build/'
# BASE_DATA = '/eos/experiment/ntof/data/x17/feb_beam/'

DECODE_EXECUTABLE = f'{BASE_SOFT}decoder/decode'
WAVEFORM_ANALYSIS_EXECUTABLE = f'{BASE_SOFT}waveform_analysis/analyze_waveforms'
COMBINE_HITS_EXECUTABLE = f'{BASE_SOFT}feu_hit_combiner/combine_feus_hits'

# ---- Dream shaping (peaking) time -> matched-filter gate width ----
# The analyzer's gate width should match the shaped pulse width. The Dream
# peaking time is encoded in register-1 lines of the DAQ .cfg
# ('Feu <id|*> Dream * 1 <w1> <w2> ...', code = (w2 >> 4) & 0xF). The DAQ
# writes an exact copy of the used config (*.cfg_cpy) next to the raw data on
# the nTOF layout; the cosmic bench only records the template path in
# run_config.json, resolved against these local checkouts:
DREAM_CFG_SEARCH_DIRS = [
    '/home/dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/dream_config/',
]
DREAM_PEAKING_NS = {0: 76, 1: 123, 2: 180, 3: 228, 4: 283, 5: 328, 6: 388, 7: 433, 8: 578}
MF_WIDTH_OVER_PEAKING = 1.7   # gate window / peaking time (measured pulse FWHM ~= 1.7 x Tp)

DECODE = True
# DECODE = False
REDECODE_PEDS = True
# REDECODE_PEDS = False
ANALYZE = True
COMBINE = True

FREE_THREADS = 2      # Leave this many threads free, or None to use all available

RAW_DREAM_DIR_NAME = 'raw_daq_data'
DECODED_ROOT_DIR_NAME = 'decoded_root'
HITS_DIR_NAME = 'hits_root'
COMBINED_HITS_DIR_NAME = 'combined_hits_root'

# PEDESTAL_LOC = 'find'  # 'same' for same dir as data, 'abs' for absolute path, 'find' to search from .prg files
PEDESTAL_LOC = 'same'  # 'same' for same dir as data, 'abs' for absolute path, 'find' to search from .prg files
# PEDESTAL_LOC = 'abs'  # 'same' for same dir as data, 'abs' for absolute path, 'find' to search from .prg files
PEDESTAL_DIR = f'{BASE_DATA}pedestals/'
# PEDESTAL_DIR = f'{BASE_DATA}pedestals/pedestals_02-02-26_10-49-02/pedestals/'


def main():
    # runs_dir = f'{BASE_DATA}runs/'
    # runs = ['run_19']
    runs_dir = f'{BASE_DATA}det_4/'
    runs = ['mx17_det3_HV_Scan_5-5-26']

    if len(sys.argv) == 2:
        runs = [sys.argv[1]]

    n_threads = effective_thread_count()
    print(f"Using {n_threads} worker threads")

    for run in runs:
        run_dir = os.path.join(runs_dir, run)

        for subrun in os.listdir(run_dir):
            subrun_dir = os.path.join(run_dir, subrun)
            if not os.path.isdir(subrun_dir):
                continue

            raw_dir = os.path.join(subrun_dir, RAW_DREAM_DIR_NAME)
            decoded_dir = os.path.join(subrun_dir, DECODED_ROOT_DIR_NAME)
            hits_dir = os.path.join(subrun_dir, HITS_DIR_NAME)
            combined_hits_dir = os.path.join(subrun_dir, COMBINED_HITS_DIR_NAME)

            # Determine whether we're working from raw fdfs or already-decoded roots
            decoded_only = False
            if not os.path.exists(raw_dir):
                if DECODE:
                    print(f"No raw data in {raw_dir}, skipping")
                    continue
                if not os.path.exists(decoded_dir):
                    print(f"No raw data in {raw_dir} and no decoded data in {decoded_dir}, skipping")
                    continue
                decoded_only = True
                print(f"No raw data found; working from decoded roots in {decoded_dir}")

            if ANALYZE:
                make_dir_if_not_exists(hits_dir)

            # data_root_stems: list of .root filenames (no path) for data files
            if decoded_only:
                data_root_stems = [f for f in os.listdir(decoded_dir)
                                   if '_datrun_' in f and f.endswith('.root')]
            else:
                fdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.fdf')]
                data_fdfs = [f for f in fdf_files if '_datrun_' in f]
                data_root_stems = [f.replace('.fdf', '.root') for f in data_fdfs]

            ped_fdf_path = ''
            if PEDESTAL_LOC == 'same':
                ped_fdf_path = decoded_dir if decoded_only else raw_dir
            elif PEDESTAL_LOC == 'abs':
                ped_fdf_path = PEDESTAL_DIR
            elif PEDESTAL_LOC == 'find':
                # Look for pedestal_run.txt file and get ped_run name from there
                ped_run_txt_name = 'pedestal_run.txt'
                ped_run_txt_path = os.path.join(subrun_dir, 'raw_daq_data', ped_run_txt_name)
                if os.path.exists(ped_run_txt_path):
                    with open(ped_run_txt_path, 'r') as f:
                        ped_run_name = f.read().strip()
                    ped_fdf_path = os.path.join(PEDESTAL_DIR, ped_run_name, 'pedestals')
                else:
                    print(f"No pedestal_run.txt found at {ped_run_txt_path}, cannot find pedestals, skipping run")
                    continue

            # ---- Decode pedestals serially ----
            if DECODE:
                ped_fdfs = [f for f in os.listdir(ped_fdf_path) if '_pedthr_' in f and f.endswith('.fdf')]
                if not REDECODE_PEDS:
                    # Check if decoded root files already there
                    already_decoded = []
                    for f in ped_fdfs:
                        root_path = os.path.join(ped_fdf_path, f.replace('.fdf', '.root'))
                        if os.path.exists(root_path):
                            already_decoded.append(f)

                    # Remove already_decoded from ped_fdfs
                    ped_fdfs = [f for f in ped_fdfs if f not in already_decoded]
                for f in ped_fdfs:
                    decode_file(
                        os.path.join(ped_fdf_path, f),
                        os.path.join(ped_fdf_path, f.replace('.fdf', '.root'))
                    )

            # ---- Decode + analyze data in parallel ----
            if DECODE:
                tasks = []
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    for f in data_fdfs:
                        fdf_path = os.path.join(raw_dir, f)
                        root_path = os.path.join(decoded_dir, f.replace('.fdf', '.root'))

                        if DECODE:
                            tasks.append(pool.submit(decode_file, fdf_path, root_path))

                    for t in as_completed(tasks):
                        t.result()

            if ANALYZE:
                # Gate width from the run's Dream shaping time when the cfg is
                # findable; otherwise the analyzer's auto default applies.
                sample_period_ns = run_sample_period_ns(run_dir)
                dream_cfg = find_dream_cfg(raw_dir, run_dir)
                peaking = parse_dream_peaking(dream_cfg) if dream_cfg else {}
                if peaking:
                    print(f"Dream peaking times from {dream_cfg}: {peaking} ns "
                          f"(sample period {sample_period_ns} ns)")
                zs_baseline = run_zs_baseline(run_dir)
                if zs_baseline:
                    print("Run is zero-suppressed with on-FEU pedestal "
                          "subtraction: passing --zs-baseline 1")

                tasks = []
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    for f in data_root_stems:
                        root_path = os.path.join(decoded_dir, f)
                        hits_path = os.path.join(hits_dir, f.replace('.root', '_hits.root'))

                        tasks.append(pool.submit(
                            analyze_file,
                            root_path,
                            ped_fdf_path,
                            hits_path,
                            sample_period_ns,
                            peaking,
                            zs_baseline
                        ))

                    for t in as_completed(tasks):
                        t.result()


            if COMBINE:
                # Get file numbers from data files
                file_nums = get_file_numbers_set(data_root_stems)
                print(f'File numbers to combine: {file_nums}')

                tasks = []
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    for fnum in file_nums:
                        feu_path_map = get_feu_files_for_filenum(hits_dir, fnum)
                        if not feu_path_map:
                            continue
                        tasks.append(pool.submit(
                            combine_hits,
                            feu_path_map,
                            combined_hits_dir
                        ))
                    for t in as_completed(tasks):
                        t.result()

    print("donzo")


# =========================
# UTILS
# =========================

def effective_thread_count() -> int:
    threads = os.cpu_count() or 1
    if FREE_THREADS is not None:
        threads -= FREE_THREADS
    return max(1, threads)


def make_dir_if_not_exists(path: str):
    os.makedirs(path, exist_ok=True)


def extract_file_numbers_tuple(filename: str, end_dot=True) -> Optional[Tuple[int, int]]:
    if end_dot:
        match = re.match(r'.*_(\d{3})_(\d{2})\..*', filename)
    else:
        match = re.match(r'.*_(\d{3})_(\d{2}).*', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def get_file_numbers_set(filenames: List[str]) -> set:
    file_nums_set = set()
    for f in filenames:
        nums = extract_file_numbers_tuple(f)
        if nums:
            fnum, feunum = nums
            file_nums_set.add(fnum)
    return file_nums_set


def get_feu_files_for_filenum(
    search_dir: str,
    file_num: int) -> Dict[int, str]:
    """
    Find all files in directory matching given file number.
    Map FEU number to filename paths.
    Args:
        search_dir (str): Directory to search in.
        file_num (int): File number to match.
    Returns:
        Dict[int, str]: Mapping of FEU number to filename.
    """
    feu_file_map = {}
    for f in os.listdir(search_dir):
        nums = extract_file_numbers_tuple(f, end_dot=False)
        if nums:
            fnum, feunum = nums
            if fnum == file_num:
                feu_file_map[feunum] = os.path.join(search_dir, f)
    return feu_file_map


def replace_feu_number_in_filename(
    filename: str,
    feu_num_replacement: str,
    end_dot=False) -> str:
    if end_dot:
        match = re.match(r'(.*_(\d{3})_)(\d{2})(\..*)', filename)
    else:
        match = re.match(r'(.*_(\d{3})_)(\d{2})(.*)', filename)
    if match:
        prefix = match.group(1)
        suffix = match.group(4)
        new_feu_str = f"{feu_num_replacement}"
        return f"{prefix}{new_feu_str}{suffix}"
    return filename


def find_file_feu_file(
    search_dir: str,
    file_num=None,
    feu_num=None,
    extension=None,
    flag=None
) -> List[str]:

    matches = []
    for f in os.listdir(search_dir):
        if extension and not f.endswith(extension):
            continue
        if flag and flag not in f:
            continue

        nums = extract_file_numbers_tuple(f)
        if nums:
            fnum, feunum = nums
            if file_num is not None and fnum != file_num:
                continue
            if feu_num is not None and feunum != feu_num:
                continue
            matches.append(f)

    return matches


# =========================
# WORKER FUNCTIONS
# =========================

def decode_file(fdf_path: str, out_root_path: str):
    make_dir_if_not_exists(os.path.dirname(out_root_path))
    cmd = f"{DECODE_EXECUTABLE} {fdf_path} {out_root_path}"
    print(f"[decode] {os.path.basename(fdf_path)}")
    os.system(cmd)


def parse_dream_peaking(cfg_path: str) -> Dict[str, int]:
    """Per-FEU Dream peaking time (ns) from a Dream .cfg / .cfg_cpy.

    Register-1 lines look like 'Feu <id|*> Dream <id|*> 1 <w1> <w2> ...';
    the peaking-time code sits in bits [7:4] of the second word (the cfg's own
    comment gives the code table). Later lines override earlier ones, so the
    usual wildcard-then-specific layout resolves naturally.
    """
    peaking: Dict[str, int] = {}
    try:
        with open(cfg_path) as f:
            for line in f:
                tok = line.split('#')[0].split()
                if len(tok) >= 7 and tok[0] == 'Feu' and tok[2] == 'Dream' and tok[4] == '1':
                    try:
                        code = (int(tok[6], 16) >> 4) & 0xF
                    except ValueError:
                        continue
                    ns = DREAM_PEAKING_NS.get(code)
                    if ns:
                        peaking[tok[1]] = ns
    except OSError:
        return {}
    return peaking


def find_dream_cfg(raw_dir: str, run_dir: str) -> Optional[str]:
    """Locate the Dream config for a run: the DAQ's exact *.cfg_cpy next to
    the raw data if present (nTOF layout), else the template named in
    run_config.json resolved against DREAM_CFG_SEARCH_DIRS."""
    if os.path.isdir(raw_dir):
        cpys = sorted(f for f in os.listdir(raw_dir) if f.endswith('.cfg_cpy'))
        preferred = [f for f in cpys if 'datrun' in f] or cpys
        if preferred:
            return os.path.join(raw_dir, preferred[0])
    rc_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(rc_path):
        try:
            with open(rc_path) as f:
                tmpl = json.load(f).get('dream_daq_info', {}).get('daq_config_template_path')
            if tmpl:
                base = os.path.basename(tmpl)
                for d in DREAM_CFG_SEARCH_DIRS:
                    p = os.path.join(d, base)
                    if os.path.exists(p):
                        return p
        except (OSError, ValueError):
            pass
    return None


def run_sample_period_ns(run_dir: str) -> Optional[float]:
    rc_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(rc_path):
        try:
            with open(rc_path) as f:
                sp = json.load(f).get('dream_daq_info', {}).get('sample_period')
            return float(sp) if sp else None
        except (OSError, ValueError, TypeError):
            pass
    return None


def run_zs_baseline(run_dir: str) -> bool:
    """True when the run took zero-suppressed data with ON-FEU pedestal
    subtraction (dream_daq_info in run_config.json): the decoded waveforms are
    then re-centred at 256 and analyze_waveforms needs --zs-baseline 1 (the
    pedestal file's per-channel means would be the wrong baseline; its RMS is
    still used for thresholds). Ported from the banco P2 fork, 2026-07-24."""
    rc_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(rc_path):
        try:
            with open(rc_path) as f:
                daq = json.load(f).get('dream_daq_info', {})
            return bool(daq.get('zero_suppress')) and bool(daq.get('pedestal_subtraction'))
        except (OSError, ValueError, TypeError):
            pass
    return False


def analyze_file(root_path: str, pedestal_dir: str, hits_out_path: str,
                 sample_period_ns: Optional[float] = None,
                 peaking: Optional[Dict[str, int]] = None,
                 zs_baseline: bool = False):
    file_num, feu_num = extract_file_numbers_tuple(os.path.basename(root_path))

    if pedestal_dir == 'same':
        pedestal_dir = os.path.dirname(root_path)

    ped_files = find_file_feu_file(
        pedestal_dir,
        feu_num=feu_num,
        extension='.root',
        flag='_pedthr_'
    )

    ped_path = ''
    if len(ped_files) == 1:
        ped_path = os.path.join(pedestal_dir, ped_files[0])
        print(f"[analyze] Using pedestal {ped_files[0]}")
    elif len(ped_files) > 1:
        print(f"[analyze] Multiple pedestals for FEU {feu_num}, skipping {root_path}")
        return
    else:
        print(f"[analyze] No pedestal for FEU {feu_num}, continuing without")

    extra = ''
    if sample_period_ns:
        extra += f" --tps {sample_period_ns:g}"
    # Gate width from the run's measured shaping time; without a cfg the
    # analyzer falls back to its own auto width (~300 ns / tps).
    if peaking and sample_period_ns:
        peak_ns = peaking.get(str(feu_num), peaking.get('*'))
        if peak_ns:
            mf = max(3, round(MF_WIDTH_OVER_PEAKING * peak_ns / sample_period_ns))
            extra += f" --mf {mf}"
    if zs_baseline:
        extra += " --zs-baseline 1"

    make_dir_if_not_exists(os.path.dirname(hits_out_path))
    cmd = f"{WAVEFORM_ANALYSIS_EXECUTABLE} {root_path} {hits_out_path} {ped_path}{extra}"
    print(f"[analyze] {os.path.basename(root_path)}{extra}")
    os.system(cmd)


def combine_hits(feu_hits_path_map: Dict[int, str], combined_out_dir_path: str):
    make_dir_if_not_exists(combined_out_dir_path)

    first_path = next(iter(feu_hits_path_map.values()), None)
    if first_path is None:
        print("No FEU files to combine, skipping")
        return
    first_file_name = os.path.basename(first_path)
    combined_file_name = replace_feu_number_in_filename(first_file_name, 'feu-combined', False)
    combined_out_path = os.path.join(combined_out_dir_path, combined_file_name)
    print(f"Combining hits into {combined_out_path}")

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tmp:
        for feu, path in feu_hits_path_map.items():
            tmp.write(f"{path} {feu}\n")
        tmp.flush()  # important!

        cmd = [
            COMBINE_HITS_EXECUTABLE,
            tmp.name,
            combined_out_path,
        ]

        print(f"[combine] {combined_file_name}")

        subprocess.run(cmd, check=True)


# BASE_SOFT = '/home/dylan/CLionProjects/mm_strip_reconstruction/cmake-build-debug/'
# BASE_DATA = '/media/dylan/data/x17/cosmic_bench/'
#
# # BASE_SOFT = '/home/mx17/CLionProjects/mm_strip_reconstruction/cmake-build-debug/'
# # BASE_DATA = '/mnt/data/x17/cosmic_bench/'
#
# DECODE_EXECUTABLE = f'{BASE_SOFT}decoder/decode'
# WAVEFORM_ANALYSIS_EXECUTABLE = f'{BASE_SOFT}waveform_analysis/analyze_waveforms'
#
# decode = True
# analyze = True
#
# def main():
#     # runs_dir = '/media/dylan/data/x17/nov_25_beam_test/dream_run/'
#     runs_dir = f'{BASE_DATA}det_1/'
#     # runs = ['mx17_det1_overnight_run_1-27-26']
#     runs = ['mx17_det1_daytime_run_1-28-26']
#     # pedestal_dir = 'ped_thresh_1_12_25_18_30'
#     # pedestal_dir = 'ped_thresh_30_11_25_14_40'
#     # runs_dir = '/media/dylan/data/sps_beam_test_25/run_54/rotation_30_test_0/'
#     # runs_dir = '/media/dylan/data/sps_beam_test_25/run_67/rotation_-60_banco_scan_0/'
#     # runs = ['raw_daq_data']
#     # pedestal_dir = 'dummy_peds'
#     # pedestal_dir = ''
#     pedestal_dir = 'same'
#
#     # raw_dream_dir_name = 'raw_dream'
#     raw_dream_dir_name = 'raw_daq_data'
#     decoded_root_dir_name = 'decoded_root'
#     hits_dir_name = 'hits_root'
#
#     for run in runs:
#         run_dir = os.path.join(runs_dir, run)
#         run_dir_dirs = os.listdir(run_dir)
#         for sub_run_name in run_dir_dirs:
#             if not os.path.isdir(os.path.join(run_dir, sub_run_name)):
#                 continue
#
#             sub_run_dir = os.path.join(run_dir, sub_run_name)
#             raw_dream_dir = os.path.join(sub_run_dir, raw_dream_dir_name)
#             if not os.path.exists(raw_dream_dir):
#                 print(f'No raw dream directory found at {raw_dream_dir}, skipping...')
#                 continue
#             if analyze:
#                 make_dir_if_not_exists(f'{sub_run_dir}/{hits_dir_name}')
#             print(f'Processing run directory: {raw_dream_dir}')
#             fdf_files = [file for file in os.listdir(raw_dream_dir) if file.endswith('.fdf')]
#
#             if decode:
#                 ped_fdf_files = [file for file in fdf_files if '_pedthr_' in file]
#                 for file in ped_fdf_files:
#                         file_path = os.path.join(raw_dream_dir, file)
#                         decoded_root_out_path = f'{sub_run_dir}/{decoded_root_dir_name}/{file.replace(".fdf", ".root")}'
#                         print(f'\nDecoding pedestal file {file_path}...')
#                         decode_fdf(file_path, decoded_root_out_path)
#                         print(f'Decoded to {decoded_root_out_path}\n')
#
#             data_fdf_files = [file for file in fdf_files if '_datrun_' in file]
#
#             for file in data_fdf_files:
#                 file_path = os.path.join(raw_dream_dir, file)
#                 decoded_root_out_path = f'{sub_run_dir}/{decoded_root_dir_name}/{file.replace(".fdf", ".root")}'
#                 if decode:
#                     print(f'\nDecoding {file_path}...')
#                     decode_fdf(file_path, decoded_root_out_path)
#                     print(f'Decoded to {decoded_root_out_path}')
#                 hits_out_path = f'{sub_run_dir}/{hits_dir_name}/{file.replace(".fdf", "_hits.root")}'
#                 if analyze:
#                     print(f'\nAnalyzing waveforms in {decoded_root_out_path}...')
#                     if pedestal_dir == 'same':
#                         sub_run_ped_path = os.path.join(sub_run_dir, decoded_root_dir_name)
#                     else:
#                         sub_run_ped_path = os.path.join(runs_dir, sub_run_dir, pedestal_dir)
#                     analyze_waveforms(decoded_root_out_path, sub_run_ped_path, hits_out_path)
#                     print(f'Analyzed waveforms in {decoded_root_out_path} to {hits_out_path}\n')
#     print('donzo')


# def decode_fdf(file_path, out_root_path=None):
#     """
#     Decode .fdf file and extract relevant data.
#     :param file_path: Path to the .fdf file.
#     :param out_root_path: Optional path for the output .root file. If None, replaces .fdf with .root in the input path.
#     """
#     if out_root_path is None:
#         out_root_path = file_path.replace('.fdf', '.root')
#     else:
#         make_dir_if_not_exists(os.path.dirname(out_root_path))
#     command = f"{DECODE_EXECUTABLE} {file_path} {out_root_path}"
#     os.system(command)
#
#
# def analyze_waveforms(root_path, pedestal_dir, hits_out_path=None):
#     """
#     Analyze waveforms from the decoded .root file.
#     :param root_path: Path to the decoded .root file.
#     :param pedestal_dir: Directory containing pedestal files.
#     :param hits_out_path: Optional path for the output hits .root file.
#     If None, replaces .root with _hits.root in the input path.
#     """
#     file_num, feu_num = extract_file_numbers_tuple(os.path.basename(root_path))
#
#     if pedestal_dir == 'same':
#         pedestal_dir = os.path.dirname(root_path)
#
#     ped_files = find_file_feu_file(pedestal_dir, file_num=None, feu_num=feu_num, file_extension='.root', flag='_pedthr_')
#     if not ped_files:
#         print(f'No pedestal file found for file number {file_num} and FEU number {feu_num}')
#         ped_file_path = ''
#     elif len(ped_files) > 1:
#         print(f'Multiple pedestal files found for file number {file_num} and FEU number {feu_num}: {ped_files}')
#         return
#     else:
#         ped_file_path = os.path.join(pedestal_dir, ped_files[0])
#         print(f'Using pedestal file: {ped_file_path}')
#     if hits_out_path is None:
#         hits_out_path = root_path.replace('.root', '_hits.root')
#     else:
#         make_dir_if_not_exists(os.path.dirname(hits_out_path))
#     print(f'\nhits_out_path: {hits_out_path}\n')
#     command = f"{WAVEFORM_ANALYSIS_EXECUTABLE} {root_path} {hits_out_path} {ped_file_path}"
#     print(f'Analyzing waveforms with command: {command}')
#     os.system(command)


# def find_file_feu_file(search_dir_path, file_num=None, feu_num=None, file_extension=None, flag=None) -> list:
#     """
#     Finds a list of files in the specified directory matching the given file number and FEU number.
#
#     Args:
#         search_dir_path (str): The directory to search in.
#         file_num (int): The file number to match (xxx).
#         feu_num (int): The FEU number to match (yy).
#         file_extension (str): The file extension to filter by.
#         Flag (str): A specific flag that should be present in the filename.
#         Returns:
#         List[str]: A list of matching filenames.
#     """
#     if file_num is None and feu_num is None and file_extension is None and flag is None:  # If no criteria provided, return all files
#         return os.listdir(search_dir_path)
#
#     matching_files = []
#     for filename in os.listdir(search_dir_path):
#         if file_extension is not None and not filename.endswith(file_extension):
#             continue
#
#         if flag is not None and flag not in filename:
#             continue
#
#         extracted = extract_file_numbers_tuple(filename)
#         if extracted is not None:
#             extracted_file_num, extracted_feu_num = extracted
#             if file_num is not None and extracted_file_num != file_num:
#                 continue
#             if feu_num is not None and extracted_feu_num != feu_num:
#                 continue
#             matching_files.append(filename)
#     return matching_files
#
#
#
# def extract_file_numbers_tuple(filename: str) -> Optional[Tuple[int, int]]:
#     """
#     Extracts the file number (xxx) and FEU number (yy) from a filename
#     following the pattern ..._xxx_yy.ext
#
#     Args:
#         filename (str): The name of the file.
#
#     Returns:
#         Optional[Tuple[int, int]]: A tuple (file_number, feu_number) as integers,
#                                     or None if the pattern is not found.
#     """
#     # Pattern explanation:
#     # r'.*': Matches any characters at the start (non-greedy)
#     # '_': Matches the literal underscore
#     # '(\d{3})': Capturing group 1: exactly three digits (xxx)
#     # '_': Matches the literal underscore
#     # '(\d{2})': Capturing group 2: exactly two digits (yy)
#     # '\..*': Matches a dot (start of extension) followed by any characters
#     pattern = r'.*_(\d{3})_(\d{2})\..*'
#
#     match = re.match(pattern, filename)
#
#     if match:
#         # Group 1 is the file number (xxx), Group 2 is the FEU number (yy)
#         file_num_str, feu_num_str = match.groups()
#         # Convert the strings to integers and return
#         return (int(file_num_str), int(feu_num_str))
#     else:
#         return None
#
#
# def make_dir_if_not_exists(dir_path: str):
#     """
#     Creates the directory if it does not already exist.
#     :param dir_path: Path to the directory.
#     """
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)


if __name__ == '__main__':
    main()
