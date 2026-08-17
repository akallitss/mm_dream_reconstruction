#!/usr/bin/env python3
# -- coding: utf-8 --
"""Full-chain reprocessing driver for the rays machine (sedipcaa28).

Runs raw .fdf -> decoded_root -> hits_root -> combined_hits_root over every
subrun under a base directory (default /mnt/cosmic_data/P2/Run).

Differs from process_run.py in the ways an unattended multi-hour run needs:

  * real exit codes -- process_run.py drives the executables with os.system()
    and discards the return value, so a decode that dies leaves a truncated
    .root behind and the run carries on. Here every step is subprocess.run()
    with the return code checked and stderr captured to a per-file log.
  * output validation -- rc==0 is necessary but not sufficient; the decoder
    prints "Decoder finished, events: 0" even on success, so log scraping is
    unreliable. Each product is size-checked instead.
  * a manifest -- every task result is journalled to JSON, so an interrupted
    run resumes where it stopped instead of redoing ~10 h of work.
  * a disk guard -- the volume is tight; the run halts cleanly before filling
    it rather than dying halfway through a write.

Python 3.7 compatible (rays has 3.7.4): no walrus, no 3.8+ syntax.

Usage
    ./reprocess_rays.py run                     # full chain, resumable
    ./reprocess_rays.py run --subrun NAME       # one subrun (validation)
    ./reprocess_rays.py status                  # progress + failures
    ./reprocess_rays.py clean --yes             # delete regenerable products
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG (rays)
# =========================

BASE_SOFT = '/local/home/usernsw/mm_dream_reconstruction/build/'
BASE_DATA = '/mnt/cosmic_data/P2/Run/'

# The machine default ROOTSYS is /workspace/root, which is ROOT 5.34 and is NOT
# ABI-compatible with these binaries (built against 6.30.02). Without this the
# executables die at startup on an undefined TNamed::ShowMembers symbol.
ROOTSYS = '/local/home/usernsw/root_6_30_02/root-build'

DECODE_EXECUTABLE = os.path.join(BASE_SOFT, 'decoder/decode')
WAVEFORM_ANALYSIS_EXECUTABLE = os.path.join(BASE_SOFT, 'waveform_analysis/analyze_waveforms')
COMBINE_HITS_EXECUTABLE = os.path.join(BASE_SOFT, 'feu_hit_combiner/combine_feus_hits')

RAW_DREAM_DIR_NAME = 'raw_daq_data'
DECODED_ROOT_DIR_NAME = 'decoded_root'
HITS_DIR_NAME = 'hits_root'
COMBINED_HITS_DIR_NAME = 'combined_hits_root'

# Dream shaping time -> matched-filter gate width (same constants as process_run.py)
DREAM_CFG_SEARCH_DIRS = [
    '/mnt/cosmic_data/P2/dream_config/',
    '/local/home/usernsw/Cosmic_Bench_DAQ_Control/dream_config/',
]
DREAM_PEAKING_NS = {0: 76, 1: 123, 2: 180, 3: 228, 4: 283, 5: 328, 6: 388, 7: 433, 8: 578}
MF_WIDTH_OVER_PEAKING = 1.7

# FEU 01 is the M3 tracker, read out by the separate cosmic_bench_m3_tracking
# repo (its gen_joblist.sh hardcodes FEU=01) into m3_tracking_root. It has never
# been part of the dream chain: across P2/Run all 145 FEU-01 datrun files have
# zero decoded/hits products, while FEU 03/04/06/07 are at 100% coverage.
# Including it would fold tracker data into the detector hits.
EXCLUDE_FEUS = {1}

LOG_DIR = '/local/home/usernsw/reprocess_p2_logs'
MANIFEST = os.path.join(LOG_DIR, 'manifest.json')
RUN_LOG = os.path.join(LOG_DIR, 'run.log')

MIN_FREE_GB = 25          # halt before the data volume fills
MIN_OUTPUT_BYTES = 10240  # a product smaller than this is treated as failed

_manifest_lock = threading.Lock()
_log_lock = threading.Lock()
_manifest = {}
_stop = threading.Event()


# =========================
# LOGGING / MANIFEST
# =========================

def log(msg):
    line = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    with _log_lock:
        print(line, flush=True)
        try:
            with open(RUN_LOG, 'a') as f:
                f.write(line + '\n')
        except OSError:
            pass


def load_manifest():
    global _manifest
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST) as f:
                _manifest = json.load(f)
        except (OSError, ValueError):
            log('WARNING: manifest unreadable, starting fresh')
            _manifest = {}
    else:
        _manifest = {}


def save_manifest():
    """Atomic write -- a crash mid-save must not corrupt the resume state."""
    with _manifest_lock:
        snapshot = dict(_manifest)
    tmp = MANIFEST + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(snapshot, f, indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


def mark(key, status, **kw):
    entry = {'status': status, 'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
    entry.update(kw)
    with _manifest_lock:
        _manifest[key] = entry


def done(key):
    with _manifest_lock:
        return _manifest.get(key, {}).get('status') == 'ok'


# =========================
# EXECUTION
# =========================

def child_env():
    env = dict(os.environ)
    env['ROOTSYS'] = ROOTSYS
    env['LD_LIBRARY_PATH'] = os.path.join(ROOTSYS, 'lib') + ':/usr/lib64'
    return env


def run_step(key, cmd, out_path, log_name):
    """Run one executable. Returns True on success.

    Success requires rc == 0 AND an output file that is plausibly non-empty:
    the decoder in particular can exit 0 having written a stub.
    """
    if done(key):
        return True
    if _stop.is_set():
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'steps', log_name + '.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=child_env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True)
        rc = proc.returncode
        output = proc.stdout or ''
    except OSError as e:
        rc, output = -1, 'exec failed: {}'.format(e)
    dt = time.time() - t0

    try:
        with open(log_path, 'w') as f:
            f.write('CMD: {}\n\n'.format(' '.join(cmd)))
            f.write(output)
            f.write('\n\nRC={} WALL={:.1f}s\n'.format(rc, dt))
    except OSError:
        pass

    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    if rc != 0:
        mark(key, 'failed', rc=rc, reason='nonzero exit', log=log_path, secs=round(dt, 1))
        log('FAIL rc={} {} (log: {})'.format(rc, os.path.basename(out_path), log_path))
        return False
    if size < MIN_OUTPUT_BYTES:
        mark(key, 'failed', rc=rc, reason='output {} bytes'.format(size),
             log=log_path, secs=round(dt, 1))
        log('FAIL empty output {} ({} bytes)'.format(os.path.basename(out_path), size))
        return False

    mark(key, 'ok', rc=0, bytes=size, secs=round(dt, 1))
    return True


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024.0 ** 3)


# =========================
# NAME / CONFIG HELPERS  (ported from process_run.py)
# =========================

def extract_file_numbers_tuple(filename, end_dot=True):
    pat = r'.*_(\d{3})_(\d{2})\..*' if end_dot else r'.*_(\d{3})_(\d{2}).*'
    m = re.match(pat, filename)
    return (int(m.group(1)), int(m.group(2))) if m else None


def replace_feu_number_in_filename(filename, repl):
    m = re.match(r'(.*_(\d{3})_)(\d{2})(.*)', filename)
    return '{}{}{}'.format(m.group(1), repl, m.group(4)) if m else filename


def parse_dream_peaking(cfg_path):
    peaking = {}
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


def find_dream_cfg(raw_dir, run_dir):
    """DAQ's exact .cfg_cpy next to the raw data if present, else the template
    named in run_config.json (absolute path first, then by basename)."""
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
                if os.path.exists(tmpl):
                    return tmpl
                for d in DREAM_CFG_SEARCH_DIRS:
                    p = os.path.join(d, os.path.basename(tmpl))
                    if os.path.exists(p):
                        return p
        except (OSError, ValueError):
            pass
    return None


def run_daq_info(run_dir):
    rc_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(rc_path):
        try:
            with open(rc_path) as f:
                return json.load(f).get('dream_daq_info', {})
        except (OSError, ValueError):
            pass
    return {}


# =========================
# PIPELINE
# =========================

def decode_task(fdf_path, root_path, tag):
    key = 'decode:' + root_path
    return run_step(key, [DECODE_EXECUTABLE, fdf_path, root_path], root_path, tag)


def analyze_task(root_path, hits_path, ped_path, tps, peaking, zs, tag):
    key = 'analyze:' + hits_path
    if done(key):
        return True
    nums = extract_file_numbers_tuple(os.path.basename(root_path))
    feu_num = nums[1] if nums else None

    cmd = [WAVEFORM_ANALYSIS_EXECUTABLE, root_path, hits_path, ped_path or '']
    if tps:
        cmd += ['--tps', '{:g}'.format(tps)]
    if peaking and tps:
        peak_ns = peaking.get(str(feu_num), peaking.get('*'))
        if peak_ns:
            mf = max(3, int(round(MF_WIDTH_OVER_PEAKING * peak_ns / tps)))
            cmd += ['--mf', str(mf)]
    if zs:
        cmd += ['--zs-baseline', '1']
    return run_step(key, cmd, hits_path, tag)


def combine_task(feu_map, out_dir, tag):
    first = os.path.basename(sorted(feu_map.values())[0])
    out_path = os.path.join(out_dir, replace_feu_number_in_filename(first, 'feu-combined'))
    key = 'combine:' + out_path
    if done(key):
        return True
    os.makedirs(out_dir, exist_ok=True)
    fd, listfile = tempfile.mkstemp(suffix='.txt')
    try:
        with os.fdopen(fd, 'w') as f:
            for feu, path in sorted(feu_map.items()):
                f.write('{} {}\n'.format(path, feu))
        return run_step(key, [COMBINE_HITS_EXECUTABLE, listfile, out_path], out_path, tag)
    finally:
        os.unlink(listfile)


def process_subrun(run_dir, sub, workers, keep_decoded, decoded_base):
    sub_dir = os.path.join(run_dir, sub)
    raw_dir = os.path.join(sub_dir, RAW_DREAM_DIR_NAME)
    if not os.path.isdir(raw_dir):
        return

    if decoded_base:
        rel = os.path.relpath(sub_dir, BASE_DATA)
        decoded_dir = os.path.join(decoded_base, rel, DECODED_ROOT_DIR_NAME)
    else:
        decoded_dir = os.path.join(sub_dir, DECODED_ROOT_DIR_NAME)
    hits_dir = os.path.join(sub_dir, HITS_DIR_NAME)
    combined_dir = os.path.join(sub_dir, COMBINED_HITS_DIR_NAME)

    label = '{}/{}'.format(os.path.basename(run_dir), sub)
    def wanted(fname):
        nums = extract_file_numbers_tuple(fname)
        return bool(nums) and nums[1] not in EXCLUDE_FEUS

    files = os.listdir(raw_dir)
    data_fdfs = sorted(f for f in files
                       if '_datrun_' in f and f.endswith('.fdf') and wanted(f))
    ped_fdfs = sorted(f for f in files
                      if '_pedthr_' in f and f.endswith('.fdf') and wanted(f))
    if not data_fdfs:
        log('SKIP {} (no datrun fdf)'.format(label))
        return

    avail = free_gb(sub_dir)
    if avail < MIN_FREE_GB:
        log('HALT: only {:.1f} GB free on {} (floor {} GB)'.format(avail, sub_dir, MIN_FREE_GB))
        _stop.set()
        return

    daq = run_daq_info(run_dir)
    tps = daq.get('sample_period')
    tps = float(tps) if tps else None
    zs = bool(daq.get('zero_suppress')) and bool(daq.get('pedestal_subtraction'))
    cfg = find_dream_cfg(raw_dir, run_dir)
    peaking = parse_dream_peaking(cfg) if cfg else {}

    log('=== {} : {} datrun, {} pedthr, tps={} zs={} peaking={}'.format(
        label, len(data_fdfs), len(ped_fdfs), tps, zs, peaking or 'auto'))

    # ---- pedestals first: analyze depends on them ----
    for f in ped_fdfs:
        if _stop.is_set():
            return
        decode_task(os.path.join(raw_dir, f),
                    os.path.join(raw_dir, f.replace('.fdf', '.root')),
                    '{}__ped_{}'.format(label.replace('/', '__'), f[:-4]))

    def ped_root_for(feu_num):
        cands = [f for f in os.listdir(raw_dir)
                 if '_pedthr_' in f and f.endswith('.root')]
        for f in cands:
            nums = extract_file_numbers_tuple(f)
            if nums and nums[1] == feu_num:
                return os.path.join(raw_dir, f)
        return ''

    # ---- decode -> analyze, one chained task per file ----
    def chain(fdf_name):
        if _stop.is_set():
            return
        stem = fdf_name[:-4]
        tag = '{}__{}'.format(label.replace('/', '__'), stem)
        root_path = os.path.join(decoded_dir, stem + '.root')
        hits_path = os.path.join(hits_dir, stem + '_hits.root')
        if not decode_task(os.path.join(raw_dir, fdf_name), root_path, tag + '__decode'):
            return
        nums = extract_file_numbers_tuple(fdf_name)
        ped = ped_root_for(nums[1]) if nums else ''
        if not ped:
            log('WARN no pedestal for {} -- analyzing without'.format(stem))
        ok = analyze_task(root_path, hits_path, ped, tps, peaking, zs, tag + '__analyze')
        if ok and not keep_decoded:
            try:
                os.remove(root_path)
            except OSError:
                pass

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(chain, f) for f in data_fdfs]
        for fu in as_completed(futs):
            fu.result()
    save_manifest()

    if _stop.is_set():
        return

    # ---- combine per file number ----
    if not os.path.isdir(hits_dir):
        return
    groups = {}
    for f in os.listdir(hits_dir):
        nums = extract_file_numbers_tuple(f, end_dot=False)
        if nums and nums[1] not in EXCLUDE_FEUS:
            groups.setdefault(nums[0], {})[nums[1]] = os.path.join(hits_dir, f)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(combine_task, m, combined_dir,
                            '{}__combine_{:03d}'.format(label.replace('/', '__'), n))
                for n, m in sorted(groups.items())]
        for fu in as_completed(futs):
            fu.result()
    save_manifest()


def iter_subruns(base, only=None):
    for run in sorted(os.listdir(base)):
        run_dir = os.path.join(base, run)
        if not os.path.isdir(run_dir):
            continue
        for sub in sorted(os.listdir(run_dir)):
            if not os.path.isdir(os.path.join(run_dir, sub)):
                continue
            if not os.path.isdir(os.path.join(run_dir, sub, RAW_DREAM_DIR_NAME)):
                continue
            if only and only not in sub and only not in run:
                continue
            yield run_dir, sub


# =========================
# MODES
# =========================

def cmd_run(args):
    os.makedirs(os.path.join(LOG_DIR, 'steps'), exist_ok=True)
    load_manifest()
    for exe in (DECODE_EXECUTABLE, WAVEFORM_ANALYSIS_EXECUTABLE, COMBINE_HITS_EXECUTABLE):
        if not os.path.exists(exe):
            sys.exit('missing executable: {}'.format(exe))

    subs = list(iter_subruns(args.base, args.subrun))
    log('START {} subruns, {} workers, keep_decoded={}, decoded_base={}'.format(
        len(subs), args.workers, not args.drop_decoded, args.decoded_base or 'in place'))
    t0 = time.time()
    try:
        for run_dir, sub in subs:
            if _stop.is_set():
                break
            process_subrun(run_dir, sub, args.workers,
                           not args.drop_decoded, args.decoded_base)
    except KeyboardInterrupt:
        _stop.set()
        log('INTERRUPTED -- manifest saved, rerun to resume')
    finally:
        save_manifest()
    log('END after {:.1f} h'.format((time.time() - t0) / 3600.0))
    cmd_status(args)


def cmd_status(args):
    load_manifest()
    counts = {}
    fails = []
    for k, v in _manifest.items():
        stage = k.split(':', 1)[0]
        st = v.get('status')
        counts.setdefault(stage, {}).setdefault(st, 0)
        counts[stage][st] += 1
        if st == 'failed':
            fails.append((k, v))
    print('\n--- progress ---')
    for stage in sorted(counts):
        print('  {:9s} {}'.format(stage, dict(counts[stage])))
    secs = sum(v.get('secs', 0) for v in _manifest.values())
    print('  cpu-time in steps: {:.1f} h'.format(secs / 3600.0))
    if fails:
        print('\n--- {} FAILURES ---'.format(len(fails)))
        for k, v in fails[:40]:
            print('  {}\n      {} (rc={})  {}'.format(
                k, v.get('reason'), v.get('rc'), v.get('log', '')))
        if len(fails) > 40:
            print('  ... {} more'.format(len(fails) - 40))
    else:
        print('\n  no failures recorded')


# Products this chain regenerates. m3_tracking_root is NOT here: it comes from
# the separate cosmic_bench_m3_tracking repo and this pipeline cannot rebuild it.
CLEAN_DIRS = [DECODED_ROOT_DIR_NAME, HITS_DIR_NAME, COMBINED_HITS_DIR_NAME,
              'hits_root_reped', 'combined_hits_root_reped']


def cmd_clean(args):
    total = 0
    targets = []
    for run_dir, sub in iter_subruns(args.base, args.subrun):
        sub_dir = os.path.join(run_dir, sub)
        for d in CLEAN_DIRS:
            p = os.path.join(sub_dir, d)
            if os.path.isdir(p):
                sz = sum(os.path.getsize(os.path.join(p, f))
                         for f in os.listdir(p)
                         if os.path.isfile(os.path.join(p, f)))
                targets.append(('dir', p, sz))
                total += sz
        # decoded pedestal roots live *inside* raw_daq_data next to the .fdf --
        # only ever remove .root here, never the raw data.
        raw = os.path.join(sub_dir, RAW_DREAM_DIR_NAME)
        roots = [f for f in os.listdir(raw) if f.endswith('.root')]
        for f in roots:
            p = os.path.join(raw, f)
            sz = os.path.getsize(p)
            targets.append(('file', p, sz))
            total += sz

    print('{} targets, {:.1f} GB'.format(len(targets), total / 1024.0 ** 3))
    for kind, p, sz in targets[:15]:
        print('  {} {} ({:.1f} MB)'.format(kind, p, sz / 1024.0 ** 2))
    if len(targets) > 15:
        print('  ... {} more'.format(len(targets) - 15))
    if not args.yes:
        print('\nDRY RUN -- pass --yes to delete')
        return
    for kind, p, _ in targets:
        if kind == 'dir':
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass
    print('deleted. free now: {:.1f} GB'.format(free_gb(args.base)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'status', 'clean'])
    ap.add_argument('--base', default=BASE_DATA)
    ap.add_argument('--subrun', default=None, help='substring filter on run/subrun name')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--drop-decoded', action='store_true',
                    help='delete each decoded root once its hits are made')
    ap.add_argument('--decoded-base', default=None,
                    help='write decoded_root under this base instead of in place')
    ap.add_argument('--yes', action='store_true', help='clean: actually delete')
    ap.add_argument('--exclude-feus', default='1',
                    help='comma-separated FEU numbers to skip (default 1, the M3 '
                         'tracker); pass empty string to process every FEU')
    args = ap.parse_args()

    global EXCLUDE_FEUS
    EXCLUDE_FEUS = set(int(x) for x in args.exclude_feus.split(',') if x.strip())
    {'run': cmd_run, 'status': cmd_status, 'clean': cmd_clean}[args.mode](args)


if __name__ == '__main__':
    main()
