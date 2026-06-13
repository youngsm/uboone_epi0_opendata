"""Full-event PILArNet-style labeled 3D point-cloud dataset.

Where make_pilarnet.py truth-filters to ONE target shower per event (e/pi0)
and labels every point identically, this script keeps ALL truth-associated
hits, runs the spacepoint solver on them, and labels each spacepoint with
its dominant truth particle's topology class and per-particle instance.

Important: OpenSamples NoWire reco hits include cosmic overlay; edep_table
covers only the simulated neutrino interaction. We therefore filter to
"any-edep-association" hits before solving, which yields a clean
multi-class point cloud of the neutrino interaction only (no cosmics).
For a cosmic-inclusive dataset, OpenSamples would need to ship cosmic
truth as well.

Semantic classes (Drielsma et al. lartpc_mlreco3d / SLAC ML convention):
  0 HIP    protons, nuclei         (heavily-ionizing)
  1 MIP    mu+/-, pi+/-, K+/-      (minimum-ionizing tracks)
  2 shower e+/-, gamma             (EM showers, not Michel/delta)
  3 delta  e+/- from ionization    (muIoni/hIoni/eIoni)
  4 Michel e+/- from mu->e nu nu   (start_process == 'Decay', parent mu)

Instance grouping: EM-shower particles collapse to their topmost e+-/gamma
ancestor (so a gamma->e+e-gamma cascade is ONE instance under the gamma).
Tracks, deltas, and Michel electrons keep their own instance id, distinct
from their parents.

Stored per event:
  points       vlen float32, reshape (-1, 4): x, y, z, value_adc
  semantic     vlen int8  in [0..4]
  instance     vlen int16 (full events can have hundreds of particles)
  inst_pdg     vlen int32 per-instance PDG
  inst_mom     vlen float32 per-instance momentum (GeV)
  inst_sem     vlen int8 per-instance semantic class
  event_id     (N,3) run/subrun/event
  n_boundary   (N,) deposits within 5 cm of TPC boundary
  n_points     (N,) total spacepoints
  n_instances  (N,) number of distinct particle instances

Usage:
  python make_pilarnet_full.py --files 'data/nue/*.h5' 'data/bnb/*.h5' \\
      --out pilarnet_full.h5 [--max-events N] [--workers K]
"""
import argparse, glob, os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py

from geometry import hit_x, load_plane_offsets
from preprocess import near_boundary_count
from make_pilarnet import _fast_read, _event_groups, _pipelined, _solve_one


# ------------------------------------------------------- truth classification

HIP, MIP, SHOWER, DELTA, MICHEL = 0, 1, 2, 3, 4

MIP_PDG = {13, 211, 321}        # mu+-, pi+-, K+-
HIP_PDG = {2212, 2112}          # p, n (n deposits via recoil-p ionization)
DELTA_PROC = {'muIoni', 'hIoni', 'eIoni'}
# Free mu+/- decay -> Michel; stopped mu- nuclear capture also emits a
# Michel-like e- in ~7% of captures (Drielsma's lartpc_mlreco3d convention).
MICHEL_PROC = {'Decay', 'muMinusCaptureAtRest'}
PDG_E, PDG_GAMMA = 11, 22


def _decode_str(arr):
    if arr.dtype.kind in 'SO':
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in arr.tolist()]
    return [str(x) for x in arr.tolist()]


def classify_particles(g4_id, pdg, parent_id, start_process):
    """Return {g4_id: semantic_class} for every particle in this event."""
    proc = _decode_str(start_process)
    pdg_of = {int(g): int(p) for g, p in zip(g4_id, pdg)}
    par_of = {int(g): int(p) for g, p in zip(g4_id, parent_id)}
    out = {}
    for g, pd, pr in zip(g4_id.tolist(), pdg.tolist(), proc):
        g = int(g); pd = int(pd)
        apd = abs(pd)
        if pd in HIP_PDG or pd >= 1_000_000_000:        # nuclei
            out[g] = HIP
        elif apd in MIP_PDG:
            out[g] = MIP
        elif apd == PDG_E:
            ppid = par_of.get(g, -1)
            ppdg = abs(pdg_of.get(ppid, 0))
            if pr in MICHEL_PROC and ppdg == 13:
                out[g] = MICHEL
            elif pr in DELTA_PROC:
                out[g] = DELTA
            else:
                out[g] = SHOWER
        elif pd == PDG_GAMMA:
            out[g] = SHOWER
        else:
            # Conservative default: hadronic / unknown deposits group with
            # HIP, not shower. (Neutral hadrons leave hits via charged
            # recoils — calling them "shower" would skew the class balance
            # toward shower for every neutrino-hadronic event.)
            out[g] = HIP
    return out


def build_instance_map(g4_id, pdg, parent_id, sem_of):
    """g4_id -> instance id. EM-shower particles collapse to their topmost
    e+-/gamma ancestor; tracks/delta/Michel keep their own instance."""
    pdg_of = {int(g): int(p) for g, p in zip(g4_id, pdg)}
    par_of = {int(g): int(p) for g, p in zip(g4_id, parent_id)}
    head = {}
    for g in g4_id.tolist():
        g = int(g)
        cur = g
        if sem_of.get(cur) != SHOWER:
            head[g] = cur
            continue
        # walk up while parent is also a SHOWER e+-/gamma
        while True:
            pid = par_of.get(cur, -1)
            if pid not in pdg_of:
                break
            if abs(pdg_of[pid]) not in (PDG_E, PDG_GAMMA):
                break
            if sem_of.get(pid) != SHOWER:
                break
            cur = pid
        head[g] = cur
    uniq = {h: i for i, h in enumerate(sorted(set(head.values())))}
    return {g: uniq[h] for g, h in head.items()}, uniq


# -------------------------------------------------------------- per-event I/O

def _event_payloads(path):
    """Yield (meta, hit-arrays) per event, with full (unfiltered) hits.
    meta = (key, hit_g4, sem_of, inst_of, inst_g4_by_inst, p_pdg_by_g4,
            p_mom_by_g4, vis_total)."""
    with h5py.File(path, 'r') as f:
        h_pl, h_w, h_t, h_q, h_id = [
            _fast_read(f['hit_table'][k])
            for k in ('local_plane', 'local_wire', 'local_time',
                      'integral', 'hit_id')]
        e_hid, e_gid, e_fr, e_en = [
            _fast_read(f['edep_table'][k])
            for k in ('hit_id', 'g4_id', 'energy_fraction', 'energy')]
        p_gid, p_pdg, p_par, p_mom = [
            _fast_read(f['particle_table'][k])
            for k in ('g4_id', 'g4_pdg', 'parent_id', 'momentum')]
        p_proc = _fast_read(f['particle_table']['start_process'])

        hi, ei, pi = (_event_groups(f, t)
                      for t in ('hit_table', 'edep_table',
                                'particle_table'))
        ntot, scanned = len(pi), 0
        for key, pidx in pi.items():
            scanned += 1
            if scanned % 200 == 0:
                print(f"    scanned {scanned}/{ntot} events", flush=True)
            if key not in hi or key not in ei:
                continue
            ev_gid = p_gid[pidx]
            sem_of = classify_particles(ev_gid, p_pdg[pidx],
                                        p_par[pidx], p_proc[pidx])
            inst_of, head_map = build_instance_map(
                ev_gid, p_pdg[pidx], p_par[pidx], sem_of)

            # per-instance metadata: PDG/momentum of the *head* particle
            inst_pdg = np.zeros(len(head_map), np.int32)
            inst_mom = np.zeros(len(head_map), np.float32)
            inst_sem = np.zeros(len(head_map), np.int8)
            pdg_arr  = p_pdg[pidx]; mom_arr = p_mom[pidx]
            for h, i_id in head_map.items():
                row = np.flatnonzero(ev_gid == h)
                if len(row):
                    j = int(row[0])
                    inst_pdg[i_id] = int(pdg_arr[j])
                    inst_mom[i_id] = float(mom_arr[j])
                    inst_sem[i_id] = sem_of.get(int(h), SHOWER)

            # dominant g4 per hit (from edep_table, energy-fraction max)
            eidx = ei[key]
            hg, gg, fr = e_hid[eidx], e_gid[eidx], e_fr[eidx]
            best = {}
            for h, g, w in zip(hg.tolist(), gg.tolist(), fr.tolist()):
                if w > best.get(h, (None, -1.0))[1]:
                    best[h] = (g, w)
            hit_g4 = {h: g for h, (g, _) in best.items()}

            # OpenSamples NoWire reco hits include cosmic overlay; edep_table
            # only covers neutrino-truth deposits. Filter to truth-associated
            # hits so the solver runs on the labeled fraction only.
            idx = hi[key]
            hid_evt = h_id[idx]
            keep_hits = set(hit_g4.keys())
            if not keep_hits:
                continue
            sel = np.fromiter(
                (int(h) in keep_hits for h in hid_evt.tolist()),
                bool, count=len(hid_evt))
            if not sel.any():
                continue

            pl, w = h_pl[idx][sel], h_w[idx][sel]
            xx, qq = hit_x(h_t[idx][sel], pl), h_q[idx][sel]
            ids = hid_evt[sel]
            coll_ids_arr = ids[pl == 2]

            # visible energy: sum edep over collection-plane hits only
            coll_set = set(int(h) for h in coll_ids_arr.tolist())
            mvis = np.fromiter(
                (int(h) in coll_set for h in hg.tolist()),
                bool, count=len(hg))
            vis = float(e_en[eidx][mvis].sum())

            meta = (key, hit_g4, sem_of, inst_of,
                    coll_ids_arr, inst_pdg, inst_mom, inst_sem, vis)
            yield meta, (pl, w, xx, qq)


# ------------------------------------------------------------------ converter

def events_to_pilarnet_full(path, max_events=None, min_points=20,
                            backend='cpp', pool=None, depth=8):
    payloads = _event_payloads(path)
    if pool is None:
        results = ((meta, _solve_one(args + (backend,)))
                   for meta, args in payloads)
    else:
        results = _pipelined(pool, payloads, backend, depth)
    n = 0
    for meta, (pts, q, cidx) in results:
        if max_events is not None and n >= max_events:
            break
        if len(pts) < min_points:
            continue
        (key, hit_g4, sem_of, inst_of, coll_ids,
         inst_pdg, inst_mom, inst_sem, vis) = meta
        # spacepoint -> collection hit_id -> dominant g4 -> sem / instance
        coll_for_sp = coll_ids[cidx]
        sem = np.empty(len(pts), np.int8)
        inst = np.empty(len(pts), np.int16)
        for i, h in enumerate(coll_for_sp.tolist()):
            g = hit_g4.get(int(h), -1)
            sem[i] = sem_of.get(g, SHOWER)
            inst[i] = inst_of.get(g, -1)
        n += 1
        yield (key, pts.astype(np.float32), q.astype(np.float32),
               sem, inst, inst_pdg, inst_mom, inst_sem, vis,
               near_boundary_count(pts))


def convert(files, out_path, max_events=None, min_points=20,
            backend='cpp', workers=None):
    if workers is None:
        workers = os.cpu_count() or 1
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        n = 0
        print(f"processing {len(files)} file(s), {workers} worker(s)"
              + (f", target {max_events} events" if max_events else ""))
        for fi, path in enumerate(files, 1):
            n0 = n
            for r in events_to_pilarnet_full(
                    path,
                    None if max_events is None else max_events - n,
                    min_points, backend, pool, depth=4 * workers):
                rows.append(r)
                n += 1
                if n % 50 == 0:
                    print(f"  {n} events kept (file {fi}/{len(files)})",
                          flush=True)
            print(f"  file {fi}/{len(files)} done: +{n - n0} events "
                  f"({n} total) — {path}", flush=True)
            if max_events is not None and n >= max_events:
                break
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    print(f"writing {len(rows)} events to {out_path} ...", flush=True)
    vf32 = h5py.vlen_dtype(np.float32)
    vi8  = h5py.vlen_dtype(np.int8)
    vi16 = h5py.vlen_dtype(np.int16)
    vi32 = h5py.vlen_dtype(np.int32)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('points', dtype=vf32, data=np.array(
            [np.column_stack([r[1], r[2][:, None]]).ravel()
             for r in rows], object))
        f.create_dataset('semantic',   dtype=vi8,  data=np.array(
            [r[3] for r in rows], object))
        f.create_dataset('instance',   dtype=vi16, data=np.array(
            [r[4] for r in rows], object))
        f.create_dataset('inst_pdg',   dtype=vi32, data=np.array(
            [r[5] for r in rows], object))
        f.create_dataset('inst_mom',   dtype=vf32, data=np.array(
            [r[6] for r in rows], object))
        f.create_dataset('inst_sem',   dtype=vi8,  data=np.array(
            [r[7] for r in rows], object))
        f.create_dataset('event_id',
                         data=np.array([r[0] for r in rows], np.int64))
        f.create_dataset('visible_energy',
                         data=np.array([r[8] for r in rows], np.float32))
        f.create_dataset('n_boundary',
                         data=np.array([r[9] for r in rows], np.int32))
        f.create_dataset('n_points',
                         data=np.array([len(r[3]) for r in rows], np.int32))
        f.create_dataset('n_instances',
                         data=np.array([len(r[5]) for r in rows], np.int32))
        f.attrs['points_columns'] = 'x,y,z,value_adc'
        f.attrs['semantic_classes'] = (
            '0:HIP (protons, nuclei); 1:MIP (mu, pi+-, K+-); '
            '2:shower (e+-, gamma); 3:delta (eIoni/hIoni/muIoni e); '
            '4:Michel (e from mu->e nu nu)')
        f.attrs['instance_grouping'] = (
            'EM-shower particles collapse to topmost e+-/gamma ancestor; '
            'tracks, deltas, and Michels keep their own instance id.')
        f.attrs['per_instance_tables'] = (
            'inst_pdg, inst_mom (GeV), inst_sem indexed by instance id; '
            'length = n_instances per event.')
    return len(rows)


# ------------------------------------------------------------------- CLI

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True,
                    help='input NoWire HDF5 files / globs '
                         '(any mix of nue and bnb)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--min-points', type=int, default=20)
    ap.add_argument('--offsets', default=None,
                    help='offsets.json from calibrate_offsets.py')
    ap.add_argument('--backend', choices=('cpp', 'python'), default='cpp')
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    if a.offsets:
        off = load_plane_offsets(a.offsets)
        print('plane offsets [cm]:', off)
    files = sorted(sum([glob.glob(p) for p in a.files], []))
    if not files:
        raise SystemExit('no files matched')
    n = convert(files, a.out, a.max_events, a.min_points, a.backend,
                a.workers)
    print(f'kept {n} events -> {a.out}')
