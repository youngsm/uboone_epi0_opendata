"""Build a PILArNet-style labeled 3D point-cloud dataset reproducing the
inputs of arXiv:2506.09238 (PRD 113, 072005): truth-EM-filtered
SpacePointSolver spacepoints from the MicroBooNE OpenSamples NoWire files,
for electron (nue sample) and pi0 (inclusive sample) events.

Stored per event:
  points            vlen float32, reshape (-1, 4): x, y, z, value
                    (value = solved spacepoint charge, ADC — the same
                    collection-plane-derived intensity the paper uses)
  semantic          vlen int8 — all 0 (shower) by construction; kept for
                    format compatibility
  instance          vlen int8 — which EM shower the point belongs to:
                    0 = the electron (electron events)
                    0 / 1 = leading / subleading photon (pi0 events)
                    -1 = ambiguous/unmatched
  event_id          (N,3) run/subrun/event
  label             (N,) 0 = electron, 1 = pi0   (event-level)
  true_energy       (N,) GeV, of the target particle (paper's binning var)
  n_boundary        (N,) deposits within 5 cm of the detector boundary
                    (store, don't cut: the paper's filter is <= 10)

Deliberately NOT applied (OT preprocessing, not sample definition):
1 cm downsampling, 2.5 cm clustering, WPCA alignment, COM centering,
energy binning, class balancing. Apply those at analysis time; the stored
fields make the paper's selection exactly reproducible:
  mask = (n_boundary <= 10) & (true_energy in bin) & balanced sampling.

Usage:
  python make_pilarnet.py --electron-files 'data/nue/*.h5' \
      --pi0-files 'data/bnb/*.h5' --out pilarnet_epi0.h5 [--max-events N]
"""
import argparse, glob, os, zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import h5py
from geometry import hit_x, load_plane_offsets
from spacepoints import (find_target, descendants,
                         em_hit_ids, PDG_GAMMA)
from preprocess import near_boundary_count
from larreco_port import solve_spacepoints_larreco


def _instance_map(mode, target, g4_id, parent_id, g4_pdg, momentum):
    """g4_id -> instance for the target's shower(s).
    electron: everything under the primary electron -> 0.
    pi0: descendants of the leading photon -> 0, subleading -> 1."""
    if mode == 'electron':
        return {g: 0 for g in descendants(target, g4_id, parent_id)}
    # pi0: find direct photon daughters of the target
    dau = [(int(g), float(mm)) for g, p, pdg, mm
           in zip(g4_id, parent_id, g4_pdg, momentum)
           if int(p) == target and int(pdg) == PDG_GAMMA]
    dau.sort(key=lambda t: -t[1])
    out = {}
    for inst, (g, _) in enumerate(dau[:2]):
        for d in descendants(g, g4_id, parent_id):
            out[d] = inst
    return out


def _fast_read(ds, threads=8):
    """Read a gzip-chunked column ~8x faster than np.asarray(ds).

    h5py decompresses chunks serially under the HDF5 global lock; here the
    raw compressed chunks are fetched via read_direct_chunk and inflated
    in a thread pool (zlib releases the GIL). Bitwise-identical output;
    falls back to np.asarray for any layout this doesn't cover."""
    if (ds.compression != 'gzip' or ds.shuffle or ds.chunks is None
            or len(ds.shape) != 2 or ds.shape[1] != 1
            or ds.chunks[1] != 1):
        a = np.asarray(ds)
        # match the fast path's output shape for (n,1) column datasets
        return a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a
    n, c0 = ds.shape[0], ds.chunks[0]
    out = np.empty(n, ds.dtype)
    dsid = ds.id

    def inflate(start, comp):
        stop = min(start + c0, n)
        raw = zlib.decompress(comp)
        out[start:stop] = np.frombuffer(raw, ds.dtype)[:stop - start]

    # HDF5 is not threadsafe: raw chunk reads stay on this thread, only
    # the zlib inflation fans out. Bounded queue caps compressed bytes
    # held in flight.
    pending = deque()
    with ThreadPoolExecutor(threads) as tp:
        for i in range(-(-n // c0)):
            start = i * c0
            fm, comp = dsid.read_direct_chunk((start, 0))
            if fm != 0:                   # filter skipped for this chunk
                out[start:min(start + c0, n)] = ds[start:start + c0, 0]
                continue
            pending.append(tp.submit(inflate, start, comp))
            while len(pending) > 4 * threads:
                pending.popleft().result()
        while pending:
            pending.popleft().result()
    return out


def _event_groups(f, table):
    """{(run,subrun,event): row-slice} from the event_id.seq_cnt companion
    dataset — avoids reading the full event_id column (674M x 3 rows in
    128-row gzip chunks for the BNB hit_table: minutes of chunk overhead).
    Replaces _group_index; rows per event are contiguous by construction
    (seq_cnt is (event_table index, row count) in storage order)."""
    if 'event_id.seq_cnt' not in f[table]:
        # stripped/synthetic file: fall back to row-scan grouping
        from spacepoints import _group_index
        return _group_index(np.asarray(f[table]['event_id']))
    sc = np.asarray(f[table]['event_id.seq_cnt'])
    ev = np.asarray(f['event_table']['event_id'])
    assert len(np.unique(sc[:, 0])) == len(sc), f"{table}: non-unique seq"
    starts = np.concatenate([[0], np.cumsum(sc[:, 1])[:-1]])
    return {tuple(ev[int(s)].tolist()): slice(int(a), int(a + c))
            for s, a, c in zip(sc[:, 0], starts, sc[:, 1])}


def _event_payloads(path, mode, min_frac=0.0):
    """Pre-solve part of the pipeline (HDF5 I/O + truth matching), kept in
    the parent process. Yields (meta, hit-arrays) per candidate event,
    where meta = (key, label, p_tgt, mass, vis, inst_coll) and inst_coll
    pre-resolves the instance of each plane-2 hit so workers stay purely
    numeric. Drift x already has plane offsets applied here (hit_x), so
    workers never need geometry.PLANE_OFFSETS_CM."""
    label = 0 if mode == 'electron' else 1
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
        scanned = 0
        ntot = len(pi)
        for key, pidx in pi.items():
            scanned += 1
            if scanned % 200 == 0:
                print(f"    scanned {scanned}/{ntot} events", flush=True)
            tgt = find_target(p_gid[pidx], p_pdg[pidx], p_par[pidx],
                              p_proc[pidx], p_mom[pidx], mode)
            if tgt is None or key not in hi or key not in ei:
                continue
            target, p_tgt, mass = tgt
            keep = descendants(target, p_gid[pidx], p_par[pidx])
            inst_of = _instance_map(mode, target, p_gid[pidx], p_par[pidx],
                                    p_pdg[pidx], p_mom[pidx])
            eidx = ei[key]
            keep_hits = em_hit_ids(keep, e_hid[eidx], e_gid[eidx],
                                   e_fr[eidx], min_frac=min_frac)
            if not keep_hits:
                continue
            # dominant kept-shower g4 per hit (for instance attribution)
            hg, gg, fr = e_hid[eidx], e_gid[eidx], e_fr[eidx]
            m = np.isin(gg, np.fromiter(keep, int))
            best = {}
            for h, g, w in zip(hg[m].tolist(), gg[m].tolist(),
                               fr[m].tolist()):
                if w > best.get(h, (None, -1))[1]:
                    best[h] = (g, w)
            hit_g4 = {h: g for h, (g, _) in best.items()}

            idx = hi[key]
            hid_evt = h_id[idx]
            sel = np.fromiter((h in keep_hits for h in hid_evt.tolist()),
                              bool, count=len(hid_evt))
            pl, w = h_pl[idx][sel], h_w[idx][sel]
            xx, qq = hit_x(h_t[idx][sel], pl), h_q[idx][sel]
            ids = h_id[idx][sel]
            coll_ids = ids[pl == 2]
            # instance per plane-2 hit, in input order (indexed by the
            # solver's cidx after the solve)
            inst_coll = np.array([inst_of.get(hit_g4.get(int(h), -1), -1)
                                  for h in coll_ids], np.int8)
            # visible energy: true edep summed over kept COLLECTION hits
            # only (each deposit contributes to all 3 planes; restricting
            # to plane 2 avoids triple counting). Units as in edep_table
            # 'energy' (undocumented in file-content-hdf5.md; assumed MeV).
            kept_coll = set(int(h) for h in coll_ids)
            mvis = np.isin(gg, np.fromiter(keep, int)) &                    np.isin(hg, np.fromiter(kept_coll, int))
            vis = float(e_en[eidx][mvis].sum())
            meta = (key, label, float(p_tgt), float(mass), vis, inst_coll)
            yield meta, (pl, w, xx, qq)


def _solve_one(args):
    """Worker entry: the expensive triplet-finding + charge solve.
    Top-level so it pickles under the macOS spawn start method."""
    pl, w, xx, qq, backend = args
    return solve_spacepoints_larreco(pl, w, xx, qq, return_hits=True,
                                     backend=backend)


def _pipelined(pool, payloads, backend, depth):
    """Submit payloads to the pool, keeping <= depth futures in flight;
    yield (meta, result) in submission order, so output is identical to
    the serial run."""
    pending = deque()
    for meta, args in payloads:
        pending.append((meta, pool.submit(_solve_one, args + (backend,))))
        while len(pending) >= depth:
            m, fut = pending.popleft()
            yield m, fut.result()
    while pending:
        m, fut = pending.popleft()
        yield m, fut.result()


def events_to_pilarnet(path, mode, max_events=None, min_points=20,
                       backend='cpp', pool=None, depth=8, min_frac=0.0):
    """Yield (event_id, label, p, mass, vis, pts, q, sem, inst,
    n_boundary). With pool (a ProcessPoolExecutor), events are solved in
    parallel; ordering and values match the serial run exactly."""
    payloads = _event_payloads(path, mode, min_frac)
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
        key, label, p_tgt, mass, vis, inst_coll = meta
        inst = inst_coll[cidx]
        sem = np.zeros(len(pts), np.int8)        # all shower
        n += 1
        yield (key, label, p_tgt, mass, vis,
               pts.astype(np.float32), q.astype(np.float32), sem, inst,
               near_boundary_count(pts))


def convert(e_files, p_files, out_path, max_events=None, min_points=20,
            backend='cpp', workers=None, min_frac=0.0):
    if workers is None:
        workers = os.cpu_count() or 1
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        for files, mode in ((e_files, 'electron'), (p_files, 'pi0')):
            n = 0
            print(f"[{mode}] processing {len(files)} file(s), "
                  f"{workers} worker(s)"
                  + (f", target {max_events} events" if max_events else ""))
            for fi, path in enumerate(files, 1):
                n0 = n
                for r in events_to_pilarnet(
                        path, mode,
                        None if max_events is None else max_events - n,
                        min_points, backend, pool, depth=4 * workers,
                        min_frac=min_frac):
                    rows.append(r)
                    n += 1
                    if n % 50 == 0:
                        print(f"  [{mode}] {n} events kept "
                              f"(file {fi}/{len(files)})", flush=True)
                print(f"  [{mode}] file {fi}/{len(files)} done: "
                      f"+{n - n0} events ({n} total) — {path}", flush=True)
                if max_events is not None and n >= max_events:
                    break
            print(f"[{mode}] {n} events")
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    print(f"writing {len(rows)} events to {out_path} ...", flush=True)
    with h5py.File(out_path, 'w') as f:
        vf = h5py.vlen_dtype(np.float32)
        vi = h5py.vlen_dtype(np.int8)
        f.create_dataset('points', dtype=vf, data=np.array(
            [np.column_stack([r[5], r[6][:, None]]).ravel()
             for r in rows], object))
        f.create_dataset('semantic', dtype=vi,
                         data=np.array([r[7] for r in rows], object))
        f.create_dataset('instance', dtype=vi,
                         data=np.array([r[8] for r in rows], object))
        f.create_dataset('event_id',
                         data=np.array([r[0] for r in rows], np.int64))
        f.create_dataset('label',
                         data=np.array([r[1] for r in rows], np.int8))
        p = np.array([r[2] for r in rows], np.float64)
        mass = np.array([r[3] for r in rows], np.float64)
        etot = np.sqrt(p ** 2 + mass ** 2)
        f.create_dataset('true_momentum', data=p.astype(np.float32))
        f.create_dataset('true_ke', data=(etot - mass).astype(np.float32))
        f.create_dataset('true_energy_total', data=etot.astype(np.float32))
        f.create_dataset('visible_energy',
                         data=np.array([r[4] for r in rows], np.float32))
        # back-compat alias = the paper-matching default (momentum):
        f.create_dataset('true_energy', data=p.astype(np.float32))
        f.attrs['energy_convention'] = (
            'true_energy == true_momentum (GeV). Paper binning variable '
            'cannot be total energy (pi0 floor at 0.135 GeV vs paper Fig.2 '
            'reaching ~0 and a populated 0.05-0.1 bin); momentum vs KE vs '
            'visible is unconfirmed — all stored, re-bin freely.')
        f.create_dataset('n_boundary',
                         data=np.array([r[9] for r in rows], np.int32))
        f.attrs['points_columns'] = 'x,y,z,value_adc'
        f.attrs['semantic_classes'] = '0:shower'
        f.attrs['instance_classes'] = \
            'electron events: 0=e shower; pi0 events: 0=leading gamma, ' \
            '1=subleading gamma; -1=unmatched'
        f.attrs['paper_selection'] = \
            'n_boundary<=10; energy bins ' \
            '[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.9] GeV; ' \
            'balanced N_e=N_pi per bin'
    return len(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--electron-files', nargs='+', default=[],
                    help='nue source files; pass none to skip electrons '
                         '(useful for per-file HPC sharding)')
    ap.add_argument('--pi0-files', nargs='+', default=[],
                    help='bnb inclusive source files; pass none to skip pi0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--min-points', type=int, default=20)
    ap.add_argument('--offsets', default=None,
                    help='offsets.json from calibrate_offsets.py '
                         '(strongly recommended for real data)')
    ap.add_argument('--backend', choices=('cpp', 'python'), default='cpp',
                    help='charge solver: cpp (compiled, ~exact, fast) or '
                         'python (reference). build cpp via cpp/build.sh')
    ap.add_argument('--min-frac', type=float, default=0.0,
                    help='min truth-energy fraction for a hit to count as '
                         'EM (0 = any association, paper reading; 0.5 = '
                         'dominance, plane-fragile in overlay data)')
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel solver processes (default: all cores; '
                         '1 = serial)')
    a = ap.parse_args()
    if a.offsets:
        off = load_plane_offsets(a.offsets)
        print('plane offsets [cm]:', off)
    ef = sorted(sum([glob.glob(p) for p in a.electron_files], []))
    pf = sorted(sum([glob.glob(p) for p in a.pi0_files], []))
    n = convert(ef, pf, a.out, a.max_events, a.min_points, a.backend,
                a.workers, a.min_frac)
    print(f"wrote {n} events to {a.out}")
