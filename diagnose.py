"""Stage-by-stage diagnostic for sparse spacepoint output. Prints, per
event: raw hits per plane -> truth-EM hits per plane -> drift-coincident
doublet yields at zero offset vs calibrated offsets -> triplets ->
solved spacepoints. Localizes the failing stage in one run.

Usage:
  python diagnose.py --in data/nue/FILE.h5 --mode electron \
      [--offsets offsets.json] [--n 5]
"""
import argparse
import numpy as np
import h5py
import geometry
from geometry import tick_to_x, hit_x, collection_wire_z, GEOM
from spacepoints import (_group_index, find_target, descendants,
                         em_hit_ids)
from larreco_port import find_triplets, MICROBOONE, solve_spacepoints_larreco


def doublet_yield(pl, w, x, p, tol=0.3):
    cm, pm = pl == 2, pl == p
    cnt = 0
    z = collection_wire_z(w[cm])
    for zc, xc in zip(z, x[cm]):
        y = GEOM[p].y_at_z(w[pm], zc)
        ok = (y > -116.5) & (y < 118.0) & (np.abs(x[pm] - xc) < tol)
        cnt += int(ok.sum())
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--mode', choices=['electron', 'pi0'],
                    default='electron')
    ap.add_argument('--offsets', default=None)
    ap.add_argument('--n', type=int, default=5)
    a = ap.parse_args()
    if a.offsets:
        off = geometry.load_plane_offsets(a.offsets)
        print('loaded offsets [cm]:', np.round(off, 3))
    else:
        print('NO offsets loaded (geometry.PLANE_OFFSETS_CM = 0)')

    with h5py.File(a.inp, 'r') as f:
        h_eid = np.asarray(f['hit_table']['event_id'])
        h_pl, h_w, h_t, h_q, h_id = [
            np.asarray(f['hit_table'][k]).squeeze()
            for k in ('local_plane', 'local_wire', 'local_time',
                      'integral', 'hit_id')]
        e_eid = np.asarray(f['edep_table']['event_id'])
        e_hid, e_gid, e_fr = [
            np.asarray(f['edep_table'][k]).squeeze()
            for k in ('hit_id', 'g4_id', 'energy_fraction')]
        p_eid = np.asarray(f['particle_table']['event_id'])
        p_gid, p_pdg, p_par, p_mom = [
            np.asarray(f['particle_table'][k]).squeeze()
            for k in ('g4_id', 'g4_pdg', 'parent_id', 'momentum')]
        p_proc = np.asarray(f['particle_table']['start_process']).squeeze()
        hi, ei, pi = (_group_index(x) for x in (h_eid, e_eid, p_eid))

        done = 0
        for key, pidx in pi.items():
            if done >= a.n:
                break
            tgt = find_target(p_gid[pidx], p_pdg[pidx], p_par[pidx],
                              p_proc[pidx], p_mom[pidx], a.mode)
            if tgt is None or key not in hi or key not in ei:
                continue
            target, p_tgt, mass = tgt
            keep = descendants(target, p_gid[pidx], p_par[pidx])
            eidx = ei[key]
            keep_hits = em_hit_ids(keep, e_hid[eidx], e_gid[eidx],
                                   e_fr[eidx])
            idx = hi[key]
            print(f"\n== event {tuple(int(v) for v in key)}  "
                  f"target p={p_tgt:.3f} GeV  "
                  f"|shower particles|={len(keep)}")
            pl_all = h_pl[idx]
            print("  raw hits/plane:        ",
                  [int((pl_all == p).sum()) for p in range(3)])
            sel = np.fromiter((h in keep_hits
                               for h in h_id[idx].tolist()), bool,
                              count=len(idx))
            pl, w = pl_all[sel], h_w[idx][sel]
            print("  truth-EM hits/plane:   ",
                  [int((pl == p).sum()) for p in range(3)])
            if sel.sum() < 10:
                print("  -> LOSS AT TRUTH FILTER (check edep coverage, "
                      "dominance threshold, target selection)")
                done += 1
                continue
            x0 = tick_to_x(h_t[idx][sel])                # no offsets
            xc = hit_x(h_t[idx][sel], pl)                # with offsets
            for name, xx in (('zero-offset', x0), ('calibrated', xc)):
                dU = doublet_yield(pl, w, xx, 0)
                dV = doublet_yield(pl, w, xx, 1)
                print(f"  doublets ({name:11s}): XU={dU:5d}  XV={dV:5d}")
            trips, _ = find_triplets(pl, w, xc, h_q[idx][sel], MICROBOONE)
            pts, q = solve_spacepoints_larreco(pl, w, xc, h_q[idx][sel])
            print(f"  triplets={len(trips)}  spacepoints(q>0)={len(pts)}")
            if len(trips) == 0 and doublet_yield(pl, w, xc, 0) > 0:
                print("  -> doublets exist but no triplets: check U-V "
                      "drift coincidence / y-agreement (tol_y)")
            done += 1


if __name__ == '__main__':
    main()
