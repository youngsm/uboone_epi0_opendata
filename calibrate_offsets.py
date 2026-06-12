"""Calibrate per-plane drift-time offsets from the data itself.

WHY THIS EXISTS: larreco's TripletFinder converts hit time to drift x via
detprop->ConvertTicksToX(time, wireID) — a PLANE-DEPENDENT conversion
(wire planes at x = 0/-0.3/-0.6 cm plus per-plane signal-processing
timing). The OpenSamples HDF5 carries only raw peak ticks and a
plane-agnostic tpcTimeToX. Uncorrected, x_induction - x_collection is
systematically offset; once the offset exceeds the 0.3 cm MICROBOONE
drift tolerance, genuine triplets fail and only accidentals survive
(symptom: ~10 scattered spacepoints where a shower should have hundreds).

METHOD: for each induction plane p, scan a trial offset over +-3.5 cm and
count "valid doublets" — (collection, plane-p) hit pairs that are
drift-coincident within the solver tolerance AND whose wires cross inside
the detector (the solver's own acceptance criteria, so the calibration is
correct by construction for any offset magnitude). The yield curve peaks
at the true offset; refine with the median residual of accepted pairs at
the peak. offsets_cm are relative to the collection plane (offset_2 = 0).

Usage:
  python calibrate_offsets.py --in 'data/nue/*.h5' --max-events 100 \
      --out offsets.json [--plot offsets.png]
Then: make_pilarnet.py --offsets offsets.json
"""
import argparse, glob, json
import numpy as np
import h5py
from geometry import GEOM, tick_to_x, collection_wire_z
from spacepoints import _group_index

TOL = 0.3          # MICROBOONE WireIntersectThresholdDriftDir [cm]
SCAN = np.arange(-3.5, 3.5001, 0.05)


def event_hits(files, max_events):
    """Yield (plane, wire, x_raw) truth-matched hit arrays per event."""
    n = 0
    for path in files:
        with h5py.File(path, 'r') as f:
            h_eid = np.asarray(f['hit_table']['event_id'])
            h_pl, h_w, h_t, h_id = [np.asarray(f['hit_table'][k]).squeeze()
                                    for k in ('local_plane', 'local_wire',
                                              'local_time', 'hit_id')]
            e_eid = np.asarray(f['edep_table']['event_id'])
            e_hid = np.asarray(f['edep_table']['hit_id']).squeeze()
            hi, ei = _group_index(h_eid), _group_index(e_eid)
            for key, idx in hi.items():
                if n >= max_events:
                    return
                if key not in ei:
                    continue
                m = np.isin(h_id[idx], np.unique(e_hid[ei[key]]))
                if m.sum() < 30:
                    continue
                yield h_pl[idx][m], h_w[idx][m], tick_to_x(h_t[idx][m])
                n += 1


def doublet_residuals(pl, w, x, p):
    """For every (collection, plane-p) pair with an in-detector wire
    crossing, the raw drift residual x_p - x_coll. Returned once; yield
    at trial offset d = #(|res - d| < TOL)."""
    cm, pm = pl == 2, pl == p
    if cm.sum() < 5 or pm.sum() < 5:
        return np.zeros(0)
    z = collection_wire_z(w[cm])
    res = []
    xp, wp = x[pm], w[pm]
    for zc, xc in zip(z, x[cm]):
        y = GEOM[p].y_at_z(wp, zc)
        ok = (y > -116.5) & (y < 118.0)
        if ok.any():
            res.append(xp[ok] - xc)
    return np.concatenate(res) if res else np.zeros(0)


def calibrate(files, max_events=100):
    allres = {0: [], 1: []}
    for pl, w, x in event_hits(files, max_events):
        for p in (0, 1):
            r = doublet_residuals(pl, w, x, p)
            allres[p].append(r[np.abs(r) < 4.0])
    off, curves, stats = [0.0, 0.0, 0.0], {}, {}
    for p in (0, 1):
        r = np.concatenate(allres[p]) if allres[p] else np.zeros(0)
        if len(r) < 100:
            raise SystemExit(f'plane {p}: too few pairs ({len(r)})')
        yield_curve = np.array([(np.abs(r - d) < TOL).sum() for d in SCAN])
        peak = SCAN[int(np.argmax(yield_curve))]
        sel = np.abs(r - peak) < TOL
        off[p] = float(np.median(r[sel]))
        curves[p] = (SCAN, yield_curve)
        stats[f'plane{p}'] = dict(offset_cm=off[p],
                                  offset_ticks=off[p] / 0.0548965,
                                  pairs_at_peak=int(sel.sum()))
        print(f"plane {p}: offset = {off[p]:+.3f} cm "
              f"({off[p]/0.0548965:+.1f} ticks, "
              f"{int(sel.sum())} pairs at peak)")
    return off, curves, stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', nargs='+', required=True)
    ap.add_argument('--max-events', type=int, default=100)
    ap.add_argument('--out', default='offsets.json')
    ap.add_argument('--plot', default=None)
    a = ap.parse_args()
    files = sorted(sum([glob.glob(p) for p in a.inp], []))
    off, curves, stats = calibrate(files, a.max_events)
    with open(a.out, 'w') as fh:
        json.dump(dict(offsets_cm=off, reference='plane2', tol_cm=TOL,
                       stats=stats), fh, indent=2)
    print('wrote', a.out)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for p, ax in zip((0, 1), axes):
            d, y = curves[p]
            ax.plot(d, y, lw=1.2)
            ax.axvline(off[p], color='r', ls='--',
                       label=f'offset {off[p]:+.2f} cm')
            ax.set_xlabel(f'trial offset, plane {p} [cm]')
            ax.set_ylabel('valid doublets')
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print('wrote', a.plot)
