"""Equivalence test: Python port (larreco_port.minimize) vs the ACTUAL
compiled larreco Solver.cxx (cpp/spsolver_cpp), on identical triplet
systems. Run after any change to larreco_port.py.

Both backends receive the same triplets from find_triplets; the comparison
isolates the charge solve, which is the numerically subtle component.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp'))
import spsolver_cpp
from geometry import GEOM
from larreco_port import (find_triplets, build_system, minimize,
                          MICROBOONE, STANDARD)


def system_arrays(trips):
    """Index-encode a triplet system for the C++ binding."""
    xyz = np.array([t[3] for t in trips])
    u_ids = sorted({t[1] for t in trips})
    v_ids = sorted({t[2] for t in trips})
    x_ids = sorted({t[0] for t in trips})
    umap = {g: i for i, g in enumerate(u_ids)}
    vmap = {g: i + len(u_ids) for i, g in enumerate(v_ids)}
    xmap = {g: i for i, g in enumerate(x_ids)}
    iw1 = np.array([umap[t[1]] for t in trips], np.int64)
    iw2 = np.array([vmap[t[2]] for t in trips], np.int64)
    cw = np.array([xmap[t[0]] for t in trips], np.int64)
    return xyz, iw1, iw2, cw, u_ids, v_ids, x_ids


def make_event(seed=7, N=80, same_x=True):
    """Ghost-rich toy event (shared drift time => ambiguous triplets)."""
    rng = np.random.default_rng(seed)
    xdrift = np.full(N, 130.0) if same_x else rng.uniform(20, 240, N)
    true = np.column_stack([xdrift, rng.uniform(-100, 100, N),
                            rng.uniform(20, 1020, N)])
    q = rng.gamma(2, 50, N)
    plane, wire, x, integ = [], [], [], []
    for (px, py, pz), c in zip(true, q):
        for pl in range(3):
            plane.append(pl)
            wire.append(int(GEOM[pl].wire_of(py, pz)))
            x.append(px)
            integ.append(c if pl == 2 else 0.8 * c)
    return (np.array(plane), np.array(wire), np.array(x), np.array(integ))


def run_case(cfg, seed, same_x):
    plane, wire, x, integ = make_event(seed, same_x=same_x)
    trips, (xw, xx, xq, uq, vq) = find_triplets(plane, wire, x, integ, cfg)
    if not trips:
        return None
    # --- python port
    cwires, scs, _ = build_system(trips, xq, uq, vq, return_map=True)
    minimize(cwires, 0.0, cfg['max_iter_noreg'])
    minimize(cwires, cfg['alpha'], cfg['max_iter_reg'])
    py_pred = np.array([s.pred for s in scs])
    # --- actual compiled Solver.cxx; build_system orders SCs grouped by
    # X hit (dict of lists) — replicate the same SC order for comparison
    order = sorted(range(len(trips)),
                   key=lambda k: list(dict.fromkeys(t[0] for t in trips)
                                      ).index(trips[k][0]))
    trips_o = [trips[k] for k in order]
    xyz, iw1, iw2, cw, u_ids, v_ids, x_ids = system_arrays(trips_o)
    iwq = np.concatenate([uq[u_ids], vq[v_ids]]).astype(float)
    cwq = xq[x_ids].astype(float)
    cpp_pred = spsolver_cpp.solve_system(
        xyz, iw1, iw2, cw, iwq, cwq, cfg['alpha'],
        cfg['max_iter_noreg'], cfg['max_iter_reg'])
    # un-permute
    cpp_full = np.empty_like(cpp_pred)
    cpp_full[order] = cpp_pred
    d = np.abs(py_pred - cpp_full)
    rel = d.max() / max(py_pred.max(), 1e-9)
    return len(trips), d.max(), rel, py_pred.sum(), cpp_full.sum()


if __name__ == '__main__':
    ok = True
    for name, cfg in (('MICROBOONE', MICROBOONE), ('STANDARD', STANDARD)):
        for seed, same_x in ((7, True), (3, True), (11, False)):
            r = run_case(cfg, seed, same_x)
            if r is None:
                continue
            n, dmax, rel, s_py, s_cpp = r
            status = 'PASS' if rel < 1e-9 else 'FAIL'
            ok &= status == 'PASS'
            print(f"{name:10s} seed={seed} ambiguous={same_x} "
                  f"n_trips={n:4d}  max|dq|={dmax:.3e} "
                  f"(rel {rel:.1e})  sums {s_py:.3f}/{s_cpp:.3f}  {status}")
    print('EQUIVALENCE:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
