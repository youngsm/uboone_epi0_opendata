"""Truth-filtered 3D spacepoint reconstruction from MicroBooNE OpenSamples
NoWire HDF5 files, following Caratelli, Craig, Fang & Howard,
arXiv:2506.09238 (PRD 113, 072005), Sec. VI.B:

  "only hits associated with true electron or photon energy deposits are
   used for 3D reconstruction via the SpacePointSolver algorithm. The
   output ... are spacepoints with position vectors in 3D space, each
   associated with two or three hits from different wire planes. The
   intensities of these spacepoints are determined using the charge
   deposits of their corresponding hits in the collection plane."

Default solver (method='larreco') is the faithful port in
larreco_port.py — see its header for exact source provenance (repo,
commit, per-function line permalinks) and the configuration ambiguity.
method='fast' keeps the original heuristic proxy (one spacepoint per
collection hit, raw integral charge, hand-chosen tolerances): NOT from
larreco, retained only as a ~5x faster sanity-check mode.

Truth selection (find_target / descendants / em_hit_ids below) implements
the paper's Sec. VI.B prose; the >50% energy-dominance threshold is this
package's reading, not a documented value — confirm with the authors.
HDF5 field semantics follow
https://github.com/uboone/OpenSamples/blob/main/file-content-hdf5.md.
"""
from dataclasses import dataclass
import numpy as np
import h5py
from geometry import GEOM, tick_to_x, hit_x, collection_wire_z

PDG_E, PDG_PI0, PDG_GAMMA = 11, 111, 22


@dataclass
class Event:
    event_id: tuple          # (run, subrun, event)
    label: int               # 0 = electron, 1 = pi0
    true_energy: float       # GeV, of the target particle
    points: np.ndarray       # (N,3) x,y,z cm
    charge: np.ndarray       # (N,) collection-plane integral (ADC)


# ---------------------------------------------------------------- I/O helpers

def _read(g, *names):
    return [np.asarray(g[n]) for n in names]


def _group_index(event_ids):
    """Map (run,subrun,event) row blocks -> slices, assuming rows for one
    event are contiguous (true for these files; falls back to sorting)."""
    eid = np.ascontiguousarray(event_ids).view(
        [(f'f{k}', event_ids.dtype) for k in range(event_ids.shape[1])]).ravel()
    order = None
    # contiguity check
    change = np.nonzero(eid[1:] != eid[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    keys = eid[starts]
    if len(np.unique(keys)) != len(keys):       # not contiguous -> sort
        order = np.argsort(eid, kind='stable')
        eid = eid[order]
        change = np.nonzero(eid[1:] != eid[:-1])[0] + 1
        starts = np.concatenate([[0], change])
        keys = eid[starts]
    ends = np.concatenate([starts[1:], [len(eid)]])
    out = {}
    for k, s, e in zip(keys, starts, ends):
        idx = np.arange(s, e) if order is None else order[s:e]
        out[tuple(np.asarray(k.tolist()))] = idx
    return out


# ------------------------------------------------------- truth / target logic

def _decode(arr):
    if arr.dtype.kind in 'SO':
        return np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in arr])
    return arr.astype(str)


def find_target(g4_id, g4_pdg, parent_id, start_process, momentum, mode):
    """Return (target_g4_id, true_energy_GeV) or None.

    mode='electron': primary e+/e- from the nu interaction (nue CC sample).
    mode='pi0'     : leading primary pi0 (inclusive sample).
    Primaries are identified by start_process == 'primary' (see
    file-content-hdf5.md), with parent_id == -1 ancestry as fallback.
    """
    proc = _decode(start_process)
    primary = (np.char.lower(proc) == 'primary')
    if not primary.any():
        primary = (parent_id == -1)
    if mode == 'electron':
        m = primary & (np.abs(g4_pdg) == PDG_E)
        mass = 0.000511
    elif mode == 'pi0':
        m = primary & (g4_pdg == PDG_PI0)
        mass = 0.13498
    else:
        raise ValueError(mode)
    if not m.any():
        return None
    i = np.flatnonzero(m)[np.argmax(momentum[m])]   # leading
    # NOTE on energy convention: the paper's binning variable cannot be
    # total energy (its 0.05-0.1 GeV bin contains pi0s, impossible above
    # the 0.135 GeV rest-mass floor; Fig. 2's pi0 spectrum reaches ~0, and
    # Sec. III text quotes 'momentum, peaking at 200 MeV'). Momentum vs
    # kinetic vs deposited energy is not determinable from the paper —
    # confirm with the authors. We return the raw momentum and the mass;
    # callers derive the convention they want.
    return int(g4_id[i]), float(momentum[i]), mass


def descendants(target, g4_id, parent_id):
    """Set of g4 ids in the target's shower (target + all descendants)."""
    children = {}
    for gid, pid in zip(g4_id.tolist(), parent_id.tolist()):
        children.setdefault(pid, []).append(gid)
    keep, stack = set(), [target]
    while stack:
        cur = stack.pop()
        if cur in keep:
            continue
        keep.add(cur)
        stack.extend(children.get(cur, ()))
    return keep


def em_hit_ids(keep, edep_hit_id, edep_g4_id, edep_efrac, min_frac=0.5):
    """Hits whose energy is dominantly (> min_frac) from the kept shower."""
    in_keep = np.isin(edep_g4_id, np.fromiter(keep, int))
    tot = np.zeros(0)
    # accumulate fraction per hit
    hid = edep_hit_id
    n = hid.max() + 1 if len(hid) else 0
    tot_all = np.bincount(hid, weights=edep_efrac, minlength=n)
    tot_keep = np.bincount(hid[in_keep], weights=edep_efrac[in_keep],
                           minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(tot_all > 0, tot_keep / tot_all, 0.0)
    return set(np.flatnonzero(frac > min_frac).tolist())


# --------------------------------------------------------- spacepoint solver

def solve_spacepoints(plane, wire, x, integral, hit_id, keep_hits,
                      tol_x=0.45, tol_y=0.6, allow_doublets=True,
                      method='larreco'):
    """method='larreco' (default): faithful port of larreco v08
    sps::SpacePointSolver with MicroBooNE config (see larreco_port.py) —
    all passing triplets per collection hit, iterative least-squares
    charge solve. method='fast': simple best-match proxy (one spacepoint
    per collection hit, raw collection charge); ~5x faster, positions
    agree at the wire-pitch level."""
    sel = np.fromiter((h in keep_hits for h in hit_id.tolist()), bool,
                      count=len(hit_id))
    if method == 'larreco':
        from larreco_port import solve_spacepoints_larreco
        return solve_spacepoints_larreco(plane[sel], wire[sel], x[sel],
                                         integral[sel])
    pts, q = [], []
    cm = sel & (plane == 2)
    um = sel & (plane == 0)
    vm = sel & (plane == 1)
    if not cm.any():
        return np.zeros((0, 3)), np.zeros(0)
    xu, wu = x[um], wire[um]
    xv, wv = x[vm], wire[vm]
    ou, ov = np.argsort(xu), np.argsort(xv)
    xu, wu = xu[ou], wu[ou]
    xv, wv = xv[ov], wv[ov]
    for xc, wc, qc in zip(x[cm], wire[cm], integral[cm]):
        z = float(collection_wire_z(wc))
        iu = slice(*np.searchsorted(xu, [xc - tol_x, xc + tol_x]))
        iv = slice(*np.searchsorted(xv, [xc - tol_x, xc + tol_x]))
        yu = GEOM[0].y_at_z(wu[iu], z)
        yv = GEOM[1].y_at_z(wv[iv], z)
        okU = (yu > -118) & (yu < 118)
        okV = (yv > -118) & (yv < 118)
        yu, yv = yu[okU], yv[okV]
        if len(yu) and len(yv):                       # triplet
            d = np.abs(yu[:, None] - yv[None, :])
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] < tol_y:
                pts.append((xc, 0.5 * (yu[i] + yv[j]), z))
                q.append(qc)
                continue
        if allow_doublets and (len(yu) or len(yv)):   # doublet fallback
            yy = yu if len(yu) else yv
            xx = xu[iu][okU] if len(yu) else xv[iv][okV]
            k = int(np.argmin(np.abs(xx - xc)))
            pts.append((xc, yy[k], z))
            q.append(qc)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.asarray(pts, float), np.asarray(q, float)


# ------------------------------------------------------------------- driver

def events_from_file(path, mode, max_events=None, min_points=20):
    """Yield Event objects (mode = 'electron' or 'pi0')."""
    label = 0 if mode == 'electron' else 1
    with h5py.File(path, 'r') as f:
        h_eid, = _read(f['hit_table'], 'event_id')
        h_pl, h_w, h_t, h_q, h_id = [
            np.asarray(f['hit_table'][k]).squeeze()
            for k in ('local_plane', 'local_wire', 'local_time',
                      'integral', 'hit_id')]
        e_eid, = _read(f['edep_table'], 'event_id')
        e_hid, e_gid, e_fr = [
            np.asarray(f['edep_table'][k]).squeeze()
            for k in ('hit_id', 'g4_id', 'energy_fraction')]
        p_eid, = _read(f['particle_table'], 'event_id')
        p_gid, p_pdg, p_par, p_mom = [
            np.asarray(f['particle_table'][k]).squeeze()
            for k in ('g4_id', 'g4_pdg', 'parent_id', 'momentum')]
        p_proc = np.asarray(f['particle_table']['start_process']).squeeze()

        hi = _group_index(h_eid)
        ei = _group_index(e_eid)
        pi = _group_index(p_eid)

        n = 0
        for key, pidx in pi.items():
            if max_events is not None and n >= max_events:
                break
            tgt = find_target(p_gid[pidx], p_pdg[pidx], p_par[pidx],
                              p_proc[pidx], p_mom[pidx], mode)
            if tgt is None or key not in hi or key not in ei:
                continue
            target, p_tgt, mass = tgt
            E = float(p_tgt)  # paper-matching default: momentum
            keep = descendants(target, p_gid[pidx], p_par[pidx])
            eidx = ei[key]
            keep_hits = em_hit_ids(keep, e_hid[eidx], e_gid[eidx], e_fr[eidx])
            if not keep_hits:
                continue
            idx = hi[key]
            pts, q = solve_spacepoints(
                h_pl[idx], h_w[idx], hit_x(h_t[idx], h_pl[idx]),
                h_q[idx], h_id[idx], keep_hits)
            if len(pts) < min_points:
                continue
            n += 1
            yield Event(key, label, E, pts, q)
