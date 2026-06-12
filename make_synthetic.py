"""Generate a synthetic HDF5 file with the exact OpenSamples NoWire schema
(hit_table / edep_table / particle_table / event_table fields used by the
pipeline) populated with toy electron (1 cone) and pi0 (2 photon cones,
conversion gaps) events. Used to validate the full pipeline end-to-end:
known 3D truth -> per-plane hits -> spacepoint solver -> preprocessing -> OT.
"""
import numpy as np
import h5py
from geometry import GEOM, TIME2CM, TRIG_OFFSET

# Per-plane arrival-time offsets [ticks], mimicking the plane-dependent
# ConvertTicksToX physics (planes at x = 0/-0.3/-0.6 cm: U arrives first,
# collection last). ~0.3 cm per gap => ~5.5 ticks. Injected so that a
# plane-agnostic time->x conversion FAILS triplet matching, as on real
# data; the pipeline must calibrate these out (calibrate_offsets.py).
PLANE_TICK_OFFSETS = np.array([0.0, 12.75, 25.5])


def _cone(rng, vtx, direction, energy, n_per_gev=400, length_per_gev=40.0):
    """Toy EM shower: deposits spread along `direction` with growing
    transverse profile. Returns (N,3) points and (N,) charges."""
    n = max(30, int(n_per_gev * energy))
    L = min(120.0, length_per_gev * max(energy, 0.1) + 20.0)
    t = rng.beta(2.0, 2.5, n) * L
    sigma = 0.5 + 0.12 * t
    d = direction / np.linalg.norm(direction)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    r1 = rng.normal(0, sigma)
    r2 = rng.normal(0, sigma)
    pts = vtx + t[:, None] * d + r1[:, None] * u + r2[:, None] * v
    q = rng.gamma(2.0, 50.0, n) * energy
    return pts, q


def _make_event(rng, mode, energy):
    """Returns (particles, deposits) where particles is a list of dicts and
    deposits is a list of (g4_id, point, charge)."""
    vtx = np.array([rng.uniform(60, 200), rng.uniform(-70, 70),
                    rng.uniform(150, 850)])
    parts, deps = [], []
    if mode == 'electron':
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        parts.append(dict(g4_id=1, pdg=11, parent=-1, proc=b'primary',
                          mom=energy))
        pts, q = _cone(rng, vtx, d, energy)
        deps += [(1, p, c) for p, c in zip(pts, q)]
    else:  # pi0 -> two photons with conversion gaps
        p_pi0 = float(np.sqrt(max(energy ** 2 - 0.13498 ** 2, 1e-6)))
        parts.append(dict(g4_id=1, pdg=111, parent=-1, proc=b'primary',
                          mom=p_pi0))
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        perp = np.cross(axis, [0, 0, 1.0]); perp /= np.linalg.norm(perp)
        half = rng.uniform(0.15, 0.6)            # half opening angle (rad)
        frac = rng.uniform(0.3, 0.5)             # energy sharing
        for k, (sgn, f) in enumerate([(+1, frac), (-1, 1 - frac)]):
            d = np.cos(half) * axis + sgn * np.sin(half) * perp
            gap = rng.exponential(20.0)          # ~ conversion length
            gid = 2 + k
            parts.append(dict(g4_id=gid, pdg=22, parent=1, proc=b'conv',
                              mom=f * energy))
            pts, q = _cone(rng, vtx + gap * d, d, f * energy)
            deps += [(gid, p, c) for p, c in zip(pts, q)]
    return parts, deps


def write_synthetic(path, n_electron=60, n_pi0=60, seed=1,
                    e_range=(0.2, 0.25)):
    rng = np.random.default_rng(seed)
    H = dict(event_id=[], hit_id=[], local_plane=[], local_wire=[],
             local_time=[], integral=[], rms=[], tpc=[])
    E = dict(event_id=[], hit_id=[], g4_id=[], energy=[], energy_fraction=[])
    P = dict(event_id=[], g4_id=[], g4_pdg=[], parent_id=[], momentum=[],
             start_process=[])
    EV = dict(event_id=[])

    evn = 0
    for mode, n in (('electron', n_electron), ('pi0', n_pi0)):
        for _ in range(n):
            evn += 1
            eid = (1, 1, evn)
            energy = rng.uniform(*e_range)
            parts, deps = _make_event(rng, mode, energy)
            EV['event_id'].append(eid)
            for p in parts:
                P['event_id'].append(eid)
                P['g4_id'].append(p['g4_id'])
                P['g4_pdg'].append(p['pdg'])
                P['parent_id'].append(p['parent'])
                P['momentum'].append(p['mom'])
                P['start_process'].append(p['proc'])
            hid = 0
            for gid, pt, c in deps:
                x, y, z = pt
                if not (5 < x < 250 and -110 < y < 110 and 15 < z < 1030):
                    continue
                for pl in range(3):
                    t = x / TIME2CM + TRIG_OFFSET + PLANE_TICK_OFFSETS[pl]
                    H['event_id'].append(eid)
                    H['hit_id'].append(hid)
                    H['local_plane'].append(pl)
                    H['local_wire'].append(int(GEOM[pl].wire_of(y, z)))
                    H['local_time'].append(t + rng.normal(0, 0.5))
                    H['integral'].append(c * (1.0 if pl == 2 else 0.8))
                    H['rms'].append(2.0)
                    H['tpc'].append(0)
                    E['event_id'].append(eid)
                    E['hit_id'].append(hid)
                    E['g4_id'].append(gid)
                    E['energy'].append(c)
                    E['energy_fraction'].append(1.0)
                    hid += 1

    def _seq_cnt(eids, ev_order):
        """event_id.seq_cnt companion dataset, as in the OpenSamples
        schema: (event_table index, contiguous row count) per event in
        storage order."""
        eids = np.array(eids, np.int64)
        keys = [tuple(r) for r in eids]
        idx_of = {k: i for i, k in enumerate(ev_order)}
        out, start = [], 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                out.append((idx_of[keys[start]], i - start))
                start = i
        return np.array(out, np.int64)

    ev_order = [tuple(e) for e in EV['event_id']]
    with h5py.File(path, 'w') as f:
        g = f.create_group('event_table')
        g.create_dataset('event_id', data=np.array(EV['event_id'], np.int32))
        g = f.create_group('hit_table')
        g.create_dataset('event_id', data=np.array(H['event_id'], np.int32))
        g.create_dataset('event_id.seq_cnt',
                         data=_seq_cnt(H['event_id'], ev_order))
        for k, dt in (('hit_id', np.int64), ('local_plane', np.int32),
                      ('local_wire', np.int32), ('local_time', np.float64),
                      ('integral', np.float64), ('rms', np.float64),
                      ('tpc', np.int32)):
            g.create_dataset(k, data=np.array(H[k], dt)[:, None],
                             chunks=(128, 1), compression='gzip')
        g = f.create_group('edep_table')
        g.create_dataset('event_id', data=np.array(E['event_id'], np.int32))
        g.create_dataset('event_id.seq_cnt',
                         data=_seq_cnt(E['event_id'], ev_order))
        for k, dt in (('hit_id', np.int64), ('g4_id', np.int64),
                      ('energy', np.float64), ('energy_fraction', np.float64)):
            g.create_dataset(k, data=np.array(E[k], dt)[:, None],
                             chunks=(128, 1), compression='gzip')
        g = f.create_group('particle_table')
        g.create_dataset('event_id', data=np.array(P['event_id'], np.int32))
        g.create_dataset('event_id.seq_cnt',
                         data=_seq_cnt(P['event_id'], ev_order))
        for k, dt in (('g4_id', np.int64), ('g4_pdg', np.int64),
                      ('parent_id', np.int64), ('momentum', np.float64)):
            g.create_dataset(k, data=np.array(P[k], dt)[:, None],
                             chunks=(128, 1), compression='gzip')
        g.create_dataset('start_process',
                         data=np.array(P['start_process'], dtype='S16')[:, None],
                         chunks=(128, 1), compression='gzip')
    return path


if __name__ == '__main__':
    print(write_synthetic('/home/claude/ot_repro/synthetic.h5'))
