"""Vectorized MicroBooNE wire geometry.

PROVENANCE: every constant and formula derives from MicroBooNE's official
utilities, vendored here as microboone_utils.py from
https://github.com/uboone/OpenSamples (HEAD commit
e55c8ad8f8a42eaa8db3381b1d689750dbb21d80 at retrieval, 2026-06-11):
  * TIME2CM / TRIG_OFFSET <- tpcTimeToX(),
    https://github.com/uboone/OpenSamples/blob/main/microboone_utils.py#L104-L107
  * active-volume bounds  <- isPosInActiveVolume(), #L97-L102
  * wire endpoint formulas (planes 0/1/2) <- wireStartPos*/wireEndPos*,
    #L18-L40; collection_wire_z() below is wireStartPosPlane2's z term.

WHY a refit instead of calling those functions directly: they are scalar
Python with clipped endpoints; the solver needs vectorized inverse maps
(position -> wire, wire -> y(z)). Since each plane is a family of parallel
lines, the signed perpendicular coordinate is exactly linear in wire
number; PlaneGeom fits that linear model against the official functions
(asserting exactness) rather than hand-deriving plane angles, and
self_test() validates 200 random crossings against the official
wireCrossingYZ (#L69-L94) to < 0.01 cm — the level at which the official
start/end constants (0.34641 vs 0.34640) themselves disagree.
"""
import numpy as np
import microboone_utils as mu

TIME2CM = 0.0548965   # from microboone_utils.tpcTimeToX
TRIG_OFFSET = 800.0

# Active volume (matches mu.isPosInActiveVolume)
XLO, XHI = 0.0, 255.0
YLO, YHI = -116.0, 116.0
ZLO, ZHI = 10.0, 1036.0


def tick_to_x(t):
    return (np.asarray(t, dtype=float) - TRIG_OFFSET) * TIME2CM


# Per-plane drift-coordinate offsets [cm], relative to the collection plane.
# WHY: larreco converts hit time to x via detprop->ConvertTicksToX(time,
# wireID) (TripletFinder.cxx#L41), which is PLANE-DEPENDENT: the wire
# planes sit at x = 0 / -0.3 / -0.6 cm, so the same deposit arrives at U,
# then V, then collection (~0.3 cm extra drift per gap, plus per-plane
# signal-timing offsets). The OpenSamples tpcTimeToX is plane-agnostic;
# applying it uniformly makes x_coll - x_U ~ +0.6 cm systematically, which
# exceeds the 0.3 cm MICROBOONE drift tolerance and kills genuine
# triplets. Calibrate from data with calibrate_offsets.py; geometric
# expectation is roughly (-0.6, -0.3, 0.0).
PLANE_OFFSETS_CM = np.zeros(3)


def load_plane_offsets(path):
    import json
    global PLANE_OFFSETS_CM
    with open(path) as fh:
        PLANE_OFFSETS_CM = np.asarray(json.load(fh)['offsets_cm'], float)
    return PLANE_OFFSETS_CM


def hit_x(t, plane):
    """Plane-aware drift coordinate (the ConvertTicksToX analog)."""
    return tick_to_x(t) - PLANE_OFFSETS_CM[np.asarray(plane, int)]


class PlaneGeom:
    """Parallel-wire family: n . (y,z) = a + b*w  (n unit normal to wires)."""

    def __init__(self, plane):
        self.plane = plane
        # direction from a mid-detector wire (clipping changes extent, not direction)
        wref = 1200
        s = np.array(mu.wireStartPos(plane, wref), float)
        e = np.array(mu.wireEndPos(plane, wref), float)
        d = e[1:] - s[1:]
        d /= np.linalg.norm(d)
        self.n = np.array([-d[1], d[0]])  # unit normal in (y,z)
        # fit offset(w) = a + b*w on sampled wires
        ws = np.arange(200, mu.nwires(plane) - 200, 50)
        offs = np.array([self.n @ np.array(mu.wireStartPos(plane, int(w)), float)[1:]
                         for w in ws])
        A = np.vstack([np.ones_like(ws, float), ws.astype(float)]).T
        (self.a, self.b), res, *_ = np.linalg.lstsq(A, offs, rcond=None)
        # sanity: linear fit must be exact to numerical noise
        assert np.abs(A @ np.array([self.a, self.b]) - offs).max() < 1e-3, \
            f"plane {plane} wire model not linear"

    def wire_of(self, y, z):
        """Nearest wire number for points (y,z)."""
        s = self.n[0] * np.asarray(y, float) + self.n[1] * np.asarray(z, float)
        w = np.rint((s - self.a) / self.b).astype(int)
        return np.clip(w, 0, mu.nwires(self.plane) - 1)

    def y_at_z(self, w, z):
        """y of wire(s) w at longitudinal position z. Requires n_y != 0
        (true for induction planes; collection wires are vertical in y)."""
        s = self.a + self.b * np.asarray(w, float)
        return (s - self.n[1] * np.asarray(z, float)) / self.n[0]


GEOM = {p: PlaneGeom(p) for p in range(3)}


def collection_wire_z(w):
    return 0.25 + 0.3 * np.asarray(w, float)  # from wireStartPosPlane2


def self_test():
    """Verify against the official scalar crossing function."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = int(rng.integers(0, 2))           # induction plane
        wu = int(rng.integers(300, 2100))
        wc = int(rng.integers(100, 3300))     # collection wire
        if not mu.doWiresCross(p, wu, 2, wc):
            continue
        y_ref, z_ref = mu.wireCrossingYZ(p, wu, 2, wc)
        z = collection_wire_z(wc)
        y = GEOM[p].y_at_z(wu, z)
        assert abs(z - z_ref) < 1e-6 and abs(y - y_ref) < 1e-2, \
            (p, wu, wc, y, y_ref)
    return True


if __name__ == "__main__":
    print("geometry self-test:", "PASS" if self_test() else "FAIL")
