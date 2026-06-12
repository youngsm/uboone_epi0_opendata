"""Faithful Python port of larreco sps::SpacePointSolver (C. Backhouse).

=============================== PROVENANCE ===============================
Ported from:  https://github.com/LArSoft/larreco
Commit:       6c4c0fd918a5577feeaa3865e5229de7b36f075c
              ("larreco v08_05_00 for larsoft v08_06_00",
               tag LARSOFT_SUITE_v08_06_00)
Files:        larreco/SpacePointSolver/{Solver.h, Solver.cxx,
              TripletFinder.cxx, SpacePointSolver_module.cc,
              SpacePointSolver.fcl}
Permalink base (abbreviated below as <L>):
  https://github.com/LArSoft/larreco/blob/6c4c0fd918a5577feeaa3865e5229de7b36f075c/larreco/SpacePointSolver/

Why this commit: the MicroBooNE OpenSamples were produced/analyzed in the
uboonecode v08 era. The exact contemporaneous larreco is v08_04_00_16
(commit 70bee021c9302047f55ddc1a410f2d080fd733fc, tag
LARSOFT_SUITE_v08_05_00_17 in BOTH LArSoft/larreco and the uboone/larreco
fork — same SHA, verified via `git ls-remote`). I verified by `git diff
LARSOFT_SUITE_v08_05_00_17..6c4c0fd -- larreco/SpacePointSolver/` that:
  * Solver.cxx and TripletFinder.cxx are BYTE-IDENTICAL between the tags
    (not in the diff at all), so every algorithm citation below is valid
    for both;
  * SpacePointSolver_module.cc differs ONLY in the hit-reading refactor
    (HitReaders/ tools added); the functions ported here — BuildSystem,
    AddNeighbours, Minimize, AddSpacePoint, FillSystemToSpacePoints —
    have zero diff hunks. Hit reading is replaced by HDF5 input in this
    package anyway (spacepoints.py), with one behavior retained from the
    era-tag produce(): hits with negative/NaN/inf Integral are skipped
    (70bee02 SpacePointSolver_module.cc, produce()).

CONFIGURATION PROVENANCE (genuinely ambiguous — read this):
  <L>SpacePointSolver.fcl#L8-L31  standard_spacepointsolver:
      WireIntersectThreshold 0.7 cm, DriftDir 0.4 cm, Alpha 0.05,
      MaxIterations 100/100, AllowBad{Induction,Collection}Hit true.
  <L>SpacePointSolver.fcl#L36-L43 microboone_spacepointsolver:
      0.3 cm / 0.3 cm, 20/20 iterations, AllowBad* FALSE (triplets only).
  The microboone_ block FIRST APPEARS at commit 6c4c0fd; at the
  OpenSamples-era tag (70bee02) only standard_ exists, and larreco's own
  uboone job fcl (reco3djob_uboone.fcl, identical at both tags) builds its
  producer from @local::standard_spacepointsolver. uboonecode
  v08_00_00_54's MCC9 reco fcl (github.com/uboone/uboonecode,
  fcl/reco/MCC9/reco_uboone_mcc9_8.fcl#L130) references
  @local::microboone_spacepointsolver. Which configuration the OT paper
  (arXiv:2506.09238) authors ran is NOT determinable from public code.
  Both are provided below; MICROBOONE is the default because it is the
  experiment-named tune and what the uboonecode reco chain requests.
  Confirm with the authors (cfang@ucsb.edu / jnhoward@kitp.ucsb.edu).

Wire-crossing geometry: instead of LArSoft's geo::WireIDsIntersect
(framework-bound), this port computes intersections analytically from the
wire-endpoint formulas in uboone/OpenSamples microboone_utils.py (commit
e55c8ad8f8a42eaa8db3381b1d689750dbb21d80) — see geometry.py, which is
validated against that file's own wireCrossingYZ to < 0.01 cm.
==========================================================================
"""
import numpy as np
from geometry import GEOM, collection_wire_z

# --- configurations: <L>SpacePointSolver.fcl#L8-L31 and #L36-L43 ----------
STANDARD = dict(dist_thresh=0.7, dist_thresh_drift=0.4, alpha=0.05,
                max_iter_noreg=100, max_iter_reg=100)
MICROBOONE = dict(dist_thresh=0.3, dist_thresh_drift=0.3, alpha=0.05,
                  max_iter_noreg=20, max_iter_reg=20)
# AllowBad*Hit (bad-channel doublets) is false in MICROBOONE and not
# implemented here at all: the OpenSamples HDF5 carries no channel-status
# database, so bad-channel triplet completion cannot be reproduced from
# these files under either config. For MICROBOONE this is exact; for
# STANDARD it is a documented omission.

CRIT_DIST = 5.0    # AddNeighbours kCritDist: <L>SpacePointSolver_module.cc#L177
PRIME = 1299827    # wire visiting stride:    <L>Solver.cxx#L313 (Iterate)


class IWire:
    """InductionWireHit: <L>Solver.h#L17 ctor <L>Solver.cxx#L12.
    Charge q = recob::Hit::Integral() (module BuildSystem,
    <L>SpacePointSolver_module.cc#L273: `new InductionWireHit(...,
    hit->Integral())`); here, hit_table/integral from the HDF5, which the
    OpenSamples schema documents as the same Gaussian-fit integral."""
    __slots__ = ('q', 'pred')

    def __init__(self, q):
        self.q = q
        self.pred = 0.0


class SC:
    """SpaceCharge: <L>Solver.h#L41 ctor <L>Solver.cxx#L24.
    add_charge() is a verbatim port of SpaceCharge::AddCharge,
    <L>Solver.cxx#L35: updates own prediction, both induction-wire
    predictions, and pushes dq*coupling into each neighbour's
    fNeiPotential."""
    __slots__ = ('x', 'y', 'z', 'w1', 'w2', 'pred', 'neipot', 'nei')

    def __init__(self, x, y, z, w1, w2):
        self.x, self.y, self.z = x, y, z
        self.w1, self.w2 = w1, w2
        self.pred = 0.0
        self.neipot = 0.0
        self.nei = []          # list of (SC, coupling)

    def add_charge(self, dq):
        self.pred += dq
        for sc, c in self.nei:
            sc.neipot += dq * c
        self.w1.pred += dq
        self.w2.pred += dq


# the U/V normal matrix is constant, so invert it once. np.linalg.solve
# per pair was a hot-loop LAPACK call in find_triplets; matmul against the
# cached inverse is identical to ~1e-15 and ~100x cheaper per call.
_UV_AINV = np.linalg.inv(np.array([GEOM[0].n, GEOM[1].n]))


def _uv_intersection(wu, wv):
    """(y,z) intersection of a plane-0 and plane-1 wire.
    Replaces geo::WireIDsIntersect (used via IntersectionCache,
    TripletFinder.cxx <L>TripletFinder.cxx#L83) with a 2x2 linear solve of
    the parallel-wire-family model fit in geometry.py. Chosen because the
    LArSoft geometry service can't run outside the art framework; accuracy
    vs the experiment's own crossing function is validated in
    geometry.self_test() to < 0.01 cm, below the official endpoint
    constants' own internal precision."""
    gu, gv = GEOM[0], GEOM[1]
    b0 = gu.a + gu.b * wu
    b1 = gv.a + gv.b * wv
    y = _UV_AINV[0, 0] * b0 + _UV_AINV[0, 1] * b1
    z = _UV_AINV[1, 0] * b0 + _UV_AINV[1, 1] * b1
    return y, z


def _close_space(p, q, thresh):
    """TripletFinder::CloseSpace, <L>TripletFinder.cxx#L124:
    2D (y,z) Euclidean distance between two wire intersections."""
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < thresh ** 2


def find_triplets(plane, wire, x, integral, cfg):
    """TripletFinder::Triplets(), <L>TripletFinder.cxx#L148:
      * X-U and X-V doublets (DoubletsXU #L250 / DoubletsXV #L263 via
        DoubletHelper #L276): drift coincidence |dx| < DriftDir threshold
        (CloseDrift, #L118) AND intersecting wires;
      * triplet: additionally CloseDrift(U,V) (#L185 in Triplets()) and
        all three pairwise intersections mutually CloseSpace (#L190-L192);
      * position: x = mean of available hit drift positions (#L194-L199),
        (y,z) = mean of the three intersections (#L201-L203);
      * ALL passing combinations kept — multiple SpaceCharges per X hit.
    Differences: the C++ iterates channel-sorted doublet lists with a
    catch-up pointer for speed; here numpy windows over drift-sorted hits
    give the same accepted set (the cuts are identical predicates).
    Bad-channel branches (x.hit==0 etc., #L181-L183) are not ported — see
    config note above. Callers must pass finite, non-negative integrals
    (the driver below applies the era-tag produce() skip and translates
    indices back to the caller's frame)."""
    Xm, Um, Vm = plane == 2, plane == 0, plane == 1
    xw, xx, xq = wire[Xm], x[Xm], integral[Xm]
    uw, ux, uq = wire[Um], x[Um], integral[Um]
    vw, vx, vq = wire[Vm], x[Vm], integral[Vm]
    dT, dD = cfg['dist_thresh'], cfg['dist_thresh_drift']

    gu, gv = GEOM[0], GEOM[1]
    dT2 = dT * dT
    # per-wire affine offsets for the cached-inverse U/V intersection
    bu_all = gu.a + gu.b * uw
    bv_all = gv.a + gv.b * vw
    trips = []
    for i in range(len(xw)):
        zX = float(collection_wire_z(xw[i]))
        du = np.flatnonzero(np.abs(ux - xx[i]) < dD)   # CloseDrift(X,U)
        dv = np.flatnonzero(np.abs(vx - xx[i]) < dD)   # CloseDrift(X,V)
        yu = gu.y_at_z(uw[du], zX)
        yv = gv.y_at_z(vw[dv], zX)
        inU = (yu > -116.5) & (yu < 118.0)   # wires actually cross in-detector
        inV = (yv > -116.5) & (yv < 118.0)   # (doWiresCross equivalent)
        du, yu = du[inU], yu[inU]
        dv, yv = dv[inV], yv[inV]
        if du.size == 0 or dv.size == 0:
            continue
        ua, va = ux[du], vx[dv]
        # pairwise (A,B) cuts, identical predicates to the C++ catch-up loop
        cdUV = np.abs(ua[:, None] - va[None, :]) < dD          # CloseDrift(U,V)
        dyXX = yu[:, None] - yv[None, :]
        cXUXV = dyXX * dyXX < dT2                              # CloseSpace(XU,XV)
        # U/V intersection via cached inverse (== np.linalg.solve to ~1e-15)
        b0 = bu_all[du]
        b1 = bv_all[dv]
        yUV = _UV_AINV[0, 0] * b0[:, None] + _UV_AINV[0, 1] * b1[None, :]
        zUV = _UV_AINV[1, 0] * b0[:, None] + _UV_AINV[1, 1] * b1[None, :]
        dzU = zX - zUV
        dyU = yu[:, None] - yUV
        dyV = yv[None, :] - yUV
        cXU = dyU * dyU + dzU * dzU < dT2
        cXV = dyV * dyV + dzU * dzU < dT2
        ia, ib = np.nonzero(cdUV & cXUXV & cXU & cXV)          # a-major order
        if ia.size == 0:
            continue
        xa = (xx[i] + ua[ia] + va[ib]) / 3.0                  # #L194-L199, nx=3
        pty = (yu[ia] + yv[ib] + yUV[ia, ib]) / 3.0
        ptz = (zX + zX + zUV[ia, ib]) / 3.0
        aidx, bidx = du[ia], dv[ib]
        for k in range(ia.size):
            trips.append((i, int(aidx[k]), int(bidx[k]),
                          (float(xa[k]), float(pty[k]), float(ptz[k]))))
    return trips, (xw, xx, xq, uq, vq)


def build_system(trips, xq, uq, vq, return_map=False):
    """SpacePointSolver::BuildSystem, <L>SpacePointSolver_module.cc#L273:
    one IWire per distinct induction hit; one collection wire per X hit
    with >=1 triplet, whose CollectionWireHit constructor
    (<L>Solver.cxx#L47) splits the hit charge EQUALLY among its crossings
    via AddCharge (`const double p = q/cross.size()`).
    Neighbours: AddNeighbours, <L>SpacePointSolver_module.cc#L175 —
    SpaceCharge pairs within kCritDist=5 cm (#L177, #L234), coupling
    exp(-d/2) (#L248, comment in source: 'a pretty random guess'). The
    C++ grid-bucket search is replaced by an exact cKDTree.query_pairs
    (same pair set; the bucket trick is only an optimization)."""
    iwU, iwV = {}, {}
    by_x = {}
    for ix, iu, iv, pt in trips:
        w1 = iwU.setdefault(iu, IWire(float(uq[iu])))
        w2 = iwV.setdefault(iv, IWire(float(vq[iv])))
        by_x.setdefault(ix, []).append(SC(*pt, w1, w2))

    scs, sc2x = [], []
    for ix, v in by_x.items():
        scs.extend(v)
        sc2x.extend([ix] * len(v))

    cwires = []
    for ix, crossings in by_x.items():
        q = float(xq[ix])
        p = q / len(crossings)                 # Solver.cxx#L47 ctor
        for sc in crossings:
            sc.add_charge(p)
        cwires.append((q, crossings))

    # AddNeighbours runs AFTER the CollectionWireHit constructors in
    # BuildSystem (#L362 vs #L162), so the initial q/N split must NOT seed
    # fNeiPotential. (An earlier version of this port wired neighbours
    # first — caught by equivalence-testing against the compiled original.)
    if len(scs) > 1:
        from scipy.spatial import cKDTree
        P = np.array([(s.x, s.y, s.z) for s in scs])
        tree = cKDTree(P)
        for i, j in tree.query_pairs(CRIT_DIST):
            d = float(np.linalg.norm(P[i] - P[j]))
            if d == 0:
                continue                       # #L237 'ZERO DISTANCE' guard
            c = np.exp(-d / 2.0)
            scs[i].nei.append((scs[j], c))
            scs[j].nei.append((scs[i], c))

    if return_map:
        return cwires, scs, sc2x
    return cwires, scs


def _metric(scs, alpha):
    """Metric(vector<SpaceCharge*>, alpha), <L>Solver.cxx#L83 (with the
    scalar Metric(q,p)=(q-p)^2 of #L71): sum over distinct induction wires
    of (q - pred)^2, minus alpha*(pred^2 + pred*neipot) per SpaceCharge
    (the double-counted neighbour term is intentional per the source
    comment at #L93-L95)."""
    ret = 0.0
    for sc in scs:
        if alpha:
            ret -= alpha * sc.pred ** 2
            ret -= alpha * sc.pred * sc.neipot
    seen = {}
    for sc in scs:
        for w in (sc.w1, sc.w2):
            seen[id(w)] = w
    for w in seen.values():
        ret += (w.q - w.pred) ** 2
    return ret


def _solve_pair(sci, scj, alpha):
    """QuadExpr Metric(sci, scj, alpha) + SolvePair:
    <L>Solver.cxx#L117 and #L217. The C++ builds a symbolic quadratic in x
    (charge moved scj -> sci) via QuadExpr; here the (quad, lin)
    coefficients are accumulated directly and the CONSTANT term is
    dropped — legitimate because SolvePair only compares evaluations of
    the same expression (x*, xmin, xmax, 0), in which constants cancel.
    Term-by-term mapping:
      * regularization self-energy  -a(pi+x)^2 - a(pj-x)^2      (#L129-L130)
      * neighbour potential         -2a(pi+x)nei_i - ...        (#L134-L135)
      * mutual-neighbour correction (#L138-L146): remove the miscounted
        terms, add -2a(pi+x)(pj-x)c
      * induction wires (#L156-L188): (q-(p+x))^2 per side, constant if
        the two SpaceCharges share that induction wire ('movement of
        charge cancels itself out', #L165-L167)
    Solution: argmin of the quadratic clamped to [-pred_i, +pred_j]
    (non-negativity, #L236-L242), then compare against both edges because
    the quadratic may be concave (#L249-L259)."""
    quad = lin = 0.0
    if alpha:
        pi, pj = sci.pred, scj.pred
        quad += -2 * alpha
        lin += -2 * alpha * pi + 2 * alpha * pj
        lin += -2 * alpha * sci.neipot + 2 * alpha * scj.neipot
        for sc, c in sci.nei:
            if sc is scj:
                lin += 2 * alpha * pj * c - 2 * alpha * pi * c
                quad += 2 * alpha * c
                lin += -2 * alpha * (pj - pi) * c
                break
    for wi, wj in ((sci.w1, scj.w1), (sci.w2, scj.w2)):
        if wi is wj:
            continue
        quad += 2.0
        lin += -2 * (wi.q - wi.pred) + 2 * (wj.q - wj.pred)

    x = 0.0 if quad == 0 else -lin / (2 * quad)
    xmin, xmax = -sci.pred, scj.pred
    x = min(xmax, max(xmin, x))

    def ev(t):
        return quad * t * t + lin * t
    return min([(ev(x), x), (ev(xmin), xmin), (ev(xmax), xmax)])[1]


def minimize(cwires, alpha, max_iter):
    """SpacePointSolver::Minimize, <L>SpacePointSolver_module.cc#L436,
    driving Iterate(vector<CollectionWireHit*>,...), <L>Solver.cxx#L313:
      * wires visited in stride-1299827 order ('helps prevent local
        artefacts', #L315-L317);
      * per wire, Iterate(cwire) (#L263) solves every crossing pair i<j
        and applies the transfer immediately;
      * converged when |dMetric| < 1e-3 |Metric| (#L451) or the metric
        increases (#L447-L450, warning + return in the source).
    Iterate(SpaceCharge*) for orphans (#L286) is not ported: orphans only
    arise from bad collection channels, which are not modeled here."""
    scs = [sc for _, cr in cwires for sc in cr]
    prev = _metric(scs, alpha)
    n = len(cwires)
    for _ in range(max_iter):
        idx = 0
        while True:
            _, crossings = cwires[idx]
            N = len(crossings)
            for i in range(N - 1):
                sci = crossings[i]
                for j in range(i + 1, N):
                    scj = crossings[j]
                    x = _solve_pair(sci, scj, alpha)
                    if x != 0.0:
                        sci.add_charge(+x)
                        scj.add_charge(-x)
            idx = (idx + PRIME) % n
            if idx == 0:
                break
        m = _metric(scs, alpha)
        if m > prev or abs(m - prev) < 1e-3 * abs(prev):
            return
        prev = m


def _solve_cpp(trips, xq, uq, vq, cfg):
    """Charge solve via the ACTUAL compiled Solver.cxx (cpp/spsolver_cpp;
    build: cpp/build.sh). System construction order matches
    SpacePointSolver_module.cc — see cpp/solver_bind.cpp. Certified
    equivalent to the Python minimize() at <1e-12 ADC by
    test_equivalence.py; use it as arbiter after any edit here."""
    import importlib, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp'))
    sp = importlib.import_module('spsolver_cpp')
    x_order = list(dict.fromkeys(t[0] for t in trips))
    xmap = {g: i for i, g in enumerate(x_order)}
    u_ids = sorted({t[1] for t in trips})
    v_ids = sorted({t[2] for t in trips})
    umap = {g: i for i, g in enumerate(u_ids)}
    vmap = {g: i + len(u_ids) for i, g in enumerate(v_ids)}
    trips = sorted(trips, key=lambda t: xmap[t[0]])
    xyz = np.array([t[3] for t in trips])
    iw1 = np.array([umap[t[1]] for t in trips], np.int64)
    iw2 = np.array([vmap[t[2]] for t in trips], np.int64)
    cw = np.array([xmap[t[0]] for t in trips], np.int64)
    iwq = np.concatenate([uq[u_ids], vq[v_ids]]).astype(float)
    cwq = xq[x_order].astype(float)
    pred = sp.solve_system(xyz, iw1, iw2, cw, iwq, cwq, cfg['alpha'],
                           cfg['max_iter_noreg'], cfg['max_iter_reg'])
    return trips, pred


def solve_spacepoints_larreco(plane, wire, x, integral, return_hits=False,
                              config=MICROBOONE, backend='python'):
    """Full chain on one event's hits, as the module's produce()
    (<L>SpacePointSolver_module.cc, Fit=true path): triplets ->
    BuildSystem -> Minimize(alpha=0) -> Minimize(alpha=cfg) -> emit.
    Output keeps SpaceCharges with positive solved charge only
    (AddSpacePoint, #L367: 'only happens when it has charge').
    config: MICROBOONE (default) or STANDARD — see header for why this
    choice is ambiguous and should be confirmed with the paper's authors.
    With return_hits=True also returns, per spacepoint, the index of its
    collection hit within the CALLER'S plane==2 hits, in input order
    (stable under the bad-integral skip below)."""
    # era-tag produce() hit-sanity skip (70bee02 SpacePointSolver_module.cc)
    good = np.isfinite(integral) & (integral >= 0)
    # surviving plane-2 hits' positions within the caller's plane-2 ordering
    coll_orig = np.flatnonzero(good[plane == 2])
    plane, wire, x, integral = (a[good] for a in (plane, wire, x, integral))
    trips, (xw, xx, xq, uq, vq) = find_triplets(plane, wire, x, integral,
                                                config)
    if not trips:
        out = (np.zeros((0, 3)), np.zeros(0))
        return out + (np.zeros(0, int),) if return_hits else out
    if backend == 'cpp':
        trips, pred = _solve_cpp(trips, xq, uq, vq, config)
        keep = np.flatnonzero(pred > 0)
        pts = np.asarray([trips[i][3] for i in keep], float)
        q = pred[keep]
        if return_hits:
            return pts, q, coll_orig[[trips[i][0] for i in keep]]
        return pts, q
    cwires, scs, sc2x = build_system(trips, xq, uq, vq, return_map=True)
    minimize(cwires, 0.0, config['max_iter_noreg'])
    minimize(cwires, config['alpha'], config['max_iter_reg'])
    keep = [i for i, s in enumerate(scs) if s.pred > 0]
    pts = np.asarray([(scs[i].x, scs[i].y, scs[i].z) for i in keep], float)
    q = np.asarray([scs[i].pred for i in keep], float)
    if return_hits:
        cidx = np.asarray([sc2x[i] for i in keep], int)
        return pts, q, coll_orig[cidx]
    return pts, q
