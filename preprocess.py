"""Pre-processing per arXiv:2506.09238 Sec. VI:
  A. boundary filter: drop events with >10 deposits within 5 cm of boundary
  1. downsampling: merge spacepoints within 1 cm (charge-weighted, charges sum)
  2. alignment: cluster at 2.5 cm linking distance; WPCA (Delchambre,
     charge-weighted covariance) on the LARGEST cluster; rotate PC1 -> z,
     PC2 -> x; translate charge-weighted COM to origin.
"""
import numpy as np
from scipy.spatial import cKDTree
from geometry import XLO, XHI, YLO, YHI, ZLO, ZHI


def near_boundary_count(pts, margin=5.0):
    d = np.minimum.reduce([
        pts[:, 0] - XLO, XHI - pts[:, 0],
        pts[:, 1] - YLO, YHI - pts[:, 1],
        pts[:, 2] - ZLO, ZHI - pts[:, 2]])
    return int((d < margin).sum())


def passes_boundary_filter(pts, margin=5.0, max_near=10):
    return near_boundary_count(pts, margin) <= max_near


def _union_components(pts, linking):
    """Connected components under 'within linking distance' (union-find)."""
    n = len(pts)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(pts)
    for i, j in tree.query_pairs(linking):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def downsample(pts, q, vsize=1.0):
    """Voxel downsampling per paper Sec. VI.1: bin into vsize cubes; each
    occupied voxel becomes one point at the charge-weighted centroid with
    summed charge. NB: an earlier implementation used single-link clustering
    at vsize, which transitively merged dense showers (hit spacing < vsize
    along tracks) into a handful of centroids — that gutted fig4 and the
    OT classifier inputs. PILArNet preprocessing means spatial binning."""
    if len(pts) == 0:
        return pts, q
    keys = np.floor(pts / vsize).astype(np.int64)
    order = np.lexsort(keys.T[::-1])
    sk, sp, sq = keys[order], pts[order], q[order]
    boundaries = np.r_[0,
                       np.where(np.any(sk[1:] != sk[:-1], axis=1))[0] + 1,
                       len(sk)]
    n_vox = len(boundaries) - 1
    out_p = np.empty((n_vox, 3))
    out_q = np.empty(n_vox)
    for i in range(n_vox):
        a, b = boundaries[i], boundaries[i + 1]
        w = sq[a:b]
        out_p[i] = (sp[a:b] * w[:, None]).sum(0) / w.sum()
        out_q[i] = w.sum()
    return out_p, out_q


def wpca_axes(pts, q):
    """Delchambre weighted PCA: eigenvectors of the charge-weighted
    covariance of centered coordinates, sorted by decreasing eigenvalue."""
    w = q / q.sum()
    mu = (pts * w[:, None]).sum(0)
    d = pts - mu
    cov = (d * w[:, None]).T @ d
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order], mu


def align(pts, q, linking=2.5):
    """Full alignment of one event. Axes are computed on the largest cluster
    (by point count, ties broken by charge); rotation applied to all points.
    PC sign fixed by charge-weighted skewness > 0 along each axis so the
    convention is deterministic."""
    lab = _union_components(pts, linking)
    uniq, counts = np.unique(lab, return_counts=True)
    big = uniq[np.argmax(counts)]
    m = lab == big
    axes, _ = wpca_axes(pts[m], q[m])

    w = q / q.sum()
    com = (pts * w[:, None]).sum(0)
    d = pts - com
    # rotation: rows = new basis -> PC1 becomes z, PC2 becomes x, PC3 y
    pc1, pc2 = axes[:, 0], axes[:, 1]
    pc3 = np.cross(pc1, pc2)
    R = np.vstack([pc2, pc3, pc1])          # new (x, y, z)
    out = d @ R.T
    for k in range(3):                       # deterministic sign
        skew = (w * out[:, k] ** 3).sum()
        if skew < 0:
            out[:, k] *= -1
    return out, q


def preprocess_event(ev, do_boundary=True):
    """Returns (points, charge) aligned & centered, or None if filtered."""
    if do_boundary and not passes_boundary_filter(ev.points):
        return None
    pts, q = downsample(ev.points, ev.charge, 1.0)
    if len(pts) < 5:
        return None
    return align(pts, q)
