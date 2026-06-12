"""Regression test for preprocess.downsample.

Paper Sec. VI step 1 says "merge spacepoints within 1 cm (charge-weighted,
charges sum)" — which the surrounding PILArNet literature consistently uses
to mean 1 cm voxel binning (one centroid per 1 cm cube), NOT single-link
connected components at 1 cm.

Bug history: the original implementation used single-link clustering via
cKDTree.query_pairs(1.0). On dense EM showers (hit spacing well below 1 cm
along tracks), that transitively merged the entire shower into one giant
component — collapsing 100s of spacepoints to <20 centroids and gutting
fig4 displays and the OT classifier.

This test fails on the buggy implementation and passes on true voxelization.
"""
import numpy as np
from preprocess import downsample


def test_dense_line_is_not_collapsed_to_one_point():
    """A 100 cm line of points spaced 0.3 cm apart must voxelize to
    ~100 voxels at 1 cm, not 1 (single-link would chain the whole line)."""
    n = 333
    pts = np.zeros((n, 3))
    pts[:, 2] = np.linspace(0.0, 100.0, n)
    q = np.ones(n)
    p2, q2 = downsample(pts, q, 1.0)
    # 100 cm @ 1 cm voxels -> ~100 distinct voxels, never <50.
    assert len(p2) >= 50, (
        f"dense 100-cm line collapsed to {len(p2)} points; "
        "single-link chaining bug in downsample()")
    # Total charge must be conserved.
    assert np.isclose(q2.sum(), q.sum())


def test_charge_weighted_centroid_within_voxel():
    """Two points in the same 1 cm voxel must merge to one centroid."""
    pts = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    q = np.array([1.0, 3.0])
    p2, q2 = downsample(pts, q, 1.0)
    assert len(p2) == 1
    # charge-weighted centroid: (1*0.1 + 3*0.9)/4 = 0.7 on every axis
    assert np.allclose(p2[0], 0.7)
    assert np.isclose(q2[0], 4.0)


def test_distinct_voxels_kept_separate():
    """Two points separated by >1 cm in any axis must stay separate."""
    pts = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 2.5]])
    q = np.array([1.0, 1.0])
    p2, q2 = downsample(pts, q, 1.0)
    assert len(p2) == 2


def test_real_shower_voxel_count():
    """A toy EM shower (hundreds of pts over tens of cm) should retain
    tens-to-hundreds of voxels — not <20."""
    rng = np.random.default_rng(0)
    n = 400
    t = rng.beta(2.0, 2.5, n) * 80.0          # 80 cm long shower
    sigma = 0.5 + 0.12 * t
    pts = np.stack([rng.normal(0, sigma), rng.normal(0, sigma), t], axis=1)
    q = rng.gamma(2.0, 50.0, n)
    p2, q2 = downsample(pts, q, 1.0)
    assert len(p2) > 50, f"shower collapsed to {len(p2)} voxels"
    assert np.isclose(q2.sum(), q.sum())


if __name__ == '__main__':
    test_dense_line_is_not_collapsed_to_one_point()
    test_charge_weighted_centroid_within_voxel()
    test_distinct_voxels_kept_separate()
    test_real_shower_voxel_count()
    print("PASS")
