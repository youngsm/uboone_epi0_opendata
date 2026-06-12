"""Concatenate per-shard PILArNet HDF5 files (from HPC array jobs) into one.

Each shard is the output of make_pilarnet.py on a subset of input files; we
just concatenate the per-event datasets in shard order. The vlen point/sem/
instance arrays are read row-by-row and re-emitted as new vlen arrays.

Usage:
  python merge_pilarnet.py --in 'shards/*.h5' --out pilarnet_epi0.h5
  python merge_pilarnet.py --in shard_a.h5 shard_b.h5 --out merged.h5
"""
import argparse, glob, os
import h5py
import numpy as np

# Datasets we always copy through. Order matters only for human readability;
# concat is row-wise on axis 0.
SCALAR_KEYS = [
    'event_id', 'label', 'true_momentum', 'true_ke', 'true_energy_total',
    'visible_energy', 'true_energy', 'n_boundary',
]
VLEN_KEYS = ['points', 'semantic', 'instance']
ATTR_KEYS = [
    'energy_convention', 'points_columns', 'semantic_classes',
    'instance_classes', 'paper_selection',
]


def merge(in_paths, out_path):
    in_paths = sorted(in_paths)
    if not in_paths:
        raise SystemExit('no input shards matched')

    counts = []
    for p in in_paths:
        with h5py.File(p, 'r') as f:
            counts.append(len(f['label']))
    total = sum(counts)
    print(f'merging {len(in_paths)} shards, {total} events total')

    with h5py.File(out_path, 'w') as out:
        # Fixed-shape datasets: read each shard, concat, write once.
        for k in SCALAR_KEYS:
            chunks = []
            for p in in_paths:
                with h5py.File(p, 'r') as f:
                    if k in f:
                        chunks.append(f[k][:])
            if not chunks:
                continue
            out.create_dataset(k, data=np.concatenate(chunks, axis=0))

        # vlen datasets: collect Python objects, write as one vlen dataset.
        for k in VLEN_KEYS:
            rows = np.empty(total, dtype=object)
            dt_inner = None
            i = 0
            for p in in_paths:
                with h5py.File(p, 'r') as f:
                    if k not in f:
                        continue
                    if dt_inner is None:
                        dt_inner = f[k].dtype
                    arr = f[k][:]
                    for row in arr:
                        rows[i] = row
                        i += 1
            if i != total:
                raise SystemExit(
                    f"shape mismatch for {k}: got {i} rows, expected {total}")
            out.create_dataset(k, dtype=h5py.vlen_dtype(dt_inner.metadata['vlen']),
                               data=rows)

        # Attributes: take from the first shard that has each one.
        for k in ATTR_KEYS:
            for p in in_paths:
                with h5py.File(p, 'r') as f:
                    if k in f.attrs:
                        out.attrs[k] = f.attrs[k]
                        break

    print(f'wrote {total} events to {out_path} '
          f'({os.path.getsize(out_path) / 1e9:.2f} GB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', nargs='+', required=True,
                    help='shard files or glob patterns')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    paths = sorted(set(sum([glob.glob(p) if any(c in p for c in '*?[')
                            else [p] for p in a.inp], [])))
    merge(paths, a.out)
