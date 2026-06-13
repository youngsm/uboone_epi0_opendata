"""Concatenate per-shard PILArNet HDF5 files (from HPC array jobs) into one.

Auto-discovers the schema from the first shard, so the same tool merges
shards produced by make_pilarnet.py (e/pi0) or make_pilarnet_full.py
(multi-class) without configuration.

Usage:
  python merge_pilarnet.py --in 'shards/*.h5'      --out pilarnet_epi0.h5
  python merge_pilarnet.py --in 'shards_full/*.h5' --out pilarnet_full.h5
"""
import argparse, glob, os
import h5py
import numpy as np


def _schema(path):
    """Return (vlen_keys, scalar_keys, attr_keys, n_events) from a shard."""
    vlen, scalar = [], []
    with h5py.File(path, 'r') as f:
        for k in f.keys():
            md = f[k].dtype.metadata or {}
            (vlen if md.get('vlen') is not None else scalar).append(k)
        if not scalar:
            raise SystemExit(f'{path}: no fixed-shape datasets to count events')
        n = len(f[scalar[0]])
        attrs = list(f.attrs)
    return vlen, scalar, attrs, n


def merge(in_paths, out_path):
    in_paths = sorted(in_paths)
    if not in_paths:
        raise SystemExit('no input shards matched')

    vlen_keys, scalar_keys, attr_keys, _ = _schema(in_paths[0])
    counts = [_schema(p)[3] for p in in_paths]
    total = sum(counts)
    print(f'merging {len(in_paths)} shards, {total} events total')
    print(f'  scalar: {scalar_keys}')
    print(f'  vlen:   {vlen_keys}')

    with h5py.File(out_path, 'w') as out:
        # Fixed-shape datasets: concat row-wise.
        for k in scalar_keys:
            chunks = []
            for p in in_paths:
                with h5py.File(p, 'r') as f:
                    if k in f:
                        chunks.append(f[k][:])
            if chunks:
                out.create_dataset(k, data=np.concatenate(chunks, axis=0))

        # vlen datasets: object array of rows.
        for k in vlen_keys:
            rows = np.empty(total, dtype=object)
            inner_dt = None
            i = 0
            for p in in_paths:
                with h5py.File(p, 'r') as f:
                    if k not in f:
                        continue
                    if inner_dt is None:
                        inner_dt = f[k].dtype.metadata['vlen']
                    for row in f[k][:]:
                        rows[i] = row
                        i += 1
            if i != total:
                raise SystemExit(
                    f"shape mismatch for {k}: got {i} rows, expected {total}")
            out.create_dataset(k, dtype=h5py.vlen_dtype(inner_dt), data=rows)

        # Attributes: take from the first shard that has each one.
        for k in attr_keys:
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
