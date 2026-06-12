"""Download the MicroBooNE OpenSamples NoWire HDF5 files from Zenodo.

  electron source : record 7261921  (BNB intrinsic nue overlay, ~31 GB, 20 files)
  pi0 source      : record 8370883  (BNB inclusive overlay, ~195 GB, 18 files)

Supports resume (HTTP Range). You do NOT need all files to validate the
chain — one file per sample is enough for a first pass; full statistics
(>=1000 per class per energy bin) needs most of the nue sample and a few
inclusive files.

Usage:
  python download_data.py --sample nue --dest data/nue --max-files 2
  python download_data.py --sample inclusive --dest data/bnb --max-files 2
"""
import argparse, os, sys, urllib.request, json

RECORDS = {'nue': 7261921, 'inclusive': 8370883}


def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def download(url, dest):
    tmp = dest + '.part'
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if pos:
        req.add_header('Range', f'bytes={pos}-')
    with urllib.request.urlopen(req) as r, open(tmp, 'ab') as f:
        total = pos + int(r.headers.get('Content-Length', 0))
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            pos += len(chunk)
            print(f"\r  {dest}: {pos/1e9:.2f}/{total/1e9:.2f} GB",
                  end='', flush=True)
    print()
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', choices=RECORDS, required=True)
    ap.add_argument('--dest', default='data')
    ap.add_argument('--max-files', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    rec = fetch_json(f"https://zenodo.org/api/records/{RECORDS[a.sample]}")
    files = [f for f in rec['files'] if f['key'].endswith(('.h5', '.hdf5'))]
    files.sort(key=lambda f: f['key'])
    if a.max_files:
        files = files[:a.max_files]
    print(f"{a.sample}: {len(files)} files, "
          f"{sum(f['size'] for f in files)/1e9:.1f} GB")
    for f in files:
        out = os.path.join(a.dest, f['key'])
        if os.path.exists(out) and os.path.getsize(out) == f['size']:
            print(f"  {f['key']}: already complete")
            continue
        download(f['links']['self'], out)


if __name__ == '__main__':
    main()
