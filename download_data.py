"""Download the MicroBooNE OpenSamples NoWire HDF5 files from Zenodo.

  electron source : record 7261921  (BNB intrinsic nue overlay, ~31 GB, 20 files)
  pi0 source      : record 8370883  (BNB inclusive overlay, ~195 GB, 18 files)

Supports HTTP-Range resume; downloads N files concurrently (--workers,
default 4). The two sample records can also be fetched in parallel from
two shells since they hit different Zenodo records.

Usage:
  python download_data.py --sample nue       --dest data/nue --workers 4
  python download_data.py --sample inclusive --dest data/bnb --workers 4
"""
import argparse, os, urllib.request, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

RECORDS = {'nue': 7261921, 'inclusive': 8370883}
_print_lock = threading.Lock()


def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def download(url, dest, label=''):
    tmp = dest + '.part'
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if pos:
        req.add_header('Range', f'bytes={pos}-')
    with urllib.request.urlopen(req) as r, open(tmp, 'ab') as f:
        total = pos + int(r.headers.get('Content-Length', 0))
        last = pos
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            pos += len(chunk)
            if pos - last > (1 << 30):     # log per ~1 GB to avoid spam
                _log(f"  {label}: {pos/1e9:.1f}/{total/1e9:.1f} GB")
                last = pos
    os.replace(tmp, dest)
    _log(f"  {label}: done ({pos/1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', choices=RECORDS, required=True)
    ap.add_argument('--dest', default='data')
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4,
                    help='concurrent file downloads (default 4); raise '
                         'cautiously, Zenodo throttles aggressive clients')
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    rec = fetch_json(f"https://zenodo.org/api/records/{RECORDS[a.sample]}")
    files = [f for f in rec['files'] if f['key'].endswith(('.h5', '.hdf5'))]
    files.sort(key=lambda f: f['key'])
    if a.max_files:
        files = files[:a.max_files]
    print(f"{a.sample}: {len(files)} files, "
          f"{sum(f['size'] for f in files)/1e9:.1f} GB, "
          f"{a.workers} parallel worker(s)")

    todo = []
    for f in files:
        out = os.path.join(a.dest, f['key'])
        if os.path.exists(out) and os.path.getsize(out) == f['size']:
            print(f"  {f['key']}: already complete")
            continue
        todo.append((f['links']['self'], out, f['key']))

    failures = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(download, url, dest, label): label
                for url, dest, label in todo}
        for fu in as_completed(futs):
            try:
                fu.result()
            except Exception as e:
                failures.append((futs[fu], e))
                _log(f"  {futs[fu]}: FAILED ({e})")
    if failures:
        raise SystemExit(f"{len(failures)} file(s) failed; "
                         "re-run to resume (HTTP Range)")


if __name__ == '__main__':
    main()
