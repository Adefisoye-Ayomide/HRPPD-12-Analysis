"""
01_build_cache_hrppd.py
=======================
Converts HRPPD dark rate WFM files to NPZ cache.
Run this first before the analysis script.

Expected file naming:
    {DATA_DIR}/{RUN_PREFIX}_{PC}V_ch{N}_{timestamp}.wfm
    e.g. 260527/0527AA_30_ch1_20260527112423677.wfm

Usage:
    python 01_build_cache_hrppd.py [DATA_DIR] [--cache-root ROOT] [--delete-raw]

    If DATA_DIR is omitted, falls back to the DATA_DIR constant below.
    If --cache-root is omitted, falls back to CACHE_ROOT constant below.
    --delete-raw deletes the source .wfm file after its NPZ cache has been
    written (or confirmed already up to date) AND read back and verified.
    Off by default. Deleted paths are logged to
    {cache_root}/deleted_raw_manifest.log.
"""

import os, sys, glob, time, hashlib, argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import tekwfm

# ─────────────────────────────────────────────
#  CONFIG (used only if not overridden on CLI)
# ─────────────────────────────────────────────
DATA_DIR   = "/home/matthew/HRPPD_characterization/RAW_DATA/260803XY"
CACHE_ROOT = "/home/matthew/HRPPD_characterization/CACHE"   # <-- all NPZ files now live under here
EXT        = ".wfm"
# ─────────────────────────────────────────────


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def file_sig(path: str) -> str:
    st  = os.stat(path)
    raw = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cache_path_for(wfm_path: str, data_dir: str, cache_root: str) -> str:
    """
    Mirrors the WFM file's path *relative to data_dir* underneath cache_root,
    instead of dropping a CACHE/ folder next to the raw files. This keeps
    cache_root as a single, fixed, greppable location regardless of which
    drive/mount the raw data happens to live on.

    e.g. data_dir=/media/.../260801XY, wfm=.../260801XY/run1/foo.wfm,
         cache_root=/home/matthew/hrppd_cache
      -> /home/matthew/hrppd_cache/260801XY/run1/foo.npz
    """
    rel      = os.path.relpath(wfm_path, start=data_dir)
    rel_base = os.path.splitext(rel)[0] + ".npz"
    out_path = os.path.join(cache_root, os.path.basename(os.path.normpath(data_dir)), rel_base)
    ensure_dir(os.path.dirname(out_path))
    return out_path


def save_npz_atomic(path: str, **arrays):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def verify_npz(cp: str, expected_sig: str, expected_n: int) -> bool:
    """
    Read-back verification. Returns True only if the NPZ on disk:
      - has the expected signature (matches the raw file it was cached from)
      - has non-empty t/v arrays
      - t and v are the same length
      - that length matches the number of samples read from the raw WFM
    This is what raw-file deletion is gated on — not "the write call didn't
    throw," but "we reopened the file and the data is actually there."
    """
    try:
        with np.load(cp, allow_pickle=False) as z:
            if "__sig__" not in z or "t" not in z or "v" not in z:
                return False
            if z["__sig__"].tobytes().decode() != expected_sig:
                return False
            t, v = z["t"], z["v"]
            if t.size == 0 or v.size == 0:
                return False
            if t.size != v.size:
                return False
            if expected_n is not None and t.size != expected_n:
                return False
            return True
    except Exception:
        return False


def wfm_to_npz(args):
    wfm_path, data_dir, cache_root, delete_raw = args
    try:
        cp  = cache_path_for(wfm_path, data_dir, cache_root)
        sig = file_sig(wfm_path)

        already_cached = False
        if os.path.exists(cp):
            with np.load(cp, allow_pickle=False) as z:
                if "__sig__" in z and z["__sig__"].tobytes().decode() == sig:
                    already_cached = True

        if already_cached:
            status, n_expected = "skip", None
        else:
            volts, tstart, tscale, tfrac, _, _ = tekwfm.read_wfm(wfm_path)
            v    = 1e3 * volts[:, 0]                          # V → mV
            toff = tfrac[0] * tscale                          # frame 0, matches v above
            n    = volts.shape[0]
            t    = np.linspace(tstart + toff,
                               tstart + toff + n * tscale,
                               num=n, endpoint=False)
            t    = 1e9 * t                                     # s → ns

            save_npz_atomic(cp,
                            t=t, v=v,
                            __sig__=np.frombuffer(sig.encode(), dtype=np.uint8))
            status, n_expected = "written", n

        if not delete_raw:
            return (status, wfm_path, None, None)

        # Gate deletion on an actual read-back check, not just a successful write.
        if not verify_npz(cp, sig, n_expected):
            return ("error", wfm_path,
                    "NPZ verification failed after write — raw file NOT deleted", None)

        raw_size = os.path.getsize(wfm_path)
        os.remove(wfm_path)
        return (status, wfm_path, None, raw_size)

    except Exception as e:
        return ("error", wfm_path, str(e), None)


def find_wfms(data_dir: str):
    p1    = os.path.join(data_dir, "**", f"*{EXT.lower()}")
    p2    = os.path.join(data_dir, "**", f"*{EXT.upper()}")
    files = set(glob.glob(p1, recursive=True)) | \
            set(glob.glob(p2, recursive=True))
    if not files:
        for root, _, fnames in os.walk(data_dir):
            for fn in fnames:
                if fn.lower().endswith(EXT.lower()):
                    files.add(os.path.join(root, fn))
    return sorted(files)


def parse_args():
    ap = argparse.ArgumentParser(description="Build NPZ cache from HRPPD WFM files.")
    ap.add_argument("data_dir", nargs="?", default=DATA_DIR,
                     help=f"Folder containing WFM files (default: {DATA_DIR})")
    ap.add_argument("--cache-root", default=CACHE_ROOT,
                     help=f"Fixed root folder to write all NPZ cache into (default: {CACHE_ROOT})")
    ap.add_argument("--delete-raw", action="store_true",
                     help="Delete each raw .wfm after its NPZ is written/confirmed AND "
                          "read-back verified. Off by default. Logs to "
                          "{cache_root}/deleted_raw_manifest.log")
    return ap.parse_args()


def main():
    args       = parse_args()
    data_dir   = args.data_dir
    cache_root = args.cache_root

    print(f"DATA_DIR   : {data_dir}")
    print(f"CACHE_ROOT : {cache_root}")
    if args.delete_raw:
        print("[cache] --delete-raw ENABLED: raw .wfm files will be removed "
              "after verified caching.")

    if not os.path.isdir(data_dir):
        print("[ERROR] data dir does not exist.")
        sys.exit(1)

    ensure_dir(cache_root)

    wfms = find_wfms(data_dir)
    print(f"[cache] found {len(wfms)} WFM files")

    if not wfms:
        print("[ERROR] No WFM files found. Check data_dir and naming.")
        sys.exit(1)

    written, skipped, errors = 0, 0, []
    deleted_count, deleted_bytes = 0, 0
    manifest_path = os.path.join(cache_root, "deleted_raw_manifest.log")
    manifest_fh = open(manifest_path, "a") if args.delete_raw else None

    tasks = [(w, data_dir, cache_root, args.delete_raw) for w in wfms]

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for i, (status, path, err, deleted_size) in enumerate(
                pool.imap_unordered(wfm_to_npz, tasks, chunksize=200), 1):
            if status == "written":
                written += 1
            elif status == "skip":
                skipped += 1
            elif status == "error":
                errors.append((path, err))
                print(f"[ERR] {path}: {err}")

            if deleted_size is not None:
                deleted_count += 1
                deleted_bytes += deleted_size
                if manifest_fh:
                    manifest_fh.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{path}\t{deleted_size}\n")

            if i % 500 == 0:
                print(f"[cache] {i}/{len(wfms)} processed "
                      f"(written={written}, skipped={skipped}, errors={len(errors)}, "
                      f"raw_deleted={deleted_count})")

    if manifest_fh:
        manifest_fh.close()

    print(f"[cache] done. wrote {written} new NPZ, skipped {skipped} up-to-date, "
          f"{len(errors)} errors.")

    if args.delete_raw:
        print(f"[cache] deleted {deleted_count} raw .wfm files "
              f"({deleted_bytes / 1e9:.2f} GB freed). Manifest: {manifest_path}")

    if errors:
        print(f"[cache] {len(errors)} file(s) FAILED — see [ERR] lines above.")
        sys.exit(2)


if __name__ == "__main__":
    main()