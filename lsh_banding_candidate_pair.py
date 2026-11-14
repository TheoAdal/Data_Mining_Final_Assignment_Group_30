"""Stage 3: LSH banding -> candidate pairs.

Usage:
    python lsh_banding_candidate_pair.py 42                        # seed 42, h = 100, b = 20, r = 5
    python lsh_banding_candidate_pair.py 42 --h 120 --b 20 --r 6   # other parameters
    python lsh_banding_candidate_pair.py 42 --max-bucket 500       # skip buckets with more users

Reads output/signatures_h{h}_seed{seed}.npy written by stage 2 (or computes the
signatures if that file is missing) and saves the candidates to
output/candidates_h{h}_b{b}_r{r}_seed{seed}.npy.

Conventions (same as stages 1 and 2):
- sig has shape (h, n_users), one column per user, users 0-indexed.
- A candidate pair is a row (u, v) of an int32 array with u < v.

How banding works here:
- The signature rows are cut into b bands of r rows each (b * r <= h; leftover
  rows are not used for banding).
- Within one band, two users land in the same bucket only if all r of their
  values are identical. Instead of hashing, users are sorted by their r values
  (np.lexsort); identical columns then end up next to each other, so every
  bucket is a contiguous run. This is exact: no false collisions from hashing.
- Every pair inside a bucket becomes a candidate. Buckets larger than
  max_bucket are skipped (k users give k(k-1)/2 pairs, see hint 4).
- A pair that collides in several bands is reported only once, in the first
  band where it appears. This is checked directly on the signatures (does the
  pair also collide in an earlier band, in a bucket that was not skipped?)
  instead of keeping a growing list of all pairs seen so far, so memory stays
  flat across bands.
"""
import argparse
import os
import time

import numpy as np

from load_data import DEFAULT_DATA, DEFAULT_MATRIX, load_data, mem_gb
from signature_generation import compute_signatures

HERE = os.path.dirname(os.path.abspath(__file__))


def candidate_probability(s, b, r):
    """Probability that a pair with Jaccard similarity s becomes a candidate."""
    return 1.0 - (1.0 - s ** r) ** b


def band_buckets(band):
    """Group users with identical columns in one band.

    band: array (r, n_users). Returns (order, starts, sizes): users sorted so that
    each bucket is the contiguous slice order[start:start + size].
    """
    order = np.lexsort(band)                      # sort users by their r values
    sorted_band = band[:, order]
    changed = np.any(sorted_band[:, 1:] != sorted_band[:, :-1], axis=0)
    starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
    sizes = np.diff(np.concatenate((starts, [band.shape[1]])))
    return order, starts, sizes


def pairs_in_buckets(order, starts, sizes, max_bucket):
    """All pairs (u, v), u < v, inside buckets of size 2..max_bucket.

    Loops only over the distinct bucket sizes; all buckets of the same size k are
    handled at once with np.triu_indices(k, 1). Returns (pairs, n_skipped).
    """
    chunks = []
    skipped = int(np.sum(sizes > max_bucket))
    for k in np.unique(sizes):
        if k < 2 or k > max_bucket:
            continue
        sel = starts[sizes == k]
        members = order[sel[:, None] + np.arange(k)]   # (n_buckets, k)
        i, j = np.triu_indices(k, 1)
        u = members[:, i].ravel().astype(np.int32)
        v = members[:, j].ravel().astype(np.int32)
        chunks.append(np.stack((np.minimum(u, v), np.maximum(u, v)), axis=1))
        del members, u, v
    if not chunks:
        return np.empty((0, 2), dtype=np.int32), skipped
    return np.concatenate(chunks), skipped


def first_band_mask(sig, pairs, band, r, emitted, chunk=200_000):
    """True for pairs that were not already reported in a band before `band`.

    A pair was reported in band j if the two users collide there (all r rows of
    band j equal) and their bucket was not skipped; emitted[j, u] is True when
    user u's bucket in band j had at most max_bucket users. A pair that only
    collided in skipped buckets so far is still new.
    """
    keep = np.ones(len(pairs), dtype=bool)
    if band == 0:
        return keep
    earlier = sig[:band * r]
    for s in range(0, len(pairs), chunk):
        p = pairs[s:s + chunk]
        equal = earlier[:, p[:, 0]] == earlier[:, p[:, 1]]          # (band * r, n)
        collided = equal.reshape(band, r, -1).all(axis=1)            # (band, n)
        reported = collided & emitted[:band, p[:, 0]]                # same bucket, so u's size is enough
        keep[s:s + chunk] = ~reported.any(axis=0)
    return keep


def iter_band_candidates(sig, b, r, max_bucket=1000):
    """Yield (band, new_pairs, stats) for each band, without repeats across bands.

    Yielding band by band lets stage 4 verify and write pairs as soon as they are
    found, so a run that is stopped at 30 minutes still has a result file.
    """
    h, n_users = sig.shape
    if b * r > h:
        raise ValueError(f"b * r = {b * r} is larger than the signature length h = {h}")

    emitted = np.zeros((b, n_users), dtype=bool)    # emitted[j, u]: u's bucket in band j was not skipped
    for band in range(b):
        rows = sig[band * r:(band + 1) * r]
        order, starts, sizes = band_buckets(rows)
        emitted[band, order] = np.repeat(sizes <= max_bucket, sizes)
        # every user is in exactly one bucket per band, so a pair occurs at most once here
        pairs, skipped = pairs_in_buckets(order, starts, sizes, max_bucket)
        new_pairs = pairs[first_band_mask(sig, pairs, band, r, emitted)]

        stats = {
            "buckets_2plus": int(np.sum(sizes >= 2)),
            "largest_bucket": int(sizes.max()),
            "skipped_buckets": skipped,
            "pairs_in_band": int(len(pairs)),
            "new_pairs": int(len(new_pairs)),
        }
        del pairs, order, starts, sizes
        yield band, new_pairs, stats


def candidate_pairs(sig, b, r, max_bucket=1000, verbose=False):
    """All candidate pairs from all bands, as an int32 array (n, 2) with u < v."""
    chunks = []
    for band, new_pairs, stats in iter_band_candidates(sig, b, r, max_bucket):
        chunks.append(new_pairs)
        if verbose:
            print(f"band {band:3d}: buckets>=2 {stats['buckets_2plus']:7,d}  "
                  f"largest {stats['largest_bucket']:6,d}  skipped {stats['skipped_buckets']:3d}  "
                  f"pairs {stats['pairs_in_band']:9,d}  new {stats['new_pairs']:9,d}")
    return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int32)


def signature_agreement(sig, pairs, chunk=200_000):
    """Fraction of equal signature values for each pair: an estimate of Jaccard.

    Cheap compared with the exact Jaccard, so stage 4 can use it as a pre-filter.
    """
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), chunk):
        p = pairs[s:s + chunk]
        out[s:s + chunk] = (sig[:, p[:, 0]] == sig[:, p[:, 1]]).mean(axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage 3: LSH banding")
    parser.add_argument("seed", type=int, help="random seed (must match stage 2)")
    parser.add_argument("--h", type=int, default=100, help="signature length")
    parser.add_argument("--b", type=int, default=20, help="number of bands")
    parser.add_argument("--r", type=int, default=5, help="rows per band")
    parser.add_argument("--max-bucket", type=int, default=1000, help="skip larger buckets")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="matrix saved by stage 1")
    parser.add_argument("--data", default=DEFAULT_DATA, help="raw data, used if matrix is missing")
    parser.add_argument("--out", default=os.path.join(HERE, "output"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sig_path = os.path.join(args.out, f"signatures_h{args.h}_seed{args.seed}.npy")
    if os.path.exists(sig_path):
        sig = np.load(sig_path)
    else:
        print(f"{sig_path} not found, computing signatures (run signature_generation.py to save them)")
        if os.path.exists(args.matrix):
            from scipy.sparse import load_npz
            M = load_npz(args.matrix).tocsr()
        else:
            M = load_data(args.data)
        sig = compute_signatures(M, h=args.h, seed=args.seed)
        del M

    print("=== Stage 3: banding ===")
    print(f"h = {args.h}, b = {args.b}, r = {args.r}, max bucket = {args.max_bucket}")
    print(f"approx. threshold (1/b)^(1/r) = {(1 / args.b) ** (1 / args.r):.3f}")
    print("P(candidate | Jaccard s):  " + "  ".join(
        f"s={s:.1f}: {candidate_probability(s, args.b, args.r):.3f}"
        for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)))
    print()

    t0 = time.perf_counter()
    cand = candidate_pairs(sig, args.b, args.r, args.max_bucket, verbose=True)
    t_band = time.perf_counter() - t0

    agree = signature_agreement(sig, cand)
    print(f"\ncandidate pairs:  {len(cand):,}")
    print(f"banding time:     {t_band:.1f} s")
    print(f"process memory:   {mem_gb():.2f} GB")
    for t in (0.3, 0.4, 0.5):
        print(f"candidates with signature agreement >= {t}: {int(np.sum(agree >= t)):,}")

    path = os.path.join(args.out, f"candidates_h{args.h}_b{args.b}_r{args.r}_seed{args.seed}.npy")
    np.save(path, cand)
    print(f"\nsaved candidates to {path}")


if __name__ == "__main__":
    main()
