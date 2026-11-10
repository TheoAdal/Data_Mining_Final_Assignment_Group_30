"""Stage 4: check candidate pairs against their real Jaccard similarity, write result.txt.

Usage:
    python verify_pairs.py 42                           # full pipeline, seed 42
    python verify_pairs.py 42 --h 100 --b 20 --r 5      # LSH parameters
    python verify_pairs.py 42 --prefilter 0             # exact check on every candidate
    python verify_pairs.py 42 --matrix output/user_movie_csr.npz   # skip the raw load (dev runs)

Runs the whole pipeline: stage 1 (sparse matrix), stage 2 (signatures), stage 3
(banding, band by band) and then, for the new candidates of each band:
- pre-filter: drop pairs whose signature agreement is below --prefilter
- exact check: Jaccard on the two sparse rows, keep the pair only if it is > threshold
- append "u1,u2" (original 1-based IDs, u1 < u2) to result.txt, opening and closing
  the file for every pair (hint 6)

LSH only gives candidates, so nothing reaches result.txt without the exact check.
The run stops itself at --time-limit seconds; result.txt is valid at any moment.
"""
import argparse
import os
import time

import numpy as np
from scipy.sparse import load_npz

from load_data import DEFAULT_DATA, load_data, jaccard, mem_gb
from signature_generation import compute_signatures
from lsh_banding_candidate_pair import candidate_probability, iter_band_candidates, signature_agreement

HERE = os.path.dirname(os.path.abspath(__file__))


def default_data_path():
    # the graders may put the .npy next to the script instead of in data/
    next_to_script = os.path.join(HERE, "user_movie_rating.npy")
    if not os.path.exists(DEFAULT_DATA) and os.path.exists(next_to_script):
        return next_to_script
    return DEFAULT_DATA


def append_pair(result_path, user_a, user_b):
    with open(result_path, "a") as result_file:
        result_file.write(f"{user_a + 1},{user_b + 1}\n")


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Stage 4: verify candidates, write result.txt")
    parser.add_argument("seed", type=int, help="random seed")
    parser.add_argument("--h", type=int, default=100, help="signature length")
    parser.add_argument("--b", type=int, default=20, help="number of bands")
    parser.add_argument("--r", type=int, default=5, help="rows per band")
    parser.add_argument("--max-bucket", type=int, default=1000, help="skip larger buckets")
    parser.add_argument("--threshold", type=float, default=0.5, help="keep pairs with Jaccard > this")
    parser.add_argument("--prefilter", type=float, default=0.3,
                        help="minimum signature agreement before the exact check")
    parser.add_argument("--time-limit", type=float, default=28 * 60, help="seconds from start")
    parser.add_argument("--data", default=None, help="raw .npy (default: data/ or next to script)")
    parser.add_argument("--matrix", default=None, help="saved stage 1 matrix, for faster dev runs")
    parser.add_argument("--result", default=os.path.join(HERE, "result.txt"))
    parser.add_argument("--out", default=os.path.join(HERE, "output"))
    args = parser.parse_args()
    deadline = start_time + args.time_limit
    os.makedirs(args.out, exist_ok=True)

    if args.matrix:
        M = load_npz(args.matrix).tocsr()
    else:
        M = load_data(args.data or default_data_path())
    time_load = time.perf_counter() - start_time
    print(f"stage 1: {M.shape[0]:,} users x {M.shape[1]:,} movies, {M.nnz:,} ratings, {time_load:.1f} s")

    time_before_signatures = time.perf_counter()
    sig = compute_signatures(M, h=args.h, seed=args.seed)
    time_signatures = time.perf_counter() - time_before_signatures
    print(f"stage 2: signatures {sig.shape}, {time_signatures:.1f} s")

    print(f"stage 3/4: b = {args.b}, r = {args.r}, threshold ~ {(1 / args.b) ** (1 / args.r):.3f}, "
          f"P(candidate | J = 0.5) = {candidate_probability(0.5, args.b, args.r):.3f}")

    open(args.result, "w").close()      # empty result file for this run, then append pair by pair

    found_pairs = []                    # (user_a, user_b, jaccard, signature agreement), 0-based
    total_candidates = 0
    total_checked = 0
    total_skipped_buckets = 0
    time_verifying = 0.0
    stopped_by_time = False
    time_before_bands = time.perf_counter()

    for band, band_pairs, stats in iter_band_candidates(sig, args.b, args.r, args.max_bucket):
        time_before_verify = time.perf_counter()
        total_candidates += len(band_pairs)
        total_skipped_buckets += stats["skipped_buckets"]

        agreement = signature_agreement(sig, band_pairs)
        passed = agreement >= args.prefilter
        pairs_to_check = band_pairs[passed]
        agreement_to_check = agreement[passed]
        most_promising_first = np.argsort(-agreement_to_check, kind="stable")

        found_in_band = 0
        for index in most_promising_first:
            if time.perf_counter() > deadline:
                stopped_by_time = True
                break
            user_a, user_b = int(pairs_to_check[index, 0]), int(pairs_to_check[index, 1])
            similarity = jaccard(M, user_a, user_b)
            total_checked += 1
            if similarity > args.threshold:
                append_pair(args.result, user_a, user_b)
                found_pairs.append((user_a, user_b, similarity, float(agreement_to_check[index])))
                found_in_band += 1

        time_verifying += time.perf_counter() - time_before_verify
        elapsed = time.perf_counter() - start_time
        print(f"band {band:3d}: new candidates {len(band_pairs):9,d}  after pre-filter {len(pairs_to_check):7,d}  "
              f"similar {found_in_band:4d}  total {len(found_pairs):5d}  "
              f"skipped buckets {stats['skipped_buckets']:3d}  largest {stats['largest_bucket']:6,d}  {elapsed:6.0f} s")

        if stopped_by_time or time.perf_counter() > deadline:
            stopped_by_time = True
            print(f"time limit of {args.time_limit:.0f} s reached, stopping after band {band}")
            break

    time_bands_total = time.perf_counter() - time_before_bands
    total_time = time.perf_counter() - start_time

    print("\n=== Summary ===")
    print(f"seed {args.seed}, h = {args.h}, b = {args.b}, r = {args.r}, max bucket = {args.max_bucket}, "
          f"pre-filter = {args.prefilter}")
    print(f"candidate pairs:            {total_candidates:,}")
    print(f"exact Jaccard checks:       {total_checked:,}")
    print(f"similar pairs (J > {args.threshold}):  {len(found_pairs):,}")
    print(f"checked but not similar:    {total_checked - len(found_pairs):,}")
    print(f"skipped buckets (all bands): {total_skipped_buckets:,}")
    print(f"time: load {time_load:.1f} s, signatures {time_signatures:.1f} s, "
          f"banding {time_bands_total - time_verifying:.1f} s, verification {time_verifying:.1f} s, "
          f"total {total_time:.1f} s" + ("  (stopped by time limit)" if stopped_by_time else ""))
    print(f"process memory: {mem_gb():.2f} GB")
    print(f"result written to {args.result}")

    details_path = os.path.join(args.out, f"similar_pairs_seed{args.seed}_h{args.h}_b{args.b}_r{args.r}.csv")
    with open(details_path, "w") as details_file:
        details_file.write("user_1,user_2,jaccard,signature_agreement\n")
        for user_a, user_b, similarity, agree in sorted(found_pairs, key=lambda pair: -pair[2]):
            details_file.write(f"{user_a + 1},{user_b + 1},{similarity:.4f},{agree:.2f}\n")
    print(f"pairs with their similarities saved to {details_path} (for the report)")


if __name__ == "__main__":
    main()
