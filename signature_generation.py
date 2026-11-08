#Stage 2: Signature table.

#python signature_generation.py 42                # seed 42, h = 100
#python signature_generation.py 42 --h 150        # different signature length
#python signature_generation.py 42 --plot         # also save agreement-vs-Jaccard plot
#python signature_generation.py 42 --timing       # also time h = 50, 100, 150


import argparse
import os
import time

import numpy as np
from scipy.sparse import load_npz

from load_data import DEFAULT_DATA, DEFAULT_MATRIX, load_data, jaccard, mem_gb

HERE = os.path.dirname(os.path.abspath(__file__))


def compute_signatures(M, h, seed):
    #Minhash signature table, shape (h, n_users), dtype int16.

    # For each of h random permutations of the movies, a user's minhash value is the
    # smallest permuted position among the movies they rated.
    
    n_users, n_movies = M.shape
    if np.any(np.diff(M.indptr) == 0):
        raise ValueError("some users have no ratings; reduceat would give wrong values")

    rng = np.random.default_rng(seed)
    starts = M.indptr[:-1]
    sig = np.empty((h, n_users), dtype=np.int16)   # n_movies = 17,770 < 32,767
    for i in range(h):
        perm = rng.permutation(n_movies).astype(np.int16)
        sig[i] = np.minimum.reduceat(perm[M.indices], starts)
    return sig


def check_against_loop(M, seed, n_users=200, h=20):
    #Vectorised signatures must equal a slow, obvious per-user loop.
    
    sub = M[:n_users]
    fast = compute_signatures(sub, h=h, seed=seed)
    rng = np.random.default_rng(seed)
    slow = np.empty_like(fast)
    for i in range(h):
        perm = rng.permutation(M.shape[1])
        for u in range(n_users):
            slow[i, u] = perm[sub.indices[sub.indptr[u]:sub.indptr[u + 1]]].min()
    return np.array_equal(fast, slow)


def agreement_vs_jaccard(M, sig, n_pairs=2000):
    #Signature agreement and true Jaccard for random user pairs.
    
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, M.shape[0], size=(n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    true_js = np.array([jaccard(M, u, v) for u, v in pairs])
    est_js = (sig[:, pairs[:, 0]] == sig[:, pairs[:, 1]]).mean(axis=0)
    return true_js, est_js


def main():
    parser = argparse.ArgumentParser(description="Stage 2: minhash signatures")
    parser.add_argument("seed", type=int, help="random seed")
    parser.add_argument("--h", type=int, default=100, help="signature length")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="matrix saved by stage 1")
    parser.add_argument("--data", default=DEFAULT_DATA, help="raw data, used if matrix is missing")
    parser.add_argument("--out", default=os.path.join(HERE, "output"))
    parser.add_argument("--plot", action="store_true", help="save agreement-vs-Jaccard plot")
    parser.add_argument("--timing", action="store_true", help="time h = 50, 100, 150")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(args.matrix):
        M = load_npz(args.matrix).tocsr()
    else:
        print("matrix not found, building it from the raw data (run load_data.py to save it)")
        M = load_data(args.data)

    print("=== Stage 2: signatures ===")
    print("vectorised == loop check:", check_against_loop(M, args.seed))

    t0 = time.perf_counter()
    sig = compute_signatures(M, h=args.h, seed=args.seed)
    t_sig = time.perf_counter() - t0
    print(f"shape {sig.shape}, dtype {sig.dtype}, {sig.nbytes / 1e6:.0f} MB")
    print(f"time: {t_sig:.1f} s total, {t_sig / args.h:.3f} s per permutation")
    print(f"process memory:   {mem_gb():.2f} GB")

    true_js, est_js = agreement_vs_jaccard(M, sig)
    print(f"mean |signature agreement - true Jaccard| over random pairs: "
          f"{np.abs(est_js - true_js).mean():.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lim = max(true_js.max(), est_js.max()) + 0.05
        plt.scatter(true_js, est_js, s=4, alpha=0.5)
        plt.plot([0, lim], [0, lim], "r--", lw=1)
        plt.xlabel("true Jaccard")
        plt.ylabel("signature agreement")
        plt.title(f"h = {args.h}, seed = {args.seed}")
        path = os.path.join(args.out, f"agreement_h{args.h}_seed{args.seed}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print("plot saved:", path)

    if args.timing:
        print("\n=== Runtime vs signature length ===")
        for h in (50, 100, 150):
            t0 = time.perf_counter()
            compute_signatures(M, h=h, seed=args.seed)
            print(f"h = {h:3d}: {time.perf_counter() - t0:5.1f} s")

    np.save(os.path.join(args.out, f"signatures_h{args.h}_seed{args.seed}.npy"), sig)
    np.save(os.path.join(args.out, "signatures_small.npy"), sig[:, :5000])
    print(f"\nsaved signatures to {args.out}")


if __name__ == "__main__":
    main()
