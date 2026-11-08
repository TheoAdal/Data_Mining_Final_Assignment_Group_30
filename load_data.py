"""Stage 1: load the ratings AND sparse user x movie matrix.

Usage:
    python load_data.py
    python load_data.py --data path/to/user_movie_rating.npy

Reads data/user_movie_rating.npy next to this script and saves the matrix to
output/user_movie_csr.npz. Other stages can also import load_data / jaccard.

Conventions:
- Users and movies are 0-indexed (original ID - 1). Add 1 back when writing result.txt.
- M is a scipy.sparse.csr_matrix, shape (n_users, n_movies), dtype bool, row = user.
"""
import argparse
import os
import time

import numpy as np
from scipy.sparse import csr_matrix, save_npz

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "user_movie_rating.npy")
DEFAULT_MATRIX = os.path.join(HERE, "output", "user_movie_csr.npz")


def load_data(path):
    """Load the (user, movie, rating) array and return a binary CSR matrix.

    Rows are users, columns are movies, True = the user rated that movie.
    IDs are shifted to start at 0. Ratings are dropped (Jaccard ignores them).
    """
    data = np.load(path)
    users = data[:, 0].astype(np.int32) - 1
    movies = data[:, 1].astype(np.int32) - 1
    del data                                    # free the big raw array early

    n_users = int(users.max()) + 1
    n_movies = int(movies.max()) + 1
    ones = np.ones(users.shape[0], dtype=np.bool_)

    M = csr_matrix((ones, (users, movies)), shape=(n_users, n_movies))
    del users, movies, ones
    M.sum_duplicates()                          # safety: one entry per (user, movie)
    return M


def jaccard(M, u, v):
    """Exact Jaccard similarity of users u and v (0-indexed). Handy for stage 4."""
    a = M.indices[M.indptr[u]:M.indptr[u + 1]]
    b = M.indices[M.indptr[v]:M.indptr[v + 1]]
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def mem_gb():
    """Current process memory in GB (NaN if psutil is not installed)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: build the sparse matrix")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_MATRIX)
    args = parser.parse_args()

    t0 = time.perf_counter()
    M = load_data(args.data)
    t_load = time.perf_counter() - t0

    n_users, n_movies = M.shape
    counts = np.diff(M.indptr)
    csr_mb = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1e6

    print("=== Stage 1: data ===")
    print(f"load time:        {t_load:.1f} s")
    print(f"users x movies:   {n_users:,} x {n_movies:,}")
    print(f"non-zeros:        {M.nnz:,}")
    print(f"CSR size:         {csr_mb:.0f} MB")
    print(f"process memory:   {mem_gb():.2f} GB")
    print(f"ratings per user: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_npz(args.out, M)
    print(f"\nsaved matrix to {args.out}")


if __name__ == "__main__":
    main()
