"""Experiments for the report: LSH parameters (h, b, r) across several random seeds.

Put this file in the repo root, next to load_data.py, signature_generation.py,
lsh_banding_candidate_pair.py and verify_pairs.py.

Usage:
    python run_experiments.py                                   # default grid, seeds 1-5
    python run_experiments.py --seeds 1 2 3                     # fewer seeds
    python run_experiments.py --configs 100,20,5 120,30,4       # own (h,b,r) settings
    python run_experiments.py --matrix output/user_movie_csr.npz   # skip the raw load
    python run_experiments.py --plot-only                       # only redraw plots from the CSV

For every seed the signatures are computed once per signature length h and reused
for all (b, r) settings with that h, so the grid runs much faster than calling
verify_pairs.py once per setting. Every run uses exactly the same steps as
verify_pairs.py (banding, signature pre-filter, exact Jaccard check), so the
numbers in the report match the submitted program.

Outputs (in --out, default output/experiments):
- runs.csv        one row per (seed, h, b, r): pairs found, candidates, checks, times, memory
- pairs.csv       every similar pair found in any run, with its Jaccard similarity
- summary.csv     mean / std / min / max per (h, b, r) over the seeds, plus recall
- s_curves.png    theoretical P(candidate | Jaccard) for each setting
- pairs_per_config.png   pairs found per setting (mean and spread over the seeds)
- jaccard_hist.png       Jaccard similarity of all similar pairs found

Recall: the brute-force answer is not available, so recall is measured against the
union of all pairs found by all runs (an upper bound on the true recall).
"""
import argparse
import csv
import os
import time

import numpy as np
from scipy.sparse import load_npz

from load_data import DEFAULT_DATA, load_data, jaccard, mem_gb
from signature_generation import compute_signatures
from lsh_banding_candidate_pair import candidate_probability, iter_band_candidates, signature_agreement

HERE = os.path.dirname(os.path.abspath(__file__))

# Signature lengths 100-150 (hint 3). Approximate threshold (1/b)^(1/r) in brackets:
DEFAULT_CONFIGS = [
    (100, 20, 5),   # 0.55  current default of verify_pairs.py
    (100, 25, 4),   # 0.45
    (120, 30, 4),   # 0.43
    (150, 37, 4),   # 0.41
    (150, 30, 5),   # 0.51
    (120, 40, 3),   # 0.29  low threshold, many candidates
]

RUN_FIELDS = ["seed", "h", "b", "r", "approx_threshold", "p_candidate_at_0.5",
              "pairs_found", "candidates", "after_prefilter", "exact_checks",
              "skipped_buckets", "largest_bucket", "time_signatures_s",
              "time_banding_s", "time_verification_s", "est_full_runtime_s",
              "memory_gb", "stopped_by_time"]


def parse_config(text):
    h, b, r = (int(x) for x in text.split(","))
    if b * r > h:
        raise argparse.ArgumentTypeError(f"b * r = {b * r} is larger than h = {h} in '{text}'")
    return h, b, r


def default_data_path():
    next_to_script = os.path.join(HERE, "user_movie_rating.npy")
    if not os.path.exists(DEFAULT_DATA) and os.path.exists(next_to_script):
        return next_to_script
    return DEFAULT_DATA


def run_one(M, sig, b, r, max_bucket, threshold, prefilter, time_limit):
    """Banding + verification exactly as in verify_pairs.py, without writing result.txt."""
    start = time.perf_counter()
    found = []                                   # (u, v, jaccard), 0-based
    stats_total = {"candidates": 0, "after_prefilter": 0, "exact_checks": 0,
                   "skipped_buckets": 0, "largest_bucket": 0}
    time_verifying = 0.0
    stopped = False

    for _, band_pairs, stats in iter_band_candidates(sig, b, r, max_bucket):
        t_verify = time.perf_counter()
        stats_total["candidates"] += len(band_pairs)
        stats_total["skipped_buckets"] += stats["skipped_buckets"]
        stats_total["largest_bucket"] = max(stats_total["largest_bucket"], stats["largest_bucket"])

        agreement = signature_agreement(sig, band_pairs)
        to_check = band_pairs[agreement >= prefilter]
        stats_total["after_prefilter"] += len(to_check)
        for u, v in to_check:
            if time.perf_counter() - start > time_limit:
                stopped = True
                break
            similarity = jaccard(M, int(u), int(v))
            stats_total["exact_checks"] += 1
            if similarity > threshold:
                found.append((int(u), int(v), similarity))
        time_verifying += time.perf_counter() - t_verify
        if stopped:
            break

    total = time.perf_counter() - start
    return found, stats_total, total - time_verifying, time_verifying, stopped


def append_rows(path, fields, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise(out_dir):
    runs = read_csv(os.path.join(out_dir, "runs.csv"))
    pairs = read_csv(os.path.join(out_dir, "pairs.csv")) if os.path.exists(os.path.join(out_dir, "pairs.csv")) else []
    union = {(p["user_1"], p["user_2"]) for p in pairs}
    found_by_run = {}
    for p in pairs:
        found_by_run.setdefault((p["seed"], p["h"], p["b"], p["r"]), set()).add((p["user_1"], p["user_2"]))

    configs = sorted({(int(x["h"]), int(x["b"]), int(x["r"])) for x in runs})
    summary = []
    for h, b, r in configs:
        rows = [x for x in runs if (int(x["h"]), int(x["b"]), int(x["r"])) == (h, b, r)]
        n = np.array([int(x["pairs_found"]) for x in rows])
        rt = np.array([float(x["est_full_runtime_s"]) for x in rows])
        recall = [len(found_by_run.get((x["seed"], x["h"], x["b"], x["r"]), set())) / len(union)
                  for x in rows] if union else [float("nan")]
        summary.append({
            "h": h, "b": b, "r": r,
            "approx_threshold": round((1 / b) ** (1 / r), 3),
            "p_candidate_at_0.5": round(candidate_probability(0.5, b, r), 3),
            "seeds": len(rows),
            "pairs_mean": round(n.mean(), 1), "pairs_std": round(n.std(ddof=1), 1) if len(n) > 1 else 0.0,
            "pairs_min": int(n.min()), "pairs_max": int(n.max()),
            "recall_vs_union_mean": round(float(np.mean(recall)), 3),
            "candidates_mean": round(np.mean([int(x["candidates"]) for x in rows])),
            "runtime_mean_s": round(rt.mean(), 1), "runtime_max_s": round(rt.max(), 1),
            "memory_max_gb": round(max(float(x["memory_gb"]) for x in rows), 2),
        })
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n=== Summary over seeds ({len(union)} distinct similar pairs found in total) ===")
    print(f"{'h':>4} {'b':>3} {'r':>2} {'thr':>5} {'P(.5)':>6} {'pairs mean±std':>16} "
          f"{'min-max':>9} {'recall':>7} {'runtime':>9}")
    for s in summary:
        print(f"{s['h']:4d} {s['b']:3d} {s['r']:2d} {s['approx_threshold']:5.2f} {s['p_candidate_at_0.5']:6.2f} "
              f"{s['pairs_mean']:9.1f} ± {s['pairs_std']:4.1f} {s['pairs_min']:4d}-{s['pairs_max']:<4d} "
              f"{s['recall_vs_union_mean']:7.2f} {s['runtime_mean_s']:7.0f} s")
    print(f"saved {path}")
    return summary, pairs


def make_plots(out_dir, summary, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"h={s['h']}\nb={s['b']}, r={s['r']}" for s in summary]

    # 1) S-curves
    s_grid = np.linspace(0, 1, 201)
    plt.figure(figsize=(6, 4))
    for s in summary:
        plt.plot(s_grid, candidate_probability(s_grid, s["b"], s["r"]),
                 label=f"b={s['b']}, r={s['r']} (h={s['h']})")
    plt.axvline(0.5, color="k", ls="--", lw=1)
    plt.xlabel("Jaccard similarity")
    plt.ylabel("P(pair becomes a candidate)")
    plt.title("LSH S-curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "s_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) pairs found per setting, one dot per seed
    runs = read_csv(os.path.join(out_dir, "runs.csv"))
    plt.figure(figsize=(max(6, 1.2 * len(summary)), 4))
    for i, s in enumerate(summary):
        vals = [int(x["pairs_found"]) for x in runs
                if (int(x["h"]), int(x["b"]), int(x["r"])) == (s["h"], s["b"], s["r"])]
        plt.bar(i, np.mean(vals), color="lightgray", edgecolor="gray")
        plt.scatter(np.full(len(vals), i), vals, color="k", s=12, zorder=3)
    plt.xticks(range(len(summary)), labels, fontsize=8)
    plt.ylabel("similar pairs found (J > 0.5)")
    plt.title("Pairs found per setting (bar = mean, dots = seeds)")
    plt.savefig(os.path.join(out_dir, "pairs_per_config.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Jaccard of all distinct pairs found
    if pairs:
        js = {}
        for p in pairs:
            js[(p["user_1"], p["user_2"])] = float(p["jaccard"])
        plt.figure(figsize=(6, 4))
        plt.hist(list(js.values()), bins=30, color="steelblue", edgecolor="white")
        plt.xlabel("Jaccard similarity")
        plt.ylabel("number of pairs")
        plt.title(f"Similar pairs found over all runs (n = {len(js)})")
        plt.savefig(os.path.join(out_dir, "jaccard_hist.png"), dpi=200, bbox_inches="tight")
        plt.close()
    print(f"plots saved in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="LSH parameter experiments for the report")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--configs", type=parse_config, nargs="+", default=DEFAULT_CONFIGS,
                        help="settings as h,b,r (e.g. 100,20,5)")
    parser.add_argument("--max-bucket", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prefilter", type=float, default=0.3)
    parser.add_argument("--time-limit", type=float, default=25 * 60,
                        help="seconds per run for banding + verification")
    parser.add_argument("--data", default=None)
    parser.add_argument("--matrix", default=None, help="saved stage 1 matrix (faster)")
    parser.add_argument("--out", default=os.path.join(HERE, "output", "experiments"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.plot_only:
        t0 = time.perf_counter()
        M = load_npz(args.matrix).tocsr() if args.matrix else load_data(args.data or default_data_path())
        time_load = time.perf_counter() - t0
        print(f"data: {M.shape[0]:,} users x {M.shape[1]:,} movies, load {time_load:.1f} s")

        runs_path = os.path.join(args.out, "runs.csv")
        pairs_path = os.path.join(args.out, "pairs.csv")
        for seed in args.seeds:
            for h in sorted({c[0] for c in args.configs}):
                t_sig = time.perf_counter()
                sig = compute_signatures(M, h=h, seed=seed)
                time_sig = time.perf_counter() - t_sig
                for (hh, b, r) in args.configs:
                    if hh != h:
                        continue
                    found, st, t_band, t_ver, stopped = run_one(
                        M, sig, b, r, args.max_bucket, args.threshold, args.prefilter, args.time_limit)
                    row = {
                        "seed": seed, "h": h, "b": b, "r": r,
                        "approx_threshold": round((1 / b) ** (1 / r), 3),
                        "p_candidate_at_0.5": round(candidate_probability(0.5, b, r), 3),
                        "pairs_found": len(found), **st,
                        "time_signatures_s": round(time_sig, 1),
                        "time_banding_s": round(t_band, 1),
                        "time_verification_s": round(t_ver, 1),
                        # what verify_pairs.py would need for this setting on this machine
                        "est_full_runtime_s": round(time_load + time_sig + t_band + t_ver, 1),
                        "memory_gb": round(mem_gb(), 2),
                        "stopped_by_time": stopped,
                    }
                    append_rows(runs_path, RUN_FIELDS, [row])
                    append_rows(pairs_path, ["seed", "h", "b", "r", "user_1", "user_2", "jaccard"],
                                [{"seed": seed, "h": h, "b": b, "r": r, "user_1": u + 1, "user_2": v + 1,
                                  "jaccard": round(j, 4)} for u, v, j in found])
                    print(f"seed {seed}  h={h:3d} b={b:3d} r={r:2d}: {len(found):4d} pairs, "
                          f"{st['candidates']:,} candidates, {st['exact_checks']:,} checks, "
                          f"~{row['est_full_runtime_s']:.0f} s full run"
                          + ("  (STOPPED BY TIME LIMIT)" if stopped else ""))
                del sig

    summary, pairs = summarise(args.out)
    make_plots(args.out, summary, pairs)


if __name__ == "__main__":
    main()
