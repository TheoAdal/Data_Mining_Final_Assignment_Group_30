# LSH for similar Netflix users (ADM final assignment, group 30)

Finds pairs of users whose Jaccard similarity (over the sets of movies they rated) is above 0.5,
using minhashing and banding.

## Run

```
python step4_verify_pairs.py <seed>
```

This runs the whole pipeline with the settings from the report (h = 150, b = 37, r = 4,
bucket limit 5,000, pre-filter 0.4) and writes the similar pairs to `result.txt`, one pair
`u1,u2` per line with u1 < u2 (original 1-based user IDs). The file is valid at any moment,
and the program stops itself after 28 minutes.

The data file `user_movie_rating.npy` is read from `data/` or from the folder of the scripts.
Another location can be given with `--data path/to/user_movie_rating.npy`.

Requirements: Python 3, NumPy, SciPy (`psutil` is optional, only for the memory report).

## Files

| File | Step |
|---|---|
| `step1_load_data.py` | Loads the ratings into a sparse user x movie CSR matrix (Section 1) |
| `step2_minhash.py` | Minhash signatures with random permutations of the movies (Section 2) |
| `step3_banding.py` | Banding: candidate pairs per band, each pair only once (Section 3) |
| `step4_verify_pairs.py` | Pre-filter, exact Jaccard check and output; runs all steps (Section 4) |

`step4_verify_pairs.py` is the only file that needs to be run; it imports the other three.

## Options

`python step4_verify_pairs.py <seed> --h 150 --b 37 --r 4 --max-bucket 5000 --prefilter 0.4`
(these are the defaults). `--result` sets the output file, `--time-limit` the stop time in seconds.
