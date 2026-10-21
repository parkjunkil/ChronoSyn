"""
Builds cgnf90_features.npy for a unique_compositions.csv, row-aligned by
construction (reads the CSV sorted by `row`, writes features in that same
order). Used to regenerate data/mpoqmd/cgnf90_features.npy locally instead
of shipping the 147 MB array in the repo -- the feature is a deterministic
function of (formula, element_embedding.json), so there's nothing to lose
by computing it on demand.

Usage:
    python scripts/build_features.py data/mpoqmd/unique_compositions.csv
    # -> writes data/mpoqmd/cgnf90_features.npy (same directory as the CSV)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chronosyn.features import cgnf_feature, load_element_embedding


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/build_features.py path/to/unique_compositions.csv", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    out_path = csv_path.parent / "cgnf90_features.npy"

    t0 = time.time()
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[]).sort_values("row").reset_index(drop=True)
    print(f"loaded {len(df):,} compositions from {csv_path} ({time.time()-t0:.1f}s)", flush=True)

    element_embedding = load_element_embedding()
    emb_dim = len(next(iter(element_embedding.values())))

    t0 = time.time()
    features = np.zeros((len(df), emb_dim), dtype=np.float32)
    n_dropped = 0
    for i, formula in enumerate(df["reduced_formula"]):
        v = cgnf_feature(formula, element_embedding)
        if v is None:
            n_dropped += 1
            continue
        features[i] = v
    if n_dropped:
        print(f"warning: {n_dropped} composition(s) could not be featurized "
              f"(left as all-zero rows)", flush=True)

    np.save(out_path, features)
    print(f"saved {out_path}  shape={features.shape}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
