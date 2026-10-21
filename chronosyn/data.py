"""
Dataset for ChronoSyn's composition-only classifier.

Wraps a `unique_compositions.csv` (columns: row, reduced_formula, is_P, year,
...) and a row-aligned `cgnf90_features.npy` ([N, 90] float32), the
stoichiometric composition feature of Jang et al. (Matter 2024,
"Synthesizability of materials stoichiometry using semi-supervised
learning"): CGNF(composition) = sum_i(element_embedding[el_i] * amount_i) /
sum_i(amount_i), using the pretrained-CGCNN element embedding table in
element_embedding.json (100 elements x 90-dim).

Two prebuilt (csv, features) pairs are provided under data/: `mponly`
(unique compositions from the Materials Project, is_P = has an ICSD-
confirmed structure among its MP entries) and `mpoqmd` (MP + OQMD union,
same labeling rule). See the top-level README for exact provenance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_MPONLY = REPO_ROOT / "data" / "mponly"
DATA_MPOQMD = REPO_ROOT / "data" / "mpoqmd"

# Real reporting years of P-labeled compositions span exactly this range in
# both datasets; U has no year (0.0 placeholder, masked out of the
# chronology loss by label).
YEAR_MIN = 1905.0
YEAR_MAX = 2024.0


class CompositionDataset(Dataset):
    def __init__(self, csv_path: str | Path = DATA_MPONLY / "unique_compositions.csv",
                 features_path: str | Path = DATA_MPONLY / "cgnf90_features.npy"):
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[]).sort_values("row").reset_index(drop=True)
        features = np.load(features_path)
        assert len(df) == len(features), f"row count mismatch: {len(df)} rows vs {len(features)} feature vectors"

        self.formulas = df["reduced_formula"].to_numpy()
        self.labels = df["is_P"].astype(bool).to_numpy().astype(np.int64)
        self.features = features.astype(np.float32)

        year_raw = pd.to_numeric(df["year"], errors="coerce").to_numpy()
        year_norm = (year_raw - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
        year_norm[self.labels == 0] = 0.0
        assert not np.isnan(year_norm[self.labels == 1]).any(), "P-labeled composition with no year"
        self.year_norm = year_norm.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.features[i]),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.year_norm[i], dtype=torch.float32),
            str(self.formulas[i]),
        )


def stratified_split(labels: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Random train/test split, stratified by label so the P/U ratio is preserved in both."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def temporal_split(labels: np.ndarray, year_norm: np.ndarray, year_cutoff_norm: float,
                    u_test_frac: float = 0.2, seed: int = 42):
    """Split P by real synthesis year (test = P discovered AFTER the cutoff --
    a genuine future-extrapolation test, unlike stratified_split's random
    80/20). U has no year, so it's split the same random way stratified_split
    would: shuffled, u_test_frac held out."""
    rng = np.random.default_rng(seed)
    p_idx = np.where(labels == 1)[0]
    u_idx = np.where(labels == 0)[0]

    test_p = p_idx[year_norm[p_idx] > year_cutoff_norm]
    train_p = p_idx[year_norm[p_idx] <= year_cutoff_norm]

    u_idx = u_idx.copy()
    rng.shuffle(u_idx)
    n_test_u = int(round(len(u_idx) * u_test_frac))
    test_u = u_idx[:n_test_u]
    train_u = u_idx[n_test_u:]

    train_idx = np.concatenate([train_p, train_u])
    test_idx = np.concatenate([test_p, test_u])
    return train_idx, test_idx


def estimate_prior(labels: np.ndarray) -> float:
    """Lower-bound estimate of the class prior pi = P(y=1) (observed P fraction)."""
    return float((labels == 1).mean())
