"""
Score new compositions with the pretrained ChronoSyn (MP-only, 100-bag
PU-bagging ensemble, h_fea_len=450, n_h=5).

Usage:
    from chronosyn.predict import ChronoSynEnsemble
    model = ChronoSynEnsemble.load()
    scores = model.predict(["Na7PS6", "Fe2O3", "Unobtainium3"])
    # -> {"Na7PS6": 0.993, "Fe2O3": 0.999}  (unparseable/out-of-vocab formulas are omitted)

Or from the command line:
    python -m chronosyn.predict Na7PS6 Fe2O3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .features import cgnf_features, load_element_embedding
from .model import CGNFNet

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints" / "chronosyn_mponly"
N_BAGS = 100
H_FEA_LEN, N_H = 450, 5


class ChronoSynEnsemble:
    def __init__(self, bags: list[CGNFNet], device: torch.device):
        self.bags = bags
        self.device = device
        self._element_embedding = load_element_embedding()

    @classmethod
    def load(cls, ckpt_dir: str | Path = DEFAULT_CKPT_DIR, n_bags: int = N_BAGS,
             device: str | torch.device = "cpu"):
        device = torch.device(device)
        ckpt_dir = Path(ckpt_dir)
        bags = []
        for b in range(n_bags):
            model = CGNFNet(feat_dim=90, h_fea_len=H_FEA_LEN, n_h=N_H, dropout=0.5,
                             year_head=True, year_mode="ranking", year_head_dim=8).to(device)
            state = torch.load(ckpt_dir / f"bag_{b:03d}.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            bags.append(model)
        return cls(bags, device)

    @torch.no_grad()
    def predict(self, formulas: list[str], batch_size: int = 4096) -> dict[str, float]:
        """Returns {formula: P(synthesizable)}, averaged over all loaded bags.
        Formulas that fail to parse or use an element outside the training
        vocabulary are silently dropped (see chronosyn.features.cgnf_features)."""
        kept, X, dropped = cgnf_features(formulas, self._element_embedding)
        if dropped:
            print(f"chronosyn: dropped {len(dropped)} formula(s) "
                  f"(unparseable or outside the 90-element training vocabulary): {dropped}",
                  file=sys.stderr)
        if len(kept) == 0:
            return {}

        X_t = torch.from_numpy(X)
        running_sum = np.zeros(len(kept), dtype=np.float64)
        for model in self.bags:
            for start in range(0, len(kept), batch_size):
                end = min(start + batch_size, len(kept))
                logp, _year_out = model(X_t[start:end].to(self.device))
                running_sum[start:end] += torch.exp(logp[:, 1]).cpu().numpy()
        scores = running_sum / len(self.bags)
        return dict(zip(kept, scores.tolist()))


def main():
    formulas = sys.argv[1:]
    if not formulas:
        print("usage: python -m chronosyn.predict FORMULA [FORMULA ...]", file=sys.stderr)
        sys.exit(1)
    model = ChronoSynEnsemble.load()
    scores = model.predict(formulas)
    for f in formulas:
        if f in scores:
            print(f"{f}\t{scores[f]:.4f}")
        else:
            print(f"{f}\t(dropped, see stderr)")


if __name__ == "__main__":
    main()
