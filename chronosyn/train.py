"""
PU-bagging training for ChronoSyn.

Model/loss follow kaist-amsg/Synthesizability-stoi-CGNF (CGNFNet + NLLLoss
on a 2-class log-softmax output; Adam, lr=1e-3). The PU-bagging scheme
differs from that paper's Algorithm 1 on purpose: that paper resamples U
fresh from the whole unlabeled pool every iteration, which only works
because their U vastly outnumbers P (~8.6:1) so each iteration's negative
draw barely dents the pool. Our P:U ratio is closer to 1:1 (MP-only) or
~1:7 (MP+OQMD), so instead we hold out a single fixed 20% test slice of
both P and U up front (data.stratified_split); each bag draws its own
bootstrap U subsample (same size as train-P, WITH replacement by default)
only from the remaining train-U, and every bag / the final ensemble are
scored on that same fixed test set.

--year-head adds ChronoSyn's auxiliary chronology head (see model.py): a
loss on normalized reporting year, P-only (U has no year), added to the
NLLLoss with weight --year-lambda. --year-mode selects the auxiliary loss:
  "mse"     -- regress the exact normalized year.
  "ranking" (ChronoSyn's setting) -- pairwise margin loss: for every
             P-observed pair (i, j) in a batch with year_i < year_j, push
             year_pred_j - year_pred_i above --year-rank-margin.
--year-rank-pool-size decouples the ranking loss from the sparse P subset
that happens to land in any one mini-batch: each step, a fresh pool of this
many compositions is instead drawn straight from the full train-P set and
scored separately just for the ranking loss (ChronoSyn uses 512, giving
~130k pairs/step instead of the ~2k a 128-example batch would give).

Usage (reproduces the released MP-only ChronoSyn checkpoint):
    python -m chronosyn.train --run-name chronosyn_mponly_h450n5 \\
        --h-fea-len 450 --n-h 5 --n-bags 100 \\
        --year-head --year-mode ranking --year-lambda 3.0 --year-rank-pool-size 512

    # MP+OQMD dataset instead of the MP-only default:
    python -m chronosyn.train --run-name chronosyn_mpoqmd_h450n5 \\
        --csv-path data/mpoqmd/unique_compositions.csv \\
        --features-path data/mpoqmd/cgnf90_features.npy \\
        --h-fea-len 450 --n-h 5 --n-bags 100 \\
        --year-head --year-mode ranking --year-lambda 3.0 --year-rank-pool-size 512
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Subset

from .data import CompositionDataset, estimate_prior, stratified_split
from .model import CGNFNet

torch.set_num_threads(int(os.environ.get("PU_NUM_THREADS", "4")))

N_BAGS = 100
EPOCHS = 100
BS = 128
LR = 1e-3
WEIGHT_DECAY = 0.0


def class_eval(prob_pos: np.ndarray, labels: np.ndarray):
    pred = (prob_pos > 0.5).astype(np.int64)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    accuracy = float((pred == labels).mean())
    auc = roc_auc_score(labels, prob_pos) if len(np.unique(labels)) > 1 else float("nan")
    return accuracy, precision, recall, fscore, auc


def _mse_component(year_pred, year_norm, obs):
    if not obs.any():
        return None
    return torch.nn.functional.mse_loss(year_pred[obs], year_norm[obs])


def _ranking_component(year_pred, year_norm, obs, year_rank_margin):
    if obs.sum() < 2:
        return None
    yp_obs = year_pred[obs]
    yt_obs = year_norm[obs]
    pair_mask = (yt_obs.unsqueeze(0) - yt_obs.unsqueeze(1) >= 0).float()
    pair_mask.fill_diagonal_(0)
    denom = pair_mask.sum().clamp_min(1.0)
    if pair_mask.sum() == 0:
        return None
    yp_i = yp_obs.unsqueeze(1).expand_as(pair_mask)
    yp_j = yp_obs.unsqueeze(0).expand_as(pair_mask)
    gap = year_rank_margin - (yp_j - yp_i)
    rank_loss = (pair_mask * torch.nn.functional.relu(gap)).sum()
    return rank_loss / denom


def year_aux_loss(year_pred, year_norm, obs, year_mode: str, year_rank_margin: float):
    """Auxiliary year loss on P-observed samples only. Returns None if fewer
    than the minimum samples needed (1 for mse, 2 for ranking pairs)."""
    if year_mode == "mse":
        return _mse_component(year_pred, year_norm, obs)
    elif year_mode == "ranking":
        return _ranking_component(year_pred, year_norm, obs, year_rank_margin)
    else:
        raise ValueError(f"unknown year_mode: {year_mode!r} (expected mse|ranking)")


def train_one_epoch(model, loader, optim, criterion, device, year_head: bool, year_lambda: float,
                     year_mode: str, year_rank_margin: float, pool_cfg=None):
    """pool_cfg (optional): (features_P, year_norm_P, pool_size, rng) --
    decouples the ranking loss from the current batch's own P subset (see
    module docstring)."""
    model.train()
    total, total_year_loss, n = 0.0, 0.0, 0
    if pool_cfg is not None:
        features_P, year_norm_P, pool_size, rng = pool_cfg
    else:
        pool_size = 0
    for feat, label, year_norm, _ in loader:
        feat, label, year_norm = feat.to(device), label.to(device), year_norm.to(device)
        if year_head:
            logp, year_pred = model(feat)
        else:
            logp = model(feat)
        loss = criterion(logp, label)
        if year_head:
            if pool_cfg is not None and pool_size > 0:
                pool_idx = rng.integers(0, len(features_P), size=pool_size)
                pool_feat = torch.from_numpy(features_P[pool_idx]).to(device)
                pool_year_norm = torch.from_numpy(year_norm_P[pool_idx]).to(device)
                _pool_logp, pool_year_pred = model(pool_feat)
                pool_obs = torch.ones(pool_size, dtype=torch.bool, device=device)
                year_loss = year_aux_loss(pool_year_pred, pool_year_norm, pool_obs, year_mode, year_rank_margin)
            else:
                obs = label.bool()
                year_loss = year_aux_loss(year_pred, year_norm, obs, year_mode, year_rank_margin)
            if year_loss is not None:
                loss = loss + year_lambda * year_loss
                total_year_loss += year_loss.item() * feat.size(0)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * feat.size(0)
        n += feat.size(0)
    return total / n, total_year_loss / n


@torch.no_grad()
def predict_prob_pos(model, loader, device, year_head: bool):
    model.eval()
    probs, labels = [], []
    for feat, label, _year_norm, _ in loader:
        feat = feat.to(device)
        logp, _ = model(feat) if year_head else (model(feat), None)
        probs.append(torch.exp(logp[:, 1]).cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--csv-path", type=str, default=None,
                     help="override CompositionDataset's default (MP-only) unique_compositions.csv, "
                          "e.g. data/mpoqmd/unique_compositions.csv for the MP+OQMD dataset")
    ap.add_argument("--features-path", type=str, default=None,
                     help="override CompositionDataset's default (MP-only) cgnf90_features.npy")
    ap.add_argument("--u-no-replace", action="store_true",
                     help="sample each bag's pseudo-negative U subset WITHOUT replacement "
                          "(matches Jang et al.'s original Algorithm 1). Only sensible when "
                          "|U| >> |P| -- default (replace=True bootstrap) is for near-1:1 P:U "
                          "datasets (MP-only) where without-replacement sampling would make "
                          "every bag's negative set nearly identical.")
    ap.add_argument("--n-bags", type=int, default=N_BAGS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BS)
    ap.add_argument("--h-fea-len", type=int, default=450)
    ap.add_argument("--n-h", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--year-head", action="store_true",
                     help="add ChronoSyn's auxiliary chronology head (see model.py)")
    ap.add_argument("--year-lambda", type=float, default=3.0)
    ap.add_argument("--year-mode", type=str, default="ranking", choices=["mse", "ranking"])
    ap.add_argument("--year-rank-margin", type=float, default=0.1,
                     help="hinge margin for --year-mode ranking, in normalized-year units "
                          "(0.1 ~= 12 years over the 1905-2024 span)")
    ap.add_argument("--year-rank-pool-size", type=int, default=512,
                     help="0: ranking loss uses whatever P samples land in the current batch. "
                          ">0 (ChronoSyn's setting, default 512): decouple the ranking loss -- "
                          "each step, freshly sample this many compositions from train-P and "
                          "forward them separately just for the ranking loss. Only affects "
                          "--year-mode ranking.")
    ap.add_argument("--year-head-dim", type=int, default=8,
                     help="hidden width of the chronology head's bottleneck layer")
    args = ap.parse_args()

    RUN_DIR = os.path.join("runs", args.run_name)
    CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
    SUMMARY_PATH = os.path.join(RUN_DIR, "summary.json")
    PERBAG_PATH = os.path.join(RUN_DIR, "per_bag.json")
    os.makedirs(CKPT_DIR, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run_name: {args.run_name}  n_bags: {args.n_bags}  seed: {args.seed}  device: {device}", flush=True)
    print(f"settings: EPOCHS={args.epochs}, BS={args.batch_size}, LR={args.lr}, "
          f"h_fea_len={args.h_fea_len}, n_h={args.n_h}, dropout={args.dropout}, "
          f"year_head={args.year_head}"
          + (f" (mode={args.year_mode}, lambda={args.year_lambda}, head_dim={args.year_head_dim}"
             f", margin={args.year_rank_margin}, pool_size={args.year_rank_pool_size})"
             if args.year_head else ""), flush=True)

    ds_kwargs = {}
    if args.csv_path is not None:
        ds_kwargs["csv_path"] = args.csv_path
    if args.features_path is not None:
        ds_kwargs["features_path"] = args.features_path
    ds = CompositionDataset(**ds_kwargs)
    print(f"dataset size: {len(ds):,}", flush=True)

    train_idx, test_idx = stratified_split(ds.labels, test_frac=args.test_frac, seed=args.seed)
    train_P = train_idx[ds.labels[train_idx] == 1]
    train_U = train_idx[ds.labels[train_idx] == 0]
    prior = estimate_prior(ds.labels)
    print(f"train P={len(train_P):,}, U={len(train_U):,} (prior={prior:.4f}) | "
          f"test={len(test_idx):,}", flush=True)

    pool_cfg = None
    if args.year_rank_pool_size > 0:
        assert args.year_mode == "ranking", "--year-rank-pool-size only applies to --year-mode ranking"
        features_P = ds.features[train_P]  # precomputed once -- train-P is fixed across every bag
        year_norm_P = ds.year_norm[train_P]
        pool_rng = np.random.default_rng(args.seed)
        pool_cfg = (features_P, year_norm_P, args.year_rank_pool_size, pool_rng)
        print(f"year_rank_pool_size={args.year_rank_pool_size}: decoupling ranking loss from the "
              f"batch loop, sampling fresh from all {len(train_P):,} train-P each step", flush=True)

    LOADER_KW = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(Subset(ds, test_idx.tolist()), **LOADER_KW)

    criterion = torch.nn.NLLLoss()
    ckpts = []
    per_bag_log: list[dict] = []
    t_total = time.time()
    for bag in range(args.n_bags):
        t0 = time.time()
        rng_bag = np.random.default_rng(args.seed + bag)
        U_sub = rng_bag.choice(train_U, size=len(train_P), replace=not args.u_no_replace).tolist()
        bag_train_idx = train_P.tolist() + U_sub
        rng_bag.shuffle(bag_train_idx)
        loader_gen = torch.Generator()
        loader_gen.manual_seed(args.seed + bag)
        bag_loader = DataLoader(Subset(ds, bag_train_idx), shuffle=True, generator=loader_gen, **LOADER_KW)

        torch.manual_seed(args.seed + bag)
        model = CGNFNet(feat_dim=90, h_fea_len=args.h_fea_len, n_h=args.n_h, dropout=args.dropout,
                         year_head=args.year_head, year_mode=args.year_mode,
                         year_head_dim=args.year_head_dim).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        torch.manual_seed(args.seed + bag + 1_000_000)

        ckpt_path = os.path.join(CKPT_DIR, f"bag_{bag:03d}.pt")
        for epoch in range(args.epochs):
            _, year_loss = train_one_epoch(model, bag_loader, optim, criterion, device,
                                            args.year_head, args.year_lambda,
                                            args.year_mode, args.year_rank_margin, pool_cfg=pool_cfg)
        torch.save(model.state_dict(), ckpt_path)

        test_prob, test_labels_bag = predict_prob_pos(model, test_loader, device, args.year_head)
        acc, prec, recall, f1, auc = class_eval(test_prob, test_labels_bag)

        dt = time.time() - t0
        ckpts.append(ckpt_path)
        per_bag_log.append({
            "bag": bag, "bag_size": len(bag_train_idx), "time_s": dt,
            "test_acc": acc, "test_precision": prec, "test_recall": recall,
            "test_f1": f1, "test_auc": auc,
            **({"final_epoch_year_loss": year_loss} if args.year_head else {}),
        })
        with open(PERBAG_PATH, "w") as f:
            json.dump(per_bag_log, f, indent=2)
        elapsed_total = (time.time() - t_total) / 60
        eta_min = (elapsed_total / (bag + 1)) * (args.n_bags - bag - 1)
        year_tag = f" year_{args.year_mode}_loss={year_loss:.4f}" if args.year_head else ""
        print(
            f"bag {bag:3d}/{args.n_bags} | bag_size={len(bag_train_idx)} "
            f"test AUC={auc:.3f} recall={recall:.3f}{year_tag} | {dt:.1f}s | "
            f"elapsed {elapsed_total:.1f}min ETA {eta_min:.1f}min",
            flush=True,
        )

    avg_bag_auc = float(np.mean([x["test_auc"] for x in per_bag_log]))
    std_bag_auc = float(np.std([x["test_auc"] for x in per_bag_log]))
    print(f"\nmean per-bag test AUC: {avg_bag_auc:.3f} +/- {std_bag_auc:.3f}", flush=True)

    print("\nrunning ensemble eval (bootstrap aggregating) on test set...", flush=True)
    all_probs = np.zeros((len(ckpts), len(test_idx)))
    test_labels = None
    model = CGNFNet(feat_dim=90, h_fea_len=args.h_fea_len, n_h=args.n_h, dropout=args.dropout,
                     year_head=args.year_head, year_mode=args.year_mode,
                     year_head_dim=args.year_head_dim).to(device)
    for b, path in enumerate(ckpts):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        prob, labels = predict_prob_pos(model, test_loader, device, args.year_head)
        all_probs[b] = prob
        if test_labels is None:
            test_labels = labels

    ens_prob = all_probs.mean(axis=0)
    acc, prec, recall, f1, auc = class_eval(ens_prob, test_labels)
    n_pos = int((test_labels == 1).sum())
    n_neg = int((test_labels == 0).sum())

    summary = {
        "run_name": args.run_name,
        "settings": {
            "N_BAGS": args.n_bags, "EPOCHS": args.epochs, "BS": args.batch_size, "LR": args.lr,
            "h_fea_len": args.h_fea_len, "n_h": args.n_h, "dropout": args.dropout,
            "optimizer": "Adam", "seed": args.seed, "test_frac": args.test_frac,
            "year_head": args.year_head, "year_lambda": args.year_lambda if args.year_head else None,
            "year_mode": args.year_mode if args.year_head else None,
            "year_rank_margin": args.year_rank_margin if (args.year_head and args.year_mode == "ranking") else None,
            "year_rank_pool_size": args.year_rank_pool_size if args.year_rank_pool_size > 0 else None,
            "year_head_dim": args.year_head_dim if args.year_head else None,
        },
        "n_total": len(ds), "prior": prior,
        "n_train_P": len(train_P), "n_train_U": len(train_U),
        "n_test": len(test_idx),
        "mean_per_bag_test_auc": avg_bag_auc, "std_per_bag_test_auc": std_bag_auc,
        "test_n_pos": n_pos, "test_n_neg": n_neg,
        "ensemble_test_acc": acc, "ensemble_test_precision": prec, "ensemble_test_recall": recall,
        "ensemble_test_f1": f1, "ensemble_test_auc": auc,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nENSEMBLE TEST (N={len(test_labels):,}, P={n_pos:,}, U={n_neg:,})\n"
        f"  AUC       : {auc:.3f}\n"
        f"  Accuracy  : {acc:.3f}\n"
        f"  Precision : {prec:.3f}\n"
        f"  Recall    : {recall:.3f}\n"
        f"  F1        : {f1:.3f}\n"
        f"\nsaved: {SUMMARY_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
