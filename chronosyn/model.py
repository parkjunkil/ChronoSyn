"""
ChronoSyn's composition-only PU classifier: a thin MLP trunk over the 90-d
CGNF composition feature (see features.py), with an auxiliary chronology
head attached to the same shared hidden representation.

The classification head/loss (2-class LogSoftmax + NLLLoss) follows
kaist-amsg/Synthesizability-stoi-CGNF's synstoi/model_PU_learning_comp.py::Net,
with a precomputed [B, 90] feature tensor as input instead of that repo's
on-the-fly calculate_input() over a raw formula string.

The chronology head (year_head=True) is an auxiliary branch off the same
shared hidden representation `h` that feeds the binary head: a small
Linear(h_fea_len, year_head_dim) -> Softplus -> Linear(year_head_dim, out_dim)
trained on P samples only (U has no reporting year, masked out of the loss
by label). `year_mode="ranking"` (ChronoSyn's setting) trains it with a
pairwise margin loss over relative chronological order rather than
regressing the exact year -- see train.py for the full loss construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CGNFNet(nn.Module):
    def __init__(
        self,
        feat_dim: int = 90,
        h_fea_len: int = 180,
        n_h: int = 1,
        dropout: float = 0.5,
        year_head: bool = False,
        year_mode: str = "mse",
        n_year_bins: int = 1,
        year_head_dim: int = 8,
        chrono_cond_dim: int = 0,
        chrono_cond_gate: bool = False,
        year_embed_concat: bool = False,
        year_embed_detach: bool = True,
    ):
        super().__init__()
        self.cgnf_to_fc = nn.Linear(feat_dim, h_fea_len)
        self.cgnf_to_fc_softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        # chrono_cond_dim > 0: binary head reads concat([h, chrono_feat])
        # instead of h alone. chrono_feat is a FROZEN, precomputed feature
        # (build_chrono_features.py's leak-free cross-fitted PMF over year
        # bins) -- not backprop-connected to the encoder, unlike year_head
        # below. This is the "feature injection" lesson ported from the
        # deprecated CGCNN/deephit model (chrono_cond_pu), where it beat
        # every auxiliary-loss variant (year_head's mse/ranking/bin/hazard)
        # by a wide margin.
        #
        # year_embed_concat (requires year_head=True): an ONLINE, jointly-
        # trained alternative to chrono_cond -- instead of a frozen feature
        # from a separately-trained model, the binary head reads head_year's
        # own OWN hidden bottleneck z_year (the year_head_dim-wide layer
        # right before its final Linear), concatenated onto h. year_embed_detach
        # (default True) cuts the gradient from the binary loss into z_year
        # (and transitively head_year/h) at that concat point, so the binary
        # loss can only learn how much to WEIGHT the chronology signal, not
        # distort what it represents -- the same protection chrono_cond gets
        # from being frozen, but without the 2-stage cross-fitting pipeline.
        # With year_embed_detach=False, both losses freely reshape z_year,
        # which is expected to behave like the existing year_head variants
        # (mse/ranking/bin/hazard) that already plateaued well below chrono_cond.
        # chrono_cond_gate: instead of concatenating the frozen chrono_feat
        # as-is, first scale it per-sample by a gate g = sigmoid(Linear(h))
        # in [0,1]^chrono_cond_dim -- "how ambiguous is this composition,
        # judging from h alone?" chrono_feat itself is still frozen/untouched
        # (gate only rescales it, doesn't reshape what it means); the gate
        # is a normal trainable layer, learned from the binary loss, so the
        # classifier can down-weight chronology for samples where h already
        # looks decisive and lean on it for the rest -- a sample-adaptive
        # version of the fixed per-dimension weighting fc_out's first layer
        # already learns implicitly (same input, every sample).
        self.chrono_cond_dim = chrono_cond_dim
        self.chrono_cond_gate = chrono_cond_gate
        if chrono_cond_dim > 0 and chrono_cond_gate:
            self.chrono_gate = nn.Linear(h_fea_len, chrono_cond_dim)
        self.year_embed_concat = year_embed_concat
        self.year_embed_detach = year_embed_detach
        assert not year_embed_concat or year_head, "year_embed_concat requires year_head=True"
        extra_dim = chrono_cond_dim + (year_head_dim if year_embed_concat else 0)
        self.fc_out = nn.Linear(h_fea_len + extra_dim, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.year_head = year_head
        self.year_mode = year_mode
        if year_head:
            # idea#1's own head_year (Linear(180,180)+Softplus+Dropout+Linear(180,1),
            # ~32.8K params) was sized to match its binary head, which in that
            # model was an equally deep/wide MLP. Here fc_out is just the last
            # layer of the shared 5-linear trunk (Linear(180,2), 362 params) --
            # so copying idea#1's head_year verbatim would make the auxiliary
            # head ~90x heavier than the head it's supposed to be helping,
            # risking the shared trunk being shaped by the year loss more than
            # intended. Two small linear layers instead (180->8->{1,n_year_bins},
            # ~1.5K params, ~4x fc_out) keep the "auxiliary pressure" character
            # without dominating the trunk. mse/ranking regress a single raw
            # scalar; bin classifies into n_year_bins equal-width year buckets.
            # "hazard" also outputs n_year_bins raw values (one hazard logit per
            # real year bin; the implicit (n_year_bins+1)-th "not yet observed"
            # state needs no logit of its own -- see train.py's year_aux_loss
            # hazard branch for the cumulative-product PMF). ChronoSyn itself
            # uses year_mode="ranking" (see train.py).
            out_dim = n_year_bins if year_mode in ("bin", "hazard", "hazard_rank") else 1
            self.head_year = nn.Sequential(
                nn.Linear(h_fea_len, year_head_dim),
                nn.Softplus(),
                nn.Linear(year_head_dim, out_dim),
            )

    def forward(self, x: torch.Tensor, chrono_feat: torch.Tensor | None = None):
        h = self.cgnf_to_fc(x)
        h = self.cgnf_to_fc_softplus(h)
        h = self.dropout(h)
        if hasattr(self, "fcs"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                h = softplus(fc(h))
        year_out = None
        z_year = None
        if self.year_head:
            # head_year is Linear(h_fea_len,year_head_dim) -> Softplus -> Linear(year_head_dim,out_dim).
            # Indexed manually (not called as one nn.Sequential) so z_year --
            # the bottleneck right before the final Linear -- is available
            # for year_embed_concat below, without changing head_year's
            # state_dict key names (still head_year.0/.1/.2).
            z_year = self.head_year[1](self.head_year[0](h))  # [B, year_head_dim]
            year_out = self.head_year[2](z_year)
            if self.year_mode not in ("bin", "hazard", "hazard_rank"):
                year_out = year_out.squeeze(-1)  # [B], raw scalar (normalized-year scale)

        parts = [h]
        if self.chrono_cond_dim > 0:
            assert chrono_feat is not None, "chrono_cond_dim > 0 requires chrono_feat at forward time"
            if self.chrono_cond_gate:
                gate = torch.sigmoid(self.chrono_gate(h))  # [B, chrono_cond_dim], learned from binary loss
                chrono_feat = gate * chrono_feat            # chrono_feat itself stays frozen/untouched
            parts.append(chrono_feat)
        if self.year_embed_concat:
            parts.append(z_year.detach() if self.year_embed_detach else z_year)
        bin_in = torch.cat(parts, dim=1) if len(parts) > 1 else h
        logp = self.logsoftmax(self.fc_out(bin_in))  # [B, 2] log-probs; use with NLLLoss on {0,1} targets

        if self.year_head:
            return logp, year_out  # year_out is [B, n_year_bins] logits when year_mode in (bin, hazard)
        return logp
