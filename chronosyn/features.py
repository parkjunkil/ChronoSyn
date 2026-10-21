"""
The 90-d CGNF composition feature (Jang et al., Matter 2024,
"Synthesizability of materials stoichiometry using semi-supervised
learning", Eq. 4):

    CGNF(composition) = sum_i( element_embedding[el_i] * amount_i ) / sum_i(amount_i)

where element_embedding is the pretrained-CGCNN node embedding table shipped
with that paper's repo (element_embedding.json, 100 elements x 90-dim) and
amounts come from pymatgen's Composition.get_el_amt_dict().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_JSON = HERE / "element_embedding.json"


def load_element_embedding(embedding_json: str | Path = DEFAULT_EMBEDDING_JSON) -> dict[str, np.ndarray]:
    with open(embedding_json) as f:
        raw = json.load(f)
    return {el: np.array(vec, dtype=np.float64) for el, vec in raw.items()}


def cgnf_feature(formula: str, element_embedding: dict[str, np.ndarray]) -> np.ndarray | None:
    """Returns a [90] float32 vector, or None if `formula` can't be parsed or
    contains an element outside the embedding vocabulary."""
    try:
        amt_dict = Composition(formula).get_el_amt_dict()
    except Exception:
        return None
    total_amt = sum(amt_dict.values())
    if total_amt == 0:
        return None
    emb_dim = len(next(iter(element_embedding.values())))
    vec = np.zeros(emb_dim, dtype=np.float64)
    for el, amt in amt_dict.items():
        if el not in element_embedding:
            return None
        vec += element_embedding[el] * amt
    return (vec / total_amt).astype(np.float32)


def cgnf_features(formulas: list[str], element_embedding: dict[str, np.ndarray] | None = None):
    """Batched version of cgnf_feature. Returns (kept_formulas, features [M,90],
    dropped_formulas) -- `dropped_formulas` couldn't be parsed or used an
    element outside the embedding vocabulary."""
    if element_embedding is None:
        element_embedding = load_element_embedding()
    kept, feats, dropped = [], [], []
    for f in formulas:
        v = cgnf_feature(f, element_embedding)
        if v is None:
            dropped.append(f)
        else:
            kept.append(f)
            feats.append(v)
    X = np.stack(feats) if feats else np.zeros((0, 90), dtype=np.float32)
    return kept, X, dropped
