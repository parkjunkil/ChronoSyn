# ChronoSyn

[[`Paper (arxiv)`](https://arxiv.org/abs/2610.36193)]
[[`Project Page (to be updated)`](https://github.com/parkjunkil/ChronoSyn/)]


<p align="center">
  <img src="https://github.com/user-attachments/assets/89f7086d-00fa-4b76-b506-81bb6f50047d"  width="60%" alt="chronology" />
</p>


ChronoSyn is a composition-only synthesizability prediction model that considers
**materials discovery chronology** into a PU-learning (positive-unlabeled) setup. Code release for the paper 
"Learning the Chronology of Materials Discovery for Synthesizability Prediction" (arxiv, 2026).

## Installation

```bash
conda env create -f environment.yml
conda activate chronosyn
```

## Quickstart: score your own compositions

```python
from chronosyn.predict import ChronoSynEnsemble

model = ChronoSynEnsemble.load()  # loads the 100-bag MP-only checkpoint
scores = model.predict(["Na7PS6", "Fe2O3", "SrTiO3"])
print(scores)  # {"Na7PS6": 0.99..., "Fe2O3": 0.99..., "SrTiO3": 0.99...}
```

or from the command line:

```bash
python -m chronosyn.predict Na7PS6 Fe2O3 SrTiO3
```

Scores are the 100-bag ensemble-averaged P(synthesizable), in [0, 1].
Compositions that fail to parse, or that contain an element outside the
90-element training vocabulary, are silently dropped (reported on stderr).

## Repository layout

```
chronosyn/              # package: model, dataset, feature extraction, training, inference
  model.py              # CGNFNet (MLP trunk + binary head + chronology head)
  data.py               # CompositionDataset, train/test split utilities
  features.py           # 90-d CGNF composition feature extraction
  train.py              # PU-bagging training script
  predict.py            # load the pretrained ensemble and score new compositions
  element_embedding.json  # pretrained-CGCNN per-element embedding table (90-dim x 100 elements)

data/
  mponly/                # unique compositions + 90-d features, Materials Project only
  mpoqmd/                 # unique compositions (features built locally, see Dataset below)

checkpoints/
  chronosyn_mponly/      # 100 bags, h_fea_len=450, n_h=5 (the released checkpoint)

scripts/
  build_features.py     # regenerates cgnf90_features.npy for a unique_compositions.csv
```

## Dataset

Each `data/<dataset>/unique_compositions.csv` has one row per unique
reduced composition, with:

| column | meaning |
|---|---|
| `reduced_formula` | pymatgen-reduced chemical formula |
| `is_P` | `True` if this composition has an ICSD-confirmed experimental structure among its Materials Project (and, for `mpoqmd`, OQMD) entries; `False` otherwise |
| `year` | earliest ICSD-reported publication year, for `is_P == True` rows only (proxy for discovery time) |

- **mponly**: 104,837 compositions (51,010 P / 53,827 U), from the
  Materials Project. `cgnf90_features.npy` (37.7 MB) is included directly.
- **mpoqmd**: 408,638 compositions (51,651 P / 356,987 U), from the union
  of the Materials Project and OQMD. Its feature file is **not** included
  in the repo (it would be 147 MB, over GitHub's 100 MB file limit) --
  build it locally first (takes well under a minute):

  ```bash
  python scripts/build_features.py data/mpoqmd/unique_compositions.csv
  # -> writes data/mpoqmd/cgnf90_features.npy
  ```

## Reproducing training

```bash
# MP-only (the released checkpoint)
python -m chronosyn.train --run-name chronosyn_mponly --n-bags 100 \
    --year-head --year-mode ranking

# MP+OQMD (run scripts/build_features.py first, see Dataset section above)
python -m chronosyn.train --run-name chronosyn_mpoqmd \
    --csv-path data/mpoqmd/unique_compositions.csv \
    --features-path data/mpoqmd/cgnf90_features.npy \
    --n-bags 100 \
    --year-head --year-mode ranking
```

Each run writes `runs/<run-name>/checkpoints/bag_{000..099}.pt`,
`runs/<run-name>/per_bag.json` (per-bag test metrics), and
`runs/<run-name>/summary.json` (final 100-bag ensemble metrics).

Training uses only GPU if available (falls back to CPU automatically); the
released checkpoint was trained on a single GPU, ~100 epochs/bag, 100 bags.

## Citation

If you find this code helpful, please consider citing:

arxiv version
```bibtex
@article{,
  author={Park, Junkil and Walsh, Aron},
  title={Learning the Chronology of Materials Discovery for Synthesizability Prediction},
  link={https://arxiv.org/abs/2610.36193},
  year={2026},
}
```

## License

MIT (see `LICENSE`). Note `element_embedding.json` originates from
kaist-amsg/Synthesizability-stoi-CGNF and should keep that project's own
license/attribution.
