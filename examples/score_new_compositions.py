"""
Minimal example: score a handful of compositions with the pretrained
MP-only ChronoSyn ensemble and print the results sorted by score.

    python examples/score_new_compositions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronosyn.predict import ChronoSynEnsemble

FORMULAS = [
    "Na7PS6",
    "Fe2O3",
    "SrTiO3",
    "Ca(PN2)2",
    "YbAu2",
]


def main():
    model = ChronoSynEnsemble.load()
    scores = model.predict(FORMULAS)
    for formula, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{formula:12s} {score:.4f}")


if __name__ == "__main__":
    main()
