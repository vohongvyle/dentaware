"""Splits. Each radiograph comes from a different child (Zhang et al. 2023: 100 curated patients),
so image-level splits are patient-level splits. The official test split is never used for model
selection; all tuning happens inside cross-validation on the official training split.
"""

from __future__ import annotations

import numpy as np

from .labels import CLASSES


def stratified_folds(sample_ids: list[str], class_counts: list[np.ndarray], k: int, seed: int) -> list[list[str]]:
    """Greedy multilabel stratification: spread images so every fold sees each disease class.

    Rare classes are placed first; each image goes to the fold that currently has the fewest
    instances of that image's rarest class (ties broken by fold size).
    """
    rng = np.random.default_rng(seed)
    counts = np.stack(class_counts) if class_counts else np.zeros((0, len(CLASSES)))
    totals = counts.sum(axis=0)
    order = list(range(len(sample_ids)))
    rng.shuffle(order)

    def rarity(i):
        present = np.nonzero(counts[i])[0]
        return min((totals[c] for c in present), default=np.inf)

    order.sort(key=rarity)
    fold_cls = np.zeros((k, counts.shape[1]))
    fold_size = np.zeros(k)
    folds: list[list[str]] = [[] for _ in range(k)]
    for i in order:
        present = np.nonzero(counts[i])[0]
        if present.size:
            rare = min(present, key=lambda c: totals[c])
            key = fold_cls[:, rare] * 1000 + fold_size
        else:
            key = fold_size.copy()
        f = int(np.argmin(key + rng.random(k) * 1e-3))
        folds[f].append(sample_ids[i])
        fold_cls[f] += counts[i]
        fold_size[f] += 1
    return folds


def train_val_split(ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return ids[n_val:], ids[:n_val]
