"""Uncertainty for small test sets: bootstrap confidence intervals, resampling *images* (patients)."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile CI of per-image values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n: int = 1000, seed: int = 0) -> dict:
    """Difference b - a on the same images, with CI and the share of resamples where b > a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n, a.size))
    d = (b[idx] - a[idx]).mean(axis=1)
    return {"diff": float((b - a).mean()), "ci_low": float(np.quantile(d, 0.025)),
            "ci_high": float(np.quantile(d, 0.975)), "p_b_better": float((d > 0).mean())}


def bootstrap_statistic(units: list, stat: Callable[[list], float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """CI for any statistic computed over a list of units (e.g. per-image detection records -> mAP)."""
    rng = np.random.default_rng(seed)
    point = stat(units)
    vals = []
    for _ in range(n):
        sample = [units[i] for i in rng.integers(0, len(units), len(units))]
        v = stat(sample)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
