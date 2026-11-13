"""Paired metrics and chain-clustered statistics.

Headline definitions follow LIBERO-MAX (Base SR, Dynamic SR, delta, four-way transitions,
conditional regression). Additions: Pre/Mid decomposition and a chain-cluster bootstrap,
because cases on one chain share a Base rollout (D-007).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class PairedSummary:
    n: int
    base_sr: float
    dyn_sr: float
    delta: float  # dyn - base, as a fraction
    preserved: int  # (1,1)
    change_success: int  # (0,1)
    regression: int  # (1,0)
    persistent_failure: int  # (0,0)
    conditional_regression: float  # (1,0) / [(1,1) + (1,0)]
    retained_ratio: float  # dyn_sr / base_sr


def paired_summary(y_base, y_dyn) -> PairedSummary:
    y0 = np.asarray(y_base, dtype=int)
    y1 = np.asarray(y_dyn, dtype=int)
    if y0.shape != y1.shape or y0.ndim != 1:
        raise ValueError("y_base and y_dyn must be equal-length 1-D arrays")
    n = len(y0)
    p11 = int(np.sum((y0 == 1) & (y1 == 1)))
    p01 = int(np.sum((y0 == 0) & (y1 == 1)))
    p10 = int(np.sum((y0 == 1) & (y1 == 0)))
    p00 = int(np.sum((y0 == 0) & (y1 == 0)))
    base_sr = y0.mean() if n else float("nan")
    dyn_sr = y1.mean() if n else float("nan")
    base_succ = p11 + p10
    return PairedSummary(
        n=n,
        base_sr=float(base_sr),
        dyn_sr=float(dyn_sr),
        delta=float(dyn_sr - base_sr),
        preserved=p11,
        change_success=p01,
        regression=p10,
        persistent_failure=p00,
        conditional_regression=p10 / base_succ if base_succ else float("nan"),
        retained_ratio=float(dyn_sr / base_sr) if base_sr else float("nan"),
    )


def exact_mcnemar(regression: int, change_success: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (1,0) vs (0,1)."""
    m = regression + change_success
    if m == 0:
        return 1.0
    k = min(regression, change_success)
    tail = sum(comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)


def decompose(y_base, y_pre, y_mid) -> dict[str, float]:
    """Delta_total = Delta_config (Pre - Base) + Delta_timing (Mid - Pre), on matched cases."""
    y0, yp, ym = (np.asarray(a, dtype=float) for a in (y_base, y_pre, y_mid))
    return {
        "delta_total": float(ym.mean() - y0.mean()),
        "delta_config": float(yp.mean() - y0.mean()),
        "delta_timing": float(ym.mean() - yp.mean()),
    }


def chain_bootstrap(stat, chain_ids, *arrays, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile CI for stat(*arrays), resampling whole chains with replacement.

    `stat` takes the resampled arrays and returns a float. Cases sharing a chain share a Base
    rollout, so resampling cases independently would understate the variance.
    """
    chain_ids = np.asarray(chain_ids)
    arrays = [np.asarray(a) for a in arrays]
    uniq, inverse = np.unique(chain_ids, return_inverse=True)
    members = [np.flatnonzero(inverse == j) for j in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = float(stat(*arrays))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[j] for j in pick])
        draws[b] = stat(*(a[idx] for a in arrays))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def holm(pvalues) -> list[float]:
    """Holm step-down adjusted p-values, returned in input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
