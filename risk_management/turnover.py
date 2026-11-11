"""Per-day turnover cap (spec 11.1 rule 5)."""

from __future__ import annotations

import pandas as pd


def apply_turnover_cap(current: pd.Series, target: pd.Series, cap: float) -> pd.Series:
    """Scale the trade vector proportionally so one-way turnover <= ``cap``.

    One-way turnover is ``max(sum of buys, sum of sells)`` in portfolio-weight units.
    Returns the post-trade weights.
    """
    idx = current.index.union(target.index)
    cur = current.reindex(idx, fill_value=0.0).astype(float)
    tgt = target.reindex(idx, fill_value=0.0).astype(float)
    delta = tgt - cur
    turnover = max(delta.clip(lower=0.0).sum(), (-delta).clip(lower=0.0).sum())
    if turnover <= cap or turnover == 0.0:
        return tgt
    return cur + delta * (cap / turnover)
